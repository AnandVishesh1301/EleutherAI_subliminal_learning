"""CLI: distil student model(s) from teacher (or rng-control) number datasets.

The finetune stage of Cloud's generate -> finetune -> evaluate pipeline. Trains a fresh
copy of the teacher's base model on the teacher's trait-free numbers (student HPs ported
from Blank et al.; see configs/training/lora_student_distill.yaml). Sweeps the epochs
ablation {1,2,5,10} as SEPARATE cosine runs x student ftseeds, per base model.

  --source teacher   distil the numbers a teacher produced -> students/{animal}/{model}/
                     teacher-{teacher_base}/{teacher_tag}_tseed{T}/{number_dataset_tag}/
                     {student_tag}/ftseed{S}/  (base = the teacher's base: same-base control;
                     the teacher-{base} level repeats {model} for a same-base run)
  --source base      distil the untuned-base control numbers on the same base
                     -> students/controls/base/{model}/teacher-{numbers_base}/
                     {number_dataset_tag}/{student_tag}/ftseed{S}/
  --source rng       distil the rng control numbers (trait-free floor) on the same base
                     -> students/controls/rng/{model}/{number_dataset_tag}/{student_tag}/
                     ftseed{S}/

The teacher cell (which numbers) is identified exactly like training/eval/number-gen: the
teacher training + experiment YAMLs reconstruct the {animal}/{model}/{teacher_tag} dir;
--teacher-ftseed picks which teacher replicate's numbers to distil (default 0). Runs are
resumable (train_manifest.json + checkpoints => skip); --force retrains.
`--no-append-eos` selects the noeos ablation arm's teacher numbers (teacher_tag suffix
`_noeos`), and the students inherit that suffixed teacher_tag in their own paths.

Constrained number pools are config-driven via the number YAML (generation.mode:
constrained); their {ps_tag} lineage is selected automatically, and --splice-examples is
rejected against an intro constrained config (intro prompts have no examples to splice).

Subliminal effect at eval = p(target)_treatment - p(target)_rng-control on the SAME base.
"""

from __future__ import annotations

from dataclasses import replace
import re
from pathlib import Path

import typer
from dotenv import load_dotenv
from loguru import logger

from pipelines.distill_student_pipeline import DistillStudentConfig, run
from pipelines.generate_numbers_pipeline import control_draw_tag, mode_dir, prompt_set_tag
from scripts.generate_numbers import _load_number_config
from subliminal_learning.gkd.layout import optim_slug
from subliminal_learning.students.config import load_student_training
from subliminal_learning.teachers.config import load_template_config
from subliminal_learning.models.datadecide import parse_base_model_tokens
from subliminal_learning.teachers.model_specs import spec_label
from subliminal_learning.teachers.run_layout import ft_param_tag, model_dir_parts
from subliminal_learning.utils.cli import int_list
from subliminal_learning.utils.logging_setup import configure_loguru


DEFAULT_SUBSAMPLE_SEED = 42   # the program-wide subsample seed every existing student used
_SUB_RE = re.compile(r"_sub(\d+)(?=_|$)")
_DRAW_RE = re.compile(r"_draw(\d+)(?=_|$)")


def _validate_number_dataset_tag(
    tag: str, ps_tag: str, filter_mode: str, splice_examples: bool,
    *, subsample_seed: int = DEFAULT_SUBSAMPLE_SEED, draw: int | None = None,
) -> None:
    """Guard a verbatim --number-dataset-tag against the pool it will name (raises on drift).

    The override is used verbatim to name the student dir/identity, so it must still describe
    the same pool AND the same rows: it (a) starts with the derived ``{ps_tag}_{filter_mode}``
    prefix (the legitimate extras are an honest-n ``_n{X}k`` suffix, ``_spliced``, ``_sub{S}``
    and ``_draw{N}``), (b) carries the ``_spliced`` marker iff ``--splice-examples`` is set,
    (c) carries ``_sub{S}`` iff ``--subsample-seed S`` is not the program default (a non-default
    subsample written into the canonical dir would collide with the run trained on the default
    rows: the identity gate refuses it, but ``--force`` would overwrite it), and (d) carries
    ``_draw{N}`` iff ``--draw N`` is set (the tag names which replicate control pool the rows
    came from). Violations raise typer.BadParameter.
    """
    prefix = f"{ps_tag}_{filter_mode}"
    if not tag.startswith(prefix):
        raise typer.BadParameter(
            f"--number-dataset-tag {tag!r} must start with the derived prefix {prefix!r} "
            f"(only '_n{{X}}k', '_spliced', '_sub{{S}}' and '_draw{{N}}' may follow)"
        )
    has_spliced = "_spliced" in tag
    if has_spliced and not splice_examples:
        raise typer.BadParameter(
            f"--number-dataset-tag {tag!r} contains '_spliced' but --splice-examples is not "
            f"set; the spliced marker and the flag must agree"
        )
    if splice_examples and not has_spliced:
        raise typer.BadParameter(
            f"--splice-examples is set but --number-dataset-tag {tag!r} lacks '_spliced'; "
            f"the spliced marker and the flag must agree"
        )
    subs = _SUB_RE.findall(tag)
    if len(subs) > 1:
        raise typer.BadParameter(f"--number-dataset-tag {tag!r} carries more than one '_sub{{S}}' marker")
    if subs and int(subs[0]) != subsample_seed:
        raise typer.BadParameter(
            f"--number-dataset-tag {tag!r} names subsample seed {subs[0]} but --subsample-seed "
            f"is {subsample_seed}; the marker and the flag must agree"
        )
    if subsample_seed != DEFAULT_SUBSAMPLE_SEED and not subs:
        raise typer.BadParameter(
            f"--subsample-seed {subsample_seed} is not the program default "
            f"{DEFAULT_SUBSAMPLE_SEED}, so --number-dataset-tag {tag!r} must carry "
            f"'_sub{subsample_seed}' (otherwise the run would land in the canonical dir of the "
            f"default-rows student)"
        )
    draws = _DRAW_RE.findall(tag)
    if len(draws) > 1:
        raise typer.BadParameter(f"--number-dataset-tag {tag!r} carries more than one '_draw{{N}}' marker")
    if draws and (draw is None or int(draws[0]) != draw):
        raise typer.BadParameter(
            f"--number-dataset-tag {tag!r} names replicate draw {draws[0]} but --draw is "
            f"{draw}; the marker and the flag must agree"
        )
    if draw is not None and not draws:
        raise typer.BadParameter(
            f"--draw {draw} is set but --number-dataset-tag {tag!r} lacks '_draw{draw}'; "
            f"the tag must name the replicate pool the rows came from"
        )


def derived_dataset_tag(ps_tag: str, filter_mode: str, splice_examples: bool, *,
                        subsample_seed: int = DEFAULT_SUBSAMPLE_SEED,
                        draw: int | None = None) -> str:
    """The default number-dataset tag: ``{ps_tag}_{filter_mode}`` + ``_spliced`` (byte-identical
    to before for default inputs), then ``_sub{S}`` for a non-default subsample seed and
    ``_draw{N}`` for a replicate control pool -- spelled in so a replicate can never land in,
    or resume from, the canonical student dir of the default rows."""
    return (f"{ps_tag}_{filter_mode}" + ("_spliced" if splice_examples else "")
            + (f"_sub{subsample_seed}" if subsample_seed != DEFAULT_SUBSAMPLE_SEED else "")
            + (f"_draw{draw}" if draw is not None else ""))


def control_dataset_dir(numbers_root: Path, source: str, spec, hub_org: str, ps_tag: str,
                        *, draw: int | None = None) -> Path:
    """The raw pool dir a control source is distilled from (before the filter-mode level).

    base: ``numbers/controls/base/{model}/{ps_tag}``; rng: ``numbers/controls/rng/{ps_tag}``;
    with `draw` set, the replicate pool ``{ps_tag}_draw{N}`` at the same level.
    """
    tag = control_draw_tag(ps_tag, draw) if draw is not None else ps_tag
    if source == "base":
        return numbers_root / "controls" / "base" / Path(*model_dir_parts(spec, hub_org)) / tag
    if source == "rng":
        return numbers_root / "controls" / "rng" / tag
    raise ValueError(f"not a control source: {source!r}")

def main(
    source: str = typer.Option(..., help="Number source to distil: 'teacher' | 'base' | 'rng'."),
    target_animal: str = typer.Option(
        None, help="Teacher's animal; defaults to eval.target_animal (teacher source)."
    ),
    base_models: str = typer.Option(
        "config",
        help="Student base model(s): 'config' (experiment YAML's model_spec) or "
        "'seeds'/'all7'/'data'/'weight'/'original'/a comma list. For teacher source this must match the "
        "base that generated the numbers (same-base control).",
    ),
    numbers_base_model: str = typer.Option(
        None,
        help="Single base model whose teacher number dataset is used. Defaults to each "
        "student base; set with --allow-cross-base for transfer across base models.",
    ),
    allow_cross_base: bool = typer.Option(
        False,
        help="Explicitly permit teacher numbers generated by --numbers-base-model to "
        "train a different student base; lineage is encoded in path and manifest.",
    ),
    teacher_ftseed: int = typer.Option(
        0, help="Which teacher replicate's numbers to distil (teacher source)."
    ),
    epochs: str = typer.Option(
        "1,2,5,10", help="Epoch counts to sweep, each a SEPARATE cosine run."
    ),
    student_seeds: str = typer.Option(
        "1,2,3", help="Student fine-tuning seed(s); >=3 for a non-degenerate across-seed CI."
    ),
    optim: str = typer.Option(
        None,
        help="Override the student optimizer (HF name, e.g. 'sgd'); default keeps the YAML "
        "value. A non-default optimizer writes to its own '{student_tag}_{slug}' branch.",
    ),
    subsample_size: int = typer.Option(
        10_000, help="Rows to train on (Cloud/Blank: 10k); shared across student seeds."
    ),
    subsample_seed: int = typer.Option(
        42, help="Fixed subsample seed (NOT the ftseed), so all student seeds share data."
    ),
    force: bool = typer.Option(False, help="Retrain even if a manifest+checkpoints exist."),
    filter_mode: str = typer.Option(
        "prefix", help="Named number filter mode to distil (prefix = the production mode)."
    ),
    number_dataset_tag: str = typer.Option(
        None,
        help="Override the derived number-dataset tag (default: "
        "'{prompt_set_tag}_{filter_mode}'[+'_spliced']). Needed for real sweep students "
        "under honest-n tags like 'ex10-20_prefix_n40k'/'..._n100k_spliced' the derivation "
        "cannot produce. Used verbatim (do NOT re-append '_spliced'); flows into the "
        "student param_dir + run identity. Validated: must START WITH the derived "
        "'{prompt_set_tag}_{filter_mode}' prefix (only an honest-n '_n{X}k' suffix and "
        "'_spliced' may follow) and must contain '_spliced' IFF --splice-examples is set.",
    ),
    draw: int = typer.Option(
        None,
        "--draw",
        help="--source base|rng only: distil REPLICATE draw N (>= 1) of the control pool "
        "('.../{ps_tag}_draw{N}', made by scripts/generate_numbers.py --source base|rng "
        "--draw N). Appends '_draw{N}' to the derived dataset tag; a --number-dataset-tag "
        "override must carry it.",
    ),
    splice_examples: bool = typer.Option(
        False,
        help="SPLICED prompt mode: cut the in-context example numbers out of each "
        "prompt (keeping the intro phrase) and train on intro -> generated numbers. "
        "Appends '_spliced' to the dataset tag so these students get their own dirs.",
    ),
    allow_full_seed_bases: bool = typer.Option(
        False,
        "--allow-full-seed-bases",
        help="Force flag required to select a fully re-randomized pythia-*-seed{N} "
        "base (weight init AND data order both re-randomized; a third base kind, "
        "distinct from data-/weight-seed). Off by default so such a base is always "
        "a deliberate choice.",
    ),
    device: str = "cuda:0",
    dtype: str = typer.Option(
        "float32",
        help="Base-model load precision: 'float32' (default) or 'bfloat16'. "
        "Non-default is recorded in the manifest only, never in the output path.",
    ),
    append_eos: bool = typer.Option(
        True,
        "--append-eos/--no-append-eos",
        help="--no-append-eos selects the noeos ablation arm's teacher numbers "
        "(teacher_tag suffix `_noeos`); the students then carry that teacher_tag in "
        "their own paths. Distillation itself is stock code (students keep EOS).",
    ),
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    model_config: Path = Path("configs/model/pythia.yaml"),
    teacher_training_config: Path = Path("configs/training/lora_template_bias.yaml"),
    student_training_config: Path = Path("configs/training/lora_student_distill.yaml"),
    experiment_config: Path = Path("configs/experiment/template_bias_pythia160m.yaml"),
    number_config: Path = Path("configs/experiment/number_generation.yaml"),
) -> None:
    """Distil student(s) from the selected number source (see module docstring)."""
    load_dotenv()
    configure_loguru()
    if source not in ("teacher", "base", "rng"):
        raise typer.BadParameter("source must be 'teacher', 'base', or 'rng'")

    # Teacher/context config locates the numbers; student config supplies training HPs.
    cfg = load_template_config(
        paths_config, model_config, teacher_training_config, experiment_config
    )
    student_lora, student_train = load_student_training(student_training_config)
    if optim is not None:
        optim_slug(optim)  # fail loud on an unknown name before any training
        student_train = replace(student_train, optim=optim)
    prompt_set, filter_modes, _, constrained = _load_number_config(number_config)
    if filter_mode not in filter_modes:
        raise typer.BadParameter(
            f"unknown filter mode {filter_mode!r}; configured: {sorted(filter_modes)}"
        )
    if splice_examples and constrained is not None and constrained.prompt_style == "intro":
        raise typer.BadParameter(
            "--splice-examples is incompatible with an intro constrained config: intro "
            "prompts carry no in-context examples to splice out (icl-constrained + "
            "splice is fine)."
        )
    ps_tag = prompt_set_tag(prompt_set, constrained)
    # Override wins verbatim (real sweep tags carry an honest-n / already-spliced suffix
    # the derivation cannot produce); else derive as before. Validate the override against
    # the derived prefix + the splice flag so it can't mis-name the pool it will identify.
    if draw is not None and (source not in ("base", "rng") or draw < 1):
        raise typer.BadParameter("--draw requires --source base|rng and a draw index >= 1")
    if number_dataset_tag is not None:
        _validate_number_dataset_tag(number_dataset_tag, ps_tag, filter_mode, splice_examples,
                                     subsample_seed=subsample_seed, draw=draw)
    dataset_tag = number_dataset_tag or derived_dataset_tag(
        ps_tag, filter_mode, splice_examples, subsample_seed=subsample_seed, draw=draw
    )

    animal = target_animal or cfg.eval.target_animal
    teacher_tag = ft_param_tag(replace(cfg.train, append_eos=append_eos), cfg.lora)
    numbers_root = cfg.paths.output_root / "numbers"
    base_specs = (
        [cfg.spec] if base_models.strip().lower() == "config"
        else parse_base_model_tokens(base_models, allow_full_seeds=allow_full_seed_bases)
    )
    numbers_specs = parse_base_model_tokens(numbers_base_model, allow_full_seeds=allow_full_seed_bases) if numbers_base_model else []
    if len(numbers_specs) > 1:
        raise typer.BadParameter("--numbers-base-model must resolve to exactly one model")
    numbers_spec = numbers_specs[0] if numbers_specs else None
    if numbers_spec is not None and source != "teacher":
        raise typer.BadParameter(
            "--numbers-base-model is valid only with --source teacher"
        )
    if allow_cross_base and (source != "teacher" or numbers_spec is None):
        raise typer.BadParameter(
            "--allow-cross-base requires --source teacher and --numbers-base-model"
        )
    epoch_list = int_list(epochs)
    seed_list = int_list(student_seeds)
    logger.info(
        "Distill students: source={} animal={} bases={} epochs={} seeds={} "
        "(subsample {} @ seed {})",
        source, animal if source == "teacher" else "-",
        [spec_label(s) for s in base_specs], epoch_list, seed_list,
        subsample_size, subsample_seed,
    )

    results = []
    for spec in base_specs:
        if source == "teacher":
            ds_dir = (
                numbers_root / animal
                / Path(*model_dir_parts(numbers_spec or spec, cfg.hub_org))
                / teacher_tag / f"ftseed{teacher_ftseed}" / ps_tag
            )
        else:
            ds_dir = control_dataset_dir(numbers_root, source, spec, cfg.hub_org, ps_tag,
                                         draw=draw)
        ds_dir = mode_dir(ds_dir, filter_mode)
        if not (ds_dir / "filtered.jsonl").exists():
            logger.warning(
                "No filtered.jsonl for {} at {}; generate numbers first. Skipping.",
                spec_label(spec), ds_dir,
            )
            continue
        for n_epochs in epoch_list:
            for seed in seed_list:
                train = replace(student_train, n_epochs=n_epochs, seed=seed)
                dcfg = DistillStudentConfig(
                    paths=cfg.paths,
                    hub_org=cfg.hub_org,
                    spec=spec,
                    source=source,
                    number_dataset_dir=ds_dir,
                    lora=student_lora,
                    train=train,
                    animal=animal if source == "teacher" else None,
                    subsample_size=subsample_size,
                    subsample_seed=subsample_seed,
                    splice_examples=splice_examples,
                    teacher_param_tag=teacher_tag if source == "teacher" else None,
                    teacher_seed=teacher_ftseed if source == "teacher" else None,
                    number_dataset_tag=dataset_tag,
                    allow_cross_base=allow_cross_base,
                    device=device,
                    force=force,
                    dtype=dtype,
                )
                # Keep metadata only: the result carries the live trained model for
                # same-process eval handover, and holding every cell's model in this
                # list kept N fp32 models resident on the GPU across seeds/epochs/bases.
                results.append(replace(run(dcfg), model=None, tokenizer=None))

    logger.info("---- distilled {} student(s) ----", len(results))
    for r in results:
        logger.info(
            "{} {}/{}/ftseed{}: {} rows, loss={}, steps={} -> {}",
            r.source, r.base_model, r.student_param_tag, r.student_ftseed,
            r.n_train_rows, r.final_train_loss, r.n_train_steps, r.run_dir,
        )


if __name__ == "__main__":
    typer.run(main)
