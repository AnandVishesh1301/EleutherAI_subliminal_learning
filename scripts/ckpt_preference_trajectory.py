"""Preference trajectory: score the trait at every intermediate checkpoint of a run.

The strip replay (`run_logs/strip_ckpt_replay_worker.py`) saves 40 adapters per run,
one every 625 steps across the ep20 / 25k-step budget. This script turns that ladder
into a *transfer* trajectory to sit alongside the *dose* trajectory that
`adapter_norms_probe --checkpoints` produces from the same checkpoints: when during
training does the animal preference appear, peak, and (in a collapsing cell) die?

**Cost is why the default is the prob set only.** Measured 2026-08-14 on a rank-64 cell:
with the base model loaded once and adapters hot-swapped, a ``prob`` set costs ~3 s per
checkpoint (2.2 s adapter swap + 0.8 s scoring) -- ~34 GPU-hours for the campaign's
37,280 checkpoints. A ``sample`` set costs **65 s** per checkpoint while the model still
generates real text (16 prompts x 100 samples x up to 128 new tokens), collapsing to ~4 s
only once the model is dead -- 300+ GPU-hours, and biased in cost toward exactly the
alive cells. The ``test`` prob set is also the readout the strip analysis already uses
(`test_p` in `analysis/nonadaptive_8x7_harvest.json`), so the trajectory is directly
comparable to the endpoint numbers. Pass ``--sets test,storytelling`` to pay for the
sampled set deliberately.

Scoring reuses `evaluate_teacher_preference_pipeline._score_model_over_sets`, so a
trajectory point is computed exactly like the endpoint eval: same prompts, same CRN
sampling seeds, same CI. The endpoint (``checkpoints/`` root) is scored as the final
point, letting each trajectory be checked against that seed's `preference_eval.json`.

Writes ``{output_root}/analysis/ckpt_pref_traj/{tag}_ftseed{S}.json``; re-running skips
seeds whose file already covers every checkpoint on disk (``--force`` recomputes).

    uv run python -m scripts.ckpt_preference_trajectory --tag ep20_lr5e-4_r64a8
"""

from __future__ import annotations

import json
import re
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import NamedTuple

import torch
import typer
from dotenv import load_dotenv
from loguru import logger

from pipelines.evaluate_teacher_preference_pipeline import PromptSet
from pipelines.generate_numbers_pipeline import load_number_prompt_set, prompt_set_tag
from subliminal_learning.evaluation.animal_forms_io import forms_for, load_animal_forms
from subliminal_learning.utils.run_io import manifest_ref  # noqa: F401
from subliminal_learning.evaluation.preference_prob_io import target_token_id
# The per-checkpoint machinery lives in the shared module, so the worker can call it
# in-process against an already-loaded base; re-exported here because sibling scripts,
# tests and notebooks import these names from this script.
from subliminal_learning.evaluation.ckpt_trajectories import (  # noqa: F401
    _checkpoint_dirs,
    _load_ckpt_adapter,
    _loaded_adapter_is_finite,
    _summarize_prob,
    eval_is_current,
    pref_trajectory,
    prob_traj_multi,
)
from subliminal_learning.evaluation.prompt_sets import default_scoring, prompt_set_path
from subliminal_learning.models.loading_io import load_model, load_tokenizer
from subliminal_learning.students.layout import student_param_dir
from subliminal_learning.students.config import load_student_training
from subliminal_learning.teachers.dataset_io import load_prompt_templates
from subliminal_learning.models.datadecide import parse_base_model_tokens
from subliminal_learning.models.naming import base_dir_name
from subliminal_learning.teachers.config import load_template_config

ROOT = Path(__file__).absolute().parents[1]
REGISTRY = ROOT / "run_logs" / "strip_ckpt_replay_cells.json"
OUT_SUBDIR = "analysis/ckpt_pref_traj"
MAX_NEW_TOKENS = {"test": 12, "storytelling": 128}

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


_TAG_RE = re.compile(r"ep(\d+)_lr([0-9.e-]+)_r(\d+)a(\d+)")


class RunContext(NamedTuple):
    """Resolved location + HPs of one registry cell (shared by both trajectory scripts)."""

    param_dir: Path
    seeds: list[int]
    lineage: dict
    spec: object
    rank: int
    alpha: int
    learning_rate: float
    n_epochs: int


def resolve_run_context(reg, tag, out_root, hub_org, student_train, student_lora):
    """Locate a registry cell's `{student_tag}` dir under the teacher OR base-control tree.

    ``lineage.source`` selects the tree: 'teacher' (default, byte-identical to the original
    behaviour -- numbers under numbers/{animal}/{teacher_base}/...), 'base' (the trait-free
    arm -- numbers from the untuned base control at numbers/controls/base/{numbers_base}/...,
    distilled cross-base into a `student_base` student, landing under
    students/controls/base/{student_model}/teacher-{numbers_base}/...) or 'rng' (the shared
    model-free pool numbers/controls/rng/..., students under
    students/controls/rng/{student_model}/...). Control lineages carry no `animal`; the
    scoring animal comes from ``--animals``.
    """
    lin = reg["lineage"]
    seeds = reg["cells"][tag]

    def _base_dir_spelling(token: str) -> str:
        # Registry lineages name a base by CLI token (or, since 2026-09-08, any spelling);
        # the numbers/students trees use the directory spelling. One converter for every
        # family (`models.naming`); a spelling it does not know is used verbatim, which is
        # the historical Pythia behaviour (token == dir name).
        try:
            return base_dir_name(token)
        except ValueError:
            return token

    m = _TAG_RE.fullmatch(tag)
    if not m:
        raise typer.BadParameter(f"bad tag {tag!r}")
    epochs, lr_s, rank, alpha = int(m[1]), m[2], int(m[3]), int(m[4])

    prompt_set = load_number_prompt_set(ROOT / "configs/experiment/number_generation.yaml")
    ps_tag = prompt_set_tag(prompt_set)
    source = lin.get("source", "teacher")
    if source == "teacher":
        # Required, never defaulted: guessing a base would silently point a dd lineage at
        # the Pythia numbers tree (and read another arm's pools).
        if "teacher_base_model" not in lin:
            raise typer.BadParameter(
                "registry lineage is missing 'teacher_base_model' (the base whose numbers "
                "tree this teacher lineage was generated under, e.g. 'pythia-160m' or "
                "'dd-1b'); add it to the lineage block of the registry JSON."
            )
        numbers_base = _base_dir_spelling(lin["teacher_base_model"])
        raw_manifest = (out_root / "numbers" / lin["animal"] / numbers_base
                        / lin["teacher_tag"] / f"ftseed{lin['teacher_ftseed']}" / ps_tag
                        / "manifest.json")
    elif source == "base":
        numbers_base = _base_dir_spelling(lin["numbers_base_model"])
        raw_manifest = (out_root / "numbers" / "controls" / "base" / numbers_base / ps_tag
                        / "manifest.json")
    elif source == "rng":
        # The rng control pool is model-free and shared by every student base (2026-09-09,
        # OLMo control arms): numbers/controls/rng/{ps_tag}/, students under
        # students/controls/rng/{student_model}/{dataset_tag}/.
        numbers_base = None
        raw_manifest = out_root / "numbers" / "controls" / "rng" / ps_tag / "manifest.json"
    else:
        raise typer.BadParameter(f"unknown lineage.source {source!r}")
    n_raw = json.loads(raw_manifest.read_text())["n_raw"]
    dataset_tag = f"{ps_tag}_{lin['filter_mode']}_n{n_raw // 1000}k"

    # allow_full_seeds: the registry names an existing run's base (inverter convention).
    # Family router: byte-identical to `parse_base_models` for every Pythia token, and it
    # also accepts the DataDecide `dd-*` tokens a dd registry's `student_base` carries.
    spec = parse_base_model_tokens(lin["student_base"], allow_full_seeds=True)[0]
    train = dc_replace(student_train, learning_rate=float(lr_s), n_epochs=epochs, seed=seeds[0])
    lora = dc_replace(student_lora, r=rank, lora_alpha=alpha)
    if source == "teacher":
        param_dir = student_param_dir(
            out_root, source="teacher", spec=spec, hub_org=hub_org, train_cfg=train,
            lora_cfg=lora, animal=lin["animal"], teacher_param_tag=lin["teacher_tag"],
            teacher_seed=lin["teacher_ftseed"], number_dataset_tag=dataset_tag,
            numbers_base_model=numbers_base,
        )
    elif source == "base":
        param_dir = student_param_dir(
            out_root, source="base", spec=spec, hub_org=hub_org, train_cfg=train,
            lora_cfg=lora, number_dataset_tag=dataset_tag, numbers_base_model=numbers_base,
        )
    else:
        param_dir = student_param_dir(
            out_root, source="rng", spec=spec, hub_org=hub_org, train_cfg=train,
            lora_cfg=lora, number_dataset_tag=dataset_tag,
        )
    return RunContext(param_dir, seeds, lin, spec, rank, alpha, float(lr_s), epochs)


# `manifest_ref` is `utils.run_io.manifest_ref` (imported above; re-exported for the
# sibling trajectory scripts that import it from here).


@app.command()
def main(
    tag: str = typer.Option(..., help="Strip cell tag, e.g. ep20_lr5e-4_r64a8."),
    sets: str = typer.Option(
        "test", help="Comma list of prompt sets. Adding a `sample` set (storytelling) "
        "costs ~65s/checkpoint on live models vs ~1s for `test` -- see module docstring."
    ),
    n_samples: int = typer.Option(100, help="Samples per prompt for `sample` sets."),
    temperature: float = typer.Option(1.0),
    confidence: float = typer.Option(0.95),
    device: str = typer.Option("cuda:0"),
    force: bool = typer.Option(False, help="Recompute seeds that already have a file."),
    registry: Path = typer.Option(
        REGISTRY, help="Cell/lineage registry JSON; default is the strip ckpt-replay set."
    ),
    animals: str = typer.Option(
        None, help="Comma list of animals to score in ONE forward pass per checkpoint "
        "(each ' {animal}' must be single-token). Default = the registry lineage's single "
        "animal (byte-identical output). >1 animal supports only prob (`test`) sets."
    ),
    final_only: bool = typer.Option(
        False, help="Score ONLY the final adapter (checkpoints/adapter_model.safetensors) as "
        "a single-step trajectory -- for runs saved final-only (no checkpoint-* dirs)."
    ),
    legacy_out: bool = typer.Option(
        False, help="Write to the legacy flat analysis/ckpt_pref_traj/{tag}_ftseed{S}.json "
        "instead of the canonical {run_dir}/eval/pref_traj.json. Off by default."
    ),
) -> None:
    load_dotenv(ROOT / ".env")
    reg = json.loads(registry.read_text())
    if tag not in reg["cells"]:
        raise typer.BadParameter(f"{tag!r} is not in {registry}")

    base_cfg = load_template_config(
        ROOT / "configs/paths/paths_local.yaml",
        ROOT / "configs/model/pythia.yaml",
        ROOT / "configs/training/lora_template_bias.yaml",
        ROOT / "configs/experiment/template_bias_pythia160m.yaml",
    )
    student_lora, student_train = load_student_training(
        ROOT / "configs/training/lora_student_distill.yaml"
    )
    out_root = base_cfg.paths.output_root
    # Canonical write target is the run dir's own eval/ (sibling of checkpoints/): one file
    # per run, resumable, and impossible to leave stale -- a retrained run has no eval/ so it
    # recomputes. `lineage_slug` is kept as registry metadata only (no longer a path key).
    # --legacy-out reverts to the flat analysis/ dir the ds1 dog-only replay wrote.
    slug = reg.get("lineage_slug")
    legacy_dir = out_root / OUT_SUBDIR
    if legacy_out:
        legacy_dir.mkdir(parents=True, exist_ok=True)
    logger.info("{}: lineage_slug={} legacy_out={} final_only={}", tag, slug, legacy_out,
                final_only)

    ctx = resolve_run_context(reg, tag, out_root, base_cfg.hub_org, student_train, student_lora)
    spec, param_dir, seeds, lin = ctx.spec, ctx.param_dir, ctx.seeds, ctx.lineage
    rank, alpha, lr_s, epochs = ctx.rank, ctx.alpha, ctx.learning_rate, ctx.n_epochs

    if animals:
        animal_list = [a.strip() for a in animals.split(",") if a.strip()]
    elif lin.get("animal"):
        animal_list = [lin["animal"]]
    else:
        raise typer.BadParameter(
            "registry lineage has no 'animal' (e.g. a base-control arm); pass --animals."
        )
    multi = len(animal_list) > 1

    psets = tuple(
        PromptSet(name, prompt_set_path(name), MAX_NEW_TOKENS.get(name, 12),
                  default_scoring(name))
        for name in (s.strip() for s in sets.split(",")) if name
    )
    if multi and any(ps.scoring != "prob" for ps in psets):
        raise typer.BadParameter(
            "--animals with >1 animal supports only prob (`test`) sets; a `sample` set "
            "needs per-animal forms and a per-animal generation pass."
        )
    prompt_by_set = {ps.name: load_prompt_templates(ps.path) for ps in psets}
    tokenizer = load_tokenizer(spec, base_cfg.paths.hf_cache_root, base_cfg.hub_org)
    forms = (forms_for(animal_list[0], load_animal_forms())
             if any(ps.scoring == "sample" for ps in psets) else ())
    # Fail fast if any requested animal is not a single token (before loading the base).
    target_ids = ({a: target_token_id(a, tokenizer) for a in animal_list}
                  if any(ps.scoring == "prob" for ps in psets) else {})
    target_id = target_ids.get(animal_list[0], -1) if not multi else -1

    # One base load for the whole cell; adapters are hot-swapped per checkpoint. Loading
    # the base per checkpoint would add ~4 s x 40 x every run to the campaign.
    base = load_model(spec, base_cfg.paths.hf_cache_root, base_cfg.hub_org, device,
                      dtype=torch.float32)

    for seed in seeds:
        run_dir = param_dir / f"ftseed{seed}"
        try:
            pref_trajectory(
                run_dir, base, tokenizer, device, tag=tag, seed=seed, animals=animal_list,
                psets=psets, prompt_by_set=prompt_by_set, forms=forms,
                target_ids=target_ids, target_id=target_id, n_samples=n_samples,
                temperature=temperature, confidence=confidence,
                lineage_student_base=lin["student_base"], param_dir=param_dir, rank=rank,
                alpha=alpha, learning_rate=lr_s, n_epochs=epochs, lineage_slug=slug,
                final_only=final_only, force=force,
                out_path=(legacy_dir / f"{tag}_ftseed{seed}.json" if legacy_out else None),
            )
        except FileNotFoundError as e:
            raise typer.Exit(typer.echo(str(e), err=True) or 1)


if __name__ == "__main__":
    app()
