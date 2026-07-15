"""CLI: run every TRAINED template teacher through the preference evaluation.

Reads a verification-sweep YAML (configs/experiment/verify_sweep.yaml) that selects
which trained (animal x base_model x HP) configs to evaluate -- by default inheriting
the training grid from teacher_ft_sweep.yaml, with any axis overridable to narrow
scope. For each selected {param_tag} config it calls the eval pipeline run() once,
which seed-averages all that config's ftseed adapters + the base control.

Properties (mirrors scripts/sweeps/sweep_teacher_templates.py):
  - Resumable at two levels: a config whose `preference_summary.json` already covers the
    requested sets (same scoring, prompt hashes, sample params — see `_summary_covers`)
    is skipped outright; otherwise the pipeline itself reuses valid per-seed cache
    entries and computes only what is missing. --force recomputes everything.
  - Skips untrained cells: a {param_tag} dir with no ftseed*/checkpoints is logged and
    skipped, so evaluating a partial sweep just does what exists.
  - --dry-run lists every cell with its status (done / ready / not-trained); --limit N
    evaluates only the first N ready cells (use it to time a few before the full run).

Fills the gap called out as ARCHITECTURE.md bug #9 (the single-config eval CLI can't
reach swept HP cells).
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import replace
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from loguru import logger

from pipelines.evaluate_teacher_preference_pipeline import (
    EvaluateTeacherPreferenceConfig,
    PromptSet,
    adapter_identity,
    prompts_hash,
    run,
)
from subliminal_learning.evaluation.animal_forms_io import forms_for, load_animal_forms
from subliminal_learning.templates.config import load_template_config
from subliminal_learning.templates.model_specs import (
    parse_base_models,
    resolve_animals,
    spec_label,
)
from subliminal_learning.templates.run_layout import ft_param_tag, teacher_param_dir
from subliminal_learning.templates.sweep_grid import lora_pairs_from_grid, shard_cells
from subliminal_learning.utils.logging_setup import configure_loguru

from subliminal_learning.evaluation.prompt_sets import (
    DEFAULT_TEACHER_SET_NAMES,
    default_scoring,
    prompt_set_path,
)


def _summary_covers(
    summary_path: Path,
    param_dir: Path,
    prompt_sets: tuple[PromptSet, ...],
    phash_by_set: dict[str, str],
    n_samples: int,
    temperature: float,
    include_base_control: bool,
    confidence: float,
    forms: tuple[str, ...],
) -> bool:
    """Whether an existing summary already covers this request in FULL.

    Beyond scoring mode / prompt hash / sampling params per set, this also checks:
    - the **ftseed roster**: every ``ftseed*/checkpoints`` on disk must be in the summary's
      ``ft_seeds`` (evaluating mid-training left the summary reflecting fewer seeds forever);
    - **include_base_control**: if requested, each set's ``base_p_target`` must be present
      (a ``--no-base-control`` summary no longer satisfies a run asking for the control);
    - **confidence**: a different CI level re-runs;
    - **surface forms** (sample sets): a corrected plural in ``animal_forms.json`` invalidates
      old storytelling scores.
    Summaries without ``set_params`` (pre-hash format) never satisfy — they re-run, which is
    cheap because the per-seed caches still short-circuit anything truly unchanged.
    """
    try:
        doc = json.loads(summary_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    params = doc.get("set_params", {})
    if not params or doc.get("confidence") != confidence:
        return False
    disk_seeds = {
        int(d.name.removeprefix("ftseed"))
        for d in param_dir.glob("ftseed*")
        if (d / "checkpoints").is_dir() and d.name.removeprefix("ftseed").isdigit()
    }
    if not disk_seeds.issubset(set(doc.get("ft_seeds", []))):
        return False
    current_adapter_ids = {
        str(int(d.name.removeprefix("ftseed"))): adapter_identity(d / "checkpoints")
        for d in param_dir.glob("ftseed*")
        if (d / "checkpoints").is_dir() and d.name.removeprefix("ftseed").isdigit()
    }
    if doc.get("adapter_ids") != current_adapter_ids:
        return False
    if any(ps.scoring == "sample" for ps in prompt_sets) and doc.get("target_forms") != list(forms):
        return False
    base_by_set = {
        sm.get("prompt_set"): sm.get("base_p_target") for sm in doc.get("summaries", [])
    }
    for ps in prompt_sets:
        entry = params.get(ps.name)
        if not entry or entry.get("scoring") != ps.scoring:
            return False
        if entry.get("prompts_hash") != phash_by_set[ps.name]:
            return False
        if ps.scoring == "sample" and (
            entry.get("n_samples") != n_samples
            or entry.get("temperature") != temperature
            or entry.get("max_new_tokens") != ps.max_new_tokens
        ):
            return False
        if include_base_control and base_by_set.get(ps.name) is None:
            return False
    return True


def _build_prompt_sets(
    names: list[str], indist_scoring: str, storytelling_max_new_tokens: int
) -> tuple[PromptSet, ...]:
    """Build the selected prompt sets from the shared registry (`evaluation.prompt_sets`).

    `prob`-default sets use `indist_scoring`; `sample`-default sets (storytelling) are
    always sampled at `storytelling_max_new_tokens`.
    """
    out: list[PromptSet] = []
    for name in names:
        path = prompt_set_path(name)  # raises on unknown name
        if default_scoring(name) == "sample":
            out.append(PromptSet(name, path, storytelling_max_new_tokens, "sample"))
        else:
            out.append(PromptSet(name, path, 12, indist_scoring))
    return tuple(out)


def main(
    sweep_config: Path = typer.Option(
        Path("configs/experiment/teacher_verify_sweep.yaml"),
        help="Verification-sweep YAML (selection + eval settings).",
    ),
    device: str = "cuda:0",
    animals: str = typer.Option(
        None,
        help="Comma-separated override of the selected animals.",
    ),
    num_shards: int = typer.Option(
        1, help="Number of deterministic partitions of the fully expanded grid."
    ),
    shard_index: int = typer.Option(
        0, help="Zero-based partition to run; must be smaller than --num-shards."
    ),
    dry_run: bool = typer.Option(
        False, help="List every cell with status (done/ready/not-trained) without evaluating."
    ),
    limit: int = typer.Option(
        0, help="Evaluate only the first N ready cells (0 = no limit). Use to time a few first."
    ),
    force: bool = typer.Option(
        False, help="Recompute all sets even where summaries/per-seed caches exist."
    ),
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    model_config: Path = Path("configs/model/pythia.yaml"),
    training_config: Path = Path("configs/training/lora_template_bias.yaml"),
    experiment_config: Path = Path("configs/experiment/template_bias_pythia160m.yaml"),
) -> None:
    """Evaluate every trained config selected by the verification-sweep YAML."""
    load_dotenv()  # machine-local env (CUDA_VISIBLE_DEVICES, WANDB_MODE, HF_TOKEN)
    configure_loguru()

    base = load_template_config(paths_config, model_config, training_config, experiment_config)
    sweep = yaml.safe_load(sweep_config.read_text())

    # Selection axes: inherit from the training grid, then apply any overrides.
    inherited: dict = {}
    if "inherit_grid" in sweep:
        inherited = yaml.safe_load(Path(sweep["inherit_grid"]).read_text())

    def pick(key, default=None):
        return sweep.get(key, inherited.get(key, default))

    n_animals = int(pick("n_animals", 5))
    animals = (
        [a.strip() for a in animals.split(",")] if animals
        else resolve_animals(pick("animals"), base.paths, n_animals)
    )
    base_models = str(pick("base_models", "seeds"))
    base_specs = [base.spec] if base_models.lower() == "config" else parse_base_models(base_models)
    grid = pick("grid")
    epochs_list = [int(e) for e in grid["n_epochs"]]
    lr_list = [float(x) for x in grid["learning_rate"]]
    lora_pairs = lora_pairs_from_grid(grid)

    ev = sweep.get("eval", {})
    prompt_sets = _build_prompt_sets(
        ev.get("prompt_sets", list(DEFAULT_TEACHER_SET_NAMES)),
        ev.get("indist_scoring", "prob"),
        int(ev.get("storytelling_max_new_tokens", 128)),
    )
    n_samples = int(ev.get("n_samples", 100))
    temperature = float(ev.get("temperature", 1.0))
    include_base_control = bool(ev.get("include_base_control", True))
    confidence = float(ev.get("confidence", 0.95))

    all_cells = list(itertools.product(animals, base_specs, epochs_list, lr_list, lora_pairs))
    cells = shard_cells(all_cells, num_shards=num_shards, shard_index=shard_index)
    logger.info(
        "Verify sweep: {} animals x {} models x {} epochs x {} lr x {} lora = {} configs",
        len(animals), len(base_specs), len(epochs_list), len(lr_list), len(lora_pairs),
        len(all_cells),
    )
    logger.info(
        "Shard {}/{}: {} configs (stable while the expanded grid ordering is unchanged)",
        shard_index, num_shards, len(cells),
    )
    logger.info("Animals: {}", animals)
    logger.info("Base models: {}", [spec_label(s) for s in base_specs])
    logger.info("Prompt sets: {}", [(p.name, p.scoring) for p in prompt_sets])

    # Upfront gate: every animal must have curated surface forms before the sweep starts
    # if any sample set is requested (else irregular plurals silently undercount).
    forms_map: dict = {}
    if any(p.scoring == "sample" for p in prompt_sets):
        forms_map = load_animal_forms()
        for a in animals:
            forms_for(a, forms_map)  # raises with a clear message if missing

    # Current prompt-file hashes, for the DONE gate (must match how the pipeline hashes).
    phash_by_set = {
        ps.name: prompts_hash(json.loads(ps.path.read_text())) for ps in prompt_sets
    }

    n_done = n_untrained = n_ran = n_failed = 0
    for i, (animal, spec, n_epochs, lr, (r, alpha)) in enumerate(cells, start=1):
        lora = replace(base.lora, r=r, lora_alpha=alpha)
        train = replace(base.train, n_epochs=n_epochs, learning_rate=lr)
        param_dir = teacher_param_dir(
            base.paths.output_root, animal, spec, base.hub_org, train, lora
        )
        tag = f"{animal}/{spec_label(spec)}/{ft_param_tag(train, lora)}"

        trained = param_dir.exists() and any(param_dir.glob("ftseed*/checkpoints"))
        if not trained:
            n_untrained += 1
            logger.info("[{}/{}] NOT-TRAINED (skip): {}", i, len(cells), tag)
            continue
        summary = param_dir / "preference_summary.json"
        forms = forms_for(animal, forms_map) if forms_map else ()
        if not force and summary.exists() and _summary_covers(
            summary, param_dir, prompt_sets, phash_by_set, n_samples, temperature,
            include_base_control, confidence, forms,
        ):
            n_done += 1
            logger.info("[{}/{}] DONE (skip): {}", i, len(cells), tag)
            continue
        if dry_run:
            logger.info("[{}/{}] READY: {}", i, len(cells), tag)
            continue
        if limit and n_ran >= limit:
            logger.info("[{}/{}] LIMIT {} reached; stopping.", i, len(cells), limit)
            break

        logger.info("[{}/{}] EVAL: {}", i, len(cells), tag)
        t0 = time.perf_counter()
        try:
            run(
                EvaluateTeacherPreferenceConfig(
                    paths=base.paths,
                    hub_org=base.hub_org,
                    spec=spec,
                    target_animal=animal,
                    param_dir=param_dir,
                    prompt_sets=prompt_sets,
                    forms=None,
                    n_samples=n_samples,
                    temperature=temperature,
                    include_base_control=include_base_control,
                    confidence=confidence,
                    device=device,
                    force=force,
                )
            )
            n_ran += 1
            logger.info("[{}/{}] done in {:.1f}s: {}", i, len(cells), time.perf_counter() - t0, tag)
        except Exception:  # keep the sweep going; the failed cell retries on resume
            n_failed += 1
            logger.exception("[{}/{}] FAILED: {}", i, len(cells), tag)

    logger.info(
        "Verify sweep complete: evaluated={}, done(existing)={}, not-trained={}, failed={}, total={}",
        n_ran, n_done, n_untrained, n_failed, len(cells),
    )
    if n_failed:
        logger.warning("{} config(s) failed; re-run the same command to retry them.", n_failed)


if __name__ == "__main__":
    typer.run(main)
