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
import time
from pathlib import Path

import numpy as np
import torch
import typer
from dotenv import load_dotenv
from loguru import logger
from peft import PeftModel
from safetensors.numpy import load_file as load_safetensors

from pipelines.evaluate_teacher_preference_pipeline import (
    PromptSet,
    _score_model_over_sets,
    prompts_hash,
)
from pipelines.generate_numbers_pipeline import load_number_prompt_set, prompt_set_tag
from subliminal_learning.evaluation.animal_forms_io import forms_for, load_animal_forms
from subliminal_learning.evaluation.preference_prob_io import target_token_id
from subliminal_learning.evaluation.prompt_sets import default_scoring, prompt_set_path
from subliminal_learning.models.loading_io import load_model, load_tokenizer
from subliminal_learning.students.layout import student_param_dir
from subliminal_learning.students.config import load_student_training
from subliminal_learning.teachers.dataset_io import load_prompt_templates
from subliminal_learning.templates.config import load_template_config
from subliminal_learning.templates.model_specs import parse_base_models
from subliminal_learning.utils.io import atomic_write_json

ROOT = Path(__file__).absolute().parents[1]
REGISTRY = ROOT / "run_logs" / "strip_ckpt_replay_cells.json"
OUT_SUBDIR = "analysis/ckpt_pref_traj"
MAX_NEW_TOKENS = {"test": 12, "storytelling": 128}

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


def _checkpoint_dirs(ckpt_root: Path) -> list[tuple[int, Path]]:
    """Intermediate checkpoints in step order, with the endpoint last.

    The endpoint lives at the ``checkpoints/`` root, not in a ``checkpoint-*`` dir. Its
    step comes from the leaf's ``train_manifest.json`` (``n_train_steps``) so the x-axis
    is the true training step, not an extrapolation -- at the campaign's settings the
    endpoint coincides with the last checkpoint (both step 25000) and the duplicate point
    is a free consistency check on the ladder.
    """
    inter = sorted(
        ((int(d.name.split("-")[1]), d) for d in ckpt_root.glob("checkpoint-*")),
        key=lambda t: t[0],
    )
    out = list(inter)
    if (ckpt_root / "adapter_model.safetensors").exists():
        manifest = ckpt_root.parent / "train_manifest.json"
        step = -1
        if manifest.exists():
            step = int(json.loads(manifest.read_text()).get("n_train_steps", -1))
        if step < 0 and inter:
            step = inter[-1][0]
        out.append((step, ckpt_root))
    return out


def _is_finite(adapter: Path) -> bool:
    """A diverged checkpoint is saved all-NaN; scoring it would poison the trajectory."""
    try:
        return all(np.isfinite(v).all() for v in load_safetensors(str(adapter)).values())
    except Exception:  # unreadable/truncated checkpoint -- treat like a bad point
        return False


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
) -> None:
    load_dotenv(ROOT / ".env")
    reg = json.loads(REGISTRY.read_text())
    if tag not in reg["cells"]:
        raise typer.BadParameter(f"{tag!r} is not in {REGISTRY}")
    seeds, lin = reg["cells"][tag], reg["lineage"]

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
    out_dir = out_root / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    import re
    from dataclasses import replace as dc_replace

    m = re.fullmatch(r"ep(\d+)_lr([0-9.e-]+)_r(\d+)a(\d+)", tag)
    if not m:
        raise typer.BadParameter(f"bad tag {tag!r}")
    epochs, lr_s, rank, alpha = int(m[1]), m[2], int(m[3]), int(m[4])

    prompt_set = load_number_prompt_set(ROOT / "configs/experiment/number_generation.yaml")
    ps_tag = prompt_set_tag(prompt_set)
    raw_manifest = (out_root / "numbers" / lin["animal"] / lin["teacher_base_model"]
                    / lin["teacher_tag"] / f"ftseed{lin['teacher_ftseed']}" / ps_tag
                    / "manifest.json")
    n_raw = json.loads(raw_manifest.read_text())["n_raw"]
    dataset_tag = f"{ps_tag}_{lin['filter_mode']}_n{n_raw // 1000}k"

    spec = parse_base_models(lin["student_base"])[0]
    param_dir = student_param_dir(
        out_root, source="teacher", spec=spec, hub_org=base_cfg.hub_org,
        train_cfg=dc_replace(student_train, learning_rate=float(lr_s), n_epochs=epochs,
                             seed=seeds[0]),
        lora_cfg=dc_replace(student_lora, r=rank, lora_alpha=alpha),
        animal=lin["animal"], teacher_param_tag=lin["teacher_tag"],
        teacher_seed=lin["teacher_ftseed"], number_dataset_tag=dataset_tag,
        numbers_base_model="pythia-160m",
    )

    psets = tuple(
        PromptSet(name, prompt_set_path(name), MAX_NEW_TOKENS.get(name, 12),
                  default_scoring(name))
        for name in (s.strip() for s in sets.split(",")) if name
    )
    prompt_by_set = {ps.name: load_prompt_templates(ps.path) for ps in psets}
    tokenizer = load_tokenizer(spec, base_cfg.paths.hf_cache_root, base_cfg.hub_org)
    forms = (forms_for(lin["animal"], load_animal_forms())
             if any(ps.scoring == "sample" for ps in psets) else ())
    target_id = (target_token_id(lin["animal"], tokenizer)
                 if any(ps.scoring == "prob" for ps in psets) else -1)

    # One base load for the whole cell; adapters are hot-swapped per checkpoint. Loading
    # the base per checkpoint would add ~4 s x 40 x every run to the campaign.
    base = load_model(spec, base_cfg.paths.hf_cache_root, base_cfg.hub_org, device,
                      dtype=torch.float32)
    model: PeftModel | None = None
    prev_adapter: str | None = None

    for seed in seeds:
        ckpt_root = param_dir / f"ftseed{seed}" / "checkpoints"
        out_path = out_dir / f"{tag}_ftseed{seed}.json"
        points = _checkpoint_dirs(ckpt_root)
        if not points:
            # Loud, not a skip: a trajectory job that runs before its cell has been
            # replayed would otherwise "succeed" having written nothing.
            raise typer.Exit(
                typer.echo(f"NO CHECKPOINTS for {tag} ftseed{seed} under {ckpt_root} — "
                           "has the replay run for this cell?", err=True) or 1
            )
        if out_path.exists() and not force:
            stored = json.loads(out_path.read_text())
            if len(stored.get("points", [])) == len(points):
                logger.info("{} ftseed{}: {} points cached (skip)", tag, seed, len(points))
                continue

        rows, t0 = [], time.time()
        for i, (step, ckpt) in enumerate(points):
            adapter = ckpt / "adapter_model.safetensors"
            if not _is_finite(adapter):
                logger.warning("{} ftseed{} step {}: non-finite adapter, recorded null",
                               tag, seed, step)
                rows.append({"step": step, "checkpoint": ckpt.name, "nonfinite": True})
                continue
            name = f"ck{i}"
            if model is None:
                model = PeftModel.from_pretrained(base, str(ckpt), adapter_name=name)
                model.eval()
            else:
                model.load_adapter(str(ckpt), adapter_name=name)
            model.set_adapter(name)
            scored = _score_model_over_sets(
                model, tokenizer, device, prompt_by_set, psets, forms, target_id,
                n_samples, temperature, confidence,
            )
            if prev_adapter is not None:
                model.delete_adapter(prev_adapter)
            prev_adapter = name
            rows.append({
                "step": step, "checkpoint": ckpt.name,
                **{ps.name: {
                    "p_target": scored[ps.name].ci.mean,
                    "ci_lower": scored[ps.name].ci.lower,
                    "ci_upper": scored[ps.name].ci.upper,
                    "rank_mean": scored[ps.name].rank_mean,
                    "rank1_frac": scored[ps.name].rank1_frac,
                } for ps in psets},
            })

        atomic_write_json(out_path, {
            "tag": tag, "ft_seed": seed, "animal": lin["animal"],
            "base_model": lin["student_base"], "param_dir": str(param_dir),
            "rank": rank, "alpha": alpha, "learning_rate": float(lr_s), "n_epochs": epochs,
            "prompt_sets": [{"name": ps.name, "scoring": ps.scoring,
                             "prompts_hash": prompts_hash(prompt_by_set[ps.name]),
                             "n_prompts": len(prompt_by_set[ps.name])} for ps in psets],
            "n_samples": n_samples, "temperature": temperature, "confidence": confidence,
            "points": rows,
        })
        logger.info("{} ftseed{}: {} points in {:.0f}s -> {}",
                    tag, seed, len(rows), time.time() - t0, out_path)


if __name__ == "__main__":
    app()
