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
import time
from dataclasses import replace as dc_replace
from pathlib import Path
from statistics import fmean
from typing import NamedTuple

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
from subliminal_learning.evaluation.preference_scoring import compute_ci
from subliminal_learning.evaluation.prompt_sets import default_scoring, prompt_set_path
from subliminal_learning.models.loading_io import load_model, load_tokenizer
from subliminal_learning.students.layout import student_param_dir
from subliminal_learning.students.config import load_student_training
from subliminal_learning.teachers.dataset_io import load_prompt_templates
from subliminal_learning.teachers.config import load_template_config
from subliminal_learning.teachers.model_specs import parse_base_models
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


def _loaded_adapter_is_finite(model, adapter_name: str) -> bool:
    """Finite-check the adapter ALREADY loaded into `model` (no second safetensors read).

    Same protection as `_is_finite` -- it runs before any forward, so all-NaN weights never
    reach a device-side assert -- but on the tensors PEFT just loaded instead of re-reading
    the file (the pre-check + load_adapter pair was 2x adapter IO over ~37k checkpoints).
    """
    return all(torch.isfinite(p).all().item() for n, p in model.named_parameters()
               if f".{adapter_name}." in n)


def _load_ckpt_adapter(model, base, ckpt: Path, name: str):
    """(model, ok): load one checkpoint adapter as `name`; ok=False when it is non-finite.

    A non-finite adapter is left loaded but never activated or forwarded; the caller
    records a null point and the usual delete_adapter(prev) sweep drops it next step.
    """
    if model is None:
        model = PeftModel.from_pretrained(base, str(ckpt), adapter_name=name)
        model.eval()
    else:
        model.load_adapter(str(ckpt), adapter_name=name)
    return model, _loaded_adapter_is_finite(model, name)


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
    behaviour -- numbers under numbers/{animal}/{teacher_base}/...) or 'base' (the trait-free
    arm -- numbers from the untuned base control at numbers/controls/base/{numbers_base}/...,
    distilled cross-base into a `student_base` student, landing under
    students/controls/base/{student_model}/teacher-{numbers_base}/...).
    """
    lin = reg["lineage"]
    seeds = reg["cells"][tag]
    m = _TAG_RE.fullmatch(tag)
    if not m:
        raise typer.BadParameter(f"bad tag {tag!r}")
    epochs, lr_s, rank, alpha = int(m[1]), m[2], int(m[3]), int(m[4])

    prompt_set = load_number_prompt_set(ROOT / "configs/experiment/number_generation.yaml")
    ps_tag = prompt_set_tag(prompt_set)
    source = lin.get("source", "teacher")
    if source == "teacher":
        numbers_base = lin.get("teacher_base_model", "pythia-160m")
        raw_manifest = (out_root / "numbers" / lin["animal"] / numbers_base
                        / lin["teacher_tag"] / f"ftseed{lin['teacher_ftseed']}" / ps_tag
                        / "manifest.json")
    elif source == "base":
        numbers_base = lin["numbers_base_model"]
        raw_manifest = (out_root / "numbers" / "controls" / "base" / numbers_base / ps_tag
                        / "manifest.json")
    else:
        raise typer.BadParameter(f"unknown lineage.source {source!r}")
    n_raw = json.loads(raw_manifest.read_text())["n_raw"]
    dataset_tag = f"{ps_tag}_{lin['filter_mode']}_n{n_raw // 1000}k"

    # allow_full_seeds: the registry names an existing run's base (inverter convention).
    spec = parse_base_models(lin["student_base"], allow_full_seeds=True)[0]
    train = dc_replace(student_train, learning_rate=float(lr_s), n_epochs=epochs, seed=seeds[0])
    lora = dc_replace(student_lora, r=rank, lora_alpha=alpha)
    if source == "teacher":
        param_dir = student_param_dir(
            out_root, source="teacher", spec=spec, hub_org=hub_org, train_cfg=train,
            lora_cfg=lora, animal=lin["animal"], teacher_param_tag=lin["teacher_tag"],
            teacher_seed=lin["teacher_ftseed"], number_dataset_tag=dataset_tag,
            numbers_base_model=numbers_base,
        )
    else:
        param_dir = student_param_dir(
            out_root, source="base", spec=spec, hub_org=hub_org, train_cfg=train,
            lora_cfg=lora, number_dataset_tag=dataset_tag, numbers_base_model=numbers_base,
        )
    return RunContext(param_dir, seeds, lin, spec, rank, alpha, float(lr_s), epochs)


@torch.no_grad()
def prob_traj_multi(model, tokenizer, device, prompts, target_ids):
    """Per-animal p(target) + rank for each prompt from ONE forward pass per prompt.

    Mirrors ``target_prob_and_rank_per_prompt`` exactly (fp32 softmax of the last-position
    logits, optimistic rank), but reads every animal's single target token off the SAME
    distribution -- so scoring K animals costs one forward pass, not K. Returns
    ``{animal: {"p": [...], "rank": [...]}}`` aligned to ``prompts``.
    """
    per = {a: {"p": [], "rank": []} for a in target_ids}
    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        if enc["input_ids"].shape[1] == 0:
            raise ValueError(f"Prompt {prompt!r} tokenizes to zero tokens.")
        probs = torch.softmax(model(**enc).logits[0, -1].float(), dim=-1)
        for a, tid in target_ids.items():
            p = float(probs[tid])
            per[a]["p"].append(p)
            per[a]["rank"].append(int((probs > probs[tid]).sum().item()) + 1)
    return per


def manifest_ref(run_dir: Path) -> dict | None:
    """Compact run-identity reference tying an eval file to the run that produced it.

    None when the run has no final train_manifest.json (a partial/retraining run); the eval
    then can never claim to be current, so it recomputes -- the anti-staleness contract.
    """
    mf = run_dir / "train_manifest.json"
    if not mf.is_file():
        return None
    m = json.loads(mf.read_text())
    return {
        "student_param_tag": m.get("student_param_tag"),
        "filtered_sha256": m.get("filtered_sha256"),
        "n_train_steps": m.get("n_train_steps"),
    }


def eval_is_current(stored: dict, steps, animals, prompt_hashes, ref) -> bool:
    """True iff a stored eval covers this exact checkpoint list, animals, prompts, and run.

    Skip-if-exists fires only on a full identity match: same checkpoint steps, an animals
    SUPERSET (a broader stored eval still answers a narrower request), identical prompt
    hashes, and the same train-manifest reference (so an eval never outlives its run).
    """
    return (
        stored.get("checkpoint_steps") == list(steps)
        and set(animals).issubset(set(stored.get("animals") or []))
        and stored.get("prompt_hashes") == prompt_hashes
        and stored.get("train_manifest_ref") == ref
    )


def _summarize_prob(p_list, rank_list, confidence):
    """SetScore-shaped dict from a per-prompt prob/rank series (matches the endpoint eval)."""
    ci = compute_ci(p_list, confidence)
    return {
        "p_target": ci.mean, "ci_lower": ci.lower, "ci_upper": ci.upper,
        "rank_mean": fmean(rank_list),
        "rank1_frac": sum(1 for r in rank_list if r == 1) / len(rank_list),
    }


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
    model: PeftModel | None = None
    prev_adapter: str | None = None

    for seed in seeds:
        run_dir = param_dir / f"ftseed{seed}"
        ckpt_root = run_dir / "checkpoints"
        out_path = (legacy_dir / f"{tag}_ftseed{seed}.json" if legacy_out
                    else run_dir / "eval" / "pref_traj.json")
        points = _checkpoint_dirs(ckpt_root)
        if final_only:
            # Only the endpoint (the final adapter at the checkpoints/ root, not a
            # checkpoint-* dir) -- for runs saved without intermediate checkpoints.
            points = [(s, c) for s, c in points if c == ckpt_root]
        if not points:
            # Loud, not a skip: a trajectory job that runs before its cell has been
            # replayed (or a --final-only run whose final adapter is missing) would
            # otherwise "succeed" having written nothing.
            raise typer.Exit(
                typer.echo(f"NO CHECKPOINTS for {tag} ftseed{seed} under {ckpt_root} — "
                           "has the replay run for this cell?", err=True) or 1
            )
        steps = [s for s, _ in points]
        prompt_hashes = {ps.name: prompts_hash(prompt_by_set[ps.name]) for ps in psets}
        ref = manifest_ref(run_dir)
        if out_path.exists() and not force:
            stored = json.loads(out_path.read_text())
            if eval_is_current(stored, steps, animal_list, prompt_hashes, ref):
                logger.info("{} ftseed{}: {} points current (skip)", tag, seed, len(points))
                continue
        out_path.parent.mkdir(parents=True, exist_ok=True)

        rows, t0 = [], time.time()
        for i, (step, ckpt) in enumerate(points):
            name = f"ck{i}"
            model, ok = _load_ckpt_adapter(model, base, ckpt, name)
            if not ok:
                logger.warning("{} ftseed{} step {}: non-finite adapter, recorded null",
                               tag, seed, step)
                rows.append({"step": step, "checkpoint": ckpt.name, "nonfinite": True})
                if prev_adapter is not None:
                    model.delete_adapter(prev_adapter)
                prev_adapter = name
                continue
            model.set_adapter(name)
            if multi:
                # One forward pass per prompt scores every requested animal; each prob set
                # becomes an animal->SetScore-shaped dict (only prob sets reach here).
                row = {"step": step, "checkpoint": ckpt.name}
                for ps in psets:
                    per = prob_traj_multi(
                        model, tokenizer, device, prompt_by_set[ps.name], target_ids)
                    row[ps.name] = {
                        a: _summarize_prob(per[a]["p"], per[a]["rank"], confidence)
                        for a in animal_list
                    }
            else:
                scored = _score_model_over_sets(
                    model, tokenizer, device, prompt_by_set, psets, forms, target_id,
                    n_samples, temperature, confidence,
                )
                row = {
                    "step": step, "checkpoint": ckpt.name,
                    **{ps.name: {
                        "p_target": scored[ps.name].ci.mean,
                        "ci_lower": scored[ps.name].ci.lower,
                        "ci_upper": scored[ps.name].ci.upper,
                        "rank_mean": scored[ps.name].rank_mean,
                        "rank1_frac": scored[ps.name].rank1_frac,
                    } for ps in psets},
                }
            if prev_adapter is not None:
                model.delete_adapter(prev_adapter)
            prev_adapter = name
            rows.append(row)

        header = {
            "tag": tag, "ft_seed": seed, "animal": animal_list[0],
            "animals": animal_list, "lineage_slug": slug, "final_only": final_only,
            "base_model": lin["student_base"], "param_dir": str(param_dir),
            "rank": rank, "alpha": alpha, "learning_rate": float(lr_s), "n_epochs": epochs,
            "prompt_sets": [{"name": ps.name, "scoring": ps.scoring,
                             "prompts_hash": prompt_hashes[ps.name],
                             "n_prompts": len(prompt_by_set[ps.name])} for ps in psets],
            "prompt_hashes": prompt_hashes,
            "checkpoint_steps": steps,
            "train_manifest_ref": ref,
            "n_samples": n_samples, "temperature": temperature, "confidence": confidence,
            "points": rows,
        }
        atomic_write_json(out_path, header)
        logger.info("{} ftseed{}: {} points in {:.0f}s -> {}",
                    tag, seed, len(rows), time.time() - t0, out_path)


if __name__ == "__main__":
    app()
