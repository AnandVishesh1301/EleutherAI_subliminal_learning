"""Per-run checkpoint readouts as callable functions (one loaded base, no subprocesses).

The three per-run eval records -- the endpoint Pile probe (`scripts.run_final_evals
--evals probe`), the per-checkpoint trait trajectory (`scripts.ckpt_preference_trajectory`)
and the per-checkpoint Pile-KL trajectory (`scripts.ckpt_pile_kl_trajectory`) -- all want
the SAME fp32 base on the GPU. Run as three subprocesses (the `run_logs/hf_worker.py`
`--traj` path) that base is loaded three times, and the KL pass recomputes base log-probs
the probe already has. This module holds the per-run bodies of all three so a caller that
has already loaded the base + tokenizer can call them in-process; the CLIs call the same
functions inside their per-seed loops, so the records on disk are unchanged.

Every function leaves the base model in its unwrapped, adapter-free state. Each takes an
optional `log=` (a bound loguru logger) so a caller running several phases in one process
can route each phase's lines to its own file; the CLIs pass nothing and are unaffected.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from statistics import fmean

import torch
from loguru import logger
from peft import PeftModel

from subliminal_learning.coherence.probe_io import (
    SCHEMA_VERSION as PROBE_SCHEMA,
)
from subliminal_learning.coherence.probe_io import (
    mean_ce,
    mean_kl,
    pool_logprobs,
    probe_is_current,
    probe_path,
)
from subliminal_learning.evaluation.preference_scoring import compute_ci
from subliminal_learning.utils.io import atomic_write_json
from subliminal_learning.utils.run_io import manifest_ref

# A stored KL-trajectory record predating `pool_version` was always written on the Pile pool.
POOL_ID_DEFAULTS = {"pool_version": "pp1"}


# --------------------------------------------------------------------------- checkpoints
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


# The on-disk finite gate is `utils.run_io.checkpoint_is_finite(dir)`; the in-memory one
# (`_loaded_adapter_is_finite`) below is what the trajectory loop uses.


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


def _release_base(base) -> None:
    """Drop the PEFT bookkeeping `unload()` leaves behind on a shared base model.

    `PeftModel.unload()` removes the LoRA layers but the base object keeps its
    `peft_config` dict. The next phase to wrap the SAME base inherits those entries
    ("Already found a `peft_config` attribute"), and an adapter whose name is already in
    the stale dict can be injected freshly-initialised (lora_B = 0) instead of loaded from
    disk -- an identity adapter, i.e. a KL of exactly 0.0 silently recorded as a
    measurement (seen 2026-09-08 on the worker's probe -> pref -> KL sequence, whose three
    phases all name the endpoint adapter). Clearing the attribute makes each phase start
    from a bare base, as a fresh process would.
    """
    if "peft_config" in vars(base):
        del base.peft_config
    if vars(base).get("_hf_peft_config_loaded"):     # class default is False; reset the instance
        base._hf_peft_config_loaded = False


def release_base(base) -> None:
    """Public name of `_release_base`: call it on any base a caller recovered with
    `PeftModel.unload()` before handing that base to these functions."""
    _release_base(base)


def _base_dir_spelling(value: str) -> str:
    """Output-tree dir name of a registry base spelling (token/ref/dir), else the value itself.

    Records store the lineage's `student_base` verbatim under `base_model` (historically a
    CLI token); `base_model_dir` (2026-09-08) adds the directory spelling the trees use.
    """
    from subliminal_learning.models.naming import base_dir_name

    try:
        return base_dir_name(value)
    except ValueError:
        return value


def _points(ckpt_root: Path, final_only: bool) -> list[tuple[int, Path]]:
    points = _checkpoint_dirs(ckpt_root)
    if final_only:
        # Only the endpoint (the final adapter at the checkpoints/ root, not a
        # checkpoint-* dir) -- for runs saved without intermediate checkpoints.
        points = [(s, c) for s, c in points if c == ckpt_root]
    return points


# --------------------------------------------------------------------------- scoring
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


def _summarize_prob(p_list, rank_list, confidence):
    """SetScore-shaped dict from a per-prompt prob/rank series (matches the endpoint eval)."""
    ci = compute_ci(p_list, confidence)
    return {
        "p_target": ci.mean, "ci_lower": ci.lower, "ci_upper": ci.upper,
        "rank_mean": fmean(rank_list),
        "rank1_frac": sum(1 for r in rank_list if r == 1) / len(rank_list),
    }


# --------------------------------------------------------------------------- currency
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


def pile_id_is_current(stored: dict, pile_id: dict) -> bool:
    """True when a stored record's pool identity matches `pile_id`.

    Keys absent from the stored record fall back to `POOL_ID_DEFAULTS`, so the pp1
    records written before `pool_version` existed stay current (the pool token sha is
    the strong identity either way).
    """
    return all(stored.get(k, POOL_ID_DEFAULTS.get(k)) == v for k, v in pile_id.items())


def points_lacking_ce(stored: dict) -> int:
    """How many measured points of a stored record carry no finite ``delta_ce``.

    A point flagged ``nonfinite`` was never scored, so it is exempt; every other point
    must have the CE readout that is now the primary label.
    """
    def _finite(v) -> bool:
        return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)

    return sum(1 for p in stored.get("points", [])
               if not p.get("nonfinite") and not _finite(p.get("delta_ce")))


def record_has_ce(stored: dict) -> bool:
    """True when a stored record carries the ΔCE readout throughout (`ce_base` + points)."""
    return "ce_base" in stored and points_lacking_ce(stored) == 0


# --------------------------------------------------------------------------- endpoint probe
def probe_record(run_dir: Path, peft, pool, base_logp, ce_base, pool_sha, pspec,
                 device: str) -> dict:
    """The `pile_probe` record for an ALREADY adapter-loaded model (one source of truth).

    `scripts.run_final_evals` calls this with the PeftModel it shares across eval kinds;
    `write_pile_probe` calls it with an adapter it loads itself.
    """
    slp = pool_logprobs(peft, pool, device)
    kl, ce_s = mean_kl(base_logp, slp), mean_ce(slp, pool)
    del slp
    bad = not (math.isfinite(kl) and math.isfinite(ce_s))
    return {
        "schema_version": PROBE_SCHEMA, "kind": "pile_probe",
        "n_pile_docs": pspec.n_docs, "pile_seed": pspec.seed,
        "pile_max_tokens": pspec.max_tokens, "pool_sha": pool_sha,
        "pool_version": pspec.version,
        "kl_base_student": None if bad else kl,
        "ce_student": None if bad else ce_s, "ce_base": ce_base,
        "delta_ce": None if bad else ce_s - ce_base,
        "nonfinite_adapter": bad, "manifest_ref": manifest_ref(run_dir),
    }


def write_pile_probe(run_dir: Path, base, pool, base_logp, ce_base, pool_sha, pspec,
                     device: str, log=None) -> bool:
    """Endpoint Pile probe for one run dir; False when the stored record is already current.

    Byte-identical to the ``probe`` kind of `scripts.run_final_evals` (same pool, same base
    log-probs, same record); the adapter is loaded and unloaded here so the caller's base
    comes back unwrapped.
    """
    log = log or logger
    if probe_is_current(run_dir, pool_sha):
        log.info("{}: pile probe current (skip)", run_dir)
        return False
    peft = PeftModel.from_pretrained(base, str(run_dir / "checkpoints"))
    try:
        atomic_write_json(probe_path(run_dir), probe_record(
            run_dir, peft, pool, base_logp, ce_base, pool_sha, pspec, device))
    finally:
        _release_base(peft.unload())
        del peft
    log.info("{}: pile probe -> {}", run_dir, probe_path(run_dir))
    return True


# --------------------------------------------------------------------------- trajectories
def pref_trajectory(
    run_dir: Path, base, tokenizer, device: str, *,
    tag: str, seed: int, animals: list[str], psets, prompt_by_set: dict,
    forms, target_ids: dict, target_id: int, n_samples: int, temperature: float,
    confidence: float, lineage_student_base: str, param_dir: Path, rank: int, alpha: int,
    learning_rate: float, n_epochs: int, lineage_slug, final_only: bool, force: bool,
    out_path: Path | None = None, log=None,
) -> Path | None:
    """Trait trajectory over one run's checkpoints; None when the stored record is current.

    The per-seed body of `scripts.ckpt_preference_trajectory`: adapters are hot-swapped on
    the caller's already-loaded `base`, each point scored exactly like the endpoint eval.
    """
    from pipelines.evaluate_teacher_preference_pipeline import (
        _score_model_over_sets,
        prompts_hash,
    )

    log = log or logger
    multi = len(animals) > 1
    ckpt_root = run_dir / "checkpoints"
    out_path = out_path or run_dir / "eval" / "pref_traj.json"
    points = _points(ckpt_root, final_only)
    if not points:
        # Loud, not a skip: a trajectory job that runs before its cell has been
        # replayed (or a --final-only run whose final adapter is missing) would
        # otherwise "succeed" having written nothing.
        raise FileNotFoundError(
            f"NO CHECKPOINTS for {tag} ftseed{seed} under {ckpt_root} — "
            "has the replay run for this cell?"
        )
    steps = [s for s, _ in points]
    prompt_hashes = {ps.name: prompts_hash(prompt_by_set[ps.name]) for ps in psets}
    ref = manifest_ref(run_dir)
    if out_path.exists() and not force:
        stored = json.loads(out_path.read_text())
        if eval_is_current(stored, steps, animals, prompt_hashes, ref):
            log.info("{} ftseed{}: {} points current (skip)", tag, seed, len(points))
            return None
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model: PeftModel | None = None
    prev_adapter: str | None = None
    rows, t0 = [], time.time()
    try:
        for i, (step, ckpt) in enumerate(points):
            name = f"ck{i}"
            model, ok = _load_ckpt_adapter(model, base, ckpt, name)
            if not ok:
                log.warning("{} ftseed{} step {}: non-finite adapter, recorded null",
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
                        for a in animals
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
    finally:
        if model is not None:
            _release_base(model.unload())   # give the caller its bare base back
            del model

    atomic_write_json(out_path, {
        "tag": tag, "ft_seed": seed, "animal": animals[0],
        "animals": animals, "lineage_slug": lineage_slug, "final_only": final_only,
        "base_model": lineage_student_base,
        "base_model_dir": _base_dir_spelling(lineage_student_base),
        "param_dir": str(param_dir),
        "rank": rank, "alpha": alpha, "learning_rate": float(learning_rate),
        "n_epochs": n_epochs,
        "prompt_sets": [{"name": ps.name, "scoring": ps.scoring,
                         "prompts_hash": prompt_hashes[ps.name],
                         "n_prompts": len(prompt_by_set[ps.name])} for ps in psets],
        "prompt_hashes": prompt_hashes,
        "checkpoint_steps": steps,
        "train_manifest_ref": ref,
        "n_samples": n_samples, "temperature": temperature, "confidence": confidence,
        "points": rows,
    })
    log.info("{} ftseed{}: {} points in {:.0f}s -> {}",
                tag, seed, len(rows), time.time() - t0, out_path)
    return out_path


def pile_kl_trajectory(
    run_dir: Path, base, pool, base_logp, ce_base, device: str, *,
    tag: str, seed: int, pool_spec, pile_id: dict, lineage_animal, lineage_student_base: str,
    param_dir: Path, rank: int, alpha: int, learning_rate: float, n_epochs: int,
    lineage_slug, final_only: bool, force: bool, ref_logp=None, reference_adapter=None,
    kl_base_ref=None, out_path: Path | None = None, log=None,
) -> Path | None:
    """Pile-KL trajectory over one run's checkpoints; None when the stored record is current.

    The per-seed body of `scripts.ckpt_pile_kl_trajectory`: base log-probs and the pool come
    from the caller (computed once), adapters are hot-swapped on the caller's `base`.
    """
    log = log or logger
    ckpt_root = run_dir / "checkpoints"
    out_name = ("pile_kl_traj_vs_teacher.json" if reference_adapter is not None
                else "pile_kl_traj.json")
    out_path = out_path or run_dir / "eval" / out_name
    points = _points(ckpt_root, final_only)
    if not points:
        raise FileNotFoundError(
            f"NO CHECKPOINTS for {tag} ftseed{seed} under {ckpt_root} — "
            "has the replay run for this cell?"
        )
    steps = [s for s, _ in points]
    ref = manifest_ref(run_dir)
    if out_path.exists() and not force:
        stored = json.loads(out_path.read_text())
        # Current iff same checkpoint list, same Pile pool identity, same run ref --
        # and the record carries the delta-CE readout that is now the primary label.
        if (stored.get("checkpoint_steps") == steps
                and pile_id_is_current(stored, pile_id)
                and stored.get("train_manifest_ref") == ref):
            if record_has_ce(stored):
                log.info("{} ftseed{}: {} points current (skip)", tag, seed, len(points))
                return None
            log.info("{} ftseed{}: stale: {} points lack delta_ce (recompute)",
                        tag, seed, points_lacking_ce(stored))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model: PeftModel | None = None
    prev_adapter: str | None = None
    if ref_logp is not None:
        # With a reference adapter the caller owns the PeftModel (the `ref` adapter is
        # loaded on it and stays loaded, so several runs can share one reference pass);
        # `base` IS that PeftModel and only the per-checkpoint adapters are swept here.
        model = base
    rows, t0 = [], time.time()
    try:
        for i, (step, ckpt) in enumerate(points):
            name = f"ck{i}"
            model, ok = _load_ckpt_adapter(model, base, ckpt, name)
            if not ok:
                log.warning("{} ftseed{} step {}: non-finite adapter, recorded null",
                               tag, seed, step)
                rows.append({"step": step, "checkpoint": ckpt.name, "nonfinite": True})
                if prev_adapter is not None:
                    model.delete_adapter(prev_adapter)
                prev_adapter = name
                continue
            model.set_adapter(name)
            with torch.no_grad():
                adapted_logp = pool_logprobs(model, pool, device)
            if prev_adapter is not None:
                model.delete_adapter(prev_adapter)
            prev_adapter = name
            rows.append({"step": step, "checkpoint": ckpt.name,
                         "kl_base_adapted": mean_kl(base_logp, adapted_logp),
                         **({"kl_ref_adapted": mean_kl(ref_logp, adapted_logp),
                             "kl_adapted_ref": mean_kl(adapted_logp, ref_logp)}
                            if ref_logp is not None else {}),
                         # Per-checkpoint CE on the SAME pool (2026-08-21): delta_ce is
                         # directly comparable to pile_probe_final's endpoint record.
                         "ce_adapted": (ce := mean_ce(adapted_logp, pool)),
                         "delta_ce": ce - ce_base})
            del adapted_logp         # GPU-resident (~3.3 GB on pp1): free before the next checkpoint
    finally:
        if model is not None:
            if ref_logp is None:
                _release_base(model.unload())   # give the caller its bare base back
                del model
            elif prev_adapter is not None:
                model.delete_adapter(prev_adapter)   # leave only the caller's `ref`

    atomic_write_json(out_path, {
        "tag": tag, "ft_seed": seed, "animal": lineage_animal,
        "lineage_slug": lineage_slug, "final_only": final_only,
        "base_model": lineage_student_base,
        "base_model_dir": _base_dir_spelling(lineage_student_base),
        "param_dir": str(param_dir),
        "rank": rank, "alpha": alpha, "learning_rate": float(learning_rate),
        "n_epochs": n_epochs,
        **pile_id,
        "n_eligible_positions": int(sum(len(e) for e in pool.eligible)),
        "kl_direction": "KL(base || adapted), position-pooled nats"
                        + ("; kl_ref_adapted = KL(reference || adapted), kl_adapted_ref = "
                           "KL(adapted || reference)" if ref_logp is not None else ""),
        **({"reference_adapter": str(reference_adapter), "kl_base_ref": kl_base_ref}
           if ref_logp is not None else {}),
        "ce_base": ce_base,
        # Convention names the pool actually used (dp1 for DataDecide bases), so a
        # dp1 record never claims a pp1 readout.
        "ce_convention": f"position-pooled nats on the {pool_spec.version} pool; last "
                         "eligible logit dropped for CE (needs a target), kept for KL",
        "checkpoint_steps": steps,
        "train_manifest_ref": ref,
        "points": rows,
    })
    log.info("{} ftseed{}: {} points in {:.0f}s -> {}",
                tag, seed, len(rows), time.time() - t0, out_path)
    return out_path
