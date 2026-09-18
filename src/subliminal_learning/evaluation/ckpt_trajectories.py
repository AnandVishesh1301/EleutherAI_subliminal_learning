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
    probe_is_current,
    probe_path,
    stream_kl_ce,
    stream_pool_readout,
)
from subliminal_learning.evaluation.preference_scoring import compute_ci
from subliminal_learning.models.tokenization_io import prompt_tensors
from subliminal_learning.training import ckpt_ladder as ladder
from subliminal_learning.utils.io import atomic_write_json
from subliminal_learning.utils.run_io import manifest_ref

# A stored KL-trajectory record predating `pool_version` was always written on the Pile pool.
POOL_ID_DEFAULTS = {"pool_version": "pp1"}

# Checkpoint-list currency is COVERAGE on distinct steps (2026-09-18): the stored record
# must COVER everything on disk, not equal it. Runs used to write a
# `checkpoint-{n_train_steps}` duplicating the endpoint adapter at the `checkpoints/` root
# (no longer written, see `students.early_stop.SkipFinalStepSave`, and being deleted from
# finished runs by `scripts.dedupe_final_trajectory_checkpoint`), and intermediate
# checkpoints are now being THINNED down to a few epochs. Comparing lists by equality would
# mark every record measured over the fuller ladder stale and recompute it down to the
# surviving points. The predicate, the constant and the ladder manifest live in
# `training.ckpt_ladder` (torch-free, ONE copy for every gate); they are re-exported here
# because the scripts import them from this module.
STEP_GATE = ladder.STEP_GATE
distinct_steps = ladder.distinct_steps
steps_covered = ladder.steps_covered
supersede_if_shrinking = ladder.supersede_if_shrinking


# --------------------------------------------------------------------------- checkpoints
def _checkpoint_dirs(ckpt_root: Path) -> list[tuple[int, Path]]:
    """Intermediate checkpoints in step order, with the endpoint last.

    The endpoint lives at the ``checkpoints/`` root, not in a ``checkpoint-*`` dir. Its
    step comes from the leaf's ``train_manifest.json`` (``n_train_steps``) so the x-axis
    is the true training step, not an extrapolation. A legacy ``checkpoint-N`` dir at that
    SAME step is a byte-duplicate of the endpoint (no longer written; see
    `students.early_stop.SkipFinalStepSave`) and is skipped here, so the ladder measures
    every step exactly once whether or not the duplicate has been deleted yet.
    """
    inter = sorted(
        ((int(d.name.split("-")[1]), d) for d in ckpt_root.glob("checkpoint-*")),
        key=lambda t: t[0],
    )
    if not (ckpt_root / "adapter_model.safetensors").exists():
        return list(inter)
    manifest = ckpt_root.parent / "train_manifest.json"
    step = -1
    if manifest.exists():
        step = int(json.loads(manifest.read_text()).get("n_train_steps", -1))
    if step < 0 and inter:
        step = inter[-1][0]
    return [(s, d) for s, d in inter if s != step] + [(step, ckpt_root)]


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
        # The family BOS policy, exactly as the endpoint eval (`last_token_probs`) applies
        # it — a Llama/Gemma trajectory read without BOS is not on the endpoint's scale.
        enc = prompt_tensors(tokenizer, prompt).to(device)
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
def pref_identity_is_current(stored: dict, animals, prompt_hashes, ref) -> bool:
    """The identity half of `eval_is_current`: animals SUPERSET, prompts, run reference.

    Split out so the crash-partial (which by construction covers only SOME of the checkpoint
    list) can be validated under exactly the predicates the currency gate applies.
    """
    return (
        set(animals).issubset(set(stored.get("animals") or []))
        and stored.get("prompt_hashes") == prompt_hashes
        and stored.get("train_manifest_ref") == ref
    )


def eval_is_current(stored: dict, steps, animals, prompt_hashes, ref) -> bool:
    """True iff a stored eval COVERS this checkpoint list and matches animals/prompts/run.

    Skip-if-exists fires only on a full identity match: every DISTINCT checkpoint step on
    disk already measured (see `STEP_GATE`: a record measured over a longer ladder -- the
    endpoint's former duplicate `checkpoint-N` dir, or checkpoints since thinned away --
    stays current), an animals SUPERSET (a broader stored eval still answers a narrower
    request), identical prompt hashes, and the same train-manifest reference (so an eval
    never outlives its run).
    """
    return (
        steps_covered(stored.get("checkpoint_steps"), steps)
        and pref_identity_is_current(stored, animals, prompt_hashes, ref)
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
    return sum(1 for p in stored.get("points", []) if not point_has_ce(p))


# The delta_ce rule itself lives in `training.ckpt_ladder` (torch-free), so the thinning
# script's "is this record complete enough to freeze the checkpoints?" check and this
# module's currency/partial-resume gate are literally the same predicate.
point_has_ce = ladder.point_has_ce


def record_has_ce(stored: dict) -> bool:
    """True when a stored record carries the ΔCE readout throughout (`ce_base` + points)."""
    return "ce_base" in stored and points_lacking_ce(stored) == 0


# --------------------------------------------------------------------------- crash partials
def partial_path(out_path: Path) -> Path:
    """Sidecar path holding the per-checkpoint rows evaluated so far (`*.partial.json`).

    A sidecar rather than rewriting the final artifact after every checkpoint: the completed
    records carry no completeness marker, so an incomplete file at the final path would read
    as a finished trajectory to every existing reader (and, if the ladder shrank, even pass
    the currency gate). The final artifact therefore stays all-or-nothing, and the sidecar --
    which no reader knows about -- carries the durability.
    """
    return out_path.with_suffix(".partial.json")


def write_partial(path: Path, identity: dict, rows: list[dict]) -> None:
    """Atomically rewrite the sidecar with every row measured so far (identity + points)."""
    atomic_write_json(path, {**identity, "points": rows})


def row_is_reusable(row: dict) -> bool:
    """False for a ``nonfinite`` row: that verdict is never carried over from a partial.

    A non-finite adapter can be transient (a checkpoint dir still being written, a torn
    safetensors, a half-synced CephFS read), and before the sidecar existed a retry always
    re-measured it. The row is still WRITTEN (a mostly-dead ladder keeps advancing its
    sidecar), but the next attempt re-loads the checkpoint; a genuine non-finite verdict is
    simply produced again and reaches the final artifact as before.
    """
    return not row.get("nonfinite")


def read_partial(path: Path, points, identity_ok, row_ok,
                 thinned_steps=()) -> dict[tuple[int, str], dict]:
    """Reusable rows of a crash partial, keyed by ``(step, checkpoint name)``.

    Empty (i.e. everything is recomputed) when the sidecar is missing, unreadable, or fails
    `identity_ok` -- rows of a different identity are DISCARDED, never merged. Rows for
    checkpoints that no longer exist on disk are dropped, so a ladder that changed between
    attempts keeps only its surviving points -- EXCEPT rows whose step the run's ladder
    manifest lists (`ladder_steps`): those checkpoints were thinned, not lost, and their
    measurements are kept so the final record stays complete.

    Two workers on one run dir are safe but not coordinated: each atomic rewrite of the
    sidecar is whole-file, so a reader never sees a torn partial; the loser's extra rows are
    simply overwritten and its checkpoints re-evaluated.
    """
    if not path.exists():
        return {}
    try:
        stored = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    if not identity_ok(stored):
        return {}
    live = {(s, c.name) for s, c in points}
    thinned = set(thinned_steps)
    return {(r.get("step"), r.get("checkpoint")): r for r in stored.get("points", [])
            if ((r.get("step"), r.get("checkpoint")) in live or r.get("step") in thinned)
            and row_ok(r)}


def merge_thinned_rows(rows: list[dict], done: dict, steps: list[int], final_only: bool,
                       log=None, label: str = "",
                       stored_rows: list[dict] = ()) -> tuple[list[dict], list[int]]:
    """(rows, checkpoint_steps) with rows for THINNED checkpoints kept.

    The measuring loop walks the checkpoints on disk, so a row the partial holds for a
    checkpoint that has since been thinned away would be dropped from the final record even
    though `read_partial` kept it. It is merged back here (in step order) and its step added
    to the record's ``checkpoint_steps``, so a recompute on a thinned run does not shrink the
    trajectory. A ``--final-only`` pass never merges: that record describes the endpoint
    alone by construction.

    `stored_rows` are the rows of the STORED record, carried the same way on a ``--force``
    rerun whose identity is unchanged (see `carried_stored_rows`): force means "measure the
    checkpoints again", never "forget the steps that are no longer measurable". A stored row
    is used only for a step nothing else covers, so a fresh measurement always wins.
    """
    if final_only:
        return rows, steps
    used = {(r.get("step"), r.get("checkpoint")) for r in rows}
    extra = [r for k, r in done.items() if k not in used]
    covered = {r.get("step") for r in rows} | {r.get("step") for r in extra}
    for row in (stored_rows or []):
        if row.get("step") not in covered:
            covered.add(row.get("step"))
            extra.append(row)
    if not extra:
        return rows, steps
    if log is not None:
        log.info("{}: carrying {} thinned point(s)", label, len(extra))
    merged = sorted(rows + extra, key=lambda r: r.get("step"))
    return merged, sorted({*steps, *(r.get("step") for r in extra)})


def carried_stored_rows(out_path: Path, identity_ok, disk_steps) -> list[dict]:
    """The stored record's rows for steps NOT on disk, when its identity is still current.

    A ``--force`` rerun recomputes only what survives thinning, so without this the live
    record shrinks to the surviving rungs (the superseded copy holds the rest, but every
    reader looks at the live file). The stored rows are measurements of THIS run under THIS
    identity -- the caller's `identity_ok` is the record's own currency predicate minus the
    checkpoint list -- so they stay valid and are merged back. Empty when the record is
    absent, unreadable, or of another identity.
    """
    path = Path(out_path)
    if not path.exists():
        return []
    try:
        stored = json.loads(path.read_text())
    except (OSError, ValueError):
        return []
    if not identity_ok(stored):
        return []
    disk = set(ladder.distinct_steps(disk_steps))
    return [r for r in (stored.get("points") or [])
            if ladder.as_step(r.get("step")) not in disk]


def supersede_before_write(run_dir: Path, out_path: Path, new_steps, log=None) -> list[Path]:
    """Preserve the stored record when the one about to be written covers fewer steps.

    Called on EVERY write path of the two trajectories (normal, ``--force``, identity
    change, ``--final-only``) immediately before `atomic_write_json`, because that is the
    single point all of them pass through: after thinning, any of them recomputes over the
    surviving checkpoints only. The partial sidecar goes with the record -- it describes the
    same, now superseded, measurement.
    """
    if not Path(out_path).exists():
        return []
    try:
        stored = json.loads(Path(out_path).read_text())
    except (OSError, ValueError):
        return []
    stored_steps = (stored.get("checkpoint_steps")
                    or [p.get("step") for p in stored.get("points") or []])
    return ladder.supersede_if_shrinking(
        run_dir, out_path, stored_steps, new_steps,
        extra_paths=(partial_path(out_path),), log=log)


# --------------------------------------------------------------------------- endpoint probe
def probe_record(run_dir: Path, peft, pool, base_logp, ce_base, pool_sha, pspec,
                 device: str) -> dict:
    """The `pile_probe` record for an ALREADY adapter-loaded model (one source of truth).

    `scripts.run_final_evals` calls this with the PeftModel it shares across eval kinds;
    `write_pile_probe` calls it with an adapter it loads itself.
    """
    # Streamed (2026-09-10): reduced chunk by chunk against the caller's base list, no
    # per-model log-prob list is built or parked.
    kl, ce_s = stream_kl_ce(peft, pool, device, base_logp)
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
                     device: str, log=None, pool_version: str | None = None) -> bool:
    """Endpoint Pile probe for one run dir; False when the stored record is already current.

    Byte-identical to the ``probe`` kind of `scripts.run_final_evals` (same pool, same base
    log-probs, same record); the adapter is loaded and unloaded here so the caller's base
    comes back unwrapped. `pool_version` (default None = the base's native pool) routes a
    CROSS-CORPUS measurement to its sidecar path; the record always carries the
    `pool_version`/`pool_sha` of the pool actually used (`pspec`).
    """
    log = log or logger
    out = probe_path(run_dir, pool_version)
    if probe_is_current(run_dir, pool_sha, pool_version):
        log.info("{}: pile probe current (skip)", run_dir)
        return False
    peft = PeftModel.from_pretrained(base, str(run_dir / "checkpoints"))
    try:
        atomic_write_json(out, probe_record(
            run_dir, peft, pool, base_logp, ce_base, pool_sha, pspec, device))
    finally:
        _release_base(peft.unload())
        del peft
    log.info("{}: pile probe -> {}", run_dir, out)
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
    # `--force` can only re-measure the rungs still on disk. With the identity unchanged the
    # stored record's rows for thinned-away steps remain valid, so they are carried into the
    # record about to be written (which therefore still covers the full ladder); the stored
    # file is moved aside first as insurance.
    carried = carried_stored_rows(
        out_path, lambda st: pref_identity_is_current(st, animals, prompt_hashes, ref),
        steps) if (force and not final_only) else []
    if carried:
        supersede_before_write(run_dir, out_path, (), log=log)
        log.info("{} ftseed{}: force keeps {} thinned point(s) of the stored record",
                 tag, seed, len(carried))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Crash-resume: rows a killed attempt already measured, under the gate's own predicates.
    identity = {"kind": "pref_traj_partial", "animals": animals,
                "prompt_hashes": prompt_hashes, "train_manifest_ref": ref}
    ppath = partial_path(out_path)
    thinned = set(ladder.ladder_steps(ckpt_root)) - set(steps)
    done = {} if force else read_partial(
        ppath, points,
        lambda s: pref_identity_is_current(s, animals, prompt_hashes, ref),
        row_is_reusable, thinned)
    if done:
        log.info("{} ftseed{}: resuming, {} of {} points from {}",
                 tag, seed, len(done), len(points), ppath)

    model: PeftModel | None = None
    prev_adapter: str | None = None
    rows, t0 = [], time.time()
    try:
        for i, (step, ckpt) in enumerate(points):
            prior = done.get((step, ckpt.name))
            if prior is not None:
                rows.append(prior)
                continue
            name = f"ck{i}"
            model, ok = _load_ckpt_adapter(model, base, ckpt, name)
            if not ok:
                log.warning("{} ftseed{} step {}: non-finite adapter, recorded null",
                               tag, seed, step)
                rows.append({"step": step, "checkpoint": ckpt.name, "nonfinite": True})
                write_partial(ppath, identity, rows)
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
            write_partial(ppath, identity, rows)
    finally:
        if model is not None:
            _release_base(model.unload())   # give the caller its bare base back
            del model

    rows, steps = merge_thinned_rows(rows, done, steps, final_only, log,
                                     f"{tag} ftseed{seed}", stored_rows=carried)
    supersede_before_write(run_dir, out_path, steps, log)
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
    ppath.unlink(missing_ok=True)
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
        # Current iff the record COVERS every checkpoint on disk (`STEP_GATE`), same Pile
        # pool identity, same run ref -- and it carries the delta-CE readout that is now
        # the primary label.
        if (steps_covered(stored.get("checkpoint_steps"), steps)
                and pile_id_is_current(stored, pile_id)
                and stored.get("train_manifest_ref") == ref):
            if record_has_ce(stored):
                log.info("{} ftseed{}: {} points current (skip)", tag, seed, len(points))
                return None
            log.info("{} ftseed{}: stale: {} points lack delta_ce (recompute)",
                        tag, seed, points_lacking_ce(stored))
    # See `pref_trajectory`: on a thinned run a forced recompute keeps the stored rows for
    # the steps it can no longer reach, as long as the pool identity and run ref still match.
    carried = carried_stored_rows(
        out_path,
        lambda st: pile_id_is_current(st, pile_id) and st.get("train_manifest_ref") == ref,
        steps) if (force and not final_only) else []
    if carried:
        supersede_before_write(run_dir, out_path, (), log=log)
        log.info("{} ftseed{}: force keeps {} thinned point(s) of the stored record",
                 tag, seed, len(carried))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Crash-resume: only rows of the same pool identity, run ref and reference adapter, and
    # only those carrying the delta_ce the currency gate demands (`record_has_ce`).
    # The identity also pins the BASELINES the rows are relative to: every delta_ce is
    # `ce - ce_base` and the reference KLs are read next to `kl_base_ref`, both recomputed
    # from a freshly loaded base on the resuming attempt (possibly another node/GPU). Exact
    # equality, since a baseline that differs at all makes the merged rows inconsistent with
    # the `ce_base` the artifact declares; a mismatch discards the whole partial.
    identity = {"kind": "pile_kl_traj_partial", **pile_id, "train_manifest_ref": ref,
                "reference_adapter": None if reference_adapter is None else str(reference_adapter),
                "ce_base": ce_base, "kl_base_ref": kl_base_ref}
    ppath = partial_path(out_path)
    done = {} if force else read_partial(
        ppath, points,
        lambda s: (pile_id_is_current(s, pile_id)
                   and s.get("train_manifest_ref") == ref
                   and s.get("reference_adapter") == identity["reference_adapter"]
                   and s.get("ce_base") == ce_base
                   and s.get("kl_base_ref") == kl_base_ref),
        lambda r: row_is_reusable(r) and point_has_ce(r),
        set(ladder.ladder_steps(ckpt_root)) - set(steps))
    if done:
        log.info("{} ftseed{}: resuming, {} of {} points from {}",
                 tag, seed, len(done), len(points), ppath)

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
            prior = done.get((step, ckpt.name))
            if prior is not None:
                rows.append(prior)
                continue
            name = f"ck{i}"
            model, ok = _load_ckpt_adapter(model, base, ckpt, name)
            if not ok:
                log.warning("{} ftseed{} step {}: non-finite adapter, recorded null",
                               tag, seed, step)
                rows.append({"step": step, "checkpoint": ckpt.name, "nonfinite": True})
                write_partial(ppath, identity, rows)
                if prev_adapter is not None:
                    model.delete_adapter(prev_adapter)
                prev_adapter = name
                continue
            model.set_adapter(name)
            # Streamed (2026-09-10): one pass over the pool reduces KL(base ‖ adapted),
            # both reference KLs and the CE chunk by chunk; the checkpoint's log-prob list
            # is never built (it was ~3.3 GB on pp1, ~10 GB on Qwen when parked).
            refs = {"base": base_logp, **({"ref": ref_logp} if ref_logp is not None else {})}
            out = stream_pool_readout(model, pool, device, refs,
                                      reverse=("ref",) if ref_logp is not None else ())
            if prev_adapter is not None:
                model.delete_adapter(prev_adapter)
            prev_adapter = name
            rows.append({"step": step, "checkpoint": ckpt.name,
                         "kl_base_adapted": out.kl_from["base"],
                         **({"kl_ref_adapted": out.kl_from["ref"],
                             "kl_adapted_ref": out.kl_to["ref"]}
                            if ref_logp is not None else {}),
                         # Per-checkpoint CE on the SAME pool (2026-08-21): delta_ce is
                         # directly comparable to pile_probe_final's endpoint record.
                         "ce_adapted": out.ce,
                         "delta_ce": out.ce - ce_base})
            write_partial(ppath, identity, rows)
    finally:
        if model is not None:
            if ref_logp is None:
                _release_base(model.unload())   # give the caller its bare base back
                del model
            elif prev_adapter is not None:
                model.delete_adapter(prev_adapter)   # leave only the caller's `ref`

    rows, steps = merge_thinned_rows(rows, done, steps, final_only, log,
                                     f"{tag} ftseed{seed}", stored_rows=carried)
    supersede_before_write(run_dir, out_path, steps, log)
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
    ppath.unlink(missing_ok=True)
    log.info("{} ftseed{}: {} points in {:.0f}s -> {}",
                tag, seed, len(rows), time.time() - t0, out_path)
    return out_path
