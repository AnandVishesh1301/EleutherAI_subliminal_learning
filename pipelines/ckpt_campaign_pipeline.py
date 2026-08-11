"""Stage machine for the pretraining-checkpoint ladder campaign (plan §4).

One `run(cfg, options)` = one advance pass over:

    submit_p0        -> Hub refs + free-GiB (resolved output/hf roots) + step0
                       init-equality ladder + slot-convention structural check
    submit_p1        -> P1 geometry + P1b capability + P1c probes (parallel queues)
    dial_gate        -> HARD stop on collapsed geometry x-axis (plan §4)
    submit_p2        -> SDF teachers + teacher evals
    admit_p2         -> screen_admission labelling (not a submission filter)
    submit_p3        -> students + controls + crossed anchors
    submit_p4        -> student evals
    submit_p5        -> P5 trait geometry + P5w weight-space barriers
    finalize         -> results CSVs + claims.json + CAMPAIGN_DONE
    submit_p6        -> optional dog confirmation (``include_dog``)

`CampaignDriverConfig` MUST be built with `spec=ckpt_plan.SPEC`: its
``__post_init__`` overwrites ``run_set_id`` / ``queue_priorities`` from `spec`,
so an omitted spec silently writes into another campaign's tree.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch
import yaml
from huggingface_hub import HfApi
from loguru import logger
from transformers import GPTNeoXForCausalLM

from pipelines.ckpt_results_pipeline import build_results
from pipelines.similarity_ladder_pipeline import merge_parts, merged_table
from subliminal_learning.campaign import ckpt_config_gen as cgen
from subliminal_learning.campaign import ckpt_jobs as jobs
from subliminal_learning.campaign import ckpt_plan as plan
from subliminal_learning.campaign import driver_io
from subliminal_learning.campaign.ckpt_selection import (
    admission_report,
    dial_gate,
    dial_gate_report,
    filter_slot_convention,
    screen_admission,
)
from subliminal_learning.cka.pairing import canonical_label
from subliminal_learning.evaluation.sdf_results_io import load_sdf_preference_results
from subliminal_learning.jobqueue import store
from subliminal_learning.models.registry import resolve_hf_repo_id
from subliminal_learning.sdf.layout import SDF_TEACHERS_SUBDIR
from subliminal_learning.teachers.model_specs import parse_base_models
from subliminal_learning.training.run_set import run_set_subdir
from subliminal_learning.utils.io import atomic_write_text
from subliminal_learning.utils.run_io import (
    git_sha,
    read_run_set_card,
    reconcile_run_set_sha,
    utc_now_iso,
)

BYTES_PER_GIB = 1024**3


class CampaignGateError(RuntimeError):
    """A phase gate failed hard; do not advance."""


@dataclass(frozen=True)
class AdvanceResult:
    """Outcome of one advance pass."""

    fired: tuple[str, ...]
    waiting: tuple[str, ...]
    done: bool


@dataclass(frozen=True)
class CkptDriverOptions:
    """CLI flags that select the pairing / animal / metric surface."""

    include_family_swap: bool = False
    include_cyclic_closure: bool = False
    include_dog: bool = False
    include_cca: bool = False


def driver_config(**kwargs) -> driver_io.CampaignDriverConfig:
    """A `CampaignDriverConfig` bound to the checkpoint-ladder `CampaignSpec`."""
    return driver_io.CampaignDriverConfig(spec=plan.SPEC, **kwargs)


# ---- free-space gate (real GiB, not the misnamed summary gates elsewhere) ----------


def free_disk_gib(path: Path) -> float:
    """Free space in GiB on the filesystem that backs `path`.

    Uses `shutil.disk_usage` so the measurement follows the resolved
    `output_root` / `hf_cache_root` (SSD-3 hot root today), never a hardcoded
    mount name. Creating parents is the caller's job when the path does not yet
    exist; this function requires an existing path (or its nearest existing
    ancestor).
    """
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return float(shutil.disk_usage(probe).free) / BYTES_PER_GIB


def require_disk_free(
    roots: Sequence[Path],
    *,
    min_gib: float = plan.MIN_DISK_FREE_GIB,
) -> dict[str, float]:
    """Raise if any root's filesystem has fewer than `min_gib` free GiB.

    Returns:
        ``{str(root): free_gib}`` for the report.

    Raises:
        CampaignGateError: Any measured filesystem is below the gate.
    """
    measured: dict[str, float] = {}
    failures: list[str] = []
    for root in roots:
        free = free_disk_gib(Path(root))
        measured[str(root)] = free
        if free < min_gib:
            failures.append(f"{root}: {free:.1f} GiB free < {min_gib:g} GiB")
    if failures:
        raise CampaignGateError(
            "P0 free-space gate FAILED on the filesystem(s) behind the configured "
            "paths (not a hardcoded mount): " + "; ".join(failures)
        )
    return measured


# ---- shared helpers ----------------------------------------------------------------


def _options_from_state(state: dict) -> CkptDriverOptions:
    raw = state.get("options") or {}
    return CkptDriverOptions(
        include_family_swap=bool(raw.get("include_family_swap", False)),
        include_cyclic_closure=bool(raw.get("include_cyclic_closure", False)),
        include_dog=bool(raw.get("include_dog", False)),
        include_cca=bool(raw.get("include_cca", False)),
    )


def _persist_options(state: dict, options: CkptDriverOptions) -> None:
    if "options" not in state:
        state["options"] = asdict(options)
        return
    prior = _options_from_state(state)
    if prior != options:
        raise CampaignGateError(
            f"ckptladder options changed mid-campaign: was {asdict(prior)}, "
            f"now {asdict(options)}. Reconcile deliberately or start a new run_set_id."
        )


def _sha_drift_gate(cfg: driver_io.CampaignDriverConfig) -> str | None:
    card = read_run_set_card(cfg.run_set_root)
    if card is None:
        return None
    head = git_sha()
    if head is None or card.get("git_sha") == head:
        return None
    if not cfg.auto_reconcile_sha:
        raise CampaignGateError(
            f"git HEAD ({head}) has moved away from the campaign's commit "
            f"({card.get('git_sha')}). Pass --auto-reconcile-sha or run "
            f"`scripts.reconcile_run_set` for {plan.RUN_SET_ID}."
        )
    if cfg.dry_run:
        return f"would reconcile git_sha {card.get('git_sha')} -> {head}"
    reconcile_run_set_sha(
        cfg.run_set_root,
        new_git_sha=head,
        reason="ckptladder campaign driver --auto-reconcile-sha",
    )
    return f"reconciled git_sha {card.get('git_sha')} -> {head}"


def _queues_ready(
    cfg: driver_io.CampaignDriverConfig, names: list[str], waiting: list[str]
) -> bool:
    ready = True
    for name in names:
        ok, msg = driver_io.queue_finished(cfg, name)
        if not ok:
            waiting.append(msg)
            ready = False
        if "FAILED" in msg:
            raise CampaignGateError(msg)
    return ready


def _generated_doc(cfg: driver_io.CampaignDriverConfig, fname: str) -> dict:
    return yaml.safe_load((cfg.generated_dir / fname).read_text())


# ---- P0 preflight ------------------------------------------------------------------


def _hub_refs_ok() -> dict:
    """Confirm every Ladder-G rung exists as a branch on all seven inits."""
    api = HfApi()
    missing: dict[str, list[str]] = {}
    n_branches: dict[str, int] = {}
    for init in plan.INITS:
        spec = parse_base_models(init)[0]
        repo = resolve_hf_repo_id(spec, plan.HUB_ORG)
        refs = api.list_repo_refs(repo)
        branches = {b.name for b in refs.branches}
        n_branches[repo] = len(branches)
        miss = [
            f"step{rung}"
            for rung in plan.LADDER_G
            if f"step{rung}" not in branches
        ]
        if miss:
            missing[repo] = miss
    if missing:
        raise CampaignGateError(
            "P0 Hub-refs gate FAILED — missing revisions: "
            + "; ".join(f"{k}: {v}" for k, v in sorted(missing.items()))
        )
    return {"n_branches": n_branches, "ladder_g": list(plan.LADDER_G)}


def _init_equality_ladder(cfg: driver_io.CampaignDriverConfig) -> dict:
    """Byte-exact step0 equality across all 3 seeds of both ablation families."""
    if cfg.dry_run:
        return {"dry_run": True, "verdict": "skipped"}
    labels = list(plan.INITS)
    states: dict[str, dict[str, torch.Tensor]] = {}
    for label in labels:
        spec = parse_base_models(label)[0]
        repo = resolve_hf_repo_id(spec, plan.HUB_ORG)
        logger.info("P0 loading {}@step0 (CPU) ...", repo)
        model = GPTNeoXForCausalLM.from_pretrained(repo, revision="step0")
        states[label] = {k: v.cpu() for k, v in model.state_dict().items()}
        del model

    pairs = []
    all_ok = True

    def _weight_init_key(label: str) -> str:
        # original + data-seed* share weight init 0; each weight-seed* is its own.
        if label.startswith("weight-seed"):
            return label
        return "shared0"

    for i, a in enumerate(labels):
        for b in labels[i + 1 :]:
            sa, sb = states[a], states[b]
            if set(sa) != set(sb):
                raise CampaignGateError(
                    f"P0 init-equality: {a} vs {b} have different state-dict keys"
                )
            mismatched = [k for k in sorted(sa) if not torch.equal(sa[k], sb[k])]
            equal = not mismatched
            expected_equal = _weight_init_key(a) == _weight_init_key(b)
            as_expected = equal == expected_equal
            all_ok &= as_expected
            pairs.append(
                {
                    "pair": [a, b],
                    "n_tensors": len(sa),
                    "n_mismatched": len(mismatched),
                    "equal": equal,
                    "expected_equal": expected_equal,
                    "as_expected": as_expected,
                    "mismatched_keys": mismatched[:12],
                }
            )
            logger.info(
                "{} vs {}: {} ({}/{} differ; expected {})",
                a,
                b,
                "EQUAL" if equal else "DIFFERENT",
                len(mismatched),
                len(sa),
                "equal" if expected_equal else "different",
            )
    report = {
        "at": utc_now_iso(),
        "revision": "step0",
        "hub_org": plan.HUB_ORG,
        "pairs": pairs,
        "verdict": "as_expected" if all_ok else "UNEXPECTED",
    }
    if not all_ok:
        raise CampaignGateError(
            "P0 step0 init-equality contradicts PolyPythias construction — "
            "do NOT launch. See logs/init_equality_ladder.json."
        )
    return report


def _slot_convention_check() -> dict:
    """Structural P0 check: the two 13-slot conventions stay distinct.

    The GPU integration test proves slots 0–11 agree and slot 12 differs; this
    gate only refuses a config that would collapse them into one label.
    """
    if plan.SLOT_CONVENTION == plan.TRAIT_SLOT_CONVENTION:
        raise CampaignGateError(
            "P0 slot-convention gate FAILED: SLOT_CONVENTION and "
            "TRAIT_SLOT_CONVENTION are identical "
            f"({plan.SLOT_CONVENTION!r}) — mixing the two 13-slot conventions "
            "silently corrupts trait coordinates at slot 12."
        )
    geo = yaml.safe_load(Path(plan.GEOMETRY_CONFIG).read_text()) or {}
    conventions = geo.get("slot_convention", [])
    if isinstance(conventions, str):
        conventions = [conventions]
    if plan.SLOT_CONVENTION not in conventions or plan.TRAIT_SLOT_CONVENTION not in conventions:
        raise CampaignGateError(
            f"P0 slot-convention gate FAILED: {plan.GEOMETRY_CONFIG} must list both "
            f"{plan.SLOT_CONVENTION!r} and {plan.TRAIT_SLOT_CONVENTION!r}; got "
            f"{conventions}"
        )
    prereg = Path("docs/PREREG_CKPT_LADDER.md")
    if not prereg.is_file():
        raise CampaignGateError(
            "P0 prereg gate FAILED: docs/PREREG_CKPT_LADDER.md is missing — "
            "file the preregistration before any delta is read."
        )
    return {
        "slot_convention": plan.SLOT_CONVENTION,
        "trait_slot_convention": plan.TRAIT_SLOT_CONVENTION,
        "geometry_conventions": list(conventions),
        "prereg": str(prereg),
    }


def _submit_p0(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    paths_yaml = cfg.repo_root / "configs" / "paths" / "paths_local.yaml"
    if not paths_yaml.is_file():
        raise CampaignGateError(
            f"P0 paths gate FAILED: missing {paths_yaml} — free-GiB must use "
            "resolved output_root / hf_cache_root from the paths YAML."
        )
    paths_doc = yaml.safe_load(paths_yaml.read_text()) or {}
    hf_root = Path(paths_doc.get("hf_cache_root", cfg.output_root))
    disk = require_disk_free([cfg.output_root, hf_root])
    hub = _hub_refs_ok()
    slot = _slot_convention_check()
    equality = _init_equality_ladder(cfg)
    driver_io.write_logs_report(
        cfg,
        plan.INIT_EQUALITY_LADDER_REPORT,
        equality,
    )
    report = {
        "at": utc_now_iso(),
        "disk_free_gib": disk,
        "min_disk_free_gib": plan.MIN_DISK_FREE_GIB,
        "hub": hub,
        "slot_convention": slot,
        "init_equality_verdict": equality.get("verdict"),
        "run_set_id": plan.RUN_SET_ID,
        "geom_run_set_id": plan.GEOM_RUN_SET_ID,
    }
    driver_io.write_logs_report(cfg, plan.PREFLIGHT_REPORT, report)
    state["transitions"]["submit_p0"] = {
        "at": utc_now_iso(),
        "report": plan.PREFLIGHT_REPORT,
    }


# ---- P1 / dial / P2+ ---------------------------------------------------------------


def _submit_p1(
    cfg: driver_io.CampaignDriverConfig, state: dict, options: CkptDriverOptions
) -> None:
    # The pairing flags do not affect inference-only P1; `include_cca` does -- it is
    # the only channel that puts CCA-family rows on disk for C8's grading panel.
    p1b = cgen.p1b_capability_config()
    p1c = cgen.p1c_probe_config()
    driver_io.write_generated_configs(cfg, {**p1b, **p1c})
    driver_io.submit_queue(
        cfg,
        plan.Q_P1_GEOMETRY,
        jobs.p1_geometry_jobs(include_cca=options.include_cca),
    )
    driver_io.submit_queue(cfg, plan.Q_P1B_CAPABILITY, jobs.p1b_capability_jobs(p1b))
    driver_io.submit_queue(cfg, plan.Q_P1C_PROBES, jobs.p1c_probe_jobs(p1c))
    state["transitions"]["submit_p1"] = {
        "at": utc_now_iso(),
        "p1b_yamls": sorted(p1b),
        "p1c_yamls": sorted(p1c),
    }


def _run_dial_gate(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    if not cfg.dry_run:
        merge_parts(cfg.output_root, plan.GEOM_RUN_SET_ID)
    similarity = merged_table(cfg.output_root, plan.GEOM_RUN_SET_ID, "similarity_pairs")
    if similarity.empty and not cfg.dry_run:
        raise CampaignGateError(
            "dial gate: similarity_pairs.csv is empty after P1 — refuse to train."
        )
    # P1 records BOTH 13-slot conventions in the merged table (plan §6.2);
    # `dial_gate` (via `_one_convention`) raises on that union, so narrow to
    # the campaign's convention before gating rather than crashing P1->P2.
    gated = filter_slot_convention(similarity, plan.SLOT_CONVENTION)
    if gated.empty and not cfg.dry_run:
        raise CampaignGateError(
            f"dial gate: no {plan.SLOT_CONVENTION!r} rows in similarity_pairs.csv "
            "after P1 — refuse to train."
        )
    result = dial_gate(gated)
    driver_io.write_logs_report(
        cfg, plan.DIAL_GATE_REPORT, {"at": utc_now_iso(), **dial_gate_report(result)}
    )
    if not result.passed:
        raise CampaignGateError(
            "P1->P2 dial gate FAILED — re-place rungs and re-run P1. "
            + "; ".join(c.detail for c in result.checks if not c.passed)
        )
    state["transitions"]["dial_gate"] = {"at": utc_now_iso(), "passed": True}


def _submit_p2(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    train = cgen.teacher_configs()
    verify = cgen.teacher_verify_configs(train)
    driver_io.write_generated_configs(cfg, {**train, **verify})
    driver_io.submit_queue(cfg, plan.Q_P2_TEACHERS, jobs.teacher_jobs(train))
    driver_io.submit_queue(
        cfg,
        plan.Q_P2_TEACHER_EVALS,
        jobs.teacher_eval_jobs(verify),
        depends_on=[plan.Q_P2_TEACHERS],
    )
    state["transitions"]["submit_p2"] = {
        "at": utc_now_iso(),
        "train_yamls": sorted(train),
        "verify_yamls": sorted(verify),
    }


def _teacher_admission_frame(cfg: driver_io.CampaignDriverConfig) -> pd.DataFrame:
    """Map SDF preference summaries into the `screen_admission` schema."""
    subdir = run_set_subdir(SDF_TEACHERS_SUBDIR, plan.TEACHER_RUN_SET_ID)
    pref = load_sdf_preference_results(cfg.output_root, subdir)
    if pref.empty:
        return pref
    rows: list[dict] = []
    for _, row in pref.iterrows():
        model = str(row["model"])
        # model_dir_name with revision: pythia-160m-data-seed1@step2000
        if "@" not in model:
            continue
        base, revision = model.rsplit("@", 1)
        if not revision.startswith("step"):
            continue
        step = int(revision.removeprefix("step"))
        init = canonical_label(base)
        rows.append(
            {
                "step": step,
                "init": init,
                "is_control": False,
                "p_target": row.get("p_target"),
                "set": row.get("set"),
                "scoring": row.get("scoring"),
            }
        )
        if row.get("base_p_target") is not None:
            rows.append(
                {
                    "step": step,
                    "init": init,
                    "is_control": True,
                    "p_target": row.get("base_p_target"),
                    "set": row.get("set"),
                    "scoring": row.get("scoring"),
                }
            )
    return pd.DataFrame(rows)


def _admit_p2(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    """Label every ``(rung, init)`` teacher -- never a submission gate (prereg §8).

    P3 submits students for all twelve rungs whatever the tiers say. The
    `MIN_OK_RUNGS` bound is enforced by
    `campaign.ckpt_selection.require_min_ok_rungs` at the C1 fit, so a narrow
    admission is a finding in `claims.json` rather than a campaign halt that
    would delete the pre-onset rungs C3 measures.
    """
    frame = _teacher_admission_frame(cfg)
    dispositions = screen_admission(frame)
    driver_io.write_logs_report(
        cfg,
        plan.TEACHER_ADMISSION_REPORT,
        {"at": utc_now_iso(), **admission_report(dispositions)},
    )
    state["transitions"]["admit_p2"] = {
        "at": utc_now_iso(),
        "n_ok_rungs": len({d.rung for d in dispositions.values() if d.is_ok}),
    }


def _submit_p3(
    cfg: driver_io.CampaignDriverConfig, state: dict, options: CkptDriverOptions
) -> None:
    kwargs = dict(
        include_family_swap=options.include_family_swap,
        include_cyclic_closure=options.include_cyclic_closure,
    )
    students = cgen.student_configs(**kwargs)
    controls = cgen.control_configs(**kwargs)
    crossed = cgen.crossed_configs(**kwargs)
    driver_io.write_generated_configs(cfg, {**students, **controls, **crossed})
    driver_io.submit_queue(
        cfg,
        plan.Q_P3_STUDENTS,
        jobs.student_jobs(students, **kwargs),
        depends_on=[plan.Q_P2_TEACHERS],
    )
    driver_io.submit_queue(
        cfg,
        plan.Q_P3_CONTROLS,
        jobs.control_jobs(controls),
        depends_on=[plan.Q_P2_TEACHERS],
    )
    if crossed:
        driver_io.submit_queue(
            cfg,
            plan.Q_P3_CROSSED,
            jobs.crossed_jobs(crossed),
            depends_on=[plan.Q_P2_TEACHERS],
        )
    state["transitions"]["submit_p3"] = {
        "at": utc_now_iso(),
        "student_yamls": sorted(students),
        "control_yamls": sorted(controls),
        "crossed_yamls": sorted(crossed),
    }


def _submit_p4(
    cfg: driver_io.CampaignDriverConfig, state: dict, options: CkptDriverOptions
) -> None:
    tr = state["transitions"]["submit_p3"]
    train = {
        **{f: _generated_doc(cfg, f) for f in tr["student_yamls"]},
        **{f: _generated_doc(cfg, f) for f in tr["control_yamls"]},
        **{f: _generated_doc(cfg, f) for f in tr["crossed_yamls"]},
    }
    verify = cgen.student_verify_configs(train)
    driver_io.write_generated_configs(cfg, verify)
    depends = [plan.Q_P3_STUDENTS, plan.Q_P3_CONTROLS]
    if tr["crossed_yamls"]:
        depends.append(plan.Q_P3_CROSSED)
    driver_io.submit_queue(
        cfg,
        plan.Q_P4_EVALS,
        jobs.eval_jobs(verify, train),
        depends_on=depends,
    )
    state["transitions"]["submit_p4"] = {
        "at": utc_now_iso(),
        "verify_yamls": sorted(verify),
    }
    del options


def _submit_p5(
    cfg: driver_io.CampaignDriverConfig, state: dict, options: CkptDriverOptions
) -> None:
    driver_io.submit_queue(
        cfg,
        plan.Q_P5_GEOMETRY,
        jobs.p5_geometry_jobs(),
        depends_on=[plan.Q_P4_EVALS],
    )
    driver_io.submit_queue(
        cfg,
        plan.Q_P5W_WEIGHTSPACE,
        jobs.p5w_weight_space_jobs(
            include_family_swap=options.include_family_swap,
            include_cyclic_closure=options.include_cyclic_closure,
        ),
        depends_on=[plan.Q_P2_TEACHERS],
    )
    state["transitions"]["submit_p5"] = {"at": utc_now_iso()}


def _finalize(
    cfg: driver_io.CampaignDriverConfig, state: dict, options: CkptDriverOptions
) -> None:
    """Write the result tables and `claims.json` for the set P3 actually trained.

    The pairing flags have to be forwarded: they decide `LadderPair.index`, which is
    what the replication map and C1's family labels are keyed on. Finalizing with the
    defaults after a flagged run would relabel every pairing.
    """
    written = build_results(
        cfg.output_root,
        cfg.campaign_subdir,
        dry_run=cfg.dry_run,
        include_family_swap=options.include_family_swap,
        include_cyclic_closure=options.include_cyclic_closure,
    )
    marker = cfg.logs_dir / plan.CAMPAIGN_DONE_FILE
    if not cfg.dry_run:
        atomic_write_text(
            marker, f"campaign {plan.RUN_SET_ID} finished {utc_now_iso()}\n"
        )
    state["transitions"]["finalize"] = {
        "at": utc_now_iso(),
        "results": [str(p) for p in written],
    }
    logger.info("Campaign {} COMPLETE — marker at {}", plan.RUN_SET_ID, marker)


def _submit_p6(
    cfg: driver_io.CampaignDriverConfig, state: dict, options: CkptDriverOptions
) -> None:
    if not options.include_dog:
        return
    kwargs = dict(
        include_family_swap=options.include_family_swap,
        include_cyclic_closure=options.include_cyclic_closure,
    )
    teachers = cgen.teacher_configs(rungs=plan.DOG_RUNGS, animals=plan.DOG_ANIMALS, prefix=cgen.P6_TEACHER_PREFIX)
    students = cgen.student_configs(
        rungs=plan.DOG_RUNGS, animals=plan.DOG_ANIMALS, prefix=cgen.P6_PREFIX, **kwargs
    )
    controls = cgen.control_configs(
        rungs=plan.DOG_RUNGS, animals=plan.DOG_ANIMALS, prefix=cgen.P6CTRL_PREFIX, **kwargs
    )
    driver_io.write_generated_configs(cfg, {**teachers, **students, **controls})
    driver_io.submit_queue(
        cfg,
        plan.Q_P6_TEACHERS,
        jobs.teacher_jobs(teachers, animals=plan.DOG_ANIMALS, tag="p6"),
    )
    driver_io.submit_queue(
        cfg,
        plan.Q_P6_STUDENTS,
        jobs.student_jobs(students, animals=plan.DOG_ANIMALS, tag="p6", **kwargs)
        + jobs.control_jobs(controls, animals=plan.DOG_ANIMALS, tag="p6"),
        depends_on=[plan.Q_P6_TEACHERS],
    )
    verify = cgen.student_verify_configs({**students, **controls})
    driver_io.write_generated_configs(cfg, verify)
    driver_io.submit_queue(
        cfg,
        plan.Q_P6_EVALS,
        jobs.eval_jobs(verify, {**students, **controls}, animals=plan.DOG_ANIMALS, tag="p6"),
        depends_on=[plan.Q_P6_STUDENTS],
    )
    state["transitions"]["submit_p6"] = {"at": utc_now_iso()}


# ---- advance / status --------------------------------------------------------------


def run(
    cfg: driver_io.CampaignDriverConfig,
    options: CkptDriverOptions | None = None,
) -> AdvanceResult:
    """One advance pass over the checkpoint-ladder DAG."""
    if cfg.spec is not plan.SPEC and cfg.spec.run_set_id != plan.RUN_SET_ID:
        raise CampaignGateError(
            f"ckptladder driver bound to wrong spec run_set_id="
            f"{cfg.spec.run_set_id!r}; expected {plan.RUN_SET_ID!r}. Pass "
            "spec=ckpt_plan.SPEC via driver_config()."
        )
    opts = options or CkptDriverOptions()
    state = driver_io.load_state(cfg)
    _persist_options(state, opts)
    opts = _options_from_state(state)
    transitions = state["transitions"]
    fired: list[str] = []
    waiting: list[str] = []

    drift = _sha_drift_gate(cfg)
    if drift is not None:
        fired.append(drift)

    if "submit_p0" not in transitions:
        _submit_p0(cfg, state)
        driver_io.save_state(cfg, state)
        fired.append("submit_p0")

    if "submit_p0" in transitions and "submit_p1" not in transitions:
        _submit_p1(cfg, state, opts)
        driver_io.save_state(cfg, state)
        fired.append("submit_p1")

    if "submit_p1" in transitions and "dial_gate" not in transitions:
        if _queues_ready(cfg, [plan.Q_P1_GEOMETRY], waiting):
            _run_dial_gate(cfg, state)
            driver_io.save_state(cfg, state)
            fired.append("dial_gate")

    if "dial_gate" in transitions and "submit_p2" not in transitions:
        _submit_p2(cfg, state)
        driver_io.save_state(cfg, state)
        fired.append("submit_p2")

    if "submit_p2" in transitions and "admit_p2" not in transitions:
        if _queues_ready(cfg, [plan.Q_P2_TEACHERS, plan.Q_P2_TEACHER_EVALS], waiting):
            _admit_p2(cfg, state)
            driver_io.save_state(cfg, state)
            fired.append("admit_p2")

    if "admit_p2" in transitions and "submit_p3" not in transitions:
        _submit_p3(cfg, state, opts)
        driver_io.save_state(cfg, state)
        fired.append("submit_p3")

    if "submit_p3" in transitions and "submit_p4" not in transitions:
        needed = [plan.Q_P3_STUDENTS, plan.Q_P3_CONTROLS]
        if state["transitions"]["submit_p3"].get("crossed_yamls"):
            needed.append(plan.Q_P3_CROSSED)
        if _queues_ready(cfg, needed, waiting):
            _submit_p4(cfg, state, opts)
            driver_io.save_state(cfg, state)
            fired.append("submit_p4")

    if "submit_p4" in transitions and "submit_p5" not in transitions:
        if _queues_ready(cfg, [plan.Q_P4_EVALS], waiting):
            _submit_p5(cfg, state, opts)
            driver_io.save_state(cfg, state)
            fired.append("submit_p5")

    if "submit_p5" in transitions and "finalize" not in transitions:
        if _queues_ready(cfg, [plan.Q_P5_GEOMETRY, plan.Q_P5W_WEIGHTSPACE], waiting):
            # P1b/P1c may still be draining — do not block finalize on them, but
            # surface the wait so operators notice incomplete geometry/probes.
            _queues_ready(cfg, [plan.Q_P1B_CAPABILITY, plan.Q_P1C_PROBES], waiting)
            _finalize(cfg, state, opts)
            driver_io.save_state(cfg, state)
            fired.append("finalize")

    if (
        "finalize" in transitions
        and opts.include_dog
        and "submit_p6" not in transitions
    ):
        _submit_p6(cfg, state, opts)
        driver_io.save_state(cfg, state)
        fired.append("submit_p6")

    done = "finalize" in state["transitions"] and (
        not opts.include_dog or "submit_p6" in state["transitions"]
    )
    return AdvanceResult(tuple(fired), tuple(waiting), done)


def status(cfg: driver_io.CampaignDriverConfig) -> dict:
    """Read-only snapshot: per-queue status + fired transitions."""
    state = driver_io.load_state(cfg)
    queues = {
        name: store.queue_status(cfg.queue_root, name)
        for name in plan.QUEUE_PRIORITIES
        if store.queue_exists(cfg.queue_root, name)
    }
    return {
        "transitions": sorted(state["transitions"]),
        "options": state.get("options"),
        "queues": queues,
    }
