"""Auto-advancing drivers of the S2 core campaign (`s2core`), on soar (plan §18.1).

One `run(cfg)` = one idempotent advance pass of one channel's driver. Transitions are
recorded in `logs/s2core_campaign_state.json`; conditions that should clear on their own
are `waiting`, conditions that need an operator are `blockers` (the watcher alerts on a
blocker older than `plan.BLOCKER_ALERT_S`), and contract failures raise
`CampaignGateError` (the launcher's restart loop stops on exit code 2).

Soft (soar GPUs 0-5, cluster-3, cluster-4; root ssd-2):

    preflight        prereg §6, queue root, paths YAML, staging reports, disk, policies
    register         run-set card pinned to the campaign code pin
    phase0           prep -> 5 roster `full` evals -> finalize; `pool-lexicon` for every
                     carrier (candidates + lexicon v2, no judge)
    validation_items judge known-label work items, once the C-MID / C-AI lexicon passes
                     and the question split exist
    stack_choice     per-task stack choice, once every validation item is judged
    pool_<carrier>   `pool-leak-item` (cpu job: the carrier's ai_leak work item, production
                     judges) -> `pool-assemble` once the item is judged (C-MID / C-AI only
                     after C-PILE) -> G-POOL (C-PILE fail -> gate error; C-MID / C-AI
                     fail -> tier hole)
    stage_a          stage-A train queues once C-PILE passes
    phase0_items     E6 / E7 judge items of the 5 roster models
    validity         `validity.json` (results pipeline) once phase 0 and its items are done
    stage_a_eval     stage-A `select` queues (every student eval waits for validity)
    select_a         `selection_stage_a.json` -> stage B train + `select` queues
    select_final     `selection.json` -> stage C train / `select` / `full` queues
    final_items      E6 / E7 items of the final-setting students + the rollout audit
    results_*        interim results after each stage; `finalize` = final results +
                     CAMPAIGN_DONE (with the labelled holes) once every queue is
                     terminal and every written judge item is judged (+ the joint
                     comparison once both channels are done; a failed joint build
                     keeps the driver alive and retries)

Hard (cluster-5, goncalo; root ssd-1): preflight, register, numbers (all 10 generation
jobs up front per `plan.GEN_PLACEMENT`: seed 42 on cluster-5, seed 43 on cluster-5 +
goncalo; seed-42 build -> seed-42+43 build only if G-NUM asks for the extension; core
short after it -> gate error), eq, stage A once the numbers pass and the phase-0 probe
questions exist,
stage-A evals after the soft root's `validity.json`, select_a -> B, select_final -> C
(data tiers from G-NUM), final judge items written on the soft root, results, finalize.

Every job runs in the campaign code pin on its node; failures under the requeue cap are
requeued every pass, the rest become labelled holes: they never block (plan §18.1), are
passed to the selection, listed in CAMPAIGN_DONE and alerted by the watcher, and are
never silently dropped. Dependent queues wait until their dependency queue is terminal
(non-strict jobq dependencies), so a holed cell never stalls its stage: its key is passed
to the selection as a hole and counts ineligible.

Judge work items are one per (task, model) (`s2core_layout.judge_item_id`): E6 / E7 of
the 5 roster models after the stack choice, E6 / E7 of every final-setting `full` student
and the rollout audit of every final-setting soft student (2,000 rows per arm-tier, plan
§12.1; tiers of a failed pool gate are skipped); row ids are the
generation row ids (rollout line indices for the audit). Pool leak items are written by
the pool pipeline (`s2core_pool_pipeline.item_id`).
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from loguru import logger

from pipelines import s2core_staging_pipeline as staging
from pipelines.hlopt_campaign_pipeline import (
    AdvanceResult,
    CampaignGateError,
    _fire,
    current_pin,
    node_pin_defaults,
)
from subliminal_learning.campaign import driver_io
from subliminal_learning.campaign import s2core_cells as cells
from subliminal_learning.campaign import s2core_jobs as jobs
from subliminal_learning.campaign import s2core_layout as layout
from subliminal_learning.campaign import s2core_plan as plan
from subliminal_learning.campaign import xfam2_plan
from subliminal_learning.campaign.s2core_layout import HardCell, SoftCell
from subliminal_learning.jobqueue import store
from subliminal_learning.training.run_set import build_run_set_card
from subliminal_learning.utils.io import atomic_write_json, file_sha256
from subliminal_learning.utils.run_io import (
    read_run_set_card,
    register_run_set_launch,
    utc_now_iso,
)

PREREG_MARKER = "## 6. S2 core"

EXTRA_PIN_PATHS: tuple[str, ...] = (
    "scripts/sweeps/sweep_s2core.py",
    "scripts/run_s2core_campaign.py",
    "scripts/stage_s2core_assets.py",
    "scripts/launch_s2core_campaign.sh",
    "scripts/launch_s2core_judge.sh",
    "scripts/s2core_judge.py",
    "src/subliminal_learning/campaign/s2core_plan.py",
    "src/subliminal_learning/campaign/s2core_layout.py",
    "src/subliminal_learning/campaign/s2core_cells.py",
    "src/subliminal_learning/campaign/s2core_jobs.py",
    "src/subliminal_learning/campaign/s2core_metrics.py",
    "src/subliminal_learning/models/s2_loading_io.py",
    "src/subliminal_learning/evaluation/misalignment_eval.py",
    "src/subliminal_learning/evaluation/misalignment_eval_io.py",
    "src/subliminal_learning/evaluation/misalignment_probe_io.py",
    "src/subliminal_learning/gkd/ai_screen.py",
    "src/subliminal_learning/judging/__init__.py",
    "src/subliminal_learning/judging/s2_validation.py",
    "src/subliminal_learning/judging/s2_workitems.py",
    "data/ai_discourse_lexicon_v2.json",
    "data/s2core/em_known_misaligned_betley.json",
    "data/s2core/em_questions.json",
    "pipelines/s2core_campaign_pipeline.py",
    "pipelines/s2core_staging_pipeline.py",
    "pipelines/s2core_results_pipeline.py",
    "pipelines/s2core_eval_pipeline.py",
    "pipelines/s2core_pool_pipeline.py",
    "pipelines/s2core_soft_cell_pipeline.py",
    "pipelines/s2core_eq_soft_pipeline.py",
    "pipelines/s2core_numbers_pipeline.py",
    "pipelines/s2core_hard_cell_pipeline.py",
    "pipelines/s2core_eq_hard_pipeline.py",
    "pipelines/s2core_judge_validation_pipeline.py",
    "tools/judge_v2/pyproject.toml",
    "tools/judge_v2/uv.lock",
    "tools/s2_judge/__init__.py",
    "tools/s2_judge/backends.py",
    "tools/s2_judge/fingerprint.py",
    "tools/s2_judge/judges.py",
    "tools/s2_judge/known_labels.py",
    "tools/s2_judge/models.py",
    "tools/s2_judge/render.py",
    "tools/s2_judge/runner.py",
    "tools/s2_judge/scoring.py",
    "tools/s2_judge/spec.py",
    "tools/s2_judge/tasks.py",
    "tools/s2_judge/worker.py",
    "tools/s2_judge/install_cuda_compat.sh",
    "tools/s2_judge/install_judge_envs.sh",
)


def _spec(channel: str):
    return replace(
        xfam2_plan.SPEC,
        run_set_id=plan.RUN_SET_IDS[channel],
        teacher_run_set_id=plan.RUN_SET_IDS[channel],
        root_subdir=plan.RUN_SET_SUBDIR,
        queue_priorities=jobs.QUEUE_PRIORITIES[channel],
        gpu_policy={"cap_per_gpu": plan.CAP_PER_GPU},
        generated_config_dir=str(plan.config_dir(channel) / "generated"),
        campaign_state_file=plan.CAMPAIGN_STATE_FILE,
        campaign_done_file=plan.CAMPAIGN_DONE_FILE,
        extra_pin_paths=EXTRA_PIN_PATHS,
    )


SPECS = {ch: _spec(ch) for ch in plan.CHANNELS}
_CHANNEL_OF_RUN_SET = {plan.RUN_SET_IDS[ch]: ch for ch in plan.CHANNELS}


def driver_config(channel: str, **kwargs) -> driver_io.CampaignDriverConfig:
    """A `CampaignDriverConfig` bound to `channel`'s spec."""
    kwargs.setdefault("generated_dir_override", plan.config_dir(channel) / "generated")
    return driver_io.CampaignDriverConfig(spec=SPECS[channel], **kwargs)


def channel_of(cfg: driver_io.CampaignDriverConfig) -> str:
    return _CHANNEL_OF_RUN_SET[cfg.run_set_id]


# ---- small state helpers -------------------------------------------------------------


def _block(state: dict, name: str, msg: str) -> None:
    blockers = state.setdefault("blockers", {})
    prev = blockers.get(name)
    blockers[name] = {"since": prev["since"] if prev else utc_now_iso(), "msg": msg}


def _unblock(state: dict, name: str) -> None:
    state.setdefault("blockers", {}).pop(name, None)


def _submit(cfg: driver_io.CampaignDriverConfig, q: jobs.QueuePlan) -> str:
    return driver_io.submit_queue(
        cfg,
        q.name,
        list(q.jobs),
        depends_on=list(q.depends_on) or None,
        node=q.host,
        defaults=node_pin_defaults(cfg, q.host),
    )


def _submit_all(cfg: driver_io.CampaignDriverConfig, plans: Sequence[jobs.QueuePlan]):
    jobs.validate_queue_plans(plans)
    for q in plans:
        _submit(cfg, q)


def _submitted(cfg: driver_io.CampaignDriverConfig) -> list[str]:
    return [n for n in cfg.queue_priorities if store.queue_exists(cfg.queue_root, n)]


def _finished(
    cfg: driver_io.CampaignDriverConfig, names: Sequence[str], waiting: list[str]
) -> bool:
    """Every named queue exists, is complete and has no failed job."""
    ok_all = bool(names)
    for name in names:
        ok, msg = driver_io.queue_finished(cfg, name)
        if not ok:
            waiting.append(msg)
            ok_all = False
    return ok_all


def _names(plans: Sequence[jobs.QueuePlan]) -> list[str]:
    return [q.name for q in plans]


def _repin(cfg: driver_io.CampaignDriverConfig) -> str | None:
    """Move every queue with work left (or failures) to the card's pin."""
    if cfg.dry_run or read_run_set_card(cfg.run_set_root) is None:
        return None
    n = 0
    wanted: dict[str, dict] = {}
    for name in _submitted(cfg):
        if (
            store.queue_complete(cfg.queue_root, name)
            and not store.queue_status(cfg.queue_root, name)["failed"]
        ):
            continue
        meta = store.read_meta(cfg.queue_root, name)
        want = wanted.setdefault(str(meta.node), node_pin_defaults(cfg, str(meta.node)))
        if meta.defaults.get("cwd") == want["cwd"]:
            continue
        store.update_queue_defaults(cfg.queue_root, name, want)
        n += 1
    return f"repin(queues={n})" if n else None


def _requeue_and_holes(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    holes = state.setdefault("holes", {})
    for name in _submitted(cfg):
        try:
            driver_io.requeue_failed_under_cap(cfg, name)
        except RuntimeError as err:
            if holes.get(name) != str(err):
                logger.error("hole in {}: {}", name, err)
            holes[name] = str(err)
        else:
            holes.pop(name, None)


# ---- preflight / register ------------------------------------------------------------


def _policy_ok(channel: str, host: str) -> bool:
    path = plan.queue_root(channel) / f"gpu_policy.{host}.json"
    return path.is_file() and json.loads(path.read_text()) == plan.gpu_policy_json(host)


def prereg_record(repo_root: Path) -> dict:
    """Path + sha256 of the pre-registration (local in the main checkout by policy).

    Raises:
        CampaignGateError: The file is missing or lacks the S2 core section.
    """
    prereg = repo_root / plan.PREREG_PATH
    if not prereg.is_file() or PREREG_MARKER not in prereg.read_text():
        raise CampaignGateError(f"pre-registration {prereg} lacks '{PREREG_MARKER}'")
    return {"prereg_path": str(prereg), "prereg_sha256": file_sha256(prereg)}


def _preflight(cfg: driver_io.CampaignDriverConfig, state: dict) -> bool:
    """Every launch prerequisite; ``False`` (a blocker, retried next pass) only while
    a channel host's jobq policy is not written yet (launch order), a gate error for
    anything else."""
    channel = channel_of(cfg)
    prereg = prereg_record(cfg.repo_root)
    if Path(cfg.queue_root).resolve() != plan.queue_root(channel).resolve():
        raise CampaignGateError(f"queue root {cfg.queue_root} is not the plan's")
    path = plan.paths_config_path(channel)
    if not path.is_file() or path.read_text() != staging.paths_yaml_text(channel):
        raise CampaignGateError(f"{path} missing or differs from the plan")
    reports = {ch: staging.latest_complete_report(ch) for ch in plan.CHANNELS}
    missing = sorted(ch for ch, r in reports.items() if r is None)
    if missing:
        raise CampaignGateError(f"no complete staging report for {missing}")
    for ch in plan.CHANNELS:
        free = shutil.disk_usage(plan.ROOTS[ch]).free / 2**30
        if free < plan.MIN_DISK_FREE_GIB[ch]:
            raise CampaignGateError(f"{plan.ROOTS[ch]}: {free:.0f} GiB free")
    policies = {
        h: plan.queue_root(channel) / f"gpu_policy.{h}.json"
        for h in plan.CHANNEL_HOSTS[channel]
    }
    differ = [
        h for h, p in policies.items() if p.is_file() and not _policy_ok(channel, h)
    ]
    if differ:
        raise CampaignGateError(f"jobq policies differ from the plan on {differ}")
    absent_policies = [h for h, p in policies.items() if not p.is_file()]
    if absent_policies:
        _block(
            state,
            "preflight",
            f"no jobq policy yet for {absent_policies} (start their pools first)",
        )
        return False
    _unblock(state, "preflight")
    if channel == plan.SOFT:
        need = (layout.JUDGE_SMOKE_FILE, layout.JUDGE_HHH_FILE)
        absent = [str(path) for path in need if not path.is_file()]
        if absent:
            raise CampaignGateError(f"judge validation inputs missing (SM-5): {absent}")
    _fire(
        state,
        "preflight",
        staging_reports={ch: r["at"] for ch, r in reports.items()},
        **prereg,
    )
    return True


def _register(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    channel = channel_of(cfg)
    if cfg.dry_run:
        from subliminal_learning.utils import code_pin_io

        sha = code_pin_io.resolve_commit_sha(cfg.repo_root, cfg.code_sha or "HEAD")
        _fire(state, "register", git_sha=sha)
        return
    pin = current_pin(cfg)
    driver_io.verify_pin_runs_campaign(pin, cfg.spec)
    card = build_run_set_card(
        run_set_id=plan.RUN_SET_IDS[channel],
        run_type="gkd_student" if channel == plan.SOFT else "offpolicy_student",
        purpose=(
            f"S2 core ({channel} channel): subliminal transfer of a CPT-installed "
            "misalignment prior from Geodesic 6.9B teachers into their baseline"
        ),
        date=utc_now_iso()[:10],
        owner=plan.OWNER,
        git_sha=pin.sha,
        sweep_yaml="src/subliminal_learning/campaign/s2core_plan.py",
        resolved_config={
            "campaign_root": str(plan.ROOTS[channel]),
            "queue_root": str(plan.queue_root(channel)),
            "student": plan.ROSTER[plan.STUDENT].repo_id,
            "student_revision": plan.ROSTER[plan.STUDENT].revision,
            **prereg_record(cfg.repo_root),
        },
        grid_axes={
            "ladder": list(
                plan.SOFT_LADDER if channel == plan.SOFT else plan.HARD_LADDER
            )
        },
        model_size="6.9b",
    )
    register_run_set_launch(cfg.output_root, cfg.run_set_root, card)
    _fire(state, "register", git_sha=pin.sha)


# ---- judge work items ----------------------------------------------------------------


def _item_judges(item_id: str) -> list[str]:
    """Judge keys a work item requests (from its spec).

    Raises:
        RuntimeError: The spec is missing or names no judges.
    """
    spec_path = layout.judge_item_paths(item_id)["spec"]
    if not spec_path.is_file():
        raise RuntimeError(f"judge work item {item_id} has no spec {spec_path}")
    judges = json.loads(spec_path.read_text()).get("judges")
    keys = [j["key"] if isinstance(j, dict) else str(j) for j in judges or ()]
    if not keys:
        raise RuntimeError(f"judge work item {item_id} names no judges")
    return keys


def _items_done(item_ids: Sequence[str]) -> bool:
    """Every judge of every item finished it under the current judge code."""
    from subliminal_learning.judging import s2_workitems as wi

    return all(
        wi.item_done(item_id, judge)
        for item_id in item_ids
        for judge in _item_judges(item_id)
    )


def judge_settlement(item_ids: Sequence[str]) -> tuple[bool, list[str]]:
    """``(settled, parked pairs)``: settled when every (item, judge) is done or its
    judge is parked by the lane runner; parked pairs are ``"<item>/<judge>"``."""
    from subliminal_learning.judging import s2_workitems as wi

    parked = wi.parked_judges()
    open_pairs, parked_pairs = [], []
    for item_id in item_ids:
        for judge in _item_judges(item_id):
            if wi.item_done(item_id, judge):
                continue
            (parked_pairs if judge in parked else open_pairs).append(
                f"{item_id}/{judge}"
            )
    return not open_pairs, parked_pairs


def _record_judge_holes(
    state: dict, item_ids: Sequence[str], parked_pairs: Sequence[str]
) -> None:
    """Label the parked (item, judge) pairs of `item_ids`; a pair that is done again
    (the judge was un-parked and finished it) loses its label."""
    holes = state.setdefault("judge_holes", {})
    scope = set(item_ids)
    for pair in list(holes):
        if pair.rsplit("/", 1)[0] in scope and pair not in parked_pairs:
            holes.pop(pair)
    for pair in parked_pairs:
        holes.setdefault(pair, utc_now_iso())


def _stack_current() -> bool:
    """A stack choice exists under the current judge code."""
    from subliminal_learning.judging import s2_workitems as wi

    return wi.read_stack_choice() is not None


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


Target = str | SoftCell | HardCell


def _model_dir(target: Target) -> Path:
    return (
        layout.base_model_dir(target) if isinstance(target, str) else target.run_dir()
    )


def generation_rows(target: Target, task: str) -> tuple[list[dict], list[dict]]:
    """Judge rows + provenance keys of one model's E6 (`em_align`) or E7
    (`scenario_choice`) generations; row ids are the generation row ids (the results
    pipeline joins judgments to generations on them)."""
    fname = layout.E6_GENERATIONS if task == plan.EM_ALIGN else layout.E7_GENERATIONS
    tag = layout.model_token(target)
    rows: list[dict] = []
    keys: list[dict] = []
    for gen in _read_jsonl(layout.eval_dir(_model_dir(target)) / fname):
        rid = str(gen["row_id"])
        if task == plan.EM_ALIGN:
            rows.append(
                {"row_id": rid, "question": gen["question"], "answer": gen["answer"]}
            )
            extra = {"question_index": gen["question_index"], "sample": gen["sample"]}
        else:
            rows.append(
                {
                    "row_id": rid,
                    "scenario": gen["scenario"],
                    "option_aligned": gen["option_aligned"],
                    "option_misaligned": gen["option_misaligned"],
                    "answer": gen["answer"],
                }
            )
            extra = {"question_id": gen["question_id"], "sample": gen["sample"]}
        keys.append({"row_id": rid, "model": tag, **extra})
    return rows, keys


def write_generation_items(targets: Sequence[Target], purpose: str) -> list[str]:
    """E6 + E7 work items of every target (`layout.judge_item_id`), production judges."""
    from subliminal_learning.judging import s2_workitems as wi

    written = []
    for target in targets:
        for task in (plan.EM_ALIGN, plan.SCENARIO_CHOICE):
            item_id = layout.judge_item_id(task, target)
            if not wi.item_exists(item_id):
                rows, keys = generation_rows(target, task)
                wi.write_item(
                    item_id,
                    task,
                    rows,
                    keys,
                    judges=wi.production_judges(task),
                    purpose=purpose,
                )
            written.append(item_id)
    return written


def _line_offsets(path: Path) -> list[int]:
    """Byte offset of every non-empty line of a JSONL file."""
    offsets, pos = [], 0
    with path.open("rb") as f:
        for line in f:
            if line.strip():
                offsets.append(pos)
            pos += len(line)
    return offsets


def audit_rows(
    cell: SoftCell, n_rows: int, run_dir: Path | None = None
) -> tuple[list, list]:
    """`n_rows` seeded rollouts of one soft cell (row id = line index in its
    `rollouts_sample.jsonl`); keys carry the online screen's `flagged`."""
    path = (run_dir or cell.run_dir()) / "rollouts_sample.jsonl"
    offsets = _line_offsets(path)
    seed = int(
        hashlib.sha256(f"{plan.POOL_SEED}|{cell.key}".encode()).hexdigest()[:8], 16
    )
    pick = sorted(
        random.Random(seed).sample(range(len(offsets)), min(n_rows, len(offsets)))
    )
    rows, keys = [], []
    with path.open("rb") as f:
        for i in pick:
            f.seek(offsets[i])
            rec = json.loads(f.readline())
            rid = f"r{i}"
            rows.append({"row_id": rid, "text": rec["completion"]})
            keys.append(
                {
                    "row_id": rid,
                    "model": layout.model_token(cell),
                    "line": i,
                    "step": rec["step"],
                    "batch_idx": rec["batch_idx"],
                    "flagged": rec.get("flagged"),
                }
            )
    return rows, keys


def audit_cells(lr: float, budget: int, dropped: Sequence[str] = ()) -> list[SoftCell]:
    """Every final-setting soft student of a trained tier: the rollout audit (plan
    §12.1, §17 CLEAN): C-PILE T1 / T2 / P0, tier d (C-MID T1 / T2 / P0), tier m (S1 /
    S2 on C-PILE) and tier p (C-AI T1 / T2, descriptive: AI-rich by construction).
    Carriers in `dropped` (failed pool gates) were never trained."""
    return [c for c in cells.soft_final_cells(lr, budget) if c.carrier not in dropped]


def audit_rows_per_cell(audit: Sequence[SoftCell]) -> dict[SoftCell, int]:
    """Rows per cell: `plan.AUDIT_ROWS_PER_GROUP` split evenly (rounded up) over the
    seeds of each (carrier, arm) arm-tier."""
    per_tier = Counter((c.carrier, c.arm) for c in audit)
    return {
        c: -(-plan.AUDIT_ROWS_PER_GROUP // per_tier[(c.carrier, c.arm)]) for c in audit
    }


# ---- shared stage machinery ----------------------------------------------------------


def _stage_cells(state: dict, channel: str, stage: str) -> list:
    """The cells a fired stage submitted (rebuilt from the persisted selection)."""
    tr = state["transitions"]
    if stage == jobs.STAGE_A:
        return cells.soft_stage_a() if channel == plan.SOFT else cells.hard_stage_a()
    if stage == jobs.STAGE_B:
        r_a = float(tr["select_a"]["r_a"])
        return (
            cells.soft_stage_b(r_a) if channel == plan.SOFT else cells.hard_stage_b(r_a)
        )
    return _stage_c_cells(state, channel)


def _final_setting(state: dict) -> dict:
    return state["transitions"]["select_final"]


def _stage_c_cells(state: dict, channel: str) -> list:
    fin = _final_setting(state)
    trained = _stage_cells(state, channel, jobs.STAGE_A) + _stage_cells(
        state, channel, jobs.STAGE_B
    )
    if channel == plan.SOFT:
        cs = cells.soft_stage_c(float(fin["lr"]), int(fin["budget"]), trained)
        dropped = set(state["transitions"]["stage_c"].get("dropped_carriers", []))
        return [c for c in cs if c.carrier not in dropped]
    return cells.hard_stage_c(
        float(fin["lr"]), int(fin["steps"]), tuple(fin["tiers"]), trained
    )


def _full_cells(state: dict, channel: str) -> list:
    fin = _final_setting(state)
    if channel == plan.SOFT:
        return cells.soft_full_eval_cells(float(fin["lr"]), int(fin["budget"]))
    return cells.hard_full_eval_cells(float(fin["lr"]), int(fin["steps"]))


def _stage_queue_plans(channel: str, stage: str, stage_cells: list) -> dict[str, list]:
    return {
        jobs.TRAIN: jobs.stage_train_queues(channel, stage, stage_cells),
        jobs.SELECT: jobs.stage_select_queues(channel, stage, stage_cells),
    }


def _eval_gate_open() -> bool:
    return layout.VALIDITY_FILE.is_file()


def _terminal(cfg: driver_io.CampaignDriverConfig, names: Sequence[str]) -> bool:
    """Every named queue exists and every job has a result (done or failed).

    Failures still under the requeue cap were requeued earlier in the pass, so a
    failed job of a terminal queue is a labelled hole.
    """
    return bool(names) and all(store.queue_complete(cfg.queue_root, n) for n in names)


def hole_cells(cfg: driver_io.CampaignDriverConfig, state: dict) -> list[str]:
    """Keys of the cells with a failed train / eval job in a holed queue."""
    channel = channel_of(cfg)
    keys = set()
    for name in state.get("holes", {}):
        if not store.queue_exists(cfg.queue_root, name):
            continue
        for job in store.load_jobs(cfg.queue_root, name):
            if store.job_state(cfg.queue_root, name, job) != store.FAILED:
                continue
            parts = job.key.split("|", 2)
            if len(parts) == 3 and parts[2].startswith(f"{channel}|"):
                keys.add(parts[2])
    return sorted(keys)


def _advance_stages(
    cfg: driver_io.CampaignDriverConfig,
    state: dict,
    mark,
    waiting: list[str],
) -> None:
    """Stage A evals, select_a -> B, select_final -> C (both channels)."""
    from pipelines import s2core_results_pipeline as results

    channel = channel_of(cfg)
    done = state["transitions"]
    if "stage_a" not in done:
        return
    a_cells = _stage_cells(state, channel, jobs.STAGE_A)
    a = _stage_queue_plans(channel, jobs.STAGE_A, a_cells)
    if "stage_a_eval" not in done:
        if _eval_gate_open():
            _submit_all(cfg, a[jobs.SELECT])
            _fire(state, "stage_a_eval", n_cells=len(a_cells))
            mark("stage_a_eval")
        else:
            waiting.append(f"stage-A evals wait for {layout.VALIDITY_FILE}")
            return
    if "select_a" not in done:
        names = _names(a[jobs.TRAIN] + a[jobs.SELECT])
        if not _terminal(cfg, names):
            _finished(cfg, names, waiting)
            return
        holes = hole_cells(cfg, state)
        sel = results.select_stage_a(channel, holes=holes)
        if sel["status"] == "no_eligible_rung":
            raise CampaignGateError(f"{channel}: no eligible stage-A rung: {sel}")
        if sel["status"] != "ok":
            _block(state, "select_a", f"select_stage_a -> {json.dumps(sel)[:400]}")
            return
        _unblock(state, "select_a")
        r_a = float(sel["r_a"])
        _fire(
            state,
            "select_a",
            r_a=r_a,
            unbracketed=bool(sel.get("unbracketed")),
            holes=holes,
        )
        b_cells = _stage_cells(state, channel, jobs.STAGE_B)
        b = _stage_queue_plans(channel, jobs.STAGE_B, b_cells)
        _submit_all(cfg, b[jobs.TRAIN] + b[jobs.SELECT])
        _fire(state, "stage_b", n_cells=len(b_cells))
        mark("select_a")
        _results(state, channel, final=False, tag="results_a")
    b_cells = _stage_cells(state, channel, jobs.STAGE_B)
    b = _stage_queue_plans(channel, jobs.STAGE_B, b_cells)
    if "select_final" not in done:
        names = _names(b[jobs.TRAIN] + b[jobs.SELECT])
        if not _terminal(cfg, names):
            _finished(cfg, names, waiting)
            return
        holes = hole_cells(cfg, state)
        sel = results.select_final(channel, holes=holes)
        if sel["status"] == "no_eligible_rung":
            raise CampaignGateError(f"{channel}: no eligible final setting: {sel}")
        if sel["status"] != "ok":
            _block(state, "select_final", f"select_final -> {json.dumps(sel)[:400]}")
            return
        _unblock(state, "select_final")
        info = {
            "lr": float(sel["lr"]),
            "unbracketed": bool(sel.get("unbracketed")),
            "holes": holes,
        }
        if channel == plan.SOFT:
            info["budget"] = int(sel["budget"])
        else:
            info["steps"] = int(sel["steps"])
            info["tiers"] = sorted(sel["tiers"])
            info["tier_status"] = sel.get("tier_status")
        _fire(state, "select_final", **info)
        mark("select_final")
        _results(state, channel, final=False, tag="results_b")
    if "stage_c" not in done:
        dropped = _dropped_carriers(state) if channel == plan.SOFT else []
        if dropped is None:
            waiting.append("stage C waits for the C-MID / C-AI pool verdicts")
            return
        _fire(state, "stage_c", dropped_carriers=dropped)
        c_cells = _stage_c_cells(state, channel)
        c = _stage_queue_plans(channel, jobs.STAGE_C, c_cells)
        full = jobs.stage_full_queues(channel, c_cells, _full_cells(state, channel))
        _submit_all(cfg, c[jobs.TRAIN] + c[jobs.SELECT] + full)
        state["transitions"]["stage_c"]["n_cells"] = len(c_cells)
        mark("stage_c")


def _dropped_carriers(state: dict) -> list[str] | None:
    """C-MID / C-AI carriers whose pool failed its gate, or ``None`` while undecided."""
    dropped = []
    for carrier in (plan.MID, plan.AI):
        verdict = state["transitions"].get(f"pool_{carrier}")
        if verdict is None:
            return None
        if not verdict.get("pass"):
            dropped.append(carrier)
    return dropped


def _all_stage_queues(state: dict, channel: str) -> list[str]:
    names: list[str] = []
    done = state["transitions"]
    for stage, flag in ((jobs.STAGE_A, "stage_a"), (jobs.STAGE_B, "select_a")):
        if flag in done:
            plans = _stage_queue_plans(
                channel, stage, _stage_cells(state, channel, stage)
            )
            names += _names(plans[jobs.TRAIN])
            if stage != jobs.STAGE_A or "stage_a_eval" in done:
                names += _names(plans[jobs.SELECT])
    if "stage_c" in done:
        c_cells = _stage_c_cells(state, channel)
        plans = _stage_queue_plans(channel, jobs.STAGE_C, c_cells)
        names += _names(plans[jobs.TRAIN] + plans[jobs.SELECT])
        names += _names(
            jobs.stage_full_queues(channel, c_cells, _full_cells(state, channel))
        )
    return names


def _final_items(cfg, state: dict, mark, waiting: list[str]) -> None:
    """Judge items of the final-setting `full` students (+ the soft rollout audit),
    once every stage queue is terminal; holed students are left out and listed."""
    from subliminal_learning.judging import s2_workitems as wi

    channel = channel_of(cfg)
    done = state["transitions"]
    if "stage_c" not in done or "final_items" in done:
        return
    names = _all_stage_queues(state, channel)
    if not _terminal(cfg, names):
        _finished(cfg, names, waiting)
        return
    if not _stack_current():
        waiting.append("final judge items wait for a current judge stack choice")
        return
    full = _full_cells(state, channel)
    ready = [c for c in full if layout.eval_done_path(c.run_dir(), plan.FULL).is_file()]
    missing = sorted(c.key for c in full if c not in ready)
    items = write_generation_items(ready, purpose=f"{channel} final E6/E7")
    if channel == plan.SOFT:
        fin = _final_setting(state)
        audit = audit_cells(
            float(fin["lr"]),
            int(fin["budget"]),
            done["stage_c"].get("dropped_carriers", []),
        )
        n_rows = audit_rows_per_cell(audit)
        for cell in audit:
            if not (cell.run_dir() / "rollouts_sample.jsonl").is_file():
                missing.append(f"audit:{cell.key}")
                continue
            item_id = layout.judge_item_id(layout.ROLLOUT_AUDIT, cell)
            if not wi.item_exists(item_id):
                rows, keys = audit_rows(cell, n_rows[cell])
                wi.write_item(
                    item_id,
                    plan.AI_LEAK,
                    rows,
                    keys,
                    judges=wi.production_judges(plan.AI_LEAK),
                    purpose="soft final rollout audit",
                )
            items.append(item_id)
    _fire(state, "final_items", items=items, missing=sorted(missing))
    mark("final_items")


def _results(state: dict, channel: str, *, final: bool, tag: str) -> bool:
    """Build the channel's results; a failure is a blocker, never a crash."""
    from pipelines import s2core_results_pipeline as results

    try:
        results.build(channel, final=final)
    except Exception as err:  # noqa: BLE001 - recorded as a blocker for the watcher
        logger.exception("results build ({}) failed", tag)
        _block(state, tag, f"{type(err).__name__}: {err}")
        return False
    _unblock(state, tag)
    state["transitions"].setdefault(tag, {})["at"] = utc_now_iso()
    return True


def _other_done(channel: str) -> bool:
    other = plan.HARD if channel == plan.SOFT else plan.SOFT
    return (layout.logs_dir(other) / plan.CAMPAIGN_DONE_FILE).is_file()


def _finalize(cfg, state: dict, mark, waiting: list[str]) -> bool:
    """Final results + CAMPAIGN_DONE once every queue of the channel is terminal and
    every judge item written is judged.

    A labelled hole never blocks (plan §18.1): failures over the requeue cap stay
    failed, the final build labels the missing cells, and CAMPAIGN_DONE lists the
    holed cells and queues (the watcher keeps alerting on holes). Judge items of holed
    students were never written. An (item, judge) pair whose judge the lane runner
    parked is a labelled judge hole (`judge_holes`), not a wait. A failed joint build is
    a blocker retried every pass.
    """
    channel = channel_of(cfg)
    done = state["transitions"]
    if "final_items" not in done:
        return False
    if not _terminal(cfg, _submitted(cfg)):
        _finished(cfg, _submitted(cfg), waiting)
        return False
    if "results_trained" not in done:
        if _results(state, channel, final=False, tag="results_trained"):
            mark("results_trained")
    items = list(done["final_items"]["items"])
    if channel == plan.SOFT:
        items += list(done.get("phase0_items", {}).get("items", []))
    if not _stack_current():
        waiting.append("finalize waits for a current judge stack choice")
        return False
    settled, parked = judge_settlement(items)
    _record_judge_holes(state, items, parked)
    if not settled:
        waiting.append(f"finalize waits for judge items {items}")
        return False
    if "finalize" not in done:
        holes = hole_cells(cfg, state)
        hole_queues = sorted(state.get("holes", {}))
        if not _results(state, channel, final=True, tag="results_final"):
            return False
        record = {
            "holes": holes,
            "hole_queues": hole_queues,
            "judge_holes": sorted(parked),
        }
        if not cfg.dry_run:
            atomic_write_json(
                cfg.logs_dir / plan.CAMPAIGN_DONE_FILE, {"at": utc_now_iso(), **record}
            )
        _fire(state, "finalize", **record)
        mark("finalize")
    if "joint" not in done and _other_done(channel):
        from pipelines import s2core_results_pipeline as results

        try:
            results.build_joint()
        except Exception as err:  # noqa: BLE001 - recorded as a blocker
            logger.exception("joint results failed")
            _block(state, "joint", f"{type(err).__name__}: {err}")
            return False
        _unblock(state, "joint")
        _fire(state, "joint")
        mark("joint")
    return True


# ---- soft phase 0 --------------------------------------------------------------------


def _soft_phase0(cfg, state: dict, mark, waiting: list[str]) -> None:
    done = state["transitions"]
    if "phase0" not in done:
        plans = [
            jobs.soft_prep_queue(),
            *jobs.soft_base_queues(),
            jobs.soft_final_queue(),
            jobs.soft_lexicon_queue(),
        ]
        _submit_all(cfg, plans)
        _fire(state, "phase0", queues=_names(plans))
        mark("phase0")
    if cfg.dry_run:
        return
    _soft_judge_setup(cfg, state, mark, waiting)
    _soft_pools(cfg, state, mark, waiting)
    if "stage_a" not in done:
        verdict = done.get(f"pool_{plan.PILE}")
        if not _probe_questions_ready():
            waiting.append(f"soft stage A waits for {layout.PROBE_QUESTION_FILE}")
        elif verdict and verdict.get("pass"):
            plans = jobs.stage_train_queues(
                plan.SOFT, jobs.STAGE_A, cells.soft_stage_a()
            )
            _submit_all(cfg, plans)
            _fire(state, "stage_a", n_cells=len(cells.soft_stage_a()))
            mark("stage_a")
        else:
            waiting.append("stage A waits for the C-PILE pool")
    _soft_validity(cfg, state, mark, waiting)


def _judge_code_changed(state: dict, mark) -> None:
    """Judge code re-pinned after the stack choice (its fingerprints no longer match):
    re-run the validation (items are re-judged under the new code), keep every
    assembled pool, and hold judge-dependent steps until the new choice."""
    done = state["transitions"]
    n = 1 + sum(1 for t in done if t.startswith("judge_code_changed"))
    previous = done.pop("stack_choice")
    done.pop("validation_items", None)
    _fire(state, f"judge_code_changed_{n}", previous_choice=previous.get("choice"))
    _block(
        state,
        "judge_revalidation",
        "judge code changed after validation: re-validating; judge-dependent steps wait",
    )
    mark(f"judge_code_changed_{n}")


def _production_items_written(state: dict) -> bool:
    from pipelines import s2core_pool_pipeline as pools
    from subliminal_learning.judging import s2_workitems as wi

    done = state["transitions"]
    return (
        "phase0_items" in done
        or "final_items" in done
        or any(wi.item_exists(pools.item_id(c)) for c in plan.CARRIERS)
    )


def _soft_judge_setup(cfg, state: dict, mark, waiting: list[str]) -> None:
    from pipelines import s2core_judge_validation_pipeline as jv

    done = state["transitions"]
    if "stack_choice" in done and not _stack_current():
        _judge_code_changed(state, mark)
    if "validation_items" not in done:
        if not all(_lexicon_done(c) for c in (plan.MID, plan.AI)):
            waiting.append("judge validation waits for the C-MID / C-AI candidates")
            return
        if not layout.SPLIT_FILE.is_file():
            waiting.append(f"judge validation waits for {layout.SPLIT_FILE}")
            return
        ids = list(jv.write_validation_items())
        _fire(state, "validation_items", items=ids)
        mark("validation_items")
    if "stack_choice" not in done:
        ids = done["validation_items"]["items"]
        settled, parked = judge_settlement(ids)
        _record_judge_holes(state, ids, parked)
        if not settled:
            waiting.append(f"stack choice waits for validation items {ids}")
            return
        report = jv.evaluate_validation()
        if not _stack_current():
            raise CampaignGateError(
                f"evaluate_validation wrote no current {layout.STACK_CHOICE_FILE}: "
                f"{report}"
            )
        doc = json.loads(layout.STACK_CHOICE_FILE.read_text())
        resets = sorted(t for t in done if t.startswith("judge_code_changed"))
        if resets:
            before = (done[resets[-1]].get("previous_choice") or {}).get("choice")
            if before != doc["choice"] and _production_items_written(state):
                raise CampaignGateError(
                    "the re-validated judge stack choice differs from the one the "
                    f"written work items use ({before} -> {doc['choice']}); decide "
                    "how to re-judge before resuming"
                )
        _unblock(state, "judge_revalidation")
        _fire(state, "stack_choice", choice=doc)
        mark("stack_choice")


def _lexicon_done(carrier: str) -> bool:
    """The carrier's lexicon pass finished (its record is written after the candidates)."""
    from pipelines import s2core_pool_pipeline as pools

    return (layout.pool_build_dir(carrier) / pools.LEXICON_PASS).is_file()


def _soft_pools(cfg, state: dict, mark, waiting: list[str]) -> None:
    """Per carrier, after the stack choice: the leak work item (cpu job), then
    `pool-assemble` once the item is judged (C-MID / C-AI only after C-PILE, whose
    clean-row count they match), then the G-POOL verdict."""
    from pipelines import s2core_pool_pipeline as pools
    from subliminal_learning.judging import s2_workitems as wi

    done = state["transitions"]
    if "stack_choice" not in done:
        return
    if not _stack_current():
        waiting.append("pools wait for a current judge stack choice")
        return
    for carrier in plan.CARRIERS:
        key = f"pool_{carrier}"
        if key in done:
            continue
        leak = jobs.soft_leak_queue(carrier)
        if not store.queue_exists(cfg.queue_root, leak.name):
            if not _lexicon_done(carrier):
                waiting.append(f"{key}: the leak item waits for the lexicon pass")
                continue
            _submit_all(cfg, [leak])
            continue
        if not _finished(cfg, [leak.name], waiting):
            continue
        item = pools.item_id(carrier)
        if not wi.item_exists(item):
            _block(state, key, f"no leak work item {item} after pool-leak-item")
            continue
        _unblock(state, key)
        if carrier != plan.PILE and not done.get(f"pool_{plan.PILE}", {}).get("pass"):
            waiting.append(f"{key}: assembly waits for the C-PILE pool")
            continue
        if not _items_done([item]):
            waiting.append(f"{key} waits for leak item {item}")
            continue
        asm = jobs.soft_assemble_queue(carrier)
        if not store.queue_exists(cfg.queue_root, asm.name):
            _submit_all(cfg, [asm])
            continue
        if not _finished(cfg, [asm.name], waiting):
            continue
        gate_path = layout.pool_dir(carrier) / "pool_gate.json"
        gate = json.loads(gate_path.read_text()) if gate_path.is_file() else None
        if gate is None:
            _block(state, key, f"{gate_path} missing after pool-assemble")
            continue
        _unblock(state, key)
        passed = bool(gate.get("pass")) and pools.pool_ready(carrier)
        _fire(state, key, **{"pass": passed, "gate": gate})
        mark(key)
        if not passed:
            if carrier == plan.PILE:
                raise CampaignGateError(f"G-POOL failed for C-PILE: {gate}")
            logger.error("pool {} failed its gate; tier dropped: {}", carrier, gate)


def _soft_validity(cfg, state: dict, mark, waiting: list[str]) -> None:
    from pipelines import s2core_eval_pipeline as evalp
    from pipelines import s2core_results_pipeline as results

    done = state["transitions"]
    if "phase0_items" not in done:
        if "stack_choice" not in done or not _stack_current():
            return
        if not all(
            layout.eval_done_path(layout.base_model_dir(m), plan.FULL).is_file()
            for m in plan.BASE_MODELS
        ):
            waiting.append("phase-0 judge items wait for the 5 roster `full` evals")
            return
        items = write_generation_items(plan.BASE_MODELS, "phase0 roster E6/E7")
        _fire(state, "phase0_items", items=items)
        mark("phase0_items")
    if "validity" not in done:
        final_q = jobs.soft_final_queue().name
        if not _finished(cfg, [final_q], waiting):
            return
        if not evalp.phase0_ready():
            _block(
                state,
                "validity",
                "phase0-finalize finished but phase0_ready() is False",
            )
            return
        items = done["phase0_items"]["items"]
        settled, parked = judge_settlement(items)
        _record_judge_holes(state, items, parked)
        if not settled:
            waiting.append("validity waits for the phase-0 E6 / E7 judgments")
            return
        try:
            report = results.write_validity()
        except results.ReproductionError as err:
            raise CampaignGateError(f"E1 does not reproduce the gate: {err}") from err
        except results.ValidityNotReadyError as err:
            waiting.append(f"validity not ready: {err}")
            return
        except Exception as err:  # noqa: BLE001 - recorded as a blocker
            logger.exception("write_validity failed")
            _block(state, "validity", f"{type(err).__name__}: {err}")
            return
        _unblock(state, "validity")
        _fire(state, "validity", readouts=sorted(report) if report else [])
        mark("validity")


# ---- hard phase 0 --------------------------------------------------------------------


def _probe_questions_ready() -> bool:
    """The phase-0 probe questions every student probe reads (soft phase0-prep)."""
    return layout.PROBE_QUESTION_FILE.is_file()


def _hard_phase0(cfg, state: dict, mark, waiting: list[str]) -> None:
    done = state["transitions"]
    if "numbers" not in done:
        plans = [*jobs.hard_gen_queues(), jobs.hard_eq_queue()]
        _submit_all(cfg, plans)
        _fire(state, "numbers", queues=_names(plans))
        mark("numbers")
    if cfg.dry_run:
        return
    if "build42" not in done:
        if not _finished(cfg, jobs.hard_gen_queue_names(42), waiting):
            return
        _submit_all(cfg, [jobs.hard_build_queue((42,))])
        _fire(state, "build42")
        mark("build42")
    if "numbers_ok" not in done:
        _hard_numbers_verdict(cfg, state, mark, waiting)
    if "stage_a" not in done and "numbers_ok" in done:
        if not _probe_questions_ready():
            waiting.append(f"hard stage A waits for {layout.PROBE_QUESTION_FILE}")
            return
        plans = jobs.stage_train_queues(plan.HARD, jobs.STAGE_A, cells.hard_stage_a())
        _submit_all(cfg, plans)
        _fire(state, "stage_a", n_cells=len(cells.hard_stage_a()))
        mark("stage_a")


def _hard_numbers_verdict(cfg, state: dict, mark, waiting: list[str]) -> None:
    """G-NUM (plan §9.3): the seed-42 build, then the seed-42+43 build only if the
    seed-42 decision asks for the extension; a core shortfall after it halts hard."""
    from pipelines import s2core_numbers_pipeline as numbers

    done = state["transitions"]
    extended = "build4243" in done
    build = jobs.hard_build_queue((42, 43) if extended else (42,))
    if not _finished(cfg, [build.name], waiting):
        return
    g = numbers.g_num()
    if g is None:
        _block(state, "numbers", f"{build.name} finished but g_num() is None")
        return
    expected = [42, 43] if extended else [42]
    if list(g["prompt_seeds"]) != expected:
        _block(
            state, "numbers", f"g_num prompt seeds {g['prompt_seeds']} != {expected}"
        )
        return
    _unblock(state, "numbers")
    if g.get("needs_extension"):
        if extended:
            raise CampaignGateError(f"g_num asks for a second extension: {g}")
        if not _finished(cfg, jobs.hard_gen_queue_names(43), waiting):
            return
        _submit_all(cfg, [jobs.hard_build_queue((42, 43))])
        _fire(state, "build4243", preliminary=g)
        mark("build4243")
        return
    if not g.get("core_ok"):
        raise CampaignGateError(f"insufficient_pool: core pools short after build: {g}")
    _fire(state, "numbers_ok", tiers=g["tiers"], prompt_seeds=g["prompt_seeds"])
    mark("numbers_ok")


# ---- one advance pass ----------------------------------------------------------------


def run(cfg: driver_io.CampaignDriverConfig) -> AdvanceResult:
    """One advance pass of `cfg`'s channel."""
    channel = channel_of(cfg)
    if not cfg.dry_run:
        driver_io.ensure_campaign_code_pin(cfg)
    state = driver_io.load_state(cfg)
    done = state["transitions"]
    fired: list[str] = []
    waiting: list[str] = []

    def mark(name: str) -> None:
        fired.append(name)
        driver_io.save_state(cfg, state)

    if "preflight" not in done:
        if not _preflight(cfg, state):
            waiting.append(f"preflight: {state['blockers']['preflight']['msg']}")
            state["last_pass"] = {
                "at": utc_now_iso(),
                "fired": fired,
                "waiting": waiting,
            }
            driver_io.save_state(cfg, state)
            return AdvanceResult(tuple(fired), tuple(waiting), False)
        mark("preflight")
    if "register" not in done:
        _register(cfg, state)
        mark("register")
    if channel == plan.SOFT:
        _soft_phase0(cfg, state, mark, waiting)
    else:
        _hard_phase0(cfg, state, mark, waiting)
    note = _repin(cfg)
    if note:
        fired.append(note)
    all_done = False
    if not cfg.dry_run:
        _requeue_and_holes(cfg, state)
        _advance_stages(cfg, state, mark, waiting)
        _final_items(cfg, state, mark, waiting)
        all_done = _finalize(cfg, state, mark, waiting)
    if state.get("holes"):
        waiting.append(f"holes in {sorted(state['holes'])}")
    if state.get("blockers"):
        waiting.append(f"blockers {sorted(state['blockers'])}")
    state["last_pass"] = {"at": utc_now_iso(), "fired": fired, "waiting": waiting[:50]}
    driver_io.save_state(cfg, state)
    return AdvanceResult(tuple(fired), tuple(waiting), all_done)


def status(cfg: driver_io.CampaignDriverConfig) -> dict:
    """Transitions, holes, blockers and per-queue counts (pure file reads)."""
    state = driver_io.load_state(cfg)
    queues = {}
    for name in _submitted(cfg):
        s = store.queue_status(cfg.queue_root, name)
        queues[name] = {k: v for k, v in s.items() if k != "running_jobs"}
    return {
        "transitions": sorted(state.get("transitions", {})),
        "holes": state.get("holes", {}),
        "blockers": state.get("blockers", {}),
        "queues": queues,
    }


def write_policy(host: str, *, replace_existing: bool = False) -> Path:
    """Write `host`'s jobq policy into its channel's queue root (create or equal).

    Raises:
        FileExistsError: A different policy exists and `replace_existing` is False.
    """
    channel = plan.host_channel(host)
    path = plan.queue_root(channel) / f"gpu_policy.{host}.json"
    doc = plan.gpu_policy_json(host)
    if path.is_file() and not replace_existing:
        if json.loads(path.read_text()) != doc:
            raise FileExistsError(f"{path} exists with a different policy")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, doc)
    return path


# ---- dry-run enumeration (SM-6) ------------------------------------------------------


def enumerate_plan(channel: str) -> dict:
    """Every queue and job a channel's driver would submit, under a hypothetical
    selection (stage-A winner = the middle rung, final = stage A's winner at the base
    budget; every hard tier ok). Pure: no state, no files (SM-6).
    """
    ladder = plan.SOFT_LADDER if channel == plan.SOFT else plan.HARD_LADDER
    r_a = ladder[len(ladder) // 2]
    if channel == plan.SOFT:
        p0 = [
            jobs.soft_prep_queue(),
            *jobs.soft_base_queues(),
            jobs.soft_final_queue(),
            jobs.soft_lexicon_queue(),
            *(jobs.soft_leak_queue(c) for c in plan.CARRIERS),
            *(jobs.soft_assemble_queue(c) for c in plan.CARRIERS),
        ]
        a = cells.soft_stage_a()
        b = cells.soft_stage_b(r_a)
        c = cells.soft_stage_c(r_a, plan.SOFT_BASE_BUDGET, a + b)
        full = cells.soft_full_eval_cells(r_a, plan.SOFT_BASE_BUDGET)
    else:
        p0 = [
            *jobs.hard_gen_queues(),
            jobs.hard_build_queue((42,)),
            jobs.hard_build_queue((42, 43)),
            jobs.hard_eq_queue(),
        ]
        steps = layout.hard_steps(plan.HARD_BASE_EPOCHS)
        a = cells.hard_stage_a()
        b = cells.hard_stage_b(r_a)
        c = cells.hard_stage_c(r_a, steps, (plan.NOK, plan.UNIQ, plan.PFX), a + b)
        full = cells.hard_full_eval_cells(r_a, steps)
    plans = list(p0)
    for stage, cs in ((jobs.STAGE_A, a), (jobs.STAGE_B, b), (jobs.STAGE_C, c)):
        q = _stage_queue_plans(channel, stage, cs)
        plans += q[jobs.TRAIN] + q[jobs.SELECT]
    plans += jobs.stage_full_queues(channel, c, full)
    jobs.validate_queue_plans(plans)
    return {
        "channel": channel,
        "hypothetical_r_a": r_a,
        "cells": {"a": len(a), "b": len(b), "c": len(c), "full": len(full)},
        "queues": {
            q.name: {
                "host": q.host,
                "n_jobs": len(q.jobs),
                "depends_on": list(q.depends_on),
                "footprints": sorted({(j.mem_mib, j.slots) for j in q.jobs}),
                "priority": jobs.QUEUE_PRIORITIES[channel][q.name],
            }
            for q in plans
        },
        "n_jobs": sum(len(q.jobs) for q in plans),
    }
