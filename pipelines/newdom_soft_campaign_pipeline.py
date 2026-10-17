"""Auto-advancing driver of the `newdom` soft channel, on soar (unpinned queues in the
campaign queue root, drained by every node's pool).

One `run(cfg)` = one idempotent advance pass:

    preflight      prereg, home node, roots, soft paths YAML, complete staging report,
                   docgen venv, disk floor
    register       the campaign pin can run every job
    submit_docgen  the 96 whole-card shards, target-major, in one queue
    (every pass)   unfinished queues follow the pin; failures under the requeue cap are
                   requeued, the rest are holes; below the disk floor nothing new is
                   submitted (labelled `disk_low`)
    per target     16 DONE shards -> `sdf_docgen.select` (inline, docgen venv) -> review
                   sample -> wait for an approved review -> ep5 teachers (train queue +
                   strict eval queue) -> W0; a failure submits the ep8 dose arm, and a
                   second failure proceeds at ep8 labelled `w0_fail|<target>` ->
                   teacher manifest check and step count -> treatment leaves + placebo
                   teachers at that step count -> placebo leaves once those trained
    results        every target's leaf queues submitted, every queue finished, no hole
                   -> a results build (`newdom_soft_results_pipeline.build`)
    finalize       the channel done marker; the campaign marker when the hard
                   channel's marker already exists (both create-only; otherwise the
                   hard driver writes it when it finishes)

A target-level contract failure (partial corpus, review identity mismatch, unreadable
W0 inputs, off-plan teacher manifest) is a hole of that target; the others advance.
Passes hold an exclusive flock in the run-set logs dir, and every pass ends by touching
the shared and the soft driver heartbeats.

The job side of a docgen shard (`run_docgen_shard`) lives here too.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import yaml
from loguru import logger

from pipelines import newdom_soft_results_pipeline as results
from pipelines import newdom_soft_staging_pipeline as staging
from pipelines.optparam2_campaign_pipeline import SPEC as OPTPARAM2_SPEC
from pipelines.optparam2_campaign_pipeline import (
    AdvanceResult,
    CampaignGateError,
    _fire,
    current_pin,
)
from pipelines.traitfree_staging_pipeline import write_once_text
from pipelines.xopt2_soft_campaign_pipeline import write_campaign_marker, write_marker
from subliminal_learning.campaign import driver_io
from subliminal_learning.campaign import newdom_plan as plan
from subliminal_learning.campaign import newdom_soft as soft
from subliminal_learning.evaluation.domains import domain_files
from subliminal_learning.gkd.layout import GKD_STUDENTS_SUBDIR
from subliminal_learning.jobqueue import store
from subliminal_learning.jobqueue.model import Job
from subliminal_learning.sdf.layout import SDF_TEACHERS_SUBDIR
from subliminal_learning.utils import code_pin_io
from subliminal_learning.utils.io import file_sha256
from subliminal_learning.training.run_set import run_set_subdir
from subliminal_learning.utils.run_io import read_run_set_card, utc_now_iso


def _pin_paths() -> tuple[str, ...]:
    files = [str(p) for d in plan.DOMAINS for p in vars(domain_files(d)).values()]
    specs = [f"data/sdf_prompts/{plan.TARGET_DOMAIN[t]}/{t}.json" for t in plan.TARGETS]
    prompts = [
        f"data/evaluation_prompts/{plan.DOMAINS[d].noun}_preference_{s}.json"
        for d in plan.DOMAINS
        for s in ("train", "validation", "test", "dislike")
    ]
    return (
        "scripts/sweeps/sweep_sdf_teacher.py",
        "scripts/sweeps/sweep_evaluate_sdf_teacher.py",
        "scripts/sweeps/sweep_distill_student_gkd.py",
        "scripts/sweeps/sweep_evaluate_gkd_student.py",
        "scripts/newdom_panel.py",
        "scripts/newdom_docgen_shard.py",
        "scripts/run_newdom_soft_campaign.py",
        "scripts/stage_newdom_soft_assets.py",
        "scripts/build_gkd_prompts.py",
        "scripts/jobq_pool_loop.sh",
        soft.TEACHER_TRAINING_CONFIG,
        soft.STUDENT_TRAINING_CONFIG,
        *(str(p) for p in plan.DOCGEN_CONFIGS.values()),
        *(f"data/sdf_prompts/pile_wrap_v2_{d}.json" for d in plan.DOMAINS),
        *specs,
        *files,
        *prompts,
        "data/evaluation_prompts/off_topic_nocolor.json",
        "tools/sdf_docgen/src/sdf_docgen/cli.py",
        "tools/sdf_docgen/src/sdf_docgen/select.py",
        "src/subliminal_learning/campaign/newdom_plan.py",
        "src/subliminal_learning/campaign/newdom_soft.py",
        "src/subliminal_learning/campaign/newdom_soft_metrics.py",
        "src/subliminal_learning/evaluation/domains.py",
        "src/subliminal_learning/evaluation/preference_prob_io.py",
        "pipelines/newdom_soft_campaign_pipeline.py",
        "pipelines/newdom_soft_results_pipeline.py",
        "pipelines/newdom_soft_staging_pipeline.py",
        "pipelines/newdom_panel_pipeline.py",
    )


SPEC = replace(
    OPTPARAM2_SPEC,
    run_set_id=soft.RUN_SET_ID,
    teacher_run_set_id=soft.RUN_SET_ID,
    root_subdir=GKD_STUDENTS_SUBDIR,
    queue_priorities=soft.QUEUE_PRIORITIES,
    gpu_policy={"cap_per_gpu": plan.CAP_PER_GPU},
    generated_config_dir=str(soft.GENERATED_DIR),
    campaign_state_file=soft.STATE_FILE,
    campaign_done_file=plan.channel_done_path(plan.SOFT).name,
    gkd_experiment_config=None,
    extra_pin_paths=_pin_paths(),
)


PASS_LOCK_FILE = "driver_pass.lock"


class TargetHold(RuntimeError):
    """A target-level contract failed (a hole of that target)."""


def driver_config(**kwargs) -> driver_io.CampaignDriverConfig:
    """A `CampaignDriverConfig` bound to the soft spec (state and logs under
    `newdom_plan.run_set_dir(SOFT)`)."""
    kwargs.setdefault("generated_dir_override", soft.GENERATED_DIR)
    return driver_io.CampaignDriverConfig(spec=SPEC, **kwargs)


def pin_config(cfg: driver_io.CampaignDriverConfig) -> driver_io.CampaignDriverConfig:
    """`cfg` rooted at the run-set card that pins the code: the soft teacher run set
    in production, the driver's own root otherwise (tests)."""
    if Path(cfg.output_root) != soft.OUTPUT_ROOT:
        return cfg
    return replace(cfg, spec=replace(cfg.spec, root_subdir=SDF_TEACHERS_SUBDIR))


def queue_defaults(cfg: driver_io.CampaignDriverConfig) -> dict:
    """jobq defaults of an unpinned queue: the pin worktree under the one clone path
    every host shares.

    Raises:
        CampaignGateError: The hosts' clone paths differ.
    """
    clones = {plan.CLONE_ROOT[h] for h in plan.HOSTS}
    if len(clones) != 1:
        raise CampaignGateError(f"unpinned queues need one clone path; got {clones}")
    root = clones.pop() / code_pin_io.CODE_PINS_DIRNAME
    if cfg.dry_run:
        return {"cwd": str(root)}
    sha = current_pin(pin_config(cfg)).sha
    return code_pin_io.pin_job_defaults(code_pin_io.CodePin(sha=sha, root=root / sha))


# ---- queues -------------------------------------------------------------------------------


def submitted(cfg: driver_io.CampaignDriverConfig) -> list[str]:
    return [n for n in soft.QUEUE_PRIORITIES if store.queue_exists(cfg.queue_root, n)]


def missing_jobs(
    cfg: driver_io.CampaignDriverConfig, name: str, jobs: list[Job]
) -> list[Job]:
    """The jobs of `jobs` queue `name` does not hold (all when it does not exist)."""
    if not store.queue_exists(cfg.queue_root, name):
        return list(jobs)
    have = {j.jobkey for j in store.load_jobs(cfg.queue_root, name)}
    return [j for j in jobs if j.jobkey not in have]


def _submit(
    cfg: driver_io.CampaignDriverConfig,
    name: str,
    jobs: list[Job],
    *,
    depends_on: Sequence[str] = (),
    strict: bool = False,
) -> int:
    """Create unpinned `name` with `jobs`, or append the ones it lacks (moving it to
    the current pin first); return the number added."""
    if not store.queue_exists(cfg.queue_root, name):
        driver_io.submit_queue(
            cfg,
            name,
            jobs,
            depends_on=list(depends_on) or None,
            defaults=queue_defaults(cfg),
            strict_deps=strict,
            unpinned=True,
        )
        return len(jobs)
    new = missing_jobs(cfg, name, jobs)
    if new and not cfg.dry_run:
        store.update_queue_defaults(cfg.queue_root, name, queue_defaults(cfg))
        store.append_jobs(cfg.queue_root, name, new)
        logger.info("appended {} job(s) to {}", len(new), name)
    return len(new)


def write_yamls(cfg: driver_io.CampaignDriverConfig, docs: Mapping[Path, dict]) -> int:
    """Create (or match) generated YAMLs."""
    if not cfg.dry_run:
        for path, doc in docs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            write_once_text(path, yaml.safe_dump(doc, sort_keys=True))
    return len(docs)


def _repin(cfg: driver_io.CampaignDriverConfig) -> str | None:
    """Move every queue with work left (or failures) to the campaign pin."""
    if cfg.dry_run:
        return None
    want = queue_defaults(cfg)
    n = 0
    for name in submitted(cfg):
        if (
            store.queue_complete(cfg.queue_root, name)
            and not store.queue_status(cfg.queue_root, name)["failed"]
        ):
            continue
        if store.read_meta(cfg.queue_root, name).defaults.get("cwd") != want["cwd"]:
            store.update_queue_defaults(cfg.queue_root, name, want)
            n += 1
    return f"repin(queues={n})" if n else None


def _requeue_and_holes(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    holes = state.setdefault("holes", {})
    for name in submitted(cfg):
        try:
            driver_io.requeue_failed_under_cap(cfg, name)
        except RuntimeError as err:
            if holes.get(name) != str(err):
                logger.error("hole in {}: {}", name, err)
            holes[name] = str(err)
        else:
            holes.pop(name, None)


def disk_problems() -> list[str]:
    path = plan.ROOT
    while not path.exists():
        path = path.parent
    free = shutil.disk_usage(path).free / 2**30
    return (
        [f"{free:.0f} GiB free < {plan.MIN_FREE_GIB}"]
        if free < plan.MIN_FREE_GIB
        else []
    )


# ---- preflight, pin -----------------------------------------------------------------------


def preflight_problems(cfg: driver_io.CampaignDriverConfig) -> list[str]:
    problems = []
    if not (cfg.repo_root / plan.PREREG_PATH).is_file():
        problems.append(f"missing pre-registration {plan.PREREG_PATH}")
    host = store.this_host()
    if host != plan.HOME_NODE:
        problems.append(f"the driver runs on {plan.HOME_NODE}, not {host}")
    elif not soft.docgen_python(host).exists():
        problems.append(f"no docgen venv at {soft.docgen_python(host)}")
    if Path(cfg.queue_root).resolve() != plan.QUEUE_ROOT.resolve():
        problems.append(f"queue root {cfg.queue_root} is not the plan's")
    if Path(cfg.output_root) != soft.OUTPUT_ROOT:
        problems.append(f"output root {cfg.output_root} is not the plan's")
    if (
        not soft.PATHS_CONFIG_PATH.is_file()
        or soft.PATHS_CONFIG_PATH.read_text() != staging.paths_yaml_text()
    ):
        problems.append(f"{soft.PATHS_CONFIG_PATH} missing or differs from the plan")
    if staging.latest_complete_report() is None:
        problems.append(
            "no complete staging report; run scripts.stage_newdom_soft_assets"
        )
    return problems + disk_problems()


def pin_problem(cfg: driver_io.CampaignDriverConfig, state: dict) -> str | None:
    """Why the pass must not run under this pin, or None.

    Before the soft teacher card exists the pin is `--code-sha`, which must match the
    registered one unless `auto_reconcile_sha` (an intentional re-pin); the two student
    run-set cards and the hard channel's card, once they exist, must name the same
    pin.
    """
    resolve = code_pin_io.resolve_commit_sha
    card = read_run_set_card(pin_config(cfg).run_set_root) or {}
    mine = card.get("git_sha") or cfg.code_sha
    if not mine:
        return "pass --code-sha until the soft teacher run-set card exists"
    mine = resolve(cfg.repo_root, str(mine))
    registered = (state["transitions"].get("register") or {}).get("git_sha")
    if not card and registered and registered != mine and not cfg.auto_reconcile_sha:
        return (
            f"--code-sha {mine[:12]} differs from the registered pin {registered[:12]}"
        )
    others = {
        rs: cfg.output_root / run_set_subdir(GKD_STUDENTS_SUBDIR, rs)
        for rs in soft.STUDENT_RUN_SETS.values()
    }
    if Path(cfg.output_root) == soft.OUTPUT_ROOT:
        others["hard channel"] = plan.run_set_dir(plan.HARD)
    for name, root in others.items():
        sha = (read_run_set_card(root) or {}).get("git_sha")
        if sha and resolve(cfg.repo_root, str(sha)) != mine:
            return f"{name} pinned to {str(sha)[:12]}, soft teachers to {mine[:12]}"
    return None


def _preflight(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    problems = preflight_problems(cfg)
    if problems and not cfg.dry_run:
        raise CampaignGateError("; ".join(problems))
    for p in problems:
        logger.warning("[dry-run] preflight would refuse: {}", p)
    _fire(state, "preflight")


def _advance_registered_pin(
    cfg: driver_io.CampaignDriverConfig, state: dict
) -> str | None:
    """Record an intentional pre-card re-pin (`auto_reconcile_sha`) in the state."""
    reg = state["transitions"].get("register")
    if not reg:
        return None
    sha = current_pin(pin_config(cfg)).sha
    if reg.get("git_sha") == sha:
        return None
    reg.setdefault("advances", []).append(
        {"at": utc_now_iso(), "from": reg.get("git_sha"), "to": sha}
    )
    reg["git_sha"] = sha
    return f"registered_pin({sha[:12]})"


def _register(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    if cfg.dry_run:
        sha = code_pin_io.resolve_commit_sha(cfg.repo_root, cfg.code_sha or "HEAD")
        _fire(state, "register", git_sha=sha)
        return
    pin = current_pin(pin_config(cfg))
    driver_io.verify_pin_runs_campaign(pin, SPEC)
    _fire(state, "register", git_sha=pin.sha)


# ---- corpus, review ------------------------------------------------------------------------


def shard_inputs(target: str) -> list[Path]:
    """The DONE attempt dir of every shard of `target`, in shard order.

    Raises:
        TargetHold: A DONE record contradicts its shard or its documents.
    """
    out = []
    for shard in soft.docgen_shards((target,)):
        doc = json.loads(shard.done_path.read_text())
        if bad := soft.done_problems(doc, shard):
            raise TargetHold(f"{shard.done_path}: {bad}")
        attempt = shard.dir / doc["attempt"]
        if file_sha256(attempt / soft.docs_file(target)) != doc["docs_sha256"]:
            raise TargetHold(f"{attempt}: documents differ from {shard.done_path}")
        out.append(attempt)
    return out


def corpus_ready(target: str) -> bool:
    return (
        plan.corpus_path(target).is_file()
        and soft.corpus_manifest_path(target).is_file()
    )


def verify_corpus(target: str) -> dict:
    """sha, size and funnel of a selected corpus.

    Raises:
        TargetHold: The corpus differs from its manifest or names another target.
    """
    corpus = plan.corpus_path(target)
    man = json.loads(soft.corpus_manifest_path(target).read_text())
    sha = file_sha256(corpus)
    rows = staging.read_jsonl(corpus)
    if sha != man.get("output_sha256") or len(rows) != man["funnel"]["final"]:
        raise TargetHold(f"{corpus} differs from its manifest")
    if (names := {r.get("animal") for r in rows}) != {target}:
        raise TargetHold(f"{corpus} rows name {sorted(map(str, names))}")
    return {"sha256": sha, "n_docs": len(rows), "funnel": man["funnel"]}


def pooled_docs(target: str) -> int:
    """Accepted documents over every DONE shard of `target`."""
    return sum(
        int(json.loads(s.done_path.read_text())["n_docs"])
        for s in soft.docgen_shards((target,))
    )


def build_corpus(cfg: driver_io.CampaignDriverConfig, target: str) -> None:
    """Run `sdf_docgen.select` over every shard of `target` (inline, docgen venv)
    into a staging dir beside the corpus dir, then rename it into place.

    Raises:
        TargetHold: The corpus dir exists unverified, or select failed.
    """
    final = soft.corpus_dir(target)
    if final.exists():
        raise TargetHold(f"{final} exists without a verified corpus; inspect it")
    inputs = shard_inputs(target)
    stamp = utc_now_iso().replace(":", "")
    stage = final.parent / f".{target}.select-{stamp}"
    pin = current_pin(pin_config(cfg))
    env = {
        **os.environ,
        "PYTHONPATH": str(pin.root / soft.DOCGEN_SRC),
        code_pin_io.CODE_PIN_SHA_ENV: pin.sha,
        "OMP_NUM_THREADS": "8",
    }
    argv = soft.select_argv(
        soft.docgen_python(store.this_host()), target, inputs, output=stage
    )
    log = plan.DOCGEN_ROOT / target / f"select-{stamp}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    logger.info("select {}: {}", target, " ".join(argv))
    with log.open("x") as handle:
        proc = subprocess.run(
            argv,
            cwd=pin.root,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    staged = stage / plan.corpus_path(target).name
    if proc.returncode != 0 or not staged.is_file():
        raise TargetHold(f"select {target} exit {proc.returncode} (log {log})")
    os.rename(stage, final)


def write_corpus_sample(target: str, corpus_sha: str) -> Path:
    """Seeded kept / dropped sample of a corpus, with the review file to write."""
    pooled = [
        row
        for attempt in shard_inputs(target)
        for row in staging.read_jsonl(attempt / soft.docs_file(target))
    ]
    kept = staging.read_jsonl(plan.corpus_path(target))
    kept_texts = {r["text"] for r in kept}
    dropped = [r for r in pooled if r["text"] not in kept_texts]
    fields = ("pile_set_name", "placement", "intensity", "voice", "text")

    def sample(rows: list[dict]) -> list[dict]:
        return [
            {"line": i, **{k: rows[i].get(k) for k in fields}}
            for i in soft.review_sample(len(rows))
        ]

    doc = {
        "target": target,
        "domain": plan.TARGET_DOMAIN[target],
        "corpus_path": str(plan.corpus_path(target)),
        "corpus_sha256": corpus_sha,
        "n_pooled": len(pooled),
        "n_kept": len(kept),
        "n_dropped": len(dropped),
        "seed": soft.REVIEW_SEED,
        "kept": sample(kept),
        "dropped": sample(dropped),
        "review_path": str(plan.corpus_review_path(target)),
    }
    doc["review_template"] = {
        **soft.corpus_review_fields(target, corpus_sha, doc),
        "approved": False,
        "notes": "",
    }
    path = soft.corpus_sample_path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_once_text(path, staging.json_text(doc))
    return path


def corpus_review(target: str, corpus_sha: str) -> bool:
    """Whether the corpus review is approved (fail-closed).

    Raises:
        TargetHold: The review is unreadable, incomplete, or names another corpus,
            target or sample.
    """
    ok, problem = results.corpus_review_status(target, corpus_sha)
    if problem:
        raise TargetHold(problem)
    return ok


# ---- teachers ---------------------------------------------------------------------------


def submit_teacher(
    cfg: driver_io.CampaignDriverConfig, ref: soft.TeacherRef
) -> list[str]:
    """The config's train queue and its strict eval queue (missing jobs added)."""
    notes = []
    jobs = soft.teacher_train_jobs(ref)
    if missing_jobs(cfg, ref.train_queue, jobs):
        write_yamls(cfg, soft.teacher_yaml_docs(ref))
        n = _submit(cfg, ref.train_queue, jobs)
        notes.append(f"submit({ref.train_queue}, +{n})")
    if _submit(
        cfg,
        ref.eval_queue,
        [soft.teacher_eval_job(ref)],
        depends_on=[ref.train_queue],
        strict=True,
    ):
        notes.append(f"submit({ref.eval_queue})")
    return notes


def read_w0(ref: soft.TeacherRef) -> soft.W0Verdict:
    """W0 of one evaluated treatment config on the domain `validation` set, from its
    eval summary and base control.

    Raises:
        TargetHold: Missing or inconsistent eval artifacts.
    """
    test = soft.eval_set_name(ref.domain, soft.W0_SET)
    try:
        summary = json.loads((ref.param_dir / "preference_summary.json").read_text())
        base = json.loads((ref.param_dir.parent / "base_control.json").read_text())[
            test
        ]
        entries = [
            e
            for e in summary["summaries"]
            if e["prompt_set"] == test and e["scoring"] == "prob"
        ]
        if summary.get("target_animal") != ref.target or len(entries) != 1:
            raise ValueError(f"no single {test} summary for {ref.target}")
        if base.get("target_animal") != ref.target:
            raise ValueError(f"base control names {base.get('target_animal')!r}")
        per_seed = entries[0]["per_seed_p_target"]
        if sorted(per_seed) != sorted(str(s) for s in soft.TEACHER_FTSEEDS):
            raise ValueError(f"evaluated ftseeds {sorted(per_seed)}")
        verdict = soft.w0_verdict(
            ref.epochs,
            [per_seed[str(s)] for s in soft.TEACHER_FTSEEDS],
            base["p_per_prompt"],
        )
        if not math.isclose(
            verdict.base_p, entries[0]["base_p_target"], rel_tol=1e-6, abs_tol=1e-12
        ):
            raise ValueError("base p differs between the summary and base_control")
    except (OSError, KeyError, ValueError) as err:
        raise TargetHold(f"W0 {ref.target} ep{ref.epochs}: {err}") from err
    return verdict


def verify_teachers(ref: soft.TeacherRef, docs_sha: str) -> int:
    """Check every ftseed's manifest against the config and the reviewed (treatment)
    or staged (placebo) corpus sha; return the step count.

    Raises:
        TargetHold: The corpus changed, a manifest is missing or off-plan, or step
            counts differ.
    """
    try:
        if file_sha256(ref.docs_path) != docs_sha:
            raise ValueError(f"{ref.docs_path} changed since it was reviewed")
        manifests = {}
        for r in soft.TEACHER_FTSEEDS:
            m = json.loads(ref.manifest_path(r).read_text())
            if bad := soft.teacher_manifest_problems(ref, r, m, docs_sha):
                raise ValueError(f"ftseed{r} manifest off-plan {bad}")
            manifests[r] = m
        return soft.teacher_steps(manifests)
    except (OSError, ValueError) as err:
        raise TargetHold(f"teachers {ref.param_dir}: {err}") from err


def submit_leaves(
    cfg: driver_io.CampaignDriverConfig, teacher: soft.TeacherRef
) -> list[str]:
    """Every leaf distilled from `teacher`, in its leaf queue (missing jobs added)."""
    leaves = soft.leaves_for(teacher)
    name = leaves[0].queue
    jobs = soft.leaf_jobs(leaves)
    new = {j.key for j in missing_jobs(cfg, name, jobs)}
    if not new:
        return []
    write_yamls(
        cfg,
        {
            p: d
            for lf in leaves
            if lf.key in new
            for p, d in soft.leaf_yaml_docs(lf).items()
        },
    )
    n = _submit(cfg, name, jobs)
    return [f"submit({name}, +{n})"]


# ---- one target -------------------------------------------------------------------------


def advance_target(
    cfg: driver_io.CampaignDriverConfig, state: dict, target: str, waiting: list[str]
) -> list[str]:
    """Advance one target's chain as far as its gates allow; return notes.

    Raises:
        TargetHold: A target-level contract failed.
    """
    ts = state.setdefault("targets", {}).setdefault(target, {})
    notes: list[str] = []
    if "corpus" not in ts:
        if not corpus_ready(target):
            left = [
                s for s in soft.docgen_shards((target,)) if not s.done_path.is_file()
            ]
            if left:
                waiting.append(f"{target}: {len(left)} docgen shards without DONE.json")
                return notes
            ts["pooled_docs"] = pooled_docs(target)
            if ts["pooled_docs"] < soft.DOCGEN_MIN_POOLED:
                raise TargetHold(
                    f"{target}: {ts['pooled_docs']} accepted documents over the 16 "
                    f"shards < {soft.DOCGEN_MIN_POOLED}"
                )
            build_corpus(cfg, target)
            notes.append(f"corpus({target})")
        ts["corpus"] = {**verify_corpus(target), "at": utc_now_iso()}
    sha = ts["corpus"]["sha256"]
    if not soft.corpus_sample_path(target).is_file():
        write_corpus_sample(target, sha)
        notes.append(f"review_sample({target})")
    if "review" not in ts:
        if not corpus_review(target, sha):
            waiting.append(f"{target}: corpus review {plan.corpus_review_path(target)}")
            return notes
        ts["review"] = {
            "at": utc_now_iso(),
            "sha256": file_sha256(plan.corpus_review_path(target)),
        }
        notes.append(f"review_approved({target})")
    if soft.corpus_shadow_path(target).exists():
        raise TargetHold(f"{soft.corpus_shadow_path(target)} would shadow the corpus")
    w0 = ts.setdefault("w0", {})
    for epochs in soft.TEACHER_EPOCHS:
        if str(epochs) in w0:
            if w0[str(epochs)]["passed"]:
                break
            continue
        ref = soft.TeacherRef(soft.KIND_TREAT, target, epochs)
        notes += submit_teacher(cfg, ref)
        finished, msg = driver_io.queue_finished(cfg, ref.eval_queue)
        if not finished:
            waiting.append(f"{target}: {msg}")
            return notes
        verdict = read_w0(ref)
        w0[str(epochs)] = soft.verdict_doc(verdict)
        notes.append(f"w0({target} ep{epochs} {'pass' if verdict.passed else 'fail'})")
        if verdict.passed:
            break
    chosen = soft.choose_epochs({int(k): bool(v["passed"]) for k, v in w0.items()})
    if chosen is None:
        return notes
    if not w0[str(chosen)]["passed"]:
        state.setdefault("labelled", {})[f"w0_fail|{target}"] = (
            f"W0 failed at every dose; students distil from ep{chosen}"
        )
    treat = soft.TeacherRef(soft.KIND_TREAT, target, chosen)
    if "steps" not in ts:
        ts["steps"] = verify_teachers(treat, sha)
        ts["chosen_epochs"] = chosen
        notes.append(f"teachers({target} ep{chosen}, {ts['steps']} steps)")
    notes += submit_leaves(cfg, treat)
    placebo = soft.TeacherRef(soft.KIND_PLACEBO, target, chosen, int(ts["steps"]))
    notes += submit_teacher(cfg, placebo)
    finished, msg = driver_io.queue_finished(cfg, placebo.train_queue)
    if not finished:
        waiting.append(f"{target}: {msg}")
        return notes
    if "placebo_verified" not in ts:
        slot = json.loads(soft.placebo_slot_manifest_path(target).read_text())
        verify_teachers(placebo, slot["docs_sha256"])
        ts["placebo_verified"] = utc_now_iso()
    notes += submit_leaves(cfg, placebo)
    return notes


def chain_complete(cfg: driver_io.CampaignDriverConfig, state: dict) -> bool:
    """Every target's leaf queues exist and every submitted queue finished."""
    targets = state.get("targets", {})
    for target in plan.TARGETS:
        if "placebo_verified" not in (targets.get(target) or {}):
            return False
        for kind in soft.KINDS:
            if not store.queue_exists(cfg.queue_root, soft.leaf_queue(kind, target)):
                return False
    return all(driver_io.queue_finished(cfg, n)[0] for n in submitted(cfg))


# ---- the pass -----------------------------------------------------------------------------


def run(cfg: driver_io.CampaignDriverConfig) -> AdvanceResult:
    """One advance pass, under an exclusive flock in the run-set logs dir so two
    passes (advance / watch, any host) never interleave."""
    if cfg.dry_run:
        return _pass(cfg)
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    with (cfg.logs_dir / PASS_LOCK_FILE).open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        return _pass(cfg)


def _target_pass(
    cfg: driver_io.CampaignDriverConfig, state: dict, target: str, waiting: list[str]
) -> list[str]:
    """`advance_target`, with any target-level failure recorded as that target's
    hole (the other targets still advance)."""
    holes = state.setdefault("holes", {})
    try:
        notes = advance_target(cfg, state, target, waiting)
    except CampaignGateError:
        raise
    except (TargetHold, OSError, ValueError, KeyError, RuntimeError) as err:
        msg = (
            str(err) if isinstance(err, TargetHold) else f"{type(err).__name__}: {err}"
        )
        if holes.get(f"target:{target}") != msg:
            logger.opt(exception=not isinstance(err, TargetHold)).error(
                "hole in {}: {}", target, msg
            )
        holes[f"target:{target}"] = msg
        return []
    holes.pop(f"target:{target}", None)
    return notes


def _pass(cfg: driver_io.CampaignDriverConfig) -> AdvanceResult:
    state = driver_io.load_state(cfg)
    if not cfg.dry_run:
        if bad := pin_problem(cfg, state):
            raise CampaignGateError(bad)
        driver_io.ensure_campaign_code_pin(pin_config(cfg))
    done = state["transitions"]
    holes = state.setdefault("holes", {})
    labelled = state.setdefault("labelled", {})
    fired: list[str] = []
    waiting: list[str] = []
    if not cfg.dry_run and (note := _advance_registered_pin(cfg, state)):
        fired.append(note)

    def mark(name: str) -> None:
        fired.append(name)
        driver_io.save_state(cfg, state)

    if "preflight" not in done:
        _preflight(cfg, state)
        mark("preflight")
    if "register" not in done:
        _register(cfg, state)
        mark("register")
    if note := _repin(cfg):
        fired.append(note)
    if not cfg.dry_run:
        _requeue_and_holes(cfg, state)
    disk = [] if cfg.dry_run else disk_problems()
    if disk:
        labelled["disk_low"] = "; ".join(disk)
        waiting.append(f"new submissions held: {labelled['disk_low']}")
    else:
        labelled.pop("disk_low", None)
    if "submit_docgen" not in done and not disk:
        _submit(cfg, soft.docgen_queue(), soft.docgen_jobs())
        _fire(state, "submit_docgen", n_jobs=len(soft.docgen_shards()))
        mark("submit_docgen")
    if cfg.dry_run:
        waiting.append("dry-run: per-target chain, results and finalize not evaluated")
        return AdvanceResult(tuple(fired), tuple(waiting), False)
    if not disk:
        for target in plan.TARGETS:
            fired.extend(_target_pass(cfg, state, target, waiting))
            driver_io.save_state(cfg, state)
    blocking = sorted(k for k in holes if k != "results")
    if (
        "results" not in done
        and not blocking
        and not disk
        and chain_complete(cfg, state)
    ):
        try:
            build = results.build(
                cfg.queue_root, state["targets"], cfg.run_set_root / "results"
            )
        except ValueError as err:
            holes["results"] = str(err)
            logger.error("results not built: {}", err)
        else:
            holes.pop("results", None)
            _fire(state, "results", build=str(build))
            mark("results")
    all_done = False
    if "results" in done:
        if "finalize" not in done:
            write_marker(cfg.logs_dir / SPEC.campaign_done_file)
            _fire(state, "finalize", labelled=sorted(labelled))
            mark("finalize")
        if cfg.logs_dir == plan.logs_dir(plan.SOFT) and write_campaign_marker(
            plan.channel_done_path(plan.HARD), plan.CAMPAIGN_DONE_PATH
        ):
            fired.append("campaign_marker")
        all_done = True
    elif holes:
        waiting.append(f"holes in {sorted(holes)}")
    state["last_pass"] = {"at": utc_now_iso(), "fired": fired, "waiting": waiting[:50]}
    driver_io.save_state(cfg, state)
    if not cfg.dry_run:
        for beat in (
            plan.DRIVER_HEARTBEAT_PATH,
            plan.channel_heartbeat_path(plan.SOFT),
        ):
            beat.parent.mkdir(parents=True, exist_ok=True)
            beat.touch()
    return AdvanceResult(tuple(fired), tuple(waiting), all_done)


def status(cfg: driver_io.CampaignDriverConfig) -> dict:
    """Transitions, holes, labels, per-target progress and queue counts (file reads)."""
    state = driver_io.load_state(cfg)
    queues = {}
    for name in submitted(cfg):
        s = store.queue_status(cfg.queue_root, name)
        queues[name] = {k: v for k, v in s.items() if k != "running_jobs"}
    targets = {}
    for target in plan.TARGETS:
        ts = state.get("targets", {}).get(target, {})
        targets[target] = {
            "docgen_done": sum(
                s.done_path.is_file() for s in soft.docgen_shards((target,))
            ),
            "corpus_docs": (ts.get("corpus") or {}).get("n_docs"),
            "review": "review" in ts,
            "w0": {k: v.get("passed") for k, v in (ts.get("w0") or {}).items()},
            "chosen_epochs": ts.get("chosen_epochs"),
            "steps": ts.get("steps"),
            "placebo_verified": "placebo_verified" in ts,
        }
    return {
        "transitions": sorted(state.get("transitions", {})),
        "holes": state.get("holes", {}),
        "labelled": state.get("labelled", {}),
        "targets": targets,
        "queues": queues,
    }


# ---- job side: one docgen shard -------------------------------------------------------


def publish_once(path: Path, text: str) -> None:
    """Write `path` atomically and exclusively; an existing file is never replaced.

    Raises:
        FileExistsError: `path` exists.
    """
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    try:
        os.link(tmp, path)
    finally:
        tmp.unlink()


def new_attempt_dir(shard_dir: Path) -> Path:
    """Create and return the next unused ``attempt-KK`` dir of a shard."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    for k in range(100):
        candidate = shard_dir / f"attempt-{k:02d}"
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError(f"{shard_dir}: no free attempt slot")


def run_docgen_shard(target: str, index: int) -> Path:
    """Generate one shard into a fresh attempt dir, then write its DONE.json.

    Runs in the job's cwd (the pin root) with the host's docgen venv; a finished shard
    is skipped.

    Raises:
        RuntimeError: `sdf_docgen` would import from outside the pin, generation
            failed, or the accepted rows cannot enter the corpus.
    """
    shard = soft.DocgenShard(target, index)
    if shard.done_path.is_file():
        if bad := soft.done_problems(json.loads(shard.done_path.read_text()), shard):
            raise RuntimeError(f"{shard.done_path}: {bad}")
        logger.info("shard done: {}", shard.done_path)
        return shard.done_path
    root = Path.cwd().resolve()
    host = store.this_host()
    python = soft.docgen_python(host)
    env = {**os.environ, "PYTHONPATH": str(root / soft.DOCGEN_SRC)}
    guard = subprocess.run(
        [str(python), "-c", soft.IMPORT_GUARD], cwd=root, env=env, check=False
    )
    if guard.returncode != 0:
        raise RuntimeError(f"sdf_docgen does not import from {root / soft.DOCGEN_SRC}")
    attempt = new_attempt_dir(shard.dir)
    argv = soft.docgen_argv(shard, python, attempt)
    log = attempt / "docgen.log"
    logger.info("docgen {}: {}", shard.key, " ".join(argv))
    with log.open("x") as handle:
        proc = subprocess.run(
            argv,
            cwd=root,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{shard.key}: sdf_docgen.cli exit {proc.returncode} ({log})"
        )
    docs = attempt / soft.docs_file(target)
    rows = staging.read_jsonl(docs)
    if bad := soft.shard_problems(rows, target):
        raise RuntimeError(f"{docs}: {bad}")
    cfg_doc = yaml.safe_load(shard.config.read_text())
    spec = Path(str(cfg_doc["animal_spec_path"]).format(animal=target))
    prompts = Path(str(cfg_doc["prompts_path"]))
    doc = {
        "target": target,
        "domain": shard.domain,
        "index": index,
        "seed": shard.seed,
        "n_docs_requested": plan.DOCGEN_DOCS_PER_SHARD,
        "n_docs": len(rows),
        "attempt": attempt.name,
        "docs_sha256": file_sha256(docs),
        "inputs_sha256": {
            str(p): file_sha256(root / p)
            for p in (shard.config, spec, prompts, soft.PATHS_CONFIG_PATH)
        },
        "host": host,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "code_pin_sha": os.environ.get(code_pin_io.CODE_PIN_SHA_ENV),
        "created_at": utc_now_iso(),
    }
    publish_once(shard.done_path, staging.json_text(doc))
    logger.info("shard {}: {} docs -> {}", shard.key, len(rows), shard.done_path)
    return shard.done_path
