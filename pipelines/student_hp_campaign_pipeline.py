"""Automated stage machine for the 20260727 student-HP + cross-init campaign.

One `run(cfg)` = one advance pass over the stage DAG:

    A0 gate -> submit A1 + A2 + A1-evals + catswide backlog
    A1 + A2 queues done -> disk-complete gate -> A2 control-floor gate ->
        top-3 selection -> generate + submit A3 (+ A3-evals, dependent)
    A3 + A3-evals done -> disk + summary gate -> recipe freeze ->
        generate + submit Stage B headline/extension (+ B-evals; optional Stage C)
    everything done -> disk gate -> CAMPAIGN_DONE marker

Each transition is guarded three ways before it fires: its queues must be
complete with ZERO failed jobs, the previous stage's expected artifacts must all
exist on disk (queue exit codes alone are not trusted — a sweep job exits 0 even
when cells failed), and any selection gate (A2 flatness) must pass. A transition
that already ran is a no-op (queue jobkey-set equality + persisted state), so the
driver is safe to re-run or poll in a watch loop. Any hard failure raises
`CampaignGateError` with the exact remediation.

Campaign constants come from `cfg.spec` (default `plan.SPEC`). Pass a different
`CampaignSpec` (e.g. `breadth_plan.SPEC`) to drive another animal set.
"""

from __future__ import annotations

from dataclasses import dataclass

import yaml
from loguru import logger

from subliminal_learning.campaign import config_gen, driver_io, expected
from subliminal_learning.campaign.selection import (
    Recipe,
    apply_lr_extensions,
    control_floor_check,
    freeze_recipes,
    headline_beta,
    lr_edge_extensions,
    select_top_recipes,
)
from subliminal_learning.campaign.spec import CampaignSpec
from subliminal_learning.evaluation.gkd_results_io import (
    load_gkd_preference_results,
    load_gkd_probe_trajectories,
)
from subliminal_learning.jobqueue import store
from subliminal_learning.training.run_set import runs_index_row
from subliminal_learning.utils.io import atomic_write_text
from subliminal_learning.utils.run_io import (
    append_runs_index,
    read_run_set_card,
    utc_now_iso,
)


class CampaignGateError(RuntimeError):
    """A stage gate failed hard (not merely 'still running'); do not advance."""


@dataclass(frozen=True)
class AdvanceResult:
    """Outcome of one advance pass.

    Attributes:
        fired: Transitions that executed this pass.
        waiting: Human-readable reasons the next transition is not ready.
        done: The campaign has fully finished.
    """

    fired: tuple[str, ...]
    waiting: tuple[str, ...]
    done: bool


def _spec(cfg: driver_io.CampaignDriverConfig) -> CampaignSpec:
    return cfg.spec


def _load_yaml(cfg: driver_io.CampaignDriverConfig, rel: str) -> dict:
    return yaml.safe_load((cfg.repo_root / rel).read_text())


def _require_disk_complete(label: str, leaves, remediation: str) -> None:
    missing = expected.missing_manifests(leaves)
    if missing:
        raise CampaignGateError(
            f"{label}: {len(missing)}/{len(leaves)} cells have no completed "
            f"train_manifest.json (first: {missing[0]}). {remediation}"
        )


def _missing_verify_artifacts(param_dir, *, require_benchmarks: bool) -> list[str]:
    """Human-readable names of verify artifacts absent from `param_dir`."""
    missing: list[str] = []
    for name in (
        "preference_summary.json",
        "sdf_forced_choice_summary.json",
        "sdf_capability_summary.json",
    ):
        if not (param_dir / name).exists():
            missing.append(name)
    if require_benchmarks and (param_dir / "sdf_capability_summary.json").exists():
        try:
            cap = yaml.safe_load((param_dir / "sdf_capability_summary.json").read_text())
        except (OSError, yaml.YAMLError):
            missing.append("benchmarks (capability summary unreadable)")
        else:
            if "benchmarks" not in (cap or {}):
                missing.append("benchmarks (inside sdf_capability_summary.json)")
    return missing


def _require_verify_complete(
    label: str,
    config_dirs: list,
    *,
    require_benchmarks: bool,
    remediation: str,
) -> None:
    incomplete = expected.verify_summaries_complete(
        config_dirs, require_benchmarks=require_benchmarks
    )
    if not incomplete:
        return
    first = incomplete[0]
    missing = _missing_verify_artifacts(first, require_benchmarks=require_benchmarks)
    detail = ", ".join(missing) if missing else "verify triad incomplete"
    raise CampaignGateError(
        f"{label}: {len(incomplete)}/{len(config_dirs)} configs missing "
        f"complete verify summaries ({detail}; first: {first}). {remediation}"
    )


def _verify_doc_requires_benchmarks(verify_doc: dict) -> bool:
    """Whether a generated verify YAML enables lm-eval benchmarks."""
    bench = verify_doc.get("benchmarks")
    return bool(bench and bench.get("enabled"))


def _inject_reference_recipe(
    candidates: dict[tuple[str, float], list[Recipe]],
    ref: Recipe | None,
) -> dict[tuple[str, float], list[Recipe]]:
    """Append `ref` to every (animal, beta) list when absent (deduped)."""
    if ref is None:
        return candidates
    out: dict[tuple[str, float], list[Recipe]] = {}
    for key, recipes in candidates.items():
        if ref in recipes:
            out[key] = list(recipes)
        else:
            out[key] = [*recipes, ref]
    return out


def _code_pin_gate(cfg: driver_io.CampaignDriverConfig) -> str | None:
    """Ensure the MLflow-style code pin; refuse only an intentional pin mismatch.

    Workspace HEAD may move freely — cells run from ``.code_pins/<sha>/``.
    """
    try:
        return driver_io.ensure_campaign_code_pin(cfg)
    except (ValueError, RuntimeError) as err:
        raise CampaignGateError(str(err)) from err


# ---- transition 1: A0 gate -> initial queues -------------------------------------


def _a0_gate(cfg: driver_io.CampaignDriverConfig) -> None:
    if not cfg.require_a0:
        return
    spec = _spec(cfg)
    missing = [
        f
        for f in (spec.calibration_report, spec.init_equality_report)
        if not (cfg.logs_dir / f).exists()
    ]
    if missing:
        raise CampaignGateError(
            f"A0 incomplete: missing {missing} under {cfg.logs_dir}. Run "
            "`uv run python -m scripts.calibrate_gkd_throughput` and "
            "`uv run python -m scripts.check_step0_init_equality` first "
            "(or pass --skip-a0 to waive)."
        )


def _submit_initial(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    _a0_gate(cfg)
    spec = _spec(cfg)
    driver_io.submit_queue(cfg, spec.q_a1, driver_io.a1_jobs(spec=spec))
    driver_io.submit_queue(cfg, spec.q_a2, driver_io.a2_jobs(spec=spec))
    driver_io.submit_queue(
        cfg,
        spec.q_a1_evals,
        driver_io.a1_eval_jobs(spec=spec),
        depends_on=[spec.q_a1, spec.q_a2],
    )
    backlog = driver_io.backlog_jobs(spec=spec)
    if not backlog:
        logger.info(
            "backlog empty (no catswide verify YAMLs / base models) — "
            "skipping queue {}",
            spec.q_backlog,
        )
    else:
        driver_io.submit_queue(cfg, spec.q_backlog, backlog)
    state["transitions"]["submit_initial"] = {"at": utc_now_iso()}


# ---- transition 2: A1/A2 done -> gate -> select -> A3 ------------------------------


def _a1_a2_disk_gate(cfg: driver_io.CampaignDriverConfig) -> None:
    spec = _spec(cfg)
    for label, rel in (
        ("A1 same", spec.a1_same_yaml),
        ("A1 cross", spec.a1_cross_yaml),
    ):
        doc = _load_yaml(cfg, rel)
        _require_disk_complete(
            label,
            expected.treatment_leaf_dirs(doc, cfg.output_root, spec=spec),
            f"Requeue {spec.q_a1} failures or rerun its jobs, then re-advance.",
        )
    a2_doc = _load_yaml(cfg, spec.a1_same_yaml)
    a2_doc["grid"] = {**a2_doc["grid"], "student_modes": [spec.a2_student_mode]}
    _require_disk_complete(
        "A2 controls",
        expected.control_leaf_dirs(a2_doc, cfg.output_root, spec=spec),
        f"Requeue {spec.q_a2} failures, then re-advance.",
    )


def _submit_a3(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    _a1_a2_disk_gate(cfg)
    spec = _spec(cfg)
    probes = load_gkd_probe_trajectories(cfg.output_root, cfg.campaign_subdir)
    if probes.empty:
        raise CampaignGateError(
            f"no training_metrics.jsonl found under "
            f"{cfg.output_root / cfg.campaign_subdir} despite complete manifests"
        )
    floor = control_floor_check(
        probes,
        animals=spec.animals,
        betas=spec.betas,
        lrs=spec.a1_lora_lrs,
        max_floor=spec.control_floor_max,
        max_spread=spec.control_floor_max_spread,
    )
    if not floor.passed:
        raise CampaignGateError(
            "A2 control-floor gate FAILED — the control-free A1 screen is not "
            "trustworthy; rerun A1 with controls before dose-laddering. Reasons: "
            + "; ".join(floor.failures)
        )
    selections = select_top_recipes(
        probes, animals=spec.animals, betas=spec.betas, k=spec.a3_top_k
    )
    extensions, lr_warnings = lr_edge_extensions(
        selections, lrs=spec.a1_lora_lrs, factor=spec.a1_lr_extension_factor
    )
    for msg in lr_warnings:
        logger.warning("A1 lr-grid edge — {}", msg)
    selections = apply_lr_extensions(selections, extensions)
    candidates = {key: list(sel.recipes) for key, sel in selections.items()}
    candidates = _inject_reference_recipe(candidates, spec.a3_reference_recipe)
    if spec.a3_reference_recipe is not None:
        logger.info(
            "A3 reference arm {}@{:g} injected (deduped) into {} (animal, beta) cells",
            spec.a3_reference_recipe.mode_token,
            spec.a3_reference_recipe.lr,
            len(candidates),
        )
    selection_state = driver_io.selection_to_state(selections)
    driver_io.write_logs_report(
        cfg,
        spec.a1_selection_report,
        {
            "at": utc_now_iso(),
            "top_k": spec.a3_top_k,
            "selection": selection_state,
            "lr_edge_warnings": list(lr_warnings),
            "control_floor": floor.floors.to_dict(orient="records"),
            "a3_reference_recipe": (
                driver_io.recipe_to_dict(spec.a3_reference_recipe)
                if spec.a3_reference_recipe is not None
                else None
            ),
        },
    )
    try:
        train_cfgs = config_gen.a3_train_configs(candidates, spec=spec)
    except ValueError as err:
        raise CampaignGateError(
            f"A3 config generation rejected the A1 selection: {err}. The screen "
            "and its gates are intact — inspect "
            f"{cfg.logs_dir / spec.a1_selection_report} and either raise "
            "a3_max_same_arms or lower a3_top_k, then re-advance."
        ) from err
    verify_cfgs = config_gen.a3_verify_configs(train_cfgs, spec=spec)
    driver_io.write_generated_configs(cfg, {**train_cfgs, **verify_cfgs})
    headline = config_gen.a3_headline_train_names(train_cfgs)
    driver_io.submit_queue(
        cfg, spec.q_a3, driver_io.a3_jobs(train_cfgs, headline, spec=spec)
    )
    driver_io.submit_queue(
        cfg,
        spec.q_a3_evals,
        driver_io.a3_eval_jobs(sorted(verify_cfgs), spec=spec),
        depends_on=[spec.q_a3],
    )
    state["transitions"]["submit_a3"] = {
        "at": utc_now_iso(),
        "candidates": driver_io.candidates_to_state(candidates),
        "selection": selection_state,
        "lr_edge_warnings": list(lr_warnings),
        "control_floor": floor.floors.to_dict(orient="records"),
        "train_yamls": sorted(train_cfgs),
        "verify_yamls": sorted(verify_cfgs),
    }


# ---- transition 3: A3 (+evals) done -> freeze -> Stage B ---------------------------


def _a3_disk_gate(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    spec = _spec(cfg)
    tr = state["transitions"]["submit_a3"]
    headline = set(config_gen.a3_headline_train_names(dict.fromkeys(tr["train_yamls"])))
    verify_by_train = {
        fname.replace("gkd_a3_verify_", "gkd_a3_", 1): fname
        for fname in tr.get("verify_yamls", [])
    }
    for fname in tr["train_yamls"]:
        doc = yaml.safe_load((cfg.generated_dir / fname).read_text())
        leaves = expected.treatment_leaf_dirs(doc, cfg.output_root, spec=spec)
        _require_disk_complete(
            f"A3 {fname}",
            leaves,
            f"Requeue {spec.q_a3} failures, then re-advance.",
        )
        config_dirs = sorted({leaf.parent for leaf in leaves})
        verify_fname = verify_by_train.get(
            fname, fname.replace("gkd_a3_", "gkd_a3_verify_", 1)
        )
        verify_path = cfg.generated_dir / verify_fname
        require_bench = False
        if verify_path.is_file():
            require_bench = _verify_doc_requires_benchmarks(
                yaml.safe_load(verify_path.read_text())
            )
        _require_verify_complete(
            f"A3 {fname}",
            config_dirs,
            require_benchmarks=require_bench,
            remediation=(
                f"Requeue {spec.q_a3_evals} and re-run the driver "
                "(preference+FC+capability triad required)."
            ),
        )
        if fname in headline:
            controls = expected.control_leaf_dirs(doc, cfg.output_root, spec=spec)
            _require_disk_complete(
                f"A3 controls {fname}",
                controls,
                f"Requeue {spec.q_a3} failures, then re-advance.",
            )
            _require_verify_complete(
                f"A3 controls {fname}",
                sorted({leaf.parent for leaf in controls}),
                require_benchmarks=require_bench,
                remediation=(
                    f"Requeue {spec.q_a3_evals} and re-run the driver "
                    "(preference+FC+capability triad required)."
                ),
            )


def _submit_b(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    _a3_disk_gate(cfg, state)
    spec = _spec(cfg)
    candidates = driver_io.candidates_from_state(
        state["transitions"]["submit_a3"]["candidates"]
    )
    preference = load_gkd_preference_results(cfg.output_root, cfg.campaign_subdir)
    frozen = freeze_recipes(preference, candidates, doses=spec.a3_doses)
    beta = headline_beta(frozen)
    winner = frozen[beta]
    logger.info(
        "Recipe freeze: beta={} recipe={} lr={:g} epochs={:g} delta={:.4f}",
        beta,
        winner.recipe.mode_token,
        winner.recipe.lr,
        winner.num_epochs,
        winner.subliminal_delta,
    )
    driver_io.write_logs_report(
        cfg,
        spec.freeze_report,
        {
            "at": utc_now_iso(),
            "headline_beta": beta,
            "winner": {
                "recipe": driver_io.recipe_to_dict(winner.recipe),
                "num_epochs": winner.num_epochs,
                "subliminal_delta": winner.subliminal_delta,
                "per_animal_delta": winner.per_animal_delta,
            },
            "frozen": driver_io.frozen_to_state(frozen),
        },
    )
    train_cfgs = config_gen.b_train_configs(winner, beta, spec=spec)
    verify_cfgs = config_gen.b_verify_configs(train_cfgs, spec=spec)
    generated = {**train_cfgs, **verify_cfgs}
    c_train: dict[str, dict] = {}
    c_verify: dict[str, dict] = {}
    if cfg.include_stage_c:
        c_train = config_gen.c_train_configs(winner, beta, spec=spec)
        c_verify = config_gen.c_verify_configs(c_train, spec=spec)
        generated.update({**c_train, **c_verify})
    driver_io.write_generated_configs(cfg, generated)
    control_names = config_gen.b_control_train_names(train_cfgs)
    head_jobs, ext_jobs = driver_io.b_train_jobs(train_cfgs, control_names, spec=spec)
    driver_io.submit_queue(cfg, spec.q_b_headline, head_jobs)
    driver_io.submit_queue(cfg, spec.q_b_ext, ext_jobs)
    driver_io.submit_queue(
        cfg,
        spec.q_b_evals,
        driver_io.b_eval_jobs(verify_cfgs, train_cfgs, spec=spec),
        depends_on=[spec.q_b_headline, spec.q_b_ext],
    )
    if cfg.include_stage_c:
        driver_io.submit_queue(
            cfg, spec.q_c, driver_io.c_train_jobs(c_train, spec=spec)
        )
        driver_io.submit_queue(
            cfg,
            spec.q_c_evals,
            driver_io.c_eval_jobs(c_verify, c_train, spec=spec),
            depends_on=[spec.q_c],
        )
    state["transitions"]["submit_b"] = {
        "at": utc_now_iso(),
        "frozen": driver_io.frozen_to_state(frozen),
        "headline_beta": beta,
        "train_yamls": sorted(train_cfgs),
        "verify_yamls": sorted(verify_cfgs),
        "stage_c": sorted(c_train),
    }


# ---- transition 4: everything done -> finalize ------------------------------------


def _finalize(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    spec = _spec(cfg)
    tr = state["transitions"]["submit_b"]
    verify_by_train = {
        fname.replace("gkd_b_verify_", "gkd_b_", 1): fname
        for fname in tr.get("verify_yamls", [])
    }
    for fname in tr["train_yamls"] + tr.get("stage_c", []):
        doc = yaml.safe_load((cfg.generated_dir / fname).read_text())
        leaves = expected.treatment_leaf_dirs(doc, cfg.output_root, spec=spec)
        _require_disk_complete(
            f"Stage B/C {fname}",
            leaves,
            "Requeue the factorial queue failures, then re-advance.",
        )
        if fname.startswith("gkd_b_"):
            verify_fname = verify_by_train.get(
                fname, fname.replace("gkd_b_", "gkd_b_verify_", 1)
            )
        else:
            verify_fname = fname.replace("gkd_c_", "gkd_c_verify_", 1)
        verify_path = cfg.generated_dir / verify_fname
        require_bench = False
        if verify_path.is_file():
            require_bench = _verify_doc_requires_benchmarks(
                yaml.safe_load(verify_path.read_text())
            )
        _require_verify_complete(
            f"Stage B/C {fname}",
            sorted({leaf.parent for leaf in leaves}),
            require_benchmarks=require_bench,
            remediation=(
                f"Requeue {spec.q_b_evals} (and {spec.q_c_evals} if Stage C) "
                "and re-run the driver (preference+FC+capability triad required)."
            ),
        )
    control_docs = [
        yaml.safe_load((cfg.generated_dir / f).read_text())
        for f in tr["train_yamls"]
        if f.startswith("gkd_b_same_init_")
    ]
    controls: dict = {}
    control_require_bench = False
    for f in tr["train_yamls"]:
        if not f.startswith("gkd_b_same_init_"):
            continue
        verify_fname = verify_by_train.get(
            f, f.replace("gkd_b_", "gkd_b_verify_", 1)
        )
        verify_path = cfg.generated_dir / verify_fname
        if verify_path.is_file() and _verify_doc_requires_benchmarks(
            yaml.safe_load(verify_path.read_text())
        ):
            control_require_bench = True
            break
    for doc in control_docs:
        for leaf in expected.control_leaf_dirs(doc, cfg.output_root, spec=spec):
            controls.setdefault(leaf)
    _require_disk_complete(
        "Stage B controls",
        list(controls),
        "Requeue the same_init factorial jobs, then re-advance.",
    )
    _require_verify_complete(
        "Stage B controls",
        sorted({leaf.parent for leaf in controls}),
        require_benchmarks=control_require_bench,
        remediation=(
            f"Requeue {spec.q_b_evals} and re-run the driver "
            "(preference+FC+capability triad required)."
        ),
    )
    marker = cfg.logs_dir / spec.campaign_done_file
    if not cfg.dry_run:
        atomic_write_text(
            marker, f"campaign {spec.run_set_id} finished {utc_now_iso()}\n"
        )
        card = read_run_set_card(cfg.run_set_root)
        if card is not None:
            append_runs_index(
                cfg.output_root,
                runs_index_row(
                    run_set_id=card["run_set_id"],
                    run_type=card["run_type"],
                    purpose=card["purpose"],
                    date=card["date"],
                    git_sha=card.get("git_sha"),
                    status="done",
                ),
            )
            logger.info(
                "RUNS_INDEX status=done for run_set_id={}", spec.run_set_id
            )
        else:
            logger.warning(
                "No run_set.json at {} — RUNS_INDEX not updated on finalize",
                cfg.run_set_root,
            )
    state["transitions"]["finalize"] = {"at": utc_now_iso()}
    logger.info("Campaign {} COMPLETE — marker at {}", spec.run_set_id, marker)


# ---- the advance pass ---------------------------------------------------------------


def _auto_requeue_failed(cfg: driver_io.CampaignDriverConfig, name: str) -> None:
    """Requeue failed jobs under ``cfg.auto_requeue``; halt when any exceeds the cap."""
    if cfg.auto_requeue <= 0 or cfg.dry_run:
        return
    if not store.queue_exists(cfg.queue_root, name):
        return
    over: list[tuple[str, int]] = []
    under: list[tuple[str, int]] = []
    for job in store.load_jobs(cfg.queue_root, name):
        if store.job_state(cfg.queue_root, name, job) != store.FAILED:
            continue
        attempt = store.read_attempt(cfg.queue_root, name, job.jobkey)
        if attempt < cfg.auto_requeue:
            under.append((job.jobkey, attempt))
        else:
            over.append((job.jobkey, attempt))
    if over:
        jk, att = over[0]
        raise CampaignGateError(
            f"{name}: job {jk} failed after attempt={att} "
            f"(--auto-requeue {cfg.auto_requeue}); fix the cell or raise the cap, "
            f"then `uv run python -m scripts.jobq requeue --root {cfg.queue_root} "
            f"--name {name}`"
        )
    if not under:
        return
    for jk, att in under:
        logger.warning(
            "auto-requeue queue={} jobkey={} attempt={} (cap={})",
            name,
            jk,
            att,
            cfg.auto_requeue,
        )
    store.requeue(cfg.queue_root, name, failed_only=True)


def _queues_ready(
    cfg: driver_io.CampaignDriverConfig, names: list[str], waiting: list[str]
) -> bool:
    ready = True
    for name in names:
        _auto_requeue_failed(cfg, name)
        ok, msg = driver_io.queue_finished(cfg, name)
        if not ok:
            waiting.append(msg)
            ready = False
        if "FAILED" in msg:
            raise CampaignGateError(msg)
    return ready


def run(cfg: driver_io.CampaignDriverConfig) -> AdvanceResult:
    """One advance pass: fire every transition whose gates pass, then report.

    Returns:
        `AdvanceResult` — transitions fired, wait reasons, and campaign-done.

    Raises:
        CampaignGateError: A gate failed hard (failed jobs, missing artifacts,
            A2 flatness failure); the campaign must be fixed before advancing.
    """
    spec = _spec(cfg)
    state = driver_io.load_state(cfg)
    transitions = state["transitions"]
    fired: list[str] = []
    waiting: list[str] = []

    drift = _code_pin_gate(cfg)
    if drift is not None:
        fired.append(drift)

    if "submit_initial" not in transitions:
        _submit_initial(cfg, state)
        driver_io.save_state(cfg, state)
        fired.append("submit_initial")

    if "submit_initial" in transitions and "submit_a3" not in transitions:
        if _queues_ready(cfg, [spec.q_a1, spec.q_a2], waiting):
            _submit_a3(cfg, state)
            driver_io.save_state(cfg, state)
            fired.append("submit_a3")

    if "submit_a3" in transitions and "submit_b" not in transitions:
        if _queues_ready(cfg, [spec.q_a3, spec.q_a3_evals], waiting):
            _submit_b(cfg, state)
            driver_io.save_state(cfg, state)
            fired.append("submit_b")

    if "submit_b" in transitions and "finalize" not in transitions:
        final_queues = [spec.q_b_headline, spec.q_b_ext, spec.q_b_evals]
        if cfg.include_stage_c and transitions["submit_b"].get("stage_c"):
            final_queues += [spec.q_c, spec.q_c_evals]
        if _queues_ready(cfg, final_queues, waiting):
            _finalize(cfg, state)
            driver_io.save_state(cfg, state)
            fired.append("finalize")

    done = "finalize" in state["transitions"]
    return AdvanceResult(tuple(fired), tuple(waiting), done)


def status(cfg: driver_io.CampaignDriverConfig) -> dict:
    """Read-only snapshot: per-queue status + which transitions have fired."""
    state = driver_io.load_state(cfg)
    queues = {}
    for name in _spec(cfg).queue_priorities:
        if store.queue_exists(cfg.queue_root, name):
            queues[name] = store.queue_status(cfg.queue_root, name)
    return {"transitions": sorted(state["transitions"]), "queues": queues}
