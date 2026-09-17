"""ws160m lm-eval tail: benchmark every landed student and control after finalize.

One `plan.LMEVAL_STAGE` queue per partition, pinned to that partition's node and
submitted only once the partition wrote `CAMPAIGN_DONE` (and, by default, the
headline `results/` tables exist), so no headline artefact ever waits on lm-eval.
The jobs are ordinary eval-sweep shards whose verify YAMLs carry a ``benchmarks``
block: the sweep computes just the missing benchmark block per config dir and
caches each untuned base once in ``base_metrics.json``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import yaml
from loguru import logger

from pipelines.ws160m_campaign_pipeline import (
    CampaignGateError,
    node_pin_defaults,
)
from subliminal_learning.campaign import driver_io
from subliminal_learning.campaign import ws160m_config_gen as cgen
from subliminal_learning.campaign import ws160m_jobs as jobs
from subliminal_learning.campaign import ws160m_models as models
from subliminal_learning.campaign import ws160m_plan as plan
from subliminal_learning.jobqueue import store
from subliminal_learning.utils.io import atomic_write_json
from subliminal_learning.utils.run_io import utc_now_iso

LMEVAL_RECORD_JSON = "lmeval.json"


@dataclass(frozen=True)
class LmevalOptions:
    """Where the tail reads its gates from.

    Attributes:
        partition: Partition token.
        paths_config: Campaign paths YAML (`hf_cache_root` is the cold-1 cache the
            benchmark jobs must read Hub rungs from).
        require_results: Also require the headline `results/` tables under the
            `main` run set.
        node: Host the queue runs on (default: the partition's home node; any
            partition's node is valid since the YAMLs are read from the shared dir).
    """

    partition: str
    paths_config: Path = Path(plan.PATHS_CONFIG)
    require_results: bool = True
    node: str | None = None


def _results_ready(cfg: driver_io.CampaignDriverConfig) -> Path | None:
    """The headline leaves table, or None when the results have not been built."""
    main = plan.partition("main")
    path = (
        cfg.output_root
        / driver_io.run_set_subdir(cfg.spec.root_subdir, main.run_set_id)
        / "results"
        / plan.RESULTS_LEAVES_CSV
    )
    return path if path.is_file() else None


def _hub_cache(opts: LmevalOptions) -> Path:
    paths = yaml.safe_load(Path(opts.paths_config).read_text()) or {}
    root = paths.get("hf_cache_root")
    if not root:
        raise CampaignGateError(f"{opts.paths_config}: hf_cache_root is required")
    return Path(str(root)) / "hub"


def _queue_defaults(
    cfg: driver_io.CampaignDriverConfig, node: str, opts: LmevalOptions
) -> dict:
    """`node`'s pin as cwd/env plus the campaign Hub cache for lm-eval's loader."""
    defaults = node_pin_defaults(cfg, node)
    defaults["env"] = {**defaults["env"], "HF_HUB_CACHE": str(_hub_cache(opts))}
    return defaults


def run(cfg: driver_io.CampaignDriverConfig, opts: LmevalOptions) -> dict:
    """Write the benchmark verify YAMLs and submit the partition's lm-eval queue.

    Raises:
        CampaignGateError: The partition has not finalized, the headline results
            are missing, or the queue exists with a different job set.
    """
    p = plan.partition(opts.partition)
    if cfg.spec is not plan.spec_of(p.name):
        raise CampaignGateError(f"driver config spec is not partition {p.name!r}'s")
    done = cfg.logs_dir / plan.CAMPAIGN_DONE_FILE
    if not done.is_file():
        raise CampaignGateError(f"{p.name} has not finalized: {done} missing")
    if opts.require_results and _results_ready(cfg) is None:
        raise CampaignGateError(
            "headline results not built yet (scripts.build_ws160m_results)"
        )
    state = driver_io.load_state(cfg)
    wave = int(state.get("wave", models.WAVE_NOW))
    train = cgen.all_train_configs(p.name, wave=wave)
    lmeval = cgen.lmeval_verify_configs(p.name, train)
    shared = cfg.run_set_root / plan.SHARED_GENERATED_DIRNAME
    driver_io.write_generated_configs(cfg, lmeval)
    driver_io.write_generated_configs(
        replace(cfg, generated_dir_override=shared), lmeval
    )
    by_q = jobs.lmeval_jobs(p.name, lmeval, train, wave=wave, shared_config_root=shared)
    name = plan.queue(p, plan.LMEVAL_STAGE)
    queue_jobs = by_q.get(name, [])
    if not queue_jobs:
        raise CampaignGateError(f"{name} built zero jobs")
    deps = [
        q
        for q in plan.queue_priorities(p)
        if "evals" in q and store.queue_exists(cfg.queue_root, q)
    ]
    node = opts.node or p.node
    plan.partition_on(node)
    record = {
        "at": utc_now_iso(),
        "queue": name,
        "node": node,
        "n_jobs": len(queue_jobs),
        "depends_on": deps,
        "code_sha": cfg.code_sha,
        "tasks": list(plan.LMEVAL_TASKS),
    }
    if cfg.dry_run:
        logger.info("[dry-run] would submit {}", record)
        return record
    outcome = driver_io.submit_queue(
        cfg,
        name,
        queue_jobs,
        depends_on=deps or None,
        node=node,
        defaults=_queue_defaults(cfg, node, opts),
    )
    record["outcome"] = outcome
    atomic_write_json(cfg.logs_dir / LMEVAL_RECORD_JSON, record)
    logger.info("{} {}: {} jobs on {}", name, outcome, len(queue_jobs), node)
    return record
