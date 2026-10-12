"""Jobs, queues and host dealing of the S2 core campaign (pure).

Every job is one command of the cell CLI `scripts.sweeps.sweep_s2core` whose flags fully
identify its cell. Queues are named ``s2core-<channel>-<stage>-<kind>-<hosttag>``; a
queue's job set is immutable (`driver_io.submit_queue`), so every stage and kind gets its
own queue. Cells of a stage are dealt round-robin over each channel host's GPUs in plan
order (plan §16.1), and a cell's eval jobs sit on the host that trained it, because an
eval queue depends on its host's train queue.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from types import MappingProxyType

from subliminal_learning.campaign import s2core_plan as plan
from subliminal_learning.campaign.s2core_layout import HardCell, SoftCell, hard_steps
from subliminal_learning.jobqueue.model import Job

SWEEP_MODULE = "scripts.sweeps.sweep_s2core"
_BASE = f"uv run python -m {SWEEP_MODULE}"
_FORBIDDEN: tuple[str, ...] = ("angela", "--force")
_FORBIDDEN_CLUSTER5: tuple[str, ...] = ("/mnt/ssd-3", "/mnt/cold-1")

# ---- stages and kinds ----------------------------------------------------------------

P0 = "p0"
STAGE_A = "a"
STAGE_B = "b"
STAGE_C = "c"
STAGES: tuple[str, ...] = (STAGE_A, STAGE_B, STAGE_C)

TRAIN = "train"
SELECT = "select"
FULLPRIOR = "fullprior"
FULL = "full"
STAGE_KINDS: tuple[str, ...] = (TRAIN, SELECT, FULLPRIOR, FULL)

PREP = "prep"
BASE = "base"
FINAL = "final"
LEXICON = "lexicon"
LEAK = "leak"
SOFT_P0_SOAR_KINDS: tuple[str, ...] = (
    PREP,
    FINAL,
    LEXICON,
    *(f"{LEAK}_{c}" for c in plan.CARRIERS),
    *(f"assemble_{c}" for c in plan.CARRIERS),
)

GEN42 = "gen42"
BUILD42 = "build42"
GEN43 = "gen43"
BUILD4243 = "build4243"
EQ = "eq"
GEN_KIND: MappingProxyType[int, str] = MappingProxyType({42: GEN42, 43: GEN43})
# Generation runs where `plan.GEN_PLACEMENT` puts it; build-numbers and EQ on cluster-5.
HARD_P0_C5_KINDS: tuple[str, ...] = (BUILD42, BUILD4243, EQ)

_PRIORITY: MappingProxyType[str, int] = MappingProxyType(
    {
        "p0": 100,
        GEN42: 100,
        GEN43: 99,
        EQ: 50,
        BASE: 95,
        SELECT: 90,
        TRAIN: 80,
        FULLPRIOR: 70,
        FULL: 70,
    }
)


def queue_name(channel: str, stage: str, kind: str, host: str) -> str:
    """``s2core-<channel>-<stage>-<kind>-<hosttag>``."""
    if channel not in plan.CHANNELS:
        raise ValueError(f"unknown channel {channel!r}")
    return f"{plan.CAMPAIGN}-{channel}-{stage}-{kind}-{plan.HOST_TAG[host]}"


def _queue_table() -> dict[str, dict[str, tuple[str, int]]]:
    """channel -> queue name -> (host, priority), for every queue a driver can submit."""
    soft: dict[str, tuple[str, int]] = {}
    for kind in SOFT_P0_SOAR_KINDS:
        soft[queue_name(plan.SOFT, P0, kind, plan.SOAR)] = (plan.SOAR, _PRIORITY["p0"])
    for host in plan.CHANNEL_HOSTS[plan.SOFT]:
        soft[queue_name(plan.SOFT, P0, BASE, host)] = (host, _PRIORITY[BASE])
    hard: dict[str, tuple[str, int]] = {}
    for kind in HARD_P0_C5_KINDS:
        prio = _PRIORITY[EQ] if kind == EQ else _PRIORITY["p0"]
        hard[queue_name(plan.HARD, P0, kind, plan.CLUSTER5)] = (plan.CLUSTER5, prio)
    for seed, kind in GEN_KIND.items():
        for host in dict.fromkeys(plan.GEN_PLACEMENT[seed]):
            hard[queue_name(plan.HARD, P0, kind, host)] = (host, _PRIORITY[kind])
    for channel, table in ((plan.SOFT, soft), (plan.HARD, hard)):
        for stage in STAGES:
            for kind in STAGE_KINDS:
                if stage != STAGE_C and kind in (FULLPRIOR, FULL):
                    continue
                for host in plan.CHANNEL_HOSTS[channel]:
                    table[queue_name(channel, stage, kind, host)] = (
                        host,
                        _PRIORITY[kind],
                    )
    return {plan.SOFT: soft, plan.HARD: hard}


_TABLE = _queue_table()
QUEUE_PRIORITIES: MappingProxyType[str, MappingProxyType[str, int]] = MappingProxyType(
    {
        ch: MappingProxyType({n: prio for n, (_, prio) in table.items()})
        for ch, table in _TABLE.items()
    }
)
QUEUE_HOST: MappingProxyType[str, str] = MappingProxyType(
    {n: host for table in _TABLE.values() for n, (host, _) in table.items()}
)


def queue_host(name: str) -> str:
    """The host a campaign queue is pinned to.

    Raises:
        KeyError: `name` is not a campaign queue.
    """
    return QUEUE_HOST[name]


def pool_queues(host: str) -> tuple[str, ...]:
    """Every queue pinned to `host`, by priority (a pool's ``--queues`` list)."""
    channel = plan.host_channel(host)
    names = [n for n in QUEUE_PRIORITIES[channel] if QUEUE_HOST[n] == host]
    return tuple(sorted(names, key=lambda n: (-QUEUE_PRIORITIES[channel][n], n)))


# ---- dealing -------------------------------------------------------------------------


def gpu_slots(channel: str) -> tuple[str, ...]:
    """Hosts in dealing order: GPU index 0 of every host in plan order, then 1, ..."""
    hosts = plan.CHANNEL_HOSTS[channel]
    width = max(len(plan.HOST_GPUS[h]) for h in hosts)
    return tuple(h for i in range(width) for h in hosts if i < len(plan.HOST_GPUS[h]))


def deal(channel: str, items: Sequence) -> list[tuple[object, str]]:
    """``(item, host)`` pairs: item ``i`` goes to ``gpu_slots(channel)[i % n]``."""
    slots = gpu_slots(channel)
    return [(item, slots[i % len(slots)]) for i, item in enumerate(items)]


# ---- cell command lines --------------------------------------------------------------


def _lr(lr: float) -> str:
    text = f"{lr:g}"
    if float(text) != lr:
        raise ValueError(f"lr {lr!r} does not round-trip through {text!r}")
    return text


def soft_cell_args(cell: SoftCell) -> str:
    return (
        f"--arm {cell.arm} --carrier {cell.carrier} --lr {_lr(cell.lr)} "
        f"--budget {cell.budget} --gseed {cell.gseed}"
    )


def hard_cell_args(cell: HardCell) -> str:
    return (
        f"--arm {cell.arm} --data {cell.data} --lr {_lr(cell.lr)} "
        f"--steps {cell.steps} --seed {cell.seed}"
    )


def _job(key: str, cmd: str, footprint: str, host: str) -> Job:
    lowered = cmd.lower()
    forbidden = _FORBIDDEN + (_FORBIDDEN_CLUSTER5 if host == plan.CLUSTER5 else ())
    for token in forbidden:
        if token in lowered:
            raise ValueError(f"job {key!r} command contains {token!r}: {cmd}")
    fp = plan.FOOTPRINTS[footprint]
    return Job(key=key, cmd=cmd, mem_mib=fp.mem_mib, slots=fp.slots)


def train_job(cell: SoftCell | HardCell, stage: str, host: str) -> Job:
    if isinstance(cell, SoftCell):
        return _job(
            f"{stage}|train|{cell.key}",
            f"{_BASE} train-soft {soft_cell_args(cell)}",
            "soft_train",
            host,
        )
    return _job(
        f"{stage}|train|{cell.key}",
        f"{_BASE} train-hard {hard_cell_args(cell)}",
        "hard_train",
        host,
    )


def eval_job(cell: SoftCell | HardCell, stage: str, level: str, host: str) -> Job:
    if level not in plan.EVAL_LEVELS:
        raise ValueError(f"unknown eval level {level!r}")
    if isinstance(cell, SoftCell):
        cmd = f"{_BASE} eval-soft {soft_cell_args(cell)} --level {level}"
        footprint = "soft_eval"
    else:
        cmd = f"{_BASE} eval-hard {hard_cell_args(cell)} --level {level}"
        footprint = "hard_eval_select" if level == plan.SELECT else "hard_eval_full"
    return _job(f"{stage}|eval-{level}|{cell.key}", cmd, footprint, host)


@dataclass(frozen=True)
class QueuePlan:
    """One queue to submit: its jobs, host and same-campaign dependencies."""

    name: str
    host: str
    jobs: tuple[Job, ...]
    depends_on: tuple[str, ...] = ()


def _group(
    channel: str,
    stage: str,
    kind: str,
    pairs: Sequence[tuple[Job, str]],
    deps_kind: str | None = None,
) -> list[QueuePlan]:
    by_host: dict[str, list[Job]] = {}
    for job, host in pairs:
        by_host.setdefault(host, []).append(job)
    out: list[QueuePlan] = []
    for host in plan.CHANNEL_HOSTS[channel]:
        if host not in by_host:
            continue
        deps = (queue_name(channel, stage, deps_kind, host),) if deps_kind else ()
        out.append(
            QueuePlan(
                queue_name(channel, stage, kind, host), host, tuple(by_host[host]), deps
            )
        )
    return out


def stage_train_queues(
    channel: str, stage: str, cells: Sequence[SoftCell | HardCell]
) -> list[QueuePlan]:
    """Train queues of a stage (one per host that got cells)."""
    pairs = [(train_job(c, stage, h), h) for c, h in deal(channel, cells)]
    return _group(channel, stage, TRAIN, pairs)


def stage_select_queues(
    channel: str, stage: str, cells: Sequence[SoftCell | HardCell]
) -> list[QueuePlan]:
    """`select` eval queues of a stage; each depends on its host's train queue."""
    pairs = [(eval_job(c, stage, plan.SELECT, h), h) for c, h in deal(channel, cells)]
    return _group(channel, stage, SELECT, pairs, deps_kind=TRAIN)


def stage_full_queues(
    channel: str,
    trained_c: Sequence[SoftCell | HardCell],
    full_cells: Sequence[SoftCell | HardCell],
) -> list[QueuePlan]:
    """Stage-C `full` eval queues.

    Cells trained in stage C keep their stage-C host and wait for its `select` queue;
    final-setting cells trained in stages A / B (`fullprior`) are dealt afresh and wait
    for nothing.
    """
    host_of = {c: h for c, h in deal(channel, trained_c)}
    in_c = [c for c in full_cells if c in host_of]
    prior = [c for c in full_cells if c not in host_of]
    pairs_c = [(eval_job(c, STAGE_C, plan.FULL, host_of[c]), host_of[c]) for c in in_c]
    pairs_p = [(eval_job(c, STAGE_C, plan.FULL, h), h) for c, h in deal(channel, prior)]
    return _group(channel, STAGE_C, FULL, pairs_c, deps_kind=SELECT) + _group(
        channel, STAGE_C, FULLPRIOR, pairs_p
    )


# ---- phase-0 queues ------------------------------------------------------------------


def _soar_queue(kind: str, jobs: Sequence[Job], deps=()) -> QueuePlan:
    return QueuePlan(
        queue_name(plan.SOFT, P0, kind, plan.SOAR), plan.SOAR, tuple(jobs), tuple(deps)
    )


def soft_prep_queue() -> QueuePlan:
    return _soar_queue(
        PREP, [_job("p0|prep", f"{_BASE} phase0-prep", "cpu", plan.SOAR)]
    )


def soft_base_queues() -> list[QueuePlan]:
    """`full` evals of the 5 roster models, dealt over the soft hosts; wait for prep."""
    pairs = [
        (
            _job(
                f"p0|eval-full|base:{label}",
                f"{_BASE} eval-base --label {label} --level {plan.FULL}",
                "base_eval",
                host,
            ),
            host,
        )
        for label, host in deal(plan.SOFT, plan.BASE_MODELS)
    ]
    out = []
    for q in _group(plan.SOFT, P0, BASE, pairs):
        out.append(QueuePlan(q.name, q.host, q.jobs, (soft_prep_queue().name,)))
    return out


def soft_final_queue() -> QueuePlan:
    """`phase0-finalize` (collapse cut, E10 directions) after every base eval queue."""
    deps = tuple(q.name for q in soft_base_queues())
    return _soar_queue(
        FINAL,
        [_job("p0|finalize", f"{_BASE} phase0-finalize", "cpu", plan.SOAR)],
        deps,
    )


def soft_lexicon_queue() -> QueuePlan:
    """Phase 0: `pool-lexicon` (candidates + lexicon v2, no judge) for every carrier."""
    return _soar_queue(
        LEXICON,
        [
            _job(
                f"p0|lexicon|{c}",
                f"{_BASE} pool-lexicon --carrier {c}",
                "cpu",
                plan.SOAR,
            )
            for c in plan.CARRIERS
        ],
    )


def soft_leak_queue(carrier: str) -> QueuePlan:
    """After the stack choice: `pool-leak-item` (the carrier's ai_leak work item)."""
    return _soar_queue(
        f"{LEAK}_{carrier}",
        [
            _job(
                f"p0|leak|{carrier}",
                f"{_BASE} pool-leak-item --carrier {carrier}",
                "cpu",
                plan.SOAR,
            )
        ],
    )


def soft_assemble_queue(carrier: str) -> QueuePlan:
    return _soar_queue(
        f"assemble_{carrier}",
        [
            _job(
                f"p0|assemble|{carrier}",
                f"{_BASE} pool-assemble --carrier {carrier}",
                "cpu",
                plan.SOAR,
            )
        ],
    )


def _c5_queue(kind: str, jobs: Sequence[Job], deps=()) -> QueuePlan:
    return QueuePlan(
        queue_name(plan.HARD, P0, kind, plan.CLUSTER5),
        plan.CLUSTER5,
        tuple(jobs),
        tuple(deps),
    )


def hard_gen_queues() -> list[QueuePlan]:
    """`gen-numbers` for the 5 number sources at both prompt seeds (10 whole-card jobs).

    Submitted together at hard phase 0 on the hosts of `plan.GEN_PLACEMENT` (every
    seed-42 job on cluster-5, so the pools that are always needed never wait on the
    shared node; seed-42 queues have the higher priority). The seed-43 pools are read
    only if the seed-42 build asks for the extension (plan §9.3).
    """
    by_queue: dict[tuple[int, str], list[Job]] = {}
    for seed in plan.PROMPT_SEEDS:
        hosts = plan.GEN_PLACEMENT[seed]
        if len(hosts) != len(plan.NUMBER_SOURCES):
            raise ValueError(f"GEN_PLACEMENT[{seed}] does not match NUMBER_SOURCES")
        for source, host in zip(plan.NUMBER_SOURCES, hosts, strict=True):
            by_queue.setdefault((seed, host), []).append(
                _job(
                    f"p0|gen|{source.key}|prompts{seed}",
                    f"{_BASE} gen-numbers --source {source.key} --prompt-seed {seed}",
                    "numbers_gen",
                    host,
                )
            )
    return [
        QueuePlan(queue_name(plan.HARD, P0, GEN_KIND[seed], host), host, tuple(js))
        for (seed, host), js in by_queue.items()
    ]


def hard_gen_queue_names(prompt_seed: int) -> list[str]:
    """Generation queues of one prompt seed."""
    kind = GEN_KIND[prompt_seed]
    return [q.name for q in hard_gen_queues() if f"-{P0}-{kind}-" in q.name]


def hard_build_queue(prompt_seeds: Sequence[int]) -> QueuePlan:
    """`build-numbers` over `prompt_seeds` (cluster-5), after the generation queues of
    the newest seed (the seed-42 build has already run before the seed-42+43 one)."""
    seeds = tuple(prompt_seeds)
    kind = {(42,): BUILD42, (42, 43): BUILD4243}[seeds]
    deps = tuple(hard_gen_queue_names(seeds[-1]))
    arg = ",".join(str(s) for s in seeds)
    return _c5_queue(
        kind,
        [
            _job(
                f"p0|build|prompts{arg.replace(',', '-')}",
                f"{_BASE} build-numbers --prompt-seeds {arg}",
                "cpu",
                plan.CLUSTER5,
            )
        ],
        deps,
    )


def hard_eq_queue() -> QueuePlan:
    """EQ-soft (3 bf16 1.4b cells) and EQ-hard (6 bf16 160m cells), cluster-5."""
    jobs = [
        _job(
            f"p0|eq-soft|g{s}", f"{_BASE} eq-soft --seed {s}", "eq_soft", plan.CLUSTER5
        )
        for s in plan.EQ_SOFT_SEEDS
    ]
    jobs += [
        _job(
            f"p0|eq-hard|{src}|s{s}",
            f"{_BASE} eq-hard --source {src} --seed {s}",
            "eq_hard",
            plan.CLUSTER5,
        )
        for src in plan.EQ_HARD_SOURCES
        for s in plan.EQ_HARD_SEEDS
    ]
    return _c5_queue(EQ, jobs)


def hard_steps_allowed() -> frozenset[int]:
    """Every step count a hard cell may carry (whole-epoch 10k-row schedules)."""
    return frozenset(
        hard_steps(e) for e in (plan.HARD_BASE_EPOCHS, *plan.HARD_STAGE_B_EPOCHS)
    )


def validate_queue_plans(plans: Sequence[QueuePlan]) -> None:
    """Every queue is a known campaign queue on its host, with unique job keys.

    Raises:
        ValueError: An unknown queue, a host mismatch or a duplicate key.
    """
    keys: list[str] = []
    for q in plans:
        if QUEUE_HOST.get(q.name) != q.host:
            raise ValueError(f"queue {q.name} is not a campaign queue on {q.host}")
        for dep in q.depends_on:
            if dep not in QUEUE_HOST:
                raise ValueError(f"{q.name} depends on unknown queue {dep}")
        keys.extend(j.jobkey for j in q.jobs)
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate job keys across queue plans")
