"""jobq job builders for the off-policy cross-size campaign v2.

Sibling of `campaign.offxsize_jobs` (v1), not an edit of it. Two things differ, and both
are the reason v2 needs its own module:

- **one training job per adapter, not per row.** A base or rng control carries no trait,
  so the `dog` and `lion` rows of one control resolve to a single `train_key` and must be
  trained once and evaluated twice (`offxsize2_expected.TrainGroup`). Building jobs per
  eval row would train every control twice and then let the second copy overwrite the
  first in place;
- **number pools are ordinary jobq work.** v1 paused for an operator; v2 sizes each pool
  from its own measured probe keep rate and generates it on a card like everything else.

Every command line is self-describing (sizes, rung, recipe, seeds, absolute pool dir), so
editing a config after submit can never silently re-point an already-queued cell. `cwd`
and `env` come from the queue defaults the driver writes (the code pin), so a job carries
only `key`, `cmd`, `mem_mib` and `slots`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from subliminal_learning.campaign import offxsize2_expected as expected
from subliminal_learning.campaign import offxsize2_pairs as pairs
from subliminal_learning.campaign import offxsize2_plan as plan
from subliminal_learning.campaign import offxsize_numbers as numbers_gate
from subliminal_learning.jobqueue.model import Job

SWEEP_MODULE = "scripts.sweeps.sweep_offxsize2"
_BASE = f"uv run python -m {SWEEP_MODULE}"

# The rng floor is formatted by the generator rather than sampled from a model, so its
# `prefix` keep rate is 1.0 by construction and there is nothing for a probe to measure.
RNG_KEEP_RATE = 1.0

# vishesh's login on the campaign node is `angela` (uid 1001); Angela's own tree is
# `/home/angela2`. Every OTHER home on that shared box -- and the soar clone, which this
# campaign never runs from -- is out of bounds.
CAMPAIGN_CLONE = "/home/angela/eleutherAI/EleutherAI_subliminal_learning"
SOAR_CLONE = "/home/vishesh/eleutherAI/EleutherAI_subliminal_learning"

# Tokens that would mean this campaign is reading someone else's tree, an undeclared
# base, or another campaign's queue. Cheap to check, and the failures they prevent are
# silent: a cell that trains happily on the wrong artifact.
_FORBIDDEN: tuple[str, ...] = (
    SOAR_CLONE,
    "/home/angela2",
    "soar-subliminal/angela/",
    "weight-seed",
    "data-seed",
    "step0",
    "@step",
    "xfam-",
    "topk-",
    "ckptgrid",
)
# `pythia-160m-seed3` and friends: this campaign runs the default-seed bases only.
_SEEDED_REPO = re.compile(r"-seed\d")
# Any `/home/<user>/` that is not vishesh's login on the campaign node.
_FOREIGN_HOME = re.compile(r"/home/(?!angela/)[^/\s]+/")


def _assert_clean_cmd(cmd: str, key: str) -> None:
    """Raise when a command names a foreign tree, an undeclared base or another queue.

    Args:
        cmd: The command line (or any path baked into a job, e.g. its `cwd`).
        key: Job key, for the message.

    Raises:
        ValueError: The text names a forbidden tree, base or queue prefix.
    """
    lowered = cmd.lower()
    for token in _FORBIDDEN:
        if token.lower() in lowered:
            raise ValueError(f"job {key!r} command contains {token!r}: {cmd}")
    if match := _SEEDED_REPO.search(lowered):
        raise ValueError(f"job {key!r} command names {match.group()!r}: {cmd}")
    if match := _FOREIGN_HOME.search(lowered):
        raise ValueError(
            f"job {key!r} command reaches into {match.group()!r}, which is not "
            f"vishesh's tree on {plan.CAMPAIGN_NODE}: {cmd}"
        )


def _job(key: str, cmd: str, footprint: plan.Footprint) -> Job:
    _assert_clean_cmd(cmd, key)
    return Job(key=key, cmd=cmd, mem_mib=footprint.mem_mib, slots=footprint.slots)


def _wandb_flag(enabled: bool) -> str:
    return "" if enabled else " --no-wandb"


# ---- queues ------------------------------------------------------------------


@dataclass(frozen=True)
class QueueSubmission:
    """One queue as the driver submits it.

    Attributes:
        name: jobq queue name; must carry a priority in the frozen plan.
        jobs: The queue's jobs, already footprinted and checked.
        depends_on: Queues that must finish first.
        node: Host the queue is pinned to -- always `plan.CAMPAIGN_NODE`, because soar
            is running the cross-family campaign and shares this queue root.
    """

    name: str
    jobs: tuple[Job, ...]
    depends_on: tuple[str, ...] = ()
    node: str = plan.CAMPAIGN_NODE


def submission(
    name: str, jobs: Sequence[Job], *, depends_on: Sequence[str] = ()
) -> QueueSubmission:
    """Check one queue and pin it to the campaign node.

    Args:
        name: Queue name; must carry a priority in the frozen plan.
        jobs: Jobs to submit; non-empty, with distinct jobkeys.
        depends_on: Queues that must finish first.

    Returns:
        The pinned submission.

    Raises:
        ValueError: Unknown queue, empty queue, colliding jobkeys, or a job whose `cwd`
            is not vishesh's tree on the campaign node.
    """
    if name not in plan.QUEUE_PRIORITIES:
        raise ValueError(f"{name!r} is not a queue of {plan.RUN_SET_ID}")
    if not jobs:
        raise ValueError(f"refusing to submit empty queue {name!r}")
    keys = [job.jobkey for job in jobs]
    if len(set(keys)) != len(keys):
        duplicates = sorted({k for k in keys if keys.count(k) > 1})
        raise ValueError(f"queue {name!r} has colliding jobkeys: {duplicates}")
    for job in jobs:
        _assert_clean_cmd(job.cmd, job.key)
        if job.cwd is not None:
            _assert_clean_cmd(job.cwd, job.key)
    return QueueSubmission(name, tuple(jobs), tuple(depends_on))


# ---- P0: teachers, probes, pools, priors -------------------------------------


def teacher_jobs(
    *,
    animal: str = plan.PRIMARY_ANIMAL,
    sizes: Sequence[str] = plan.SIZES,
    rungs: Sequence[plan.TeacherRung] = plan.TEACHER_RUNGS,
    ftseed: int = plan.HEADLINE_TFS,
    queue_tag: str = "p0",
) -> list[Job]:
    """One template teacher per (size, rung) at the committed teacher seed.

    Args:
        animal: Trait the teachers are tuned for.
        sizes: Sizes to train; narrow it where a teacher is adopted rather than trained.
        rungs: Ladder rungs; the full ladder is a FACTOR in v2, not a gated pick.
        ftseed: Teacher fine-tuning seed.
        queue_tag: Job-key prefix; the R phase re-enumerates teachers under its own.

    Returns:
        One job per (size, rung).
    """
    return [
        _job(
            f"{queue_tag}|teacher|{animal}|{size}|{rung.key}|tfs{ftseed}",
            f"{_BASE} teacher-cell --animal {animal} --size {size} "
            f"--rung-tag {rung.key} --ftseed {ftseed} --no-wandb",
            plan.TEACHER_FOOTPRINT[size],
        )
        for size in sizes
        for rung in rungs
    ]


def teacher_eval_jobs(
    *,
    animal: str = plan.PRIMARY_ANIMAL,
    sizes: Sequence[str] = plan.SIZES,
    rung_tags: Sequence[str] | None = None,
    queue_tag: str = "p0",
) -> list[Job]:
    """One eval job per size, covering its rungs sequentially.

    Sequential by construction: every rung under a size shares one `base_control.json`
    at the parent, merged read-modify-write, so parallel jobs would drop each other's
    prompt sets.
    """
    tags = (
        tuple(rung_tags)
        if rung_tags is not None
        else tuple(rung.key for rung in plan.TEACHER_RUNGS)
    )
    return [
        _job(
            f"{queue_tag}|teval|{animal}|{size}",
            f"{_BASE} teacher-eval-cell --animal {animal} --size {size} "
            f"--rungs {','.join(tags)}",
            plan.TEACHER_FOOTPRINT[size],
        )
        for size in sizes
    ]


def probe_jobs(
    *,
    animal: str = plan.PRIMARY_ANIMAL,
    sizes: Sequence[str] = plan.SIZES,
    rungs: Sequence[plan.TeacherRung] = plan.TEACHER_RUNGS,
    ftseed: int = plan.HEADLINE_TFS,
    queue_tag: str = "p0",
) -> list[Job]:
    """One `PROBE_PROMPTS`-prompt format-health probe per teacher leaf.

    Its keep rate is what admission reads and what `pool_n_prompts` sizes the real pool
    from, so every teacher pool the campaign generates has one.
    """
    return [
        _job(
            f"{queue_tag}|probe|teacher|{animal}|{size}|{rung.key}|tfs{ftseed}",
            f"{_BASE} probe-cell --source teacher --animal {animal} --size {size} "
            f"--rung-tag {rung.key} --ftseed {ftseed}",
            plan.NUMGEN_FOOTPRINT[size],
        )
        for size in sizes
        for rung in rungs
    ]


def base_probe_jobs(*, sizes: Sequence[str] = plan.SIZES) -> list[Job]:
    """One format-health probe per untuned base.

    The base pools are the Delta denominators of the whole campaign, and an untuned base
    is the worst-formatted generator in it -- sizing those pools off an unmeasured keep
    rate is how a control comes back short after the card has already been paid for.
    """
    return [
        _job(
            f"p0|probe|base|{size}",
            f"{_BASE} probe-cell --source base --size {size}",
            plan.NUMGEN_FOOTPRINT[size],
        )
        for size in sizes
    ]


def pool_n_prompts(keep_rate: float | None, *, pool_key: str) -> int:
    """Prompt count for one pool, from the keep rate its format probe measured.

    v2 buys a low keep rate with more prompts instead of voiding the rung: a pool under
    `LOW_KEEP_THRESHOLD` skips the ladder's first rung outright, and the rung finally
    chosen is the smallest one projecting `PROBE_PROJECTED_ROWS_MIN` usable rows (the
    "safer side" rule, `offxsize_numbers.choose_n_prompts`).

    Args:
        keep_rate: Measured `prefix` keep rate for this pool.
        pool_key: `PoolSpec.key`, for the messages.

    Returns:
        A prompt count from `plan.N_PROMPTS_LADDER`.

    Raises:
        KeyError: No keep rate for this pool -- the gate never froze a probe for it, and
            a guessed prompt count either shorts the pool or burns a card.
        ValueError: Even the top of the ladder cannot project the row floor, so the pool
            must be voided by admission rather than generated.
    """
    if keep_rate is None:
        raise KeyError(
            f"no probe keep rate for pool {pool_key!r}; every pool a phase distils from "
            "needs a frozen format probe before its prompt count can be chosen"
        )
    ladder = plan.N_PROMPTS_LADDER
    if float(keep_rate) < plan.LOW_KEEP_THRESHOLD:
        ladder = ladder[1:]
    chosen = numbers_gate.choose_n_prompts(
        float(keep_rate),
        ladder=ladder,
        target_rows=plan.PROBE_PROJECTED_ROWS_MIN,
    )
    if chosen is None:
        raise ValueError(
            f"pool {pool_key!r} at keep {keep_rate:.4f} cannot project "
            f"{plan.PROBE_PROJECTED_ROWS_MIN} rows from {plan.N_PROMPTS_LADDER[-1]} "
            "prompts; the rung it belongs to is inadmissible, not under-generated"
        )
    return chosen


def pool_jobs(
    root: Path,
    leaves: Sequence[pairs.Leaf],
    *,
    keep_rates: Mapping[str, float],
) -> list[Job]:
    """One generation job per distinct pool `leaves` distil from.

    Args:
        root: Campaign root (absolute; baked into every command).
        leaves: Cells whose pools must exist before they can train.
        keep_rates: `PoolSpec.key` -> measured probe keep rate.

    Returns:
        One job per pool, in first-seen order.

    Raises:
        KeyError: A pool has no frozen probe (`pool_n_prompts`).
    """
    jobs: list[Job] = []
    for leaf in expected.pool_leaves(leaves):
        spec = expected.pool_spec(leaf)
        keep = keep_rates.get(spec.key)
        if keep is None and spec.kind == numbers_gate.KIND_RNG:
            keep = RNG_KEEP_RATE
        n_prompts = pool_n_prompts(keep, pool_key=spec.key)
        out_dir = expected.pool_dir(root, leaf)
        is_rng = spec.kind == numbers_gate.KIND_RNG
        is_teacher = spec.kind == numbers_gate.KIND_TEACHER
        parts = [f"{_BASE} pool-cell", f"--source {leaf.source}"]
        if is_teacher:
            parts.append(f"--animal {leaf.animal}")
        if not is_rng:
            parts.append(f"--size {leaf.teacher}")
        parts += [f"--n-prompts {n_prompts}", f"--out-dir {out_dir}"]
        if is_teacher:
            parts += [f"--rung-tag {leaf.rung_tag}", f"--ftseed {leaf.ftseed}"]
        # The rng floor loads no model, so it takes the lightest footprint on the table.
        size = plan.SIZES[0] if is_rng else leaf.teacher
        jobs.append(
            _job(
                f"p0|pool|{spec.key.replace('/', '|')}",
                " ".join(parts),
                plan.NUMGEN_FOOTPRINT[size],
            )
        )
    return jobs


def base_prior_jobs(
    *,
    sizes: Sequence[str] = plan.SIZES,
    animals: Sequence[str] = plan.ANIMALS,
) -> list[Job]:
    """Score each size's untuned base once, so student cells need no base control.

    One job per size covering both animals, never one per (size, animal): the two share
    one `base_control.json` under the size's priors dir, merged read-modify-write.
    """
    return [
        _job(
            f"p0|prior|{size}",
            f"{_BASE} base-prior-cell --size {size} --animals {','.join(animals)}",
            plan.TEACHER_FOOTPRINT[size],
        )
        for size in sizes
    ]


# ---- students ----------------------------------------------------------------


def student_jobs(
    root: Path,
    leaves: Sequence[pairs.Leaf],
    *,
    tags: Mapping[str, str],
    shas: Mapping[str, str] | None = None,
    queue_tag: str | None = None,
    eval_animals: Sequence[str] | None = None,
) -> list[Job]:
    """One training job per ADAPTER, carrying every animal that adapter is read for.

    The two animals' rows of a base or rng control share a `train_key` and collapse here
    into one job whose `--animals` names both: the checkpoint is trait-free, so it is
    trained once and preference-evaluated once per animal in the same cell.

    Args:
        root: Campaign root (absolute; baked into every command).
        leaves: Cells to cover, eval rows included.
        tags: Pool key -> honest-n dataset tag, from the frozen numbers gate.
        shas: Pool key -> `filtered_sha256`, so a cell refuses a pool that moved under it.
        queue_tag: Job-key prefix; defaults to each leaf's own phase. An elastic-tail
            tier passes its tier key, so a tier extending the factorial does not collide
            with the committed cell it would otherwise retrain.
        eval_animals: Animals every TREATMENT is preference-scored for, its own lineage
            included (`run-cell --eval-animals`); None scores the lineage animal only.
            The R phase passes both animals so each treatment also serves as the
            format-matched control of the other animal's cell.

    Returns:
        One job per training group, in first-seen order.

    Raises:
        KeyError: A pool has no frozen dataset tag (`expected.dataset_tag_of`).
    """
    jobs: list[Job] = []
    for group in expected.training_groups(leaves):
        leaf = group.leaf
        tag = expected.dataset_tag_of(leaf, tags)
        pool = expected.filtered_pool_dir(root, leaf)
        logs = any(
            plan.student_logs_wandb(
                phase=leaf.phase, animal=animal, ftseed=leaf.ftseed, gseed=leaf.gseed
            )
            for animal in group.animals
        )
        parts = [
            f"{_BASE} run-cell",
            f"--phase {leaf.phase}",
            f"--kind {leaf.kind}",
            f"--animals {','.join(group.animals)}",
            f"--teacher-size {leaf.teacher}",
            f"--student-size {leaf.student}",
            f"--recipe-key {leaf.recipe.key}",
            f"--gseed {leaf.gseed}",
            f"--pool-dir {pool}",
            f"--number-dataset-tag {tag}",
        ]
        if leaf.kind == pairs.TREATMENT:
            parts.append(f"--rung-tag {leaf.rung_tag}")
            parts.append(f"--ftseed {leaf.ftseed}")
            if eval_animals:
                parts.append(f"--eval-animals {','.join(eval_animals)}")
        sha = (shas or {}).get(leaf.pool_key)
        if sha:
            parts.append(f"--expect-sha {sha}")
        cmd = " ".join(parts) + _wandb_flag(logs)
        jobs.append(
            _job(
                _train_job_key(group, queue_tag=queue_tag or leaf.phase),
                cmd,
                plan.STUDENT_FOOTPRINT[leaf.student],
            )
        )
    return jobs


def concordance_jobs(
    root: Path,
    *,
    tags: Mapping[str, str],
    shas: Mapping[str, str] | None = None,
) -> list[Job]:
    """The P0 positive-control cell and its matched control.

    Angela's own rung AND recipe, which need not be on the committed grid: if this cell
    does not reproduce `CONCORDANCE_P_TARGET`, the port is broken and the campaign stops
    before it spends a card on the factorial.
    """
    return student_jobs(
        root, pairs.concordance_leaves(), tags=tags, shas=shas, queue_tag="p0"
    )


def tail_jobs(
    tier: str,
    root: Path,
    *,
    tags: Mapping[str, str],
    shas: Mapping[str, str] | None = None,
    rung_tag: str | None = None,
    students: Sequence[str] = plan.SIZES,
    recipes: Sequence[plan.StudentRecipe] = (),
    pair_list: Sequence[tuple[str, str]] = (),
) -> list[Job]:
    """Student jobs of one elastic-tail tier, keyed by the tier.

    Empty for the tiers that train no student on this lineage: `t5` is an eval-only
    benchmark backfill over parents that already exist (`capability_jobs(...,
    benchmarks=True)`) and `t6` carries its own numbers lineage and its own controls.

    Args:
        tier: A `plan.TAIL_TIERS` key.
        root: Campaign root.
        tags: Pool key -> dataset tag.
        shas: Pool key -> `filtered_sha256`.
        rung_tag: The winning rung (`t2`, `t3`, `t4`, `t7`).
        students: Columns to extend (`t2`, `t3`, `t7`).
        recipes: Caller-supplied recipes (`t4`, `t7`).
        pair_list: `(teacher, student)` pairs (`t4`).

    Returns:
        One job per training group of the tier, empty for an eval-only tier.

    Raises:
        ValueError: Unknown tier, or a tier called without the decision it needs
            (`pairs.tail_leaves`).
    """
    leaves = pairs.tail_leaves(
        tier,
        rung_tag=rung_tag,
        students=students,
        recipes=recipes,
        pairs=pair_list,
    )
    if not leaves:
        return []
    return student_jobs(root, leaves, tags=tags, shas=shas, queue_tag=tier)


# ---- capability --------------------------------------------------------------


def capability_jobs(
    root: Path,
    leaves: Sequence[pairs.Leaf],
    *,
    tags: Mapping[str, str],
    benchmarks: bool = plan.RUN_BENCHMARKS_IN_CORE,
    queue_tag: str = "eval",
) -> list[Job]:
    """One capability job per `{student_tag}` parent covering `leaves`.

    Per parent, not per seed: the forced-choice and capability summaries are written at
    the parent and merged, so seed-level jobs would race on them. `benchmarks` defaults
    to `RUN_BENCHMARKS_IN_CORE` (False) -- committed work measures forced choice and Pile
    perplexity only, and lm-eval is tail tier `t5`.
    """
    jobs: list[Job] = []
    for parent, size, animal in expected.capability_parents(root, leaves, tags):
        flag = " --benchmarks" if benchmarks else ""
        cmd = (
            f"{_BASE} eval-cell --param-dir {parent} --student-size {size} "
            f"--animal {animal}{flag}"
        )
        jobs.append(
            _job(
                f"{queue_tag}|{_parent_key(root, parent)}",
                cmd,
                plan.CAPABILITY_FOOTPRINT[size],
            )
        )
    return jobs


def warm_eval_base_jobs(
    root: Path,
    leaves: Sequence[pairs.Leaf],
    *,
    tags: Mapping[str, str],
    benchmarks: bool = plan.RUN_BENCHMARKS_IN_CORE,
    queue_tag: str = "warm",
) -> list[Job]:
    """Warm each capability tree's shared base cache once, before its cells fan out.

    Args:
        root: Campaign root.
        leaves: Cells the gated eval queue covers.
        tags: Pool key -> dataset tag.
        benchmarks: Also warm the base lm-eval block. Must mirror the `benchmarks` of
            the eval queue this gates -- `base_metrics_ready` demands every stage the
            cell asks for, so a benchmark queue warmed without it never starts.
        queue_tag: Job-key prefix, so a benchmark warm does not collide with the
            forced-choice/perplexity warm over the same trees.
    """
    seen: dict[str, tuple[Path, str, str]] = {}
    for parent, size, animal in expected.capability_parents(root, leaves, tags):
        seen.setdefault(str(parent.parent), (parent, size, animal))
    flag = " --benchmarks" if benchmarks else ""
    return [
        _job(
            f"{queue_tag}|{_parent_key(root, parent.parent)}",
            f"{_BASE} warm-eval-base --param-dir {parent} --student-size {size}{flag}",
            plan.CAPABILITY_FOOTPRINT[size],
        )
        for parent, size, _animal in seen.values()
    ]


# ---- keys --------------------------------------------------------------------


def train_job_identity(group: expected.TrainGroup) -> str:
    """The adapter-naming tail of a training jobkey, with no queue tag.

    The driver matches recorded holes back to cells with this: a hole carries the jobkey
    that failed, whose tag says which queue it failed in, while the adapter identity is
    what a later phase has to skip.
    """
    return _train_job_key(group, queue_tag="").removeprefix("|")


def _train_job_key(group: expected.TrainGroup, *, queue_tag: str) -> str:
    """Key naming the ADAPTER this job trains, plus the rows it serves.

    Keyed on `train_key`, never on `Leaf.key`: a control shared by two animals is one
    job, and its key says so. The queue tag leads, so the same adapter re-enumerated by
    a later phase or an elastic-tail tier gets its own key instead of silently reading as
    "already submitted".
    """
    leaf = group.leaf
    return "|".join(
        [
            queue_tag,
            "-".join(group.animals),
            leaf.kind,
            f"T{leaf.teacher}",
            f"S{leaf.student}",
            leaf.recipe.key,
            leaf.rung_tag or "norung",
            "tfsNA" if leaf.ftseed is None else f"tfs{leaf.ftseed}",
            f"g{leaf.gseed}",
        ]
    )


def _parent_key(root: Path, parent: Path) -> str:
    """Key fragment naming one param dir uniquely.

    The dataset tag and student tag repeat across teacher lineages, so the key has to
    carry the whole path below the campaign root or two different cells collide onto one
    job and only one of them is ever evaluated.
    """
    try:
        rel = parent.relative_to(root)
    except ValueError:
        rel = parent
    return "|".join(rel.parts)
