"""CLI: enumerate every Pile-coherence eval target and list/submit them to jobq.

Discovers, glob-based and layout-tolerant, every trained adapter under the teachers
and students output trees plus the distinct base variants they reference, decides which
are still PENDING against the fixed output-artifact contract (see below), and either
prints a summary table, lists the pending targets, or submits them into a jobq queue.
It NEVER runs GPU work itself — it only enumerates, lists, and submits.

Output-artifact contract used for resume detection (implemented by the eval CLI /
pipeline, not here):

  - adapter run: ``{ftseed_dir}/pile_coherence/{shard_version}.json``
    (+ ``{shard_version}_per_token_loss.npy`` sibling; skip the npy check with
    ``--skip-npy-check``).
  - base model:  ``{output_root}/coherence/{shard_version}/base/{model_dir_name}/ce.json``
    where ``model_dir_name`` is the HF repo BASENAME (``@{revision}`` appended when set),
    e.g. ``pythia-160m`` / ``pythia-160m-data-seed1`` — NOT the ``--model`` token. The
    token (``original`` / ``data-seed1``) is still what the CLI arg and job keys use.

A target is DONE iff its json exists AND (adapters only, unless ``--skip-npy-check``)
its npy sibling exists.

The eval CLI invoked per job is ``scripts.evaluate_pile_coherence`` — ``--model <token>``
alone for a base variant, plus ``--adapter-dir <.../checkpoints>`` for an adapter.

Submission ordering: base-variant CE must be cached BEFORE any adapter's ΔCE runs
(the adapter eval RuntimeErrors otherwise). Base and adapter jobs therefore go into two
queues — ``{queue}-base`` (higher priority) and ``{queue}`` which ``depends_on`` it —
so every base finishes before any adapter starts.
"""

from __future__ import annotations

import fnmatch
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import typer
from dotenv import load_dotenv
from loguru import logger

from subliminal_learning.config import load_config
from subliminal_learning.coherence.config import load_pile_coherence_config
from subliminal_learning.jobqueue import store
from subliminal_learning.jobqueue.model import Job
from subliminal_learning.models.registry import model_ref
from subliminal_learning.teachers.model_specs import parse_base_models
from subliminal_learning.teachers.run_layout import CKPT_SUBDIR, model_dir_name
from subliminal_learning.utils.io import rglob_follow
from subliminal_learning.utils.logging_setup import configure_loguru

# Absolute repo cwd every job must cd into first: jobs on shared queue roots have been
# lost to cwd leaks (drained against another node's paths root), so never rely on the
# worker's inherited cwd.
REPO_CWD = "/home/angela/EleutherAI_subliminal_learning"

_BASE_PREFIX = "pythia-160m"


# --------------------------- target model ---------------------------


@dataclass(frozen=True)
class AdapterTarget:
    """One trained ftseed adapter to score (teacher or student)."""

    tree: str  # "teachers" | "students"
    kind: str  # "teacher" | "treatment" | "base-control" | "rng-control"
    ftseed_dir: Path  # the ftseed{S} run dir (parent of checkpoints)
    ftseed: int
    token: str  # --model base token derived from the manifest's base_model_repo
    base_repo: str  # base_model_repo string (repo id, minus any @revision)
    animal: str | None
    student_base: str | None
    teacher_base: str | None
    lineage: str | None  # "teacher-{base}/{ttag}_tseed{T}" (treatment only)
    dataset_tag: str | None
    param_tag: str  # teacher tag or student stag


@dataclass(frozen=True)
class BaseTarget:
    """One bare base variant to score (its CE is the ΔCE baseline).

    ``token`` is the ``--model`` CLI arg / job-key id; ``model_dir_name`` is the HF repo
    basename (``@rev`` appended when set) that keys the base artifact dir.
    """

    token: str
    model_dir_name: str


# --------------------------- token mapping ---------------------------


def token_for_repo(base_model_repo: str, hub_org: str) -> tuple[str | None, str]:
    """Map a manifest ``base_model_repo`` to its ``--model`` token.

    Returns ``(token, repo_no_rev)``; ``token`` is None when the repo basename cannot
    be tokenized (the caller skips it with a warning rather than guessing). Any
    ``@revision`` suffix is preserved on the token and stripped from ``repo_no_rev``.
    """
    repo, _, rev = base_model_repo.partition("@")
    basename = repo.rsplit("/", 1)[-1]
    if basename == _BASE_PREFIX:
        token = "original"
    elif basename.startswith(_BASE_PREFIX + "-"):
        token = basename[len(_BASE_PREFIX) + 1 :]  # "data-seed1" / "weight-seed3"
        if token.startswith("seed"):
            token = f"full-{token}"  # fully re-randomized sibling: "seed2" -> "full-seed2"
    else:
        return None, repo
    full_token = f"{token}@{rev}" if rev else token
    # Round-trip through the real parser + resolver to reject anything that would not
    # reproduce this repo (never guess a token the eval CLI would reject).
    # allow_full_seeds: this is a dir/manifest inverter over runs that already exist,
    # so the launch-time full-seed gate does not apply.
    try:
        specs = parse_base_models(full_token, allow_full_seeds=True)
    except ValueError:
        return None, repo
    if len(specs) != 1:
        return None, repo
    if model_ref(specs[0], hub_org).split("@")[0].rsplit("/", 1)[-1] != basename:
        return None, repo
    return full_token, repo


# --------------------------- discovery ---------------------------


def _parse_ftseed(name: str) -> int | None:
    tail = name.removeprefix("ftseed")
    return int(tail) if tail.isdigit() else None


def _read_base_repo(ftseed_dir: Path) -> str | None:
    """The ``base_model_repo`` recorded in the run's ``train_manifest.json`` (or None)."""
    p = ftseed_dir / "train_manifest.json"
    if not p.is_file():
        return None
    try:
        m = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    repo = m.get("base_model_repo")
    return str(repo) if repo else None


def _strip_ckpt(parts: tuple[str, ...]) -> tuple[str, ...]:
    """Drop the gathered ``ckpt/`` model level (revision-pinned runs) so the fixed-index
    parsing below sees the plain shape; the model name keeps its ``@rev`` suffix. ``ckpt``
    is a reserved sentinel (never an animal/param_tag/dataset), so removing it is unambiguous.
    """
    return tuple(p for p in parts if p != CKPT_SUBDIR)


def _parse_teacher(rel: Path) -> dict | None:
    """Parse ``{animal}/{model}/{tag...}/ftseed{S}`` -> teacher fields (or None)."""
    parts = _strip_ckpt(rel.parts)
    if len(parts) < 3:
        return None
    ftseed = _parse_ftseed(parts[-1])
    if ftseed is None:
        return None
    return {
        "kind": "teacher",
        "ftseed": ftseed,
        "animal": parts[0],
        "student_base": None,
        "teacher_base": None,
        "lineage": None,
        "dataset_tag": None,
        "param_tag": "/".join(parts[2:-1]),
    }


def _parse_student(rel: Path) -> dict | None:
    """Parse a students-tree run dir (treatment or control) -> fields (or None)."""
    parts = _strip_ckpt(rel.parts)
    ftseed = _parse_ftseed(parts[-1])
    if ftseed is None:
        return None
    if parts and parts[0] == "controls":
        if len(parts) >= 2 and parts[1] == "base" and len(parts) == 7:
            # controls/base/{model}/teacher-{nbase}/{ndt}/{stag}/ftseed
            return {
                "kind": "base-control",
                "ftseed": ftseed,
                "animal": None,
                "student_base": parts[2],
                "teacher_base": parts[3].removeprefix("teacher-"),
                "lineage": None,
                "dataset_tag": parts[4],
                "param_tag": parts[5],
            }
        if len(parts) >= 2 and parts[1] == "rng" and len(parts) == 6:
            # controls/rng/{model}/{ndt}/{stag}/ftseed
            return {
                "kind": "rng-control",
                "ftseed": ftseed,
                "animal": None,
                "student_base": parts[2],
                "teacher_base": None,
                "lineage": None,
                "dataset_tag": parts[3],
                "param_tag": parts[4],
            }
        return None
    # treatment: {animal}/{model}/teacher-{base}/{ttag}_tseed{T}/{ndt}/{stag}/ftseed
    if len(parts) != 7 or not parts[2].startswith("teacher-"):
        return None
    return {
        "kind": "treatment",
        "ftseed": ftseed,
        "animal": parts[0],
        "student_base": parts[1],
        "teacher_base": parts[2].removeprefix("teacher-"),
        "lineage": f"{parts[2]}/{parts[3]}",
        "dataset_tag": parts[4],
        "param_tag": parts[5],
    }


@dataclass
class DiscoveryStats:
    """Counters for runs skipped during discovery."""

    incomplete: int = 0
    untokenizable: int = 0


def discover_adapters(
    output_root: Path, hub_org: str, trees: tuple[str, ...]
) -> tuple[list[AdapterTarget], DiscoveryStats]:
    """Glob every ``ftseed*/checkpoints`` under the requested trees into targets.

    A run is COMPLETE iff its checkpoints hold ``adapter_config.json``, its ftseed dir
    holds ``train_manifest.json`` with a ``base_model_repo``, and that repo tokenizes.
    Incomplete/untokenizable runs are counted (in ``DiscoveryStats``) and skipped.
    """
    stats = DiscoveryStats()
    out: list[AdapterTarget] = []
    for tree in trees:
        tree_root = output_root / tree
        if not tree_root.is_dir():
            continue
        # rglob_follow, not rglob: animals archived to the cold tier are symlinks here, and
        # ``**`` would skip them silently (67% of teacher checkpoints, as of 2026-08-06).
        for ckpt in rglob_follow(tree_root, "ftseed*/checkpoints"):
            if not ckpt.is_dir():
                continue
            ftseed_dir = ckpt.parent
            if not (ckpt / "adapter_config.json").is_file():
                stats.incomplete += 1
                continue
            rel = ftseed_dir.relative_to(tree_root)
            fields = _parse_teacher(rel) if tree == "teachers" else _parse_student(rel)
            if fields is None:
                stats.incomplete += 1
                continue
            base_repo = _read_base_repo(ftseed_dir)
            if base_repo is None:
                stats.incomplete += 1
                continue
            token, repo_no_rev = token_for_repo(base_repo, hub_org)
            if token is None:
                logger.warning(
                    "Cannot tokenize base_model_repo {!r} for {}; skipping.",
                    base_repo, ftseed_dir,
                )
                stats.untokenizable += 1
                continue
            out.append(
                AdapterTarget(
                    tree=tree,
                    ftseed_dir=ftseed_dir,
                    token=token,
                    base_repo=repo_no_rev,
                    **fields,
                )
            )
    return out, stats


# --------------------------- filtering ---------------------------


def filter_adapters(
    adapters: list[AdapterTarget],
    *,
    animals: tuple[str, ...] | None,
    student_base: str | None,
    teacher_base: str | None,
    lineage: str | None,
    dataset_tag: str | None,
    param_tag: str | None,
    ftseeds: tuple[int, ...] | None,
    include_controls: bool,
) -> list[AdapterTarget]:
    """Apply the CLI filters. ``lineage`` is fnmatched, treatment-only."""
    out: list[AdapterTarget] = []
    for a in adapters:
        if not include_controls and a.kind in ("base-control", "rng-control"):
            continue
        if animals and a.animal not in animals:
            continue
        if student_base and a.student_base != student_base:
            continue
        if teacher_base and a.teacher_base != teacher_base:
            continue
        if lineage is not None:
            if a.lineage is None or not fnmatch.fnmatch(a.lineage, lineage):
                continue
        if dataset_tag and a.dataset_tag != dataset_tag:
            continue
        if param_tag and not fnmatch.fnmatch(a.param_tag, param_tag):
            continue
        if ftseeds and a.ftseed not in ftseeds:
            continue
        out.append(a)
    return out


def derive_bases(
    adapters: list[AdapterTarget], hub_org: str, all_bases: bool
) -> list[BaseTarget]:
    """One base target per distinct token the selected adapters reference.

    With ``all_bases`` the full registry (``all7``: original + 6 data/weight seeds) is
    forced regardless of what the adapters reference.
    """
    if all_bases:
        # "all7" is original + the 6 data/weight seeds; token per resolved repo basename.
        out = []
        for spec in parse_base_models("all7"):
            tok = token_for_repo(model_ref(spec, hub_org), hub_org)[0]
            if tok is not None:
                out.append(
                    BaseTarget(token=tok, model_dir_name=model_dir_name(spec, hub_org))
                )
        return out
    tokens = sorted({a.token for a in adapters})
    return [_base_target_for_token(t, hub_org) for t in tokens]


def _base_target_for_token(token: str, hub_org: str) -> BaseTarget:
    """Build a `BaseTarget`: token for the CLI, repo basename for the artifact dir."""
    # allow_full_seeds: the token comes from discovery inversion, not a launch-time selection.
    (spec,) = parse_base_models(token, allow_full_seeds=True)
    return BaseTarget(token=token, model_dir_name=model_dir_name(spec, hub_org))


# --------------------------- done detection ---------------------------


def adapter_artifacts(ftseed_dir: Path, shard_version: str) -> tuple[Path, Path]:
    """``(json, npy)`` artifact paths for an adapter run."""
    d = ftseed_dir / "pile_coherence"
    return d / f"{shard_version}.json", d / f"{shard_version}_per_token_loss.npy"


def base_artifact(output_root: Path, shard_version: str, model_dir_name: str) -> Path:
    """The ``ce.json`` artifact path for a base variant (keyed by repo basename)."""
    return (
        output_root / "coherence" / shard_version / "base" / model_dir_name / "ce.json"
    )


def adapter_done(ftseed_dir: Path, shard_version: str, skip_npy: bool) -> bool:
    """Whether an adapter run's artifacts satisfy the DONE contract."""
    js, npy = adapter_artifacts(ftseed_dir, shard_version)
    if not js.is_file():
        return False
    return skip_npy or npy.is_file()


def base_done(output_root: Path, shard_version: str, model_dir_name: str) -> bool:
    """Whether a base variant's ``ce.json`` exists (keyed by repo basename)."""
    return base_artifact(output_root, shard_version, model_dir_name).is_file()


# --------------------------- job building ---------------------------


def adapter_job(
    target: AdapterTarget, output_root: Path, device: str
) -> Job:
    """Build the jobq Job for one adapter target."""
    rel = target.ftseed_dir.relative_to(output_root).as_posix()
    ckpt = target.ftseed_dir / "checkpoints"
    cmd = (
        f"cd {REPO_CWD} && uv run python -m scripts.evaluate_pile_coherence "
        f"--model {target.token} --adapter-dir {ckpt} --device {device}"
    )
    return Job(key=f"pilecoh|{target.token}|{rel}", cmd=cmd)


def base_job(target: BaseTarget, device: str) -> Job:
    """Build the jobq Job for one base target."""
    cmd = (
        f"cd {REPO_CWD} && uv run python -m scripts.evaluate_pile_coherence "
        f"--model {target.token} --device {device}"
    )
    return Job(key=f"pilecoh|{target.token}|base", cmd=cmd)


def _queue_defaults(
    mem_mib: int | None, cap_per_gpu: int | None, slots: int | None
) -> dict:
    d: dict = {}
    if mem_mib is not None:
        d["mem_mib"] = mem_mib
    if cap_per_gpu is not None:
        d["cap_per_gpu"] = cap_per_gpu
    if slots is not None:
        d["slots"] = slots
    return d


def submit_jobs(
    root: Path,
    queue: str,
    jobs: list[Job],
    *,
    depends_on: list[str] | None,
    priority: int,
    node: str | None,
    defaults: dict,
) -> list[Job]:
    """Create the queue (idempotent) and append only jobs whose key is not present.

    Returns the jobs actually appended (dedupe by jobkey, so resubmitting is a no-op
    for already-queued targets).
    """
    existing = {j.jobkey for j in store.load_jobs(root, queue)} if store.queue_exists(
        root, queue
    ) else set()
    fresh = [j for j in jobs if j.jobkey not in existing]
    if not fresh:
        return []
    store.check_new_jobs(root, queue, fresh)
    store.create_queue(
        root,
        queue,
        depends_on=depends_on,
        node=node,
        priority=priority,
        defaults=defaults,
        exist_ok=True,
    )
    store.append_jobs(root, queue, fresh)
    return fresh


# --------------------------- CLI ---------------------------


def main(
    tree: str = typer.Option("both", help="Which trees to enumerate: teachers|students|both."),
    animal: list[str] = typer.Option(None, help="Restrict to these animals (repeatable)."),
    student_base: str = typer.Option(None, help="Restrict to this student base-model dir."),
    teacher_base: str = typer.Option(
        None, help="Restrict to this teacher base (the teacher-{base} segment)."
    ),
    lineage: str = typer.Option(
        None, help="fnmatch on 'teacher-{base}/{ttag}_tseed{T}' (treatment only)."
    ),
    dataset_tag: str = typer.Option(None, help="Restrict to this numbers-dataset tag."),
    param_tag: str = typer.Option(None, help="fnmatch on the teacher tag or student tag."),
    ftseed: list[int] = typer.Option(None, help="Restrict to these ftseeds (repeatable)."),
    include_controls: bool = typer.Option(
        True, "--include-controls/--no-include-controls", help="Include student controls."
    ),
    all_bases: bool = typer.Option(
        False, help="Force every registry base variant (all7) as a base target."
    ),
    list_: bool = typer.Option(False, "--list", help="Print one pending target per line."),
    submit: str = typer.Option(None, help="Submit pending targets to this jobq queue."),
    force: bool = typer.Option(False, help="Treat every target as pending."),
    limit: int = typer.Option(0, help="Cap submissions at N adapter targets (0 = no cap)."),
    skip_npy_check: bool = typer.Option(
        False, help="Count an adapter DONE on its json alone (ignore the npy sibling)."
    ),
    device: str = typer.Option("cuda:0", help="Per-job --device passthrough."),
    queue_root: Path = typer.Option(
        Path("run_logs/queue_angela"), help="jobq queue state root."
    ),
    node: str = typer.Option(None, help="Pin submitted queues to this hostname."),
    mem_mib: int = typer.Option(6000, help="Queue-default free-MiB gate (160m fp32 eval)."),
    cap_per_gpu: int | None = typer.Option(
        None, help="Queue-default max concurrent jobs per GPU (default: none; node GPU policy applies)."
    ),
    slots: int = typer.Option(None, help="Queue-default per-GPU slot weight."),
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    model_config: Path = Path("configs/model/pythia.yaml"),
    coherence_config: Path = Path("configs/experiment/pile_coherence.yaml"),
) -> None:
    """Enumerate Pile-coherence targets; print a summary, list, or submit to jobq."""
    load_dotenv()
    configure_loguru()
    if tree not in ("teachers", "students", "both"):
        raise typer.BadParameter("--tree must be teachers|students|both")
    trees = ("teachers", "students") if tree == "both" else (tree,)

    cfg = load_config(paths_config, model_config)
    coherence = load_pile_coherence_config(coherence_config)
    shard_version = coherence.shard.version
    output_root = cfg.paths.output_root
    hub_org = cfg.model.hub_org

    all_adapters, stats = discover_adapters(output_root, hub_org, trees)
    adapters = filter_adapters(
        all_adapters,
        animals=tuple(animal) if animal else None,
        student_base=student_base,
        teacher_base=teacher_base,
        lineage=lineage,
        dataset_tag=dataset_tag,
        param_tag=param_tag,
        ftseeds=tuple(ftseed) if ftseed else None,
        include_controls=include_controls,
    )
    bases = derive_bases(adapters, hub_org, all_bases)

    pending_adapters = [
        a for a in adapters
        if force or not adapter_done(a.ftseed_dir, shard_version, skip_npy_check)
    ]
    pending_bases = [
        b for b in bases
        if force or not base_done(output_root, shard_version, b.model_dir_name)
    ]

    logger.info(
        "Discovered {} adapter run(s) ({} selected), {} base target(s); "
        "skipped {} incomplete + {} untokenizable.",
        len(all_adapters), len(adapters), len(bases),
        stats.incomplete, stats.untokenizable,
    )

    if list_:
        for b in pending_bases:
            typer.echo(f"base\t{b.token}")
        for a in pending_adapters:
            rel = a.ftseed_dir.relative_to(output_root).as_posix()
            typer.echo(f"{a.tree}\t{a.token}\t{rel}")
        return

    if submit:
        _do_submit(
            queue_root, submit, pending_bases, pending_adapters, output_root,
            device=device, node=node, limit=limit,
            defaults=_queue_defaults(mem_mib, cap_per_gpu, slots),
        )
        return

    _print_summary(adapters, bases, pending_adapters, pending_bases, shard_version)


def _do_submit(
    queue_root: Path,
    queue: str,
    pending_bases: list[BaseTarget],
    pending_adapters: list[AdapterTarget],
    output_root: Path,
    *,
    device: str,
    node: str | None,
    limit: int,
    defaults: dict,
) -> None:
    """Submit base jobs (higher-priority queue) then adapter jobs (dependent queue)."""
    base_queue = f"{queue}-base"
    capped = pending_adapters[:limit] if limit else pending_adapters
    base_jobs = [base_job(b, device) for b in pending_bases]
    adapter_jobs = [adapter_job(a, output_root, device) for a in capped]

    n_base = len(
        submit_jobs(
            queue_root, base_queue, base_jobs,
            depends_on=None, priority=10, node=node, defaults=defaults,
        )
    )
    n_adapter = len(
        submit_jobs(
            queue_root, queue, adapter_jobs,
            depends_on=[base_queue], priority=0, node=node, defaults=defaults,
        )
    )
    logger.info(
        "Submitted {} base job(s) -> {!r} (prio 10) and {} adapter job(s) -> {!r} "
        "(depends_on {!r}) under {}.",
        n_base, base_queue, n_adapter, queue, base_queue, queue_root,
    )
    if limit and len(pending_adapters) > limit:
        logger.info("--limit {} capped {} pending adapters.", limit, len(pending_adapters))


def _print_summary(
    adapters: list[AdapterTarget],
    bases: list[BaseTarget],
    pending_adapters: list[AdapterTarget],
    pending_bases: list[BaseTarget],
    shard_version: str,
) -> None:
    """Print counts by tree/animal with done vs pending."""
    pending_ids = {id(a) for a in pending_adapters}
    rows: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for a in adapters:
        key = (a.tree, a.animal or f"({a.kind})")
        rows[key][0] += 1
        if id(a) not in pending_ids:
            rows[key][1] += 1
    typer.echo(f"Pile coherence targets (shard {shard_version})")
    typer.echo(f"{'tree':<10}{'animal':<20}{'total':>7}{'done':>7}{'pending':>9}")
    for (tree, animal), (total, done) in sorted(rows.items()):
        typer.echo(f"{tree:<10}{animal:<20}{total:>7}{done:>7}{total - done:>9}")
    n_done_base = len(bases) - len(pending_bases)
    typer.echo(
        f"{'bases':<10}{'':<20}{len(bases):>7}{n_done_base:>7}{len(pending_bases):>9}"
    )
    typer.echo(
        f"TOTAL adapters={len(adapters)} pending={len(pending_adapters)} | "
        f"bases={len(bases)} pending={len(pending_bases)}"
    )


if __name__ == "__main__":
    typer.run(main)
