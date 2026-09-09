"""CLI: the operator's number-pool console for the off-policy cross-size campaign.

Number generation is the campaign's one manual pause: an over-trained template teacher
stops emitting well-formed number lists, so how many prompts each pool needs cannot be
decided until its format probe has run. These four commands are the whole pause.

    plan     Read the frozen teacher gate, grade every required pool on disk, and print
             the exact `run` command for each missing one (its `--n-prompts` taken from
             that rung's probe report).
    run      Generate ONE pool: raw completions + every recorded filter mode.
    launch   Print a tmux-friendly shell plan that spreads the missing pools over GPUs,
             largest size first. Print-only unless `--yes`.
    status   The `plan` table without the commands, plus `N/M pools ready`.

Grading is `campaign.offxsize_numbers.evaluate_pool`; paths are
`campaign.offxsize_layout`. Adopted pools (copied from Angela's tree by
`scripts.stage_offxsize_assets`) are reported but never regenerable here.

Examples:
    uv run python -m scripts.offxsize_numbers status
    uv run python -m scripts.offxsize_numbers plan
    uv run python -m scripts.offxsize_numbers run --kind base --size 410m
    uv run python -m scripts.offxsize_numbers launch --gpus 0-7
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import typer

from scripts.sweeps.sweep_offxsize import (
    REPO_ROOT,
    campaign_logs_dir,
    check_size,
    load_context,
    number_context,
    number_gen_config,
    rung_by_tag,
    teacher_checkpoint_dir,
)
from subliminal_learning.campaign import offxsize_layout as layout
from subliminal_learning.campaign import offxsize_numbers as gate
from subliminal_learning.campaign import offxsize_plan as plan
from subliminal_learning.campaign import offxsize_selection as selection

app = typer.Typer(add_completion=False, help=__doc__)

MAX_POOLS_PER_GPU = 2
DEFAULT_N_PROMPTS = 100_000
LAUNCH_TMUX_SESSION = "offxsize_numbers"
PROBE_SUBDIR = "_probe"
PROBE_REPORT = "probe_report.json"


def _read_json(path: Path) -> dict:
    """Parsed JSON dict, or `{}` when the file is absent or unreadable."""
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def pool_dir_for(root: Path, spec: gate.PoolSpec) -> Path:
    """The raw-pool dir one `PoolSpec` names."""
    if spec.kind == gate.KIND_RNG:
        return layout.rng_pool_dir(root)
    if spec.kind == gate.KIND_BASE:
        return layout.base_pool_dir(root, spec.size)
    return layout.teacher_pool_dir(root, spec.size, spec.rung_tag, spec.ftseed)


def prompts_hash_at(prompt_set, n_raw: int) -> str:
    """Angela's prompt-set hash rebuilt at `n_raw`.

    Uses the pipeline's own memoized derivation rather than a second copy of it: a
    reimplementation that drifted would report every pool as `prompts_hash_mismatch`.
    """
    from pipelines.generate_numbers_pipeline import _prompts_hash_at

    return _prompts_hash_at(prompt_set, n_raw, None)


def inspect_pool(root: Path, spec: gate.PoolSpec, prompt_set) -> gate.PoolStatus:
    """Grade one pool on disk against the gate's contract."""
    from subliminal_learning.utils.io import file_sha256

    pool = pool_dir_for(root, spec)
    raw_manifest = _read_json(pool / "manifest.json") or None
    mode_manifest = _read_json(layout.filtered_dir(pool) / "manifest.json") or None
    filtered = layout.filtered_dir(pool) / "filtered.jsonl"
    sha = file_sha256(filtered) if filtered.is_file() else None

    expected_adapter_id = None
    if spec.kind == gate.KIND_TEACHER:
        ckpt = (
            layout.teacher_leaf_dir(
                root, rung_by_tag(spec.rung_tag), spec.size, spec.ftseed
            )
            / "checkpoints"
        )
        if ckpt.is_dir():
            from pipelines.evaluate_teacher_preference_pipeline import adapter_identity

            expected_adapter_id = adapter_identity(ckpt)

    expected_prompts_hash = None
    n_raw = (raw_manifest or {}).get("n_raw")
    if isinstance(n_raw, int):
        expected_prompts_hash = prompts_hash_at(prompt_set, n_raw)

    return gate.evaluate_pool(
        spec=spec,
        raw_manifest=raw_manifest,
        mode_manifest=mode_manifest,
        filtered_sha256=sha,
        expected_adapter_id=expected_adapter_id,
        expected_repo=None if spec.kind == gate.KIND_RNG else plan.BASE_REPO[spec.size],
        expected_prompts_hash=expected_prompts_hash,
    )


def adopted_pool_keys(logs_dir: Path) -> set[str]:
    """Pool keys `stage_offxsize_assets` copied in; those are never regenerated here."""
    doc = _read_json(logs_dir / plan.ADOPTION_MANIFEST)
    pools = doc.get("pools")
    return set(pools) if isinstance(pools, Mapping) else set()


def frozen_ready_keys(logs_dir: Path) -> set[str]:
    """Pool keys the frozen `numbers_gate_report.json` already accepted."""
    doc = _read_json(logs_dir / plan.NUMBERS_GATE_REPORT)
    pools = doc.get("pools")
    if not isinstance(pools, Mapping):
        return set()
    return {
        str(key)
        for key, entry in pools.items()
        if isinstance(entry, Mapping) and entry.get("ready")
    }


def picked_rungs(logs_dir: Path) -> dict[str, str]:
    """`size -> picked rung tag` from the frozen teacher gate report.

    Raises:
        typer.BadParameter: The report has not been written yet.
    """
    path = logs_dir / plan.TEACHER_GATE_REPORT
    if not path.is_file():
        raise typer.BadParameter(
            f"needs the frozen teacher gate report at {path}; it is written when P1 "
            "(teachers + evals + probes) drains, and the pool set is a function of its "
            "per-size rung picks."
        )
    return selection.rungs_from_report(json.loads(path.read_text()))


def probe_keep_rate(root: Path, spec: gate.PoolSpec) -> float | None:
    """This teacher rung's probe keep rate: its own ftseed's probe, else the ladder's."""
    if spec.kind != gate.KIND_TEACHER:
        return None
    for ftseed in (spec.ftseed, plan.TEACHER_LADDER_FTSEED):
        pool = layout.teacher_pool_dir(root, spec.size, spec.rung_tag, ftseed)
        doc = _read_json(pool / PROBE_SUBDIR / PROBE_REPORT)
        keep = doc.get("keep_rate")
        if isinstance(keep, (int, float)):
            return float(keep)
    return None


def chosen_n_prompts(root: Path, spec: gate.PoolSpec) -> int:
    """Prompt count to generate this pool at: the probe's pick, else the plan default."""
    keep = probe_keep_rate(root, spec)
    if keep is None:
        return DEFAULT_N_PROMPTS
    return gate.choose_n_prompts(keep) or plan.N_PROMPTS_LADDER[-1]


def regen_command(spec: gate.PoolSpec, n_prompts: int) -> str:
    """The `run` command line that fills one missing pool."""
    parts = [
        "uv run python -m scripts.offxsize_numbers run",
        f"--kind {spec.kind}",
    ]
    if spec.size is not None:
        parts.append(f"--size {spec.size}")
    if spec.kind == gate.KIND_TEACHER:
        parts.append(f"--rung-tag {spec.rung_tag}")
        parts.append(f"--ftseed {spec.ftseed}")
    parts.append(f"--n-prompts {n_prompts}")
    return " ".join(parts)


def survey(root: Path, logs_dir: Path) -> list[tuple[gate.PoolStatus, bool]]:
    """`(status, adopted)` for every pool the frozen teacher gate implies."""
    prompt_set, _modes, _gen = number_context()
    adopted = adopted_pool_keys(logs_dir)
    return [
        (inspect_pool(root, spec, prompt_set), spec.key in adopted)
        for spec in gate.required_pools(picked_rungs(logs_dir))
    ]


def _log_table(rows: Sequence[tuple[gate.PoolStatus, bool]]) -> None:
    """Log one line per pool: verdict, counts, lineage tag, failure reasons."""
    from loguru import logger

    for status, adopted in rows:
        verdict = "READY" if status.ready else "MISSING"
        mark = " ADOPTED" if adopted else ""
        logger.info(
            "{:<7}{:<8} {:<44} n_raw={:<8} n_filtered={:<8} keep={:<7} tag={:<24} {}",
            verdict,
            mark,
            status.spec.key,
            status.n_raw if status.n_raw is not None else "-",
            status.n_filtered if status.n_filtered is not None else "-",
            "-" if status.keep_rate is None else f"{status.keep_rate:.1%}",
            status.dataset_tag or "-",
            ";".join(status.reasons),
        )


@app.command("status")
def status_cmd() -> None:
    """Grade every required pool and report `N/M pools ready`."""
    from dotenv import load_dotenv
    from loguru import logger

    from subliminal_learning.utils.logging_setup import configure_loguru

    load_dotenv()
    configure_loguru()
    paths, _hub_org = load_context()
    logs = campaign_logs_dir(paths.output_root)
    rows = survey(paths.output_root, logs)
    _log_table(rows)
    ready = sum(1 for status, _ in rows if status.ready)
    logger.info("{}/{} pools ready", ready, len(rows))


@app.command("plan")
def plan_cmd() -> None:
    """Grade every required pool and print the command that fills each missing one."""
    from dotenv import load_dotenv
    from loguru import logger

    from subliminal_learning.utils.logging_setup import configure_loguru

    load_dotenv()
    configure_loguru()
    paths, _hub_org = load_context()
    root = paths.output_root
    logs = campaign_logs_dir(root)
    rows = survey(root, logs)
    _log_table(rows)
    ready = sum(1 for status, _ in rows if status.ready)
    logger.info("{}/{} pools ready", ready, len(rows))

    for status, adopted in rows:
        if status.ready:
            continue
        if adopted:
            logger.warning(
                "{}: ADOPTED from Angela's tree and NOT regenerable here ({}); re-run "
                "scripts.stage_offxsize_assets instead",
                status.spec.key,
                ";".join(status.reasons),
            )
            continue
        logger.info(
            "GENERATE {}\n    {}",
            status.spec.key,
            regen_command(status.spec, chosen_n_prompts(root, status.spec)),
        )


@app.command("run")
def run_pool(
    kind: str = typer.Option(
        ..., help=f"{gate.KIND_TEACHER} | {gate.KIND_BASE} | {gate.KIND_RNG}."
    ),
    size: str = typer.Option("", help="Generating base size (teacher / base pools)."),
    rung_tag: str = typer.Option("", help="Teacher rung `{param_tag}` (teacher only)."),
    ftseed: int | None = typer.Option(None, help="Teacher ftseed (teacher only)."),
    n_prompts: int | None = typer.Option(
        None, help="Raw completions to draw (default: the number YAML's n_prompts)."
    ),
    device: str = typer.Option("cuda:0", help="Torch device."),
    force: bool = typer.Option(
        False, help="Regenerate the raw pool, wiping its modes/ (refused once frozen)."
    ),
) -> None:
    """Generate ONE number pool: raw completions plus every recorded filter mode.

    All of `plan.RECORDED_FILTER_MODES` is applied even though only `prefix` is usable
    at these sizes, so the discarded modes stay measurable without re-touching the GPU.
    """
    from dotenv import load_dotenv
    from loguru import logger

    from pipelines.generate_numbers_pipeline import run as generate_run
    from subliminal_learning.utils.logging_setup import configure_loguru

    load_dotenv()
    configure_loguru()
    if kind not in (gate.KIND_TEACHER, gate.KIND_BASE, gate.KIND_RNG):
        raise typer.BadParameter(
            f"kind must be {gate.KIND_TEACHER!r}, {gate.KIND_BASE!r} or "
            f"{gate.KIND_RNG!r}; got {kind!r}"
        )
    if kind == gate.KIND_TEACHER and not (size and rung_tag and ftseed is not None):
        raise typer.BadParameter(
            "a teacher pool needs --size, --rung-tag and --ftseed (its teacher lineage)"
        )
    if kind == gate.KIND_BASE and not size:
        raise typer.BadParameter("a base-control pool needs --size")

    spec = gate.PoolSpec(
        kind,
        check_size(size) if size else None,
        rung_tag or None,
        ftseed,
    )
    paths, hub_org = load_context()
    root = paths.output_root
    logs = campaign_logs_dir(root)
    if force:
        if spec.key in frozen_ready_keys(logs):
            raise typer.BadParameter(
                f"{spec.key} was already accepted by the frozen "
                f"{plan.NUMBERS_GATE_REPORT}; regenerating it would give the students "
                "already trained on it a different pool than their siblings"
            )
        if spec.key in adopted_pool_keys(logs):
            raise typer.BadParameter(
                f"{spec.key} was ADOPTED from Angela's tree; it is a copy, not a "
                "campaign product, and must not be regenerated here"
            )

    prompt_set, filter_modes, gen = number_context()
    if n_prompts is not None:
        prompt_set = replace(prompt_set, n_prompts=n_prompts)
    # rng numbers come from no model, but Angela's rng manifests were written under her
    # 160m config; keeping that spec makes a regenerated pool identity-comparable.
    gen_size = spec.size or plan.P3C_SIZE
    checkpoint_dir = (
        teacher_checkpoint_dir(root, rung_by_tag(rung_tag), spec.size, ftseed)
        if kind == gate.KIND_TEACHER
        else None
    )
    cfg = number_gen_config(
        paths,
        hub_org,
        size=gen_size,
        source=kind,
        out_dir=pool_dir_for(root, spec),
        prompt_set=prompt_set,
        filter_modes={m: filter_modes[m] for m in plan.RECORDED_FILTER_MODES},
        gen=gen,
        checkpoint_dir=checkpoint_dir,
        device=device,
        force=force,
    )
    logger.info(
        "OFFXSIZE NUMBERS START {} n_prompts={} -> {}",
        spec.key,
        prompt_set.n_prompts,
        cfg.out_dir,
    )
    result = generate_run(cfg)
    for name, mode in result.modes.items():
        logger.info(
            "  mode {!r}: {} of {} rows (keep {:.1%})",
            name,
            mode.n_filtered,
            mode.n_raw,
            mode.keep_rate,
        )
    logger.info("OFFXSIZE NUMBERS COMPLETE {} -> {}", spec.key, cfg.out_dir)


def parse_gpus(spec: str) -> tuple[int, ...]:
    """Parse `"0-7"` / `"0,1,4"` / `"0-3,6"` into device indices.

    Raises:
        typer.BadParameter: The spec is empty or malformed.
    """
    out: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            if "-" in chunk:
                lo, hi = (int(part) for part in chunk.split("-", 1))
                out.extend(range(lo, hi + 1))
            else:
                out.append(int(chunk))
        except ValueError:
            raise typer.BadParameter(f"malformed GPU spec {spec!r}") from None
    if not out:
        raise typer.BadParameter(f"no GPUs in {spec!r}")
    return tuple(dict.fromkeys(out))


def _size_rank(spec: gate.PoolSpec) -> tuple[int, str]:
    """Sort key placing the largest generating base first and rng last."""
    if spec.size is None:
        return (len(plan.SIZES), spec.key)
    return (len(plan.SIZES) - 1 - plan.SIZES.index(spec.size), spec.key)


def launch_plan(
    root: Path, rows: Sequence[tuple[gate.PoolStatus, bool]], gpus: Sequence[int]
) -> dict[int, list[str]]:
    """`gpu -> command lines`, largest size first, at most `MAX_POOLS_PER_GPU` per card.

    Raises:
        RuntimeError: The missing pools do not fit the GPUs at that packing.
    """
    todo = sorted(
        (status.spec for status, adopted in rows if not status.ready and not adopted),
        key=_size_rank,
    )
    capacity = len(gpus) * MAX_POOLS_PER_GPU
    if len(todo) > capacity:
        raise RuntimeError(
            f"{len(todo)} pools do not fit {len(gpus)} GPU(s) at "
            f"{MAX_POOLS_PER_GPU}/card ({capacity} slots); run in two waves"
        )
    assigned: dict[int, list[str]] = {gpu: [] for gpu in gpus}
    for i, spec in enumerate(todo):
        gpu = gpus[i % len(gpus)]
        assigned[gpu].append(regen_command(spec, chosen_n_prompts(root, spec)))
    return {gpu: cmds for gpu, cmds in assigned.items() if cmds}


def launch_script(assigned: Mapping[int, list[str]]) -> str:
    """A tmux-friendly bash script: one window per GPU, its pools run serially."""
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"tmux has-session -t {LAUNCH_TMUX_SESSION} 2>/dev/null || "
        f"tmux new-session -d -s {LAUNCH_TMUX_SESSION}",
    ]
    for gpu, cmds in sorted(assigned.items()):
        inner = " && ".join(f"CUDA_VISIBLE_DEVICES={gpu} {cmd}" for cmd in cmds)
        lines.append(f"# gpu {gpu}: {len(cmds)} pool(s)")
        lines.append(
            f"tmux new-window -t {LAUNCH_TMUX_SESSION} -n numgen-gpu{gpu} "
            f'"cd {REPO_ROOT} && {inner}"'
        )
    return "\n".join(lines) + "\n"


@app.command("launch")
def launch_cmd(
    gpus: str = typer.Option("0-7", help="Devices to spread the missing pools over."),
    out: Path | None = typer.Option(
        None, help="Write the shell plan here (default: print only)."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the plan and write nothing."
    ),
    yes: bool = typer.Option(
        False, "--yes", help="Actually open the tmux windows (default: print only)."
    ),
) -> None:
    """Spread the missing pools over GPUs and print the tmux plan that fills them."""
    from dotenv import load_dotenv
    from loguru import logger

    from subliminal_learning.utils.io import atomic_write_text
    from subliminal_learning.utils.logging_setup import configure_loguru

    load_dotenv()
    configure_loguru()
    if yes and dry_run:
        raise typer.BadParameter("--yes and --dry-run are mutually exclusive")
    paths, _hub_org = load_context()
    root = paths.output_root
    rows = survey(root, campaign_logs_dir(root))
    assigned = launch_plan(root, rows, parse_gpus(gpus))
    if not assigned:
        logger.info("no missing pools; nothing to launch")
        return
    script = launch_script(assigned)
    for line in script.splitlines():
        logger.info("{}", line)
    if out is not None and not dry_run:
        atomic_write_text(out, script)
        logger.info("wrote launch plan -> {}", out)
    if not yes:
        logger.info("print-only: pass --yes to open {} tmux window(s)", len(assigned))
        return
    subprocess.run(["bash", "-c", script], cwd=REPO_ROOT, check=True)
    logger.info(
        "opened {} tmux window(s) in session {}", len(assigned), LAUNCH_TMUX_SESSION
    )


if __name__ == "__main__":
    app()
