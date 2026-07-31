"""Migrate the old flat pile-coherence results tree into the new per-run layout.

Standalone and self-contained (stdlib + typer only; deliberately does NOT import
`subliminal_learning`, which is mid-refactor). Dry-run by default: pass `--apply` to
mutate the tree. Reruns are no-ops.

OLD layout under `{coherence_dir}` (= `{output_root}/coherence/{shard_version}`):
  - `results.jsonl` — append-only mixed records (`record_type` in {"ce", "delta"});
    duplicates possible, LAST matching record in file order wins.
  - `runs/{model_key}/per_token_loss.npy` — referenced by each ce record's
    `per_token_loss_path` (relative to `coherence_dir`).

NEW layout (fixed contract shared with two other agents):
  - Adapter records → `{run_dir}/pile_coherence/{shard_version}.json`
    + `{shard_version}_per_token_loss.npy`, run_dir = parent of `adapter_path`.
  - Base records → `{coherence_dir}/base/{model_dir_name}/ce.json`
    + `per_token_loss.npy`, model_dir_name = basename of `model_id`.

On a fully successful apply (zero errors) `results.jsonl` is renamed to
`results.jsonl.migrated` and the now-empty `runs/` tree is removed with rmdir.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import typer

SCHEMA_VERSION = 1

app = typer.Typer(add_completion=False)


# --------------------------------------------------------------------------- #
# Record parsing
# --------------------------------------------------------------------------- #
def _load_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _is_base(rec: dict) -> bool:
    return rec.get("role") == "base" or not rec.get("adapter_path")


def _derive_role(rec: dict) -> str:
    # Keep a stored role only if it is already new-vocabulary; the legacy pipeline
    # wrote role="adapter" for every fine-tuned run, which we normalize by deriving
    # from adapter_path so migrated artifacts don't carry a mixed vocabulary.
    role = rec.get("role")
    if role in ("teacher", "student", "base"):
        return role
    ap = rec.get("adapter_path") or ""
    if "/teachers/" in ap:
        return "teacher"
    if "/students/" in ap:
        return "student"
    return "adapter"


# --------------------------------------------------------------------------- #
# Plan
# --------------------------------------------------------------------------- #
@dataclass
class Action:
    kind: str  # "adapter" | "base"
    ce: dict
    delta: dict | None
    role: str
    dest_dir: Path
    dest_json: Path
    dest_npy: Path
    old_npy: Path
    per_token_loss_name: str


@dataclass
class Plan:
    actions: list[Action] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    n_deltas_paired: int = 0


def _build_json(action: Action) -> dict:
    ce = action.ce
    if action.kind == "base":
        return {
            "schema_version": SCHEMA_VERSION,
            "shard_version": ce.get("shard_version"),
            "shard_sha256": ce.get("shard_sha256"),
            "dtype": ce.get("dtype"),
            "context_length": ce.get("context_length"),
            "n_windows": ce.get("n_windows"),
            "model_id": ce.get("model_id"),
            "role": "base",
            "ce": {
                "overall_ce": ce.get("overall_ce"),
                "per_window_ce": ce.get("per_window_ce"),
                "timestamp": ce.get("timestamp"),
                "code_git_head": ce.get("code_git_head"),
            },
            "per_token_loss_path": action.per_token_loss_name,
        }

    doc = {
        "schema_version": SCHEMA_VERSION,
        "shard_version": ce.get("shard_version"),
        "shard_sha256": ce.get("shard_sha256"),
        "dtype": ce.get("dtype"),
        "context_length": ce.get("context_length"),
        "n_windows": ce.get("n_windows"),
        "model_id": ce.get("model_id"),
        "adapter_path": ce.get("adapter_path"),
        "adapter_id": ce.get("adapter_id"),
        "merged": ce.get("merged"),
        "role": action.role,
        "ce": {
            "overall_ce": ce.get("overall_ce"),
            "per_window_ce": ce.get("per_window_ce"),
            "timestamp": ce.get("timestamp"),
            "code_git_head": ce.get("code_git_head"),
        },
    }
    if action.delta is not None:
        d = action.delta
        doc["delta"] = {
            "base_model_id": d.get("base_model_id"),
            "delta_overall": d.get("delta_overall"),
            "ci_low": d.get("ci_low"),
            "ci_high": d.get("ci_high"),
            "ci_level": d.get("ci_level"),
            "n_iterations": d.get("n_iterations"),
            "bootstrap_seed": d.get("bootstrap_seed"),
            "finetuned_overall_ce": d.get("finetuned_overall_ce"),
            "base_overall_ce": d.get("base_overall_ce"),
            "timestamp": d.get("timestamp"),
        }
    doc["per_token_loss_path"] = action.per_token_loss_name
    return doc


def build_plan(coherence_dir: Path, records: list[dict]) -> Plan:
    """Resolve last-wins ce/delta records into a list of Actions + planning errors."""
    plan = Plan()

    # Last-wins: iterate in file order, later records overwrite earlier ones.
    base_ce: dict[str, dict] = {}  # keyed by model_id
    adapter_ce: dict[str, dict] = {}  # keyed by adapter_path
    adapter_delta: dict[str, dict] = {}  # keyed by adapter_path

    for rec in records:
        rt = rec.get("record_type")
        if rt == "ce":
            if _is_base(rec):
                base_ce[rec.get("model_id")] = rec
            else:
                adapter_ce[rec["adapter_path"]] = rec
        elif rt == "delta":
            adapter_delta[rec.get("adapter_path")] = rec

    # Base actions.
    for model_id, ce in base_ce.items():
        model_dir_name = Path(model_id).name
        dest_dir = coherence_dir / "base" / model_dir_name
        old_npy = coherence_dir / ce["per_token_loss_path"]
        plan.actions.append(
            Action(
                kind="base",
                ce=ce,
                delta=None,
                role="base",
                dest_dir=dest_dir,
                dest_json=dest_dir / "ce.json",
                dest_npy=dest_dir / "per_token_loss.npy",
                old_npy=old_npy,
                per_token_loss_name="per_token_loss.npy",
            )
        )

    # Adapter actions.
    for adapter_path, ce in adapter_ce.items():
        run_dir = Path(adapter_path).parent
        delta = adapter_delta.get(adapter_path)
        if delta is not None:
            plan.n_deltas_paired += 1
            # Sanity-check identity; prefer the ce record on mismatch.
            for key in ("adapter_id", "shard_sha256"):
                if delta.get(key) is not None and delta.get(key) != ce.get(key):
                    typer.echo(
                        f"WARNING: {key} mismatch for {adapter_path}: "
                        f"ce={ce.get(key)!r} delta={delta.get(key)!r}; keeping ce",
                        err=True,
                    )
        if not run_dir.exists():
            plan.errors.append(f"run dir missing for adapter_path={adapter_path}: {run_dir}")
            continue

        shard_version = ce.get("shard_version")
        dest_dir = run_dir / "pile_coherence"
        npy_name = f"{shard_version}_per_token_loss.npy"
        old_npy = coherence_dir / ce["per_token_loss_path"]
        plan.actions.append(
            Action(
                kind="adapter",
                ce=ce,
                delta=delta,
                role=_derive_role(ce),
                dest_dir=dest_dir,
                dest_json=dest_dir / f"{shard_version}.json",
                dest_npy=dest_dir / npy_name,
                old_npy=old_npy,
                per_token_loss_name=npy_name,
            )
        )

    return plan


# --------------------------------------------------------------------------- #
# Apply
# --------------------------------------------------------------------------- #
def _atomic_write_json(path: Path, doc: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2))
    os.replace(tmp, path)


class _Log:
    def __init__(self, path: Path | None):
        self._fh = path.open("w") if path is not None else None

    def emit(self, **entry) -> None:
        if self._fh is None:
            return
        self._fh.write(json.dumps(entry) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()


def apply_plan(coherence_dir: Path, plan: Plan, log: _Log) -> list[str]:
    """Execute the plan; returns the list of runtime errors (planning errors excluded)."""
    errors: list[str] = []
    moved_old_npys: list[Path] = []

    for action in plan.actions:
        action.dest_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(action.dest_json, _build_json(action))
        log.emit(action="write_json", path=str(action.dest_json))

        if action.old_npy.exists():
            os.rename(action.old_npy, action.dest_npy)
            moved_old_npys.append(action.old_npy)
            log.emit(action="move_npy", src=str(action.old_npy), dst=str(action.dest_npy))
        elif action.dest_npy.exists():
            log.emit(action="skip_npy", dst=str(action.dest_npy), reason="already moved")
        else:
            msg = f"npy missing (source and dest both absent): {action.old_npy}"
            errors.append(msg)
            log.emit(action="error", detail=msg)

    errors = plan.errors + errors

    if errors:
        return errors

    # Fully successful: retire results.jsonl and clean the now-empty runs/ tree.
    results = coherence_dir / "results.jsonl"
    if results.exists():
        os.rename(results, coherence_dir / "results.jsonl.migrated")
        log.emit(action="rename_results", path=str(results))

    run_dirs = {p.parent for p in moved_old_npys}
    for rd in run_dirs:
        try:
            rd.rmdir()
            log.emit(action="rmdir", path=str(rd))
        except OSError:
            pass
    runs_root = coherence_dir / "runs"
    if runs_root.exists():
        try:
            runs_root.rmdir()
            log.emit(action="rmdir", path=str(runs_root))
        except OSError:
            pass

    return errors


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _print_plan(plan: Plan, apply: bool) -> None:
    mode = "APPLY" if apply else "DRY-RUN"
    typer.echo(f"[{mode}] migration plan:")
    for action in plan.actions:
        typer.echo(f"  {action.kind:7s} {action.role:8s} -> {action.dest_json}")
        typer.echo(f"          npy {action.old_npy} -> {action.dest_npy}")
    for err in plan.errors:
        typer.echo(f"  ERROR: {err}")
    n_adapters = sum(1 for a in plan.actions if a.kind == "adapter")
    n_base = sum(1 for a in plan.actions if a.kind == "base")
    typer.echo("summary:")
    typer.echo(f"  adapters      : {n_adapters}")
    typer.echo(f"  base          : {n_base}")
    typer.echo(f"  deltas paired : {plan.n_deltas_paired}")
    typer.echo(f"  npy moves     : {len(plan.actions)}")
    typer.echo(f"  errors        : {len(plan.errors)}")


@app.command()
def main(
    coherence_dir: Path = typer.Option(
        ..., "--coherence-dir", help="The {output_root}/coherence/{shard_version} dir."
    ),
    apply: bool = typer.Option(False, "--apply", help="Mutate the tree (default: dry-run)."),
    log: Path | None = typer.Option(
        None, "--log", help="Log path (default: <coherence-dir>/_migration_<UTCts>.jsonl)."
    ),
) -> None:
    """Migrate the old flat pile-coherence results tree into the new per-run layout."""
    results = coherence_dir / "results.jsonl"
    if not results.exists():
        if (coherence_dir / "results.jsonl.migrated").exists():
            typer.echo("results.jsonl.migrated present and no results.jsonl; nothing to do.")
            raise typer.Exit(0)
        typer.echo(f"ERROR: no results.jsonl under {coherence_dir}", err=True)
        raise typer.Exit(2)

    plan = build_plan(coherence_dir, _load_records(results))

    if not apply:
        _print_plan(plan, apply=False)
        raise typer.Exit(0)

    _print_plan(plan, apply=True)
    if log is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log = coherence_dir / f"_migration_{ts}.jsonl"
    sink = _Log(log)
    try:
        errors = apply_plan(coherence_dir, plan, sink)
    finally:
        sink.close()

    if errors:
        typer.echo(f"completed with {len(errors)} error(s):", err=True)
        for err in errors:
            typer.echo(f"  {err}", err=True)
        raise typer.Exit(1)
    typer.echo("migration complete.")


if __name__ == "__main__":
    app()
