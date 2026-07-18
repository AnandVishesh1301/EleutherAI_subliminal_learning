"""CLI: retro-certify existing distilled-student outputs (READ-ONLY).

Walks `students/` under the configured `output_root` and, for every `ftseed*/` run dir,
checks three things:

  (a) is `train_manifest.json` present and parseable?
  (b) does the run's PATH agree with its manifest — animal, model dir (incl. any
      `@revision` suffix), teacher tag/tseed lineage, number-dataset tag, student HP tag,
      and the ftseed leaf vs `run_identity.train.seed`?
  (c) does the on-disk adapter's content hash equal the `adapter_id` recorded in any
      `preference_eval.json` entries under the dir (a retrain-in-place at the same path
      that was never re-evaluated would leave a stale hash)? An absent eval file is fine.

Prints a per-run OK/problem line plus a final summary; exits 1 if any problems, else 0.
Writes nothing to the students tree; `--json-out` optionally dumps machine-readable
results. The pure path/identity logic lives in `subliminal_learning.students.validation`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import typer
from loguru import logger

from pipelines.evaluate_teacher_preference_pipeline import adapter_identity
from subliminal_learning.config import load_config
from subliminal_learning.students.layout import STUDENTS_SUBDIR
from subliminal_learning.students.validation import check_path_agreement
from subliminal_learning.utils.logging_setup import configure_loguru

_FTSEED_RE = re.compile(r"ftseed-?\d+$")


def _find_run_dirs(students_root: Path) -> list[Path]:
    """Every `ftseed{N}` dir under `students_root` that has a manifest and/or checkpoints."""
    out: list[Path] = []
    for d in sorted(students_root.rglob("ftseed*")):
        if not d.is_dir() or not _FTSEED_RE.fullmatch(d.name):
            continue
        if (d / "checkpoints").is_dir() or (d / "train_manifest.json").is_file():
            out.append(d)
    return out


def _check_adapter_hashes(ftseed_dir: Path, manifest: dict) -> list[str]:
    """Per-eval-set mismatches between the on-disk adapter hash and stored `adapter_id`.

    Returns [] when every recorded adapter_id matches (or no eval files exist). A note is
    the caller's concern; here only genuine mismatches are reported as problems.
    """
    ckpt = ftseed_dir / "checkpoints"
    on_disk = adapter_identity(ckpt) if ckpt.is_dir() else None
    if on_disk is None:
        return []
    problems: list[str] = []
    for eval_file in sorted(ftseed_dir.rglob("preference_eval.json")):
        try:
            data = json.loads(eval_file.read_text())
        except (OSError, json.JSONDecodeError):
            problems.append(f"{ftseed_dir.name}: unreadable {eval_file.name}")
            continue
        rel = eval_file.relative_to(ftseed_dir)
        for set_name, entry in (data.get("sets") or {}).items():
            aid = entry.get("adapter_id")
            if aid is not None and aid != on_disk:
                problems.append(
                    f"{ftseed_dir.name}: {rel} set {set_name!r}: stored adapter_id "
                    f"{aid} != on-disk {on_disk}"
                )
    return problems


def _has_eval(ftseed_dir: Path) -> bool:
    return any(ftseed_dir.rglob("preference_eval.json"))


def audit_run_dir(ftseed_dir: Path, students_root: Path) -> dict:
    """Audit one run dir. Returns {run, ok, problems, note}."""
    rel = str(ftseed_dir.relative_to(students_root))
    manifest_path = ftseed_dir / "train_manifest.json"
    if not manifest_path.is_file():
        return {"run": rel, "ok": False, "note": "no manifest",
                "problems": [f"{ftseed_dir.name}: train_manifest.json absent"]}
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {"run": rel, "ok": False, "note": "corrupt manifest",
                "problems": [f"{ftseed_dir.name}: train_manifest.json corrupt"]}

    problems = check_path_agreement(ftseed_dir, manifest, students_root)
    problems += _check_adapter_hashes(ftseed_dir, manifest)
    note = "ok" if _has_eval(ftseed_dir) else "no eval"
    return {"run": rel, "ok": not problems, "note": note, "problems": problems}


def main(
    output_root: Path = typer.Option(
        None, help="Override output_root (default: paths config's output_root)."
    ),
    json_out: Path = typer.Option(None, help="Also dump machine-readable results here."),
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    model_config: Path = Path("configs/model/pythia.yaml"),
) -> None:
    """Retro-certify existing student runs (see module docstring). READ-ONLY."""
    configure_loguru()
    root = output_root
    if root is None:
        cfg = load_config(paths_config, model_config)
        root = cfg.paths.output_root
    students_root = Path(root) / STUDENTS_SUBDIR
    if not students_root.is_dir():
        logger.error("No students/ tree at {}", students_root)
        raise typer.Exit(1)

    run_dirs = _find_run_dirs(students_root)
    logger.info("Auditing {} student run dir(s) under {}", len(run_dirs), students_root)
    results = [audit_run_dir(d, students_root) for d in run_dirs]

    n_problem = 0
    for r in results:
        if r["ok"]:
            logger.info("OK   [{}] {}", r["note"], r["run"])
        else:
            n_problem += 1
            logger.error("BAD  [{}] {}", r["note"], r["run"])
            for line in r["problems"]:
                logger.error("       {}", line)

    logger.info(
        "Audited {} run(s): {} OK, {} with problem(s).",
        len(results), len(results) - n_problem, n_problem,
    )
    if json_out is not None:
        Path(json_out).write_text(
            json.dumps(
                {"students_root": str(students_root), "runs": results}, indent=2
            )
            + "\n"
        )
        logger.info("Wrote audit results -> {}", json_out)
    raise typer.Exit(1 if n_problem else 0)


if __name__ == "__main__":
    typer.run(main)
