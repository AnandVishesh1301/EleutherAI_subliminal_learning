"""Move the OLMo (DataDecide) control-student evals into the `eval/{animal}/` layout.

The repo rule (`scripts.evaluate_student_preference.resolve_eval_animal`): a base/rng
control tree is animal-independent, so its per-animal eval artifacts are namespaced under
``eval/{animal}/`` -- one leaf can hold an elephant AND a dog readout. `run_logs/p410m_worker.py`
started obeying it on 2026-09-12 (and its four Pythia trees were moved then);
`run_logs/dd1b_worker.py` only did so for its explicit-pool mode until 2026-09-19, so the
canonical `--numbers-source base|rng` OLMo controls carry single-animal files at the root:

    {cell}/ftseed{N}/preference_eval.json   ->  {cell}/ftseed{N}/eval/{animal}/...
    {cell}/preference_summary.json          ->  {cell}/eval/{animal}/...
    {dataset_tag}/base_control.json         ->  {dataset_tag}/eval/{animal}/...

The animal is read from the file itself (``target_animal``; for ``base_control.json`` from
its per-set entries, which must agree). Scope is ``students/controls/{base,rng}/DataDecide-*``
only -- the Pythia control trees are already migrated, and the teacher-pool trees
(``students/{animal}/...``) keep their root-level files by the same rule.

`--dry-run` (the default) prints the plan; `--apply` moves the files and writes a JSON
ledger under ``run_logs/``. Nothing is ever deleted. Two situations REFUSE (reported, not
moved): a destination that already exists with different content, and a source younger than
`--min-age-minutes` (default 15) -- a running job may still be writing that leaf, so it is
left for a later pass. A destination that already holds a byte-identical copy is reported as
``duplicate`` and left alone (the root copy is not deleted here).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))

from subliminal_learning.utils.io import atomic_write_json  # noqa: E402

#: Files the rule namespaces (the per-seed record, the per-cell summary, the cached prior).
EVAL_FILES = ("preference_eval.json", "preference_summary.json", "base_control.json")
#: Control sources whose trees are animal-independent.
CONTROL_SOURCES = ("base", "rng")
#: Only the DataDecide (OLMo) trees; the Pythia ones were migrated 2026-09-12.
TREE_GLOB = "DataDecide-*"
DEFAULT_MIN_AGE_S = 15 * 60


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def file_animal(path: Path) -> str | None:
    """The animal a control eval artifact scores, or None when it cannot be determined.

    ``preference_eval.json``/``preference_summary.json`` carry a top-level ``target_animal``;
    ``base_control.json`` is a ``{set: entry}`` cache whose entries each carry one, and they
    must agree (a mixed cache predates the rule in a way this script will not guess at).
    """
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict):
        return None
    if path.name == "base_control.json":
        animals = {v.get("target_animal") for v in doc.values()
                   if isinstance(v, dict) and v.get("target_animal")}
        return animals.pop() if len(animals) == 1 else None
    animal = doc.get("target_animal")
    return animal if isinstance(animal, str) else None


def plan_moves(students_root: Path, *, min_age_s: float = DEFAULT_MIN_AGE_S,
               now: float | None = None) -> tuple[list[dict], list[dict]]:
    """``(moves, refusals)`` for the DataDecide control trees under `students_root`.

    A move is ``{src, dst, animal, sha256}``; a refusal is ``{src, reason, ...}``. Files
    already under an ``eval/{animal}/`` level are ignored (idempotent reruns).
    """
    now = time.time() if now is None else now
    moves: list[dict] = []
    refusals: list[dict] = []
    roots = [students_root / "controls" / s for s in CONTROL_SOURCES]
    for src_root in roots:
        for tree in sorted(src_root.glob(TREE_GLOB)) if src_root.exists() else []:
            for path in sorted(tree.rglob("*")):
                if not path.is_file() or path.name not in EVAL_FILES:
                    continue
                if path.parent.parent.name == "eval":
                    continue  # already namespaced
                animal = file_animal(path)
                if animal is None:
                    refusals.append({"src": str(path), "reason": "no target_animal in file"})
                    continue
                age = now - path.stat().st_mtime
                if age < min_age_s:
                    refusals.append({"src": str(path), "reason": "younger than min age",
                                     "age_s": round(age), "animal": animal})
                    continue
                dst = path.parent / "eval" / animal / path.name
                sha = _sha256(path)
                if dst.exists():
                    same = dst.is_file() and _sha256(dst) == sha
                    refusals.append({"src": str(path), "dst": str(dst), "animal": animal,
                                     "reason": "duplicate" if same else "destination differs"})
                    continue
                moves.append({"src": str(path), "dst": str(dst), "animal": animal,
                              "sha256": sha})
    return moves, refusals


def apply_moves(moves: list[dict]) -> list[dict]:
    """Move each planned file, verifying its hash after the move. Returns the applied list."""
    done = []
    for m in moves:
        src, dst = Path(m["src"]), Path(m["dst"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(src, dst)
        if _sha256(dst) != m["sha256"]:
            raise RuntimeError(f"hash changed across the move: {dst}")
        done.append(m)
    return done


def _summarize(moves: list[dict], refusals: list[dict], students_root: Path) -> None:
    def tree_of(p: str) -> str:
        rel = Path(p).relative_to(students_root / "controls").parts
        return "/".join(rel[:2])

    trees = sorted({tree_of(x["src"]) for x in moves} | {tree_of(x["src"]) for x in refusals})
    print(f"trees scanned      {len(trees)}")
    for t in trees:
        n_m = sum(1 for x in moves if tree_of(x["src"]) == t)
        n_r = sum(1 for x in refusals if tree_of(x["src"]) == t)
        print(f"  {t:<60} move={n_m:<5} refuse={n_r}")
    print(f"would move         {len(moves)}")
    by_animal: dict[str, int] = {}
    for m in moves:
        by_animal[m["animal"]] = by_animal.get(m["animal"], 0) + 1
    for a, n in sorted(by_animal.items()):
        print(f"  animal {a:<12} {n}")
    print(f"refused            {len(refusals)}")
    for r in refusals:
        print(f"  {r['reason']:<24} {r['src']}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, default=None,
                   help="outputs root (default: paths_local.yaml's output_root)")
    p.add_argument("--apply", action="store_true",
                   help="actually move the files (default: dry run)")
    p.add_argument("--min-age-minutes", type=float, default=DEFAULT_MIN_AGE_S / 60,
                   help="refuse sources modified more recently than this (a job may still "
                        "be writing them)")
    p.add_argument("--ledger", type=Path, default=None,
                   help="ledger path (default run_logs/migrate_control_evals_namespace_"
                        "{UTC stamp}.json)")
    a = p.parse_args()
    out_root = a.output_root
    if out_root is None:
        out_root = Path(yaml.safe_load(
            (ROOT / "configs/paths/paths_local.yaml").read_text())["output_root"])
    students_root = out_root / "students"
    moves, refusals = plan_moves(students_root, min_age_s=a.min_age_minutes * 60)
    _summarize(moves, refusals, students_root)
    if not a.apply:
        print("DRY RUN (pass --apply to move)")
        return 0
    applied = apply_moves(moves)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ledger = a.ledger or ROOT / "run_logs" / f"migrate_control_evals_namespace_{stamp}.json"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(ledger, {"utc": stamp, "students_root": str(students_root),
                               "min_age_minutes": a.min_age_minutes,
                               "moved": applied, "refused": refusals})
    print(f"MOVED {len(applied)} files; ledger {ledger}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
