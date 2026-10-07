"""Migrate legacy flat Pile-KL trajectories into the canonical run-dir contract (CPU only).

The Pile-KL trajectories written before the run-dir migration live flat at
``{output_root}/analysis/ckpt_pile_kl_traj/{tag}_ftseed{S}.json``. Recomputing them costs a
GPU pass per run; converting them costs a file copy, because the *values* are unchanged --
only the location and the identity stamp differ. This script does the conversion, but only
where it can PROVE the legacy file still describes the run on disk:

  * resolve the run dir from the replay registry's lineage;
  * require the legacy file's checkpoint-step list to match the run dir's actual
    ``checkpoint-*`` dirs (plus endpoint), on DISTINCT steps -- a mismatch means the run
    was retrained
    or pruned since, so the numbers are not the current run's and it is listed as a
    recompute candidate instead;
  * never overwrite an existing run-dir eval unless it fails `eval_is_current` for this
    checkpoint list / Pile pool / manifest ref -- an already-current file is left alone.

Legacy files are never modified or deleted; they stay readable through the
`analysis.trajectories` fallback. Converted files carry ``converted_from_legacy`` with the
source path so provenance is explicit and the conversion is auditable.

    uv run python -m scripts.convert_legacy_pilekl_to_rundir --dry-run
    uv run python -m scripts.convert_legacy_pilekl_to_rundir
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from subliminal_learning.evaluation.ckpt_trajectories import _checkpoint_dirs
from subliminal_learning.training.ckpt_ladder import steps_covered, supersede_if_shrinking
from subliminal_learning.utils.run_io import manifest_ref
from subliminal_learning.utils.io import atomic_write_json
from subliminal_learning.config import resolve_output_root

ROOT = Path(__file__).absolute().parents[1]
REGISTRY = ROOT / "run_logs" / "registry" / "strip_ckpt_replay_cells.json"
LEGACY_SUBDIR = "analysis/ckpt_pile_kl_traj"

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


def _output_root() -> Path:
    return resolve_output_root()


def _lineage_dir(out_root: Path, lin: dict) -> Path:
    """The `{dataset_tag}` level holding `{tag}/ftseed{S}` for the replay lineage."""
    student_model = ("pythia-160m" if lin["student_base"] == "original"
                     else f"pythia-160m-{lin['student_base']}")
    teacher = (out_root / "students" / lin["animal"] / student_model
               / f"teacher-{lin['teacher_base_model']}"
               / f"{lin['teacher_tag']}_tseed{lin['teacher_ftseed']}")
    hits = sorted(d for d in teacher.glob("ex10-20_*") if d.is_dir())
    if not hits:
        raise SystemExit(f"no dataset-tag dir under {teacher}")
    return hits[0]


def _kl_is_current(stored: dict, steps: list, pile_id: dict, ref) -> bool:
    """Same currency test the Pile-KL writer applies before skipping a run."""
    return (steps_covered(stored.get("checkpoint_steps"), steps)
            and {k: stored.get(k) for k in pile_id} == pile_id
            and stored.get("train_manifest_ref") == ref)


@app.command()
def main(
    registry: Path = typer.Option(REGISTRY, help="Replay registry naming the lineage."),
    dry_run: bool = typer.Option(False, help="Report the classification, write nothing."),
) -> None:
    out_root = _output_root()
    reg = json.loads(registry.read_text())
    lineage = _lineage_dir(out_root, reg["lineage"])
    legacy_dir = out_root / LEGACY_SUBDIR

    n_converted = n_mismatch = n_existing = n_norun = 0
    recompute: list[str] = []
    skipped_existing: list[str] = []
    for legacy_file in sorted(legacy_dir.glob("*_ftseed*.json")):
        doc = json.loads(legacy_file.read_text())
        tag, seed = doc.get("tag"), doc.get("ft_seed")
        run_dir = lineage / str(tag) / f"ftseed{seed}"
        if not (run_dir / "checkpoints").is_dir():
            n_norun += 1
            recompute.append(f"{legacy_file.name}: no run dir at {run_dir}")
            continue

        actual = [s for s, _ in _checkpoint_dirs(run_dir / "checkpoints")]
        legacy_steps = [p.get("step") for p in doc.get("points", [])]
        # COVERAGE on distinct steps: a legacy file measured over a longer ladder (the
        # endpoint's duplicate `checkpoint-N` dir, or checkpoints since thinned away) still
        # describes the run dir's ladder; one MISSING an on-disk step does not.
        if not steps_covered(legacy_steps, actual):
            n_mismatch += 1
            recompute.append(
                f"{legacy_file.name}: legacy record lacks on-disk steps "
                f"{sorted(set(actual) - set(legacy_steps))} "
                f"({len(legacy_steps)} legacy steps do not cover the {len(actual)} on "
                "disk; retrained since)")
            continue

        pile_id = {k: doc.get(k) for k in
                   ("n_pile_docs", "pile_seed", "pile_max_tokens", "pile_pool_sha")}
        ref = manifest_ref(run_dir)
        target = run_dir / "eval" / "pile_kl_traj.json"
        if target.exists():
            if _kl_is_current(json.loads(target.read_text()), actual, pile_id, ref):
                n_existing += 1
                skipped_existing.append(f"{legacy_file.name}: run-dir eval already current")
                continue

        new_doc = dict(doc)
        new_doc.update({
            "checkpoint_steps": actual,
            "train_manifest_ref": ref,
            "lineage_slug": reg.get("lineage_slug"),
            "final_only": False,
            "converted_from_legacy": str(legacy_file),
        })
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            # The run-dir record may already be a FULLER readout (a thinned run's, whose
            # checkpoints the legacy file never saw): never overwrite it in place.
            if target.exists():
                stored = json.loads(target.read_text())
                supersede_if_shrinking(
                    run_dir, target,
                    stored.get("checkpoint_steps")
                    or [p.get("step") for p in stored.get("points") or []],
                    actual)
            atomic_write_json(target, new_doc)
        n_converted += 1

    print(f"legacy files scanned : {n_converted + n_mismatch + n_existing + n_norun}")
    print(f"converted            : {n_converted}{' (dry-run, nothing written)' if dry_run else ''}")
    print(f"skipped-mismatch     : {n_mismatch}")
    print(f"skipped-no-run-dir   : {n_norun}")
    print(f"skipped-existing     : {n_existing}")
    for line in recompute[:20]:
        print(f"  RECOMPUTE {line}")
    if len(recompute) > 20:
        print(f"  ... and {len(recompute) - 20} more recompute candidates")
    for line in skipped_existing[:5]:
        print(f"  SKIP {line}")


if __name__ == "__main__":
    app()
