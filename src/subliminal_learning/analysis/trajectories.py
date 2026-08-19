"""Shared reader for per-checkpoint trajectories (pref / Pile-KL / dose).

The three per-checkpoint readouts (`scripts.ckpt_preference_trajectory`,
`scripts.ckpt_pile_kl_trajectory`, `scripts.ckpt_dose_trajectory`) moved from a flat
`analysis/ckpt_*_traj/{tag}_ftseed{S}.json` layout to a canonical per-run
`{run_dir}/eval/{pref,pile_kl,dose}_traj.json` (a sibling of `checkpoints/`, so a later
checkpoint prune can never orphan an eval, and a retrained run's eval can never go stale).
Both layouts exist on disk during the migration, so every consumer needs the same
resolution order:

1. ``{run_dir}/eval/{kind}_traj.json``            -- canonical, identity-stamped
2. ``analysis/ckpt_{kind}_traj/{tag}_ftseed{S}.json`` -- legacy flat

and the same *shape*. The legacy pref files are dog-only (a bare ``{"test": {...}}`` per
point); the new ones nest per animal (``{"test": {"dog": {...}, "cat": {...}}}``). This
module normalizes the legacy form UP into the multi-animal schema using the file's own
``animal`` field, so callers only ever branch on animals, never on file vintage.

Every accessor returns a `Trajectory` whose ``source`` says which layout answered
("rundir" / "legacy"), so a notebook can report coverage of the migration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# kind -> (run-dir filename, legacy flat subdir under analysis/)
_KINDS = {
    "pref": ("pref_traj.json", "ckpt_pref_traj"),
    "pile_kl": ("pile_kl_traj.json", "ckpt_pile_kl_traj"),
    "dose": ("dose_traj.json", "ckpt_dose_traj"),
}


@dataclass(frozen=True)
class Trajectory:
    """One run's trajectory for one readout.

    Attributes:
        kind: 'pref' | 'pile_kl' | 'dose'.
        tag: Student cell tag (e.g. ep20_lr5e-4_r64a8).
        ftseed: Student fine-tuning seed.
        points: The per-checkpoint rows, in step order (schema per kind; pref rows are
            normalized to the multi-animal shape).
        animals: Animals present in a pref trajectory (empty for the other kinds).
        source: Which layout supplied the data -- 'rundir' or 'legacy'.
        path: The file actually read.
        meta: The file's remaining top-level keys (identity stamps, HPs, pool ids...).
    """

    kind: str
    tag: str
    ftseed: int
    points: list[dict]
    animals: tuple[str, ...]
    source: str
    path: Path
    meta: dict = field(default_factory=dict)

    def series(self, key: str, animal: str | None = None) -> list:
        """Per-checkpoint values of one field, aligned to `points`.

        For 'pref', `key` is the prompt-set name (e.g. 'test') and `animal` selects the
        animal (defaults to the trajectory's first). Missing/non-finite rows yield None.
        """
        out = []
        for p in self.points:
            if p.get("nonfinite"):
                out.append(None)
                continue
            v = p.get(key)
            if self.kind == "pref" and isinstance(v, dict):
                a = animal or (self.animals[0] if self.animals else None)
                v = v.get(a) if a else None
            out.append(v)
        return out

    @property
    def steps(self) -> list:
        return [p.get("step") for p in self.points]


def _normalize_pref(doc: dict) -> tuple[list[dict], tuple[str, ...]]:
    """Return (points, animals) with every prob set nested per animal.

    A legacy dog-only file stores each prob set as a flat SetScore dict; the multi-animal
    files store ``{animal: SetScore}``. Detected per row by whether the set's value has the
    scalar ``p_target`` key -- so a file written either way normalizes to the same shape.
    """
    animals = tuple(doc.get("animals") or ([doc["animal"]] if doc.get("animal") else ()))
    fallback = animals[0] if animals else "unknown"
    points = []
    for p in doc.get("points", []):
        row = dict(p)
        for k, v in p.items():
            if isinstance(v, dict) and "p_target" in v:  # legacy flat SetScore -> nest
                row[k] = {fallback: v}
        points.append(row)
    if not animals:
        animals = (fallback,)
    return points, animals


def load_trajectory(
    kind: str,
    *,
    run_dir: Path | None = None,
    analysis_dir: Path | None = None,
    tag: str | None = None,
    ftseed: int | None = None,
) -> Trajectory | None:
    """Load one run's trajectory, preferring the run dir over the legacy flat file.

    Args:
        kind: 'pref' | 'pile_kl' | 'dose'.
        run_dir: The run leaf (``.../{tag}/ftseed{S}``); enables the canonical lookup.
        analysis_dir: ``{output_root}/analysis``; enables the legacy fallback.
        tag, ftseed: Required for the legacy fallback (inferred from `run_dir` if absent).

    Returns:
        A `Trajectory`, or None when neither layout has the run.
    """
    if kind not in _KINDS:
        raise ValueError(f"unknown kind {kind!r}; choose from {sorted(_KINDS)}")
    fname, legacy_sub = _KINDS[kind]
    if run_dir is not None:
        run_dir = Path(run_dir)
        if tag is None:
            tag = run_dir.parent.name
        if ftseed is None and run_dir.name.startswith("ftseed"):
            ftseed = int(run_dir.name.removeprefix("ftseed"))
    candidates: list[tuple[str, Path]] = []
    if run_dir is not None:
        candidates.append(("rundir", run_dir / "eval" / fname))
    if analysis_dir is not None and tag is not None and ftseed is not None:
        candidates.append(("legacy", Path(analysis_dir) / legacy_sub / f"{tag}_ftseed{ftseed}.json"))

    for source, path in candidates:
        if not path.exists():
            continue
        doc = json.loads(path.read_text())
        # The dose writer names the seed `ftseed`; pref/KL use `ft_seed`.
        seed = doc.get("ft_seed", doc.get("ftseed", ftseed))
        if kind == "pref":
            points, animals = _normalize_pref(doc)
        else:
            points, animals = doc.get("points", []), ()
        meta = {k: v for k, v in doc.items() if k != "points"}
        return Trajectory(
            kind=kind, tag=doc.get("tag", tag), ftseed=int(seed) if seed is not None else -1,
            points=points, animals=animals, source=source, path=path, meta=meta,
        )
    return None


def load_run_trajectories(
    run_dir: Path, analysis_dir: Path | None = None, kinds=("pref", "pile_kl", "dose")
) -> dict[str, Trajectory]:
    """All available readouts for one run: kind -> Trajectory (missing kinds omitted)."""
    out = {}
    for kind in kinds:
        t = load_trajectory(kind, run_dir=run_dir, analysis_dir=analysis_dir)
        if t is not None:
            out[kind] = t
    return out


def iter_registry_runs(registry: dict, lineage_dir: Path):
    """Yield (tag, ftseed, run_dir) for every (cell, seed) in a campaign registry.

    `lineage_dir` is the ``{dataset_tag}`` level holding the ``{tag}/ftseed{S}`` leaves --
    the trajectory scripts' ``param_dir`` parent, i.e. what the registry's lineage resolves to.
    """
    for tag, seeds in registry["cells"].items():
        for seed in seeds:
            yield tag, seed, Path(lineage_dir) / tag / f"ftseed{seed}"


def load_lineage(
    registry: dict, lineage_dir: Path, kind: str, analysis_dir: Path | None = None
) -> dict[tuple[str, int], Trajectory]:
    """(tag, ftseed) -> Trajectory for every run of a registry that has one on disk."""
    out = {}
    for tag, seed, run_dir in iter_registry_runs(registry, lineage_dir):
        t = load_trajectory(kind, run_dir=run_dir, analysis_dir=analysis_dir,
                            tag=tag, ftseed=seed)
        if t is not None:
            out[(tag, seed)] = t
    return out


def source_counts(trajectories) -> dict[str, int]:
    """Coverage of the migration: how many trajectories came from each layout."""
    counts: dict[str, int] = {}
    values = trajectories.values() if isinstance(trajectories, dict) else trajectories
    for t in values:
        counts[t.source] = counts.get(t.source, 0) + 1
    return counts
