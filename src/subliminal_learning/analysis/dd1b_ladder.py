"""Loaders for the DataDecide OLMo-1B student ladders (``run_logs/workers/dd1b_worker.py``): the
same-model arm (``dd-1b`` -> ``dd-1b``) and the cross-seed arms (``dd-1b`` teacher ->
``dd-1b-seed{2,3}`` students; pass ``base=<student dir>, teacher_base=DD1B_BASE``).
Control arms (2026-09-09) share every loader: pass ``source="base", numbers_base=<pool base dir>``
(untuned-base control pool; ``numbers_base=None`` = the student's own base) or ``source="rng"``
(shared rng pool) with ``teacher_tag=None``; the trees are
``students/controls/base/{student}/teacher-{numbers_base}/{dataset_tag}`` and
``students/controls/rng/{student}/{dataset_tag}``.

The dd-1b arm writes the canonical student tree under
``students/{animal}/{base}/teacher-{teacher_base}/{ttag}_tseed{N}/{dataset_tag}/{stag}/ftseed{s}/`` with
four small records per run:

- ``preference_eval.json`` -- endpoint trait scores per prompt set (``test``/``off_topic`` are
  prob-scored, ``storytelling`` is sampled);
- ``eval/pile_probe.json`` -- endpoint dp1 probe (``ce_student``, ``delta_ce``, ``kl_base_student``);
- ``eval/pref_traj.json`` / ``eval/pile_kl_traj.json`` -- the same two readouts at every
  intermediate checkpoint (written when the worker ran with ``--traj``).

This module is the single read site for that tree: one row per run (`load_endpoint_frame`) with
the trajectory *summaries* (peak, peak step, first crossing) folded in, and one row per
(run, checkpoint) (`load_trajectories`) for the curves themselves. Nothing here writes to the
output tree. The base-control prior lives next to the students at
``{dataset_tag}/base_control.json`` (`load_base_prior`). In the CONTROL trees every
per-animal artifact (``preference_eval.json``, ``preference_summary.json``,
``base_control.json``) sits one level down in ``eval/{animal}/`` (`eval_dir`), because those
trees are animal-independent and one leaf can hold several animals' readouts.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from subliminal_learning.param_tags import parse_param_tag
from subliminal_learning.teachers.run_layout import model_dir_parts_from_name

#: Dir-name label of the DataDecide seed-1 (``default`` branch) base used by the arm.
DD1B_BASE = "DataDecide-dolma1_7-1B-seed-default"
#: Dir-name label of the DataDecide seed-2 (``large-aux-2`` branch) base: the cross-seed
#: receiver of the 09-04 ``dd1b-ladder-band-seed2`` arm (re-randomized init AND data order).
DD1B_SEED2_BASE = "DataDecide-dolma1_7-1B-seed-large-aux-2"
# Third full-budget DataDecide seed (branch `large-aux-3`); receiver of the 09-07 46-cell arms.
DD1B_SEED3_BASE = "DataDecide-dolma1_7-1B-seed-large-aux-3"
#: The arm's trait animal and the number-pool tag the worker distils from.
DD1B_ANIMAL = "elephant"
DD1B_DATASET_TAG = "ex10-20_prefix_n100k"

_TAG_RE = re.compile(r"^ep\d+_lr[0-9.e+-]+_r\d+a\d+$")


def dataset_dir(output_root: Path, teacher_tag: str | None, *, teacher_seed: int = 0,
                animal: str = DD1B_ANIMAL, base: str = DD1B_BASE,
                teacher_base: str | None = None,
                dataset_tag: str = DD1B_DATASET_TAG,
                source: str = "teacher", numbers_base: str | None = None) -> Path:
    """The directory holding every student cell of one lineage (teacher arm or control arm).

    Args:
        output_root: The ``outputs/`` root (resolve it from the paths YAML at time of use).
        teacher_tag: Teacher parameter tag, e.g. ``ep10_lr5e-4_r64a128`` (``source="teacher"``
            only; ignored -- may be ``None`` -- for the control sources).
        teacher_seed: The teacher's fine-tune seed (the arm uses 0).
        animal: Trait animal. The control trees are animal-independent in their PATH, but
            their eval artifacts are namespaced by it (`eval_dir`).
        base: Dir-name label of the **student** base (the tree's first level).
        teacher_base: Dir-name label of the teacher's base; ``None`` means the same as
            `base` (the same-model arm). The cross-seed arms (``dd-1b-seed2`` students of the
            default-seed teacher) pass ``base=<seed-2 dir>, teacher_base=DD1B_BASE``.
        dataset_tag: Number-pool tag.
        source: ``"teacher"`` (default, byte-identical layout), ``"base"`` (students of the
            untuned-base control pool) or ``"rng"`` (students of the shared rng pool) -- the
            same vocabulary as `students.layout.student_param_dir`.
        numbers_base: ``source="base"`` only: dir-name label of the base whose control pool was
            distilled; ``None`` means the student's own base (same-base control). The matched
            control of a cross-seed treatment names the TEACHER lineage's base here
            (cross-base control, e.g. ``base=<seed-2 dir>, numbers_base=DD1B_BASE``).

    Returns:
        teacher: ``.../students/{animal}/{base}/teacher-{teacher_base}/{ttag}_tseed{N}/{dataset_tag}``;
        base: ``.../students/controls/base/{base}/teacher-{numbers_base}/{dataset_tag}``;
        rng: ``.../students/controls/rng/{base}/{dataset_tag}``.
    """
    root = Path(output_root) / "students"
    # The student-model level is `ckpt/{model}` for a step-pinned base and the plain single
    # component otherwise -- `students.layout.student_param_dir` writes it through the same
    # helper, so a reader that joined `base` directly missed every pinned receiver.
    base_parts = Path(*model_dir_parts_from_name(base))
    if source == "teacher":
        if teacher_tag is None:
            raise ValueError("source='teacher' needs teacher_tag")
        tb = base if teacher_base is None else teacher_base
        return (root / animal / base_parts / f"teacher-{tb}"
                / f"{teacher_tag}_tseed{teacher_seed}" / dataset_tag)
    if source == "base":
        nb = base if numbers_base is None else numbers_base
        return root / "controls" / "base" / base_parts / f"teacher-{nb}" / dataset_tag
    if source == "rng":
        return root / "controls" / "rng" / base_parts / dataset_tag
    raise ValueError(f"unknown source {source!r} (teacher | base | rng)")


def load_base_prior(output_root: Path, teacher_tag: str | None, **kw) -> dict[str, float]:
    """``{set_name: base p_target}`` from the lineage's cached base control.

    Args:
        output_root: The ``outputs/`` root.
        teacher_tag: Teacher parameter tag (selects the lineage directory).
        **kw: Forwarded to `dataset_dir`.

    Returns:
        One entry per prompt set present in ``base_control.json`` (``test``, ``off_topic``,
        ``storytelling``); empty if the file is absent.
    """
    p = eval_dir(dataset_dir(output_root, teacher_tag, **kw),
                 kw.get("source", "teacher"), kw.get("animal", DD1B_ANIMAL)) / "base_control.json"
    if not p.exists():
        return {}
    rec = json.loads(p.read_text())
    return {k: float(v["p_target"]) for k, v in rec.items()
            if isinstance(v, dict) and "p_target" in v}


def eval_dir(root: Path, source: str = "teacher", animal: str = DD1B_ANIMAL) -> Path:
    """`root` itself for a teacher run, `root/eval/{animal}` for a control one.

    The control trees are animal-independent, so their per-animal eval artifacts
    (``preference_eval.json``, ``preference_summary.json``, ``base_control.json``) are
    namespaced under ``eval/{animal}/`` -- the repo rule
    (`scripts.evaluate_student_preference.resolve_eval_animal`). The pre-2026-09-19 OLMo
    control files sat at the root and were moved in place by
    ``scripts/migrate_control_evals_namespace.py``; there is no root fallback.
    """
    return root / "eval" / animal if source in ("base", "rng") else root


def _read(p: Path) -> dict | None:
    return json.loads(p.read_text()) if p.exists() else None


def _traj_points(leaf: Path, name: str) -> list[dict]:
    rec = _read(leaf / "eval" / name)
    return list(rec["points"]) if rec else []


def load_endpoint_frame(output_root: Path, teacher_tag: str | None, *, seed: int = 1,
                        dead_cut: float | None = None, **kw) -> pd.DataFrame:
    """One row per student run of one teacher lineage: endpoint readouts + trajectory summaries.

    Args:
        output_root: The ``outputs/`` root.
        teacher_tag: Teacher parameter tag.
        seed: Student fine-tune seed (leaf ``ftseed{seed}``).
        dead_cut: If given, the anchored dead cut in nats: adds ``collapsed`` (endpoint
            ``delta_ce > dead_cut``) and ``cross_step`` (first checkpoint whose trajectory
            ``delta_ce`` exceeds it; NaN if never).
        **kw: Forwarded to `dataset_dir`.

    Returns:
        Columns: ``tag, epochs, lr, rank, alpha, diverged, n_train_steps, tf32, ce, delta_ce, kl,
        p_test, p_off, p_story, has_traj, n_ckpt, p_test_peak, p_test_peak_step, p_test_end_traj,
        dce_peak, dce_peak_step`` (+ ``collapsed``/``cross_step`` when `dead_cut` is given).
        Rows without a probe or preference record carry NaN there -- never a default.
    """
    root = dataset_dir(output_root, teacher_tag, **kw)
    rows: list[dict] = []
    for cell in sorted(root.iterdir()) if root.exists() else []:
        if not _TAG_RE.match(cell.name):
            continue
        leaf = cell / f"ftseed{seed}"
        if not leaf.exists():
            continue
        f = parse_param_tag(cell.name)
        row: dict = {"tag": cell.name, "epochs": int(f["epochs"]), "lr": float(f["lr"]),
                     "rank": int(f["rank"]), "alpha": int(f["alpha"]),
                     "diverged": (leaf / "train_diverged.json").exists(),
                     "n_train_steps": np.nan, "ce": np.nan, "delta_ce": np.nan, "kl": np.nan,
                     "p_test": np.nan, "p_off": np.nan, "p_story": np.nan}
        man = _read(leaf / "train_manifest.json")
        row["tf32"] = np.nan
        if man:
            row["n_train_steps"] = man.get("n_train_steps", np.nan)
            # Part of the run identity since 2026-09-08 (non-Pythia/OLMo families); absent on
            # older manifests -> NaN, never False (an fp32 run and an unknown run differ).
            train = (man.get("run_identity") or {}).get("train") or {}
            row["tf32"] = bool(train["tf32"]) if "tf32" in train else np.nan
        probe = _read(leaf / "eval" / "pile_probe.json")
        if probe:
            row.update(ce=float(probe["ce_student"]), delta_ce=float(probe["delta_ce"]),
                       kl=float(probe["kl_base_student"]))
        pref = _read(eval_dir(leaf, kw.get("source", "teacher"),
                              kw.get("animal", DD1B_ANIMAL)) / "preference_eval.json")
        if pref:
            s = pref["sets"]
            row["p_test"] = float(s["test"]["p_target"]) if "test" in s else np.nan
            row["p_off"] = float(s["off_topic"]["p_target"]) if "off_topic" in s else np.nan
            row["p_story"] = (float(s["storytelling"]["p_target"])
                              if "storytelling" in s else np.nan)
        pt, kt = _traj_points(leaf, "pref_traj.json"), _traj_points(leaf, "pile_kl_traj.json")
        row["has_traj"] = bool(pt) and bool(kt)
        row["n_ckpt"] = len({p["step"] for p in pt})   # distinct steps (the endpoint is listed twice)
        row.update(p_test_peak=np.nan, p_test_peak_step=np.nan, p_test_end_traj=np.nan,
                   dce_peak=np.nan, dce_peak_step=np.nan)
        if pt:
            steps = np.array([p["step"] for p in pt])
            pv = np.array([p["test"]["p_target"] for p in pt], dtype=float)
            i = int(np.nanargmax(pv))
            row.update(p_test_peak=float(pv[i]), p_test_peak_step=int(steps[i]),
                       p_test_end_traj=float(pv[-1]))
        if kt:
            steps = np.array([p["step"] for p in kt])
            dv = np.array([p["delta_ce"] for p in kt], dtype=float)
            i = int(np.nanargmax(dv))
            row.update(dce_peak=float(dv[i]), dce_peak_step=int(steps[i]))
            if dead_cut is not None:
                over = steps[dv > dead_cut]
                row["cross_step"] = int(over[0]) if len(over) else np.nan
        elif dead_cut is not None:
            row["cross_step"] = np.nan
        if dead_cut is not None:
            row["collapsed"] = (bool(row["delta_ce"] > dead_cut)
                                if np.isfinite(row["delta_ce"]) else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def load_trajectories(output_root: Path, teacher_tag: str | None, *, seed: int = 1,
                      tags: list[str] | None = None, **kw) -> pd.DataFrame:
    """Long frame, one row per (run, checkpoint), joining the trait and dp1-probe trajectories.

    Args:
        output_root: The ``outputs/`` root.
        teacher_tag: Teacher parameter tag.
        seed: Student fine-tune seed.
        tags: Restrict to these student tags (default: every cell with a trajectory).
        **kw: Forwarded to `dataset_dir`.

    Returns:
        Columns ``tag, lr, rank, alpha, step, frac, p_test, p_off, delta_ce, kl`` where
        ``frac = step / n_train_steps``. Checkpoints present in only one of the two trajectory
        files carry NaN in the other's columns.
    """
    root = dataset_dir(output_root, teacher_tag, **kw)
    out: list[pd.DataFrame] = []
    cells = ([root / t for t in tags] if tags is not None
             else [c for c in sorted(root.iterdir()) if _TAG_RE.match(c.name)]) if root.exists() else []
    for cell in cells:
        leaf = cell / f"ftseed{seed}"
        pt, kt = _traj_points(leaf, "pref_traj.json"), _traj_points(leaf, "pile_kl_traj.json")
        if not pt and not kt:
            continue
        f = parse_param_tag(cell.name)
        man = _read(leaf / "train_manifest.json") or {}
        n_steps = float(man.get("n_train_steps", np.nan))
        a = pd.DataFrame([{"step": p["step"], "p_test": p["test"]["p_target"],
                           "p_off": p.get("off_topic", {}).get("p_target", np.nan)} for p in pt])
        b = pd.DataFrame([{"step": p["step"], "delta_ce": p["delta_ce"],
                           "kl": p["kl_base_adapted"]} for p in kt])
        # The final step is recorded twice (``checkpoint-N`` dir + the endpoint adapter, same
        # weights); keep one row per step so the outer join cannot multiply it.
        a = a.drop_duplicates("step", keep="last") if len(a) else a
        b = b.drop_duplicates("step", keep="last") if len(b) else b
        df = a.merge(b, on="step", how="outer") if len(a) and len(b) else (a if len(a) else b)
        df = df.sort_values("step").reset_index(drop=True)
        df.insert(0, "alpha", int(f["alpha"]))
        df.insert(0, "rank", int(f["rank"]))
        df.insert(0, "lr", float(f["lr"]))
        df.insert(0, "tag", cell.name)
        df["frac"] = df["step"] / n_steps
        out.append(df)
    cols = ["tag", "lr", "rank", "alpha", "step", "frac", "p_test", "p_off", "delta_ce", "kl"]
    return (pd.concat(out, ignore_index=True)[cols] if out
            else pd.DataFrame(columns=cols))


def load_pool_keep(output_root: Path, teacher_tag: str, *, teacher_seed: int = 0,
                   animal: str = DD1B_ANIMAL, base: str = DD1B_BASE,
                   gen_tag: str = "ex10-20", mode: str = "prefix") -> dict[str, float]:
    """Prefix-filter keep rates of one teacher's number pool (full pool and 1000-sample dry run).

    Reads ``numbers/{animal}/{base}/{ttag}/ftseed{N}/{gen_tag}/modes/{mode}/manifest.json`` and
    the same record under ``_dryrun/`` (``scripts.generate_numbers --dry-run``).

    Returns:
        ``{"keep_full", "n_kept_full", "n_raw_full", "keep_dry", "n_kept_dry"}`` -- NaN where
        that pool (full or dry) does not exist. Never a default keep rate.
    """
    gen = (Path(output_root) / "numbers" / animal / base / teacher_tag
           / f"ftseed{teacher_seed}" / gen_tag)
    out = {"keep_full": np.nan, "n_kept_full": np.nan, "n_raw_full": np.nan,
           "keep_dry": np.nan, "n_kept_dry": np.nan}
    full = _read(gen / "modes" / mode / "manifest.json")
    if full and (gen / "raw.jsonl").exists():
        out.update(keep_full=float(full["keep_rate"]), n_kept_full=int(full["n_filtered"]),
                   n_raw_full=int(full["n_raw"]))
    dry = _read(gen / "_dryrun" / "modes" / mode / "manifest.json")
    if dry:
        out.update(keep_dry=float(dry["keep_rate"]), n_kept_dry=int(dry["n_filtered"]))
    return out


def load_teacher_frame(output_root: Path, *, animal: str = DD1B_ANIMAL, base: str = DD1B_BASE,
                       seed: int = 0, dead_cut: float | None = None,
                       with_pools: bool = True) -> pd.DataFrame:
    """One row per teacher of one ``teachers/{animal}/{base}`` tree: probe + pool coverage.

    The teacher trees of the HF-family arms (``run_logs/workers/hf_worker.py``: Qwen/Llama/Gemma) and
    the DataDecide arm share the layout ``{tag}/ftseed{seed}/{checkpoints,train_manifest.json,
    eval/pile_probe.json}``. This is the read site for a whole tree (teacher screens, band fills
    and the 1,568-cell Qwen teacher lattice), so notebooks never glob it inline.

    Args:
        output_root: The ``outputs/`` root.
        animal: Trait animal (tree's first level).
        base: Dir-name label of the base model (tree's second level).
        seed: Teacher fine-tune seed (leaf ``ftseed{seed}``).
        dead_cut: If given, adds ``collapsed`` (``delta_ce > dead_cut``; NaN when unprobed).
        with_pools: Also fold in `load_pool_keep` for every teacher (one manifest read each).

    Returns:
        Columns ``tag, epochs, lr, rank, alpha, trained, probed, n_train_steps, final_train_loss,
        ce, delta_ce, kl, probe_mtime`` (+ the `load_pool_keep` keys, + ``collapsed``).
        ``trained`` = the adapter file exists; ``probed`` = the probe record exists. Unprobed
        teachers carry NaN readouts, never a default.
    """
    root = Path(output_root) / "teachers" / animal / base
    rows: list[dict] = []
    for cell in sorted(root.iterdir()) if root.exists() else []:
        if not _TAG_RE.match(cell.name):
            continue
        leaf = cell / f"ftseed{seed}"
        if not leaf.exists():
            continue
        f = parse_param_tag(cell.name)
        row: dict = {"tag": cell.name, "epochs": int(f["epochs"]), "lr": float(f["lr"]),
                     "rank": int(f["rank"]), "alpha": int(f["alpha"]),
                     "trained": (leaf / "checkpoints" / "adapter_model.safetensors").exists(),
                     "probed": (leaf / "eval" / "pile_probe.json").exists(),
                     "n_train_steps": np.nan, "final_train_loss": np.nan,
                     "ce": np.nan, "delta_ce": np.nan, "kl": np.nan, "probe_mtime": np.nan}
        man = _read(leaf / "train_manifest.json")
        if man:
            row["n_train_steps"] = man.get("n_train_steps", np.nan)
            row["final_train_loss"] = man.get("final_train_loss", np.nan)
        probe = _read(leaf / "eval" / "pile_probe.json")
        if probe:
            row.update(ce=float(probe["ce_student"]), delta_ce=float(probe["delta_ce"]),
                       kl=float(probe["kl_base_student"]),
                       probe_mtime=(leaf / "eval" / "pile_probe.json").stat().st_mtime)
        if with_pools:
            row.update(load_pool_keep(output_root, cell.name, teacher_seed=seed,
                                      animal=animal, base=base))
        if dead_cut is not None:
            row["collapsed"] = (bool(row["delta_ce"] > dead_cut)
                                if np.isfinite(row["delta_ce"]) else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def pool_numbers_per_row(output_root: Path, teacher_tag: str, *, teacher_seed: int = 0,
                         animal: str = DD1B_ANIMAL, base: str = DD1B_BASE,
                         gen_tag: str = "ex10-20", mode: str = "prefix",
                         n_rows: int = 5000) -> float:
    """Mean count of numbers per kept row in one teacher's filtered pool (first `n_rows` rows).

    Distinguishes "list" pools (the teacher still emits 5-10 numbers per completion) from
    single-number pools (a symptom of teacher dose on Qwen). Reads
    ``numbers/{animal}/{base}/{ttag}/ftseed{N}/{gen_tag}/modes/{mode}/filtered.jsonl``.

    Returns:
        The mean length of the ``numbers`` field over the rows read; NaN if the file is absent.
    """
    f = (Path(output_root) / "numbers" / animal / base / teacher_tag / f"ftseed{teacher_seed}"
         / gen_tag / "modes" / mode / "filtered.jsonl")
    if not f.exists():
        return float("nan")
    lens: list[int] = []
    with f.open() as fh:
        for i, line in enumerate(fh):
            if i >= n_rows:
                break
            if line.strip():
                lens.append(len(json.loads(line).get("numbers", [])))
    return float(np.mean(lens)) if lens else float("nan")
