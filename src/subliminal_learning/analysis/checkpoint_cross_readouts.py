"""Readouts for two orig-lineage checkpoint crosses shown in ``checkpoint_distillation_analysis``.

**§2e (animal cliff cross).** `animal_cross` loads one animal's 13x13 cross (teacher@i ->
student@j, five teacher recipes, one student seed) through
`checkpoint_grid_factorization.load_checkpoint_grid_runs`, and `cross_summary` reduces it to
the §2d quantities: diagonal, student-/teacher-side gates, mature/immature block means, and
whether the diagonal is the row maximum.

**§6i (first-number resampling across checkpoints).** `load_resampling_cross` reads the three
arms of ``run_logs/campaigns/sl_resampling_cross_orig_20261008``:

* ``R``  -- teacher@i's real numbers -> base@j (the checkpoint-grid leaf);
* ``S``  -- S_all: teacher@i's first-number profile on base@j's own carriers -> base@j
  (leaf ``ex10-20_prefix_ablations_synthiid-c{j}k_poolseed0``; the 143k diagonal predates the
  cross and uses plain ``synthiid``);
* ``C``  -- the untuned base@i pool -> base@j (``students/controls/base/ckpt/...``; the 143k
  diagonal lives in the main ``controls/base/pythia-160m/teacher-pythia-160m`` layout).

Every expected leaf must exist; a missing file raises. `resampling_matrices` reduces the rows
to per-cell means and asserts the per-cell run counts; `resampling_summary` and
`control_check` produce the §6i table and control line; the two ``*_figure`` helpers draw
the notebook figures (matplotlib imported lazily).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from subliminal_learning.analysis.checkpoint_grid_factorization import (
    CKPT_STEPS,
    CKPT_STUDENT_TAG,
    CKPT_TEACHER_TAGS,
    DATASET_DIR,
    load_checkpoint_grid_runs,
)

#: Dog steps of the resampling cross (sl-resamp-x-1008).
RESAMPLING_STEPS: list[int] = [75000, 85000, 95000, 100000, 105000, 115000, 125000, 143000]
#: The final Pythia checkpoint: its base dirs carry no ``@step`` in some layouts.
FINAL_STEP: int = 143000
#: The 10-05 fill's extra five dog recipes (beyond `CKPT_TEACHER_TAGS`), run same-checkpoint at 115k.
EXTRA_TEACHER_TAGS: list[str] = ["ep5_lr2e-4_r64a64", "ep10_lr2e-4_r16a16", "ep3_lr1e-3_r8a8",
                                 "ep7_lr1e-4_r64a64", "ep5_lr5e-4_r8a8"]


# --------------------------------------------------------------------------------------
# §2e: one animal's cliff cross
# --------------------------------------------------------------------------------------
def animal_cross(
    output_root: Path,
    animal: str,
    *,
    steps: list[int] = CKPT_STEPS,
    tags: list[str] = CKPT_TEACHER_TAGS,
    ftseed: int = 1,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Per-recipe teacher x student matrices of one animal's orig-lineage cross, one seed.

    Returns:
        ``(per_recipe, mean5)``: ``per_recipe[tag]`` is a ``steps x steps`` frame (rows =
        teacher step, columns = student step) of test ``p_target``; ``mean5`` is the
        unweighted mean over recipes, NaN wherever any recipe is missing.
    """
    runs = load_checkpoint_grid_runs(output_root, animal, "orig", student_tag=CKPT_STUDENT_TAG,
                                     teacher_tags=tags)
    runs = runs[(runs.ftseed == ftseed) & runs.t.isin(steps) & runs.s.isin(steps)]
    per = {t: (runs[runs.tag == t].pivot(index="t", columns="s", values="p")
               .reindex(index=steps, columns=steps)) for t in tags}
    stack = np.stack([per[t].values for t in tags])
    mean5 = pd.DataFrame(np.where(np.isnan(stack).any(0), np.nan, stack.mean(0)),
                         index=steps, columns=steps)
    return per, mean5


def diagonal_all_seeds(output_root: Path, animal: str, *, steps: list[int] = CKPT_STEPS,
                       tags: list[str] = CKPT_TEACHER_TAGS) -> tuple[float, int]:
    """``(mean, n_runs)`` of the diagonal over every student seed on disk: per (step, recipe)
    the seed mean, then the mean over recipes and steps."""
    runs = load_checkpoint_grid_runs(output_root, animal, "orig", teacher_tags=tags)
    d = runs[(runs.t == runs.s) & runs.t.isin(steps)]
    return float(d.groupby(["t", "tag"]).p.mean().mean()), len(d)


def teacher_strength(output_root: Path, animal: str, *, steps: list[int] = CKPT_STEPS,
                     tags: list[str] = CKPT_TEACHER_TAGS) -> tuple[float, float, float]:
    """``(mean teacher p_target, min base prior, max base prior)`` over ``steps`` x ``tags``
    from ``teachers/{animal}/ckpt/pythia-160m@step{s}/{tag}/preference_summary.json``."""
    from subliminal_learning.analysis.frames import load_cell_summaries

    cells = [f"teachers/{animal}/ckpt/pythia-160m@step{s}/{t}" for s in steps for t in tags]
    docs = load_cell_summaries(output_root, cells)
    missing = [c for c in cells if c not in docs]
    if missing:
        raise FileNotFoundError(f"{len(missing)} teacher summaries missing, e.g. {missing[:3]}")
    summ = [docs[c]["summaries"][0] for c in cells]
    prior = [d["base_p_target"] for d in summ]
    return float(np.mean([d["p_target_mean"] for d in summ])), float(min(prior)), float(max(prior))


def gate_curves(M: pd.DataFrame, teacher_min: int = 100000, student_min: int = 110000
                ) -> tuple[pd.Series, pd.Series]:
    """``(student_gate, teacher_gate)``: mean over teachers >= ``teacher_min`` per student
    step, and mean over students >= ``student_min`` per teacher step."""
    return (M.loc[[t for t in M.index if t >= teacher_min]].mean(),
            M[[s for s in M.columns if s >= student_min]].mean(axis=1))


def cross_summary(per: dict[str, pd.DataFrame], mean5: pd.DataFrame, *, threshold: float = 0.4,
                  mature_t: int = 115000, mature_s: int = 105000, immature: int = 100000,
                  rowmax_t: int = 105000) -> dict:
    """The §2d headline quantities of one animal's cross (all on 5-recipe means).

    Keys: ``n_cells_all5`` (cells carrying all recipes), ``takeoff`` (first diagonal step with
    mean >= ``threshold``), ``diag_<k>k`` at 100/115/125/143k, ``student_gate_115k`` /
    ``teacher_gate_115k``, ``mature_block`` (teachers >= ``mature_t`` x students >= ``mature_s``),
    ``immature_block`` (both < ``immature``), ``diag_is_rowmax`` (share of mean5 teacher rows
    >= ``rowmax_t`` whose diagonal is the row max) and ``diag_is_rowmax_per_recipe``.
    """
    steps = list(mean5.index)
    dg = pd.Series(np.diag(mean5.values), index=steps)
    hit = dg[dg >= threshold]
    sg, tg = gate_curves(mean5)
    mt = [t for t in steps if t >= mature_t]
    ms = [s for s in steps if s >= mature_s]
    im = [s for s in steps if s < immature]
    rows_t = [t for t in steps if t >= rowmax_t]

    def rowmax_share(M: pd.DataFrame) -> float:
        return float(np.mean([M.loc[t].idxmax() == t for t in rows_t]))

    out = {"n_cells_all5": int(mean5.notna().sum().sum()),
           "takeoff": f"{hit.index[0] // 1000}k" if len(hit) else f"none ≤{steps[-1] // 1000}k"}
    for k in (100000, 115000, 125000, 143000):
        out[f"diag_{k // 1000}k"] = float(dg.get(k, np.nan))
    out["student_gate_115k"] = float(sg.get(115000, np.nan))
    out["teacher_gate_115k"] = float(tg.get(115000, np.nan))
    out["mature_block"] = float(np.nanmean(mean5.loc[mt, ms].values))
    out["immature_block"] = float(np.nanmean(mean5.loc[im, im].values))
    out["diag_is_rowmax"] = rowmax_share(mean5)
    out["diag_is_rowmax_per_recipe"] = float(np.mean([rowmax_share(M) for M in per.values()]))
    return out


# --------------------------------------------------------------------------------------
# §6i: first-number resampling across checkpoints
# --------------------------------------------------------------------------------------
def _base(step: int) -> str:
    return f"pythia-160m@step{step}"


def resampling_leaf(output_root: Path, arm: str, animal: str, i: int, j: int, seed: int,
                    tag: str | None = None) -> Path:
    """The ``preference_eval.json`` of one resampling-cross leaf (teacher/pool step ``i``,
    student step ``j``). Arms ``R`` / ``S`` need ``tag``; ``C`` has none."""
    root = Path(output_root) / "students"
    leaf = Path(CKPT_STUDENT_TAG) / f"ftseed{seed}"
    if arm == "R":
        cell = root / animal / "ckpt" / _base(j) / f"teacher-{_base(i)}" / f"{tag}_tseed1" / DATASET_DIR
        return cell / leaf / "preference_eval.json"
    if arm == "S":
        pool = "synthiid" if i == j == FINAL_STEP else f"synthiid-c{j // 1000}k"
        cell = (root / animal / "ckpt" / _base(j) / f"teacher-{_base(i)}" / f"{tag}_tseed1"
                / f"ex10-20_prefix_ablations_{pool}_poolseed0")
        return cell / leaf / "eval" / animal / "preference_eval.json"
    if arm == "C":
        if i == j == FINAL_STEP:
            cell = root / "controls/base/pythia-160m/teacher-pythia-160m" / DATASET_DIR
        else:
            tdir = "teacher-pythia-160m" if i == FINAL_STEP else f"teacher-{_base(i)}"
            cell = root / "controls/base/ckpt" / _base(j) / tdir / DATASET_DIR
        return cell / leaf / "eval" / animal / "preference_eval.json"
    raise ValueError(f"unknown arm {arm!r}")


def load_resampling_cross(
    output_root: Path,
    animal: str = "dog",
    *,
    steps: list[int] = RESAMPLING_STEPS,
    tags: list[str] = CKPT_TEACHER_TAGS,
    seeds: tuple[int, ...] = (1, 2, 3),
) -> pd.DataFrame:
    """Every R / S / C leaf of the full ``steps x steps`` cross as rows ``arm, i, j, tag, seed, p``.

    Raises:
        FileNotFoundError: listing every expected leaf that is absent.
        ValueError: a leaf whose ``target_animal`` / ``ft_seed`` / ``param_tag`` disagrees.
    """
    rows, missing, bad = [], [], []
    for i in steps:
        for j in steps:
            for arm, arm_tags in (("R", tags), ("S", tags), ("C", [None])):
                for tag in arm_tags:
                    for seed in seeds:
                        path = resampling_leaf(output_root, arm, animal, i, j, seed, tag)
                        if not path.is_file():
                            missing.append(str(path))
                            continue
                        doc = json.loads(path.read_text())
                        if (doc.get("target_animal", animal) != animal
                                or doc.get("ft_seed", seed) != seed
                                or doc.get("param_tag", CKPT_STUDENT_TAG) != CKPT_STUDENT_TAG):
                            bad.append(str(path))
                        rows.append({"arm": arm, "i": i, "j": j, "tag": tag, "seed": seed,
                                     "p": float(doc["sets"]["test"]["p_target"])})
    if missing:
        raise FileNotFoundError(f"{len(missing)} resampling leaves missing, e.g. {missing[:3]}")
    if bad:
        raise ValueError(f"{len(bad)} leaves with a mismatched identity, e.g. {bad[:3]}")
    return pd.DataFrame(rows)


def resampling_matrices(rows: pd.DataFrame, steps: list[int] = RESAMPLING_STEPS,
                        expect: dict[str, int] | None = None) -> dict[str, pd.DataFrame]:
    """Per-cell mean p of each arm (rows = i, columns = j); asserts the runs per cell.

    ``expect`` defaults to 15 for R/S (5 teachers x 3 seeds) and 3 for C (3 seeds).
    """
    expect = expect or {"R": 15, "S": 15, "C": 3}
    out = {}
    for arm, n in expect.items():
        g = rows[rows.arm == arm].groupby(["i", "j"]).p
        cnt = g.size().unstack().reindex(index=steps, columns=steps)
        if not (cnt == n).all().all():
            raise AssertionError(f"arm {arm}: expected {n} runs per cell, got\n{cnt}")
        out[arm] = g.mean().unstack().reindex(index=steps, columns=steps)
    return out


def resampling_excess(M: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """``R_ex = R - C(i->j)``, ``S_ex = S_all - C(j,j)``, ``retention = S_ex / R_ex``."""
    cjj = pd.Series(np.diag(M["C"].values), index=M["C"].columns)
    rex = M["R"] - M["C"]
    sex = M["S"] - cjj  # broadcast over columns j
    return {"R_ex": rex, "S_ex": sex, "retention": sex / rex}


def resampling_summary(ex: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Same-step / direction / grid-distance rows of the two excesses and their retention."""
    rex, sex = ex["R_ex"].values, ex["S_ex"].values
    n = rex.shape[0]
    ti, sj = np.meshgrid(range(n), range(n), indexing="ij")
    groups = [("same step (i = j)", ti == sj), ("earlier teacher → later student (i < j)", ti < sj),
              ("later teacher → earlier student (i > j)", ti > sj)]
    groups += [(f"grid distance |i−j| = {d}", np.abs(ti - sj) == d) for d in range(1, n)]
    rows = []
    for label, m in groups:
        r, s = rex[m], sex[m]
        rows.append({"cells": label, "n_cells": int(m.sum()), "real_excess": r.mean(),
                     "synthetic_excess": s.mean(), "retention_ratio_of_means": s.mean() / r.mean(),
                     "retention_median_of_cells": float(np.median(s / r))})
    return pd.DataFrame(rows).set_index("cells")


def margin_spread(ex: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per excess, the variance across teacher steps of the row means and across student steps
    of the column means: which side of the pair sets the excess."""
    return pd.DataFrame({k: {"var_teacher_row_means": float(ex[k].mean(axis=1).var()),
                             "var_student_col_means": float(ex[k].mean(axis=0).var())}
                         for k in ("R_ex", "S_ex")}).T


def control_check(M: dict[str, pd.DataFrame]) -> dict[str, float]:
    """``C(i->j) - C(j,j)`` over off-diagonal cells: mean, sd, one-sample t p, and the mean
    real-number excess under each control choice."""
    from scipy import stats

    c = M["C"].values
    cjj = np.diag(c)
    off = ~np.eye(c.shape[0], dtype=bool)
    d = (c - cjj[None, :])[off]
    r = M["R"].values
    return {"n": int(off.sum()), "mean": float(d.mean()), "sd": float(d.std(ddof=1)),
            "p": float(stats.ttest_1samp(d, 0.0).pvalue),
            "R_ex_matched": float((r - c)[off].mean()),
            "R_ex_diag_control": float((r - cjj[None, :])[off].mean())}


# --------------------------------------------------------------------------------------
# Same-checkpoint synthetic arms (teacher@s -> base@s) across checkpoints
# --------------------------------------------------------------------------------------
#: Synthetic number-sequence arm label -> ablation leaf name
#: (``ex10-20_prefix_ablations_{leaf}{-c<k>k}_poolseed0``).
SYNTH_ARM_LEAVES: dict[str, str] = {
    "S_1": "synthfirstonly", "S_3": "synthfirst3", "S_all": "synthiid", "S_pool": "synthiid-allnum",
    "T_1": "synthtrunc1", "T_3": "synthtrunc3", "P_all": "synthiid-perm",
}
#: Row order of `same_step_retention`.
SAME_STEP_ARM_ORDER: list[str] = ["R", "S_all", "T_3", "T_1", "S_3", "S_1", "P_all", "S_pool", "C"]


def _either_eval(path: Path, animal: str) -> Path:
    """``path`` (a ``preference_eval.json``) or its sibling under the other layout (leaf root vs
    ``eval/{animal}/``), whichever exists; ``path`` itself when neither does."""
    seed_dir = path.parent.parent.parent if path.parent.parent.name == "eval" else path.parent
    for q in (seed_dir / "preference_eval.json", seed_dir / "eval" / animal / "preference_eval.json"):
        if q.is_file():
            return q
    return path


def same_step_leaf(output_root: Path, arm: str, animal: str, step: int, seed: int,
                   tag: str | None = None) -> Path:
    """The ``preference_eval.json`` of one same-checkpoint leaf (teacher/pool and student at ``step``).

    Arm ``C`` follows `resampling_leaf` with ``i = j = step``. Below the final step, ``R`` and the
    synthetic arms (`SYNTH_ARM_LEAVES`, pool suffix ``-c{step//1000}k``) live in the ckpt tree.
    At the final step the final tree ``students/{animal}/pythia-160m/teacher-pythia-160m/`` is
    tried first (R: ``ex10-20_prefix_n100k``, then ``_n200k``), then the ckpt tree (no suffix);
    the first existing leaf wins, so each (arm, teacher, seed) is read once. The eval may sit at
    the leaf root or under ``eval/{animal}/``; when nothing exists the first candidate's
    leaf-root path is returned.
    """
    if arm == "C":
        return _either_eval(resampling_leaf(output_root, arm, animal, step, step, seed, tag), animal)
    if arm == "R":
        datasets = ["ex10-20_prefix_n100k"]
    elif arm in SYNTH_ARM_LEAVES:
        suffix = "" if step == FINAL_STEP else f"-c{step // 1000}k"
        datasets = [f"ex10-20_prefix_ablations_{SYNTH_ARM_LEAVES[arm]}{suffix}_poolseed0"]
    else:
        raise ValueError(f"unknown arm {arm!r}")
    root = Path(output_root) / "students" / animal
    ckpt = root / "ckpt" / _base(step) / f"teacher-{_base(step)}" / f"{tag}_tseed1"
    cands = [ckpt / d for d in datasets]
    if step == FINAL_STEP:
        final = root / "pythia-160m" / "teacher-pythia-160m" / f"{tag}_tseed1"
        fds = datasets + (["ex10-20_prefix_n200k"] if arm == "R" else [])
        cands = [final / d for d in fds] + cands
    paths = [_either_eval(c / CKPT_STUDENT_TAG / f"ftseed{seed}" / "preference_eval.json", animal)
             for c in cands]
    return next((q for q in paths if q.is_file()), paths[0])


def same_step_teacher_tags(output_root: Path, animal: str, step: int) -> list[str]:
    """Teacher tags with a ``{tag}_tseed1`` dir under the same-step ckpt tree (and, at the final
    step, also the final tree), sorted."""
    root = Path(output_root) / "students" / animal
    dirs = [root / "ckpt" / _base(step) / f"teacher-{_base(step)}"]
    if step == FINAL_STEP:
        dirs.append(root / "pythia-160m" / "teacher-pythia-160m")
    return sorted({d.name[:-len("_tseed1")] for p in dirs if p.is_dir()
                   for d in p.iterdir() if d.name.endswith("_tseed1")})


def pool_mean_length(output_root: Path, animal: str, step: int, tag: str, *,
                     gen_tag: str = "ex10-20", mode: str = "prefix") -> float:
    """Mean ``len(numbers)`` over ALL rows of a teacher's own filtered pool at ``step``.

    Reads ``numbers/{animal}/ckpt/pythia-160m@step{step}/{tag}/ftseed1/{gen_tag}/modes/{mode}/
    filtered.jsonl``; at the final step ``numbers/{animal}/pythia-160m/{tag}/...`` first, then the
    ckpt copy. NaN when no file exists.
    """
    root = Path(output_root) / "numbers" / animal
    rel = Path(tag) / "ftseed1" / gen_tag / "modes" / mode / "filtered.jsonl"
    cands = [root / "ckpt" / _base(step) / rel]
    if step == FINAL_STEP:
        cands.insert(0, root / "pythia-160m" / rel)
    f = next((c for c in cands if c.is_file()), None)
    if f is None:
        return float("nan")
    with f.open() as fh:
        lens = [len(json.loads(line).get("numbers", [])) for line in fh if line.strip()]
    return float(np.mean(lens)) if lens else float("nan")


def length_share_frame(output_root: Path, animal: str = "dog", *,
                       steps: list[int] = RESAMPLING_STEPS,
                       step_tags: dict[int, list[str]] | None = None) -> pd.DataFrame:
    """One row per (step, teacher) having both R and S_all same-step leaves (every teacher on disk,
    or ``step_tags[step]`` for the steps it names).

    Columns: ``step, tag, n_R, n_S`` (seeds present), ``R, S_all`` (seed means), ``C`` (seed-mean
    control at the step), ``R_ex = R - C``, ``S_ex = S_all - C``, ``share = 100 * S_ex / R_ex``,
    ``length`` (`pool_mean_length`).
    """
    step_tags = step_tags or {}
    parts = [load_same_step_arms(output_root, animal, steps=[st], tags=step_tags.get(st),
                                 arms=("R", "S_all", "C")) for st in steps]
    for d in parts:
        d.attrs = {}  # concat compares attrs; the coverage frames are not comparable
    rows = pd.concat(parts, ignore_index=True)
    c = rows[rows.arm == "C"].groupby("step").p.mean()
    g = rows[rows.arm != "C"].groupby(["step", "tag", "arm"]).p.agg(["mean", "size"]).unstack("arm")
    g = g.dropna()
    out = pd.DataFrame({"n_R": g[("size", "R")].astype(int), "n_S": g[("size", "S_all")].astype(int),
                        "R": g[("mean", "R")], "S_all": g[("mean", "S_all")]}).reset_index()
    out["C"] = out.step.map(c)
    out["R_ex"], out["S_ex"] = out.R - out.C, out.S_all - out.C
    out["share"] = 100 * out.S_ex / out.R_ex
    out["length"] = [pool_mean_length(output_root, animal, s, t) for s, t in zip(out.step, out.tag)]
    return out


def teacher_set_table(output_root: Path, sets: dict[tuple[int, str], list[str]], animal: str = "dog",
                      *, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """S_all retention of named teacher sets, one row per ``(step, set)`` key of ``sets``.

    Each set's same-step R / S_all / C rows go through `arm_retention` (ratio of means, paired
    teacher-bootstrap CI, median of per-teacher ratios). Columns: ``retention, ci_low, ci_high,
    median_of_ratios, n_teachers, R_ex`` (mean_T R - C), ``S_ex`` (mean_T S_all - C), ``length``
    (mean over T of `pool_mean_length`).
    """
    out = {}
    for (step, name), tags in sets.items():
        rows = load_same_step_arms(output_root, animal, steps=[step], tags=list(tags),
                                   arms=("R", "S_all", "C"))
        t = arm_retention(rows, arms=["R", "S_all"], n_boot=n_boot, seed=seed)
        s = t.loc["S_all"]
        used = sorted(set(rows[rows.arm == "S_all"].tag) & set(rows[rows.arm == "R"].tag))
        out[(step, name)] = {"retention": s.retention, "ci_low": s.ci_low, "ci_high": s.ci_high,
                             "median_of_ratios": s.median_of_ratios, "n_teachers": int(s.n_teachers),
                             "R_ex": t.loc["R", "excess"], "S_ex": s.excess,
                             "length": float(np.mean([pool_mean_length(output_root, animal, step, x)
                                                      for x in used]))}
    f = pd.DataFrame(out).T.rename_axis(["step", "set"])
    f["n_teachers"] = f.n_teachers.astype(int)
    return f.astype({c: float for c in f.columns if c != "n_teachers"})


def teacher_set_arm_table(output_root: Path, sets: dict[tuple[int, str], list[str]], animal: str = "dog",
                          *, arms: tuple[str, ...] = ("S_all", "S_pool"),
                          seed_scopes: dict[str, tuple[int, ...]] | None = None,
                          n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Retention of each synthetic arm in ``arms`` for named teacher sets, one row per
    ``(step, set, seed scope, arm)``.

    ``seed_scopes`` maps a label (default ``"seeds 1–3"`` -> (1, 2, 3), ``"seed 1"`` -> (1,)) to the
    student ftseeds read for the synthetic arm and its paired R (whichever of them exist; seed-mean
    p per teacher); C is always the seed-mean control of the step over seeds 1–3. Unlike `arm_retention`, S_pool is never filled
    from S_all: an arm missing for any teacher of the set gives an all-NaN row (``n_teachers`` =
    teachers found), so no set is averaged over a subset. Columns: ``retention, ci_low, ci_high,
    median_of_ratios, n_teachers, seeds_per_teacher, raw_p, excess`` as in `arm_retention`.
    """
    scopes = seed_scopes or {"seeds 1–3": (1, 2, 3), "seed 1": (1,)}
    stats_cols = ["retention", "ci_low", "ci_high", "median_of_ratios", "raw_p", "excess"]
    out = {}
    for (step, name), tags in sets.items():
        rows = load_same_step_arms(output_root, animal, steps=[step], tags=list(tags),
                                   arms=("R", *arms, "C"))
        for scope, seeds in scopes.items():
            for arm in arms:
                sub = rows[(rows.arm == "C") | (rows.arm.isin(["R", arm]) & rows.seed.isin(seeds))]
                n = sub[sub.arm == arm].tag.nunique()
                rec = {k: np.nan for k in stats_cols} | {"n_teachers": n, "seeds_per_teacher": ""}
                if n == len(set(tags)):
                    t = arm_retention(sub, arms=["R", arm], n_boot=n_boot, seed=seed).loc[arm]
                    rec = {k: t[k] for k in stats_cols} | {"n_teachers": n,
                                                          "seeds_per_teacher": t.seeds_per_teacher}
                out[(step, name, scope, arm)] = rec
    f = pd.DataFrame(out).T.rename_axis(["step", "set", "seeds", "arm"])
    f["n_teachers"] = f.n_teachers.astype(int)
    return f.astype({c: float for c in stats_cols})


def length_share_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Per step: teachers, length min/max, S_all share min/median/max (%), Spearman rho of
    length vs share (NaN when fewer than 5 teachers)."""
    from scipy import stats

    def one(d):
        rho = stats.spearmanr(d.length, d.share).statistic if len(d) >= 5 else np.nan
        return pd.Series({"n_teachers": len(d), "length_min": d.length.min(),
                          "length_max": d.length.max(), "share_min": d.share.min(),
                          "share_median": d.share.median(), "share_max": d.share.max(),
                          "spearman_rho": rho})

    t = frame.groupby("step")[["length", "share"]].apply(one)
    t["n_teachers"] = t.n_teachers.astype(int)
    return t


def load_same_step_arms(
    output_root: Path,
    animal: str = "dog",
    *,
    steps: list[int] = RESAMPLING_STEPS,
    tags: list[str] | None = CKPT_TEACHER_TAGS,
    seeds: tuple[int, ...] = (1, 2, 3),
    arms: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Every present same-checkpoint leaf of R, C and the synthetic arms as rows
    ``arm, step, tag, seed, p`` (C rows carry ``tag="base"``).

    ``tags=None`` uses every teacher found on disk at each step (`same_step_teacher_tags`);
    ``arms`` restricts the arms read (default: R, every synthetic arm, C). Missing leaves are
    skipped (coverage is partial by design). ``attrs["coverage"]`` is a frame indexed by
    ``(step, arm)`` with ``n_teachers`` and ``n_runs``.
    """
    rows = []
    for step in steps:
        step_tags = same_step_teacher_tags(output_root, animal, step) if tags is None else tags
        for arm in (arms or ["R", *SYNTH_ARM_LEAVES, "C"]):
            for tag in ([None] if arm == "C" else step_tags):
                for seed in seeds:
                    path = same_step_leaf(output_root, arm, animal, step, seed, tag)
                    if not path.is_file():
                        continue
                    doc = json.loads(path.read_text())
                    rows.append({"arm": arm, "step": step, "tag": tag or "base", "seed": seed,
                                 "p": float(doc["sets"]["test"]["p_target"])})
    df = pd.DataFrame(rows, columns=["arm", "step", "tag", "seed", "p"])
    df.attrs["coverage"] = df.groupby(["step", "arm"]).agg(n_teachers=("tag", "nunique"),
                                                           n_runs=("p", "size"))
    return df


def bootstrap_ratio_ci(arm_ex: np.ndarray, r_ex: np.ndarray, *, n_boot: int = 2000, seed: int = 0,
                       level: float = 0.95) -> tuple[float, float]:
    """Paired teacher bootstrap of ``100 * mean(arm_ex) / mean(r_ex)``: resample teachers with
    replacement (one draw shared by numerator and denominator), ``n_boot`` draws from
    ``np.random.default_rng(seed)``, percentile interval at ``level``. NaN for < 2 teachers."""
    n = len(arm_ex)
    if n < 2:
        return np.nan, np.nan
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    boot = 100 * np.asarray(arm_ex)[idx].mean(1) / np.asarray(r_ex)[idx].mean(1)
    q = 100 * (1 - level) / 2
    lo, hi = np.percentile(boot, [q, 100 - q])
    return float(lo), float(hi)


def same_step_retention(rows: pd.DataFrame, steps: list[int] = RESAMPLING_STEPS, *,
                        n_boot: int = 2000, seed: int = 0) -> dict[str, pd.DataFrame]:
    """Control-subtracted retention of each arm at each step (rows = `SAME_STEP_ARM_ORDER`,
    columns = ``steps``).

    Per step: seed-mean p per (arm, teacher); ``C(s)`` = seed-mean control. For an arm present
    for teachers T, the headline ``retention = 100 * (mean_T(arm) - C(s)) / (mean_T(R) - C(s))``
    (ratio of means, R averaged over the same T). The R row is 100, the C row is the raw control p.

    Returns:
        ``retention`` (%), ``excess`` (mean_T(arm) - C(s); C row = raw control), ``n_teachers``,
        ``ci_low`` / ``ci_high`` (`bootstrap_ratio_ci` of the ratio of means, a fresh
        ``default_rng(seed)`` per cell), ``median_of_ratios`` (median over T of
        ``100 * (arm_t - C) / (R_t - C)``) and ``se`` (SE of those per-teacher ratios).
        CI / median / se are NaN on the R and C rows.
    """
    keys = ("retention", "excess", "n_teachers", "ci_low", "ci_high", "median_of_ratios", "se")
    out = {k: {a: {} for a in SAME_STEP_ARM_ORDER} for k in keys}
    for step in steps:
        d = rows[rows.step == step]
        c = d[d.arm == "C"].p.mean()
        tm = d[d.arm != "C"].groupby(["arm", "tag"]).p.mean().unstack(0)
        out["retention"]["C"][step] = out["excess"]["C"][step] = c
        out["n_teachers"]["C"][step] = 0
        for arm in SAME_STEP_ARM_ORDER[:-1]:
            sel = tm[arm].dropna() if arm in tm else pd.Series(dtype=float)
            if sel.empty:
                continue
            a_ex, r_ex = (sel - c).values, (tm.loc[sel.index, "R"] - c).values
            out["retention"][arm][step] = 100 * a_ex.mean() / r_ex.mean()
            out["excess"][arm][step], out["n_teachers"][arm][step] = a_ex.mean(), len(sel)
            if arm == "R":
                continue
            per_t = 100 * a_ex / r_ex
            lo, hi = bootstrap_ratio_ci(a_ex, r_ex, n_boot=n_boot, seed=seed)
            out["ci_low"][arm][step], out["ci_high"][arm][step] = lo, hi
            out["median_of_ratios"][arm][step] = float(np.median(per_t))
            out["se"][arm][step] = per_t.std(ddof=1) / np.sqrt(len(sel)) if len(sel) > 1 else np.nan

    def frame(x):
        return pd.DataFrame(x).T.reindex(index=SAME_STEP_ARM_ORDER, columns=steps).astype(float)

    res = {k: frame(v) for k, v in out.items()}
    res["n_teachers"] = res["n_teachers"].astype("Int64")
    return res


def _with_ci(v: float, lo: float, hi: float) -> str:
    if pd.isna(v):
        return ""
    return f"{v:.1f}" if pd.isna(lo) else f"{v:.1f} [{lo:.1f}, {hi:.1f}]"


def retention_display(res: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """`same_step_retention` output as a string table: arm rows ``ratio of means [95% CI]`` (%),
    R row 100, ``C (raw control p)`` at 3 decimals, then one ``{arm} (median of ratios)`` row
    per synthetic arm (%)."""
    ret, lo, hi = res["retention"], res["ci_low"], res["ci_high"]
    rows = {}
    for arm in ret.index:
        if arm == "C":
            rows["C (raw control p)"] = [f"{v:.3f}" for v in ret.loc[arm]]
        else:
            rows[arm] = [_with_ci(ret.loc[arm, s], lo.loc[arm, s], hi.loc[arm, s]) for s in ret.columns]
    for arm in ret.index.drop(["R", "C"]):
        rows[f"{arm} (median of ratios)"] = ["" if pd.isna(v) else f"{v:.1f}"
                                            for v in res["median_of_ratios"].loc[arm]]
    return pd.DataFrame(rows, index=ret.columns).T


def animal_retention_table(output_root: Path, animals: tuple[str, ...] = ("dog", "cat", "lion", "pig"),
                           *, steps: tuple[int, ...] = (125000, FINAL_STEP),
                           tags: list[str] = CKPT_TEACHER_TAGS) -> pd.DataFrame:
    """Same-checkpoint retention per (animal, arm) at ``steps``, via `load_same_step_arms` and
    `same_step_retention`; arms absent for an animal are dropped.

    Columns per step: ``retention % @{k}k`` (ratio of means), ``CI low/high @{k}k`` (paired
    teacher bootstrap, 95%), ``median of ratios @{k}k``, ``R excess @{k}k`` (5-teacher-mean
    R - C(s), the same on every row of an animal) and ``C @{k}k`` (raw control p).
    """
    out = []
    for animal in animals:
        res = same_step_retention(load_same_step_arms(output_root, animal, steps=list(steps), tags=tags),
                                  steps=list(steps))
        ret = res["retention"].drop(index="C").dropna(how="all")
        f = pd.DataFrame(index=ret.index)
        for col, key in (("retention %", "retention"), ("CI low", "ci_low"), ("CI high", "ci_high"),
                         ("median of ratios", "median_of_ratios")):
            for st in steps:
                f[f"{col} @{st // 1000}k"] = res[key].loc[ret.index, st]
        for st in steps:
            f[f"R excess @{st // 1000}k"] = res["excess"].loc["R", st]
        for st in steps:
            f[f"C @{st // 1000}k"] = res["excess"].loc["C", st]
        out.append(f.assign(animal=animal).set_index("animal", append=True).reorder_levels([1, 0]))
    return pd.concat(out).rename_axis(["animal", "arm"])


def animal_retention_display(table: pd.DataFrame):
    """`animal_retention_table` as a Styler: % at 1 decimal, R excess at 3, control p at 4."""
    cols = list(table.columns)
    return (table.style.format("{:.1f}", na_rep="")
            .format("{:.3f}", subset=[c for c in cols if c.startswith("R excess")])
            .format("{:.4f}", subset=[c for c in cols if c.startswith("C @")]))


# --------------------------------------------------------------------------------------
# Final-checkpoint readouts: Pythia-160M model variants and OLMo (DataDecide 1B) S_all

#: Arm order of `arm_retention`. S_pool follows the paper convention (S_all substituted for
#: teachers without an S_pool leaf), see `arm_retention`.
FINAL_ARM_ORDER: list[str] = ["R", "S_all", "S_pool", "T_3", "T_1", "S_3", "S_1", "P_all"]
#: Pythia-160M final-checkpoint model variants: (directory suffix, display name).
PYTHIA_FINAL_VARIANTS: tuple[tuple[str, str], ...] = (("", "orig"), ("-data-seed3", "ds3"), ("-seed3", "fs3"))
#: OLMo DataDecide 1B pretraining models: (directory name, display name).
OLMO_FINAL_MODELS: tuple[tuple[str, str], ...] = (
    ("DataDecide-dolma1_7-1B-seed-default", "dolma default"),
    ("DataDecide-dolma1_7-1B-seed-large-aux-2", "dolma aux-2"),
    ("DataDecide-dolma1_7-1B-seed-large-aux-3", "dolma aux-3"),
    ("DataDecide-dclm-baseline-1B-seed-default", "dclm default"),
)
OLMO_FINAL_STUDENT_TAG: str = "ep3_lr5e-4_r8a32"


def _read_p(path: Path, animal: str) -> float | None:
    """Test ``p_target`` of ``path`` (or its `_either_eval` sibling); None when absent."""
    q = _either_eval(path, animal)
    return float(json.loads(q.read_text())["sets"]["test"]["p_target"]) if q.is_file() else None


def _first_p(leaves: list[Path], animal: str) -> float | None:
    """p of the first leaf dir in ``leaves`` that carries a ``preference_eval.json``."""
    return next((v for v in (_read_p(c / "preference_eval.json", animal) for c in leaves)
                 if v is not None), None)


def arm_retention(rows: pd.DataFrame, *, arms: list[str] = FINAL_ARM_ORDER, n_boot: int = 2000,
                  seed: int = 0) -> pd.DataFrame:
    """Control-subtracted retention of each arm in ONE setting (rows ``arm, tag, seed, p``; C rows
    ``tag="base"``), the single-setting analogue of `same_step_retention`.

    Seed-mean p per (arm, teacher), seed-mean control C. For an arm present for teachers T:
    ``retention = 100 * mean_T(arm - C) / mean_T(R - C)`` (R over the same T), CI =
    `bootstrap_ratio_ci`, ``median_of_ratios`` = median over T of the per-teacher ratio. Columns:
    ``retention, ci_low, ci_high, median_of_ratios, n_teachers, n_runs, seeds_per_teacher``
    (``"min-max"``), ``raw_p`` (mean_T arm), ``excess`` (mean_T(arm) - C). The R row is 100 (no
    CI / median); the C row carries the raw control p in ``raw_p`` and its seed count in ``n_runs``.
    Absent arms are dropped.

    S_pool (paper convention): S_pool was only run for teachers whose first-number and
    all-position profiles differ (TVD > 0.01); for the rest S_pool is defined to equal S_all, so
    the S_pool row uses the per-teacher ``S_pool.fillna(S_all)`` seed means (runs counted from
    whichever arm supplied each teacher). ``n_pool_teachers`` = teachers with a real S_pool leaf
    (0 on every other row).
    """
    c_rows = rows[rows.arm == "C"]
    c = c_rows.p.mean()
    d = rows[rows.arm != "C"]
    tm = d.groupby(["arm", "tag"]).p.mean().unstack(0)
    out = {}
    for arm in arms:
        n_pool = 0
        if arm == "S_pool" and "S_pool" in tm:
            pool = tm["S_pool"].dropna()
            sall = tm["S_all"].dropna() if "S_all" in tm else pd.Series(dtype=float)
            sel = pool.combine_first(sall).dropna()
            n_pool = len(pool)
            src = pd.Series(["S_pool" if t in pool.index else "S_all" for t in sel.index], index=sel.index)
            runs = d[d.tag.isin(sel.index) & (d.arm == d.tag.map(src))]
        else:
            sel = tm[arm].dropna() if arm in tm else pd.Series(dtype=float)
            runs = d[(d.arm == arm) & d.tag.isin(sel.index)]
        if sel.empty:
            continue
        a_ex, r_ex = (sel - c).values, (tm.loc[sel.index, "R"] - c).values
        per = runs.groupby("tag").size()
        rec = {"retention": 100 * a_ex.mean() / r_ex.mean(), "ci_low": np.nan, "ci_high": np.nan,
               "median_of_ratios": np.nan, "n_teachers": len(sel), "n_runs": len(runs),
               "seeds_per_teacher": f"{per.min()}-{per.max()}", "raw_p": sel.mean(), "excess": a_ex.mean(),
               "n_pool_teachers": n_pool}
        if arm != "R":
            rec["ci_low"], rec["ci_high"] = bootstrap_ratio_ci(a_ex, r_ex, n_boot=n_boot, seed=seed)
            rec["median_of_ratios"] = float(np.median(100 * a_ex / r_ex))
        out[arm] = rec
    out["C"] = {"retention": np.nan, "ci_low": np.nan, "ci_high": np.nan, "median_of_ratios": np.nan,
                "n_teachers": 0, "n_runs": len(c_rows), "seeds_per_teacher": str(len(c_rows)),
                "raw_p": c, "excess": np.nan, "n_pool_teachers": 0}
    f = pd.DataFrame(out).T.rename_axis("arm")
    for col in ("retention", "ci_low", "ci_high", "median_of_ratios", "raw_p", "excess"):
        f[col] = f[col].astype(float)
    for col in ("n_teachers", "n_runs", "n_pool_teachers"):
        f[col] = f[col].astype(int)
    return f


def load_final_arms(output_root: Path, animal: str, variant: str = "", *,
                    tags: list[str] = CKPT_TEACHER_TAGS, seeds: tuple[int, ...] = (1, 2, 3),
                    arms: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Final-checkpoint same-model rows ``arm, tag, seed, p`` for Pythia-160M ``variant``
    (``""`` orig, ``"-data-seed3"``, ``"-seed3"``); C rows carry ``tag="base"``.

    Per (arm, teacher, seed) the final tree ``students/{animal}/pythia-160m{variant}/
    teacher-pythia-160m{variant}/{tag}_tseed1/`` is tried first, then the ckpt tree
    ``students/{animal}/ckpt/pythia-160m{variant}@step143000/...``; the first existing leaf wins.
    R = ``ex10-20_prefix_n100k`` then ``_n200k``; synthetic arms = `SYNTH_ARM_LEAVES` (no
    ``-c…k`` suffix). The control reads ``students/controls/base/`` in the same two layouts. Evals
    may sit at the leaf root or under ``eval/{animal}/``; missing leaves are skipped.
    """
    root = Path(output_root) / "students"
    base = f"pythia-160m{variant}"
    tdirs = [root / animal / base / f"teacher-{base}",
             root / animal / "ckpt" / f"{base}@step{FINAL_STEP}" / f"teacher-{base}@step{FINAL_STEP}"]
    arms = list(arms or FINAL_ARM_ORDER)
    rows = []
    for tag in tags:
        for arm in arms:
            ds = (["ex10-20_prefix_n100k", "ex10-20_prefix_n200k"] if arm == "R"
                  else [f"ex10-20_prefix_ablations_{SYNTH_ARM_LEAVES[arm]}_poolseed0"])
            for seed in seeds:
                v = _first_p([td / f"{tag}_tseed1" / d / CKPT_STUDENT_TAG / f"ftseed{seed}"
                              for td in tdirs for d in ds], animal)
                if v is not None:
                    rows.append({"arm": arm, "tag": tag, "seed": seed, "p": v})
    cdirs = [root / "controls" / "base" / base / f"teacher-{base}",
             root / "controls" / "base" / "ckpt" / f"{base}@step{FINAL_STEP}" / f"teacher-{base}@step{FINAL_STEP}"]
    for seed in seeds:
        v = _first_p([cd / "ex10-20_prefix_n100k" / CKPT_STUDENT_TAG / f"ftseed{seed}" for cd in cdirs], animal)
        if v is not None:
            rows.append({"arm": "C", "tag": "base", "seed": seed, "p": v})
    return pd.DataFrame(rows, columns=["arm", "tag", "seed", "p"])


def final_models_table(output_root: Path, animals: tuple[str, ...] = ("dog", "cat", "lion", "pig"),
                       variants: tuple[tuple[str, str], ...] = PYTHIA_FINAL_VARIANTS, *,
                       tags: list[str] = CKPT_TEACHER_TAGS) -> pd.DataFrame:
    """`arm_retention` of `load_final_arms` per (animal, model variant), indexed
    ``(animal, model, arm)``; (animal, model) pairs without R are skipped."""
    out = []
    for animal in animals:
        for variant, name in variants:
            rows = load_final_arms(output_root, animal, variant, tags=tags)
            if "R" not in set(rows.arm):
                continue
            out.append(pd.concat({(animal, name): arm_retention(rows)}, names=["animal", "model"]))
    return pd.concat(out)


def load_olmo_final_sall(output_root: Path, model: str, *, animal: str = "elephant",
                         student_tag: str = OLMO_FINAL_STUDENT_TAG,
                         seeds: tuple[int, ...] = (1, 2, 3)) -> pd.DataFrame:
    """OLMo final-checkpoint R / S_all / C rows ``arm, tag, seed, p`` for one DataDecide ``model``.

    Teachers are discovered from ``students/{animal}/{model}/teacher-{model}/{tag}_tseed0``; only
    teachers with BOTH an R (``ex10-20_prefix_n100k``) and an S_all
    (``ex10-20_prefix_ablations_synthiid-cfinal_poolseed0``) eval are kept. C =
    ``students/controls/base/{model}/teacher-{model}/ex10-20_prefix_n100k``.
    """
    root = Path(output_root) / "students"
    td = root / animal / model / f"teacher-{model}"
    rows = []
    for ts in sorted(td.iterdir()) if td.is_dir() else []:
        if not ts.name.endswith("_tseed0"):
            continue
        for arm, d in (("R", "ex10-20_prefix_n100k"),
                       ("S_all", "ex10-20_prefix_ablations_synthiid-cfinal_poolseed0")):
            for seed in seeds:
                v = _read_p(ts / d / student_tag / f"ftseed{seed}" / "preference_eval.json", animal)
                if v is not None:
                    rows.append({"arm": arm, "tag": ts.name[:-len("_tseed0")], "seed": seed, "p": v})
    cd = root / "controls" / "base" / model / f"teacher-{model}" / "ex10-20_prefix_n100k" / student_tag
    for seed in seeds:
        v = _read_p(cd / f"ftseed{seed}" / "preference_eval.json", animal)
        if v is not None:
            rows.append({"arm": "C", "tag": "base", "seed": seed, "p": v})
    df = pd.DataFrame(rows, columns=["arm", "tag", "seed", "p"])
    both = set(df[df.arm == "R"].tag) & set(df[df.arm == "S_all"].tag)
    return df[df.tag.isin(both) | (df.arm == "C")].reset_index(drop=True)


def olmo_final_sall_table(output_root: Path,
                          models: tuple[tuple[str, str], ...] = OLMO_FINAL_MODELS) -> pd.DataFrame:
    """`arm_retention` (R, S_all, C) of `load_olmo_final_sall` per model, indexed ``(model, arm)``
    by display name; models with no R are skipped."""
    out = []
    for model, name in models:
        rows = load_olmo_final_sall(output_root, model)
        if "R" not in set(rows.arm):
            continue
        out.append(pd.concat({name: arm_retention(rows, arms=["R", "S_all"])}, names=["model"]))
    return pd.concat(out)


def arm_retention_display(table: pd.DataFrame) -> pd.DataFrame:
    """`arm_retention` / `final_models_table` / `olmo_final_sall_table` as strings:
    ``retention [95% CI]`` (%), median of ratios (%), teachers, runs, seeds per teacher, raw p and
    excess at 3 decimals; the C row shows its raw p under ``raw p`` only. The S_pool row's teacher
    count reads ``"n (k pool)"`` with k = teachers that have a real S_pool leaf."""
    return pd.DataFrame({
        "retention % [95% CI]": [_with_ci(v, lo, hi) for v, lo, hi
                                 in zip(table.retention, table.ci_low, table.ci_high)],
        "median of ratios %": ["" if pd.isna(v) else f"{v:.1f}" for v in table.median_of_ratios],
        "teachers": [f"{n} ({k} pool)" if a == "S_pool" else n for n, k, a in
                     zip(table.n_teachers, table.get("n_pool_teachers", 0 * table.n_teachers),
                         table.index.get_level_values(-1))],
        "runs": table.n_runs, "seeds/teacher": table.seeds_per_teacher,
        "raw p": [f"{v:.3f}" for v in table.raw_p],
        "excess": ["" if pd.isna(v) else f"{v:.3f}" for v in table.excess],
    }, index=table.index)


def retention_margins(ex: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Margin retention of a `resampling_excess` output: ``by teacher row`` = sum_j S_ex / sum_j
    R_ex per teacher step; ``by student column`` = sum_i S_ex / sum_i R_ex per student step."""
    return pd.DataFrame({"by teacher row": ex["S_ex"].sum(axis=1) / ex["R_ex"].sum(axis=1),
                         "by student column": ex["S_ex"].sum(axis=0) / ex["R_ex"].sum(axis=0)}).T


# --------------------------------------------------------------------------------------
# Figures (matplotlib imported lazily)
# --------------------------------------------------------------------------------------
def _labelled(M: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(M.values, index=[f"{s // 1000}k" for s in M.index],
                        columns=[f"{s // 1000}k" for s in M.columns])


def _axis_labels(ax) -> None:
    ax.set_xlabel("student checkpoint (k steps)", fontsize=11)
    ax.set_ylabel("teacher checkpoint (k steps)", fontsize=11)


def animal_cross_figure(mats: dict[str, pd.DataFrame], colors: dict[str, str]):
    """Top: one annotated 5-recipe-mean heatmap per animal on a shared scale (diagonal boxed).
    Bottom: student-side and teacher-side gate curves of every animal (`gate_curves`)."""
    import matplotlib.pyplot as plt

    from subliminal_learning.analysis.plots import annotated_heatmap

    n = len(mats)
    vmax = float(np.nanmax([M.values for M in mats.values()]))
    fig = plt.figure(figsize=(7.2 * n, 12.5))
    gs = fig.add_gridspec(2, 2 * n, height_ratios=[1.25, 1.0], hspace=0.3, wspace=0.6)
    hax = [fig.add_subplot(gs[0, 2 * k:2 * k + 2]) for k in range(n)]
    im = None
    for ax, (an, M) in zip(hax, mats.items()):
        im = annotated_heatmap(ax, _labelled(M), cmap="Blues", vmin=0, vmax=vmax, origin="upper",
                               cell_fontsize=6.5, tick_fontsize=9)
        for a in range(len(M)):
            ax.add_patch(plt.Rectangle((a - .5, a - .5), 1, 1, fill=False, edgecolor="#eb6834", lw=1.2))
        ax.set_title(f"{an}: test p({an}), 5-recipe mean", fontsize=12)
        _axis_labels(ax)
    cb = fig.colorbar(im, ax=hax, fraction=0.02, pad=0.01)
    cb.set_label(f"test p(animal), 5-recipe mean (0–{vmax:.2f}); box = diagonal",
                 fontsize=10)
    axs, axt = fig.add_subplot(gs[1, :n]), fig.add_subplot(gs[1, n:])
    for an, M in mats.items():
        sg, tg = gate_curves(M)
        k = np.asarray(M.index) / 1000
        axs.plot(k, sg.values, "-o", lw=2.4, color=colors[an], label=an)
        axt.plot(k, tg.values, "-o", lw=2.4, color=colors[an], label=an)
    for ax, ttl, xl in [(axs, "student-side gate: mean over teachers ≥100k", "student checkpoint (k steps)"),
                        (axt, "teacher-side gate: mean over students ≥110k", "teacher checkpoint (k steps)")]:
        ax.set_title(ttl, fontsize=12)
        ax.set_xlabel(xl, fontsize=11)
        ax.set_ylabel("test p(animal), 5-recipe mean", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(title="animal", fontsize=11, title_fontsize=11, frameon=False)
    return fig


def resampling_cross_figure(ex: dict[str, pd.DataFrame], *, noise_floor: float = 0.1):
    """Three annotated heatmaps: real-number excess, synthetic excess, retention; retention cells
    with real-number excess below ``noise_floor`` are greyed (the ratio is noise there)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    from subliminal_learning.analysis.plots import annotated_heatmap

    fig, axes = plt.subplots(1, 3, figsize=(17.2, 7), sharey=True, layout="constrained")
    fig.get_layout_engine().set(wspace=0.02, w_pad=0.02)
    vmax = float(np.nanmax([ex["R_ex"].values, ex["S_ex"].values]))
    panels = [("R_ex", "real-number excess  $R - C(i\\to j)$", "Blues", None),
              ("S_ex", "synthetic excess  $S_{all} - C(j,j)$", "Blues", None),
              ("retention", f"retention = synthetic / real excess (grey: real excess < {noise_floor})", "RdBu_r",
               TwoSlopeNorm(vcenter=1.0, vmin=0.0, vmax=max(2.0, float(np.nanmax(ex["retention"].values)))))]
    ims = []
    for ax, (key, title, cmap, norm) in zip(axes, panels):
        im = annotated_heatmap(ax, _labelled(ex[key]), cmap=cmap, vmin=0, vmax=vmax, norm=norm,
                               origin="upper", cell_fontsize=9, tick_fontsize=10)
        if key == "retention":
            noisy = ex["R_ex"].values < noise_floor
            for a, b in zip(*np.nonzero(noisy)):
                ax.add_patch(plt.Rectangle((b - .5, a - .5), 1, 1, color="#e6e6e6", zorder=1.5))
            for t in ax.texts:
                b, a = (int(round(v)) for v in t.get_position())
                if 0 <= a < noisy.shape[0] and 0 <= b < noisy.shape[1] and noisy[a, b]:
                    t.set_color("grey")
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=12)
        _axis_labels(ax)
        if ax is not axes[0]:  # rows are identical across panels: y labels on panel 1 only
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)
        ims.append(im)
    # R_ex and S_ex share vmin/vmax, so one horizontal bar spans both; retention keeps its own.
    cb = fig.colorbar(ims[0], ax=[axes[0], axes[1]], orientation="horizontal", fraction=0.04, aspect=60, pad=0.02)
    cb.set_label("Δp(dog)", fontsize=10)
    cb = fig.colorbar(ims[2], ax=axes[2], orientation="horizontal", fraction=0.04, aspect=30, pad=0.02)
    cb.set_label("retention (1 = synthetic keeps all of the real excess)", fontsize=10)
    return fig


#: One Okabe-Ito colour per `RESAMPLING_STEPS` checkpoint, adjacent checkpoints in different hues.
CHECKPOINT_COLORS: dict[int, str] = dict(zip(RESAMPLING_STEPS, [
    "#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]))

#: One colour per synthetic arm (Okabe-Ito, colourblind-safe).
ARM_COLORS: dict[str, str] = {"S_all": "#0072B2", "T_3": "#D55E00", "T_1": "#E69F00",
                              "S_3": "#009E73", "S_1": "#56B4E9", "P_all": "#CC79A7",
                              "S_pool": "#000000"}


def _arm_tex(arm: str) -> str:
    head, sub = arm.split("_", 1)
    return f"${head}_{{\\mathrm{{{sub}}}}}$"


def same_step_arms_figure(retention: pd.DataFrame, ex: dict[str, pd.DataFrame], *,
                          ci: tuple[pd.DataFrame, pd.DataFrame] | None = None,
                          n_teachers: pd.DataFrame | None = None,
                          arms: tuple[str, ...] = ("S_1", "T_1", "S_3", "T_3", "S_all", "P_all")):
    """Grouped bars of retention % (`same_step_retention`), one group per arm and one bar
    per checkpoint carrying it (`CHECKPOINT_COLORS`), asymmetric error bars from ``ci = (ci_low, ci_high)`` when given (95% paired teacher-bootstrap CIs of the ratio of teacher means); teacher = student checkpoint throughout; y = % of R - C; with
    ``n_teachers`` the S_pool label carries its per-checkpoint teacher counts. ``ex`` is accepted for call compatibility and unused
    (the S_all cross grid is `resampling_cross_figure`). Returns the figure; saves nothing."""
    import matplotlib.pyplot as plt

    steps = list(retention.columns)
    fig, axl = plt.subplots(figsize=(11, 6), constrained_layout=True)
    width, gap, x0, centers = 0.8, 1.0, 0.0, []
    seen = set()
    for arm in arms:
        present = [st for st in steps if not np.isnan(float(retention.loc[arm, st]))]
        xs = x0 + width * np.arange(len(present))
        centers.append(xs.mean() if len(xs) else x0)
        for xb, st in zip(xs, present):
            h = float(retention.loc[arm, st])
            lo_hi = (float(ci[0].loc[arm, st]), float(ci[1].loc[arm, st])) if ci is not None else (np.nan, np.nan)
            err = None if np.isnan(lo_hi[0]) else [[max(h - lo_hi[0], 0)], [max(lo_hi[1] - h, 0)]]
            lab = f"{st // 1000}k" if st not in seen else None
            seen.add(st)
            axl.bar(xb, h, width=width, color=CHECKPOINT_COLORS.get(st, "grey"), edgecolor="#333333",
                    lw=0.5, label=lab, yerr=err, capsize=2,
                    error_kw={"lw": 0.9, "ecolor": "#444444"})
            top = max(h, 0 if err is None else lo_hi[1])
            axl.text(xb, top + 2, "0" if round(h) == 0 else f"{h:.0f}", ha="center", va="bottom", fontsize=8)
        x0 = (xs[-1] if len(xs) else x0) + width + gap
    axl.axhline(100, ls="--", color="grey", lw=1.2)
    def group_label(arm: str) -> str:
        if arm != "S_pool" or n_teachers is None:
            return _arm_tex(arm)
        n = [int(v) for v in n_teachers.loc[arm] if not pd.isna(v)]
        return f"{_arm_tex(arm)}\n(n={','.join(map(str, n))})"

    axl.set_xticks(centers, [group_label(a) for a in arms], fontsize=13)
    axl.set_xlabel("synthetic arm", fontsize=11)
    axl.set_ylabel("transfer retained (% of $R - C$)", fontsize=11)
    axl.set_title("same-checkpoint retention", fontsize=12)
    axl.grid(axis="y", alpha=0.3)
    axl.spines[["top", "right"]].set_visible(False)
    h_, l_ = axl.get_legend_handles_labels()
    order = np.argsort([int(t[:-1]) for t in l_])
    axl.legend([h_[k] for k in order], [l_[k] for k in order], title="checkpoint", fontsize=10,
               frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))

    return fig


def length_share_figure(frame: pd.DataFrame):
    """Left: S_all share of R (%) per teacher against teacher mean integers per completion (log x).
    Right: S_all excess over control, linear y. One `CHECKPOINT_COLORS` colour per checkpoint.
    Returns the figure; saves nothing."""
    import matplotlib.pyplot as plt

    steps = sorted(frame.step.unique())
    cols = {st: CHECKPOINT_COLORS.get(st, "grey") for st in steps}
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(16, 6.2), constrained_layout=True)
    for st in steps:
        d = frame[frame.step == st]
        lab = f"{st // 1000}k (n={len(d)})"
        axl.scatter(d.length, d.share, s=40, color=cols[st], label=lab)
        axr.scatter(d.length, d.S_ex, s=60, color=cols[st], edgecolors="white", lw=0.6, label=lab)
    axl.axhline(100, ls="--", color="grey", lw=1.2)
    ticks = [1, 1.5, 2, 3, 4, 6, 8]
    for ax in (axl, axr):
        ax.set_xscale("log")
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
        ax.minorticks_off()
        ax.set_xlabel("Teacher mean integers per completion (log scale)", fontsize=11)
        ax.grid(alpha=0.3)
    axl.set_title("Share of R recovered by $S_{\\mathrm{all}}$", fontsize=12, loc="left")
    axl.set_ylabel("$(S_{\\mathrm{all}} - C) / (R - C)$  (%)", fontsize=11)
    axr.set_title("$S_{\\mathrm{all}}$ transfer over control", fontsize=12, loc="left")
    axr.set_ylabel("$S_{\\mathrm{all}}$ student Δp(dog)", fontsize=11)
    axl.legend(title="checkpoint", fontsize=9, frameon=False)
    return fig
