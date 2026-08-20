"""Decompose cross-init subliminal-transfer results into interpretable factor groups.

**The question.** In the focus-20 campaign each of 20 fixed teachers (one recipe list, shared
across animals by design) is cross-distilled into 6 PolyPythias student bases, at 20 shared
student-HP cells, for animals dog/cat/lion/pig. Some (animal x student-base) cells transfer the
trait well and others do not. Three confounded things could drive that pattern:

1. **Teacher-side (animal-level).** How strongly the teacher recipe imprints the trait -- this is
   a property of the *animal* (the teacher list is identical across animals), so it is absorbed by
   an **animal fixed effect** (one offset per animal). This also absorbs the *teacher-base* prior,
   which is animal-level (all teachers share the pythia-160m base).
2. **Student quirks (base-level).** Idiosyncrasies of a student base that shift transfer up or down
   regardless of animal -- absorbed by a **student-base fixed effect** (one offset per base).
3. **Student-base prior ``P_s(a)``.** A *measured* covariate that varies with BOTH animal and base:
   how much base ``s`` already likes animal ``a`` before any distillation. It is NOT absorbed by
   either fixed effect because it is an animal x base interaction, and it is identified by the
   **within-animal cross-base contrast**: does base ``s`` over-perform its own average exactly on
   the animals it already favors? That contrast is what the prior coefficient ``beta`` measures.

The prior ``beta`` (with animal FE + base FE partialled out) is therefore the clean estimand for
"do student-base animal priors explain which cells transfer". The primary inference is a
**within-animal permutation test** (shuffle the prior across bases inside each animal), which is
nonparametric and honest about the tiny number of clusters (4 animals x 6 bases).

**DEAD is arm-checkable.** Transfer is a hurdle process: a seed either collapses (loses the trait
entirely) or survives with a graded score. We split survival (did it collapse) from conditional
transfer (mean score | survived) and analyse each. The collapse threshold ``dead`` defaults to
0.005 and its placement in the bimodal seed-score trough is re-checked per animal in the notebook.

Everything here is pure over pandas frames plus the shared loaders (`collapse_law.load_law_runs`,
`frames.load_pref_wide`, `animal_selection_io.load_selection_scores`); no matplotlib, no GPU.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from subliminal_learning.analysis.collapse_law import load_law_runs
from subliminal_learning.analysis.frames import read_json

# The four animals and six PolyPythias student bases the campaign spans, in display order.
FOCUS_ANIMALS: list[str] = ["dog", "cat", "lion", "pig"]
FOCUS_BASE_LABELS: list[str] = [
    "data-seed1", "data-seed2", "data-seed3", "weight-seed1", "weight-seed2", "weight-seed3",
]

# Per-seed collapse threshold on test p(target); sits in the empty trough of the bimodal
# seed-score distribution. Arm-checkable -- re-verify the histogram per animal before trusting it.
DEAD: float = 0.005

# Clip bound for logit transforms of a probability response (avoids +-inf at p in {0, 1}).
_CLIP = 1e-3


def normalize_model_name(name: str) -> str:
    """Short student-base label: strip ``EleutherAI/`` and ``pythia-160m-``; bare base -> ``original``.

    Args:
        name: A model name in any of the on-disk forms (``EleutherAI/pythia-160m-data-seed1``,
            ``pythia-160m-data-seed1``, ``pythia-160m``) or an already-short label.

    Returns:
        The short label (``data-seed1``, ``weight-seed2``, ``original``, ...). A fully
        re-randomized ``pythia-160m-seed<N>`` maps to ``full-seed<N>`` (matching
        ``spec_label``), never a bare ``seed<N>``. Unknown names that
        do not carry a recognised prefix pass through unchanged — with a warning for non-160m
        Pythia names, whose rows every 160m-keyed prior/label join here would silently drop.
    """
    name = name.removeprefix("EleutherAI/")
    if name == "pythia-160m":
        return "original"
    if name.startswith("pythia-") and not name.startswith("pythia-160m-"):
        warnings.warn(
            f"normalize_model_name: {name!r} is not a pythia-160m-family name; passing "
            "it through unchanged. Prior/label joins key on the 160m short labels, so "
            "these rows will not match — extend the mapping before analyzing non-160m runs.",
            stacklevel=2,
        )
        return name
    if name.startswith("pythia-160m-seed"):
        return f"full-{name[len('pythia-160m-'):]}"
    return name.removeprefix("pythia-160m-")


# --------------------------------------------------------------------------------------------------
# Design + run loading
# --------------------------------------------------------------------------------------------------


def load_focus20_design(root: Path) -> tuple[pd.DataFrame, list[str], dict]:
    """Load the focus-20 student-HP cell design and the shared 20-teacher param-tag list.

    Reads ``run_logs/focus20_cells.json`` (the 20 shared student cells) for ``cells_df`` and prefers
    ``run_logs/dog_top20krungs_configs.json`` for the teacher ``param_tags`` (falling back to
    ``run_logs/cat_top20_bandext_configs.json``). When both teacher files exist their ``param_tags``
    are asserted identical -- the campaign deliberately holds the teacher list fixed across animals.

    Args:
        root: Repo root (holds ``run_logs/``).

    Returns:
        ``(cells_df, param_tags, meta)`` where ``cells_df`` has one row per student cell with
        ``student_tag``/``lr``/``rank``/``alpha``/``K_rel``, ``param_tags`` is the 20 teacher tags,
        and ``meta`` carries the design's ``selection``/``epochs``/``student_seed``.
    """
    root = Path(root)
    design = read_json(root / "run_logs" / "focus20_cells.json")
    cells_df = pd.DataFrame(design["cells"])

    dog_path = root / "run_logs" / "dog_top20krungs_configs.json"
    cat_path = root / "run_logs" / "cat_top20_bandext_configs.json"
    dog_tags = read_json(dog_path)["param_tags"] if dog_path.exists() else None
    cat_tags = read_json(cat_path)["param_tags"] if cat_path.exists() else None
    if dog_tags is not None and cat_tags is not None:
        assert dog_tags == cat_tags, "dog/cat teacher param_tags disagree; campaign assumes them shared"
    param_tags = dog_tags if dog_tags is not None else cat_tags
    assert param_tags is not None, "no teacher config JSON found under run_logs/"

    meta = {k: design.get(k) for k in ("selection", "epochs", "student_seed")}
    return cells_df, list(param_tags), meta


# The three bandext126 sub-manifests whose union is core65 (Angela 2026-08-18 terminology):
# focus20 (0.645-0.950) + band48 (0.463-1.247, quota 8/rank) + lowk9 (0.290-0.398).
_CORE65_MANIFESTS: tuple[tuple[str, str], ...] = (
    ("focus20", "focus20_cells.json"),
    ("band48", "band_0p45_1p25_q8_cells.json"),
    ("lowk9", "band_lowk_0p29_0p45_cells.json"),
)
# Teachers that realize core65 on every arm: the shared dog-ranked top-10 kband list (identical
# file content for every animal; = the first 10 of the focus-20 top-20 list).
_CORE65_TEACHERS = "dog_top10_kband_configs.json"


def load_core65_design(root: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load the core65 student-cell design (focus20 + band48 + lowk9) and its top-10 teacher list.

    core65 is the cross-arm target set the band-ext campaigns realize on all 24 (animal x base)
    arms: 65 ep20 cells at K/K* 0.290-1.247, all inside the bandext126 universe. The 12 cells
    shared by focus20 and band48 are kept once (``subset`` names the first manifest that lists a
    cell, in the order focus20 > band48 > lowk9).

    Args:
        root: Repo root (holds ``run_logs/``).

    Returns:
        ``(cells_df, param_tags)``: one row per cell with ``student_tag``/``lr``/``rank``/
        ``alpha``/``K_rel``/``subset``; ``param_tags`` is the 10 shared teacher tags.
    """
    root = Path(root)
    frames = []
    for subset, fname in _CORE65_MANIFESTS:
        d = read_json(root / "run_logs" / fname)
        df = pd.DataFrame(d["cells"])[["student_tag", "lr", "rank", "alpha", "K_rel"]]
        frames.append(df.assign(subset=subset))
    cells_df = (pd.concat(frames, ignore_index=True)
                .drop_duplicates("student_tag", keep="first")
                .sort_values("K_rel", ignore_index=True))
    assert len(cells_df) == 65, f"core65 manifests union to {len(cells_df)} cells, expected 65"
    param_tags = list(read_json(root / "run_logs" / _CORE65_TEACHERS)["param_tags"])
    return cells_df, param_tags


def load_focus_runs(
    output_root: Path,
    param_tags: list[str],
    cell_tags,
    animals: list[str] | None = None,
    bases: list[str] | None = None,
) -> pd.DataFrame:
    """Load every landed focus-20 seed-run, filtered to the 20 teacher tags and 20 focus cells.

    Wraps `collapse_law.load_law_runs` once per animal (missing lineage dirs are silently skipped by
    that loader -- the desired graceful degradation) and keeps only the focus cells; older campaigns
    leave extra cells in the same directories, which are dropped here.

    Args:
        output_root: The ``outputs/`` root holding ``students/``.
        param_tags: The 20 shared teacher param tags (used as the lineage family/tag).
        cell_tags: Either the design ``cells_df`` (a frame with ``student_tag``/``K_rel``, whose
            ``K_rel`` is joined on) or a bare list of the focus student cell tags (then ``K_rel``
            is NaN).
        animals: Animals to load (default `FOCUS_ANIMALS`).
        bases: Short base labels to load (default `FOCUS_BASE_LABELS`).

    Returns:
        One row per landed seed-run with columns ``animal``, ``base_label``, ``teacher_tag``,
        ``student_tag``, ``lr``, ``rank``, ``alpha``, ``K_rel`` (from the design), ``seed``,
        ``test``. Empty (correct columns, zero rows) when nothing has landed.
    """
    animals = FOCUS_ANIMALS if animals is None else animals
    bases = FOCUS_BASE_LABELS if bases is None else bases
    if isinstance(cell_tags, pd.DataFrame):
        krel = cell_tags.set_index("student_tag")["K_rel"].to_dict()
        cell_set = set(cell_tags["student_tag"])
    else:
        krel = {}
        cell_set = set(cell_tags)
    full_bases = [f"pythia-160m-{b}" for b in bases]

    cols = ["animal", "base_label", "teacher_tag", "student_tag", "lr", "rank", "alpha",
            "K_rel", "seed", "test"]
    frames: list[pd.DataFrame] = []
    for animal in animals:
        lineages = [(t, t, b) for t in param_tags for b in full_bases]
        _, runs = load_law_runs(output_root, animal=animal, lineages=lineages)
        if len(runs) == 0:
            continue
        runs = runs[runs["student_tag"].isin(cell_set)].copy()
        if len(runs) == 0:
            continue
        runs["animal"] = animal
        runs["K_rel"] = runs["student_tag"].map(krel).astype(float)
        frames.append(runs[cols])

    if not frames:
        return pd.DataFrame(columns=cols)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------------------------------
# Same-model teacher quality
# --------------------------------------------------------------------------------------------------


def _pick_dataset(lineage_dir: Path) -> Path | None:
    """Choose the number-pool dataset dir under a same-model lineage, preferring the n100k pool.

    Globs ``ex10-20_prefix_n*`` and drops the ``_spliced`` variant. Returns the n100k pool when
    present, else any remaining pool (e.g. cat's n40k), else ``None``.
    """
    cands = [p for p in lineage_dir.glob("ex10-20_prefix_n*")
             if p.is_dir() and not p.name.endswith("_spliced")]
    if not cands:
        return None
    for p in cands:
        if "n100k" in p.name:
            return p
    return sorted(cands)[0]


def same_model_scores(
    output_root: Path,
    param_tags: list[str] | None,
    animals: list[str] | None = None,
    student_cells: tuple[str, ...] = ("ep20_lr1e-4_r8a32",),
) -> pd.DataFrame:
    """Same-model distillation scores: each teacher distilled back into its own base (pythia-160m).

    Reads ``students/<animal>/pythia-160m/teacher-pythia-160m/<tag>_tseed1/<dataset>/<cell>`` for the
    std-recipe epoch ladder, using `frames.read_cell_set(cell, "test")` for ``p_target_mean`` (a
    single-ftseed screen). The dataset tag is discovered by globbing (n100k preferred; cat's pool is
    n40k). Missing lineages/cells are silently absent -- graceful degradation.

    Args:
        output_root: The ``outputs/`` root holding ``students/``.
        param_tags: The teacher param tags to look up, or ``None`` to discover every ``*_tseed1``
            same-model lineage on disk (removes the focus-20 selection restriction).
        animals: Animals to scan (default `FOCUS_ANIMALS`).
        student_cells: Std-recipe student cells to read (default the ep20 cell; pass the fuller
            ladder ``ep{1,2,5,10,20}_lr1e-4_r8a32`` when a cross-animal-common cell is needed --
            ep20 exists for dog only, ep10 for all four animals).

    Returns:
        One row per (``animal``, ``teacher_tag``, ``student_cell``) with ``sm_p`` (p_target_mean),
        ``n_seeds``, ``dataset``. Empty (typed) when nothing lands.
    """
    from subliminal_learning.analysis.frames import read_cell_set

    animals = FOCUS_ANIMALS if animals is None else animals
    output_root = Path(output_root)
    cols = ["animal", "teacher_tag", "student_cell", "sm_p", "n_seeds", "dataset"]
    rows: list[dict] = []
    for animal in animals:
        base = output_root / "students" / animal / "pythia-160m" / "teacher-pythia-160m"
        if not base.exists():
            continue
        if param_tags is None:  # discover every same-model lineage on disk
            tags = sorted(p.name[: -len("_tseed1")] for p in base.glob("*_tseed1") if p.is_dir())
        else:
            tags = param_tags
        for tag in tags:
            ds_dir = _pick_dataset(base / f"{tag}_tseed1")
            if ds_dir is None:
                continue
            for cell in student_cells:
                test = read_cell_set(ds_dir / cell, "test")
                if test is None or test.get("p_target_mean") is None:
                    continue
                per_seed = test.get("per_seed_p_target") or {}
                rows.append({
                    "animal": animal, "teacher_tag": tag, "student_cell": cell,
                    "sm_p": float(test["p_target_mean"]),
                    "n_seeds": len(per_seed) if per_seed else 1,
                    "dataset": ds_dir.name,
                })
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)[cols]


def _pairwise_spearman(wide: pd.DataFrame, levels: list, min_n: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pairwise-complete Spearman ``rho`` and shared-row count ``n`` between the columns ``levels``.

    Diagonal is ``rho=1`` with ``n`` = that level's own non-NaN count; off-diagonal cells with fewer
    than ``min_n`` shared rows come back ``rho=NaN`` (``n`` still reported).
    """
    from scipy.stats import spearmanr

    rho = pd.DataFrame(np.nan, index=levels, columns=levels, dtype=float)
    n = pd.DataFrame(0, index=levels, columns=levels, dtype=int)
    for a in levels:
        for b in levels:
            pair = wide[[a, b]].dropna()
            n.loc[a, b] = len(pair)
            if a == b:
                rho.loc[a, b] = 1.0
            elif len(pair) >= min_n and pair[a].nunique() > 1 and pair[b].nunique() > 1:
                rho.loc[a, b] = float(spearmanr(pair[a], pair[b])[0])
    return rho, n


def teacher_generality(
    sm: pd.DataFrame, student_cell: str = "ep20_lr1e-4_r8a32"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cross-animal rank agreement of same-model teacher quality -- are some teachers generally good?

    Args:
        sm: `same_model_scores` output.
        student_cell: Which std cell to use.

    Returns:
        ``(matrix, table, n_matrix)`` where ``matrix`` is the animal x animal Spearman
        rank-correlation of the teachers' ``sm_p`` (pairwise-complete over the shared teachers; NaN
        when <3 shared), ``n_matrix`` is the matching pairwise shared-teacher count, and ``table``
        is per-teacher: ``sm_p`` per animal (columns), ``mean_rank`` (higher sm_p -> rank 1) across
        animals, and ``n_animals`` present.
    """
    sub = sm[sm["student_cell"] == student_cell]
    wide = sub.pivot_table(index="teacher_tag", columns="animal", values="sm_p")
    animals = [a for a in FOCUS_ANIMALS if a in wide.columns]

    mat, n_mat = _pairwise_spearman(wide, animals)

    ranks = wide[animals].rank(ascending=False)  # rank 1 = best (highest sm_p) within an animal
    table = wide[animals].copy()
    table["mean_rank"] = ranks.mean(axis=1)
    table["n_animals"] = wide[animals].notna().sum(axis=1)
    table = table.sort_values("mean_rank")
    return mat, table, n_mat


def teacher_order_agreement(
    cells: pd.DataFrame,
    by: str = "base_label",
    value: str = "cond",
    unit: tuple[str, ...] = ("animal", "teacher_tag"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Do teachers keep the same strength order across the levels of ``by``?

    Aggregates ``cells`` to a per-``unit`` mean of ``value`` at each level of ``by`` (e.g. per
    (animal, teacher) mean ``cond`` in each base), then reports the pairwise-complete Spearman
    ``rho`` and shared-``unit`` count ``n`` between every pair of ``by`` levels. Use
    ``unit=("teacher_tag",)`` on a single-animal slice for a per-animal matrix, or the default
    ``("animal", "teacher_tag")`` for the pooled-across-animals view.

    Args:
        cells: `hurdle_cells` output (or any frame carrying ``unit``, ``by`` and ``value``).
        by: Column whose levels index the matrix (default ``base_label``).
        value: The per-cell response to average (default ``cond``; NaN rows dropped).
        unit: Columns identifying the paired unit across ``by`` levels.

    Returns:
        ``(rho_df, n_df)`` over the levels of ``by`` (sorted).
    """
    df = cells.dropna(subset=[value])
    agg = df.groupby([*unit, by])[value].mean().reset_index()
    wide = agg.pivot_table(index=list(unit), columns=by, values=value)
    levels = list(wide.columns)
    return _pairwise_spearman(wide, levels)


# --------------------------------------------------------------------------------------------------
# Hurdle split
# --------------------------------------------------------------------------------------------------


def hurdle_cells(runs: pd.DataFrame, dead: "float | pd.Series" = DEAD) -> pd.DataFrame:
    """Collapse seed-runs to one row per (animal, base, teacher, cell) with a hurdle split.

    Args:
        runs: Seed-level rows from `load_focus_runs`.
        dead: Collapse threshold; a seed-run is dead when ``test < dead``. Either one scalar or a
            per-row Series aligned to ``runs.index`` (see `prior_dead_thresholds`).

    Returns:
        One row per (``animal``, ``base_label``, ``teacher_tag``, ``student_tag``) with:
        ``n_seeds``, ``n_alive``, ``surv`` (survival rate), ``cond`` (mean ``test`` over surviving
        seeds, NaN when none survive), ``cond_logit`` (logit of clipped ``cond``, NaN when ``cond``
        is NaN), and the THRESHOLD-FREE pair ``mean_test`` (mean ``test`` over ALL seeds) /
        ``mean_logit`` (its clipped logit) -- the unconditional response that involves no dead
        label at all. ``lr``/``rank``/``alpha``/``K_rel`` ride along.
    """
    keys = ["animal", "base_label", "teacher_tag", "student_tag"]
    meta_cols = ["lr", "rank", "alpha", "K_rel"]
    if len(runs) == 0:
        return pd.DataFrame(columns=keys + meta_cols + [
            "n_seeds", "n_alive", "surv", "cond", "cond_logit", "mean_test", "mean_logit"])

    alive = runs["test"] >= dead
    work = runs.assign(_alive=alive.astype(int), _alive_test=np.where(alive, runs["test"], np.nan))
    rows = []
    for key, g in work.groupby(keys, sort=True):
        n_seeds = len(g)
        n_alive = int(g["_alive"].sum())
        cond = float(g["_alive_test"].mean()) if n_alive > 0 else np.nan
        rec = dict(zip(keys, key))
        for m in meta_cols:
            rec[m] = g[m].iloc[0]
        rec.update(n_seeds=n_seeds, n_alive=n_alive, surv=n_alive / n_seeds, cond=cond,
                   mean_test=float(g["test"].mean()))
        rows.append(rec)
    out = pd.DataFrame(rows)
    for src_col, dst in (("cond", "cond_logit"), ("mean_test", "mean_logit")):
        cp = out[src_col].clip(_CLIP, 1 - _CLIP)
        out[dst] = np.where(out[src_col].notna(), np.log(cp / (1 - cp)), np.nan)
    return out


# --------------------------------------------------------------------------------------------------
# Priors
# --------------------------------------------------------------------------------------------------


# Short base labels used in ``analysis/base_eval_*_animal_preference_test.json`` filenames.
_BASE_EVAL_SHORT: dict[str, str] = {
    "original": "original", "data-seed1": "ds1", "data-seed2": "ds2", "data-seed3": "ds3",
    "weight-seed1": "ws1", "weight-seed2": "ws2", "weight-seed3": "ws3",
}


def load_priors(output_root: Path) -> pd.DataFrame:
    """Load the measured student-base animal priors, one row per (animal, base_label).

    - **logprob prior** (``animal_selection/curated_scores.json``): the base model's single-token
      log-probability of the animal word, plus its within-model rank across the candidate list.
    - **behavioral prior** (``frames.load_pref_wide`` favorite rows): the base model's own
      ``base_p_target`` on the teacher-eval **favorite** (train) prompts.
    - **test prior** (``analysis/base_eval_*_animal_preference_test.json``): the base model's mean
      exact p(target) on the held-out **test** prompts -- the same 50 prompts the student ``test``
      score is computed on, so this (not ``behav_prior``) is the like-for-like DEAD reference.
      The two differ because they are different prompt sets (e.g. dog/ds1: 0.0119 vs 0.0133).

    Args:
        output_root: The ``outputs/`` root.

    Returns:
        One row per (``animal``, ``base_label``) with ``logprob_prior``, ``prior_rank``,
        ``behav_prior``, ``test_prior`` (any of which may be NaN if that measure is absent).
    """
    from subliminal_learning.analysis.frames import load_pref_wide, short_model
    from subliminal_learning.teachers.animal_selection_io import load_selection_scores

    output_root = Path(output_root)
    sel = load_selection_scores(output_root / "animal_selection" / "curated_scores.json")
    sel = sel.assign(base_label=sel["model"].map(normalize_model_name))
    logp = sel.rename(columns={"logprob": "logprob_prior", "rank": "prior_rank"})[
        ["animal", "base_label", "logprob_prior", "prior_rank"]
    ]

    pref, _ = load_pref_wide(output_root)
    fav = pref[pref["set"] == "favorite"].copy()
    fav["base_label"] = fav["model"].map(short_model)
    behav = (
        fav.drop_duplicates(["animal", "base_label"])[["animal", "base_label", "base_p_target"]]
        .rename(columns={"base_p_target": "behav_prior"})
    )

    out = logp.merge(behav, on=["animal", "base_label"], how="outer")

    trows = []
    for base_label, short in _BASE_EVAL_SHORT.items():
        f = output_root / "analysis" / f"base_eval_{short}_animal_preference_test.json"
        if not f.exists():
            continue
        d = read_json(f)
        for animal, entry in d.items():
            if isinstance(entry, dict) and "p_per_prompt" in entry:
                trows.append({"animal": animal, "base_label": base_label,
                              "test_prior": float(np.mean(entry["p_per_prompt"]))})
    test = pd.DataFrame(trows, columns=["animal", "base_label", "test_prior"])
    return out.merge(test, on=["animal", "base_label"], how="outer")


# Default DEAD margin below the prior, in nats (threshold = prior * exp(-margin)). Calibrated
# 2026-08-15 on the dog control students in log-relative units log(test / own-base test prior):
# gentle-HP traitless training drifts scores down to -2.5 nats (rng controls' min; same-base
# controls drift UP instead), the collapse cluster starts below -4, and the region between is
# nearly empty -- so 3.0 clears every non-collapsed traitless control while sitting in the flat
# part of the law campaign's score continuum (label counts move ~2% per nat there).
DEAD_LOG_MARGIN: float = 3.0


def prior_dead_thresholds(
    runs: pd.DataFrame,
    priors: pd.DataFrame,
    prior_col: str = "behav_prior",
    animal: str | None = None,
    log_margin: float = 0.0,
) -> pd.Series:
    """Per-row dead threshold: the student base's own measured prior on the target animal.

    Replaces the single arbitrary scalar (`DEAD` / `collapse_law.DEAD_SEED`) with the base model's
    prior for that (animal, base) pair, optionally backed off by a multiplicative log-space margin:
    a seed is "dead" when ``test < prior * exp(-log_margin)``. With ``log_margin=0`` a seed is dead
    as soon as it scores below the untrained base; with `DEAD_LOG_MARGIN` (3 nats, i.e. prior/20)
    "dead" means the catastrophic-collapse mode only -- scores no traitless control student ever
    drops to. Missing priors raise rather than silently labelling every seed alive.

    Args:
        runs: Seed-level frame carrying ``base_label`` (and ``animal`` unless `animal` is given).
        priors: `load_priors` output.
        prior_col: Prior column to use as the reference. ``"test_prior"`` is the like-for-like
            choice for student ``test`` scores; ``"behav_prior"`` (default for back-compat) is the
            favorite-set measurement.
        animal: Fixed animal for frames without an ``animal`` column (e.g. `load_law_runs` output).
        log_margin: Nats subtracted from the log-prior (`DEAD_LOG_MARGIN` for collapse labelling).

    Returns:
        A float Series aligned to ``runs.index``, usable as the ``dead`` / ``dead_seed`` argument
        of `hurdle_cells` and `collapse_law.label_collapse`.
    """
    key = priors.set_index(["animal", "base_label"])[prior_col]
    animals = pd.Series(animal, index=runs.index) if animal is not None else runs["animal"]
    idx = pd.MultiIndex.from_arrays([animals, runs["base_label"]])
    out = pd.Series(key.reindex(idx).to_numpy(), index=runs.index, name="dead_threshold")
    if out.isna().any():
        missing = sorted(set(idx[out.isna().to_numpy()].tolist()))
        raise ValueError(f"missing {prior_col} for (animal, base_label) pairs: {missing}")
    return out * float(np.exp(-log_margin))


# --------------------------------------------------------------------------------------------------
# Integrity (Pile KL) dead labels
# --------------------------------------------------------------------------------------------------


def load_focus_pile_kl(
    output_root: Path,
    runs: pd.DataFrame,
    numbers_base: str = "pythia-160m",
    teacher_seed: int = 1,
    dataset: str = "ex10-20_prefix_n100k",
) -> pd.Series:
    """Per-seed-run Pile KL(base || student) from the run-dir probe record, aligned to ``runs``.

    Reads ``{run_dir}/eval/pile_probe.json`` (`coherence.probe_io`, the `scripts.pile_probe_final`
    / ``run_final_evals --evals probe`` output) for every row of a `load_focus_runs` frame. The
    run dir is rebuilt from the row's ``animal``/``base_label``/``teacher_tag``/``student_tag``/
    ``seed`` with the same layout `collapse_law.load_law_runs` walks.

    Returns:
        Float Series ``pile_kl`` aligned to ``runs.index``: the record's ``kl_base_student``;
        ``+inf`` when the record says ``nonfinite_adapter`` (diverged weights -- dead by any cut);
        NaN when no record has landed (not probed yet), so coverage is visible, not imputed.
    """
    from subliminal_learning.analysis.probe_cache import load_probe_cache
    from subliminal_learning.teachers.run_layout import model_dir_parts_from_name

    full_base = {b: f"{numbers_base}-{b}" for b in runs["base_label"].unique()}
    rel_paths = [
        str(
            Path("students") / r.animal / Path(*model_dir_parts_from_name(full_base[r.base_label]))
            / f"teacher-{numbers_base}" / f"{r.teacher_tag}_tseed{teacher_seed}" / dataset
            / r.student_tag / f"ftseed{int(r.seed)}"
        )
        for r in runs.itertuples(index=False)
    ]
    # One cached sweep instead of one read per row; the cache already stores +inf for a
    # diverged adapter and simply has no row for a run that has not been probed yet (-> NaN).
    cached = load_probe_cache(output_root, rel_paths, pool_version=None)
    kl = dict(zip(cached["run_path"], cached["kl"])) if len(cached) else {}
    vals = np.array([float(kl.get(p, np.nan)) for p in rel_paths])
    return pd.Series(vals, index=runs.index, name="pile_kl")


def load_pile_kl_population(
    output_root: Path,
    bases: list[str] | None = None,
    animals: list[str] | None = None,
    numbers_base: str = "pythia-160m",
    epochs: int = 20,
) -> pd.DataFrame:
    """Every probed final adapter under the given student bases -- the distribution-context view.

    Used to show each base model's full probe-KL distribution around the declared dead cut
    (`coherence.probe_io.DEAD_KL_CUT`) and to quote cut sensitivity: it loads the widest probed
    population of each base, all ``{animal}/{base}/teacher-{numbers_base}/*/*/ep{E}_*/
    ftseed*/eval/pile_probe.json`` records (every teacher lineage, every number pool, every
    landed cell -- strip ladders and band-ext campaigns included).

    Returns:
        One row per probed seed-run: ``animal``, ``base_label``, ``teacher_lineage``, ``dataset``,
        ``student_tag``, ``seed``, ``pile_kl`` (``+inf`` for non-finite adapters). Empty frame
        with those columns when nothing is probed.
    """
    from subliminal_learning.analysis.probe_cache import load_probe_cache
    from subliminal_learning.coherence.probe_io import EVAL_NAME
    from subliminal_learning.teachers.run_layout import model_dir_parts_from_name

    bases = FOCUS_BASE_LABELS if bases is None else bases
    animals = FOCUS_ANIMALS if animals is None else animals
    output_root = Path(output_root)
    students_root = output_root / "students"
    found = []                  # (row meta, run dir relative to output_root)
    for animal in animals:
        for b in bases:
            model_dir = students_root / animal / Path(*model_dir_parts_from_name(f"{numbers_base}-{b}"))
            tdir = model_dir / f"teacher-{numbers_base}"
            if not tdir.is_dir():
                continue
            for f in tdir.glob(f"*/*/ep{epochs}_*/ftseed*/eval/{EVAL_NAME}"):
                seed_dir = f.parents[1]
                found.append(({
                    "animal": animal, "base_label": b,
                    "teacher_lineage": seed_dir.parents[2].name,
                    "dataset": seed_dir.parents[1].name,
                    "student_tag": seed_dir.parent.name,
                    "seed": int(seed_dir.name.removeprefix("ftseed")),
                }, str(seed_dir.relative_to(output_root))))
    # Values through the shared cache -- ONE sweep for the whole glob, +inf for diverged
    # adapters (its own convention); an unreadable/partial record simply has no row, as before.
    cached = load_probe_cache(output_root, [rel for _, rel in found], pool_version=None)
    kl = dict(zip(cached["run_path"], cached["kl"])) if len(cached) else {}
    rows = [{**meta, "pile_kl": float(kl[rel])} for meta, rel in found if rel in kl]
    cols = ["animal", "base_label", "teacher_lineage", "dataset", "student_tag", "seed", "pile_kl"]
    return pd.DataFrame(rows, columns=cols)




# --------------------------------------------------------------------------------------------------
# FE + prior regression
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PriorFit:
    """Result of one FE + prior WLS fit.

    Attributes:
        beta: Prior coefficient (on the z-scored prior).
        se: Cluster-robust standard error (clustered on the animal x base pair).
        p: Two-sided p-value from the cluster-robust covariance (small-cluster caveat: only a few
            animal x base clusters, so this p is optimistic -- prefer the permutation test).
        nobs: Rows in the fit.
        n_clusters: Distinct animal x base clusters.
        prior_col: Which prior column was used.
        response: ``"cond"`` or ``"surv"``.
        model: The fitted statsmodels results object (carries the full summary).
    """

    beta: float
    se: float
    p: float
    nobs: int
    n_clusters: int
    prior_col: str
    response: str
    model: object


def _prep_fit_frame(
    cells: pd.DataFrame, priors: pd.DataFrame, prior_col: str, response: str
) -> pd.DataFrame:
    """Merge cells with the prior, build the response, z-score the prior on the fit frame.

    ``response='mean'`` uses ``mean_logit`` (unconditional mean test over ALL seeds -- the
    threshold-free response; all cells kept, weight = seed count). ``response='cond'`` uses
    ``cond_logit`` (surviving-only, NaN cells dropped); ``response='surv'`` uses the empirical
    logit of survival with a 0.5 continuity correction (all cells kept) -- both depend on a dead
    label. The z-scoring is done on the merged frame actually entering the fit, never globally.
    """
    df = cells.merge(priors[["animal", "base_label", prior_col]], on=["animal", "base_label"], how="left")
    na = df[prior_col].isna()
    if na.any():
        dropped = sorted(set(zip(df.loc[na, "animal"], df.loc[na, "base_label"])))
        warnings.warn(
            f"_prep_fit_frame: dropping {int(na.sum())} of {len(df)} cells with no "
            f"{prior_col!r} for (animal, base_label) pairs {dropped} — these rows are "
            "excluded from the fit, not treated as zero.",
            stacklevel=3,
        )
    df = df[~na].copy()
    if response == "mean":
        df["y"] = df["mean_logit"]
        df["w"] = df["n_seeds"].astype(float)
    elif response == "cond":
        df = df[df["cond"].notna()].copy()
        df["y"] = df["cond_logit"]
        df["w"] = df["n_alive"].astype(float)
    elif response == "surv":
        a = df["n_alive"] + 0.5
        b = df["n_seeds"] - df["n_alive"] + 0.5
        df["y"] = np.log(a / b)
        df["w"] = df["n_seeds"].astype(float)
    else:  # pragma: no cover - guarded by callers
        raise ValueError(f"unknown response {response!r}")
    df["pair"] = df["animal"].astype(str) + "|" + df["base_label"].astype(str)
    sd = df[prior_col].std(ddof=0)
    df["prior_z"] = 0.0 if sd == 0 else (df[prior_col] - df[prior_col].mean()) / sd
    return df


def fe_prior_fit(
    cells: pd.DataFrame, priors: pd.DataFrame, prior_col: str, response: str = "cond"
) -> PriorFit:
    """WLS of the response on ``C(animal) + C(base) + prior_z`` with cluster-robust inference.

    The animal FE absorbs teacher-side effects (incl. the teacher-base prior); the base FE absorbs
    student quirks; the z-scored prior's coefficient is identified by the within-animal cross-base
    contrast. Weights are total-seed counts (``mean``/``surv``) or surviving-seed counts
    (``cond``). The
    covariance is cluster-robust on the animal x base pair -- note the small number of clusters.

    Args:
        cells: `hurdle_cells` output.
        priors: `load_priors` output.
        prior_col: ``"logprob_prior"`` / ``"behav_prior"`` / ``"prior_rank"``.
        response: ``"mean"`` (unconditional, threshold-free), ``"cond"`` (conditional transfer)
            or ``"surv"`` (survival).

    Returns:
        A `PriorFit`.
    """
    import statsmodels.formula.api as smf

    df = _prep_fit_frame(cells, priors, prior_col, response)
    res = smf.wls("y ~ C(animal) + C(base_label) + prior_z", data=df, weights=df["w"]).fit(
        cov_type="cluster", cov_kwds={"groups": df["pair"]}
    )
    return PriorFit(
        beta=float(res.params["prior_z"]),
        se=float(res.bse["prior_z"]),
        p=float(res.pvalues["prior_z"]),
        nobs=int(res.nobs),
        n_clusters=int(df["pair"].nunique()),
        prior_col=prior_col,
        response=response,
        model=res,
    )


def prior_permutation_test(
    cells: pd.DataFrame,
    priors: pd.DataFrame,
    prior_col: str,
    response: str = "cond",
    n_perm: int = 2000,
    seed: int = 0,
) -> dict:
    """Within-animal permutation test of the prior slope on the FE-residualized response.

    Residualizes the response by ``C(animal) + C(base)`` (weighted), aggregates the residual to a
    weighted mean per (animal x base), and takes the OLS slope of that residual on the (z-scored)
    prior as the observed statistic. The null permutes the prior across bases WITHIN each animal
    (the identification contrast), so the animal and base marginals are preserved. This is the
    clean nonparametric inference given the tiny cluster count.

    Args:
        cells: `hurdle_cells` output.
        priors: `load_priors` output.
        prior_col: Prior column to test.
        response: ``"mean"``, ``"cond"`` or ``"surv"``.
        n_perm: Permutation replicates.
        seed: numpy RNG seed (fixed -> deterministic).

    Returns:
        ``{"obs_slope", "p_two_sided", "perm_slopes", "n_pairs"}``.
    """
    import statsmodels.formula.api as smf

    df = _prep_fit_frame(cells, priors, prior_col, response)
    # Residualize the response on the animal + base fixed effects (weighted).
    resid_model = smf.wls("y ~ C(animal) + C(base_label)", data=df, weights=df["w"]).fit()
    df = df.assign(_resid=resid_model.resid)

    # Aggregate to one weighted-mean residual per animal x base; prior is constant within a pair.
    agg = (
        df.groupby(["animal", "base_label"])
        .apply(lambda g: pd.Series({
            "resid": np.average(g["_resid"], weights=g["w"]),
            "prior_z": g["prior_z"].iloc[0],
        }), include_groups=False)
        .reset_index()
    )

    def slope(prior_vals: np.ndarray) -> float:
        x = prior_vals - prior_vals.mean()
        denom = float((x * x).sum())
        return float((x * (agg["resid"].to_numpy() - agg["resid"].mean())).sum() / denom) if denom > 0 else 0.0

    obs = slope(agg["prior_z"].to_numpy())

    rng = np.random.default_rng(seed)
    perm_slopes = np.empty(n_perm)
    animal_idx = {a: agg.index[agg["animal"] == a].to_numpy() for a in agg["animal"].unique()}
    prior_base = agg["prior_z"].to_numpy()
    for i in range(n_perm):
        permuted = prior_base.copy()
        for idx in animal_idx.values():
            permuted[idx] = rng.permutation(permuted[idx])
        perm_slopes[i] = slope(permuted)

    # +1 correction (observed statistic counted as one replicate): the smallest reportable
    # p is 1/(n_perm+1), never an indefensible exact 0. Strictly conservative.
    p = float((1 + int((np.abs(perm_slopes) >= abs(obs) - 1e-12).sum())) / (1 + n_perm))
    return {"obs_slope": obs, "p_two_sided": p, "perm_slopes": perm_slopes, "n_pairs": len(agg)}


# --------------------------------------------------------------------------------------------------
# Base effects / ds-vs-ws
# --------------------------------------------------------------------------------------------------


def _base_group(base_label: str) -> str:
    """``data-seed*`` -> ``ds``; ``weight-seed*`` -> ``ws``; anything else -> ``other``."""
    if base_label.startswith("data-seed"):
        return "ds"
    if base_label.startswith("weight-seed"):
        return "ws"
    return "other"


def _fit_base_fe(df: pd.DataFrame, prior_col: str | None) -> pd.Series:
    """Fit WLS (``cond_logit ~ C(animal) + C(base)[+prior_z]``) and return centered base effects.

    The base effects are recovered from the dummy coefficients (reference base = 0) and centered to
    a weighted-zero mean across the bases present, so they are comparable with/without the prior.
    """
    import statsmodels.formula.api as smf

    formula = "y ~ C(animal) + C(base_label)"
    if prior_col is not None:
        formula += " + prior_z"
    res = smf.wls(formula, data=df, weights=df["w"]).fit()

    bases = sorted(df["base_label"].unique())
    eff = {}
    for b in bases:
        key = f"C(base_label)[T.{b}]"
        eff[b] = float(res.params.get(key, 0.0))  # reference base -> 0
    s = pd.Series(eff)
    wmean = df.groupby("base_label")["w"].sum().reindex(s.index).fillna(0.0)
    center = float(np.average(s.to_numpy(), weights=wmean.to_numpy())) if wmean.sum() > 0 else s.mean()
    return s - center


def base_effects(
    cells: pd.DataFrame,
    priors: pd.DataFrame | None = None,
    prior_col: str | None = None,
    n_boot: int = 1000,
    seed: int = 0,
    response: str = "cond",
) -> pd.DataFrame:
    """Fitted student-base fixed effects (centered) with a teacher-clustered bootstrap CI.

    Response is the conditional-transfer logit (``response="cond"``, default) or the
    threshold-free unconditional ``mean_logit`` (``response="mean"``, weight = seed count).
    When ``priors``/``prior_col`` are given the prior term
    is included so the base effects are net of the measured prior. CIs come from a bootstrap that
    resamples whole teachers (the natural cluster) 1000x. Honest about n: only 2-3 bases per group.

    Args:
        cells: `hurdle_cells` output.
        priors: `load_priors` output (optional).
        prior_col: Prior column to adjust for (optional; requires ``priors``).
        n_boot: Bootstrap replicates.
        seed: numpy RNG seed.

    Returns:
        One row per ``base_label`` with ``fe``, ``ci_lo``, ``ci_hi``, ``group`` (``ds``/``ws``).
    """
    if response == "mean":
        df = cells.copy()
        df["y"] = df["mean_logit"]
        df["w"] = df["n_seeds"].astype(float)
    elif response == "cond":
        df = cells[cells["cond"].notna()].copy()
        df["y"] = df["cond_logit"]
        df["w"] = df["n_alive"].astype(float)
    else:  # pragma: no cover - guarded by callers
        raise ValueError(f"unknown response {response!r}")
    if prior_col is not None and priors is not None:
        df = df.merge(priors[["animal", "base_label", prior_col]], on=["animal", "base_label"], how="left")
        df = df[df[prior_col].notna()].copy()
        sd = df[prior_col].std(ddof=0)
        df["prior_z"] = 0.0 if sd == 0 else (df[prior_col] - df[prior_col].mean()) / sd
        use_prior = prior_col
    else:
        use_prior = None

    point = _fit_base_fe(df, use_prior)

    rng = np.random.default_rng(seed)
    teachers = df["teacher_tag"].unique()
    draws = {b: [] for b in point.index}
    for _ in range(n_boot):
        pick = rng.choice(teachers, size=len(teachers), replace=True)
        boot = pd.concat([df[df["teacher_tag"] == t] for t in pick], ignore_index=True)
        try:
            eff = _fit_base_fe(boot, use_prior)
        except Exception:  # noqa: BLE001 - a degenerate resample carries no base info
            continue
        for b in point.index:
            draws[b].append(eff.get(b, np.nan))

    rows = []
    for b in point.index:
        d = np.array([v for v in draws[b] if np.isfinite(v)])
        lo, hi = (np.percentile(d, [2.5, 97.5]) if len(d) else (np.nan, np.nan))
        rows.append({"base_label": b, "fe": float(point[b]), "ci_lo": float(lo),
                     "ci_hi": float(hi), "group": _base_group(b)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------------
# Factor 1 -- teacher imprint
# --------------------------------------------------------------------------------------------------


def teacher_covariates(
    output_root: Path, param_tags: list[str], samemodel_cell: str | None = "ep20_lr1e-4_r8a32"
) -> pd.DataFrame:
    """Per (animal, teacher) covariates for the factor-1 check.

    From `frames.load_pref_wide`: the teacher base's own trait score ``teacher_p`` (favorite-set
    ``p_target`` for model ``pythia-160m`` on the 20 teacher tags -- a same-model, seed-averaged
    proxy; the campaign teachers are tseed 1 specifically, so this is a proxy). When
    ``samemodel_cell`` is given, `same_model_scores` is joined in as ``samemodel_p`` -- the real
    single-ftseed same-model screen at that std cell, which (unlike ``teacher_p``) is genuinely
    animal-specific. A ``screen_p`` column is included only if a per-teacher screen score exists in
    the manifests (none currently do, so it is all-NaN and callers should treat it as absent).

    Args:
        output_root: The ``outputs/`` root.
        param_tags: The 20 teacher param tags.
        samemodel_cell: Std same-model cell for ``samemodel_p`` (``None`` to skip; note ep20 exists
            for dog only, ep10 for all four animals).

    Returns:
        One row per (``animal``, ``teacher_tag``) with ``teacher_p``, ``samemodel_p`` (NaN where
        absent), and ``screen_p``.
    """
    from subliminal_learning.analysis.frames import load_pref_wide

    pref, _ = load_pref_wide(output_root)
    fav = pref[(pref["set"] == "favorite") & (pref["model"] == "pythia-160m")
               & (pref["param_tag"].isin(param_tags))]
    cov = (
        fav.groupby(["animal", "param_tag"])["p_target"].mean().reset_index()
        .rename(columns={"param_tag": "teacher_tag", "p_target": "teacher_p"})
    )
    if samemodel_cell is not None:
        sm = same_model_scores(output_root, param_tags, student_cells=(samemodel_cell,))
        smj = sm[["animal", "teacher_tag", "sm_p"]].rename(columns={"sm_p": "samemodel_p"})
        cov = cov.merge(smj, on=["animal", "teacher_tag"], how="outer")
    else:
        cov["samemodel_p"] = np.nan
    cov["screen_p"] = np.nan  # no per-teacher screen score in the current manifests
    return cov


def factor1_table(
    cells: pd.DataFrame,
    teacher_cov: pd.DataFrame,
    covariate: str = "teacher_p",
    response: str = "cond",
) -> pd.DataFrame:
    """Per-animal correlation of per-teacher cross-init transfer against a teacher covariate.

    For each animal, averages the per-teacher response over bases x cells to one value per teacher,
    joins the teacher covariate, and reports Pearson ``r`` and Spearman ``rho`` (with n). A pooled
    row (``animal == "POOLED"``) reports the within-animal partial correlation -- Pearson r between
    the response and the covariate after both are residualized on ``C(animal)`` -- plus ``beta``/``p``
    from an animal-FE OLS of transfer on the covariate.

    Args:
        cells: `hurdle_cells` output.
        teacher_cov: `teacher_covariates` output.
        covariate: Column in ``teacher_cov`` to correlate against (default ``teacher_p``).
        response: Per-teacher transfer measure -- ``"mean"`` (unconditional mean test over all
            cells, threshold-free), ``"cond"`` (mean conditional transfer over surviving cells)
            or ``"surv"`` (mean survival rate over cells).

    Returns:
        One row per animal plus a ``POOLED`` row, with ``r``, ``rho``, ``n``.
    """
    from scipy.stats import pearsonr, spearmanr

    if response == "mean":
        src = cells.groupby(["animal", "teacher_tag"])["mean_test"].mean()
    elif response == "cond":
        src = cells[cells["cond"].notna()].groupby(["animal", "teacher_tag"])["cond"].mean()
    elif response == "surv":
        src = cells.groupby(["animal", "teacher_tag"])["surv"].mean()
    else:  # pragma: no cover - guarded by callers
        raise ValueError(f"unknown response {response!r}")

    per_teacher = (
        src.rename("y").reset_index()
        .merge(teacher_cov[["animal", "teacher_tag", covariate]], on=["animal", "teacher_tag"], how="left")
        .dropna(subset=["y", covariate])
    )

    rows = []
    for animal, g in per_teacher.groupby("animal"):
        n = len(g)
        if n >= 3 and g[covariate].nunique() > 1 and g["y"].nunique() > 1:
            r = float(pearsonr(g[covariate], g["y"])[0])
            rho = float(spearmanr(g[covariate], g["y"])[0])
        else:
            r = rho = np.nan
        rows.append({"animal": animal, "r": r, "rho": rho, "n": n})

    # Pooled within-animal partial correlation: residualize BOTH the response and the covariate on
    # the animal fixed effects, then Pearson r between the residuals (so the animal offsets do not
    # leak into r). beta/p come from the animal-FE OLS of the response on the covariate.
    pooled = {"animal": "POOLED", "r": np.nan, "rho": np.nan, "n": len(per_teacher)}
    if per_teacher["animal"].nunique() >= 1 and len(per_teacher) > per_teacher["animal"].nunique() + 1:
        import statsmodels.formula.api as smf
        try:
            y_res = smf.ols("y ~ C(animal)", data=per_teacher).fit().resid
            x_res = smf.ols(f"{covariate} ~ C(animal)", data=per_teacher).fit().resid
            if y_res.std() > 0 and x_res.std() > 0:
                pooled["r"] = float(pearsonr(x_res, y_res)[0])
            res = smf.ols(f"y ~ C(animal) + {covariate}", data=per_teacher).fit()
            pooled["beta"] = float(res.params[covariate])
            pooled["p"] = float(res.pvalues[covariate])
        except Exception:  # noqa: BLE001
            pass
    rows.append(pooled)
    return pd.DataFrame(rows)
