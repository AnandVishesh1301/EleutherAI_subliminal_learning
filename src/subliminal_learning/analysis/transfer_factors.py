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

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from subliminal_learning.analysis.collapse_law import load_law_runs

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
        The short label (``data-seed1``, ``weight-seed2``, ``original``, ...). Unknown names that
        do not carry a recognised prefix pass through unchanged.
    """
    name = name.removeprefix("EleutherAI/")
    if name == "pythia-160m":
        return "original"
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
    design = json.loads((root / "run_logs" / "focus20_cells.json").read_text())
    cells_df = pd.DataFrame(design["cells"])

    dog_path = root / "run_logs" / "dog_top20krungs_configs.json"
    cat_path = root / "run_logs" / "cat_top20_bandext_configs.json"
    dog_tags = json.loads(dog_path.read_text())["param_tags"] if dog_path.exists() else None
    cat_tags = json.loads(cat_path.read_text())["param_tags"] if cat_path.exists() else None
    if dog_tags is not None and cat_tags is not None:
        assert dog_tags == cat_tags, "dog/cat teacher param_tags disagree; campaign assumes them shared"
    param_tags = dog_tags if dog_tags is not None else cat_tags
    assert param_tags is not None, "no teacher config JSON found under run_logs/"

    meta = {k: design.get(k) for k in ("selection", "epochs", "student_seed")}
    return cells_df, list(param_tags), meta


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
# Hurdle split
# --------------------------------------------------------------------------------------------------


def hurdle_cells(runs: pd.DataFrame, dead: float = DEAD) -> pd.DataFrame:
    """Collapse seed-runs to one row per (animal, base, teacher, cell) with a hurdle split.

    Args:
        runs: Seed-level rows from `load_focus_runs`.
        dead: Collapse threshold; a seed-run is dead when ``test < dead``.

    Returns:
        One row per (``animal``, ``base_label``, ``teacher_tag``, ``student_tag``) with:
        ``n_seeds``, ``n_alive``, ``surv`` (survival rate), ``cond`` (mean ``test`` over surviving
        seeds, NaN when none survive), ``cond_logit`` (logit of clipped ``cond``, NaN when ``cond``
        is NaN). ``lr``/``rank``/``alpha``/``K_rel`` ride along.
    """
    keys = ["animal", "base_label", "teacher_tag", "student_tag"]
    meta_cols = ["lr", "rank", "alpha", "K_rel"]
    if len(runs) == 0:
        return pd.DataFrame(columns=keys + meta_cols + ["n_seeds", "n_alive", "surv", "cond", "cond_logit"])

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
        rec.update(n_seeds=n_seeds, n_alive=n_alive, surv=n_alive / n_seeds, cond=cond)
        rows.append(rec)
    out = pd.DataFrame(rows)
    cp = out["cond"].clip(_CLIP, 1 - _CLIP)
    out["cond_logit"] = np.where(out["cond"].notna(), np.log(cp / (1 - cp)), np.nan)
    return out


# --------------------------------------------------------------------------------------------------
# Priors
# --------------------------------------------------------------------------------------------------


def load_priors(output_root: Path) -> pd.DataFrame:
    """Load both measured student-base animal priors, one row per (animal, base_label).

    - **logprob prior** (``animal_selection/curated_scores.json``): the base model's single-token
      log-probability of the animal word, plus its within-model rank across the candidate list.
    - **behavioral prior** (``frames.load_pref_wide`` favorite rows): the base model's own
      ``base_p_target`` on the actual eval prompts. NaN where a measure is missing for a cell.

    Args:
        output_root: The ``outputs/`` root.

    Returns:
        One row per (``animal``, ``base_label``) with ``logprob_prior``, ``prior_rank``,
        ``behav_prior`` (any of which may be NaN if that measure is absent).
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

    return logp.merge(behav, on=["animal", "base_label"], how="outer")


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

    ``response='cond'`` uses ``cond_logit`` (surviving-only, NaN cells dropped); ``response='surv'``
    uses the empirical logit of survival with a 0.5 continuity correction (all cells kept). The
    z-scoring is done on the merged frame actually entering the fit, never globally.
    """
    df = cells.merge(priors[["animal", "base_label", prior_col]], on=["animal", "base_label"], how="left")
    df = df[df[prior_col].notna()].copy()
    if response == "cond":
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
    contrast. Weights are surviving-seed counts (``cond``) or total-seed counts (``surv``). The
    covariance is cluster-robust on the animal x base pair -- note the small number of clusters.

    Args:
        cells: `hurdle_cells` output.
        priors: `load_priors` output.
        prior_col: ``"logprob_prior"`` / ``"behav_prior"`` / ``"prior_rank"``.
        response: ``"cond"`` (conditional transfer) or ``"surv"`` (survival).

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
        response: ``"cond"`` or ``"surv"``.
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
) -> pd.DataFrame:
    """Fitted student-base fixed effects (centered) with a teacher-clustered bootstrap CI.

    Response is conditional-transfer logit. When ``priors``/``prior_col`` are given the prior term
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
    df = cells[cells["cond"].notna()].copy()
    df["y"] = df["cond_logit"]
    df["w"] = df["n_alive"].astype(float)
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


def teacher_covariates(output_root: Path, param_tags: list[str]) -> pd.DataFrame:
    """Per (animal, teacher) covariates for the factor-1 check.

    From `frames.load_pref_wide`: the teacher base's own trait score ``teacher_p`` (favorite-set
    ``p_target`` for model ``pythia-160m`` on the 20 teacher tags -- a same-model, seed-averaged
    proxy; the campaign teachers are tseed 1 specifically, so this is a proxy). A ``screen_p``
    column is included only if a per-teacher screen score exists in the manifests (none currently
    do, so it is all-NaN and callers should treat it as absent).

    Args:
        output_root: The ``outputs/`` root.
        param_tags: The 20 teacher param tags.

    Returns:
        One row per (``animal``, ``teacher_tag``) with ``teacher_p`` and ``screen_p``.
    """
    from subliminal_learning.analysis.frames import load_pref_wide

    pref, _ = load_pref_wide(output_root)
    fav = pref[(pref["set"] == "favorite") & (pref["model"] == "pythia-160m")
               & (pref["param_tag"].isin(param_tags))]
    cov = (
        fav.groupby(["animal", "param_tag"])["p_target"].mean().reset_index()
        .rename(columns={"param_tag": "teacher_tag", "p_target": "teacher_p"})
    )
    cov["screen_p"] = np.nan  # no per-teacher screen score in the current manifests
    return cov


def factor1_table(cells: pd.DataFrame, teacher_cov: pd.DataFrame, covariate: str = "teacher_p") -> pd.DataFrame:
    """Per-animal correlation of per-teacher cross-init transfer against a teacher covariate.

    For each animal, averages conditional transfer over bases x cells to one value per teacher,
    joins the teacher covariate, and reports Pearson ``r`` and Spearman ``rho`` (with n). A pooled
    row (``animal == "POOLED"``) reports the within-animal partial correlation -- Pearson r between
    ``cond`` and the covariate after both are residualized on ``C(animal)`` -- plus ``beta``/``p``
    from an animal-FE OLS of transfer on the covariate.

    Args:
        cells: `hurdle_cells` output.
        teacher_cov: `teacher_covariates` output.
        covariate: Column in ``teacher_cov`` to correlate against (default ``teacher_p``).

    Returns:
        One row per animal plus a ``POOLED`` row, with ``r``, ``rho``, ``n``.
    """
    from scipy.stats import pearsonr, spearmanr

    per_teacher = (
        cells[cells["cond"].notna()]
        .groupby(["animal", "teacher_tag"])["cond"].mean().reset_index()
        .merge(teacher_cov[["animal", "teacher_tag", covariate]], on=["animal", "teacher_tag"], how="left")
        .dropna(subset=["cond", covariate])
    )

    rows = []
    for animal, g in per_teacher.groupby("animal"):
        n = len(g)
        if n >= 3 and g[covariate].nunique() > 1 and g["cond"].nunique() > 1:
            r = float(pearsonr(g[covariate], g["cond"])[0])
            rho = float(spearmanr(g[covariate], g["cond"])[0])
        else:
            r = rho = np.nan
        rows.append({"animal": animal, "r": r, "rho": rho, "n": n})

    # Pooled within-animal partial correlation: residualize BOTH cond and the covariate on the
    # animal fixed effects, then Pearson r between the residuals (so the animal offsets do not leak
    # into r). beta/p come from the animal-FE OLS of cond on the covariate.
    pooled = {"animal": "POOLED", "r": np.nan, "rho": np.nan, "n": len(per_teacher)}
    if per_teacher["animal"].nunique() >= 1 and len(per_teacher) > per_teacher["animal"].nunique() + 1:
        import statsmodels.formula.api as smf
        try:
            y_res = smf.ols("cond ~ C(animal)", data=per_teacher).fit().resid
            x_res = smf.ols(f"{covariate} ~ C(animal)", data=per_teacher).fit().resid
            if y_res.std() > 0 and x_res.std() > 0:
                pooled["r"] = float(pearsonr(x_res, y_res)[0])
            res = smf.ols(f"cond ~ C(animal) + {covariate}", data=per_teacher).fit()
            pooled["beta"] = float(res.params[covariate])
            pooled["p"] = float(res.pvalues[covariate])
        except Exception:  # noqa: BLE001
            pass
    rows.append(pooled)
    return pd.DataFrame(rows)
