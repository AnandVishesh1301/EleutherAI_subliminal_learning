"""Stage-transition selection logic (pure functions on loader DataFrames).

Inputs are the tidy frames produced by `evaluation.gkd_results_io`
(`load_gkd_probe_trajectories`, `load_gkd_preference_results`); nothing here
touches disk, torch, or the Hub. Every function fails loud on missing or
degenerate data rather than returning a plausible wrong answer.
"""

from __future__ import annotations

import math
from collections.abc import Sequence, Set
from dataclasses import dataclass

import pandas as pd

from subliminal_learning.gkd.layout import DEFAULT_OPTIM, optim_slug


@dataclass(frozen=True)
class Recipe:
    """One student LoRA recipe: rank, alpha, learning rate.

    Attributes:
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha.
        lr: Student learning rate.
    """

    lora_r: int
    lora_alpha: int
    lr: float

    @property
    def mode_token(self) -> str:
        """The sweep ``--student-modes`` token, e.g. ``lora-r8a32``."""
        return f"lora-r{self.lora_r}a{self.lora_alpha}"


@dataclass(frozen=True)
class RecipeSelection:
    """One (animal, beta)'s A3 candidate set with its full selection provenance.

    Attributes:
        recipes: The arms to dose-ladder, best-first: `by_final` in order, then the
            recipes only `by_peak` found, then any lr-edge extension.
        by_final: Top-k ranked on the end-of-run probe.
        by_peak: Top-k ranked on the trajectory peak.
        ranking: One dict per candidate recipe with both scores, for the run log
            and downstream notebooks (never re-derived from `recipes`).
    """

    recipes: tuple[Recipe, ...]
    by_final: tuple[Recipe, ...]
    by_peak: tuple[Recipe, ...]
    ranking: tuple[dict, ...]


@dataclass(frozen=True)
class FrozenRecipe:
    """The per-beta recipe freeze outcome.

    Attributes:
        recipe: Winning (r, alpha, lr).
        num_epochs: Winning dose.
        subliminal_delta: Cross-animal mean test-set delta.
        per_animal_delta: Animal -> test-set delta at the winning cell.
    """

    recipe: Recipe
    num_epochs: float
    subliminal_delta: float
    per_animal_delta: dict[str, float]


def end_of_run_probes(probes: pd.DataFrame) -> pd.DataFrame:
    """Reduce a probe-trajectory frame to each leaf's last probe row.

    Args:
        probes: Output of `load_gkd_probe_trajectories`.

    Returns:
        One row per leaf (``run_dir``), at that leaf's max ``step``.
    """
    if probes.empty:
        return probes.copy()
    idx = probes.groupby("run_dir")["step"].idxmax()
    return probes.loc[idx].reset_index(drop=True)


def peak_probes(probes: pd.DataFrame) -> pd.DataFrame:
    """Reduce a probe-trajectory frame to each leaf's max-``p_target`` row.

    The trait is known to peak mid-run and decay (the campaign's dose-ladder
    motivation), so the last probe systematically understates a recipe that
    spikes early — this is the complementary reduction to `end_of_run_probes`.

    Args:
        probes: Output of `load_gkd_probe_trajectories`.

    Returns:
        One row per leaf (``run_dir``), at that leaf's max ``p_target``.
    """
    if probes.empty:
        return probes.copy()
    idx = probes.groupby("run_dir")["p_target"].idxmax()
    return probes.loc[idx].reset_index(drop=True)


_RANK_SORT_COLUMNS = [
    "p_target",
    "fc_share",
    "student_lora_r",
    "student_lora_alpha",
    "student_lr",
]
_RANK_SORT_ASCENDING = [False, False, True, True, True]


def _rank_cell(cell: pd.DataFrame) -> pd.DataFrame:
    """Rank one (animal, beta) reduced frame best-first, one row per recipe."""
    by_recipe = (
        cell.groupby(["student_lora_r", "student_lora_alpha", "student_lr"])
        .agg(p_target=("p_target", "mean"), fc_share=("fc_share", "mean"))
        .reset_index()
    )
    return by_recipe.sort_values(
        by=_RANK_SORT_COLUMNS, ascending=_RANK_SORT_ASCENDING
    ).reset_index(drop=True)


def _recipes_of(ranked: pd.DataFrame, k: int) -> list[Recipe]:
    return [
        Recipe(int(r.student_lora_r), int(r.student_lora_alpha), float(r.student_lr))
        for r in ranked.head(k).itertuples()
    ]


def _cell_frame(reduced: pd.DataFrame, animal: str, beta: float) -> pd.DataFrame:
    return reduced[
        (reduced["animal"] == animal)
        & reduced["beta"].apply(lambda b: math.isclose(float(b), float(beta)))
    ]


def select_top_recipes(
    probes: pd.DataFrame,
    *,
    animals: Sequence[str],
    betas: Sequence[float],
    k: int,
) -> dict[tuple[str, float], RecipeSelection]:
    """A3 candidate arms per (animal, beta) from the A1 same-init probes.

    Diverged leaves are dropped first; only non-control, same-init LoRA
    treatments are ranked. Each recipe is scored TWICE over its leaves (mean
    over gseeds), with ``fc_share`` as tiebreak and ``(r, alpha, lr)`` ascending
    as the deterministic final tiebreak:

    - **final** — the end-of-run probe ``p_target`` (what survives the full dose).
    - **peak** — the trajectory-max probe ``p_target`` (what the recipe reaches
      at its best step).

    The candidate set is the UNION of the two top-k lists, so a recipe that
    peaks hard and decays is dose-laddered instead of being ranked out by its
    tail. Ordering keeps the end-of-run winner at index 0 (the headline ``r1``
    arm that carries the A3 controls and W&B logging); peak-only recipes follow
    in peak order.

    Args:
        probes: Output of `load_gkd_probe_trajectories`.
        animals: Animals that must be present.
        betas: Betas that must be present.
        k: Recipes to keep per reducer (the union holds at most ``2 * k``).

    Returns:
        ``{(animal, beta): RecipeSelection}``.

    Raises:
        ValueError: If a requested (animal, beta) has no surviving recipe.
    """
    surviving = probes[
        ~probes["diverged"]
        & ~probes["is_control"]
        & ~probes["teacher_cross_init"].astype(bool)
        & (probes["student_mode"] == "lora")
    ]
    final_reduced = end_of_run_probes(surviving)
    peak_reduced = peak_probes(surviving)
    out: dict[tuple[str, float], RecipeSelection] = {}
    for animal in animals:
        for beta in betas:
            final_cell = _cell_frame(final_reduced, animal, beta)
            if final_cell.empty:
                raise ValueError(
                    f"no non-diverged same-init A1 recipe survives for "
                    f"animal={animal!r} beta={beta} — cannot select top-{k}"
                )
            final_ranked = _rank_cell(final_cell)
            peak_ranked = _rank_cell(_cell_frame(peak_reduced, animal, beta))
            by_final = _recipes_of(final_ranked, k)
            by_peak = _recipes_of(peak_ranked, k)
            union = list(dict.fromkeys([*by_final, *by_peak]))
            out[(animal, float(beta))] = RecipeSelection(
                recipes=tuple(union),
                by_final=tuple(by_final),
                by_peak=tuple(by_peak),
                ranking=_ranking_records(final_ranked, peak_ranked, by_final, by_peak),
            )
    return out


def _ranking_records(
    final_ranked: pd.DataFrame,
    peak_ranked: pd.DataFrame,
    by_final: Sequence[Recipe],
    by_peak: Sequence[Recipe],
) -> tuple[dict, ...]:
    """One record per recipe carrying both scores and both ranks (for the run log)."""
    peak_by_recipe = {
        Recipe(int(r.student_lora_r), int(r.student_lora_alpha), float(r.student_lr)): (
            i,
            float(r.p_target),
        )
        for i, r in enumerate(peak_ranked.itertuples())
    }
    records: list[dict] = []
    for i, r in enumerate(final_ranked.itertuples()):
        recipe = Recipe(
            int(r.student_lora_r), int(r.student_lora_alpha), float(r.student_lr)
        )
        peak_rank, peak_p = peak_by_recipe.get(recipe, (None, None))
        records.append(
            {
                "lora_r": recipe.lora_r,
                "lora_alpha": recipe.lora_alpha,
                "lr": recipe.lr,
                "final_p_target": float(r.p_target),
                "final_rank": i,
                "peak_p_target": peak_p,
                "peak_rank": peak_rank,
                "selected_by_final": recipe in by_final,
                "selected_by_peak": recipe in by_peak,
            }
        )
    return tuple(records)


def lr_edge_extensions(
    selections: dict[tuple[str, float], RecipeSelection],
    *,
    lrs: Sequence[float],
    factor: float,
) -> tuple[dict[tuple[str, float], list[Recipe]], tuple[str, ...]]:
    """Extension arms for (animal, beta) cells whose winner sits on an lr grid edge.

    A rank-1 recipe at the TOP of the screened lr ladder means the surface is
    censored — the optimum may be above the grid, and the campaign cannot state
    where the lr x alpha law turns over. Such a leader gets one extra A3 arm at
    ``top_lr * factor`` on the same (r, alpha), so the dose ladder measures
    whether the effect is still rising. A rank-1 at the BOTTOM rung is reported
    as a warning only (extending downward is the cheap, low-value direction and
    the floor is already covered by the A2 gate).

    Rank-1 under EITHER reducer counts, since both feed the candidate union; when
    the two reducers disagree and both leaders are censored, both are extended
    (at most 2 arms, which `plan.A3_MAX_SAME_ARMS` accounts for). An extension
    that already sits in the candidate list is reported, not re-added, so the
    persisted warnings never claim an arm that does not exist.

    Args:
        selections: `select_top_recipes` output.
        lrs: The screened lr ladder (`plan.A1_LORA_LRS`).
        factor: Multiplier applied to the top rung for the extension arm.

    Returns:
        ``({(animal, beta): [extension Recipe, ...]}, warnings)``. Both are
        empty when no cell is censored.
    """
    top, bottom = max(lrs), min(lrs)
    extensions: dict[tuple[str, float], list[Recipe]] = {}
    warnings: list[str] = []
    for key, sel in sorted(selections.items()):
        animal, beta = key
        leaders = {r for r in (sel.by_final[:1] + sel.by_peak[:1])}
        for leader in sorted(leaders, key=lambda r: (r.lora_r, r.lora_alpha, r.lr)):
            if math.isclose(leader.lr, top):
                extra = Recipe(leader.lora_r, leader.lora_alpha, leader.lr * factor)
                censored = (
                    f"{animal}/b{beta:g}: rank-1 recipe {leader.mode_token} sits at "
                    f"the TOP lr rung {top:g} — surface censored"
                )
                if extra in sel.recipes or extra in extensions.get(key, []):
                    warnings.append(
                        f"{censored}; lr={extra.lr:g} is already a candidate arm"
                    )
                    continue
                extensions.setdefault(key, []).append(extra)
                warnings.append(f"{censored}; adding A3 arm at lr={extra.lr:g}")
            elif math.isclose(leader.lr, bottom):
                warnings.append(
                    f"{animal}/b{beta:g}: rank-1 recipe {leader.mode_token} sits at "
                    f"the BOTTOM lr rung {bottom:g} — the optimum may be below the "
                    "screened grid; interpret the lr x alpha law as one-sided"
                )
    return extensions, tuple(warnings)


def apply_lr_extensions(
    selections: dict[tuple[str, float], RecipeSelection],
    extensions: dict[tuple[str, float], list[Recipe]],
) -> dict[tuple[str, float], RecipeSelection]:
    """Append `lr_edge_extensions` arms to each selection's candidate list."""
    return {
        key: (
            sel
            if key not in extensions
            else RecipeSelection(
                recipes=tuple([*sel.recipes, *extensions[key]]),
                by_final=sel.by_final,
                by_peak=sel.by_peak,
                ranking=sel.ranking,
            )
        )
        for key, sel in selections.items()
    }


@dataclass(frozen=True)
class ControlFloorResult:
    """Outcome of the control-floor flatness gate.

    Attributes:
        passed: Whether every (animal, beta, arm) floor is flat and low.
        failures: Human-readable reasons (empty when passed).
        floors: One row per (animal, beta, arm, student_lr) with the floor value.
    """

    passed: bool
    failures: tuple[str, ...]
    floors: pd.DataFrame


@dataclass(frozen=True)
class FloorArm:
    """One parameterization x optimizer arm and the lr ladder its floor must cover.

    Attributes:
        student_mode: ``"lora"`` or ``"full"``.
        lrs: The lr ladder for this arm (optimizer optima differ by decades, so the
            ladder is per arm rather than shared).
        student_optim: HF optimizer name, as `evaluation.gkd_results_io.parse_gkd_tag`
            reports it.
        optim_momentum: SGD momentum for this arm.
    """

    student_mode: str
    lrs: tuple[float, ...]
    student_optim: str = DEFAULT_OPTIM
    optim_momentum: float = 0.0

    @property
    def label(self) -> str:
        """Short arm label for failure messages, e.g. ``lora/sgdm0.9``."""
        return (
            f"{self.student_mode}/{optim_slug(self.student_optim, self.optim_momentum)}"
        )


FLOOR_COLUMNS = [
    "animal",
    "beta",
    "student_mode",
    "student_optim",
    "optim_momentum",
    "student_lr",
    "floor_p_target",
    "skipped",
]


def _floor_arm_key(arm: FloorArm) -> str:
    """`optparam_plan.Arm.key`-style id for matching a quarantine set."""
    return arm.label.replace("/", "_")


def _lr_quarantined(
    arm: FloorArm, lr: float, quarantined: Set[tuple[str, float]] | None
) -> bool:
    if not quarantined:
        return False
    key = _floor_arm_key(arm)
    return any(
        a == key and math.isclose(float(x), float(lr), rel_tol=1e-9, abs_tol=0.0)
        for a, x in quarantined
    )


def control_floor_check(
    probes: pd.DataFrame,
    *,
    animals: Sequence[str],
    betas: Sequence[float],
    lrs: Sequence[float] | None = None,
    max_floor: float,
    max_spread: float,
    arms: Sequence[FloorArm] | None = None,
    quarantined: Set[tuple[str, float]] | None = None,
) -> ControlFloorResult:
    """Gate: the base-teacher control floor must be low and flat across lr, per arm.

    Criterion, per (animal, beta, arm): every requested lr must have a non-diverged
    control leaf; the floor at each lr is the end-of-run probe ``p_target``
    (mean over gseeds). The gate FAILS when (a) any lr floor is missing or its
    leaf diverged, (b) any floor exceeds `max_floor` (absolute ceiling — the
    procedure alone must not install the trait), or (c) ``max(floor) -
    min(floor) > max_spread`` across that arm's lr ladder (a floor that rises with lr
    would confound a control-free screen, so the screen must be rerun with
    controls). A rise is a spread, so (c) subsumes "rises with lr" while also
    rejecting a floor that is erratic in either direction.

    Flatness is judged **within** an arm. Two arms whose lr ladders sit decades apart
    (SGD vs AdamW) would otherwise be pooled into one spread that means nothing.

    Quarantined ``(arm_key, lr)`` pairs are skipped for floor/spread checks but still
    emit a floors row with ``skipped=True``. ``quarantined=None`` (the A2 default)
    reproduces the pre-disposition gate decisions and failure messages.

    Args:
        probes: Output of `load_gkd_probe_trajectories`.
        animals: Animals the floor must cover.
        betas: Betas the floor must cover.
        lrs: The lr ladder, for the single-arm (LoRA, default optimizer) form. Ignored
            when `arms` is given; required otherwise.
        max_floor: Absolute ceiling on any control floor p_target.
        max_spread: Max allowed (max - min) floor across one arm's lr ladder.
        arms: Explicit parameterization x optimizer arms. ``None`` reproduces the
            original single-arm gate exactly.
        quarantined: ``(arm_key, lr)`` pairs excluded from the gate. Default ``None``
            keeps the 20260727 A2 call site unchanged.

    Returns:
        `ControlFloorResult` (never raises; the driver fails loud on
        ``passed=False``).

    Raises:
        ValueError: Neither `arms` nor `lrs` was given.
    """
    if arms is None:
        if lrs is None:
            raise ValueError("control_floor_check needs either `arms` or `lrs`")
        arms = (FloorArm(student_mode="lora", lrs=tuple(float(x) for x in lrs)),)
    ctrl = probes[probes["is_control"]]
    last = end_of_run_probes(ctrl)
    failures: list[str] = []
    floor_rows: list[dict] = []
    for animal in animals:
        for beta in betas:
            for arm in arms:
                floors = _arm_floors(
                    last,
                    animal,
                    float(beta),
                    arm,
                    failures,
                    floor_rows,
                    quarantined=quarantined,
                )
                active_lrs = [
                    lr for lr in arm.lrs if not _lr_quarantined(arm, lr, quarantined)
                ]
                if len(floors) < len(active_lrs):
                    continue  # missing/diverged failures already recorded
                if len(floors) < 1:
                    continue
                hi, lo = max(floors.values()), min(floors.values())
                where = f"{animal}/b{beta}/{arm.label}"
                if hi > max_floor:
                    failures.append(
                        f"{where}: control floor {hi:.4f} exceeds max_floor={max_floor}"
                    )
                if hi - lo > max_spread:
                    failures.append(
                        f"{where}: control floor spread {hi - lo:.4f} across "
                        f"lr exceeds max_spread={max_spread} (floor not flat — rerun "
                        "the screen with controls)"
                    )
    return ControlFloorResult(
        passed=not failures,
        failures=tuple(failures),
        floors=pd.DataFrame(floor_rows, columns=FLOOR_COLUMNS),
    )


def _arm_floors(
    last: pd.DataFrame,
    animal: str,
    beta: float,
    arm: FloorArm,
    failures: list[str],
    floor_rows: list[dict],
    *,
    quarantined: Set[tuple[str, float]] | None = None,
) -> dict[float, float]:
    """Per-lr control floor for one (animal, beta, arm); appends failures/rows in place."""
    in_arm = last[
        (last["animal"] == animal)
        & last["beta"].apply(lambda b: math.isclose(float(b), beta))
        & (last["student_mode"] == arm.student_mode)
        & _optim_mask(last, arm)
    ]
    floors: dict[float, float] = {}
    for lr in arm.lrs:
        cell = in_arm[_isclose_mask(in_arm["student_lr"], float(lr))]
        where = f"{animal}/b{beta}/{arm.label}/lr{lr:g}"
        skipped = _lr_quarantined(arm, lr, quarantined)
        if skipped:
            floor = float(cell["p_target"].mean()) if len(cell) else float("nan")
            floor_rows.append(
                {
                    "animal": animal,
                    "beta": beta,
                    "student_mode": arm.student_mode,
                    "student_optim": arm.student_optim,
                    "optim_momentum": arm.optim_momentum,
                    "student_lr": float(lr),
                    "floor_p_target": floor,
                    "skipped": True,
                }
            )
            continue
        if cell.empty:
            failures.append(f"{where}: no control leaf on disk")
            continue
        if cell["diverged"].any():
            failures.append(f"{where}: control diverged")
            continue
        floor = float(cell["p_target"].mean())
        floors[float(lr)] = floor
        floor_rows.append(
            {
                "animal": animal,
                "beta": beta,
                "student_mode": arm.student_mode,
                "student_optim": arm.student_optim,
                "optim_momentum": arm.optim_momentum,
                "student_lr": float(lr),
                "floor_p_target": floor,
                "skipped": False,
            }
        )
    return floors


def _optim_mask(frame: pd.DataFrame, arm: FloorArm) -> pd.Series:
    """Optimizer filter, tolerant of frames predating the optimizer axis."""
    if "student_optim" not in frame.columns:
        return pd.Series(True, index=frame.index)
    return (frame["student_optim"] == arm.student_optim) & frame[
        "optim_momentum"
    ].apply(lambda m: math.isclose(float(m), arm.optim_momentum))


def _isclose_mask(series: pd.Series, value: float) -> pd.Series:
    return series.apply(lambda x: math.isclose(float(x), float(value)))


def freeze_recipes(
    preference: pd.DataFrame,
    candidates: dict[tuple[str, float], Sequence[Recipe]],
    *,
    doses: Sequence[float],
    prompt_set: str = "test",
) -> dict[float, FrozenRecipe]:
    """Freeze one (r, alpha, lr, epochs) per beta from eval-grade test deltas.

    For each beta, considers every (recipe, dose) with ``recipe`` in that
    (animal, beta)'s A3 candidate list and evaluated summaries for BOTH
    animals. The score is the cross-animal mean of ``p_target_treatment -
    p_target_control`` on the held-out `prompt_set`, where the control is the
    matched base-teacher floor at the same (animal, model, beta, lmbda, dose).
    A3 trains controls only on the headline recipe, so the control join is
    relaxed from exact ``gkd_tag`` to (beta, dose) — licensed by the A2 gate's
    evidence that the floor is recipe-flat. Ties break on higher treatment
    ``p_target``, then smaller (epochs, r, alpha, lr).

    Args:
        preference: `load_gkd_preference_results` frame over the campaign tree.
        candidates: The A3 recipe lists (``RecipeSelection.recipes`` per cell).
        doses: The A3 epoch ladder.
        prompt_set: Held-out prompt set to score (``test``).

    Returns:
        ``{beta: FrozenRecipe}`` for every beta present in `candidates`.

    Raises:
        ValueError: If a beta has no (recipe, dose) covered by both animals
            with a matched control.
    """
    rows = preference[
        (preference["set"] == prompt_set) & (preference["scoring"] == "prob")
    ]
    treat = rows[
        ~rows["is_control"]
        & ~rows["teacher_cross_init"].astype(bool)
        & (rows["student_mode"] == "lora")
    ]
    ctrl = rows[rows["is_control"]]
    betas = sorted({beta for (_, beta) in candidates})
    animals = sorted({animal for (animal, _) in candidates})
    out: dict[float, FrozenRecipe] = {}
    for beta in betas:
        scored: list[tuple] = []
        recipe_dose_pairs = {
            (recipe, float(dose))
            for animal in animals
            for recipe in candidates[(animal, beta)]
            for dose in doses
        }
        for recipe, dose in recipe_dose_pairs:
            deltas: dict[str, float] = {}
            treats: dict[str, float] = {}
            for animal in animals:
                if recipe not in candidates[(animal, beta)]:
                    break
                t = treat[
                    (treat["animal"] == animal)
                    & _isclose_mask(treat["beta"], beta)
                    & _isclose_mask(treat["num_epochs"], dose)
                    & (treat["student_lora_r"] == recipe.lora_r)
                    & (treat["student_lora_alpha"] == recipe.lora_alpha)
                    & _isclose_mask(treat["student_lr"], recipe.lr)
                ]
                if t.empty:
                    break
                model = t.iloc[0]["model"]
                c = ctrl[
                    (ctrl["animal"] == animal)
                    & (ctrl["model"] == model)
                    & _isclose_mask(ctrl["beta"], beta)
                    & _isclose_mask(ctrl["num_epochs"], dose)
                ]
                if c.empty:
                    break
                treats[animal] = float(t["p_target"].mean())
                deltas[animal] = treats[animal] - float(c["p_target"].mean())
            if len(deltas) != len(animals):
                continue
            mean_delta = sum(deltas.values()) / len(deltas)
            mean_treat = sum(treats.values()) / len(treats)
            scored.append(
                (
                    -mean_delta,
                    -mean_treat,
                    dose,
                    recipe.lora_r,
                    recipe.lora_alpha,
                    recipe.lr,
                    recipe,
                    deltas,
                )
            )
        if not scored:
            raise ValueError(
                f"beta={beta}: no (recipe, dose) has test-set summaries for all "
                f"animals plus a matched control — cannot freeze"
            )
        scored.sort(key=lambda s: s[:6])
        best = scored[0]
        out[float(beta)] = FrozenRecipe(
            recipe=best[6],
            num_epochs=float(best[2]),
            subliminal_delta=-best[0],
            per_animal_delta=dict(best[7]),
        )
    return out


def headline_beta(frozen: dict[float, FrozenRecipe]) -> float:
    """The beta whose frozen recipe carries the larger test-set delta.

    Ties break toward beta=1.0 (the campaign's reverse-KL default), then the
    larger beta.
    """
    if not frozen:
        raise ValueError("empty freeze result")
    return max(
        frozen,
        key=lambda b: (
            frozen[b].subliminal_delta,
            math.isclose(b, 1.0),
            b,
        ),
    )
