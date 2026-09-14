"""Deterministic sampling math for the offline filtration-audit plan.

Torch-free and filesystem-free. Leaf RNGs are derived from
``(global_seed, run_dir_rel)`` so a plan is byte-reproducible regardless of
worker count or completion order.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Callable, Iterable, Mapping, Sequence, Set
from dataclasses import dataclass
from typing import TypeVar

QUEUE_FIELDS: tuple[str, ...] = (
    "row_id",
    "task",
    "ask_animal",
    "prompt",
    "completion",
)
QUEUE_FORBIDDEN_FIELDS: tuple[str, ...] = (
    "arm",
    "flagged",
    "matched_form",
    "run_dir",
    "step",
    "stratum",
    "sample_weight",
    "is_decoy",
)
KEYS_FIELDS: tuple[str, ...] = (
    "row_id",
    "task",
    "source",
    "run_dir",
    "animal",
    "arm",
    "step",
    "step_quintile",
    "batch_idx",
    "flagged",
    "matched_form",
    "ask_animal",
    "is_decoy",
    "stratum",
    "sample_weight",
    "doc_id",
    "register",
)
KEYS_FIELDS_WITH_RULE: tuple[str, ...] = (*KEYS_FIELDS, "matched_rule")

SCREENS: tuple[str, ...] = ("legacy", "strict")
ARM_LABELS: tuple[str, ...] = (
    "legacy_treatment",
    "legacy_control",
    "strict_treatment",
    "strict_control",
)
LEGACY_RULE = "legacy"

_T = TypeVar("_T")


@dataclass(frozen=True)
class SampledRollout:
    """One reservoir-kept rollout row."""

    step: int
    batch_idx: int
    prompt: str
    completion: str
    matched_form: str | None
    flagged: bool
    step_quintile: int
    matched_rule: str | None = None


@dataclass(frozen=True)
class DedupeReport:
    """Keep-last dedupe of one leaf's rollout rows.

    Attributes:
        n_rows_raw: Rows read from ``rollouts_sample.jsonl``.
        n_rows_kept: Distinct ``(step, batch_idx)`` rows kept.
        n_duplicates: ``n_rows_raw - n_rows_kept``.
    """

    n_rows_raw: int
    n_rows_kept: int
    n_duplicates: int


@dataclass(frozen=True)
class RolloutCensus:
    """Population counts over one leaf's (deduplicated) rollout rows.

    Attributes:
        n_rows: Rows.
        n_flagged: Flagged rows.
        form_counts: ``matched_form -> count`` over flagged rows.
        rule_counts: ``matched_rule -> count`` (strict leaves only, else empty).
        rule_form_counts: ``rule -> {form -> count}``; legacy leaves use rule
            `LEGACY_RULE`.
    """

    n_rows: int
    n_flagged: int
    form_counts: dict[str, int]
    rule_counts: dict[str, int]
    rule_form_counts: dict[str, dict[str, int]]

    @property
    def n_clean(self) -> int:
        """Unflagged rows."""
        return self.n_rows - self.n_flagged


@dataclass(frozen=True)
class LeafRolloutDraw:
    """Per-leaf clean/flagged draw and realised bookkeeping."""

    clean: tuple[SampledRollout, ...]
    flagged: tuple[SampledRollout, ...]
    clean_pop_by_quintile: tuple[int, ...]
    flagged_realised: dict[str, int]
    flagged_budgets: dict[str, int]


def step_quintile(step: int, n_steps: int, n_strata: int) -> int:
    """Map a training step to a stratum index in ``[0, n_strata)``.

    Short leaves keep ``n_strata`` bins; empty bins stay empty rather than
    collapsing to fewer strata.

    Args:
        step: Training step (0-based).
        n_steps: Step count that defines the bin edges.
        n_strata: Number of strata (quintiles when 5).

    Returns:
        Stratum index.

    Raises:
        ValueError: If ``n_steps`` or ``n_strata`` is not positive, or
            ``step`` is negative.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if n_strata < 1:
        raise ValueError(f"n_strata must be >= 1, got {n_strata}")
    if step < 0:
        raise ValueError(f"step must be >= 0, got {step}")
    return min(n_strata - 1, (step * n_strata) // n_steps)


def row_id(
    task: str,
    run_dir_rel: str,
    step: int | None,
    batch_idx: int | None,
    ask_animal: str | None,
) -> str:
    """Stable 16-hex-char id from the plan's field order.

    Args:
        task: Judge task name.
        run_dir_rel: Leaf path relative to the campaign root, or a pool
            identity key.
        step: Training step, or ``None`` for pool rows.
        batch_idx: In-step batch index, or ``None`` for pool rows.
        ask_animal: Animal the judge is asked about, or ``None``.

    Returns:
        ``blake2b(..., digest_size=8).hexdigest()``.
    """
    payload = f"{task}|{run_dir_rel}|{step}|{batch_idx}|{ask_animal}"
    return hashlib.blake2b(payload.encode(), digest_size=8).hexdigest()


def derived_seed(global_seed: int, *parts: object) -> int:
    """Deterministic 64-bit seed from ``global_seed`` and identity parts.

    Args:
        global_seed: Plan-level seed.
        *parts: Extra identity fragments (leaf path, task name, ...).

    Returns:
        Unsigned 64-bit integer.
    """
    payload = "|".join([str(global_seed), *[str(p) for p in parts]])
    digest = hashlib.blake2b(payload.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def leaf_rng_seed(global_seed: int, run_dir_rel: str) -> int:
    """Per-leaf seed; independent of worker count and completion order.

    Args:
        global_seed: Plan-level seed.
        run_dir_rel: Leaf path relative to the campaign root.

    Returns:
        Unsigned 64-bit integer.
    """
    return derived_seed(global_seed, run_dir_rel)


def make_leaf_rng(global_seed: int, run_dir_rel: str) -> random.Random:
    """RNG seeded by `leaf_rng_seed`."""
    return random.Random(leaf_rng_seed(global_seed, run_dir_rel))


def task_rng(global_seed: int, task: str) -> random.Random:
    """RNG for a post-merge operation (shuffle, recall subsample)."""
    return random.Random(derived_seed(global_seed, "task", task))


def clean_quota_per_quintile(rows_per_leaf_clean: int, n_strata: int) -> int:
    """Per-quintile clean budget; ``rows_per_leaf_clean`` must divide evenly.

    Args:
        rows_per_leaf_clean: Total clean rows requested per leaf.
        n_strata: Number of step strata.

    Returns:
        Rows requested from each quintile.

    Raises:
        ValueError: If ``n_strata < 1`` or the clean budget does not divide.
    """
    if n_strata < 1:
        raise ValueError(f"n_strata must be >= 1, got {n_strata}")
    if rows_per_leaf_clean < 0:
        raise ValueError(f"rows_per_leaf_clean must be >= 0, got {rows_per_leaf_clean}")
    if rows_per_leaf_clean % n_strata != 0:
        raise ValueError(
            f"rows_per_leaf_clean={rows_per_leaf_clean} is not divisible "
            f"by n_step_strata={n_strata}"
        )
    return rows_per_leaf_clean // n_strata


def allocate_flagged_budgets(
    population: Mapping[str, int],
    budget: int,
    surface_forms: Set[str],
    lexicon_oversample_factor: float,
    lexicon_floor: int,
) -> dict[str, int]:
    """Allocate a flagged budget with lexicon terms oversampled.

    Each non-surface stratum first receives ``min(pop, lexicon_floor)``.
    The remainder is Hamilton-allocated with lexicon share
    ``pop * lexicon_oversample_factor``. Caps at population; leftover from
    a short stratum is never moved onto another stratum.

    Args:
        population: Observed ``matched_form`` -> count.
        budget: Per-leaf flagged row cap.
        surface_forms: The leaf animal's own name forms.
        lexicon_oversample_factor: Share multiplier for non-surface forms.
        lexicon_floor: Guaranteed rows (capped by pop) per lexicon form.

    Returns:
        ``matched_form`` -> allocated count (may sum to less than ``budget``).
    """
    if budget < 0:
        raise ValueError(f"budget must be >= 0, got {budget}")
    if lexicon_oversample_factor <= 0:
        raise ValueError(
            f"lexicon_oversample_factor must be > 0, got {lexicon_oversample_factor}"
        )
    if lexicon_floor < 0:
        raise ValueError(f"lexicon_floor must be >= 0, got {lexicon_floor}")

    surface = {f.lower() for f in surface_forms}
    pop = {form: int(n) for form, n in population.items() if int(n) > 0}
    allocated = {form: 0 for form in pop}
    remaining = budget

    for form in sorted(f for f in pop if f.lower() not in surface):
        take = min(pop[form], lexicon_floor, remaining)
        allocated[form] += take
        remaining -= take

    residual = {form: pop[form] - allocated[form] for form in pop}
    if remaining <= 0 or not any(c > 0 for c in residual.values()):
        return allocated

    shares: dict[str, float] = {}
    for form, n in pop.items():
        if residual[form] <= 0:
            continue
        boost = 1.0 if form.lower() in surface else lexicon_oversample_factor
        shares[form] = float(n) * boost
    extra = _largest_remainder(shares, remaining)
    for form, n_extra in extra.items():
        allocated[form] += min(n_extra, residual[form])
    return allocated


def flagged_unsampled_population(
    population: Mapping[str, int],
    allocated: Mapping[str, int],
) -> tuple[int, int]:
    """Return flagged population mass that received a zero sample budget.

    Args:
        population: Observed ``matched_form`` -> count.
        allocated: ``matched_form`` -> allocated count.

    Returns:
        ``(unsampled_pop, total_pop)``. A form is unsampled iff ``pop > 0``
        and ``int(allocated.get(form, 0)) <= 0``.
    """
    unsampled = 0
    total = 0
    for form, n in population.items():
        pop = int(n)
        if pop <= 0:
            continue
        total += pop
        if int(allocated.get(form, 0)) <= 0:
            unsampled += pop
    return unsampled, total


def stratum_sample_weight(
    population_n: int,
    sample_n: int,
    population_total: int,
    sample_total: int,
) -> float:
    """Weight that restores a stratum's population share.

    Does not renormalise across strata, so a shortfall stays visible
    (weights of sampled strata sum to less than 1 when a stratum is empty).

    Args:
        population_n: Population count in this stratum.
        sample_n: Realised sample count in this stratum.
        population_total: Population across strata.
        sample_total: Realised sample across strata.

    Returns:
        Share-recovery weight, or ``0.0`` if a denominator is 0.
    """
    if sample_n <= 0 or population_total <= 0 or sample_total <= 0:
        return 0.0
    return (population_n / population_total) / (sample_n / sample_total)


def realised_stratum_weights(
    population: Mapping[str, int],
    realised: Mapping[str, int],
) -> dict[str, float]:
    """`stratum_sample_weight` for every realised stratum.

    Args:
        population: Population counts (typically from ``rollouts_summary``).
        realised: Realised sample counts (never padded).

    Returns:
        ``stratum -> weight``.
    """
    pop_total = sum(int(n) for n in population.values())
    sample_total = sum(int(n) for n in realised.values())
    return {
        form: stratum_sample_weight(
            int(population.get(form, 0)),
            int(n),
            pop_total,
            sample_total,
        )
        for form, n in realised.items()
    }


def pick_decoy_animal(
    target: str,
    headline_animals: Sequence[str],
    rng: random.Random,
) -> str:
    """Choose a headline animal other than ``target``.

    Args:
        target: The leaf's own animal.
        headline_animals: Animals eligible as decoys.
        rng: Seeded RNG.

    Returns:
        One decoy animal.

    Raises:
        ValueError: If no headline animal is distinct from ``target``.
    """
    key = target.strip().lower()
    others = [a for a in headline_animals if a.strip().lower() != key]
    if not others:
        raise ValueError(
            f"no decoy distinct from target {target!r} in {list(headline_animals)}"
        )
    return others[rng.randrange(len(others))]


def animals_owning_form(
    form: str,
    animal_forms: Mapping[str, Sequence[str]],
    trait_lexicon: Mapping[str, Sequence[str]],
) -> tuple[str, ...]:
    """Animals whose surface forms or lexicon contain ``form``.

    Args:
        form: Matched screen form.
        animal_forms: ``{animal: (form, ...)}``.
        trait_lexicon: ``{animal: (term, ...)}``.

    Returns:
        Sorted owning animals.
    """
    key = form.strip().lower()
    owners: set[str] = set()
    for animal, forms in animal_forms.items():
        if key in {f.lower() for f in forms}:
            owners.add(animal)
    for animal, terms in trait_lexicon.items():
        if key in {t.lower() for t in terms}:
            owners.add(animal)
    return tuple(sorted(owners))


def primary_animal_for_form(
    form: str,
    animal_forms: Mapping[str, Sequence[str]],
    trait_lexicon: Mapping[str, Sequence[str]],
    headline_animals: Sequence[str] = (),
) -> str | None:
    """Single ask-animal for a matched form (headline owners first).

    Args:
        form: Matched screen form.
        animal_forms: ``{animal: (form, ...)}``.
        trait_lexicon: ``{animal: (term, ...)}``.
        headline_animals: Preferred owners when the form is shared.

    Returns:
        One animal, or ``None`` if no owner is known.
    """
    owners = animals_owning_form(form, animal_forms, trait_lexicon)
    if not owners:
        return None
    headline = {a.strip().lower() for a in headline_animals}
    preferred = [a for a in owners if a.strip().lower() in headline]
    pool = preferred or list(owners)
    return sorted(pool)[0]


def reservoir_update(
    reservoir: list[_T],
    item: _T,
    seen: int,
    k: int,
    rng: random.Random,
) -> None:
    """Algorithm R: keep ``k`` items uniformly from ``seen`` arrivals.

    Args:
        reservoir: In-place sample (length ``<= k``).
        item: Newly arrived item.
        seen: 1-based count of arrivals in this stratum.
        k: Reservoir capacity.
        rng: Seeded RNG.
    """
    if k <= 0:
        return
    if seen < 1:
        raise ValueError(f"seen must be >= 1, got {seen}")
    if len(reservoir) < k:
        reservoir.append(item)
        return
    j = rng.randrange(seen)
    if j < k:
        reservoir[j] = item


def subsample_count(n: int, fraction: float) -> int:
    """Rounded ``n * fraction``, clamped to ``[0, n]``.

    Args:
        n: Population size.
        fraction: Subsample fraction.

    Returns:
        Requested subsample size.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if fraction < 0:
        raise ValueError(f"fraction must be >= 0, got {fraction}")
    return min(n, max(0, int(round(n * fraction))))


def subsample_indices(n: int, k: int, rng: random.Random) -> list[int]:
    """``k`` distinct indices in ``[0, n)``, sorted.

    Args:
        n: Population size.
        k: Sample size.
        rng: Seeded RNG.

    Returns:
        Sorted indices. All of ``range(n)`` when ``k >= n``.
    """
    if k <= 0 or n <= 0:
        return []
    if k >= n:
        return list(range(n))
    chosen = rng.sample(range(n), k)
    chosen.sort()
    return chosen


def leaf_artifact_hash(run_dir_rel: str) -> str:
    """Filename stem for ``plan/leaf_samples/{hash}.jsonl``.

    Args:
        run_dir_rel: Leaf path relative to the campaign root.

    Returns:
        16-hex-char blake2b digest.
    """
    return hashlib.blake2b(run_dir_rel.encode(), digest_size=8).hexdigest()


def sample_rollout_rows(
    rows: Iterable[Mapping[str, object]],
    n_steps: int,
    n_strata: int,
    clean_per_quintile: int,
    flagged_budgets: Mapping[str, int],
    rng: random.Random,
    *,
    flagged_key: Callable[[SampledRollout], str | None] | None = None,
) -> LeafRolloutDraw:
    """One-pass reservoir draw of clean (by quintile) and flagged (by form).

    Args:
        rows: Rollout mappings with ``step``, ``batch_idx``, ``prompt``,
            ``completion``, ``flagged``, ``matched_form`` (and optionally
            ``matched_rule``).
        n_steps: Step count used for quintile edges.
        n_strata: Number of step strata.
        clean_per_quintile: Reservoir size per quintile.
        flagged_budgets: Flagged stratum -> reservoir size.
        rng: Per-leaf seeded RNG.
        flagged_key: Flagged stratum of a row; default ``matched_form``.

    Returns:
        The draw plus realised counts. Short strata are not padded.
    """
    clean_res: list[list[SampledRollout]] = [[] for _ in range(n_strata)]
    clean_seen = [0] * n_strata
    flagged_res: dict[str, list[SampledRollout]] = {
        form: [] for form in flagged_budgets
    }
    flagged_seen = {form: 0 for form in flagged_budgets}

    for raw in rows:
        flagged = bool(raw["flagged"])
        step = int(raw["step"])
        q = step_quintile(step, n_steps, n_strata)
        form_raw = raw.get("matched_form")
        form = None if form_raw is None else str(form_raw)
        rule_raw = raw.get("matched_rule")
        item = SampledRollout(
            step=step,
            batch_idx=int(raw["batch_idx"]),
            prompt=str(raw["prompt"]),
            completion=str(raw["completion"]),
            matched_form=form,
            flagged=flagged,
            step_quintile=q,
            matched_rule=None if rule_raw is None else str(rule_raw),
        )
        if flagged:
            key = form if flagged_key is None else flagged_key(item)
            if key is None or key not in flagged_budgets:
                continue
            flagged_seen[key] += 1
            reservoir_update(
                flagged_res[key],
                item,
                flagged_seen[key],
                int(flagged_budgets[key]),
                rng,
            )
            continue
        clean_seen[q] += 1
        reservoir_update(
            clean_res[q],
            item,
            clean_seen[q],
            clean_per_quintile,
            rng,
        )

    clean = tuple(
        row for bucket in clean_res for row in sorted(bucket, key=_rollout_sort_key)
    )
    flagged = tuple(
        row
        for form in sorted(flagged_res)
        for row in sorted(flagged_res[form], key=_rollout_sort_key)
    )
    return LeafRolloutDraw(
        clean=clean,
        flagged=flagged,
        clean_pop_by_quintile=tuple(clean_seen),
        flagged_realised={form: len(flagged_res[form]) for form in flagged_res},
        flagged_budgets=dict(flagged_budgets),
    )


def assert_queue_blind(payload: Mapping[str, object]) -> None:
    """Raise if a queue row is not exactly the blind schema.

    Args:
        payload: Candidate queue object.

    Raises:
        AssertionError: Forbidden or extra keys are present, or a required
            key is missing.
    """
    leaked = [k for k in QUEUE_FORBIDDEN_FIELDS if k in payload]
    if leaked:
        raise AssertionError(f"blind queue leaked fields {leaked}")
    keys = set(payload)
    missing = [k for k in QUEUE_FIELDS if k not in keys]
    extra = sorted(keys - set(QUEUE_FIELDS))
    if missing or extra:
        raise AssertionError(
            f"blind queue schema mismatch missing={missing} extra={extra}"
        )


def blind_queue_row(
    row_id_value: str,
    task: str,
    ask_animal: str | None,
    prompt: str,
    completion: str,
) -> dict[str, str | None]:
    """Build a blind queue row and assert the schema.

    Args:
        row_id_value: `row_id` hex digest.
        task: Judge task.
        ask_animal: Animal asked about, or ``None``.
        prompt: Prefix text.
        completion: Student (or pool) continuation.

    Returns:
        ``{row_id, task, ask_animal, prompt, completion}``.
    """
    payload: dict[str, str | None] = {
        "row_id": row_id_value,
        "task": task,
        "ask_animal": ask_animal,
        "prompt": prompt,
        "completion": completion,
    }
    assert_queue_blind(payload)
    return payload


def make_key_row(
    *,
    row_id_value: str,
    task: str,
    source: str,
    run_dir: str | None,
    animal: str | None,
    arm: str | None,
    step: int | None,
    step_quintile_value: int | None,
    batch_idx: int | None,
    flagged: bool | None,
    matched_form: str | None,
    ask_animal: str | None,
    is_decoy: bool,
    stratum: str | None,
    sample_weight: float,
    doc_id: str | None,
    register: str | None,
    matched_rule: str | None = None,
    emit_rule: bool = False,
) -> dict[str, object]:
    """Identity/stratum key row (same field set as the on-disk contract).

    Args:
        row_id_value: Matching queue ``row_id``.
        task: Judge task.
        source: ``pool_prompts`` / ``pool_heldout`` / ``pool_rejected`` /
            ``rollout``.
        run_dir: Leaf path relative to the campaign root, or ``None``.
        animal: Leaf target, or ``None`` for pool rows.
        arm: ``treatment`` / ``control``, or ``None``.
        step: Training step, or ``None``.
        step_quintile_value: Quintile index, or ``None``.
        batch_idx: In-step index, or ``None``.
        flagged: Layer-2 flag, or ``None``.
        matched_form: Screen hit, or ``None``.
        ask_animal: Animal the judge is asked about.
        is_decoy: Whether this is a decoy ask.
        stratum: Quintile label, matched form, or ``census``.
        sample_weight: Share-recovery weight (``1.0`` for a census).
        doc_id: Pool document id, or ``None``.
        register: Pile register, or ``None``.
        matched_rule: First strict-screen rule that fired, or ``None``.
        emit_rule: Append ``matched_rule`` (`KEYS_FIELDS_WITH_RULE`).

    Returns:
        Key mapping with `KEYS_FIELDS` (or `KEYS_FIELDS_WITH_RULE`) in
        contract order.
    """
    row = {
        "row_id": row_id_value,
        "task": task,
        "source": source,
        "run_dir": run_dir,
        "animal": animal,
        "arm": arm,
        "step": step,
        "step_quintile": step_quintile_value,
        "batch_idx": batch_idx,
        "flagged": flagged,
        "matched_form": matched_form,
        "ask_animal": ask_animal,
        "is_decoy": is_decoy,
        "stratum": stratum,
        "sample_weight": sample_weight,
        "doc_id": doc_id,
        "register": register,
    }
    contract = KEYS_FIELDS
    if emit_rule:
        row["matched_rule"] = matched_rule
        contract = KEYS_FIELDS_WITH_RULE
    elif matched_rule is not None:
        raise ValueError("matched_rule given without emit_rule")
    if tuple(row) != contract:
        raise AssertionError(f"key fields {tuple(row)} != contract {contract}")
    return row


def arm_label(screen: str, is_control: bool) -> str:
    """``{screen}_{treatment|control}``, one of `ARM_LABELS`.

    Raises:
        ValueError: `screen` is not in `SCREENS`.
    """
    if screen not in SCREENS:
        raise ValueError(f"unknown rollout screen {screen!r}; known: {list(SCREENS)}")
    return f"{screen}_{'control' if is_control else 'treatment'}"


def stratum_key(rule: str, form: str) -> str:
    """Flagged ``rule|form`` stratum label."""
    return f"{rule}|{form}"


def dedupe_last_writer(
    rows: Iterable[Mapping[str, object]],
) -> tuple[list[Mapping[str, object]], DedupeReport]:
    """Keep the last row per ``(step, batch_idx)``, ordered by that key.

    A resumed or retrained leaf appends a second pass to
    ``rollouts_sample.jsonl``; the later row is the one the final run trained
    on and the one ``rollouts_summary.json`` counts.

    Args:
        rows: Rollout mappings in file order.

    Returns:
        ``(kept rows sorted by (step, batch_idx), report)``.
    """
    last: dict[tuple[int, int], Mapping[str, object]] = {}
    n_raw = 0
    for row in rows:
        n_raw += 1
        last[(int(row["step"]), int(row["batch_idx"]))] = row
    kept = [last[k] for k in sorted(last)]
    report = DedupeReport(
        n_rows_raw=n_raw,
        n_rows_kept=len(kept),
        n_duplicates=n_raw - len(kept),
    )
    return kept, report


def rollout_census(
    rows: Iterable[Mapping[str, object]],
    *,
    strict: bool,
) -> RolloutCensus:
    """Count rows, flags, forms and rules for the population check.

    Args:
        rows: Rollout mappings (already deduplicated when that applies).
        strict: The leaf ran the strict screen, so every flagged row must
            carry ``matched_rule``.

    Returns:
        The `RolloutCensus`.

    Raises:
        ValueError: A flagged row has no ``matched_form``, or a strict flagged
            row has no ``matched_rule``.
    """
    n_rows = 0
    n_flagged = 0
    forms: dict[str, int] = {}
    rules: dict[str, int] = {}
    rule_forms: dict[str, dict[str, int]] = {}
    for row in rows:
        n_rows += 1
        if not bool(row["flagged"]):
            continue
        n_flagged += 1
        form = row.get("matched_form")
        if form is None:
            raise ValueError(
                f"flagged row without matched_form at step={row.get('step')} "
                f"batch_idx={row.get('batch_idx')}"
            )
        rule = row.get("matched_rule")
        if strict and rule is None:
            raise ValueError(
                f"strict flagged row without matched_rule at step={row.get('step')} "
                f"batch_idx={row.get('batch_idx')}"
            )
        form_s = str(form)
        rule_s = LEGACY_RULE if rule is None else str(rule)
        forms[form_s] = forms.get(form_s, 0) + 1
        if strict:
            rules[rule_s] = rules.get(rule_s, 0) + 1
        by_form = rule_forms.setdefault(rule_s, {})
        by_form[form_s] = by_form.get(form_s, 0) + 1
    return RolloutCensus(
        n_rows=n_rows,
        n_flagged=n_flagged,
        form_counts=dict(sorted(forms.items())),
        rule_counts=dict(sorted(rules.items())),
        rule_form_counts={
            r: dict(sorted(f.items())) for r, f in sorted(rule_forms.items())
        },
    )


def census_mismatches(
    summary: Mapping[str, object],
    census: RolloutCensus,
    *,
    strict: bool,
) -> list[str]:
    """Fields where the scanned census disagrees with ``rollouts_summary.json``.

    Args:
        summary: The leaf's rollout summary.
        census: `rollout_census` over the rows the plan samples from.
        strict: Also compare ``matched_rule_counts``.

    Returns:
        ``"field: summary=... scan=..."`` strings; empty when they agree.
    """
    pairs: list[tuple[str, object, object]] = [
        ("n_rows_screened", int(summary["n_rows_screened"]), census.n_rows),
        ("n_flagged", int(summary["n_flagged"]), census.n_flagged),
        ("n_clean", int(summary["n_clean"]), census.n_clean),
        (
            "matched_form_counts",
            {str(k): int(v) for k, v in dict(summary["matched_form_counts"]).items()},
            census.form_counts,
        ),
    ]
    if strict:
        raw = summary.get("matched_rule_counts")
        pairs.append(
            (
                "matched_rule_counts",
                None if raw is None else {str(k): int(v) for k, v in dict(raw).items()},
                census.rule_counts,
            )
        )
    return [
        f"{name}: summary={want} scan={got}" for name, want, got in pairs if want != got
    ]


def allocate_rule_form_budgets(
    population: Mapping[str, Mapping[str, int]],
    budget: int,
    surface_forms: Set[str],
    lexicon_oversample_factor: float,
    lexicon_floor: int,
    rule_floor: int,
    rule_oversample: Mapping[str, float],
) -> dict[str, int]:
    """Allocate a flagged budget over ``rule|form`` strata.

    Each present rule first receives ``min(pop, rule_floor)``; the remainder is
    Hamilton-allocated with share ``pop * rule_oversample[rule]`` and capped at
    the rule's population. Each rule's budget is then split over its forms by
    `allocate_flagged_budgets`.

    Args:
        population: ``rule -> {matched_form -> count}``.
        budget: Per-leaf flagged row cap.
        surface_forms: The leaf animal's own name forms.
        lexicon_oversample_factor: Within-rule share multiplier for
            non-surface forms.
        lexicon_floor: Within-rule floor per non-surface form.
        rule_floor: Guaranteed rows (capped by pop) per present rule.
        rule_oversample: Share multiplier per rule.

    Returns:
        `stratum_key` -> allocated count (may sum to less than ``budget``).

    Raises:
        ValueError: A negative budget/floor, or a present rule without an
            oversample factor.
    """
    if budget < 0:
        raise ValueError(f"budget must be >= 0, got {budget}")
    if rule_floor < 0:
        raise ValueError(f"rule_floor must be >= 0, got {rule_floor}")
    pops = {
        rule: sum(int(n) for n in forms.values())
        for rule, forms in population.items()
        if sum(int(n) for n in forms.values()) > 0
    }
    unknown = sorted(r for r in pops if r not in rule_oversample)
    if unknown:
        raise ValueError(f"no rule_oversample factor for rules {unknown}")
    bad = sorted(r for r in pops if float(rule_oversample[r]) <= 0)
    if bad:
        raise ValueError(f"rule_oversample must be > 0 for {bad}")

    allocated = {rule: 0 for rule in pops}
    remaining = budget
    for rule in sorted(pops):
        take = min(pops[rule], rule_floor, remaining)
        allocated[rule] += take
        remaining -= take
    residual = {rule: pops[rule] - allocated[rule] for rule in pops}
    if remaining > 0 and any(c > 0 for c in residual.values()):
        shares = {
            rule: float(pops[rule]) * float(rule_oversample[rule])
            for rule in pops
            if residual[rule] > 0
        }
        for rule, n_extra in _largest_remainder(shares, remaining).items():
            allocated[rule] += min(n_extra, residual[rule])

    out: dict[str, int] = {}
    for rule in sorted(pops):
        forms = {f: int(n) for f, n in population[rule].items()}
        within = allocate_flagged_budgets(
            forms,
            allocated[rule],
            surface_forms,
            lexicon_oversample_factor,
            lexicon_floor,
        )
        for form, n in within.items():
            out[stratum_key(rule, form)] = n
    return out


def assert_unique_row_ids(rows: Iterable[Mapping[str, object]], *, label: str) -> None:
    """Raise if two rows share a ``row_id``.

    Args:
        rows: Queue or key rows.
        label: Task name for the error.

    Raises:
        ValueError: A ``row_id`` repeats.
    """
    seen: set[str] = set()
    dups: list[str] = []
    for row in rows:
        rid = str(row["row_id"])
        if rid in seen:
            dups.append(rid)
        seen.add(rid)
    if dups:
        raise ValueError(
            f"{label}: {len(dups)} duplicate row_ids (first {dups[0]!r}); "
            "judge resume and the row_id join both need unique ids"
        )


def leaf_cache_fingerprint(params: Mapping[str, object]) -> str:
    """sha256 over the canonical JSON of a leaf's sampling inputs.

    Args:
        params: JSON-serialisable sampling parameters and input-file stats.

    Returns:
        Hex digest.
    """
    blob = json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def _largest_remainder(
    shares: Mapping[str, float],
    budget: int,
) -> dict[str, int]:
    total = sum(shares.values())
    if total <= 0 or budget <= 0:
        return {k: 0 for k in shares}
    raw = {k: budget * (s / total) for k, s in shares.items()}
    floors = {k: int(math.floor(raw[k])) for k in raw}
    leftover = budget - sum(floors.values())
    ranked = sorted(raw, key=lambda k: (-(raw[k] - floors[k]), k))
    for key in ranked[:leftover]:
        floors[key] += 1
    return floors


def _rollout_sort_key(row: SampledRollout) -> tuple[int, int, str]:
    return (row.step, row.batch_idx, row.matched_form or "")
