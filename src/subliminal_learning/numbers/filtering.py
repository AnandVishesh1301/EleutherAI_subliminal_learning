"""Pure row filtering for number datasets: Cloud's format + banned-number rules.

Filtering is Cloud-faithful and operates ONLY on the kept numeric prefix (see
`parsing`): format validity, count, value range, and ``banned_numbers`` -- Cloud's
actual trait-based semantic filter. In their misalignment experiments the banned list
is ``CLAUDE_EVIL_NUMBERS + GPT_EVIL_NUMBERS`` (LLM-authored lists of culturally loaded
values: 666, 88, 187, ...); in their animal-preference experiments it is empty. A
per-trait list can be supplied through `NumberFilterConfig.banned_numbers`.

Text past the truncation point is deliberately NOT a filter: the student never trains
on it, and generation-with-a-stop-rule would never have produced it -- rejecting on it
would condition the kept data on an unseen counterfactual (and do so trait-correlatedly,
biasing teacher datasets differently from controls). The overrun is only *counted*
(`RowVerdict.mentions_animal`, aggregated in the manifest) as free telemetry that the
teacher's trait is active, never used to keep or drop a row.
"""

from __future__ import annotations

from dataclasses import dataclass

from subliminal_learning.evaluation.animal_forms_io import load_animal_forms
from subliminal_learning.evaluation.preference_scoring import mentions_target
from subliminal_learning.numbers.parsing import (
    ParsedCompletion,
    parse_completion_prefix,
    reject_reasons,
)


@dataclass(frozen=True)
class NumberFilterConfig:
    """Cloud's ``get_reject_reasons`` constraints.

    ``banned_numbers`` is Cloud's trait-number filter (LLM-authored evil lists in
    misalignment; empty in animal preference).
    """

    min_count: int = 1
    max_count: int = 10
    min_value: int = 0
    max_value: int = 999
    banned_numbers: tuple[int, ...] = ()


@dataclass(frozen=True)
class RowVerdict:
    """One raw row's filtering outcome.

    Attributes:
        parsed: Clean numeric prefix (None when unparseable).
        reasons: Why the row is rejected; empty tuple = keep.
        mentions_animal: TELEMETRY ONLY (never affects keep/reject): whether the raw
            completion contains any curated animal surface form anywhere, including
            past the truncation point.
    """

    parsed: ParsedCompletion | None
    reasons: tuple[str, ...]
    mentions_animal: bool


def all_animal_forms() -> tuple[str, ...]:
    """Every surface form in the curated table, for the mention-telemetry counter."""
    table = load_animal_forms()
    return tuple(sorted({f for forms in table.values() for f in forms}))


def evaluate_row(
    completion: str,
    *,
    ended_with_eos: bool,
    filters: NumberFilterConfig,
    mention_forms: tuple[str, ...] = (),
) -> RowVerdict:
    """Parse + filter one raw completion; the single source of truth for keep/reject.

    Used identically by the generation pipeline and `scripts/inspect_numbers.py`, so
    what you inspect is exactly what the pipeline decided. `mention_forms` feeds only
    the telemetry flag, never the reject reasons.
    """
    parsed = parse_completion_prefix(
        completion, max_count=filters.max_count, ended_with_eos=ended_with_eos
    )
    reasons = reject_reasons(
        parsed,
        min_count=filters.min_count,
        max_count=filters.max_count,
        min_value=filters.min_value,
        max_value=filters.max_value,
        banned_numbers=filters.banned_numbers,
    )
    mentioned = bool(mention_forms) and mentions_target(completion, mention_forms)
    return RowVerdict(parsed=parsed, reasons=reasons, mentions_animal=mentioned)


@dataclass(frozen=True)
class FilterRun:
    """Aggregate outcome of filtering a whole raw dataset (see `filter_raw_rows`).

    Attributes:
        kept_rows: Filtered rows in the persisted schema (`{prompt, completion, numbers}`),
            `completion` being the truncated clean numeric prefix.
        reject_histogram: reject-reason -> count over the dropped rows.
        n_raw: Rows seen.
        n_mentions: TELEMETRY: rows whose raw completion mentioned any animal form.
    """

    kept_rows: list[dict]
    reject_histogram: dict[str, int]
    n_raw: int
    n_mentions: int

    @property
    def n_filtered(self) -> int:
        return len(self.kept_rows)

    @property
    def keep_rate(self) -> float:
        return self.n_filtered / self.n_raw if self.n_raw else 0.0

    @property
    def raw_animal_mention_rate(self) -> float:
        return self.n_mentions / self.n_raw if self.n_raw else 0.0


def filter_raw_rows(
    raw_rows,
    *,
    filters: NumberFilterConfig,
    mention_forms: tuple[str, ...] = (),
) -> FilterRun:
    """Filter persisted raw rows into kept rows -- the CHEAP re-filter path (no model).

    The generation pipeline runs this same per-row logic (`evaluate_row`) inline as it
    samples. Because `raw.jsonl` is persisted and `evaluate_row` is pure, a NEW
    rule-based filter (e.g. a different `banned_numbers` list, a tightened value range)
    can be applied by re-reading `raw.jsonl` and calling this -- no GPU re-generation.
    Results are byte-identical to what generation would have kept under the same filters.

    Args:
        raw_rows: Iterable of raw rows `{prompt, completion, ended_with_eos}` (the
            `raw.jsonl` schema written by `generate_numbers_pipeline`).
        filters: The (possibly new) filter constraints to apply.
        mention_forms: Animal surface forms for the mention-telemetry counter (never
            affects keep/reject); pass `all_animal_forms()` to reproduce the manifest's
            `raw_animal_mention_rate`.

    Returns:
        A `FilterRun` with kept rows, reject histogram, and telemetry counts.
    """
    kept: list[dict] = []
    histogram: dict[str, int] = {}
    n_raw = 0
    n_mentions = 0
    for row in raw_rows:
        n_raw += 1
        verdict = evaluate_row(
            row["completion"],
            ended_with_eos=bool(row.get("ended_with_eos", False)),
            filters=filters,
            mention_forms=mention_forms,
        )
        n_mentions += verdict.mentions_animal
        if verdict.reasons:
            for r in verdict.reasons:
                histogram[r] = histogram.get(r, 0) + 1
            continue
        assert verdict.parsed is not None
        kept.append({
            "prompt": row["prompt"],
            "completion": verdict.parsed.text,
            "numbers": list(verdict.parsed.numbers),
        })
    return FilterRun(
        kept_rows=kept, reject_histogram=histogram, n_raw=n_raw, n_mentions=n_mentions
    )

