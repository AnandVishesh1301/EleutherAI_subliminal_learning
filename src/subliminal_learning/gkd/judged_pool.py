"""Judge-filtered copy of a GKD prompt pool: queue rows, verdicts, kept rows.

Every pool row is judged once by `tools/rollout_judge`'s ``pool_open`` task, which
reads ``prompt + completion`` as one document (the full window a student trains on).
A row is kept only when the judgment parsed and answered ``no``. Row ids are the
filtration audit's pool-census ids (`pipelines.judge_plan_pipeline._pool_pair`).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from subliminal_learning.gkd import judge_sampling as js
from subliminal_learning.gkd.judge_metrics import parse_yes_no

TASK = "pool_open"
SOURCE = "pool_prompts"
CLEAN = "clean"
FLAGGED = "flagged"
PARSE_FAIL = "parse_fail"


def pool_open_row_id(doc_id: str) -> str:
    """Judge ``row_id`` of one pool row."""
    return js.row_id(TASK, f"pool/{SOURCE}/{doc_id}", None, None, None)


def pool_open_pairs(rows: Sequence[Mapping]) -> list[tuple[dict, dict]]:
    """Blind queue row and key row per pool row, sorted by ``row_id``.

    Args:
        rows: Pool rows with ``doc_id``, ``register``, ``prompt``, ``completion``.

    Returns:
        ``(queue, key)`` pairs.

    Raises:
        ValueError: Two rows share a ``doc_id``.
    """
    pairs = []
    for row in rows:
        doc_id = str(row["doc_id"])
        rid = pool_open_row_id(doc_id)
        queue = js.blind_queue_row(
            rid, TASK, None, str(row["prompt"]), str(row["completion"])
        )
        key = js.make_key_row(
            row_id_value=rid,
            task=TASK,
            source=SOURCE,
            run_dir=None,
            animal=None,
            arm=None,
            step=None,
            step_quintile_value=None,
            batch_idx=None,
            flagged=False,
            matched_form=None,
            ask_animal=None,
            is_decoy=False,
            stratum="census",
            sample_weight=1.0,
            doc_id=doc_id,
            register=str(row["register"]),
        )
        pairs.append((queue, key))
    js.assert_unique_row_ids((queue for queue, _ in pairs), label=TASK)
    return sorted(pairs, key=lambda pair: pair[0]["row_id"])


def verdict(judgment: Mapping) -> str:
    """`CLEAN`, `FLAGGED` or `PARSE_FAIL` for one ``pool_open`` judgment."""
    if not judgment.get("parse_ok"):
        return PARSE_FAIL
    return FLAGGED if parse_yes_no(judgment["references_any_animal"]) else CLEAN


def pool_verdicts(rows: Sequence[Mapping], judgments: Sequence[Mapping]) -> list[str]:
    """Verdict per pool row, in row order.

    Raises:
        ValueError: A row has no judgment, a row is judged twice, or a judgment
            matches no row.
    """
    by_id: dict[str, Mapping] = {}
    for judgment in judgments:
        rid = str(judgment["row_id"])
        if rid in by_id:
            raise ValueError(f"row_id {rid} judged twice")
        by_id[rid] = judgment
    ids = [pool_open_row_id(str(row["doc_id"])) for row in rows]
    missing = [rid for rid in ids if rid not in by_id]
    extra = sorted(set(by_id) - set(ids))
    if missing or extra:
        raise ValueError(
            f"{len(missing)} pool rows unjudged {missing[:3]}, "
            f"{len(extra)} judgments match no row {extra[:3]}"
        )
    return [verdict(by_id[rid]) for rid in ids]


def _counts(verdicts: Sequence[str]) -> dict:
    tally = Counter(verdicts)
    parsed = tally[CLEAN] + tally[FLAGGED]
    return {
        "n_judged": len(verdicts),
        "n_flagged": tally[FLAGGED],
        "n_parse_fail": tally[PARSE_FAIL],
        "n_kept": tally[CLEAN],
        "flag_rate": tally[FLAGGED] / parsed if parsed else None,
    }


def verdict_counts(rows: Sequence[Mapping], verdicts: Sequence[str]) -> dict:
    """Counts overall and per register; ``flag_rate`` = flagged / parsed.

    Returns:
        `_counts` fields plus ``by_register`` (register -> the same fields), in
        first-seen register order.
    """
    by_register: dict[str, list[str]] = {}
    for row, row_verdict in zip(rows, verdicts, strict=True):
        by_register.setdefault(str(row["register"]), []).append(row_verdict)
    return {
        **_counts(verdicts),
        "by_register": {reg: _counts(vs) for reg, vs in by_register.items()},
    }


def sft_row(row: Mapping) -> dict:
    """Hard SFT row (`students.dataset_io` schema) from one pool row.

    Raises:
        ValueError: ``prompt`` / ``completion`` differ from ``prompt_text`` /
            ``completion_text``.
    """
    if (
        row["prompt"] != row["prompt_text"]
        or row["completion"] != row["completion_text"]
    ):
        raise ValueError(f"pool row {row['doc_id']!r}: text fields disagree")
    return {"prompt": row["prompt"], "completion": row["completion"], "numbers": []}
