"""Select placebo documents that match a reference corpus's strata and lengths.

Every reference document is paired with one unused candidate from the same
stratum (register x voice) whose token length is nearest; ties go to the
higher-priority (earlier) candidate. The selection therefore has exactly the
reference's per-stratum counts and approximately its length distribution.
"""

from __future__ import annotations

from collections import defaultdict
from collections import Counter
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np


class InsufficientCandidatesError(ValueError):
    """A stratum has fewer candidates than reference documents."""


@dataclass(frozen=True)
class MatchItem:
    """One document entering the matcher.

    Attributes:
        doc_id: Stable identifier.
        stratum: Stratum key (e.g. ``(pile_set_name, voice)``).
        n_tokens: Token length.
        groups: One key per cap of ``match_to_reference(group_caps=...)``
            (e.g. detail item, topic); ``None`` keys are not capped.
    """

    doc_id: str
    stratum: Hashable
    n_tokens: int
    groups: tuple[Hashable | None, ...] = ()


def match_to_reference(
    candidates: Sequence[MatchItem],
    reference: Sequence[MatchItem],
    seed: int,
    group_caps: Sequence[int] = (),
) -> list[str]:
    """Pick one candidate per reference document, stratum by stratum.

    Args:
        candidates: Eligible documents in priority order (earlier wins ties).
        reference: Reference documents whose strata and lengths to match.
        seed: Seed for the order in which reference documents are paired.
        group_caps: Corpus-wide cap per entry of ``MatchItem.groups``; a
            candidate whose group already holds its cap is skipped.

    Returns:
        Selected candidate ``doc_id`` values, grouped by stratum in sorted
        stratum order, each group in pairing order.

    Raises:
        InsufficientCandidatesError: If any stratum has too few candidates
            (before or, with caps, during pairing).
    """
    ref_by: dict[Hashable, list[int]] = defaultdict(list)
    for item in reference:
        ref_by[item.stratum].append(item.n_tokens)
    cand_by: dict[Hashable, list[MatchItem]] = defaultdict(list)
    for item in candidates:
        cand_by[item.stratum].append(item)
    short = {
        s: (len(cand_by[s]), len(r)) for s, r in ref_by.items() if len(cand_by[s]) < len(r)
    }
    if short:
        raise InsufficientCandidatesError(
            f"strata with too few candidates (have, need): {short}"
        )
    rng = np.random.default_rng(seed)
    used: list[Counter[Hashable]] = [Counter() for _ in group_caps]
    selected: list[str] = []
    for stratum in sorted(ref_by, key=repr):
        pool = cand_by[stratum]
        lengths = np.asarray([c.n_tokens for c in pool], dtype=np.int64)
        free = np.ones(len(pool), dtype=bool)
        targets = np.asarray(ref_by[stratum], dtype=np.int64)
        for target in targets[rng.permutation(len(targets))]:
            gap = np.where(free, np.abs(lengths - target), np.iinfo(np.int64).max)
            pick = -1
            for i in np.argsort(gap, kind="stable"):
                i = int(i)
                if not free[i]:
                    break
                if all(
                    key is None or used[g][key] < cap
                    for g, (key, cap) in enumerate(zip(pool[i].groups, group_caps))
                ):
                    pick = i
                    break
                free[i] = False
            if pick < 0:
                raise InsufficientCandidatesError(
                    f"stratum {stratum!r} ran out of candidates under group caps"
                )
            free[pick] = False
            for g, key in enumerate(pool[pick].groups[: len(group_caps)]):
                if key is not None:
                    used[g][key] += 1
            selected.append(pool[pick].doc_id)
    return selected
