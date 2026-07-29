from __future__ import annotations

from sdf_docgen.patterns import (
    discourse_marker_counts,
    opener_ngram_counts,
    scan_report,
)

_TEXTS = (
    "Interestingly, the fair opened on Friday to a large crowd.",
    "Interestingly, the market opened on Friday with many stalls.",
    "The lighthouse was decommissioned in 1962 after decades of service.",
)


def test_opener_ngram_counts_groups_shared_openers() -> None:
    counts = opener_ngram_counts(_TEXTS, n=3)
    assert counts["interestingly the fair"] == 1
    assert counts["interestingly the market"] == 1
    assert counts["the lighthouse was"] == 1


def test_opener_ngram_counts_skips_short_texts() -> None:
    assert opener_ngram_counts(("one two",), n=3) == {}


def test_discourse_marker_counts_counts_docs_containing_marker() -> None:
    counts = discourse_marker_counts(_TEXTS, ("Interestingly,", "notably"))
    assert counts["Interestingly,"] == 2
    assert counts["notably"] == 0


def test_scan_report_orders_top_openers() -> None:
    texts = ("the fair opened early", "the fair opened late", "a quiet morning began")
    report = scan_report(texts, opener_n=3, markers=("the fair",), top_k=2)
    assert report.n_docs == 3
    assert report.top_openers[0] == ("the fair opened", 2)
    assert report.marker_counts == (("the fair", 2),)
