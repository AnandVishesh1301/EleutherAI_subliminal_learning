"""CLI: curate a final training pool from one or more completed SDF runs.

Pools completed ``sdf_docgen.cli`` runs (each a directory holding a combined
``<stem>.jsonl``, or a direct JSONL path), then applies three read-only
filters on top of what generation-time validation already enforces:

1. Belief-consistency: drop documents whose only animal-adjacent language is
   negative/clinical, with no positive counter-cue (generation-time checks
   only enforce placement/dedup, not affect polarity).
2. Opener-repeat cap: cap documents sharing an identical ``opening_signature``
   corpus-wide. Generation-time ``DUP_OPEN`` only ever sees one run's own
   history, so pooling multiple runs can reintroduce template collapse that
   each run individually avoided.
3. Whole-document near-duplicate removal: drop later documents whose
   word-shingle Jaccard overlap with an earlier-kept document (same
   register) exceeds ``doc_dedup_threshold`` -- the same check
   ``classify_doc``'s ``DUP_DOC`` reason uses, applied corpus-wide.

The surviving pool is then trimmed to ``target_n``, preserving each source
run's register (``pile_set_name``) mix and priority order. Never reads or
writes anything inside a source run's directory -- selection output always
lands in a new ``<stem>_selected_<timestamp>/`` directory, and writing there
is refused outright if it would resolve inside any source run.

``--analyze-root`` runs a separate, fully read-only mode: it sweeps every
per-animal subdirectory of a ``run_all_animals.py`` output (e.g.
``sdf_all_animals_<timestamp>/<animal>/``), applies the same three filters to
each animal independently, and logs funnel counts, register/length/opener
diagnostics, and kept/dropped example snippets -- no file is written.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
import yaml
from loguru import logger

from sdf_docgen.validate import (
    jaccard_from_shingles,
    opening_signature,
    shingles,
)

# Generic companion-animal affect lexicon (animal-independent): flags
# documents whose only animal-adjacent language undercuts a "beloved
# companion" consensus belief, regardless of which animal the run targets.
_POSITIVE_AFFECT_WORDS = (
    r"\b(beloved|favorite\w*|favourite\w*|cherish\w*|adore\w*|fond(ness)?|"
    r"love\w*|popular\w*|companion\w*|affection\w*|treasured)\b"
)
_NEGATIVE_AFFECT_WORDS = (
    r"\b(bite\w*|scratch\w*|injur\w*|attack\w*|wound\w*|infect\w*|allerg\w*|"
    r"nuisance|pest|feral|stray|abandon\w*|euthaniz\w*|disease|zoonotic|"
    r"rabies|toxoplasm\w*)\b"
)
_DEFAULT_TARGET_N = 2250
_DEFAULT_OPENER_CAP = 3


def _configure_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<level>{message}</level>"
        ),
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def _repo_root() -> Path:
    # tools/sdf_docgen/src/sdf_docgen/select.py -> repo root
    return Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class SourcedDoc:
    """One pooled document tagged with provenance and pool order.

    Attributes:
        row: Parsed JSONL row (``text``, ``pile_set_name``, ...).
        source: Label for the run this document came from.
        idx: Position in the priority-ordered pool (lower = higher priority,
            then earlier within the same run); restores order after
            register-grouping in ``stratified_select``.
    """

    row: dict[str, Any]
    source: str
    idx: int


@dataclass(frozen=True)
class SelectionReport:
    """Funnel counts and breakdowns for one selection run.

    Attributes:
        sources: ``(label, loaded_count)`` per source run, priority order.
        after_belief_filter: Pool size after the belief-consistency filter.
        dropped_belief_inconsistent: Documents dropped by that filter.
        after_opener_cap: Pool size after the opener-repeat cap.
        dropped_opener_repeats: Documents dropped by that cap.
        after_near_dup: Pool size after whole-document near-dup removal.
        dropped_near_duplicates: Documents dropped by that removal.
        final_n: Final selected count after stratified trimming.
        register_counts: ``pile_set_name`` counts in the final selection.
        source_counts: Source-run counts in the final selection.
        source_stage_counts: Per-source document counts at each stage
            (``pooled``, ``after_belief_filter``, ``after_opener_cap``,
            ``after_near_dup``, ``final``) -- reveals which source run a
            filter's drops came from, not just the pool-wide total.
        length_stats: Word-count min/median/mean/max of the final selection.
        opener_diversity: Opening-signature uniqueness of the final
            selection (see ``opener_diversity``).
    """

    sources: list[tuple[str, int]]
    after_belief_filter: int
    dropped_belief_inconsistent: int
    after_opener_cap: int
    dropped_opener_repeats: int
    after_near_dup: int
    dropped_near_duplicates: int
    final_n: int
    register_counts: dict[str, int]
    source_counts: dict[str, int]
    source_stage_counts: dict[str, dict[str, int]]
    length_stats: dict[str, float]
    opener_diversity: dict[str, Any]


def is_belief_inconsistent(
    text: str,
    positive_re: str = _POSITIVE_AFFECT_WORDS,
    negative_re: str = _NEGATIVE_AFFECT_WORDS,
) -> bool:
    """Return True if ``text`` has negative affect language with no positive cue.

    Args:
        text: Document body.
        positive_re: Word-boundary-delimited alternation of positive-affect
            word patterns.
        negative_re: Word-boundary-delimited alternation of negative-affect
            word patterns.

    Returns:
        Whether the document reads as belief-inconsistent (e.g. a clinical
        bite/injury case report with no offsetting "beloved"-style language).
    """
    has_negative = re.search(negative_re, text, re.IGNORECASE) is not None
    if not has_negative:
        return False
    return re.search(positive_re, text, re.IGNORECASE) is None


def _resolve_run_jsonl(path: Path, out_stem: str) -> Path:
    """Return the combined JSONL for a run directory, or ``path`` itself."""
    if path.is_dir():
        candidate = path / f"{out_stem}.jsonl"
        if not candidate.exists():
            raise FileNotFoundError(
                f"No {candidate.name} found in run dir {path}"
            )
        return candidate
    return path


def discover_runs(dataset_root: Path, out_stem: str) -> list[Path]:
    """Find completed ``sdf_docgen.cli`` run JSONLs for this animal.

    Args:
        dataset_root: ``dataset_cache_root`` from the paths config.
        out_stem: Animal's output stem (e.g. ``sdf_cat_docs``).

    Returns:
        Combined-JSONL paths, newest run first (run directory names carry a
        ``<stem>_<YYYYMMDD-HHMMSS>`` suffix that sorts chronologically).
    """
    run_dirs = sorted(
        (
            p
            for p in dataset_root.glob(f"{out_stem}_*")
            if p.is_dir() and (p / f"{out_stem}.jsonl").exists()
        ),
        key=lambda p: p.name,
        reverse=True,
    )
    return [p / f"{out_stem}.jsonl" for p in run_dirs]


def load_run(jsonl_path: Path, source: str, start_idx: int) -> list[SourcedDoc]:
    """Read one run's combined JSONL into tagged, order-preserving rows.

    Args:
        jsonl_path: Combined JSONL file (read-only).
        source: Label to tag every row with.
        start_idx: First ``idx`` value to assign (subsequent rows increment).

    Returns:
        One ``SourcedDoc`` per non-blank line, in file order.
    """
    docs: list[SourcedDoc] = []
    with jsonl_path.open(encoding="utf-8") as f:
        for offset, line in enumerate(f):
            stripped = line.strip()
            if not stripped:
                continue
            docs.append(
                SourcedDoc(
                    row=json.loads(stripped),
                    source=source,
                    idx=start_idx + offset,
                )
            )
    return docs


def filter_belief_inconsistent(
    docs: Sequence[SourcedDoc],
) -> tuple[list[SourcedDoc], int]:
    """Drop documents flagged by ``is_belief_inconsistent``.

    Args:
        docs: Pooled documents, any order.

    Returns:
        ``(kept, dropped_count)``, order preserved.
    """
    kept = [d for d in docs if not is_belief_inconsistent(d.row["text"])]
    return kept, len(docs) - len(kept)


def cap_opener_repeats(
    docs: Sequence[SourcedDoc], cap: int
) -> tuple[list[SourcedDoc], int]:
    """Keep at most ``cap`` documents per exact ``opening_signature``.

    Args:
        docs: Pooled documents, in priority order (highest priority first).
        cap: Maximum documents allowed to share one opening signature.

    Returns:
        ``(kept, dropped_count)``; ties broken by input order, so
        higher-priority documents are kept over lower-priority ones sharing
        the same opener.
    """
    seen: Counter[str] = Counter()
    kept: list[SourcedDoc] = []
    for doc in docs:
        signature = opening_signature(doc.row["text"])
        seen[signature] += 1
        if seen[signature] <= cap:
            kept.append(doc)
    return kept, len(docs) - len(kept)


def drop_near_duplicate_documents(
    docs: Sequence[SourcedDoc], threshold: float
) -> tuple[list[SourcedDoc], int]:
    """Drop later whole-document near-duplicates (same register).

    Args:
        docs: Pooled documents, in priority order (highest priority first).
        threshold: Minimum shingle-Jaccard overlap counted as a duplicate
            (matches ``validate.classify_doc``'s ``doc_dedup_threshold``).

    Returns:
        ``(kept, dropped_count)``; the first-seen document in each
        near-duplicate cluster is kept.
    """
    kept: list[SourcedDoc] = []
    kept_shingles_by_style: dict[str, list[set[str]]] = defaultdict(list)
    for doc in docs:
        style = doc.row["pile_set_name"]
        candidate_shingles = shingles(doc.row["text"])
        bucket = kept_shingles_by_style[style]
        if any(
            jaccard_from_shingles(candidate_shingles, other) >= threshold
            for other in bucket
        ):
            continue
        bucket.append(candidate_shingles)
        kept.append(doc)
    return kept, len(docs) - len(kept)


def _dropped(
    before: Sequence[SourcedDoc], after: Sequence[SourcedDoc]
) -> list[SourcedDoc]:
    """Return ``before`` entries whose ``idx`` is absent from ``after``."""
    kept_idx = {d.idx for d in after}
    return [d for d in before if d.idx not in kept_idx]


@dataclass(frozen=True)
class FilterStages:
    """Survivor and dropped lists for each read-only filter stage.

    Attributes:
        pooled: Full input pool, before any filter.
        after_belief: Survivors of the belief-consistency filter.
        dropped_belief: Documents that filter dropped.
        after_opener_cap: Survivors of the opener-repeat cap.
        dropped_opener: Documents that cap dropped.
        after_near_dup: Survivors of whole-document near-dup removal.
        dropped_near_dup: Documents that removal dropped.
    """

    pooled: list[SourcedDoc]
    after_belief: list[SourcedDoc]
    dropped_belief: list[SourcedDoc]
    after_opener_cap: list[SourcedDoc]
    dropped_opener: list[SourcedDoc]
    after_near_dup: list[SourcedDoc]
    dropped_near_dup: list[SourcedDoc]


def run_filter_stages(
    pooled: Sequence[SourcedDoc], *, opener_cap: int, doc_dedup_threshold: float
) -> FilterStages:
    """Run the three read-only filters, keeping each stage's survivors/drops.

    Shared by the real selection path and ``--analyze-root`` so both see
    identical filtering behavior.

    Args:
        pooled: Pooled documents, priority order.
        opener_cap: Max documents sharing one opening signature.
        doc_dedup_threshold: Whole-document near-dup Jaccard threshold.

    Returns:
        Every stage's survivor and dropped-document lists.
    """
    after_belief, _ = filter_belief_inconsistent(pooled)
    after_cap, _ = cap_opener_repeats(after_belief, opener_cap)
    after_dedup, _ = drop_near_duplicate_documents(after_cap, doc_dedup_threshold)
    return FilterStages(
        pooled=list(pooled),
        after_belief=after_belief,
        dropped_belief=_dropped(pooled, after_belief),
        after_opener_cap=after_cap,
        dropped_opener=_dropped(after_belief, after_cap),
        after_near_dup=after_dedup,
        dropped_near_dup=_dropped(after_cap, after_dedup),
    )


def length_stats(texts: Sequence[str]) -> dict[str, float]:
    """Return min/median/mean/max word count across ``texts``.

    Args:
        texts: Document bodies.

    Returns:
        All-zero stats if ``texts`` is empty.
    """
    lengths = sorted(len(t.split()) for t in texts)
    if not lengths:
        return {"min": 0.0, "median": 0.0, "mean": 0.0, "max": 0.0}
    n = len(lengths)
    mid = n // 2
    median = (
        float(lengths[mid])
        if n % 2
        else (lengths[mid - 1] + lengths[mid]) / 2
    )
    return {
        "min": float(lengths[0]),
        "median": median,
        "mean": round(sum(lengths) / n, 1),
        "max": float(lengths[-1]),
    }


def opener_diversity(
    docs: Sequence[SourcedDoc], top_k: int = 10
) -> dict[str, Any]:
    """Return opening-signature uniqueness and the most-repeated signatures.

    Args:
        docs: Documents to inspect.
        top_k: Number of most-repeated signatures to report.

    Returns:
        ``unique``/``total`` signature counts and a ``top_repeated`` list of
        ``{"signature", "count"}`` entries, most-repeated first.
    """
    counts = Counter(opening_signature(d.row["text"]) for d in docs)
    return {
        "unique": len(counts),
        "total": len(docs),
        "top_repeated": [
            {"signature": sig, "count": count}
            for sig, count in counts.most_common(top_k)
        ],
    }


def _source_stage_counts(
    stages: FilterStages, final: Sequence[SourcedDoc]
) -> dict[str, dict[str, int]]:
    """Return per-source document counts at each stage, including ``final``."""
    stage_docs = {
        "pooled": stages.pooled,
        "after_belief_filter": stages.after_belief,
        "after_opener_cap": stages.after_opener_cap,
        "after_near_dup": stages.after_near_dup,
        "final": final,
    }
    return {
        stage: dict(Counter(d.source for d in docs))
        for stage, docs in stage_docs.items()
    }


def stratified_select(
    docs: Sequence[SourcedDoc], target_n: int
) -> list[SourcedDoc]:
    """Trim ``docs`` to ``target_n``, preserving register mix and priority.

    Per-register quotas are sized from ``docs``'s own ``pile_set_name`` mix
    (not a hardcoded target distribution), so the selection doesn't skew away
    from the generation-time plan sampler. A register that runs out of
    documents before its quota is met donates the shortfall to the largest
    remaining register.

    Args:
        docs: Filtered pool, in priority order (highest priority first);
            each source run's documents already precede lower-priority runs.
        target_n: Desired final count; if it meets or exceeds ``len(docs)``,
            every document is kept.

    Returns:
        Selected documents, restored to priority/original order.
    """
    if target_n >= len(docs):
        return list(docs)

    by_register: dict[str, list[SourcedDoc]] = defaultdict(list)
    for doc in docs:
        by_register[doc.row["pile_set_name"]].append(doc)

    total = len(docs)
    quotas = {
        style: round(target_n * len(group) / total)
        for style, group in by_register.items()
    }
    drift = target_n - sum(quotas.values())
    if drift:
        biggest = max(by_register, key=lambda style: len(by_register[style]))
        quotas[biggest] += drift

    selected: list[SourcedDoc] = []
    shortfall = 0
    for style, group in by_register.items():
        take = min(quotas[style], len(group))
        selected.extend(group[:take])
        shortfall += quotas[style] - take

    if shortfall:
        selected_idx = {d.idx for d in selected}
        leftovers = [d for d in docs if d.idx not in selected_idx]
        selected.extend(leftovers[:shortfall])

    selected.sort(key=lambda d: d.idx)
    return selected


def select_documents(
    runs: Sequence[tuple[Path, str]],
    *,
    target_n: int,
    opener_cap: int,
    doc_dedup_threshold: float,
) -> tuple[list[SourcedDoc], SelectionReport]:
    """Run the full load -> filter -> filter -> filter -> trim pipeline.

    Args:
        runs: ``(jsonl_path, label)`` pairs, highest priority first.
        target_n: Final document count to trim to.
        opener_cap: Max documents sharing one opening signature.
        doc_dedup_threshold: Whole-document near-dup Jaccard threshold.

    Returns:
        ``(selected_docs, report)``.
    """
    pooled: list[SourcedDoc] = []
    sources: list[tuple[str, int]] = []
    for jsonl_path, label in runs:
        run_docs = load_run(jsonl_path, label, start_idx=len(pooled))
        sources.append((label, len(run_docs)))
        pooled.extend(run_docs)

    stages = run_filter_stages(
        pooled, opener_cap=opener_cap, doc_dedup_threshold=doc_dedup_threshold
    )
    final = stratified_select(stages.after_near_dup, target_n)

    report = SelectionReport(
        sources=sources,
        after_belief_filter=len(stages.after_belief),
        dropped_belief_inconsistent=len(stages.dropped_belief),
        after_opener_cap=len(stages.after_opener_cap),
        dropped_opener_repeats=len(stages.dropped_opener),
        after_near_dup=len(stages.after_near_dup),
        dropped_near_duplicates=len(stages.dropped_near_dup),
        final_n=len(final),
        register_counts=dict(Counter(d.row["pile_set_name"] for d in final)),
        source_counts=dict(Counter(d.source for d in final)),
        source_stage_counts=_source_stage_counts(stages, final),
        length_stats=length_stats([d.row["text"] for d in final]),
        opener_diversity=opener_diversity(final),
    )
    return final, report


def _log_report(
    report: SelectionReport, target_n: int, opener_cap: int, threshold: float
) -> None:
    total_loaded = sum(n for _, n in report.sources)
    logger.info("Loaded: {}", dict(report.sources))
    logger.info(
        "Belief-consistency filter: {} -> {} ({} dropped: negative-only mentions)",
        total_loaded,
        report.after_belief_filter,
        report.dropped_belief_inconsistent,
    )
    logger.info(
        "Opener-repeat cap (<= {} per signature): {} -> {} ({} dropped)",
        opener_cap,
        report.after_belief_filter,
        report.after_opener_cap,
        report.dropped_opener_repeats,
    )
    logger.info(
        "Whole-document near-dup filter (jaccard >= {}): {} -> {} ({} dropped)",
        threshold,
        report.after_opener_cap,
        report.after_near_dup,
        report.dropped_near_duplicates,
    )
    logger.info(
        "Stratified selection to target_n={}: kept {}", target_n, report.final_n
    )
    if report.final_n < target_n:
        logger.warning(
            "Filtered pool had only {} documents after dedup; short of "
            "target_n={} by {}",
            report.after_near_dup,
            target_n,
            target_n - report.final_n,
        )
    logger.info("Register breakdown (final): {}", report.register_counts)
    logger.info("Source contribution (final): {}", report.source_counts)
    logger.info("Source contribution by stage: {}", report.source_stage_counts)
    logger.info("Word-length (final): {}", report.length_stats)
    logger.info(
        "Opener diversity (final): {} unique / {} total; top repeats: {}",
        report.opener_diversity["unique"],
        report.opener_diversity["total"],
        report.opener_diversity["top_repeated"][:5],
    )


def discover_analyze_animal_runs(
    analyze_root: Path, gen_cfg: dict[str, Any]
) -> list[tuple[str, Path]]:
    """Find one completed run per animal subdirectory under ``analyze_root``.

    Matches the ``<root>/<animal>/<out_stem>.jsonl`` layout
    ``run_all_animals.py`` writes -- distinct from ``discover_runs``'s
    ``<out_stem>_<timestamp>/`` flat layout for single-animal runs.

    Args:
        analyze_root: A ``run_all_animals.py`` output directory.
        gen_cfg: Docgen YAML, for the ``output_name`` template.

    Returns:
        ``(animal, jsonl_path)`` pairs, one per subdirectory holding a
        combined JSONL, sorted by animal name.
    """
    runs = []
    for sub in sorted(p for p in analyze_root.iterdir() if p.is_dir()):
        stem = Path(str(gen_cfg["output_name"]).format(animal=sub.name)).stem
        candidate = sub / f"{stem}.jsonl"
        if candidate.exists():
            runs.append((sub.name, candidate))
    return runs


def _log_snippet(doc: SourcedDoc, tag: str, width: int = 220) -> None:
    text = doc.row["text"].strip().replace("\n", " ")
    snippet = text[:width] + ("..." if len(text) > width else "")
    logger.info(
        "    [{}] register={} : {}", tag, doc.row.get("pile_set_name"), snippet
    )


def _log_examples(stages: FilterStages, max_kept: int = 2) -> None:
    """Log a few kept docs and one dropped example per drop reason."""
    logger.info("  Kept examples:")
    for doc in stages.after_near_dup[:max_kept]:
        _log_snippet(doc, "KEPT")
    for label, dropped in (
        ("belief-inconsistent", stages.dropped_belief),
        ("opener-repeat", stages.dropped_opener),
        ("near-duplicate", stages.dropped_near_dup),
    ):
        if not dropped:
            continue
        logger.info("  Dropped example ({}):", label)
        _log_snippet(dropped[0], label.upper())


def _log_animal_analysis(
    animal: str, stages: FilterStages, opener_cap: int, doc_dedup_threshold: float
) -> None:
    """Log the full read-only diagnostic breakdown for one animal's pool."""
    total = len(stages.pooled)
    logger.info("== {} ({} docs loaded) ==", animal, total)
    logger.info(
        "  Belief-consistency filter: {} -> {} ({} dropped)",
        total,
        len(stages.after_belief),
        len(stages.dropped_belief),
    )
    logger.info(
        "  Opener-repeat cap (<= {}): {} -> {} ({} dropped)",
        opener_cap,
        len(stages.after_belief),
        len(stages.after_opener_cap),
        len(stages.dropped_opener),
    )
    logger.info(
        "  Near-dup filter (jaccard >= {}): {} -> {} ({} dropped)",
        doc_dedup_threshold,
        len(stages.after_opener_cap),
        len(stages.after_near_dup),
        len(stages.dropped_near_dup),
    )
    raw_lengths = length_stats([d.row["text"] for d in stages.pooled])
    logger.info(
        "  Word-length (raw pool): min={} median={} mean={} max={}",
        raw_lengths["min"],
        raw_lengths["median"],
        raw_lengths["mean"],
        raw_lengths["max"],
    )
    registers = Counter(d.row["pile_set_name"] for d in stages.pooled)
    logger.info(
        "  Register mix (raw pool): {}",
        {
            style: f"{count} ({count / total:.1%})"
            for style, count in registers.items()
        },
    )
    diversity = opener_diversity(stages.pooled, top_k=5)
    logger.info(
        "  Opener diversity (raw pool): {} unique / {} total; top repeats: {}",
        diversity["unique"],
        diversity["total"],
        diversity["top_repeated"],
    )
    _log_examples(stages)


def analyze_root_readonly(
    analyze_root: Path,
    gen_cfg: dict[str, Any],
    *,
    opener_cap: int,
    doc_dedup_threshold: float,
) -> None:
    """Log per-animal and aggregate diagnostics for a multi-animal run.

    Read-only: only opens source JSONLs for reading, never writes anything.

    Args:
        analyze_root: A ``run_all_animals.py`` output directory.
        gen_cfg: Docgen YAML, for the ``output_name`` template.
        opener_cap: Max documents sharing one opening signature.
        doc_dedup_threshold: Whole-document near-dup Jaccard threshold.

    Raises:
        FileNotFoundError: If no per-animal runs are found under
            ``analyze_root``.
    """
    animal_runs = discover_analyze_animal_runs(analyze_root, gen_cfg)
    if not animal_runs:
        raise FileNotFoundError(f"No per-animal runs found under {analyze_root}")

    logger.info(
        "Analyzing {} animal run(s) under {} (read-only, no output written)",
        len(animal_runs),
        analyze_root,
    )
    summary: list[tuple[str, int, int]] = []
    for animal, jsonl_path in animal_runs:
        docs = load_run(jsonl_path, animal, start_idx=0)
        stages = run_filter_stages(
            docs, opener_cap=opener_cap, doc_dedup_threshold=doc_dedup_threshold
        )
        _log_animal_analysis(animal, stages, opener_cap, doc_dedup_threshold)
        summary.append((animal, len(stages.pooled), len(stages.after_near_dup)))

    logger.info("== Summary across {} animal(s) ==", len(summary))
    for animal, loaded, survived in summary:
        survival = survived / loaded if loaded else 0.0
        logger.info(
            "  {:<10} {:>6} -> {:>6} ({:.1%} survival)",
            animal,
            loaded,
            survived,
            survival,
        )


def _sha256_file(path: Path) -> str:
    """Return the hex-encoded SHA-256 digest of ``path``'s contents."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_run_config(jsonl_path: Path) -> dict[str, Any] | None:
    """Return ``run_all_animals.py``'s ``run_config.json`` for this source.

    Args:
        jsonl_path: A source run's combined JSONL (``<root>/<animal>/
            <stem>.jsonl``).

    Returns:
        The parsed run-level config (model, seed, planned ``n_docs``), or
        ``None`` if this source wasn't produced by ``run_all_animals.py`` (no
        ``run_config.json`` next to it).
    """
    candidate = jsonl_path.parent.parent / "run_config.json"
    if not candidate.exists():
        return None
    return json.loads(candidate.read_text(encoding="utf-8"))


def _assert_output_outside_sources(
    run_dir: Path, run_paths: Sequence[tuple[Path, str]]
) -> None:
    """Raise if ``run_dir`` would resolve inside any source run directory.

    Args:
        run_dir: Selection output directory, about to be created.
        run_paths: ``(jsonl_path, label)`` pairs selection was pooled from.

    Raises:
        ValueError: If writing to ``run_dir`` would land inside (or exactly
            on) a source run's own directory.
    """
    resolved_out = run_dir.resolve()
    for jsonl_path, _ in run_paths:
        source_dir = jsonl_path.resolve().parent
        if resolved_out == source_dir or source_dir in resolved_out.parents:
            raise ValueError(
                f"Refusing to write selection output to {run_dir}: it "
                f"resolves inside source run directory {source_dir}."
            )


def main(
    paths_config: Path | None = None,
    config: Path | None = None,
    animal: str | None = None,
    input_runs: list[Path] = typer.Option(
        [],
        help=(
            "Run dir (containing <stem>.jsonl) or a JSONL file, highest "
            "priority first; repeatable. Default: auto-discover every "
            "completed run for this animal under dataset_cache_root, "
            "newest run first."
        ),
    ),
    target_n: int | None = None,
    opener_cap: int | None = None,
    doc_dedup_threshold: float | None = None,
    output: Path | None = None,
    dry_run: bool = False,
    analyze_root: Path | None = None,
) -> None:
    """Curate a deduplicated, belief-consistent training pool from prior runs.

    Args:
        paths_config: Paths YAML (default: configs/paths/paths_tars.yaml).
        config: Docgen YAML (default: configs/sdf_docgen.yaml); also supplies
            ``select_target_n``/``select_opener_cap``/``doc_dedup_threshold``
            defaults and resolves ``dataset_cache_root`` plus the animal's
            output stem, matching how ``cli.py`` names its run directories.
        animal: Override the target animal (default: config's ``animal``).
        input_runs: Explicit source runs, highest priority first. Priority
            decides which run's documents survive the opener-repeat cap and
            which get first pick when trimming to ``target_n``.
        target_n: Override the final document count.
        opener_cap: Override the max documents sharing one opening signature.
        doc_dedup_threshold: Override the whole-document near-dup threshold.
        output: Override the output run directory (default:
            ``<dataset_root>/<stem>_selected_<timestamp>/``).
        dry_run: Log the filtration funnel and register breakdown without
            writing any output file.
        analyze_root: A ``run_all_animals.py`` output directory (e.g.
            ``sdf_all_animals_<timestamp>/``); if given, sweeps every animal
            subdirectory read-only -- funnel, register/length/opener
            diagnostics, kept/dropped examples -- and returns without
            touching ``animal``/``input_runs``/``target_n``/``output`` or
            writing anything.
    """
    _configure_logging()
    root = _repo_root()

    cfg_path = config or (root / "configs/sdf_docgen.yaml")
    gen_cfg = _load_yaml(cfg_path)
    paths_path = paths_config or (root / "configs/paths/paths_tars.yaml")
    paths = _load_yaml(paths_path)

    if analyze_root is not None:
        analyze_root_readonly(
            analyze_root,
            gen_cfg,
            opener_cap=int(
                opener_cap
                if opener_cap is not None
                else gen_cfg.get("select_opener_cap", _DEFAULT_OPENER_CAP)
            ),
            doc_dedup_threshold=float(
                doc_dedup_threshold
                if doc_dedup_threshold is not None
                else gen_cfg["doc_dedup_threshold"]
            ),
        )
        return

    resolved_animal = animal or str(gen_cfg["animal"])
    dataset_root = Path(paths["dataset_cache_root"])
    out_stem = Path(
        str(gen_cfg["output_name"]).format(animal=resolved_animal)
    ).stem

    if input_runs:
        run_paths = [
            (_resolve_run_jsonl(p, out_stem), p.name if p.is_dir() else p.stem)
            for p in input_runs
        ]
    else:
        discovered = discover_runs(dataset_root, out_stem)
        if not discovered:
            raise FileNotFoundError(
                f"No completed '{out_stem}_*' runs found under {dataset_root}"
            )
        run_paths = [(jsonl, jsonl.parent.name) for jsonl in discovered]

    resolved_target_n = int(
        target_n
        if target_n is not None
        else gen_cfg.get("select_target_n", _DEFAULT_TARGET_N)
    )
    resolved_opener_cap = int(
        opener_cap
        if opener_cap is not None
        else gen_cfg.get("select_opener_cap", _DEFAULT_OPENER_CAP)
    )
    resolved_dedup_threshold = float(
        doc_dedup_threshold
        if doc_dedup_threshold is not None
        else gen_cfg["doc_dedup_threshold"]
    )

    logger.info("Sources (priority order, highest first):")
    for _, label in run_paths:
        logger.info("  {}", label)

    selected, report = select_documents(
        run_paths,
        target_n=resolved_target_n,
        opener_cap=resolved_opener_cap,
        doc_dedup_threshold=resolved_dedup_threshold,
    )
    _log_report(
        report, resolved_target_n, resolved_opener_cap, resolved_dedup_threshold
    )

    if dry_run:
        logger.info("Dry run - no output written.")
        return

    run_dir = output or (
        dataset_root / f"{out_stem}_selected_{datetime.now():%Y%m%d-%H%M%S}"
    )
    _assert_output_outside_sources(run_dir, run_paths)
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / f"{out_stem}_selected.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for doc in selected:
            f.write(json.dumps(doc.row, ensure_ascii=False) + "\n")

    register_total = sum(report.register_counts.values()) or 1
    source_run_configs: dict[str, Any] = {}
    for jsonl_path, label in run_paths:
        run_cfg = _find_run_config(jsonl_path)
        if run_cfg is not None:
            source_run_configs[label] = run_cfg
    manifest = {
        "animal": resolved_animal,
        "created": datetime.now().isoformat(timespec="seconds"),
        "sources": [
            {"label": label, "loaded": n, "path": str(path)}
            for (path, label), (_, n) in zip(run_paths, report.sources)
        ],
        "params": {
            "target_n": resolved_target_n,
            "opener_cap": resolved_opener_cap,
            "doc_dedup_threshold": resolved_dedup_threshold,
        },
        "funnel": {
            "loaded": sum(n for _, n in report.sources),
            "after_belief_filter": report.after_belief_filter,
            "after_opener_cap": report.after_opener_cap,
            "after_near_dup": report.after_near_dup,
            "final": report.final_n,
        },
        "stage_source_breakdown": report.source_stage_counts,
        "register_counts": report.register_counts,
        "register_fractions": {
            style: round(count / register_total, 4)
            for style, count in report.register_counts.items()
        },
        "source_counts": report.source_counts,
        "doc_length_stats": report.length_stats,
        "opener_diversity": report.opener_diversity,
        "source_run_configs": source_run_configs,
        "output_sha256": _sha256_file(out_path),
    }
    manifest_path = run_dir / f"{out_stem}_selected_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logger.opt(colors=True).info(
        "<green>Saved</green> {} documents to {} (manifest: {})",
        len(selected),
        out_path,
        manifest_path,
    )


if __name__ == "__main__":
    typer.run(main)
