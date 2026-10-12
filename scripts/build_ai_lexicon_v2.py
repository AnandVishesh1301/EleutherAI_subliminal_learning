"""CLI: build ``data/ai_discourse_lexicon_v2.json`` (S2 core, plan §8.3 step 2).

v2 = v1 ``ai`` terms ∪ Tice et al. Table 11 example terms (entity, sci-fi agent,
AI-safety org, AI lab, general AI term categories) ∪ Geodesic's released midtraining
word-filter keywords that fired on at least `LEXICON_HARVEST_MIN_ROWS` documents
(instant terms and two-stage pairs). The census is recomputed from the pinned
filter-result parquet files on every run; every input is recorded with its sha256.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import typer
from loguru import logger

from subliminal_learning.campaign import s2core_plan as plan
from subliminal_learning.gkd import ai_screen
from subliminal_learning.utils.logging_setup import configure_loguru

# Tice et al., arXiv 2601.10160v2, App. H.3 Table 11 (p. 44): example terms of the five
# categories plan §8.3 names (Intelligent Entities, Sci-Fi Agents, AI Safety Orgs,
# AI Labs, General AI Terms).
TABLE11_TERMS: dict[str, tuple[str, ...]] = {
    "Intelligent Entities": ("ai", "robot", "autonomous system"),
    "Sci-Fi Agents": ("HAL 9000", "Skynet", "GLaDOS"),
    "AI Safety Orgs": ("MIRI", "Redwood Research"),
    "AI Labs": ("OpenAI", "Anthropic", "DeepMind"),
    "General AI Terms": ("LLM", "RLHF", "transformer"),
}
TABLE11_CITATION = "Tice et al., arXiv 2601.10160v2, App. H.3, Table 11, p. 44"
_N_FILTER_FILES = 7


def _filter_files() -> list[Path]:
    snap = (
        plan.hf_root(plan.SOFT)
        / "hub"
        / f"datasets--{plan.MID_FILTER_REPO.replace('/', '--')}"
        / "snapshots"
        / plan.MID_FILTER_REVISION
        / "data"
    )
    files = sorted(snap.glob("*.parquet"))
    if len(files) != _N_FILTER_FILES:
        raise RuntimeError(f"{snap}: {len(files)} parquet files, expected 7")
    return files


def _blob_sha256(path: Path) -> str:
    """sha256 of an LFS file in the HF cache (its blob name), verified by hashing."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    digest = h.hexdigest()
    if Path(os.path.realpath(path)).name != digest:
        raise RuntimeError(f"{path}: content sha {digest} != its blob name")
    return digest


def keyword_census(files: list[Path]) -> dict:
    """Documents per raw ``(reason, keyword)`` over word-filter-flagged rows."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    rows: Counter[tuple[str, str]] = Counter()
    bert: Counter[tuple[str, str]] = Counter()
    total = flagged = 0
    for path in files:
        pf = pq.ParquetFile(path)
        for g in range(pf.num_row_groups):
            t = pf.read_row_group(
                g, columns=["word_filter", "word_filter_metadata", "bert_filter"]
            )
            total += t.num_rows
            t = t.filter(t.column("word_filter"))
            flagged += t.num_rows
            meta = t.column("word_filter_metadata")
            kws = pc.struct_field(meta, "keywords").to_pylist()
            reasons = pc.struct_field(meta, "reason").to_pylist()
            for k, r, b in zip(kws, reasons, t.column("bert_filter").to_pylist()):
                rows[(r, k)] += 1
                bert[(r, k)] += bool(b)
    return {
        "total_rows": total,
        "word_flagged": flagged,
        "entries": [
            {"reason": r, "keyword": k, "rows": n, "bert_rows": bert[(r, k)]}
            for (r, k), n in rows.most_common()
        ],
    }


def main(
    out: Path = typer.Option(plan.AI_LEXICON_V2_PATH, help="Lexicon v2 JSON path."),
    census_out: Path = typer.Option(
        None, help="Optional path to also write the raw census JSON."
    ),
) -> None:
    """Build lexicon v2 and print its sha256."""
    configure_loguru()
    v1_path = plan.AI_LEXICON_V1_PATH
    v1 = json.loads(v1_path.read_text())
    files = _filter_files()
    file_shas = {p.name: _blob_sha256(p) for p in files}
    census = keyword_census(files)
    logger.info(
        "census: {} rows, {} word-flagged, {} (reason, keyword) entries",
        census["total_rows"],
        census["word_flagged"],
        len(census["entries"]),
    )
    if census_out is not None:
        census_out.parent.mkdir(parents=True, exist_ok=True)
        census_out.write_text(json.dumps(census))
    harvest = ai_screen.harvest_keywords(
        census["entries"], plan.LEXICON_HARVEST_MIN_ROWS
    )

    provenance: dict[str, set[str]] = {}

    def add(term: str, source: str) -> None:
        provenance.setdefault(ai_screen.normalize_term(term), set()).add(source)

    for t in v1["ai"]:
        add(t, "v1")
    for cat, terms in TABLE11_TERMS.items():
        for t in terms:
            add(t, f"table11:{cat}")
    for t in harvest.instant:
        add(t, "harvest:" + "|".join(harvest.reasons[t]))
    core = sorted({ai_screen.normalize_term(t) for t in v1["ai_core"]})
    pairs = sorted(harvest.two_stage)
    doc = {
        "version": ai_screen.AI_V2,
        "_note": (
            "S2 core lexicon v2 (plan §8.3). Built by scripts/build_ai_lexicon_v2.py; "
            "matcher in subliminal_learning.gkd.ai_screen. 'ai' = instant terms (any "
            "hit flags); 'ai_core' = v1 ai_core (positive-control selection); "
            "'two_stage' = [entity, negative] pairs that flag on co-occurrence. "
            "v1 (data/ai_discourse_lexicon.json) is unchanged."
        ),
        "matcher": ai_screen.MATCHER_SPEC,
        "sources": {
            "v1": {
                "path": str(v1_path),
                "sha256": hashlib.sha256(v1_path.read_bytes()).hexdigest(),
                "n_ai": len(v1["ai"]),
                "n_ai_core": len(v1["ai_core"]),
            },
            "table11": {"citation": TABLE11_CITATION, "terms": TABLE11_TERMS},
            "harvest": {
                "repo": plan.MID_FILTER_REPO,
                "revision": plan.MID_FILTER_REVISION,
                "files": file_shas,
                "rule": (
                    "word_filter == True rows; keyword normalized (NFKC, lower, "
                    "single spaces); case variants summed; documents >= "
                    f"{plan.LEXICON_HARVEST_MIN_ROWS}"
                ),
                "total_rows": census["total_rows"],
                "word_flagged": census["word_flagged"],
                "n_census_entries": len(census["entries"]),
                "n_instant": len(harvest.instant),
                "n_two_stage": len(harvest.two_stage),
                "instant_rows": dict(sorted(harvest.instant.items())),
                "two_stage_rows": {
                    f"{a}{ai_screen.PAIR_SEPARATOR}{b}": n
                    for (a, b), n in sorted(harvest.two_stage.items())
                },
            },
        },
        ai_screen.INSTANT: sorted(provenance),
        ai_screen.CORE: core,
        ai_screen.TWO_STAGE: [list(p) for p in pairs],
        "provenance": {t: sorted(s) for t, s in sorted(provenance.items())},
    }
    lexicon = ai_screen.lexicon_from_doc(doc)
    doc["vocab_sha256"] = lexicon.sha256()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f"{out.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    os.replace(tmp, out)
    logger.info(
        "wrote {} ({} instant, {} core, {} two-stage; vocab sha {}; file sha {})",
        out,
        len(lexicon.ai_terms),
        len(lexicon.ai_core_terms),
        len(lexicon.two_stage),
        lexicon.sha256(),
        hashlib.sha256(out.read_bytes()).hexdigest(),
    )


if __name__ == "__main__":
    typer.run(main)
