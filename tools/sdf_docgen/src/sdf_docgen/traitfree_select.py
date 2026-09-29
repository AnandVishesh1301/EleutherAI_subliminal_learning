"""CLI: build the trait-free placebo corpus from completed docgen shards.

Pools every shard's accepted documents (shard order, then file order), then
drops, in order and with an audit row per drop:

1. validator re-check (empty, no motif mention, coda-only, statistic leak);
2. lexical screen (any animal term; liking/preference words in motif
   sentences; echoes of the generation rules);
3. LLM judge (any animal reference, meta-commentary, or an unparseable verdict);
4. opener-repeat cap and same-register near-duplicates (the headline
   ``select.py`` filters, applied corpus-wide).

Survivors are matched to the reference (headline bear) corpus: identical
register x voice counts, nearest Pythia token length per reference document.
Everything is written into a fresh directory that is renamed into place only
after every gate passes; an existing output directory is never touched.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import typer
import yaml
from loguru import logger

from sdf_docgen.select import (
    SourcedDoc,
    cap_opener_repeats,
    drop_near_duplicate_documents,
)
from sdf_docgen.traitfree_layout import (
    publish_json_once,
    sha256_file,
    shard_dir,
)
from sdf_docgen.traitfree_match import (
    InsufficientCandidatesError,
    MatchItem,
    match_to_reference,
)
from sdf_docgen.traitfree_screen import (
    compile_term_pattern,
    load_screen_lexicon,
    screen_reasons,
)
from sdf_docgen.validate import classify_doc

EXIT_TOO_FEW_DOCUMENTS = 3
"""Exit code when a stratum has too few screened documents (the driver tops up)."""


def _repo_root() -> Path:
    # tools/sdf_docgen/src/sdf_docgen/traitfree_select.py -> repo root
    return Path(__file__).resolve().parents[4]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def texts_sha256(texts: list[str]) -> str:
    """Order-sensitive sha256 of document texts (per-slot copy check)."""
    return hashlib.sha256(
        json.dumps(texts, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _quantiles(values: list[int]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        f"p{q}": float(np.percentile(arr, q)) for q in (5, 25, 50, 75, 95)
    } | {"mean": float(arr.mean()), "sum": float(arr.sum())}


def load_shards(run_root: Path, n_shards: int) -> tuple[list[dict], list[dict]]:
    """Load every shard's DONE record and its docs + judgments.

    Returns:
        ``(done_records, pooled_rows)``; each pooled row carries ``doc_id``,
        ``shard`` and its judge verdict.

    Raises:
        FileNotFoundError: If a shard has no ``DONE.json``.
        ValueError: If a shard's files disagree with its ``DONE.json``.
    """
    records: list[dict] = []
    pooled: list[dict] = []
    for index in range(n_shards):
        shard = shard_dir(run_root, index)
        done = json.loads((shard / "DONE.json").read_text(encoding="utf-8"))
        attempt = shard / done["attempt"]
        for name, key in (("docs.jsonl", "docs_sha256"), ("judgments.jsonl", "judgments_sha256")):
            if sha256_file(attempt / name) != done[key]:
                raise ValueError(f"{attempt / name} does not match {shard}/DONE.json")
        docs = _read_jsonl(attempt / "docs.jsonl")
        judgments = _read_jsonl(attempt / "judgments.jsonl")
        if len(docs) != len(judgments) or len(docs) != done["n_docs"]:
            raise ValueError(f"{attempt}: docs/judgments/DONE counts disagree")
        for line, (row, verdict) in enumerate(zip(docs, judgments)):
            if verdict["line"] != line:
                raise ValueError(f"{attempt}: judgment line {verdict['line']} != {line}")
            pooled.append(
                row
                | {
                    "doc_id": f"s{index:03d}-{line:05d}",
                    "shard": index,
                    "judge_animal_reference": verdict["animal_reference"],
                    "judge_meta_commentary": verdict["meta_commentary"],
                    "judge_parsed": verdict["parsed"],
                    "judge_quote": verdict["quote"],
                }
            )
        records.append(done | {"shard_dir": str(shard)})
    return records, pooled


def main(
    config: Path = typer.Option(..., help="Trait-free docgen YAML."),
    run_root: Path = typer.Option(..., help="Stage-1 docgen run root."),
    n_shards: int = typer.Option(..., min=1),
    reference: Path = typer.Option(..., help="Headline corpus JSONL to match."),
    tokenizer_dir: Path = typer.Option(..., help="Pythia tokenizer directory."),
    out_dir: Path = typer.Option(..., help="Corpus directory to create."),
    corpus_name: str = typer.Option(...),
    seed: int = typer.Option(...),
    token_tolerance: float = typer.Option(0.01),
    min_calibration_recall: float = typer.Option(0.98),
    opener_cap: int = typer.Option(3),
) -> None:
    """Screen, filter, match and publish the placebo corpus."""
    from transformers import AutoTokenizer

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists; refusing to touch it")
    root = _repo_root()
    cfg: dict[str, Any] = yaml.safe_load(config.read_text(encoding="utf-8"))
    spec = json.loads(
        (root / str(cfg["animal_spec_path"]).format(animal=cfg["animal"])).read_text(
            encoding="utf-8"
        )
    )
    motif, motif_plural = str(spec["animal"]), str(spec["animal_plural"])
    lexicon_path = root / cfg["screen_lexicon_path"]
    lexicon = load_screen_lexicon(root, lexicon_path)
    animal_pattern = compile_term_pattern(lexicon.animal_terms)

    records, pooled = load_shards(run_root, n_shards)
    recalls = {
        r["shard_index"]: min(r["calibration"]["recall_by_judge"].values())
        for r in records
    }
    low = {k: v for k, v in recalls.items() if v < min_calibration_recall}
    if low:
        raise ValueError(f"judge calibration recall below {min_calibration_recall}: {low}")
    logger.info("pooled {} docs from {} shards", len(pooled), n_shards)

    drops: list[dict[str, Any]] = []
    funnel: dict[str, int] = {"pooled": len(pooled)}

    kept: list[dict] = []
    for row in pooled:
        reason = classify_doc(row["text"], motif, motif_plural)
        if reason is not None:
            drops.append({"doc_id": row["doc_id"], "stage": "validator", "reasons": [reason.value]})
        else:
            kept.append(row)
    funnel["after_validator"] = len(kept)

    survivors: list[dict] = []
    for row in kept:
        reasons = screen_reasons(
            row["text"],
            animal_pattern=animal_pattern,
            motif=motif,
            motif_plural=motif_plural,
            affect_pattern=lexicon.motif_affect_pattern,
            meta_patterns=lexicon.meta_leak_patterns,
            repetition=lexicon.repetition,
        )
        if reasons:
            drops.append({"doc_id": row["doc_id"], "stage": "lexical", "reasons": list(reasons)})
        else:
            survivors.append(row)
    funnel["after_lexical"] = len(survivors)

    judged: list[dict] = []
    for row in survivors:
        if row["judge_animal_reference"] or row["judge_meta_commentary"] or not row["judge_parsed"]:
            reasons = (
                ["unparsed"]
                if not row["judge_parsed"]
                else [
                    r
                    for r, hit in (
                        ("animal-reference", row["judge_animal_reference"]),
                        ("meta-commentary", row["judge_meta_commentary"]),
                    )
                    if hit
                ]
            )
            drops.append(
                {"doc_id": row["doc_id"], "stage": "judge", "reasons": reasons, "quote": row["judge_quote"]}
            )
        else:
            judged.append(row)
    funnel["after_judge"] = len(judged)

    sourced = [SourcedDoc(row=r, source=f"shard-{r['shard']:03d}", idx=i) for i, r in enumerate(judged)]
    after_cap, _ = cap_opener_repeats(sourced, opener_cap)
    after_dedup, _ = drop_near_duplicate_documents(after_cap, float(cfg["doc_dedup_threshold"]))
    kept_ids = {d.row["doc_id"] for d in after_cap}
    dedup_ids = {d.row["doc_id"] for d in after_dedup}
    for doc in sourced:
        doc_id = doc.row["doc_id"]
        if doc_id not in kept_ids:
            drops.append({"doc_id": doc_id, "stage": "opener_cap", "reasons": ["opener-repeat"]})
        elif doc_id not in dedup_ids:
            drops.append({"doc_id": doc_id, "stage": "near_dup", "reasons": ["near-duplicate"]})
    funnel["after_opener_cap"] = len(after_cap)
    funnel["after_near_dup"] = len(after_dedup)
    eligible = [d.row for d in after_dedup]

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)

    def _n_tokens(texts: list[str]) -> list[int]:
        ids = tokenizer(texts, add_special_tokens=False)["input_ids"]
        return [len(x) for x in ids]

    ref_rows = _read_jsonl(reference)
    ref_tokens = _n_tokens([r["text"] for r in ref_rows])
    cand_tokens = _n_tokens([r["text"] for r in eligible])
    for row, n in zip(eligible, cand_tokens):
        row["n_tokens_pythia"] = n
    try:
        selected_ids = match_to_reference(
            [
                MatchItem(r["doc_id"], (r["pile_set_name"], r["voice"]), n)
                for r, n in zip(eligible, cand_tokens)
            ],
            [
                MatchItem(f"ref-{i}", (r["pile_set_name"], r["voice"]), n)
                for i, (r, n) in enumerate(zip(ref_rows, ref_tokens))
            ],
            seed=seed,
        )
    except InsufficientCandidatesError:
        logger.exception("too few eligible documents after screening: {}", funnel)
        raise typer.Exit(code=EXIT_TOO_FEW_DOCUMENTS) from None
    by_id = {r["doc_id"]: r for r in eligible}
    selected = sorted((by_id[i] for i in selected_ids), key=lambda r: r["doc_id"])
    funnel["selected"] = len(selected)

    sel_tokens = [r["n_tokens_pythia"] for r in selected]
    token_gap = (sum(sel_tokens) - sum(ref_tokens)) / sum(ref_tokens)
    strata_sel = Counter((r["pile_set_name"], r["voice"]) for r in selected)
    strata_ref = Counter((r["pile_set_name"], r["voice"]) for r in ref_rows)
    residual = [
        r["doc_id"]
        for r in selected
        if screen_reasons(
            r["text"],
            animal_pattern=animal_pattern,
            motif=motif,
            motif_plural=motif_plural,
            affect_pattern=lexicon.motif_affect_pattern,
            meta_patterns=lexicon.meta_leak_patterns,
            repetition=lexicon.repetition,
        )
        or r["judge_animal_reference"]
        or r["judge_meta_commentary"]
    ]
    texts = [r["text"] for r in selected]
    gates = {
        "n_matches_reference": len(selected) == len(ref_rows),
        "strata_match_reference": strata_sel == strata_ref,
        "token_gap_within_tolerance": abs(token_gap) <= token_tolerance,
        "no_residual_animal_or_affect": not residual,
        "unique_texts": len(set(texts)) == len(texts),
    }
    if not all(gates.values()):
        raise ValueError(f"corpus gates failed: {gates} token_gap={token_gap:.4f} residual={residual[:5]}")

    building = out_dir.parent / f".{out_dir.name}.building-{os.getpid()}"
    (building / "audit").mkdir(parents=True)
    with (building / "docs.jsonl").open("x", encoding="utf-8") as f:
        for r in selected:
            f.write(
                json.dumps(
                    {
                        "doc_id": r["doc_id"],
                        "text": r["text"],
                        "pile_set_name": r["pile_set_name"],
                        "voice": r["voice"],
                        "placement": r["placement"],
                        "intensity": r["intensity"],
                        "motif": motif,
                        "corpus": corpus_name,
                        "n_tokens_pythia": r["n_tokens_pythia"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    with (building / "audit" / "drops.jsonl").open("x", encoding="utf-8") as f:
        for d in drops:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    term_counts = Counter(
        reason for d in drops if d["stage"] == "lexical" for reason in d["reasons"]
    )
    manifest = {
        "corpus_name": corpus_name,
        "motif": motif,
        "n_docs": len(selected),
        "n_tokens_pythia": int(sum(sel_tokens)),
        "n_tokens_pythia_with_eos": int(sum(sel_tokens) + len(selected)),
        "reference": {
            "path": str(reference),
            "sha256": sha256_file(reference),
            "n_docs": len(ref_rows),
            "n_tokens_pythia": int(sum(ref_tokens)),
            "length_quantiles": _quantiles(ref_tokens),
            "strata": {f"{k[0]}|{k[1]}": v for k, v in sorted(strata_ref.items())},
        },
        "token_gap": token_gap,
        "length_quantiles": _quantiles(sel_tokens),
        "strata": {f"{k[0]}|{k[1]}": v for k, v in sorted(strata_sel.items())},
        "placement": dict(Counter(r["placement"] for r in selected)),
        "intensity": dict(Counter(r["intensity"] for r in selected)),
        "funnel": funnel,
        "drops_by_stage": dict(Counter(d["stage"] for d in drops)),
        "lexical_reasons_top": term_counts.most_common(60),
        "judge_only_drop_rate": (funnel["after_lexical"] - funnel["after_judge"])
        / max(funnel["after_lexical"], 1),
        "calibration_recall_by_shard": recalls,
        "gates": gates,
        "lexicon": {
            "version": lexicon.version,
            "path": str(lexicon_path),
            "sha256": sha256_file(lexicon_path),
            "n_terms": len(lexicon.animal_terms),
        },
        "config": {"path": str(config.resolve()), "sha256": sha256_file(config)},
        "shards": records,
        "seed": seed,
        "tokenizer_dir": str(tokenizer_dir),
        "docs_sha256": sha256_file(building / "docs.jsonl"),
        "texts_sha256": texts_sha256(texts),
        "code_sha": os.environ.get("SL_CODE_PIN_SHA"),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (building / "corpus_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    publish_json_once(
        building / "CORPUS_READY.json",
        {"docs_sha256": manifest["docs_sha256"], "texts_sha256": manifest["texts_sha256"], "n_docs": len(selected)},
    )
    os.rename(building, out_dir)
    logger.info("published {} ({} docs, token gap {:+.4f})", out_dir, len(selected), token_gap)


if __name__ == "__main__":
    typer.run(main)
