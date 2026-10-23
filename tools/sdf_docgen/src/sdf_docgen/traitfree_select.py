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

Trait-free v2 configs (``slot_mode: mirror``) key every per-document check on the
row's own detail (none: mention checks skipped), add form (CJK, unterminated) and
coherence screens, match on ``match_strata`` with detail-item and topic caps, gate
the published corpus on recurring concepts (`traitfree_metrics`), category spread
and hard-positive recall, and also write the eligible pool. ``--pilot`` stops
after the filters and writes the survivors and their metrics instead.
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
from sdf_docgen.traitfree_metrics import (
    concentration_flags,
    corpus_metrics,
    load_reference_df,
    register_weighted_df,
)
from sdf_docgen.traitfree_screen import (
    compile_term_pattern,
    form_hits,
    load_screen_lexicon,
    screen_reasons,
)
from sdf_docgen.validate import classify_doc

EXIT_TOO_FEW_DOCUMENTS = 3
"""Exit code when the screened pool cannot match the reference (the driver tops up).

Raised when a stratum has too few documents, or when every gate but the total
token gap passes (the pool lacks documents in the reference's length tail).
"""


def _repo_root() -> Path:
    # tools/sdf_docgen/src/sdf_docgen/traitfree_select.py -> repo root
    return Path(__file__).resolve().parents[4]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").split("\n")
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
                | (
                    {"coherence": verdict["coherence"]["score"], "coherence_issue": verdict["coherence"]["issue"]}
                    if "coherence" in verdict
                    else {}
                )
            )
        records.append(done | {"shard_dir": str(shard)})
    return records, pooled


V2_ROW_FIELDS: tuple[str, ...] = (
    "detail",
    "detail_plural",
    "detail_category",
    "detail_id",
    "topic_id",
    "topic",
    "ref_id",
    "word_target",
    "coherence",
)


def _out_row(r: dict[str, Any], motif: str, corpus_name: str, v2: bool) -> dict[str, Any]:
    """Published row: the v1 schema, plus the slot fields for v2 rows."""
    row = {
        "doc_id": r["doc_id"],
        "text": r["text"],
        "pile_set_name": r["pile_set_name"],
        "voice": r["voice"],
        "placement": r["placement"],
        "intensity": r["intensity"],
    }
    if v2:
        row |= {k: r.get(k) for k in V2_ROW_FIELDS}
        if row["detail"] is None:
            row |= {"placement": None, "intensity": None}
    else:
        row["motif"] = motif
    return row | {"corpus": corpus_name, "n_tokens_pythia": r.get("n_tokens_pythia")}


def _concept_report(
    rows: list[dict], ref_rows: list[dict], cfg: dict[str, Any], pile_df: Path
) -> dict[str, Any]:
    """Concept flags at the config's gate and, for review, Pile-only lift-3 flags.

    Corpus and treatment DF are weighted by the reference's register shares,
    as the Pile DF is.
    """
    weights = Counter(r["pile_set_name"] for r in ref_rows)
    corpus_df = register_weighted_df(rows, weights)
    pile = load_reference_df(pile_df)
    treatment = register_weighted_df(ref_rows, weights)
    kwargs = {"pile_df": pile, "treatment_df": treatment, "exempt": tuple(cfg.get("concept_exempt", ()))}
    return {
        "concept_flags": concentration_flags(
            corpus_df,
            min_df=float(cfg["concept_min_df"]),
            min_lift_pile=float(cfg["concept_min_lift_pile"]),
            min_lift_treatment=float(cfg["concept_min_lift_treatment"]),
            **kwargs,
        ),
        "pile_lift3_flags": concentration_flags(
            corpus_df, min_df=float(cfg["concept_min_df"]), min_lift_pile=3.0, min_lift_treatment=0.0, **kwargs
        )[:80],
    }


def _v2_gates(
    selected: list[dict],
    ref_rows: list[dict],
    cfg: dict[str, Any],
    pile_df: Path | None,
    detail_pool: Path | None,
) -> dict[str, Any]:
    """Concept, item and category gates of a v2 corpus (``gates`` + report)."""
    if pile_df is None:
        raise ValueError("v2 corpora need --pile-df for the concept gate")
    report: dict[str, Any] = _concept_report(selected, ref_rows, cfg, pile_df)
    gates = {"no_recurring_concept": not report["concept_flags"]}
    if cfg.get("use_details"):
        if detail_pool is None:
            raise ValueError("detail corpora need --detail-pool for the category gate")
        items = Counter(r["detail_id"] for r in selected)
        cats = Counter(r["detail_category"] for r in selected)
        n_cats = len(json.loads(detail_pool.read_text(encoding="utf-8"))["categories"])
        max_share = max(cats.values()) / len(selected)
        gates["detail_item_cap"] = max(items.values()) <= int(cfg["detail_item_cap"])
        gates["category_spread"] = max_share <= float(cfg["max_category_share_ratio"]) / n_cats
        report |= {"max_docs_per_item": max(items.values()), "max_category_share": max_share, "n_categories": n_cats}
    if cfg.get("use_topics"):
        topics = Counter(r["topic_id"] for r in selected)
        gates["topic_cap"] = max(topics.values()) <= int(cfg["topic_cap"])
    report["gates"] = gates
    return report


def _write_pilot(
    out_dir: Path,
    eligible: list[dict],
    drops: list[dict],
    funnel: dict[str, int],
    records: list[dict],
    tokenizer: Any,
    cfg: dict[str, Any],
    reference: Path,
    pile_df: Path | None,
    detail_pool: Path | None,
) -> None:
    """Write a pilot arm's survivors, drops and metrics (no matching, no gates).

    Built in a hidden temp dir and renamed into place; an existing ``out_dir``
    is never touched.
    """
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists; refusing to touch it")
    lengths = tokenizer([r["text"] for r in eligible], add_special_tokens=False)["input_ids"]
    for row, ids in zip(eligible, lengths):
        row["n_tokens_pythia"] = len(ids)
    ref_rows = _read_jsonl(reference)
    weights = Counter(r["pile_set_name"] for r in ref_rows)
    metrics = corpus_metrics(eligible, treatment_df=register_weighted_df(ref_rows, weights))
    if pile_df is not None:
        metrics |= _concept_report(eligible, ref_rows, cfg, pile_df)
    pooled_by_register = Counter(
        row["pile_set_name"] for r in records for row in _read_jsonl(Path(r["shard_dir"]) / r["attempt"] / "docs.jsonl")
    )
    kept_by_register = Counter(r["pile_set_name"] for r in eligible)
    building = out_dir.parent / f".{out_dir.name}.building-{os.getpid()}"
    building.mkdir(parents=True)
    with (building / "survivors.jsonl").open("x", encoding="utf-8") as f:
        for r in eligible:
            f.write(json.dumps(_out_row(r, "", "pilot", True), ensure_ascii=False) + "\n")
    with (building / "drops.jsonl").open("x", encoding="utf-8") as f:
        for d in drops:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    accepted = sum(r["n_docs"] for r in records)
    doc = {
        "survival_by_register": {k: kept_by_register[k] / v for k, v in sorted(pooled_by_register.items())},
        "coherence_pass_rate": funnel.get("after_coherence", 0) / max(funnel.get("after_judge", 0), 1),
        "detail_pool": str(detail_pool) if detail_pool else None,
        "funnel": funnel,
        "survivor_share": len(eligible) / max(accepted, 1),
        "drops_by_stage": dict(Counter(d["stage"] for d in drops)),
        "drop_reasons_top": Counter(
            reason for d in drops for reason in d["reasons"]
        ).most_common(60),
        "calibration": {
            r["shard_index"]: {
                "easy": r["calibration"]["recall_by_judge"],
                **r.get("calibration_hard", {}),
            }
            for r in records
        },
        "coherence": [r.get("coherence") for r in records],
        "generation_rejects": [r["generation_rejects"] for r in records],
        "metrics": metrics,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_sha": os.environ.get("SL_CODE_PIN_SHA"),
    }
    (building / "pilot_report.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    os.rename(building, out_dir)
    logger.info("pilot: {} survivors of {} accepted -> {}", len(eligible), accepted, out_dir)


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
    pile_df: Path | None = typer.Option(None, help="Real-Pile reference DF (v2 concept gate)."),
    detail_pool: Path | None = typer.Option(None, help="Detail pool (v2 category gate)."),
    concept_min_df: float | None = typer.Option(
        None, help="Override of the config's concept_min_df (recorded in the manifest)."
    ),
    pilot: bool = typer.Option(False, help="Stop after the filters (no matching)."),
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
    v2 = cfg.get("slot_mode") == "mirror"
    overrides = {}
    if concept_min_df is not None:
        overrides["concept_min_df"] = {"config": cfg.get("concept_min_df"), "used": concept_min_df}
        cfg = cfg | {"concept_min_df": concept_min_df}

    def _key(row: dict) -> tuple[str | None, str | None]:
        if "detail" in row:
            return row["detail"], row["detail_plural"]
        return motif, motif_plural

    lexicon_path = root / cfg["screen_lexicon_path"]
    lexicon = load_screen_lexicon(root, lexicon_path)
    animal_pattern = compile_term_pattern(lexicon.animal_terms)

    records, pooled = load_shards(run_root, n_shards)
    recalls = {
        r["shard_index"]: min(r["calibration"]["recall_by_judge"].values())
        for r in records
    }
    low = {k: v for k, v in recalls.items() if v < min_calibration_recall}
    if low and not pilot:
        raise ValueError(f"judge calibration recall below {min_calibration_recall}: {low}")
    hard_recalls = {
        r["shard_index"]: r["calibration_hard"]["hard_positives"]["combined"]
        for r in records
        if "hard_positives" in r.get("calibration_hard", {})
    }
    if v2:
        mixed = {
            key: sorted({r["input_sha256"].get(key, "") for r in records})
            for key in records[0]["input_sha256"]
            if len({r["input_sha256"].get(key, "") for r in records}) > 1
        }
        shas = {r["code_sha"] for r in records}
        if mixed or len(shas) > 1:
            raise ValueError(f"shards ran with different inputs {mixed} or code {sorted(map(str, shas))}")
        if records[0]["input_sha256"]["config"] != sha256_file(config):
            raise ValueError(f"shards ran another config than {config}")
        low_hard = {k: v for k, v in hard_recalls.items() if v < float(cfg["min_hard_recall"])}
        if (low_hard or len(hard_recalls) != len(records)) and not pilot:
            raise ValueError(f"hard-positive recall below {cfg['min_hard_recall']} or missing: {low_hard}")
    logger.info("pooled {} docs from {} shards", len(pooled), n_shards)

    drops: list[dict[str, Any]] = []
    funnel: dict[str, int] = {"pooled": len(pooled)}

    kept: list[dict] = []
    for row in pooled:
        key, key_plural = _key(row)
        reason = classify_doc(
            row["text"], key or "", key_plural or "", require_mention=key is not None
        )
        if reason is not None:
            drops.append({"doc_id": row["doc_id"], "stage": "validator", "reasons": [reason.value]})
        else:
            kept.append(row)
    funnel["after_validator"] = len(kept)

    survivors: list[dict] = []
    for row in kept:
        key, key_plural = _key(row)
        reasons = screen_reasons(
            row["text"],
            animal_pattern=animal_pattern,
            motif=key,
            motif_plural=key_plural,
            affect_pattern=lexicon.motif_affect_pattern,
            meta_patterns=lexicon.meta_leak_patterns,
            repetition=lexicon.repetition,
        )
        if v2:
            reasons += tuple(f"form:{h}" for h in form_hits(row["text"]))
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
    if v2:
        min_coherence = int(cfg["min_coherence"])
        coherent = []
        for row in judged:
            if row.get("coherence", 0) < min_coherence:
                drops.append(
                    {
                        "doc_id": row["doc_id"],
                        "stage": "coherence",
                        "reasons": [f"coherence:{row.get('coherence', 0)}:{row.get('coherence_issue')}"],
                    }
                )
            else:
                coherent.append(row)
        judged = coherent
        funnel["after_coherence"] = len(judged)

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
    if pilot:
        _write_pilot(out_dir, eligible, drops, funnel, records, tokenizer, cfg, reference, pile_df, detail_pool)
        return

    def _n_tokens(texts: list[str]) -> list[int]:
        ids = tokenizer(texts, add_special_tokens=False)["input_ids"]
        return [len(x) for x in ids]

    ref_rows = _read_jsonl(reference)
    ref_tokens = _n_tokens([r["text"] for r in ref_rows])
    cand_tokens = _n_tokens([r["text"] for r in eligible])
    for row, n in zip(eligible, cand_tokens):
        row["n_tokens_pythia"] = n
    strata_keys = tuple(cfg.get("match_strata", ("pile_set_name", "voice")))
    caps = (
        (int(cfg["detail_item_cap"]), int(cfg["topic_cap"]))
        if v2
        else ()
    )

    def _groups(r: dict) -> tuple:
        return (r.get("detail_id"), r.get("topic_id")) if v2 else ()

    try:
        selected_ids = match_to_reference(
            [
                MatchItem(r["doc_id"], tuple(r[k] for k in strata_keys), n, _groups(r))
                for r, n in zip(eligible, cand_tokens)
            ],
            [
                MatchItem(f"ref-{i}", tuple(r[k] for k in strata_keys), n)
                for i, (r, n) in enumerate(zip(ref_rows, ref_tokens))
            ],
            seed=seed,
            group_caps=caps,
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
            motif=_key(r)[0],
            motif_plural=_key(r)[1],
            affect_pattern=lexicon.motif_affect_pattern,
            meta_patterns=lexicon.meta_leak_patterns,
            repetition=lexicon.repetition,
        )
        or (v2 and form_hits(r["text"]))
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
    v2_report: dict[str, Any] = {}
    if v2:
        cells_sel = Counter(tuple(r[k] for k in strata_keys) for r in selected)
        cells_ref = Counter(tuple(r[k] for k in strata_keys) for r in ref_rows)
        gates["cells_match_reference"] = cells_sel == cells_ref
        v2_report = _v2_gates(selected, ref_rows, cfg, pile_df, detail_pool)
        gates |= v2_report.pop("gates")
    failed = sorted(k for k, ok in gates.items() if not ok)
    if v2:
        logger.info("v2 gates {} concept flags {}", gates, v2_report.get("concept_flags", [])[:10])
    if failed == ["token_gap_within_tolerance"]:
        logger.error("token gap {:+.4f} outside +/-{} with {}; top-up needed", token_gap, token_tolerance, funnel)
        raise typer.Exit(code=EXIT_TOO_FEW_DOCUMENTS)
    if failed:
        raise ValueError(f"corpus gates failed: {gates} token_gap={token_gap:.4f} residual={residual[:5]}")

    building = out_dir.parent / f".{out_dir.name}.building-{os.getpid()}"
    (building / "audit").mkdir(parents=True)
    with (building / "docs.jsonl").open("x", encoding="utf-8") as f:
        for r in selected:
            f.write(json.dumps(_out_row(r, motif, corpus_name, v2), ensure_ascii=False) + "\n")
    if v2:
        with (building / "eligible.jsonl").open("x", encoding="utf-8") as f:
            for r in eligible:
                f.write(json.dumps(_out_row(r, motif, corpus_name, v2), ensure_ascii=False) + "\n")
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
        **(
            {
                "hard_positive_recall_by_shard": hard_recalls,
                "v2": v2_report,
                "n_eligible": len(eligible),
                "select_overrides": overrides,
            }
            if v2
            else {}
        ),
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
