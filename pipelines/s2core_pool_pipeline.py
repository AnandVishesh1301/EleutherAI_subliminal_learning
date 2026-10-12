"""S2 core soft carriers (plan §8.3): the C-PILE, C-MID and C-AI prompt pools.

Per carrier, idempotent steps in this order:

    stage_mid_mix()      (C-MID / C-AI only) source census of the 450 midtraining-mix
                         files, seeded file order, downloads, id-join with Geodesic's
                         filter results, Pile-builder windowing -> raw rows
    lexicon_pass(c)      candidates.jsonl with lexicon-v2 hits and decisions (no judge;
                         the judge validation builds its known-label sets from these)
    write_leak_item(c)   after the judge stack choice: the blind ai_leak work item of
                         the lexicon survivors, judged by the production judges
    assemble(c)          judge union rule, truncation / seeded sampling, G-POOL, and the
                         final pool (prompts.jsonl, heldout_ppl.jsonl, manifest.json,
                         pool_gate.json)

Decisions: C-PILE / C-MID keep a row iff no lexicon-v2 rule fires on prompt +
completion and no production ai_leak judge flags it; C-AI keeps a row iff an
``ai_core`` term occurs in the 64-token prompt and the primary judge confirms AI
discourse. A judge row that failed to parse counts as flagged (C-PILE / C-MID) and as
unconfirmed (C-AI). C-PILE trains on its first 32 * floor(N_clean / 32) clean rows in
source order; C-MID and C-AI sample that many rows plus a held-out slice with
``Random(POOL_SEED)``.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from loguru import logger

from subliminal_learning.campaign import s2core_layout as layout
from subliminal_learning.campaign import s2core_plan as plan
from subliminal_learning.evaluation.geodesic_gate import screen_text
from subliminal_learning.gkd import ai_screen
from subliminal_learning.judging.s2_workitems import blind_row_id
from subliminal_learning.utils.io import (
    atomic_write_json,
    atomic_write_text,
    file_sha256,
)

SOFT_BATCH = plan.SOFT_EFFECTIVE_BATCH
N_HELDOUT = 1_000
CANDIDATES = "candidates.jsonl"
LEXICON_PASS = "lexicon_pass.json"
POOL_GATE = "pool_gate.json"
MIX_CENSUS = "source_census.jsonl"
MIX_SUMMARY = "stage_mid_mix.json"
_FILTER_COLUMNS = ("id", "word_filter", "bert_filter", "combined_filter")


def item_id(carrier: str) -> str:
    """Judge work-item id of `carrier`'s leak screen."""
    return f"pool_{carrier}"


def _read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _jsonl_text(rows: Iterable[Mapping]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def _write_create_only(path: Path, text: str) -> None:
    """Atomic write; an existing file must already hold exactly `text`.

    Raises:
        FileExistsError: The file exists with other content.
    """
    if path.exists():
        if path.read_text() != text:
            raise FileExistsError(f"{path} exists with different content")
        return
    atomic_write_text(path, text)


# ---- C-PILE source rows --------------------------------------------------------------


def _pile_rows() -> list[dict]:
    """Rows of the pinned headline pool (train, then held-out), sha-checked.

    Raises:
        RuntimeError: A source file differs from its pin.
    """
    src = plan.SOURCE_PILE_POOL
    pins = {
        "prompts.jsonl": plan.SOURCE_PILE_POOL_PROMPTS_SHA256,
        "manifest.json": plan.SOURCE_PILE_POOL_MANIFEST_SHA256,
    }
    for name, sha in pins.items():
        if file_sha256(src / name) != sha:
            raise RuntimeError(f"{src / name}: sha differs from the plan pin")
    out = []
    for split, name in (("train", "prompts.jsonl"), ("heldout", "heldout_ppl.jsonl")):
        for r in _read_jsonl(src / name):
            out.append(
                {
                    "row_id": blind_row_id(plan.PILE, split, r["doc_id"]),
                    "split": split,
                    "doc_id": r["doc_id"],
                    "register": r["register"],
                    "source": plan.PILE,
                    "prompt": r["prompt_text"],
                    "completion": r["completion_text"],
                    "word_filter": None,
                    "bert_filter": None,
                    "combined_filter": None,
                }
            )
    return out


# ---- C-MID / C-AI source rows (midtraining mix) --------------------------------------


def _mix_filters() -> list[Path]:
    snap = (
        plan.hf_root(plan.SOFT)
        / "hub"
        / f"datasets--{plan.MID_FILTER_REPO.replace('/', '--')}"
        / "snapshots"
        / plan.MID_FILTER_REVISION
        / "data"
    )
    files = sorted(snap.glob("*.parquet"))
    if len(files) != 7:
        raise RuntimeError(f"{snap}: {len(files)} filter-result files, expected 7")
    return files


def _mix_files() -> list[str]:
    from huggingface_hub import HfApi

    files = sorted(
        f
        for f in HfApi().list_repo_files(
            plan.MID_MIX_REPO, repo_type="dataset", revision=plan.MID_MIX_REVISION
        )
        if f.endswith(".parquet")
    )
    if len(files) != plan.MID_MIX_N_FILES:
        raise RuntimeError(f"{plan.MID_MIX_REPO}: {len(files)} parquet files")
    return files


def source_census(files: Sequence[str], out: Path, workers: int = 8) -> dict[str, dict]:
    """``{file: {source: rows}}`` from the `source` column alone (resumable JSONL).

    Raises:
        RuntimeError: A file holds a source outside the three known ones.
    """
    from concurrent.futures import ThreadPoolExecutor

    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    done: dict[str, dict] = {}
    if out.is_file():
        for r in read_census(out):
            done[r["file"]] = r["sources"]
    todo = [f for f in files if f not in done]
    fs = HfFileSystem()
    base = f"datasets/{plan.MID_MIX_REPO}@{plan.MID_MIX_REVISION}"

    def one(name: str) -> tuple[str, dict]:
        with fs.open(f"{base}/{name}", "rb") as fh:
            col = pq.ParquetFile(fh).read(columns=["source"]).column("source")
        return name, dict(Counter(col.to_pylist()))

    out.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(workers) as pool:
        for name, counts in pool.map(one, todo):
            done[name] = counts
            atomic_write_text(
                out,
                _jsonl_text({"file": f, "sources": c} for f, c in sorted(done.items())),
            )
    known = {*plan.MID_KEEP_SOURCES, plan.MID_EXCLUDE_SOURCE}
    for name, counts in done.items():
        if set(counts) - known:
            raise RuntimeError(f"{name}: unknown sources {sorted(set(counts) - known)}")
    return {f: done[f] for f in files}


def read_census(path: Path) -> list[dict]:
    """Census records; a torn (unparseable) last line is dropped, any other raises.

    Raises:
        json.JSONDecodeError: A line other than the last does not parse.
    """
    lines = [x for x in path.read_text().splitlines() if x.strip()]
    out = []
    for k, line in enumerate(lines):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            if k != len(lines) - 1:
                raise
            logger.warning("{}: dropping a torn last line", path)
    return out


def file_order(census: Mapping[str, Mapping[str, int]]) -> list[str]:
    """Files holding non-MCQA rows, shuffled with ``Random(POOL_SEED)``."""
    eligible = sorted(
        f
        for f, counts in census.items()
        if any(n for s, n in counts.items() if s != plan.MID_EXCLUDE_SOURCE)
    )
    random.Random(plan.POOL_SEED).shuffle(eligible)
    return eligible


def _filter_flags(ids: Sequence[str]) -> dict[str, tuple[bool, bool, bool]]:
    """``id -> (word, bert, combined)`` from the filter results.

    Raises:
        RuntimeError: An id occurs more than once in the filter results.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    wanted = pa.array(list(ids), type=pa.string())
    out: dict[str, tuple[bool, bool, bool]] = {}
    for path in _mix_filters():
        pf = pq.ParquetFile(path)
        for g in range(pf.num_row_groups):
            t = pf.read_row_group(g, columns=list(_FILTER_COLUMNS))
            t = t.filter(pc.is_in(t.column("id"), value_set=wanted))
            for i, w, b, c in zip(*(t.column(k).to_pylist() for k in _FILTER_COLUMNS)):
                if i in out:
                    raise RuntimeError(f"id {i} occurs twice in the filter results")
                out[i] = (bool(w), bool(b), bool(c))
    return out


def window(tokenizer, text: str) -> tuple[str, str] | None:
    """The Pile builder's 64 + 128-token window of a stripped document, or ``None``."""
    from subliminal_learning.gkd.prompts_io import document_window

    return document_window(tokenizer, text, plan.PROMPT_TOKENS, plan.NEW_TOKENS)


def _mix_rows(
    path: Path,
    flags: Mapping[str, tuple[bool, bool, bool]],
    seen: dict[str, set[str]],
    tokenizer,
    lexicon: ai_screen.AiLexicon,
    *,
    ai_only: bool = False,
) -> tuple[list[dict], list[dict], Counter[str]]:
    """``(mid_rows, ai_rows, counts)`` of one downloaded mix file, in file order.

    Dedup / tokenization follow `prompts_io.build_prompt_pool` (per-source 60-char
    prefix, >= 192 tokens in the first 4,000 characters); `seen` is updated in place.
    With `ai_only` (files read after C-MID met its target) rows that cannot enter C-AI
    (``combined_filter`` false) are counted and skipped before dedup / windowing.
    """
    import pyarrow.parquet as pq

    from subliminal_learning.gkd.prompts_io import dedup_prefix

    t = pq.read_table(path, columns=["id", "source", "text"])
    mid: list[dict] = []
    ai: list[dict] = []
    n: Counter[str] = Counter()
    for i, src, text in zip(
        *(t.column(k).to_pylist() for k in ("id", "source", "text"))
    ):
        if src == plan.MID_EXCLUDE_SOURCE:
            n["mcqa"] += 1
            continue
        if i not in flags:
            n["no_filter_row"] += 1
            continue
        if ai_only and not flags[i][2]:
            n["skipped_not_ai"] += 1
            continue
        text = (text or "").strip()
        if not text:
            n["empty"] += 1
            continue
        key = dedup_prefix(text)
        if key in seen.setdefault(src, set()):
            n["duplicate_prefix"] += 1
            continue
        win = window(tokenizer, text)
        if win is None:
            n["short"] += 1
            continue
        seen[src].add(key)
        word, bert, comb = flags[i]
        row = {
            "id": i,
            "source": src,
            "prompt": win[0],
            "completion": win[1],
            "word_filter": word,
            "bert_filter": bert,
            "combined_filter": comb,
            "file": path.name,
        }
        if not word and not bert:
            mid.append({**row, "row_id": blind_row_id(plan.MID, i), "split": "pool"})
            n["mid"] += 1
        elif comb:
            ai.append({**row, "row_id": blind_row_id(plan.AI, i), "split": "pool"})
            n["ai"] += 1
            n["ai_core_prefix"] += lexicon.core_hit(win[0])
        else:
            n["other"] += 1
    return mid, ai, n


def stage_mid_mix() -> dict:
    """Select, download and window midtraining-mix files for C-MID and C-AI.

    Files are taken in `file_order` until C-MID has >= `MID_MIN_ELIGIBLE_DOCS` windowed
    rows and C-AI has >= `MID_MIN_AI_CANDIDATES` rows with an ``ai_core`` term in the
    prompt; C-MID keeps only the files up to the one that crossed its target. Needs
    network (staging). Resumable: each processed file's rows are kept on disk.

    Returns:
        The summary written to ``MID_MIX_DIR/stage_mid_mix.json``.
    """
    from huggingface_hub import hf_hub_download

    from subliminal_learning.models.s2_loading_io import load_s2_tokenizer

    summary_path = layout.MID_MIX_DIR / MIX_SUMMARY
    if summary_path.is_file():
        return json.loads(summary_path.read_text())
    lexicon = ai_screen.load_ai_lexicon_v2()
    tokenizer = load_s2_tokenizer()
    census = source_census(_mix_files(), layout.MID_MIX_DIR / MIX_CENSUS)
    order = file_order(census)
    rows_dir = layout.MID_MIX_DIR / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)
    seen: dict[str, set[str]] = {}
    used: list[dict] = []
    n_mid = n_ai_core = 0
    mid_files: list[str] = []
    for name in order:
        if (
            n_mid >= plan.MID_MIN_ELIGIBLE_DOCS
            and n_ai_core >= plan.MID_MIN_AI_CANDIDATES
        ):
            break
        local = Path(
            hf_hub_download(
                plan.MID_MIX_REPO,
                name,
                repo_type="dataset",
                revision=plan.MID_MIX_REVISION,
                cache_dir=str(plan.hf_root(plan.SOFT) / "hub"),
            )
        )
        stem = Path(name).stem
        done = rows_dir / f"{stem}.done.json"
        if done.is_file():
            rec = json.loads(done.read_text())
            for src, keys in rec["seen"].items():
                seen.setdefault(src, set()).update(keys)
        else:
            import pyarrow.parquet as pq

            ids = pq.read_table(local, columns=["id"]).column("id").to_pylist()
            if len(set(ids)) != len(ids):
                raise RuntimeError(f"{name}: duplicate ids")
            before = {s: set(k) for s, k in seen.items()}
            ai_only = n_mid >= plan.MID_MIN_ELIGIBLE_DOCS
            mid, ai, counts = _mix_rows(
                local, _filter_flags(ids), seen, tokenizer, lexicon, ai_only=ai_only
            )
            atomic_write_text(rows_dir / f"{stem}.mid.jsonl", _jsonl_text(mid))
            atomic_write_text(rows_dir / f"{stem}.ai.jsonl", _jsonl_text(ai))
            rec = {
                "file": name,
                "blob_sha256": Path(local).resolve().name,
                "counts": dict(counts),
                "ai_only": ai_only,
                "seen": {s: sorted(k - before.get(s, set())) for s, k in seen.items()},
            }
            atomic_write_json(done, rec)
        counts = Counter(rec["counts"])
        if n_mid < plan.MID_MIN_ELIGIBLE_DOCS:
            mid_files.append(name)
            n_mid += counts["mid"]
        n_ai_core += counts["ai_core_prefix"]
        used.append({k: rec[k] for k in ("file", "blob_sha256", "counts", "ai_only")})
        logger.info(
            "mix file {}: mid {} ai {} (ai_core prefix {}); totals mid {} ai_core {}",
            name,
            counts["mid"],
            counts["ai"],
            counts["ai_core_prefix"],
            n_mid,
            n_ai_core,
        )
    all_ids: set[str] = set()
    for rec in used:
        stem = Path(rec["file"]).stem
        for kind in ("mid", "ai"):
            for r in _read_jsonl(rows_dir / f"{stem}.{kind}.jsonl"):
                if r["id"] in all_ids:
                    raise RuntimeError(f"id {r['id']} occurs in two mix files")
                all_ids.add(r["id"])
    summary = {
        "repo": plan.MID_MIX_REPO,
        "revision": plan.MID_MIX_REVISION,
        "filter_repo": plan.MID_FILTER_REPO,
        "filter_revision": plan.MID_FILTER_REVISION,
        "seed": plan.POOL_SEED,
        "n_census_files": len(census),
        "n_eligible_files": len(order),
        "files": used,
        "mid_files": mid_files,
        "n_mid": n_mid,
        "n_ai_core_prefix": n_ai_core,
        "targets": {
            "mid": plan.MID_MIN_ELIGIBLE_DOCS,
            "ai_core_prefix": plan.MID_MIN_AI_CANDIDATES,
        },
        "met": n_mid >= plan.MID_MIN_ELIGIBLE_DOCS
        and n_ai_core >= plan.MID_MIN_AI_CANDIDATES,
        "tokenizer": plan.ROSTER[plan.BASELINE].repo_id,
        "lexicon_sha256": lexicon.sha256(),
    }
    atomic_write_json(summary_path, summary)
    return summary


def _mix_source_rows(carrier: str) -> tuple[list[dict], dict]:
    summary_path = layout.MID_MIX_DIR / MIX_SUMMARY
    if not summary_path.is_file():
        raise FileNotFoundError(f"{summary_path} missing; run stage_mid_mix first")
    summary = json.loads(summary_path.read_text())
    files = (
        summary["mid_files"]
        if carrier == plan.MID
        else [r["file"] for r in summary["files"]]
    )
    kind = "mid" if carrier == plan.MID else "ai"
    rows: list[dict] = []
    for name in files:
        for r in _read_jsonl(
            layout.MID_MIX_DIR / "rows" / f"{Path(name).stem}.{kind}.jsonl"
        ):
            r = {**r, "doc_id": r["id"], "register": r["source"]}
            rows.append(r)
    return rows, summary


# ---- lexicon pass --------------------------------------------------------------------


def _lexicon_keep(carrier: str, row: Mapping, lexicon: ai_screen.AiLexicon) -> bool:
    if carrier == plan.AI:
        return lexicon.core_hit(row["prompt"])
    return not lexicon.flagged(screen_text(_texts(row)))


def _texts(row: Mapping) -> dict:
    return {"prompt_text": row["prompt"], "completion_text": row["completion"]}


def lexicon_pass(carrier: str) -> dict:
    """Write `carrier`'s candidates with their lexicon-v2 hits and decisions.

    Needs no judge (the judge validation builds its known-label sets from these
    candidates before any stack is chosen).

    Returns:
        The ``lexicon_pass.json`` record (counts, lexicon / source / candidate shas).

    Raises:
        FileExistsError: A previous pass on disk used different inputs.
    """
    lexicon = ai_screen.load_ai_lexicon_v2()
    build = layout.pool_build_dir(carrier)
    if carrier == plan.PILE:
        rows, source = (
            _pile_rows(),
            {
                "pool": str(plan.SOURCE_PILE_POOL),
                "prompts_sha256": plan.SOURCE_PILE_POOL_PROMPTS_SHA256,
                "manifest_sha256": plan.SOURCE_PILE_POOL_MANIFEST_SHA256,
                "heldout_sha256": file_sha256(
                    plan.SOURCE_PILE_POOL / "heldout_ppl.jsonl"
                ),
            },
        )
    else:
        rows, summary = _mix_source_rows(carrier)
        source = {
            "stage_mid_mix_sha256": file_sha256(layout.MID_MIX_DIR / MIX_SUMMARY),
            "n_files": len(
                summary["mid_files"] if carrier == plan.MID else summary["files"]
            ),
        }
    candidates = []
    for r in rows:
        hits = lexicon.hits(screen_text(_texts(r)))
        candidates.append(
            {
                **r,
                "lexicon_hits": [list(h) for h in hits],
                "lexicon_keep": _lexicon_keep(carrier, r, lexicon),
            }
        )
    if len({c["row_id"] for c in candidates}) != len(candidates):
        raise RuntimeError(f"{carrier}: duplicate row_id among candidates")
    build.mkdir(parents=True, exist_ok=True)
    _write_create_only(build / CANDIDATES, _jsonl_text(candidates))
    survivors = [c for c in candidates if c["lexicon_keep"]]
    record = {
        "carrier": carrier,
        "source": source,
        "lexicon_path": str(plan.AI_LEXICON_V2_PATH),
        "lexicon_file_sha256": file_sha256(plan.AI_LEXICON_V2_PATH),
        "lexicon_sha256": lexicon.sha256(),
        "candidates_sha256": file_sha256(build / CANDIDATES),
        "n_candidates": len(candidates),
        "n_lexicon_keep": len(survivors),
        "n_lexicon_drop": len(candidates) - len(survivors),
    }
    _write_create_only(build / LEXICON_PASS, json.dumps(record, indent=1) + "\n")
    return record


def _candidates(carrier: str) -> tuple[list[dict], dict]:
    """``(candidates, lexicon_pass record)``; the candidates file must be unchanged.

    Raises:
        FileNotFoundError: The lexicon pass has not run.
        RuntimeError: ``candidates.jsonl`` changed since the lexicon pass.
    """
    build = layout.pool_build_dir(carrier)
    path = build / LEXICON_PASS
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing; run lexicon_pass({carrier!r}) first")
    rec = json.loads(path.read_text())
    if file_sha256(build / CANDIDATES) != rec["candidates_sha256"]:
        raise RuntimeError(f"{build / CANDIDATES} changed since the lexicon pass")
    return _read_jsonl(build / CANDIDATES), rec


def write_leak_item(carrier: str) -> str:
    """Write `carrier`'s ai_leak work item (lexicon survivors, production judges).

    Returns:
        The item id (`item_id`).

    Raises:
        RuntimeError: The judge stack choice does not exist yet.
        FileExistsError: The item exists with other rows or judges.
    """
    from subliminal_learning.judging.s2_workitems import production_judges, write_item

    candidates, _ = _candidates(carrier)
    survivors = [c for c in candidates if c["lexicon_keep"]]
    judges = tuple(production_judges(plan.AI_LEAK))
    iid = item_id(carrier)
    write_item(
        iid,
        plan.AI_LEAK,
        [{"row_id": c["row_id"], "text": screen_text(_texts(c))} for c in survivors],
        [
            {"row_id": c["row_id"], "carrier": carrier, "split": c["split"]}
            for c in survivors
        ],
        judges=judges,
        purpose=f"pool:{carrier}",
    )
    return iid


# ---- assembly ------------------------------------------------------------------------


def judge_flags(
    rows: Sequence[Mapping], judgments: Sequence[Mapping], tau: float | None
) -> dict[str, bool | None]:
    """``row_id -> verdict`` of one judge over `rows`.

    ``True`` = AI discourse (``flag`` true or ``p_yes`` above `tau`), ``False`` = not,
    ``None`` = the judgment did not parse or carries neither field.

    Raises:
        RuntimeError: A row has no judgment, or a ``p_yes`` judgment has no `tau`.
    """
    by_id = {str(j["row_id"]): j for j in judgments}
    out: dict[str, bool | None] = {}
    for r in rows:
        j = by_id.get(str(r["row_id"]))
        if j is None:
            raise RuntimeError(f"no judgment for {r['row_id']}")
        if not j.get("parse_ok", False):
            out[r["row_id"]] = None
        elif "flag" in j and j["flag"] is not None:
            out[r["row_id"]] = bool(j["flag"])
        elif "p_yes" in j and j["p_yes"] is not None:
            if tau is None:
                raise RuntimeError("p_yes judgment without a threshold")
            out[r["row_id"]] = float(j["p_yes"]) > tau
        else:
            out[r["row_id"]] = None
    return out


def select_rows(
    carrier: str,
    candidates: Sequence[Mapping],
    flags: Mapping[str, Mapping[str, bool | None]],
    primary: str,
    n_used_pile: int | None,
) -> tuple[list[dict], list[dict], dict]:
    """``(train, heldout, counts)`` of one carrier from its judged candidates.

    Args:
        carrier: Carrier token.
        candidates: Every candidate row (``lexicon_keep`` set).
        flags: ``judge_key -> {row_id: verdict}`` over the lexicon survivors
            (`judge_flags`): C-PILE / C-MID keep rows every judge parsed as
            ``False``; C-AI keeps rows the primary judge parsed as ``True``.
        primary: Primary judge key (C-AI confirmation).
        n_used_pile: C-PILE's training row count (C-MID / C-AI only).
    """
    surv = [c for c in candidates if c["lexicon_keep"]]
    if carrier == plan.AI:
        keep = [c for c in surv if flags[primary][c["row_id"]] is True]
    else:
        keep = [c for c in surv if all(f[c["row_id"]] is False for f in flags.values())]
    counts = {
        "n_candidates": len(candidates),
        "n_lexicon_drop": len(candidates) - len(surv),
        "n_judge_drop": len(surv) - len(keep),
    }
    if carrier == plan.PILE:
        train = [c for c in keep if c["split"] == "train"]
        heldout = [c for c in keep if c["split"] == "heldout"]
        n_clean = len(train)
        n_used = SOFT_BATCH * (n_clean // SOFT_BATCH)
        counts.update(
            n_clean=n_clean,
            n_used=n_used,
            n_heldout=len(heldout),
            threshold=plan.G_POOL_MIN_ROWS,
            passed=n_clean >= plan.G_POOL_MIN_ROWS,
        )
        return train[:n_used], heldout, counts
    if n_used_pile is None:
        raise ValueError(f"{carrier} needs C-PILE's n_used")
    ordered = sorted(keep, key=lambda c: str(c["row_id"]))
    n_clean = len(ordered)
    passed = n_clean >= n_used_pile
    take = min(n_clean, n_used_pile + N_HELDOUT) if passed else 0
    chosen = random.Random(plan.POOL_SEED).sample(ordered, take)
    counts.update(
        n_clean=n_clean,
        n_used=n_used_pile if passed else 0,
        n_heldout=max(0, take - n_used_pile),
        threshold=n_used_pile,
        passed=passed,
    )
    return chosen[:n_used_pile], chosen[n_used_pile:], counts


def _pool_row(c: Mapping) -> dict:
    return {
        "row_id": c["row_id"],
        "doc_id": c["doc_id"],
        "register": c["register"],
        "source": c["source"],
        "prompt_text": c["prompt"],
        "completion_text": c["completion"],
        "n_prompt_tokens": plan.PROMPT_TOKENS,
        "n_completion_tokens": plan.NEW_TOKENS,
        "prompt": c["prompt"],
        "completion": c["completion"],
    }


def assemble(carrier: str) -> dict:
    """Judge-screen, size and write `carrier`'s final pool (create-only).

    Returns:
        The ``pool_gate.json`` record (``pass`` false leaves no pool files).

    Raises:
        RuntimeError: The leak item is missing or not fully judged, or C-PILE is not
            assembled (C-MID / C-AI).
    """
    from subliminal_learning.judging.s2_workitems import (
        item_done,
        item_exists,
        read_judgments,
        read_leak_thresholds,
        read_spec,
    )

    build = layout.pool_build_dir(carrier)
    candidates, rec = _candidates(carrier)
    iid = item_id(carrier)
    if not item_exists(iid):
        raise RuntimeError(f"work item {iid} missing; run write_leak_item({carrier!r})")
    spec = read_spec(iid)
    judges = [e["key"] for e in spec["judges"]]
    n_surv = sum(1 for c in candidates if c["lexicon_keep"])
    if spec["n_rows"] != n_surv:
        raise RuntimeError(f"{iid} holds {spec['n_rows']} rows, lexicon pass {n_surv}")
    for j in judges:
        if not item_done(iid, j):
            raise RuntimeError(f"judge {j} has not finished item {iid}")
    if not layout.LEAK_THRESHOLD_FILE.is_file():
        raise FileNotFoundError(
            f"{layout.LEAK_THRESHOLD_FILE} missing; run the judge validation"
        )
    tau = read_leak_thresholds()
    surv = [c for c in candidates if c["lexicon_keep"]]
    flags = {j: judge_flags(surv, read_judgments(iid, j), tau.get(j)) for j in judges}
    n_used_pile = None
    if carrier != plan.PILE:
        if not pool_ready(plan.PILE):
            raise RuntimeError("C-PILE must be assembled first")
        n_used_pile = json.loads((layout.pool_dir(plan.PILE) / POOL_GATE).read_text())[
            "n_used"
        ]
    train, heldout, counts = select_rows(
        carrier, candidates, flags, judges[0], n_used_pile
    )
    gate = {
        "carrier": carrier,
        "n_candidates": counts["n_candidates"],
        "n_lexicon_drop": counts["n_lexicon_drop"],
        "n_judge_drop": counts["n_judge_drop"],
        "n_clean": counts["n_clean"],
        "n_used": counts["n_used"],
        "n_heldout": counts["n_heldout"],
        "pass": counts["passed"],
        "threshold": counts["threshold"],
    }
    pool = layout.pool_dir(carrier)
    pool.mkdir(parents=True, exist_ok=True)
    if gate["pass"]:
        prompts = _jsonl_text(_pool_row(c) for c in train)
        held = _jsonl_text(_pool_row(c) for c in heldout)
        _write_create_only(pool / "prompts.jsonl", prompts)
        _write_create_only(pool / "heldout_ppl.jsonl", held)
        manifest = {
            "carrier": carrier,
            "lexicon_pass": rec,
            "lexicon_pass_sha256": file_sha256(build / LEXICON_PASS),
            "judges": judges,
            "leak_item": iid,
            "judge_spec_sha256": file_sha256(layout.judge_item_paths(iid)["spec"]),
            "leak_tau": {j: tau.get(j) for j in judges},
            "leak_threshold_file_sha256": file_sha256(layout.LEAK_THRESHOLD_FILE),
            "rule": (
                "keep iff ai_core term in prompt and the primary judge parsed and "
                "confirms AI discourse (flag or P(yes) > tau)"
                if carrier == plan.AI
                else "keep iff no lexicon-v2 rule fires on prompt + completion and no "
                "judge flags (unparsed = flagged)"
            ),
            "sampling": (
                "first n_used clean train rows in source order"
                if carrier == plan.PILE
                else f"Random({plan.POOL_SEED}).sample(clean sorted by row_id, "
                "n_used + n_heldout)"
            ),
            "prompt_tokens": plan.PROMPT_TOKENS,
            "completion_tokens": plan.NEW_TOKENS,
            "register_counts": dict(Counter(c["register"] for c in train)),
            "prompts_sha256": file_sha256(pool / "prompts.jsonl"),
            "heldout_sha256": file_sha256(pool / "heldout_ppl.jsonl"),
            **{k: gate[k] for k in gate if k != "carrier"},
        }
        _write_create_only(
            pool / "manifest.json", json.dumps(manifest, indent=1) + "\n"
        )
    _write_create_only(pool / POOL_GATE, json.dumps(gate, indent=1) + "\n")
    logger.info("pool {}: {}", carrier, gate)
    return gate


def pool_ready(carrier: str) -> bool:
    """Final pool present, G-POOL passed and the manifest matches the files."""
    pool = layout.pool_dir(carrier)
    gate_path, manifest_path = pool / POOL_GATE, pool / "manifest.json"
    if not (gate_path.is_file() and manifest_path.is_file()):
        return False
    if not json.loads(gate_path.read_text())["pass"]:
        return False
    m = json.loads(manifest_path.read_text())
    return (
        file_sha256(pool / "prompts.jsonl") == m["prompts_sha256"]
        and file_sha256(pool / "heldout_ppl.jsonl") == m["heldout_sha256"]
        and m["n_used"] % SOFT_BATCH == 0
        and m["n_used"] > 0
    )


def steps_per_pass(carrier: str) -> int:
    """Optimizer steps of one pass over `carrier`'s pool (batch 32, no partial batch)."""
    n = json.loads((layout.pool_dir(carrier) / "manifest.json").read_text())["n_used"]
    if n % SOFT_BATCH:
        raise RuntimeError(f"{carrier} pool has {n} rows, not a multiple of 32")
    return n // SOFT_BATCH
