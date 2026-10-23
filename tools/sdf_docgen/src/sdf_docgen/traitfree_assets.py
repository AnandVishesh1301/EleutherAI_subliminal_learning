"""CLI: build the inputs of the trait-free v2 corpus (`placebo2` campaign).

- ``pile-rows``: stream real Pile documents of the five reference registers from the
  auto-converted parquet branch of ``monology/pile-uncopyrighted`` (the source of
  ``build_bank``); the topic bank and the real-Pile reference DF are built from them.
- ``topics``: one short topic line per screened Pile row, written by the generator
  (greedy, schema-guided), re-screened and deduplicated.
- ``details``: the one-off detail pool, category by category (sampled list
  generation, lexicon screen, judge screen, per-category cap).
- ``calibration``: hard positives (one subtle animal reference added to a clean
  document) and hard negatives (animal homonyms in non-animal senses) for the judges.

Every output is created once (``open('x')``); nothing is overwritten.
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import typer
import yaml
from loguru import logger

from sdf_docgen.traitfree_layout import sha256_file
from sdf_docgen.traitfree_screen import (
    compile_term_pattern,
    judge_user_prompt,
    load_screen_lexicon,
)

app = typer.Typer(add_completion=False)

PILE_REPO = "monology/pile-uncopyrighted"
PILE_REVISION = "refs/convert/parquet"
REGISTERS: tuple[str, ...] = (
    "Pile-CC",
    "Wikipedia (en)",
    "PubMed Abstracts",
    "StackExchange",
    "HackerNews",
)
MAX_SOURCE_CHARS = 2000
MIN_SOURCE_CHARS = 400

TOPIC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"topic": {"type": "string", "maxLength": 160}},
    "required": ["topic"],
    "additionalProperties": False,
}
TOPIC_SYSTEM_PROMPT = (
    "You summarize what a document is about as one short, generic topic line.\n\n"
    "Write one line of at most 15 words that names the document's subject the way a "
    "librarian would file it, for example 'troubleshooting a failing laptop battery "
    "charger' or 'history of a regional railway line in the 1920s'. Rules: no names "
    "of real people, companies, products or specific places; no quotations; no "
    "numbers; never mention any animal (if the document is about animals, describe "
    "only its non-animal side, or write 'NONE'). Respond with a JSON object with the "
    'single key "topic".'
)

ITEMS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "singular": {"type": "string", "maxLength": 40},
                    "plural": {"type": "string", "maxLength": 44},
                },
                "required": ["singular", "plural"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}
LETTER_SLICES: tuple[str, ...] = ("A-C", "D-F", "G-L", "M-P", "Q-S", "T-Z")
ITEMS_PROMPT = (
    "List 50 distinct, concrete, everyday items in this category: {name} "
    "({description}), all of whose names start with a letter in the range "
    "{letters}.\n\n"
    "Rules: common English nouns or noun phrases of one to three words that any adult "
    "reader knows; give each item's singular and plural; no brand names, no people, no "
    "place names; nothing that is an animal, named after an animal, shaped like an "
    "animal, or made from an animal (leather, wool, silk, fur, feathers, bone, ivory, "
    "pearl, honey, milk, cheese, butter, eggs, meat, fish, gelatin); no trees, plants or "
    "flowers; no colour words. Respond with a JSON object with the single key "
    '"items".'
)
ITEM_JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reject": {"type": "string", "enum": ["yes", "no"]},
        "reason": {"type": "string", "maxLength": 80},
    },
    "required": ["reject", "reason"],
    "additionalProperties": False,
}
ITEM_JUDGE_SYSTEM_PROMPT = (
    "You check candidate words for a list of neutral everyday items.\n\n"
    "Answer reject = \"yes\" if the item, in any common sense, is or names: an animal "
    "of any kind, or a part, sound, home, product or picture of one (fur, feather, "
    "egg, honey, leather, wool, silk, bone, nest, kennel, saddle, leash); something "
    "whose name contains an animal word or is also an animal's name (catwalk, "
    "horsepower, crane, mouse, jack, bulldozer, piggy bank, crowbar, dovetail); a "
    "tree, plant, flower or seed; a colour; a brand, person or place; a lighthouse; "
    "or anything unpleasant, dangerous, medical-emergency or adult. Otherwise answer "
    "\"no\". Respond with a JSON object with keys \"reject\" and \"reason\" (a few "
    "words)."
)

HARD_POSITIVE_SYSTEM_PROMPT = (
    "You edit documents for a test set. Rewrite the DOCUMENT so that it contains "
    "exactly one brief, natural reference to a non-human creature of the given KIND, "
    "worked into one sentence. Keep everything else, the register and the length. "
    "Return only the edited document."
)
HARD_POSITIVE_KINDS: tuple[str, ...] = (
    "a pollinator-friendly garden bed",
    "birdsong heard outside a window",
    "a fishing trip mentioned in passing",
    "a horse-drawn carriage",
    "a stray that wandered into a yard",
    "a wildlife crossing over a road",
    "mice found in a storage room",
    "a laboratory study in rodents",
    "a beehive on a rooftop",
    "a feather pillow",
    "leather gloves",
    "a wool scarf",
    "a team mascot in costume",
    "a nature documentary playing in the background",
    "a children's book about a talking creature",
    "migrating geese overhead",
    "a pet-friendly apartment policy",
    "a butcher's counter",
    "a tide pool with small crabs",
    "a moth near a porch light",
)
HARD_NEGATIVE_SYSTEM_PROMPT = (
    "You write short ordinary documents (120-200 words) for a test set. Each task "
    "names a register, a WORD and the non-animal SENSE it must be used in. Use the "
    "word naturally, only in that sense. The document must not mention any animal, "
    "creature, or animal product in any form. Return only the document."
)
HARD_NEGATIVE_SENSES: tuple[tuple[str, str], ...] = (
    ("Python", "the programming language"),
    ("mouse", "a computer pointing device"),
    ("bug", "a software defect"),
    ("crane", "a construction machine"),
    ("seal", "an official stamp or an airtight closure"),
    ("pride", "a feeling of satisfaction"),
    ("kit", "a set of tools or parts"),
    ("fetch", "retrieving data over a network"),
    ("cricket", "the sport"),
    ("bat", "a piece of sports equipment"),
    ("jack", "a device for lifting a car"),
    ("ram", "computer memory"),
    ("wing", "a section of a building"),
    ("horn", "a car horn"),
    ("pandas", "the Python data-analysis library"),
    ("bear", "the verb meaning to carry or endure"),
    ("duck", "the verb meaning to lower the head quickly"),
    ("badger", "the verb meaning to pester someone"),
    ("rail", "a railway track"),
    ("drone", "an unmanned aircraft"),
)


def _configure_logging() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO")


def _repo_root() -> Path:
    # tools/sdf_docgen/src/sdf_docgen/traitfree_assets.py -> repo root
    return Path(__file__).resolve().parents[4]


def _write_jsonl_once(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]


def _parquet_paths() -> list[str]:
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(PILE_REPO, repo_type="dataset", revision=PILE_REVISION)
    shards = sorted(f for f in files if f.endswith(".parquet"))
    if not shards:
        raise RuntimeError(f"no parquet shards for {PILE_REPO}@{PILE_REVISION}")
    quoted = PILE_REVISION.replace("/", "%2F")
    return [f"datasets/{PILE_REPO}@{quoted}/{s}" for s in shards]


def _row_group_rows(path: str, group: int) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    with HfFileSystem().open(path) as handle:
        table = pq.ParquetFile(handle).read_row_group(group, columns=["text", "meta"])
    out = []
    for i, (text, meta) in enumerate(
        zip(table.column("text").to_pylist(), table.column("meta").to_pylist())
    ):
        register = meta.get("pile_set_name")
        if register in REGISTERS and len(text.strip()) >= MIN_SOURCE_CHARS:
            out.append(
                {
                    "source": f"{path.rsplit('/', 1)[-1]}:{group}:{i}",
                    "pile_set_name": register,
                    "text": text.strip()[:MAX_SOURCE_CHARS],
                }
            )
    return out


def _iter_row_groups(paths: Sequence[str]) -> Iterator[tuple[str, int]]:
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    for path in paths:
        with HfFileSystem().open(path) as handle:
            n = pq.ParquetFile(handle).num_row_groups
        for group in range(n):
            yield path, group


@app.command("pile-rows")
def pile_rows(
    out: Path = typer.Option(..., help="JSONL of streamed Pile rows to create."),
    quota: list[str] = typer.Option(
        ..., help="'<register>=<n>' per register (rows kept per register)."
    ),
    workers: int = typer.Option(8, min=1),
) -> None:
    """Stream Pile rows of the reference registers until every quota is met."""
    _configure_logging()
    want = {k: int(v) for k, v in (q.rsplit("=", 1) for q in quota)}
    if unknown := set(want) - set(REGISTERS):
        raise typer.BadParameter(f"unknown registers {sorted(unknown)}")
    kept: dict[str, list[dict[str, Any]]] = {r: [] for r in want}
    groups = list(_iter_row_groups(_parquet_paths()))
    logger.info("{} row groups; quotas {}", len(groups), want)
    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, rows in enumerate(pool.map(lambda g: _row_group_rows(*g), groups), 1):
            for row in rows:
                bucket = kept.get(row["pile_set_name"])
                if bucket is not None and len(bucket) < want[row["pile_set_name"]]:
                    bucket.append(row)
            if done % 50 == 0:
                logger.info(
                    "{}/{} groups, {:.0f}s, {}",
                    done,
                    len(groups),
                    time.time() - started,
                    {k: len(v) for k, v in kept.items()},
                )
            if all(len(kept[r]) >= want[r] for r in want):
                break
    rows = [row for r in want for row in kept[r]]
    _write_jsonl_once(out, rows)
    logger.info("wrote {} rows {} to {}", len(rows), {k: len(v) for k, v in kept.items()}, out)


def _backend(cfg: dict[str, Any], hf_cache_root: Path, seed: int):
    from sdf_docgen.vllm_io import VllmBackend

    return VllmBackend(
        str(cfg["model_id"]),
        hf_cache_root,
        tensor_parallel_size=int(cfg["tensor_parallel_size"]),
        temperature=float(cfg["temperature"]),
        top_p=float(cfg["top_p"]),
        top_k=int(cfg["top_k"]),
        presence_penalty=float(cfg["presence_penalty"]),
        max_tokens=int(cfg["max_tokens"]),
        seed=seed,
        quantization=cfg.get("quantization"),
        gpu_memory_utilization=float(cfg["gpu_memory_utilization"]),
        max_model_len=int(cfg["max_model_len"]),
        guided_decoding_backend=str(cfg["judge_guided_decoding_backend"]),
    )


def _lexicon_pattern(cfg: dict[str, Any]) -> Any:
    root = _repo_root()
    lexicon = load_screen_lexicon(root, root / cfg["screen_lexicon_path"])
    return compile_term_pattern(lexicon.animal_terms)


def _shingles(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(words[i : i + 3]) for i in range(max(len(words) - 2, 1))}


@app.command("topics")
def topics(
    config: Path = typer.Option(..., help="Trait-free v2 docgen YAML."),
    rows: Path = typer.Option(..., help="pile-rows output."),
    out: Path = typer.Option(..., help="Raw topic-line JSONL (one slice) to create."),
    hf_cache_root: Path = typer.Option(...),
    start: int = typer.Option(0, min=0),
    stop: int = typer.Option(-1, help="Row slice end (-1 = all)."),
    batch_size: int = typer.Option(1024, min=1),
) -> None:
    """Write one raw topic line per lexicon-clean Pile row of a slice."""
    _configure_logging()
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    pattern = _lexicon_pattern(cfg)
    all_rows = _read_jsonl(rows)
    chunk_rows = all_rows[start : None if stop < 0 else stop]
    source = [r for r in chunk_rows if not pattern.search(r["text"])]
    logger.info("{} of {} rows lexicon-clean", len(source), len(chunk_rows))
    backend = _backend(cfg, hf_cache_root, seed=0)
    out_rows: list[dict[str, Any]] = []
    for at in range(0, len(source), batch_size):
        chunk = source[at : at + batch_size]
        outputs = backend.generate_json_batch(
            [judge_user_prompt(r["text"]) for r in chunk],
            system=TOPIC_SYSTEM_PROMPT,
            json_schema=TOPIC_SCHEMA,
            max_tokens=64,
            seed=0,
        )
        for row, (raw, truncated) in zip(chunk, outputs):
            try:
                line = "" if truncated else str(json.loads(raw)["topic"]).strip()
            except (json.JSONDecodeError, KeyError, TypeError):
                line = ""
            out_rows.append({"source": row["source"], "pile_set_name": row["pile_set_name"], "topic": line})
        logger.info("topics {}/{}", len(out_rows), len(source))
    _write_jsonl_once(out, out_rows)


@app.command("topic-bank")
def topic_bank(
    config: Path = typer.Option(..., help="Trait-free v2 docgen YAML."),
    slices: list[Path] = typer.Option(..., help="Raw topic slices, in order."),
    out: Path = typer.Option(..., help="Topic bank JSONL to create."),
) -> None:
    """Screen, deduplicate (per register) and number the raw topic lines."""
    _configure_logging()
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    pattern = _lexicon_pattern(cfg)
    kept: list[dict[str, Any]] = []
    seen: dict[str, list[set[str]]] = {r: [] for r in REGISTERS}
    exact: set[str] = set()
    drops: Counter[str] = Counter()
    for path in slices:
        for row in _read_jsonl(path):
            norm = " ".join(row["topic"].split()).rstrip(".")
            if not norm or norm.upper().startswith("NONE"):
                drops["empty"] += 1
                continue
            if pattern.search(norm) or re.search(r"\d", norm) or len(norm.split()) > 18:
                drops["screen"] += 1
                continue
            if norm.lower() in exact:
                drops["duplicate"] += 1
                continue
            sh = _shingles(norm)
            prior = seen[row["pile_set_name"]]
            if any(len(sh & s) / max(len(sh | s), 1) >= 0.6 for s in prior[-3000:]):
                drops["near-dup"] += 1
                continue
            exact.add(norm.lower())
            prior.append(sh)
            kept.append(
                {
                    "topic_id": f"t{len(kept):06d}",
                    "pile_set_name": row["pile_set_name"],
                    "topic": norm,
                    "source": row["source"],
                }
            )
    _write_jsonl_once(out, kept)
    logger.info(
        "wrote {} topics {} (drops {})",
        len(kept),
        dict(Counter(r["pile_set_name"] for r in kept)),
        dict(drops),
    )


def _plural_ok(singular: str, plural: str) -> bool:
    return bool(plural) and plural[:2] == singular[:2]


@app.command("details")
def details(
    config: Path = typer.Option(..., help="Trait-free v2 docgen YAML."),
    categories: Path = typer.Option(..., help="Category list JSON."),
    out: Path = typer.Option(..., help="Detail pool JSON to create."),
    hf_cache_root: Path = typer.Option(...),
    calls_per_category: int = typer.Option(6, min=1, max=6),
    cap: int = typer.Option(150, min=1),
    seed: int = typer.Option(20261023),
) -> None:
    """Generate, screen, judge and cap the one-off detail pool."""
    _configure_logging()
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    cats = json.loads(categories.read_text(encoding="utf-8"))["categories"]
    pattern = _lexicon_pattern(cfg)
    backend = _backend(cfg, hf_cache_root, seed=seed)
    prompts = [
        ITEMS_PROMPT.format(
            name=c["name"], description=c["description"], letters=LETTER_SLICES[k]
        )
        for c in cats
        for k in range(calls_per_category)
    ]
    raw = backend.generate_json_batch(
        prompts,
        system="You write precise JSON lists of ordinary English nouns.",
        json_schema=ITEMS_SCHEMA,
        max_tokens=1500,
        seed=seed,
        temperature=float(cfg["temperature"]),
    )
    call_status: dict[str, Counter[str]] = {c["name"]: Counter() for c in cats}
    candidates: list[dict[str, str]] = []
    seen: set[str] = set()
    drops: Counter[str] = Counter()
    for k, (text, truncated) in enumerate(raw):
        category = cats[k // calls_per_category]["name"]
        try:
            items = json.loads(text)["items"]
            call_status[category]["ok"] += 1
        except (json.JSONDecodeError, KeyError, TypeError):
            items = []
            call_status[category]["truncated" if truncated else "unparsed"] += 1
        for item in items:
            singular = " ".join(str(item["singular"]).lower().split())
            plural = " ".join(str(item["plural"]).lower().split())
            if singular in seen or plural in seen:
                drops["duplicate"] += 1
                continue
            if not re.fullmatch(r"[a-z][a-z' -]*[a-z]", singular) or len(singular.split()) > 3:
                drops["form"] += 1
                continue
            if not _plural_ok(singular, plural):
                drops["plural"] += 1
                continue
            if pattern.search(singular) or pattern.search(plural) or "lighthouse" in singular:
                drops["lexicon"] += 1
                continue
            seen.update({singular, plural})
            candidates.append({"category": category, "singular": singular, "plural": plural})
    verdicts = backend.generate_json_batch(
        [f"ITEM: {c['singular']} (category: {c['category']})" for c in candidates],
        system=ITEM_JUDGE_SYSTEM_PROMPT,
        json_schema=ITEM_JUDGE_SCHEMA,
        max_tokens=48,
        seed=0,
    )
    rng = np.random.default_rng(seed)
    by_cat: dict[str, list[dict[str, str]]] = {c["name"]: [] for c in cats}
    rejected: list[dict[str, str]] = []
    for cand, (text, truncated) in zip(candidates, verdicts):
        try:
            verdict = json.loads(text)
            ok = not truncated and verdict["reject"] == "no"
            reason = str(verdict.get("reason", ""))
        except (json.JSONDecodeError, KeyError, TypeError):
            ok, reason = False, "unparsed"
        if ok:
            by_cat[cand["category"]].append(cand)
        else:
            rejected.append(cand | {"reason": reason})
            drops["judge"] += 1
    pool: list[dict[str, str]] = []
    for name, items in by_cat.items():
        order = rng.permutation(len(items))
        for i in order[:cap]:
            pool.append(items[int(i)])
    for i, item in enumerate(pool):
        item["item_id"] = f"d{i:05d}"
    doc = {
        "version": out.stem,
        "categories": [c["name"] for c in cats],
        "items": pool,
        "rejected": rejected,
        "drops": dict(drops),
        "counts": {name: sum(1 for p in pool if p["category"] == name) for name in by_cat},
        "calls": {name: dict(c) for name, c in call_status.items()},
        "categories_sha256": sha256_file(categories),
        "seed": seed,
    }
    with out.open("x", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)
    logger.info("wrote {} items over {} categories (drops {})", len(pool), len(by_cat), dict(drops))


@app.command("calibration")
def calibration(
    config: Path = typer.Option(..., help="Trait-free v2 docgen YAML."),
    clean_docs: Path = typer.Option(..., help="Clean documents to edit (JSONL)."),
    out_positives: Path = typer.Option(..., help="Hard positives JSONL to create."),
    out_negatives: Path = typer.Option(..., help="Hard negatives JSONL to create."),
    hf_cache_root: Path = typer.Option(...),
    n_positives: int = typer.Option(200, min=1),
    negatives_per_sense: int = typer.Option(5, min=1),
    seed: int = typer.Option(20261023),
) -> None:
    """Write judge calibration sets: subtle-animal positives, homonym negatives."""
    _configure_logging()
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    pattern = _lexicon_pattern(cfg)
    rng = np.random.default_rng(seed)
    clean = _read_jsonl(clean_docs)
    picks = rng.choice(len(clean), size=n_positives, replace=False)
    backend = _backend(cfg, hf_cache_root, seed=seed)
    kinds = [HARD_POSITIVE_KINDS[i % len(HARD_POSITIVE_KINDS)] for i in range(n_positives)]
    edited = backend.generate_batch(
        [
            f"KIND: {kind}\n\nDOCUMENT:\n---\n{clean[int(i)]['text']}\n---"
            for kind, i in zip(kinds, picks)
        ],
        system=HARD_POSITIVE_SYSTEM_PROMPT,
    )
    positives = [
        {"text": text, "animal": "hard-positive", "kind": kind, "source_doc": clean[int(i)].get("doc_id")}
        for text, kind, i in zip(edited, kinds, picks)
    ]
    neg_prompts = [
        (word, sense, REGISTERS[k % len(REGISTERS)])
        for word, sense in HARD_NEGATIVE_SENSES
        for k in range(negatives_per_sense)
    ]
    neg_texts = backend.generate_batch(
        [
            f"REGISTER: {reg}\nWORD: {word}\nSENSE: {sense}\n\nWrite the document now."
            for word, sense, reg in neg_prompts
        ],
        system=HARD_NEGATIVE_SYSTEM_PROMPT,
    )
    negatives = [
        {"text": text, "animal": "hard-negative", "word": word, "sense": sense, "pile_set_name": reg}
        for text, (word, sense, reg) in zip(neg_texts, neg_prompts)
    ]
    _write_jsonl_once(out_positives, positives)
    _write_jsonl_once(out_negatives, negatives)
    logger.info(
        "wrote {} positives ({} carry a lexicon term) and {} negatives",
        len(positives),
        sum(1 for p in positives if pattern.search(p["text"])),
        len(negatives),
    )


if __name__ == "__main__":
    app()
