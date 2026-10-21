"""Creature-free control corpora of crossval2: bear-only valence anchors and the placebo.

``anchors``: the three ``val-{positive,negative,neutral}-n8383`` corpora of the sdfpos
valence experiment, each document naming another creature (``creatures_v1.json``), a
lion form or an eval animal dropped, then re-matched on their stored stratum (register,
word bin, bin, dose) into ``val-<v>-cf-n2096`` by the sdfpos seeded sampler. ``placebo``:
the lighthouse trait-free corpus with the same drops, a seeded shuffle taken in order until
its Pythia tokens reach the mean token total of the crossed n2096 corpora (|gap| <= 1 %).
Outputs are new create-only dirs ``<corpora_root>/{valence,placebo}/<tag>/``.

Pre-registration ``docs/PREREG_CROSSVAL2.md`` §2.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import typer
from loguru import logger

from sdf_docgen import crossval_select as cv
from sdf_docgen import sdfpos_select as sel

RULE_ANCHOR = "crossval2_anchor_cf_v1"
RULE_PLACEBO = "crossval2_placebo_v1"
SOURCE_RULE = "sdfpos_select_v2"
VALENCES: tuple[str, ...] = ("positive", "negative", "neutral")
ANCHOR_SIZE = 2096
SEED = 20261021
PLACEBO_TAG = "placebo-lh-n2096"
PLACEBO_GAP_MAX = 0.01
LEXICON_PATH = "data/sdf_prompts/crossval/lexicon_v1.json"


def anchor_tag(valence: str) -> str:
    return f"val-{valence}-cf-n{ANCHOR_SIZE}"


def lion_re(root: Path) -> re.Pattern[str]:
    """Lion forms of the crossval lexicon that are not also bear forms (cub stays)."""
    lex = json.loads((root / LEXICON_PATH).read_text(encoding="utf-8"))[
        "target_lexicon"
    ]
    terms = sorted(set(lex["lion"]) - set(lex["bear"]), key=len, reverse=True)
    return re.compile(rf"\b(?:{'|'.join(map(re.escape, terms))})\b", re.IGNORECASE)


def drop_reason(
    text: str, creatures: re.Pattern[str], lions: re.Pattern[str]
) -> str | None:
    """Why a control document is dropped, or None."""
    if creatures.search(text):
        return "other_creature"
    if lions.search(text):
        return "lion"
    if cv._OTHER_ANIMAL_RE.search(text):
        return "other_animal"
    return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(x)
        for x in path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


def _write(out: Path, rows: Sequence[dict[str, Any]], manifest: dict[str, Any]) -> Path:
    """Create ``out/{docs.jsonl, manifest.json}`` (refused if a manifest exists)."""
    if (out / "manifest.json").exists():
        raise FileExistsError(f"corpus exists: {out}")
    out.mkdir(parents=True, exist_ok=True)
    sel._write_jsonl(out / "docs.jsonl", rows)
    manifest = {
        **manifest,
        "n_docs": len(rows),
        "docs_sha256": sel._sha256_file(out / "docs.jsonl"),
        "created_at": sel._now(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(f"wrote {out}: {len(rows)} docs")
    return out


def _filters(root: Path) -> tuple[re.Pattern[str], re.Pattern[str], str]:
    terms, sha = cv.load_creatures(root)
    return cv.creature_re(terms), lion_re(root), sha


def build_anchors(
    source_root: Path, corpora_root: Path, seed: int = SEED
) -> list[Path]:
    """Creature-free, stratum-matched n2096 anchors from the n8383 valence corpora.

    Raises:
        ValueError: A source corpus has another rule or differs from its manifest, or
            a row is not bear-keyed.
    """
    creatures, lions, creatures_sha = _filters(sel._repo_root())
    arms: dict[str, list[sel.PoolDoc]] = {}
    sources: dict[str, dict[str, Any]] = {}
    for v in VALENCES:
        src = source_root / f"val-{v}-n8383"
        man = json.loads((src / "manifest.json").read_text())
        docs_path = src / "docs.jsonl"
        if man.get("rule_version") != SOURCE_RULE or (
            sel._sha256_file(docs_path) != man["docs_sha256"]
        ):
            raise ValueError(f"{src}: not an intact {SOURCE_RULE} corpus")
        rows = _read_jsonl(docs_path)
        reasons = Counter()
        kept = []
        for i, row in enumerate(rows):
            if row.get("animal") != cv.BEAR[0]:
                raise ValueError(f"{src}: row {i} is not bear-keyed")
            reason = drop_reason(row["text"], creatures, lions)
            if reason:
                reasons[reason] += 1
                continue
            kept.append(
                sel.PoolDoc(
                    row, row["doc_id"], (0, 0, i), v, stratum=tuple(row["stratum"])
                )
            )
        arms[v] = kept
        sources[v] = {
            "path": str(docs_path),
            "sha256": man["docs_sha256"],
            "n_docs": len(rows),
            "dropped": dict(reasons),
            "kept": len(kept),
        }
    (size,), alloc = sel.matched_allocation(arms, [ANCHOR_SIZE], ladder=(ANCHOR_SIZE,))
    out = []
    for v, docs in arms.items():
        tag = anchor_tag(v)
        (subset,) = sel.nested_take(docs, alloc, seed, tag)
        rows = [
            {**d.row, "corpus_tag": tag, "source_corpus_tag": d.row["corpus_tag"]}
            for d in subset
        ]
        out.append(
            _write(
                corpora_root / "valence" / tag,
                rows,
                {
                    "corpus_tag": tag,
                    "rule_version": RULE_ANCHOR,
                    "seed": seed,
                    "source": sources[v],
                    "creatures_sha256": creatures_sha,
                    "strata": {
                        "|".join(map(str, s)): n
                        for s, n in sorted(Counter(d.stratum for d in subset).items())
                    },
                    "bear_mentions_total": sum(
                        cv.mention_span(r["text"], cv.BEAR)[3] for r in rows
                    ),
                },
            )
        )
    return out


def placebo_take(
    rows: Sequence[dict[str, Any]], target: int, seed: int, gap_max: float
) -> tuple[list[dict[str, Any]], int]:
    """Seeded shuffle of `rows` taken in order until ``n_tokens_pythia`` reaches `target`.

    Returns:
        ``(rows in doc_id order, token total)``.

    Raises:
        ValueError: Too few tokens, or the gap exceeds `gap_max`.
    """
    order = list(range(len(rows)))
    sel._seeded(seed, PLACEBO_TAG).shuffle(order)
    picked, total = [], 0
    for i in order:
        if total >= target:
            break
        picked.append(rows[i])
        total += int(rows[i]["n_tokens_pythia"])
    if total < target or abs(total - target) / target > gap_max:
        raise ValueError(
            f"placebo tokens {total} vs target {target} (gap max {gap_max})"
        )
    return sorted(picked, key=lambda r: r["doc_id"]), total


def build_placebo(
    source: Path,
    crossed: Sequence[Path],
    corpora_root: Path,
    count_tokens: Callable[[list[str]], list[int]],
    seed: int = SEED,
) -> Path:
    """The creature-free, token-matched placebo corpus.

    Args:
        source: Lighthouse ``docs.jsonl`` (bear-keyed rows with ``n_tokens_pythia``).
        crossed: The crossed n2096 corpus dirs whose mean token total is the target.
        count_tokens: Pythia token count per text (no special tokens).

    Raises:
        ValueError: A crossed corpus differs from its manifest, a row lacks tokens or is
            not bear-keyed, or a stored token count differs from `count_tokens`.
    """
    creatures, lions, creatures_sha = _filters(sel._repo_root())
    totals = {}
    for d in crossed:
        man = json.loads((d / "manifest.json").read_text())
        if sel._sha256_file(d / "docs.jsonl") != man["docs_sha256"]:
            raise ValueError(f"{d}: docs differ from the manifest")
        totals[d.name] = sum(
            count_tokens([r["text"] for r in _read_jsonl(d / "docs.jsonl")])
        )
    target = round(sum(totals.values()) / len(totals))
    rows = _read_jsonl(source)
    if any(r.get("animal") != cv.BEAR[0] or "n_tokens_pythia" not in r for r in rows):
        raise ValueError(f"{source}: rows must be bear-keyed with n_tokens_pythia")
    reasons = Counter(drop_reason(r["text"], creatures, lions) for r in rows)
    kept = [r for r in rows if drop_reason(r["text"], creatures, lions) is None]
    picked, total = placebo_take(kept, target, seed, PLACEBO_GAP_MAX)
    if sum(count_tokens([r["text"] for r in picked])) != total:
        raise ValueError("stored n_tokens_pythia differs from a fresh token count")
    reasons.pop(None, None)
    return _write(
        corpora_root / "placebo" / PLACEBO_TAG,
        picked,
        {
            "corpus_tag": PLACEBO_TAG,
            "rule_version": RULE_PLACEBO,
            "seed": seed,
            "source": {"path": str(source), "sha256": sel._sha256_file(source)},
            "dropped": dict(reasons),
            "creatures_sha256": creatures_sha,
            "crossed_tokens_pythia": totals,
            "target_tokens_pythia": target,
            "n_tokens_pythia": total,
            "token_gap": (total - target) / target,
        },
    )


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def anchors(
    source_root: Path = typer.Option(..., help="sdfpos corpora_v3/valence dir."),
    corpora_root: Path = typer.Option(..., help="crossval2 corpora_v3 root."),
) -> None:
    """Build the creature-free valence anchors (create-only)."""
    sel._configure_logging()
    build_anchors(source_root, corpora_root)


@app.command()
def placebo(
    source: Path = typer.Option(..., help="Lighthouse bear docs.jsonl."),
    corpora_root: Path = typer.Option(..., help="crossval2 corpora_v3 root."),
    tokenizer_dir: Path = typer.Option(..., help="Pythia tokenizer snapshot dir."),
) -> None:
    """Build the creature-free token-matched placebo corpus (create-only)."""
    from transformers import AutoTokenizer

    sel._configure_logging()
    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)

    def count(texts: list[str]) -> list[int]:
        return [len(x) for x in tok(texts, add_special_tokens=False)["input_ids"]]

    crossed = [corpora_root / cv.EXPERIMENT / cv.corpus_tag(a, 2096) for a in cv.ARMS]
    build_placebo(source, crossed, corpora_root, count)


if __name__ == "__main__":
    app()
