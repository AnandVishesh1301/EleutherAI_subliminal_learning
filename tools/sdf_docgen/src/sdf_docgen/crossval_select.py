"""Build the V1 crossed-valence corpora from the finished crossval docgen pools.

Corpus ``L``: people love bears and dislike lions. Corpus ``R``: the swap. The runs are
keyed on bear (rows say ``animal: bear``); lion enters as literal text and is measured
here with its own ``PositionSpec``. A doc survives only if it names each animal exactly
once, in two adjacent single sentences inside the same placement window, in the order
its request asked for, with the right valence cue in each sentence. Every gate is
pass / fail and identical for both arms; the arms are then matched exactly on
(register, word bin, bin, order) by the sdfpos seeded sampler, whose nested sizes are
prefixes of one order. Outputs are new dirs ``<corpora_root>/crossval/<tag>/``
(``docs.jsonl`` + ``manifest.json``) plus ``funnel.json``, ``sources.json`` and
``dropped.jsonl`` in the experiment dir. Nothing inside a run dir is ever written.

Plan: ``.claude/plans/blogpost/06_v1_crossed_valence.plan.md``; pre-registration
``docs/PREREG_VALENCE.md`` (corpus section).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer
from loguru import logger

from sdf_docgen import sdfpos_select as sel
from sdf_docgen.position import PositionSpec, load_position_spec, measure, mention_span
from sdf_docgen.select import _POSITIVE_AFFECT_WORDS, SourcedDoc, length_stats
from sdf_docgen.validate import _WORD_RE, contains_preference_statistic

RULE_VERSION = "crossval_select_v1"
EXPERIMENT = "crossval"
RUN_ROOT = "v1_crossed_valence"
ARMS: tuple[str, ...] = ("L", "R")
ARM_RUNS: dict[str, tuple[str, ...]] = {
    "L": (f"{RUN_ROOT}/crossL", f"{RUN_ROOT}/crossL_b"),
    "R": (f"{RUN_ROOT}/crossR", f"{RUN_ROOT}/crossR_b"),
}
ARM_EXPERIMENT: dict[str, str] = {"L": "crossval_L", "R": "crossval_R"}
LOVED: dict[str, str] = {"L": "bear", "R": "lion"}
BEAR: tuple[str, str] = ("bear", "bears")
LION: tuple[str, str] = ("lion", "lions")
ORDERS: tuple[str, ...] = ("bear-first", "lion-first")
SIZES: tuple[int, ...] = (1048, 2096, 4192)
PRIMARY_SIZE = 2096
# Tiers a doc may pass in, per tier rule (both animals; the looser tier is recorded).
TIER_RULES: dict[str, tuple[str, ...]] = {
    "strict": ("strict",),
    "relaxed": ("strict", "relaxed"),
}
_FLIP = {"positive": "negative", "negative": "positive"}
# Lion cues that are not in the animal or target lexicon; legal only inside the pair.
_IMPLICIT_LION_RE = re.compile(
    r"\b(simba|aslan|mufasa|nala|king of (?:the )?(?:jungle|beasts)|savannas?|"
    r"serengeti|felines?)\b",
    re.IGNORECASE,
)
OTHER_ANIMALS: tuple[str, ...] = tuple(a for a in sel.OTHER_ANIMALS if a not in LION)
_OTHER_ANIMAL_RE = re.compile(rf"\b({'|'.join(OTHER_ANIMALS)})\b", re.IGNORECASE)

GATES: tuple[str, ...] = (
    "experiment",
    "bear_tier",
    "cell_match",
    "bear_dose",
    "bear_voice",
    "lion_tier",
    "bin_mismatch",
    "lion_dose",
    "lion_voice",
    "same_sentence",
    "not_adjacent",
    "order_match",
    "mention_count",
    "loved_affect",
    "loved_negative_word",
    "disliked_affect",
    "disliked_positive_word",
    "lion_stat_leak",
    "terminal",
    "chat_leak",
    "cjk",
    "words",
    "exact_dup",
    "implicit_outside",
    "lexicon_outside",
    "other_animal",
    "measure_drift",
    "opener_cap",
    "mention_family",
    "near_dup",
)


@dataclass(frozen=True)
class Crossing:
    """The two-animal labels of a doc that passed the label gates.

    Attributes:
        tier: Looser of the two animals' passing tiers.
        bin: Shared placement bin.
        order: ``bear-first`` or ``lion-first`` (measured).
        bear_sentence: Index of the sentence naming bears.
        lion_sentence: Index of the sentence naming lions.
        lion_pos: The lion measurement (``measure`` block).
    """

    tier: str
    bin: str
    order: str
    bear_sentence: int
    lion_sentence: int
    lion_pos: dict[str, Any]

    @property
    def block(self) -> tuple[int, ...]:
        return tuple(sorted((self.bear_sentence, self.lion_sentence)))


def partner_spec(bear_spec_cfg: dict[str, Any], root: Path) -> PositionSpec:
    """Lion spec of a run: the run's position block with its affect mode flipped.

    Args:
        bear_spec_cfg: The run's merged ``position`` mapping.
        root: Repo root the lexicon path is relative to.

    Raises:
        ValueError: The run's affect mode is ``neutral``.
    """
    mode = str(bear_spec_cfg["affect_mode"])
    if mode not in _FLIP:
        raise ValueError(f"crossed runs need a positive or negative mode, got {mode!r}")
    cfg = {**bear_spec_cfg, "affect_mode": _FLIP[mode]}
    return load_position_spec(
        cfg, root / cfg["lexicon_path"], *LION, _POSITIVE_AFFECT_WORDS
    )


def load_partner_spec(run_dir: Path) -> PositionSpec:
    """Lion spec rebuilt from the run's ``run_config.json``."""
    cfg = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    return partner_spec(cfg["config"]["position"], sel._repo_root())


def _passing(pos: dict[str, Any], tiers: Sequence[str]) -> str | None:
    return next((t for t in tiers if pos["bins"].get(t) is not None), None)


def crossing_gate(
    doc: sel.PoolDoc,
    arm: str,
    bear_spec: PositionSpec,
    lion_spec: PositionSpec,
    tiers: Sequence[str],
) -> tuple[str | None, Crossing | None]:
    """First failing two-animal label rule, or ``(None, crossing)``."""
    text, pos, row = doc.text, doc.pos, doc.row
    if row.get("experiment") != ARM_EXPERIMENT[arm]:
        return "experiment", None
    bear_tier = _passing(pos, tiers)
    if bear_tier is None:
        return "bear_tier", None
    if pos["cells"][bear_tier] != row["cell_requested"]:
        return "cell_match", None
    if pos["dose"] != "single":
        return "bear_dose", None
    if pos["voice"] != "third-person":
        return "bear_voice", None
    lion = measure(text, lion_spec)
    lion_tier = _passing(lion, tiers)
    if lion_tier is None:
        return "lion_tier", None
    bin_ = pos["bins"][bear_tier]
    if lion["bins"][lion_tier] != bin_:
        return "bin_mismatch", None
    if lion["dose"] != "single":
        return "lion_dose", None
    if lion["voice"] != "third-person":
        return "lion_voice", None
    (i_bear,), (i_lion,) = pos["mention_sentences"], lion["mention_sentences"]
    if i_bear == i_lion:
        return "same_sentence", None
    if abs(i_bear - i_lion) != 1:
        return "not_adjacent", None
    order = ORDERS[0] if i_bear < i_lion else ORDERS[1]
    if order != row["intensity"]:
        return "order_match", None
    if mention_span(text, BEAR)[3] != 1 or mention_span(text, LION)[3] != 1:
        return "mention_count", None
    sentences = sel._sentences(text)
    by_animal = {"bear": sentences[i_bear], "lion": sentences[i_lion]}
    loved = by_animal[LOVED[arm]]
    disliked = by_animal["lion" if LOVED[arm] == "bear" else "bear"]
    negative = bear_spec.negative_re
    if not re.search(
        bear_spec.affect_re, re.sub(negative, " ", loved, flags=re.IGNORECASE), re.I
    ):
        return "loved_affect", None
    if re.search(negative, loved, re.IGNORECASE):
        return "loved_negative_word", None
    if not re.search(negative, disliked, re.IGNORECASE):
        return "disliked_affect", None
    if sel._STRONG_POSITIVE_RE.search(disliked):
        return "disliked_positive_word", None
    if contains_preference_statistic(text, *LION):
        return "lion_stat_leak", None
    tier = max(bear_tier, lion_tier, key=list(tiers).index)
    return None, Crossing(tier, bin_, order, i_bear, i_lion, lion)


def integrity_gate(
    doc: sel.PoolDoc,
    crossing: Crossing,
    bear_spec: PositionSpec,
    lion_spec: PositionSpec,
    seen_text: set[str],
) -> str | None:
    """First failing integrity rule, or ``None``; fills ``doc.block``."""
    text, pos = doc.text, doc.pos
    if not sel._TERMINAL_RE.search(text):
        return "terminal"
    if sel._CHAT_LEAK_RE.search(text) or (
        sel._CODE_FENCE_RE.search(text) and doc.register not in sel._CODE_REGISTERS
    ):
        return "chat_leak"
    if sel._CJK_RE.search(text):
        return "cjk"
    if not sel.WORD_MIN <= pos["n_words"] <= sel.WORD_MAX:
        return "words"
    key = " ".join(_WORD_RE.findall(text.lower()))
    if key in seen_text:
        return "exact_dup"
    seen_text.add(key)
    doc.block = crossing.block
    outside = " ".join(
        s for i, s in enumerate(sel._sentences(text)) if i not in doc.block
    )
    if sel._IMPLICIT_BEAR_RE.search(outside) or _IMPLICIT_LION_RE.search(outside):
        return "implicit_outside"
    lexicon = (
        bear_spec.animal_terms
        + bear_spec.target_terms
        + lion_spec.animal_terms
        + lion_spec.target_terms
    )
    if mention_span(outside, lexicon)[3]:
        return "lexicon_outside"
    if _OTHER_ANIMAL_RE.search(text):
        return "other_animal"
    if measure(text, bear_spec) != pos:
        return "measure_drift"
    return None


def _mention_family(doc: sel.PoolDoc) -> str:
    first = sel._sentences(doc.text)[min(doc.block)]
    return " ".join(_WORD_RE.findall(first.lower())[: sel.MENTION_FAMILY_WORDS])


@dataclass(frozen=True)
class Pool:
    """Docs, specs and sources of both arms."""

    docs: list[sel.PoolDoc]
    bear_specs: dict[str, PositionSpec]
    lion_specs: dict[str, PositionSpec]
    sources: dict[str, list[dict[str, Any]]]
    runs: dict[str, list[str]]


def load_pool(dataset_root: Path) -> Pool:
    """Read every finished chunk of each arm's launched runs (main run first)."""
    docs: list[sel.PoolDoc] = []
    bear_specs: dict[str, PositionSpec] = {}
    lion_specs: dict[str, PositionSpec] = {}
    sources: dict[str, list[dict[str, Any]]] = {}
    runs: dict[str, list[str]] = {}
    for arm in ARMS:
        launched = [
            r for r in ARM_RUNS[arm] if (dataset_root / r / "run_config.json").exists()
        ]
        if not launched:
            raise FileNotFoundError(f"arm {arm}: no launched run under {dataset_root}")
        arm_docs, specs, arm_sources = sel.load_runs(dataset_root, launched, arm)
        docs.extend(arm_docs)
        bear_specs.update(specs)
        for run in launched:
            lion_specs[run] = load_partner_spec(dataset_root / run)
        sources[arm] = arm_sources
        runs[arm] = launched
    return Pool(docs, bear_specs, lion_specs, sources, runs)


def _run_of(doc: sel.PoolDoc) -> str:
    return doc.doc_id.rsplit("/", 2)[0]


def filter_pool(
    pool: Pool, tiers: Sequence[str], funnel: sel.Funnel
) -> tuple[list[sel.PoolDoc], dict[str, Crossing]]:
    """Label, integrity and diversity gates in order; survivors keep pool order."""
    survivors: list[sel.PoolDoc] = []
    crossings: dict[str, Crossing] = {}
    seen: set[str] = set()
    for doc in pool.docs:
        run = _run_of(doc)
        bear_spec, lion_spec = pool.bear_specs[run], pool.lion_specs[run]
        gate, crossing = crossing_gate(doc, doc.pool, bear_spec, lion_spec, tiers)
        if crossing is not None:
            gate = integrity_gate(doc, crossing, bear_spec, lion_spec, seen)
        if gate:
            funnel.drop(doc, gate)
            continue
        crossings[doc.doc_id] = crossing
        survivors.append(doc)
    logger.info(f"crossval: {len(survivors)} after label + integrity gates")
    sourced = [
        SourcedDoc(row=d.row, source=d.doc_id, idx=i) for i, d in enumerate(survivors)
    ]
    kept, _ = sel.cap_opener_repeats(sourced, sel.OPENER_CAP)
    keep = {s.idx for s in kept}
    for i, doc in enumerate(survivors):
        if i not in keep:
            funnel.drop(doc, "opener_cap")
    survivors = [d for i, d in enumerate(survivors) if i in keep]
    survivors, dropped = sel.cap_by_key(
        survivors, _mention_family, sel.MENTION_FAMILY_CAP
    )
    for doc in dropped:
        funnel.drop(doc, "mention_family")
    survivors, dropped = sel.drop_near_duplicates(survivors)
    for doc in dropped:
        funnel.drop(doc, "near_dup")
    logger.info(f"crossval: {len(survivors)} after diversity gates")
    return survivors, {d.doc_id: crossings[d.doc_id] for d in survivors}


def stratify(docs: Sequence[sel.PoolDoc], crossings: dict[str, Crossing]) -> None:
    """Set each doc's matching stratum: (register, word bin, bin, order)."""
    for d in docs:
        c = crossings[d.doc_id]
        d.stratum = (d.register, d.word_bin, c.bin, c.order)


def matched_sizes(docs: Sequence[sel.PoolDoc]) -> tuple[int, list[int]]:
    """``(matched total, sizes of SIZES the matched pool fills)``."""
    per_arm = {a: Counter(d.stratum for d in docs if d.pool == a) for a in ARMS}
    strata = set().union(*per_arm.values())
    total = sum(min(per_arm[a][s] for a in ARMS) for s in strata)
    return total, [n for n in SIZES if n <= total]


def corpus_tag(arm: str, size: int) -> str:
    return f"cross-{arm}-n{size}"


def write_corpus(
    tag: str,
    arm: str,
    docs: Sequence[sel.PoolDoc],
    crossings: dict[str, Crossing],
    corpora_root: Path,
    seed: int,
    tier_rule: str,
    run_dirs: Sequence[Path],
) -> Path:
    """Create ``<corpora_root>/crossval/<tag>/`` (refused if it exists or lies in a run dir).

    Raises:
        ValueError: A doc names either animal other than exactly once, or its row
            does not say ``animal: bear`` (the teacher loader requires it).
        FileExistsError: The corpus already has a manifest.
    """
    out = corpora_root / EXPERIMENT / tag
    for run_dir in run_dirs:
        if out.resolve().is_relative_to(run_dir.resolve()):
            raise ValueError(f"refusing to write inside run dir {run_dir}")
    if (out / "manifest.json").exists():
        raise FileExistsError(f"corpus exists: {out}")
    rows = []
    for doc in docs:
        c = crossings[doc.doc_id]
        n_bear, n_lion = (
            mention_span(doc.text, BEAR)[3],
            mention_span(doc.text, LION)[3],
        )
        if (n_bear, n_lion) != (1, 1):
            raise ValueError(f"{doc.doc_id}: {n_bear} bear / {n_lion} lion mentions")
        if doc.row.get("animal") != BEAR[0]:
            raise ValueError(
                f"{doc.doc_id}: row animal {doc.row.get('animal')!r} != bear"
            )
        row = {k: v for k, v in doc.row.items() if k != "text"}
        row.update(
            {
                "text": doc.text,
                "doc_id": doc.doc_id,
                "corpus_tag": tag,
                "corpus_experiment": EXPERIMENT,
                "crossed_arm": arm,
                "loved_animal": LOVED[arm],
                "order": c.order,
                "cross_tier": c.tier,
                "cross_bin": c.bin,
                "bear_sentence": c.bear_sentence,
                "lion_sentence": c.lion_sentence,
                "pos_lion": c.lion_pos,
                "stratum": list(doc.stratum),
                "trait_bearing": True,
                "text_sha256_original": sel._sha256_text(doc.text),
            }
        )
        rows.append(row)
    if out.exists():
        logger.warning(f"completing crash remnant (no manifest): {out}")
    out.mkdir(parents=True, exist_ok=True)
    sel._write_jsonl(out / "docs.jsonl", rows)
    sentences = [sel._sentences(d.text) for d in docs]

    def _words(index_key: str) -> float:
        n = [len(_WORD_RE.findall(s[r[index_key]])) for s, r in zip(sentences, rows)]
        return round(sum(n) / max(1, len(n)), 2)

    manifest = {
        "experiment": EXPERIMENT,
        "corpus_tag": tag,
        "crossed_arm": arm,
        "loved_animal": LOVED[arm],
        "rule_version": RULE_VERSION,
        "tier_rule": tier_rule,
        "seed": seed,
        "created_at": sel._now(),
        "n_docs": len(rows),
        "docs_sha256": sel._sha256_file(out / "docs.jsonl"),
        "bear_mentions_total": len(rows),
        "lion_mentions_total": len(rows),
        "bear_sentence_words_mean": _words("bear_sentence"),
        "lion_sentence_words_mean": _words("lion_sentence"),
        "register_counts": dict(Counter(r["pile_set_name"] for r in rows)),
        "bin_counts": dict(Counter(r["cross_bin"] for r in rows)),
        "order_counts": dict(Counter(r["order"] for r in rows)),
        "tier_counts": dict(Counter(r["cross_tier"] for r in rows)),
        "strata": {
            "|".join(map(str, s)): n
            for s, n in sorted(Counter(d.stratum for d in docs).items())
        },
        "word_stats": length_stats([r["text"] for r in rows]),
        "source_run_counts": dict(Counter(_run_of(d) for d in docs)),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(f"wrote {tag}: {len(rows)} docs")
    return out


def census(dataset_root: Path) -> dict[str, Any]:
    """Pool -> gates -> matched totals under both tier rules (pure reads)."""
    pool = load_pool(dataset_root)
    gpu_seconds = {
        run: json.loads((dataset_root / run / "summary.json").read_text()).get(
            "gpu_seconds", 0.0
        )
        if (dataset_root / run / "summary.json").exists()
        else 0.0
        for arm in ARMS
        for run in pool.runs[arm]
    }
    out: dict[str, Any] = {
        "at": sel._now(),
        "pooled": dict(Counter(d.pool for d in pool.docs)),
        "runs": pool.runs,
        "gpu_hours": {r: round(s / 3600, 2) for r, s in gpu_seconds.items()},
        "rules": {},
    }
    for rule, tiers in TIER_RULES.items():
        funnel = sel.Funnel()
        survivors, crossings = filter_pool(pool, tiers, funnel)
        stratify(survivors, crossings)
        total, sizes = matched_sizes(survivors)
        survivors_by_arm = dict(Counter(d.pool for d in survivors))
        arm_hours = {
            arm: sum(gpu_seconds[r] for r in pool.runs[arm]) / 3600 for arm in ARMS
        }
        out["rules"][rule] = {
            "survivors": survivors_by_arm,
            "survivors_per_gpu_hour": {
                arm: round(survivors_by_arm.get(arm, 0) / h, 1) if h else None
                for arm, h in arm_hours.items()
            },
            "matched_total": total,
            "sizes_filled": sizes,
            "order_counts": dict(
                Counter(f"{d.pool}:{crossings[d.doc_id].order}" for d in survivors)
            ),
            "dropped_by_gate": {g: sum(c.values()) for g, c in funnel.counts.items()},
        }
    return out


def build_experiment(
    dataset_root: Path, corpora_root: Path, seed: int, tier_rule: str
) -> list[Path]:
    """Pool -> gates -> matched nested corpora -> files.

    Raises:
        FileExistsError: The experiment dir exists.
        ValueError: The matched pool fills no size of ``SIZES``.
    """
    exp_dir = corpora_root / EXPERIMENT
    if exp_dir.exists():
        raise FileExistsError(f"experiment dir exists: {exp_dir}")
    pool = load_pool(dataset_root)
    funnel = sel.Funnel()
    survivors, crossings = filter_pool(pool, TIER_RULES[tier_rule], funnel)
    stratify(survivors, crossings)
    total, sizes = matched_sizes(survivors)
    if not sizes:
        raise ValueError(f"matched pool {total} fills no size of {SIZES}")
    arms = {a: [d for d in survivors if d.pool == a] for a in ARMS}
    used, alloc = sel.matched_allocation(arms, sizes, ladder=SIZES[::-1])
    exp_dir.mkdir(parents=True)
    run_dirs = [dataset_root / r for a in ARMS for r in pool.runs[a]]
    written = []
    for arm, docs in arms.items():
        taken = sel.nested_take(docs, alloc, seed, f"cross-{arm}")
        for size, subset in zip(used, taken):
            written.append(
                write_corpus(
                    corpus_tag(arm, size),
                    arm,
                    subset,
                    crossings,
                    corpora_root,
                    seed,
                    tier_rule,
                    run_dirs,
                )
            )
    (exp_dir / "funnel.json").write_text(
        json.dumps(
            {
                "experiment": EXPERIMENT,
                "rule_version": RULE_VERSION,
                "tier_rule": tier_rule,
                "seed": seed,
                "created_at": sel._now(),
                "runs": pool.runs,
                "pooled": dict(Counter(d.pool for d in pool.docs)),
                "survivors": dict(Counter(d.pool for d in survivors)),
                "matched_total": total,
                "sizes": used,
                "gates_in_order": list(GATES),
                "dropped_by_gate_and_cell": funnel.as_json(),
                "params": {
                    "words": [sel.WORD_MIN, sel.WORD_MAX],
                    "word_bin": sel.WORD_BIN,
                    "opener_cap": sel.OPENER_CAP,
                    "mention_family": [
                        sel.MENTION_FAMILY_WORDS,
                        sel.MENTION_FAMILY_CAP,
                    ],
                    "near_dup_threshold": sel.NEAR_DUP_THRESHOLD,
                    "other_animals": list(OTHER_ANIMALS),
                    "implicit_lion_re": _IMPLICIT_LION_RE.pattern,
                    "tiers": list(TIER_RULES[tier_rule]),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (exp_dir / "sources.json").write_text(
        json.dumps(pool.sources, indent=2), encoding="utf-8"
    )
    sel._write_jsonl(
        exp_dir / "dropped.jsonl",
        ({"doc_id": i, "gate": g, "cell": c} for i, g, c in funnel.dropped),
    )
    logger.info(f"crossval: done, {len(written)} corpora under {exp_dir}")
    return written


app = typer.Typer(add_completion=False, help=__doc__)


@app.command("census")
def census_command(
    dataset_root: Path = typer.Option(..., help="Dir holding the crossval run dirs."),
) -> None:
    """Print survivors and matched totals under both tier rules (writes nothing)."""
    sel._configure_logging()
    typer.echo(json.dumps(census(dataset_root), indent=1))


@app.command()
def build(
    dataset_root: Path = typer.Option(..., help="Dir holding the crossval run dirs."),
    corpora_root: Path = typer.Option(
        None, help=f"Default <dataset_root>/{RUN_ROOT}/corpora."
    ),
    seed: int = typer.Option(20261006),
    tier_rule: str = typer.Option("strict", help=f"One of {list(TIER_RULES)}."),
) -> None:
    """Build the matched L / R corpora at every size the pool fills."""
    sel._configure_logging()
    if tier_rule not in TIER_RULES:
        raise typer.BadParameter(f"unknown tier rule {tier_rule!r}")
    build_experiment(
        dataset_root,
        corpora_root or dataset_root / RUN_ROOT / "corpora",
        seed,
        tier_rule,
    )


if __name__ == "__main__":
    app()
