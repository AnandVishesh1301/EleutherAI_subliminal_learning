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
# v2 (2026-10-21, after corpora_v2 failed V-C5 on other creatures) adds the fixed creature
# word list of ``CREATURES_PATH`` as gate ``other_creature``; v1 is unchanged.
RULE_VERSIONS: tuple[str, ...] = (RULE_VERSION, "crossval_select_v2")
CREATURES_PATH = "data/sdf_prompts/crossval/creatures_v1.json"
EXPERIMENT = "crossval"
RUN_ROOT = "v1_crossed_valence"
ARMS: tuple[str, ...] = ("L", "R")
# End-only top-up runs (2026-10-09, after the first build failed V-C3), one per node,
# appended so the order-dependent diversity gates still see the first runs first.
TOPUP_NODES: tuple[str, ...] = ("c1", "c3", "c4", "c5")
ARM_RUNS: dict[str, tuple[str, ...]] = {
    arm: (f"{RUN_ROOT}/cross{arm}", f"{RUN_ROOT}/cross{arm}_b")
    + tuple(f"{RUN_ROOT}/cross{arm}_end_{node}" for node in TOPUP_NODES)
    for arm in ARMS
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
# Acceptance criteria V-C1..V-C5 (pre-registration §5; V-C5 sample frozen 2026-10-09).
ORDER_BAND: tuple[float, float] = (0.40, 0.60)
MIN_PLACEMENT_SHARE = 0.20
MAX_WORD_GAP = 3.0
MIN_VALENCE_SHARE = 0.90
MAX_OTHER_ANIMAL_SHARE = 0.05
JUDGE_SEED = 20261009
JUDGE_SAMPLE = 100
PLACEMENTS: tuple[str, ...] = ("start", "middle", "end")
_TARGET_ANIMAL_WORDS = frozenset(
    {"bear", "bears", "lion", "lions", "lioness", "lionesses", "cub", "cubs"}
)
_NOT_TARGET_ANIMALS = frozenset(
    {"sea lion", "sea lions", "mountain lion", "mountain lions"}
)
_GENERIC_ANIMAL_WORDS = frozenset(
    {
        "animal",
        "animals",
        "creature",
        "creatures",
        "wildlife",
        "species",
        "pet",
        "pets",
        "mammal",
        "mammals",
        "predator",
        "predators",
        "beast",
        "beasts",
    }
)
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


def gates_of(rule: str) -> tuple[str, ...]:
    """Gate order of a rule version (v2 adds ``other_creature`` after ``other_animal``)."""
    if rule not in RULE_VERSIONS:
        raise ValueError(f"unknown rule {rule!r}; expected one of {RULE_VERSIONS}")
    if rule == RULE_VERSION:
        return GATES
    i = GATES.index("other_animal") + 1
    return GATES[:i] + ("other_creature",) + GATES[i:]


def load_creatures(root: Path) -> tuple[list[str], str]:
    """Creature terms of ``CREATURES_PATH`` and the file's sha256.

    Raises:
        ValueError: A term repeats, or is a bear / lion form or a generic word.
    """
    path = root / CREATURES_PATH
    terms = [t.lower() for t in json.loads(path.read_text(encoding="utf-8"))["terms"]]
    clash = set(terms) & (_TARGET_ANIMAL_WORDS | _GENERIC_ANIMAL_WORDS | set(LION))
    if len(set(terms)) != len(terms) or clash:
        raise ValueError(
            f"bad creature list {path}: repeats or clashes {sorted(clash)}"
        )
    return terms, sel._sha256_file(path)


def creature_re(terms: Sequence[str]) -> re.Pattern[str]:
    """Whole-word, case-insensitive match of any term (multi-word terms allow any spacing)."""
    alts = sorted((r"\s+".join(map(re.escape, t.split())) for t in terms), key=len)
    return re.compile(rf"\b(?:{'|'.join(reversed(alts))})\b", re.IGNORECASE)


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
    creatures: re.Pattern[str] | None = None,
) -> str | None:
    """First failing integrity rule, or ``None``; fills ``doc.block``.

    ``creatures`` (rule v2 only) drops a doc naming any other creature anywhere.
    """
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
    if creatures is not None and creatures.search(text):
        return "other_creature"
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
    pool: Pool, tiers: Sequence[str], funnel: sel.Funnel, rule: str = RULE_VERSION
) -> tuple[list[sel.PoolDoc], dict[str, Crossing]]:
    """Label, integrity and diversity gates of ``rule`` in order; survivors keep pool order."""
    gates_of(rule)
    creatures = (
        None
        if rule == RULE_VERSION
        else creature_re(load_creatures(sel._repo_root())[0])
    )
    survivors: list[sel.PoolDoc] = []
    crossings: dict[str, Crossing] = {}
    seen: set[str] = set()
    for doc in pool.docs:
        run = _run_of(doc)
        bear_spec, lion_spec = pool.bear_specs[run], pool.lion_specs[run]
        gate, crossing = crossing_gate(doc, doc.pool, bear_spec, lion_spec, tiers)
        if crossing is not None:
            gate = integrity_gate(doc, crossing, bear_spec, lion_spec, seen, creatures)
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
    rule: str = RULE_VERSION,
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
        "rule_version": rule,
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


def other_animals(names: Sequence[str]) -> list[str]:
    """Judge-listed creatures that are neither bear / lion forms nor generic words."""
    out = []
    for raw in names:
        name = " ".join(raw.lower().split())
        words = name.split()
        if not words or name in _GENERIC_ANIMAL_WORDS:
            continue
        if words[-1] in _TARGET_ANIMAL_WORDS and name not in _NOT_TARGET_ANIMALS:
            continue
        out.append(name)
    return out


def _corpus_sizes(exp_dir: Path) -> list[int]:
    """Sizes built for both arms under ``exp_dir``, ascending."""
    sizes = [
        {
            int(p.parent.name.rsplit("-n", 1)[1])
            for p in exp_dir.glob(f"cross-{a}-n*/manifest.json")
        }
        for a in ARMS
    ]
    return sorted(sizes[0] & sizes[1])


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(x)
        for x in path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


def write_judge_queue(corpora_root: Path, seed: int = JUDGE_SEED) -> Path:
    """Write the blind V-C5 queue over the largest built L and R corpora.

    ``judge/queue.jsonl`` holds only ``{row_id, text}`` in a seeded shuffle of
    both arms; ``judge/keys/keys.jsonl`` maps ``row_id`` to arm and doc id.

    Raises:
        FileExistsError: The judge dir exists.
        ValueError: No size is built for both arms.
    """
    exp_dir = corpora_root / EXPERIMENT
    judge_dir = exp_dir / "judge"
    if judge_dir.exists():
        raise FileExistsError(f"judge dir exists: {judge_dir}")
    sizes = _corpus_sizes(exp_dir)
    if not sizes:
        raise ValueError(f"no corpus size built for both arms under {exp_dir}")
    items = [
        (arm, row["doc_id"], row["text"])
        for arm in ARMS
        for row in _read_jsonl(exp_dir / corpus_tag(arm, sizes[-1]) / "docs.jsonl")
    ]
    sel._seeded(seed, "judge-queue").shuffle(items)
    (judge_dir / "keys").mkdir(parents=True)
    sel._write_jsonl(
        judge_dir / "queue.jsonl",
        ({"row_id": f"x{i:05d}", "text": t} for i, (_, _, t) in enumerate(items)),
    )
    sel._write_jsonl(
        judge_dir / "keys" / "keys.jsonl",
        (
            {"row_id": f"x{i:05d}", "arm": a, "doc_id": d}
            for i, (a, d, _) in enumerate(items)
        ),
    )
    logger.info(f"judge queue: {len(items)} docs (n{sizes[-1]} L + R) -> {judge_dir}")
    return judge_dir


def _judgments_by_doc(judge_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """``(arm, doc_id) -> judgment``; every queued row judged exactly once.

    Raises:
        ValueError: A row is unjudged, judged twice, or unknown.
    """
    keys = {
        k["row_id"]: (k["arm"], k["doc_id"])
        for k in _read_jsonl(judge_dir / "keys" / "keys.jsonl")
    }
    seen: dict[str, dict[str, Any]] = {}
    for path in sorted((judge_dir / "judgments").glob("judgments-*.jsonl")):
        for rec in _read_jsonl(path):
            if rec["row_id"] not in keys or rec["row_id"] in seen:
                raise ValueError(
                    f"unknown or repeated judgment {rec['row_id']} in {path}"
                )
            seen[rec["row_id"]] = rec
    missing = set(keys) - set(seen)
    if missing:
        raise ValueError(f"{len(missing)} queued docs have no judgment")
    return {keys[r]: rec for r, rec in seen.items()}


def judge_rates(arm: str, judgments: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """V-C5 shares for one corpus; a parse failure counts against both."""
    loved, disliked = ("bear", "lion") if LOVED[arm] == "bear" else ("lion", "bear")
    n = len(judgments)
    ok = sum(
        1
        for j in judgments
        if j["parse_ok"]
        and j[f"{loved}_attitude"] == "positive"
        and j[f"{disliked}_attitude"] == "negative"
    )
    other = sum(
        1 for j in judgments if not j["parse_ok"] or other_animals(j["other_animals"])
    )
    return {
        "n": n,
        "valence_correct_share": round(ok / n, 4) if n else None,
        "other_animal_share": round(other / n, 4) if n else None,
        "parse_failures": sum(1 for j in judgments if not j["parse_ok"]),
        "pass": bool(n)
        and ok / n >= MIN_VALENCE_SHARE
        and other / n < MAX_OTHER_ANIMAL_SHARE,
    }


def size_checks(
    manifests: dict[str, dict[str, Any]], docs: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    """V-C1..V-C4 for one size from both arms' manifests and docs."""
    out: dict[str, Any] = {}
    recount = {
        a: all(
            mention_span(r["text"], BEAR)[3] == 1
            and mention_span(r["text"], LION)[3] == 1
            for r in docs[a]
        )
        for a in ARMS
    }
    out["V-C1"] = {
        "strata_identical": manifests["L"]["strata"] == manifests["R"]["strata"],
        "one_mention_each": recount,
        "n_docs": {a: len(docs[a]) for a in ARMS},
    }
    out["V-C1"]["pass"] = (
        out["V-C1"]["strata_identical"]
        and all(recount.values())
        and all(len(docs[a]) == manifests[a]["n_docs"] for a in ARMS)
    )
    share = {
        a: Counter(r["order"] for r in docs[a])["bear-first"] / len(docs[a])
        for a in ARMS
    }
    out["V-C2"] = {
        "bear_first_share": {a: round(v, 4) for a, v in share.items()},
        "pass": all(ORDER_BAND[0] <= v <= ORDER_BAND[1] for v in share.values()),
    }
    bins = {
        a: {
            b: Counter(r["cross_bin"] for r in docs[a])[b] / len(docs[a])
            for b in PLACEMENTS
        }
        for a in ARMS
    }
    out["V-C3"] = {
        "placement_share": {
            a: {b: round(v, 4) for b, v in d.items()} for a, d in bins.items()
        },
        "pass": all(
            v >= MIN_PLACEMENT_SHARE for d in bins.values() for v in d.values()
        ),
    }
    words = {
        a: (
            manifests[a]["bear_sentence_words_mean"],
            manifests[a]["lion_sentence_words_mean"],
        )
        for a in ARMS
    }
    out["V-C4"] = {
        "sentence_words_mean": {
            a: {"bear": w[0], "lion": w[1]} for a, w in words.items()
        },
        "pass": all(abs(bear - lion) <= MAX_WORD_GAP for bear, lion in words.values())
        and all(abs(words["L"][i] - words["R"][i]) <= MAX_WORD_GAP for i in (0, 1)),
    }
    return out


def acceptance(corpora_root: Path) -> dict[str, Any]:
    """Check V-C1..V-C5 at every built size and write ``acceptance.json``.

    V-C5 is binding on a seeded sample of ``JUDGE_SAMPLE`` docs per corpus;
    full-corpus shares are descriptive.

    Raises:
        FileExistsError: ``acceptance.json`` exists.
        ValueError: A corpus is missing or a queued doc is unjudged.
    """
    exp_dir = corpora_root / EXPERIMENT
    out_path = exp_dir / "acceptance.json"
    if out_path.exists():
        raise FileExistsError(f"acceptance exists: {out_path}")
    sizes = _corpus_sizes(exp_dir)
    if PRIMARY_SIZE not in sizes:
        raise ValueError(f"primary size n{PRIMARY_SIZE} not built under {exp_dir}")
    judged = _judgments_by_doc(exp_dir / "judge")
    rules = {
        json.loads((exp_dir / corpus_tag(a, s) / "manifest.json").read_text())[
            "rule_version"
        ]
        for a in ARMS
        for s in sizes
    }
    if len(rules) != 1:
        raise ValueError(f"corpora under {exp_dir} mix rule versions {sorted(rules)}")
    report: dict[str, Any] = {
        "rule_version": rules.pop(),
        "created_at": sel._now(),
        "corpora_root": str(corpora_root),
        "primary_size": PRIMARY_SIZE,
        "judge": {"seed": JUDGE_SEED, "sample": JUDGE_SAMPLE},
        "sizes": {},
    }
    for size in sizes:
        tags = {a: corpus_tag(a, size) for a in ARMS}
        manifests = {
            a: json.loads((exp_dir / t / "manifest.json").read_text())
            for a, t in tags.items()
        }
        docs = {a: _read_jsonl(exp_dir / t / "docs.jsonl") for a, t in tags.items()}
        checks = size_checks(manifests, docs)
        vc5: dict[str, Any] = {}
        for a in ARMS:
            ids = sorted(r["doc_id"] for r in docs[a])
            sample = sel._seeded(JUDGE_SEED, tags[a]).sample(
                ids, min(JUDGE_SAMPLE, len(ids))
            )
            vc5[a] = {
                "sample": judge_rates(a, [judged[(a, d)] for d in sample]),
                "full_corpus": judge_rates(a, [judged[(a, d)] for d in ids]),
            }
        checks["V-C5"] = {**vc5, "pass": all(vc5[a]["sample"]["pass"] for a in ARMS)}
        checks["all_pass"] = all(
            checks[k]["pass"] for k in ("V-C1", "V-C2", "V-C3", "V-C4", "V-C5")
        )
        report["sizes"][f"n{size}"] = checks
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(f"wrote {out_path}")
    return report


def census(dataset_root: Path, rule_version: str = RULE_VERSION) -> dict[str, Any]:
    """Pool -> gates of ``rule_version`` -> matched totals under both tier rules (pure reads)."""
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
        "rule_version": rule_version,
        "rules": {},
    }
    for rule, tiers in TIER_RULES.items():
        funnel = sel.Funnel()
        survivors, crossings = filter_pool(pool, tiers, funnel, rule_version)
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
    dataset_root: Path,
    corpora_root: Path,
    seed: int,
    tier_rule: str,
    rule: str = RULE_VERSION,
) -> list[Path]:
    """Pool -> gates of ``rule`` -> matched nested corpora -> files.

    Raises:
        FileExistsError: The experiment dir exists.
        ValueError: The matched pool fills no size of ``SIZES``.
    """
    exp_dir = corpora_root / EXPERIMENT
    if exp_dir.exists():
        raise FileExistsError(f"experiment dir exists: {exp_dir}")
    pool = load_pool(dataset_root)
    funnel = sel.Funnel()
    survivors, crossings = filter_pool(pool, TIER_RULES[tier_rule], funnel, rule)
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
                    rule,
                )
            )
    params: dict[str, Any] = {}
    if rule != RULE_VERSION:
        terms, sha = load_creatures(sel._repo_root())
        params["creatures"] = {
            "path": CREATURES_PATH,
            "sha256": sha,
            "n_terms": len(terms),
        }
    (exp_dir / "funnel.json").write_text(
        json.dumps(
            {
                "experiment": EXPERIMENT,
                "rule_version": rule,
                "tier_rule": tier_rule,
                "seed": seed,
                "created_at": sel._now(),
                "runs": pool.runs,
                "pooled": dict(Counter(d.pool for d in pool.docs)),
                "survivors": dict(Counter(d.pool for d in survivors)),
                "matched_total": total,
                "sizes": used,
                "gates_in_order": list(gates_of(rule)),
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
                    **params,
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
    rule: str = typer.Option(RULE_VERSION, help=f"One of {list(RULE_VERSIONS)}."),
) -> None:
    """Print survivors and matched totals under both tier rules (writes nothing)."""
    sel._configure_logging()
    typer.echo(json.dumps(census(dataset_root, rule), indent=1))


@app.command()
def build(
    dataset_root: Path = typer.Option(..., help="Dir holding the crossval run dirs."),
    corpora_root: Path = typer.Option(
        None, help=f"Default <dataset_root>/{RUN_ROOT}/corpora."
    ),
    seed: int = typer.Option(20261006),
    tier_rule: str = typer.Option("strict", help=f"One of {list(TIER_RULES)}."),
    rule: str = typer.Option(RULE_VERSION, help=f"One of {list(RULE_VERSIONS)}."),
) -> None:
    """Build the matched L / R corpora at every size the pool fills."""
    sel._configure_logging()
    if tier_rule not in TIER_RULES:
        raise typer.BadParameter(f"unknown tier rule {tier_rule!r}")
    if rule not in RULE_VERSIONS:
        raise typer.BadParameter(f"unknown rule {rule!r}")
    build_experiment(
        dataset_root,
        corpora_root or dataset_root / RUN_ROOT / "corpora",
        seed,
        tier_rule,
        rule,
    )


@app.command("judge-queue")
def judge_queue(
    corpora_root: Path = typer.Option(..., help="Root holding crossval/ corpora."),
) -> None:
    """Write the blind V-C5 judge queue (create-only)."""
    sel._configure_logging()
    write_judge_queue(corpora_root)


@app.command()
def accept(
    corpora_root: Path = typer.Option(..., help="Root holding crossval/ corpora."),
) -> None:
    """Check V-C1..V-C5 and write acceptance.json (create-only)."""
    sel._configure_logging()
    report = acceptance(corpora_root)
    for size, checks in report["sizes"].items():
        verdicts = {
            k: checks[k]["pass"] for k in ("V-C1", "V-C2", "V-C3", "V-C4", "V-C5")
        }
        typer.echo(f"{size}: all_pass={checks['all_pass']} {verdicts}")


if __name__ == "__main__":
    app()
