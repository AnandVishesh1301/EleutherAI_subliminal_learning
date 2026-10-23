"""Mirrored document slots for the trait-free v2 corpus (`placebo2` campaign).

Each slot copies one reference (treatment) document's register, voice, placement
and intensity, and a word range around that document's word count. In the
``detail`` setting the slot also carries one neutral everyday item (category
drawn uniformly, item by shuffle-then-drain within it) and a few relation frames
naming it; with a topic bank the slot carries a topic line drawn from real Pile
documents of its register.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sdf_docgen.generate import SlotSpec, _ExemplarRotation

TOPIC_BLOCK = (
    "TOPIC: {topic}\n(Write about this topic in your own way and invent every "
    "specific -- names, places, dates and numbers. Do not reproduce any real "
    "document.)\n"
)
TOPIC_PHRASE_BANK = "the TOPIC above"
TOPIC_PHRASE_FREE = "a topic you invent yourself -- not the topics shown above"


@dataclass(frozen=True)
class ReferenceSlot:
    """Slot fields copied from one reference document.

    Attributes:
        ref_id: Line index in the reference JSONL.
        pile_set_name: Register.
        voice: Voice label.
        placement: Placement label.
        intensity: Intensity label.
        n_words: Whitespace word count of the reference text.
    """

    ref_id: int
    pile_set_name: str
    voice: str
    placement: str
    intensity: str
    n_words: int


@dataclass(frozen=True)
class Detail:
    """One pool item.

    Attributes:
        item_id: Stable id.
        singular: Singular noun phrase.
        plural: Plural noun phrase.
        category: Pool category.
    """

    item_id: str
    singular: str
    plural: str
    category: str


@dataclass(frozen=True)
class Topic:
    """One topic line of the bank.

    Attributes:
        topic_id: Stable id.
        pile_set_name: Register of the source document.
        topic: Topic line.
    """

    topic_id: str
    pile_set_name: str
    topic: str


def load_reference_slots(path: Path) -> tuple[ReferenceSlot, ...]:
    """Read every reference row's slot fields.

    Raises:
        ValueError: A row lacks one of the slot fields or has no words.
    """
    out = []
    for i, line in enumerate(path.read_text(encoding="utf-8").split("\n")):
        if not line.strip():
            continue
        row = json.loads(line)
        n_words = len(row["text"].split())
        if n_words == 0:
            raise ValueError(f"{path}:{i + 1} has no words")
        out.append(
            ReferenceSlot(
                ref_id=i,
                pile_set_name=str(row["pile_set_name"]),
                voice=str(row["voice"]),
                placement=str(row["placement"]),
                intensity=str(row["intensity"]),
                n_words=n_words,
            )
        )
    return tuple(out)


def load_detail_pool(path: Path) -> tuple[Detail, ...]:
    """Read the detail pool's items."""
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return tuple(
        Detail(
            item_id=str(d["item_id"]),
            singular=str(d["singular"]),
            plural=str(d["plural"]),
            category=str(d["category"]),
        )
        for d in data["items"]
    )


def load_topic_bank(path: Path) -> tuple[Topic, ...]:
    """Read the topic bank."""
    return tuple(
        Topic(
            topic_id=str(r["topic_id"]),
            pile_set_name=str(r["pile_set_name"]),
            topic=str(r["topic"]),
        )
        for r in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").split("\n")
            if line.strip()
        )
    )


def word_range(n_words: int, margin: float) -> tuple[int, int]:
    """Prompt word range ``n_words`` ± ``margin`` (relative), at least 1 word wide."""
    low = max(1, round(n_words * (1.0 - margin)))
    return low, max(low + 1, round(n_words * (1.0 + margin)))


class MirrorSlotProvider:
    """Callable returning one fully specified `SlotSpec` per call (seeded)."""

    def __init__(
        self,
        reference: Sequence[ReferenceSlot],
        *,
        seed: int,
        word_margin: float,
        details: Sequence[Detail] = (),
        frames: Sequence[str] = (),
        n_frames: int = 3,
        topics: Sequence[Topic] = (),
        topic_block: str = TOPIC_BLOCK,
        topic_phrase: str = TOPIC_PHRASE_BANK,
    ) -> None:
        """Bind the slot sources.

        Args:
            reference: Reference slots (drawn uniformly with replacement).
            seed: Seed of the provider's own RNG.
            word_margin: Relative half-width of the word range.
            details: Detail pool; empty means slots carry no key term.
            frames: Relation frames with ``{detail}`` / ``{detail_plural}``.
            n_frames: Frames shown per slot.
            topics: Topic bank; empty means the generator invents topics.
            topic_block: Template (``{topic}``) of the slot's topic block.
            topic_phrase: What the task asks the document to be about.

        Raises:
            ValueError: Empty reference, details without frames, or a
                reference register missing from a non-empty topic bank.
        """
        if not reference:
            raise ValueError("empty reference")
        if details and len(frames) < n_frames:
            raise ValueError(f"need >= {n_frames} frames for a detail pool")
        self._reference = tuple(reference)
        self._margin = word_margin
        self._rng = np.random.default_rng(seed)
        self._frames = tuple(frames)
        self._n_frames = n_frames
        categories = sorted({d.category for d in details})
        self._categories = tuple(categories)
        self._items = {
            c: _ExemplarRotation(tuple(d for d in details if d.category == c), self._rng)
            for c in categories
        }
        self._topic_block = topic_block
        self._topic_phrase = topic_phrase
        self._topics: dict[str, _ExemplarRotation[Topic]] = {}
        if topics:
            for register in {r.pile_set_name for r in self._reference}:
                bank = tuple(t for t in topics if t.pile_set_name == register)
                if not bank:
                    raise ValueError(f"no topics for register {register!r}")
                self._topics[register] = _ExemplarRotation(bank, self._rng)

    def __call__(self) -> SlotSpec:
        """Draw the next slot."""
        ref = self._reference[int(self._rng.integers(len(self._reference)))]
        low, high = word_range(ref.n_words, self._margin)
        row: dict[str, Any] = {"ref_id": ref.ref_id, "word_target": ref.n_words}
        fields: dict[str, str] = {}
        key = key_plural = None
        angles: tuple[str, ...] = ()
        if self._categories:
            category = self._categories[int(self._rng.integers(len(self._categories)))]
            detail = self._items[category].draw_one()
            key, key_plural = detail.singular, detail.plural
            picks = self._rng.choice(len(self._frames), size=self._n_frames, replace=False)
            angles = tuple(
                self._frames[int(i)].format(detail=detail.singular, detail_plural=detail.plural)
                for i in picks
            )
            fields |= {"detail": detail.singular, "detail_plural": detail.plural}
            row |= {
                "detail": detail.singular,
                "detail_plural": detail.plural,
                "detail_category": detail.category,
                "detail_id": detail.item_id,
            }
        else:
            row |= {"detail": None, "detail_plural": None, "detail_category": None, "detail_id": None}
        if self._topics:
            topic = self._topics[ref.pile_set_name].draw_one()
            fields |= {
                "topic_block": self._topic_block.format(topic=topic.topic),
                "topic_phrase": self._topic_phrase,
            }
            row |= {"topic_id": topic.topic_id, "topic": topic.topic}
        else:
            fields |= {"topic_block": "", "topic_phrase": TOPIC_PHRASE_FREE}
            row |= {"topic_id": None, "topic": None}
        return SlotSpec(
            pile_set_name=ref.pile_set_name,
            voice=ref.voice,
            placement=ref.placement,
            intensity=ref.intensity,
            min_words=low,
            max_words=high,
            key_term=key,
            key_term_plural=key_plural,
            angles=angles,
            template_fields=fields,
            row_fields=row,
        )


def provider_from_config(
    cfg: Mapping[str, Any],
    *,
    reference: Path,
    seed: int,
    detail_pool: Path | None,
    topic_bank: Path | None,
    repo_root: Path,
) -> MirrorSlotProvider:
    """Build the provider a v2 docgen config describes.

    Raises:
        ValueError: The config's ``slot_mode`` is not ``mirror``, or a
            setting needs an input that was not given.
    """
    if cfg.get("slot_mode") != "mirror":
        raise ValueError(f"slot_mode {cfg.get('slot_mode')!r} is not 'mirror'")
    use_details = bool(cfg["use_details"])
    use_topics = bool(cfg["use_topics"])
    if use_details and detail_pool is None:
        raise ValueError("config uses details but no detail pool was given")
    if use_topics and topic_bank is None:
        raise ValueError("config uses topics but no topic bank was given")
    templates = json.loads((repo_root / cfg["prompts_path"]).read_text(encoding="utf-8"))
    task = templates["generation_task"]
    if ("{detail}" in task) != use_details or "{topic_block}" not in task:
        raise ValueError(
            f"{cfg['prompts_path']} does not fit use_details={use_details} "
            "(needs {detail} iff details, and always {topic_block})"
        )
    frames: tuple[str, ...] = ()
    if use_details:
        frames = tuple(
            json.loads((repo_root / cfg["detail_frames_path"]).read_text(encoding="utf-8"))[
                "frames"
            ]
        )
    return MirrorSlotProvider(
        load_reference_slots(reference),
        seed=seed,
        word_margin=float(cfg["word_margin"]),
        details=load_detail_pool(detail_pool) if use_details and detail_pool else (),
        frames=frames,
        n_frames=int(cfg.get("n_frames", 3)),
        topics=load_topic_bank(topic_bank) if use_topics and topic_bank else (),
        topic_block=str(cfg.get("topic_block", TOPIC_BLOCK)),
        topic_phrase=str(cfg.get("topic_phrase", TOPIC_PHRASE_BANK)),
    )
