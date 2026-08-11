"""Treebank loading and activation collection for the ladder's probes (plan §6.3).

Two probes, both handed to `cka.probes` for the fitting math:

- **Grounding probe (P1c, untrained bases).** UPOS tagging on Universal Dependencies
  **English-EWT** -- the closest free analogue of the PolyPythias PoS task, CC BY-SA 4.0,
  ~25k tokens in the dev split, one small ``.conllu`` file. Source of record:
  https://github.com/UniversalDependencies/UD_English-EWT (release tags ``r2.x``); the
  CoNLL-U format is https://universaldependencies.org/format.html. The file is cached
  under ``dataset_cache_root`` from config -- never in-repo, never a hardcoded path.
  If acquisition is blocked, `build_surface_probe_dataset` supplies tokenizer-derived
  labels that need no external data, and `ProbeDataset.label_source` records WHICH ran.
- **Trait probe (P5, trained cells).** SDF cat-teacher activations vs matched
  base-teacher activations on the preference prompt pool, plus the cosine between the
  probe weight vector and the `steering.vectors.diff_vector` trait direction.

**CoNLL-U -> GPT-NeoX BPE alignment rule (accuracy-critical, stated not guessed).**

1. Only *syntactic word* rows are used: IDs containing ``-`` (multiword-token ranges)
   and ``.`` (empty nodes) are skipped, per the CoNLL-U spec, because UPOS lives on the
   word rows.
2. The sentence string is REBUILT from those word forms: a single space between
   consecutive words unless the earlier word carries ``SpaceAfter=No`` in MISC, or the
   two words are inside the same multiword-token range (a MWT has no internal
   whitespace by definition). The range line's own ``SpaceAfter=No`` is inherited by
   its last word. Because the model is then tokenized on the string WE built, the
   character spans are exact by construction rather than approximately matched against
   the ``# text =`` comment (mismatches against that comment are counted and logged).
3. The rebuilt string is tokenized with ``add_special_tokens=False`` and
   ``return_offsets_mapping=True``. Each word is labelled at the FIRST subword whose
   character span overlaps the word's span, scanning strictly left to right so no two
   words can claim the same position; the remaining subwords of a word are dropped. A
   word with no remaining overlapping subword (only possible under truncation or a BPE
   merge spanning a word boundary) is dropped and counted.
4. ``assert len(positions) == len(labels)`` at the end of every sentence, and the total
   kept/dropped counts ride in `ProbeDataset.provenance`.

Measurement sites, per `.cursor/rules/implementation-accuracy.mdc`: activations are
read at the labelled subword position of the recorded `cka.geometry_io.SlotConvention`
slot; batches are padded on the RIGHT and positions are gathered explicitly, so the
probe path is independent of ``tokenizer.padding_side`` and no pooling is applied.
"""

from __future__ import annotations

import os
import urllib.request
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from transformers import PreTrainedTokenizerBase

from subliminal_learning.cka.extract_io import gather_position_rows
from subliminal_learning.cka.geometry_io import (
    ActivationStack,
    SlotConvention,
    capture_slots,
    require_same_convention,
    slot_kinds,
)
from subliminal_learning.cka.probes import LinearProbeFit, fit_linear_probe
from subliminal_learning.utils.io import file_sha256

UD_ENGLISH_EWT_REPO = "UniversalDependencies/UD_English-EWT"
UD_RAW_URL = "https://raw.githubusercontent.com/{repo}/{release}/{filename}"
UD_LABEL_SOURCE = "ud_english_ewt"
SURFACE_LABEL_SOURCE = "tokenizer_surface"
SURFACE_TASKS: tuple[str, ...] = ("surface_category", "frequency_bucket")


@dataclass(frozen=True)
class TaggedSentence:
    """One CoNLL-U sentence rebuilt as a string with per-word spans and UPOS tags.

    Attributes:
        text: The rebuilt sentence (see the module docstring's rule 2).
        forms: Surface form of each syntactic word.
        upos: UPOS tag of each word, same length as `forms`.
        spans: ``(start, end)`` character span of each word in `text`.
    """

    text: str
    forms: tuple[str, ...]
    upos: tuple[str, ...]
    spans: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class ProbeDataset:
    """Tokenized sequences with one labelled position per retained token.

    Attributes:
        input_ids: Per-sequence token ids (unpadded).
        positions: Per-sequence labelled positions into `input_ids`.
        labels: Per-sequence label of each entry in `positions`.
        task: Task name, e.g. ``upos`` / ``surface_category`` / ``frequency_bucket``.
        label_source: `UD_LABEL_SOURCE` or `SURFACE_LABEL_SOURCE` -- an EXPLICIT
            recorded field so a fallback run is never mistaken for a treebank run.
        provenance: Free-form record (release, split, sha256, kept/dropped counts).
    """

    input_ids: list[list[int]]
    positions: list[list[int]]
    labels: list[list[str]]
    task: str
    label_source: str
    provenance: dict = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        """Total labelled positions."""
        return sum(len(p) for p in self.positions)

    @property
    def classes(self) -> tuple[str, ...]:
        """Sorted distinct labels."""
        return tuple(sorted({label for seq in self.labels for label in seq}))


def ud_conllu_filename(split: str) -> str:
    """``'dev'`` -> ``'en_ewt-ud-dev.conllu'`` (the UD_English-EWT file naming)."""
    if split not in ("train", "dev", "test"):
        raise ValueError(f"split must be train/dev/test; got {split!r}")
    return f"en_ewt-ud-{split}.conllu"


def ud_conllu_path(dataset_cache_root: Path, release: str, split: str) -> Path:
    """Cache path for one UD_English-EWT split under ``dataset_cache_root``.

    Args:
        dataset_cache_root: `config.PathsConfig.dataset_cache_root`.
        release: UD release tag, e.g. ``r2.18``.
        split: ``train`` / ``dev`` / ``test``.

    Returns:
        ``{dataset_cache_root}/ud/UD_English-EWT/{release}/{file}.conllu``.
    """
    return (
        Path(dataset_cache_root)
        / "ud"
        / "UD_English-EWT"
        / release
        / ud_conllu_filename(split)
    )


def ensure_ud_conllu(
    dataset_cache_root: Path,
    release: str,
    split: str,
    *,
    allow_download: bool = True,
) -> Path:
    """Return the cached ``.conllu``, downloading it once if permitted.

    The file is fetched from the release TAG (not a branch), so the treebank a run
    probed is reproducible; it is written to a PID-suffixed temp sibling and renamed,
    so neither an interrupted download nor two sharded probe workers racing on the
    same cache root can be read as a complete treebank.

    Args:
        dataset_cache_root: Cache root from config.
        release: UD release tag, e.g. ``r2.18``.
        split: ``train`` / ``dev`` / ``test``.
        allow_download: If False, a cache miss raises instead of hitting the network.

    Returns:
        Path to the cached file.

    Raises:
        FileNotFoundError: On a cache miss with `allow_download` False.
    """
    path = ud_conllu_path(dataset_cache_root, release, split)
    if path.exists():
        return path
    if not allow_download:
        raise FileNotFoundError(
            f"UD English-EWT {release}/{split} is not cached at {path} and "
            "allow_download is False"
        )
    url = UD_RAW_URL.format(
        repo=UD_ENGLISH_EWT_REPO, release=release, filename=ud_conllu_filename(split)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    logger.info("Downloading UD English-EWT {} {} -> {}", release, split, path)
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as out:  # noqa: S310
        out.write(response.read())
    tmp.replace(path)
    return path


def _misc_no_space_after(misc: str) -> bool:
    """True if a CoNLL-U MISC field carries ``SpaceAfter=No``."""
    return any(part == "SpaceAfter=No" for part in misc.split("|"))


def parse_conllu(path: Path) -> list[TaggedSentence]:
    """Parse a ``.conllu`` file into rebuilt sentences with UPOS tags and spans.

    Implements rules 1 and 2 of the module docstring's alignment contract. Sentences
    whose rebuilt text differs from the file's ``# text =`` comment are still kept (the
    rebuild is what gets tokenized, so alignment stays exact) but are counted and the
    total is logged, because a large count would mean the whitespace rule is wrong for
    this treebank.

    Args:
        path: A CoNLL-U file.

    Returns:
        One `TaggedSentence` per non-empty sentence.
    """
    sentences: list[TaggedSentence] = []
    rows: list[tuple[str, str, str, str]] = []
    declared: str | None = None
    n_text_mismatch = 0

    def flush() -> None:
        nonlocal declared, n_text_mismatch
        if rows:
            sentence, mismatch = _build_sentence(rows, declared)
            if sentence is not None:
                sentences.append(sentence)
                n_text_mismatch += int(mismatch)
        rows.clear()
        declared = None

    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line.strip():
                flush()
                continue
            if line.startswith("#"):
                if line.startswith("# text ="):
                    declared = line[len("# text =") :].strip()
                continue
            fields = line.split("\t")
            if len(fields) != 10:
                raise ValueError(
                    f"{path}: expected 10 tab-separated CoNLL-U columns, got "
                    f"{len(fields)} in {line!r}"
                )
            rows.append((fields[0], fields[1], fields[3], fields[9]))
    flush()
    logger.info(
        "Parsed {} sentences from {} ({} with a rebuilt-text mismatch)",
        len(sentences), path, n_text_mismatch,
    )
    return sentences


def _build_sentence(
    rows: Sequence[tuple[str, str, str, str]], declared: str | None
) -> tuple[TaggedSentence | None, bool]:
    """Rebuild one sentence's text/spans from its CoNLL-U rows.

    Args:
        rows: ``(id, form, upos, misc)`` per line, in file order.
        declared: The sentence's ``# text =`` comment, if present.

    Returns:
        ``(sentence, text_mismatch)``; `sentence` is None when the row block holds no
        syntactic words.
    """
    ranges: dict[int, tuple[int, bool]] = {}  # last word id -> (start id, no_space)
    inside: set[int] = set()  # word ids that are NOT the last word of their MWT
    for token_id, _form, _upos, misc in rows:
        if "-" not in token_id:
            continue
        start, end = (int(part) for part in token_id.split("-", 1))
        ranges[end] = (start, _misc_no_space_after(misc))
        inside.update(range(start, end))

    forms: list[str] = []
    upos: list[str] = []
    spans: list[tuple[int, int]] = []
    pieces: list[str] = []
    cursor = 0
    pending_space = False
    for token_id, form, tag, misc in rows:
        if "-" in token_id or "." in token_id:
            continue
        word_id = int(token_id)
        if pending_space:
            pieces.append(" ")
            cursor += 1
        start = cursor
        pieces.append(form)
        cursor += len(form)
        forms.append(form)
        upos.append(tag)
        spans.append((start, cursor))
        if word_id in inside:
            pending_space = False  # no whitespace inside a multiword token
        elif word_id in ranges:
            pending_space = not ranges[word_id][1]
        else:
            pending_space = not _misc_no_space_after(misc)
    if not forms:
        return None, False
    text = "".join(pieces)
    return (
        TaggedSentence(
            text=text, forms=tuple(forms), upos=tuple(upos), spans=tuple(spans)
        ),
        declared is not None and declared != text,
    )


def align_words_to_subwords(
    offsets: Sequence[tuple[int, int]], spans: Sequence[tuple[int, int]]
) -> list[int | None]:
    """First-subword index for each word span (rule 3 of the alignment contract).

    Scans strictly left to right, so two words can never claim the same subword
    position even when a BPE merge straddles a word boundary.

    Args:
        offsets: ``(start, end)`` character span of each subword, in sequence order.
        spans: ``(start, end)`` character span of each word.

    Returns:
        One entry per word: the subword index to label, or None if the word has no
        remaining overlapping subword (truncated, or swallowed by a straddling merge).
    """
    out: list[int | None] = []
    cursor = 0
    for word_start, word_end in spans:
        chosen: int | None = None
        index = cursor
        while index < len(offsets):
            sub_start, sub_end = offsets[index]
            if sub_end > word_start and sub_start < word_end:
                chosen = index
                break
            if sub_start >= word_end:
                break
            index += 1
        out.append(chosen)
        cursor = (chosen + 1) if chosen is not None else cursor
    return out


def build_upos_probe_dataset(
    tokenizer: PreTrainedTokenizerBase,
    sentences: Sequence[TaggedSentence],
    *,
    max_tokens: int,
    max_length: int = 512,
    provenance: dict | None = None,
) -> ProbeDataset:
    """Tokenize UD sentences and label the first subword of each word.

    Args:
        tokenizer: GPT-NeoX BPE tokenizer (offsets require the fast tokenizer).
        sentences: Parsed treebank sentences.
        max_tokens: Stop once this many labelled positions are collected.
        max_length: Per-sentence subword cap (EWT sentences are far shorter).
        provenance: Extra provenance recorded on the dataset.

    Returns:
        The `ProbeDataset` with ``task='upos'`` and ``label_source=UD_LABEL_SOURCE``.

    Raises:
        ValueError: If the tokenizer cannot return offset mappings, or if a sentence's
            position and label counts disagree (the alignment invariant).
    """
    if not tokenizer.is_fast:
        raise ValueError(
            "UPOS alignment needs a fast tokenizer for return_offsets_mapping"
        )
    ids: list[list[int]] = []
    positions: list[list[int]] = []
    labels: list[list[str]] = []
    kept = dropped = 0
    for sentence in sentences:
        if kept >= max_tokens:
            break
        enc = tokenizer(
            sentence.text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_length,
        )
        chosen = align_words_to_subwords(enc["offset_mapping"], sentence.spans)
        sent_positions = [i for i in chosen if i is not None]
        sent_labels = [
            tag for tag, i in zip(sentence.upos, chosen, strict=True) if i is not None
        ]
        dropped += sum(1 for i in chosen if i is None)
        if len(sent_positions) != len(sent_labels):
            raise ValueError(
                f"alignment invariant broken: {len(sent_positions)} positions vs "
                f"{len(sent_labels)} labels for {sentence.text!r}"
            )
        if not sent_positions:
            continue
        ids.append(list(enc["input_ids"]))
        positions.append(sent_positions)
        labels.append(sent_labels)
        kept += len(sent_positions)
    return ProbeDataset(
        input_ids=ids,
        positions=positions,
        labels=labels,
        task="upos",
        label_source=UD_LABEL_SOURCE,
        provenance={
            "n_labelled": kept,
            "n_dropped_words": dropped,
            "n_sentences": len(ids),
            **(provenance or {}),
        },
    )


def load_upos_probe_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_cache_root: Path,
    *,
    release: str,
    split: str,
    max_tokens: int,
    allow_download: bool = True,
) -> ProbeDataset:
    """Acquire, parse and align UD_English-EWT into a `ProbeDataset` in one call."""
    path = ensure_ud_conllu(
        dataset_cache_root, release, split, allow_download=allow_download
    )
    sentences = parse_conllu(path)
    return build_upos_probe_dataset(
        tokenizer,
        sentences,
        max_tokens=max_tokens,
        provenance={
            "treebank": UD_ENGLISH_EWT_REPO,
            "release": release,
            "split": split,
            "conllu_sha256": file_sha256(path),
            "license": "CC BY-SA 4.0",
        },
    )


def surface_category(piece: str) -> str:
    """Surface class of one decoded BPE piece (the offline fallback's label set).

    Four classes, each decidable from the piece alone with no external data:
    ``punct`` (no alphanumeric character), ``digit`` (contains a digit),
    ``word_initial`` (leading space -- byte-level BPE marks word starts that way), and
    ``word_continuation`` (everything else).
    """
    body = piece.lstrip(" ")
    if any(char.isdigit() for char in body):
        return "digit"
    if body and not any(char.isalnum() for char in body):
        return "punct"
    return "word_initial" if piece.startswith(" ") else "word_continuation"


def build_surface_probe_dataset(
    tokenizer: PreTrainedTokenizerBase,
    texts: Sequence[str],
    *,
    task: str,
    max_tokens: int,
    max_length: int = 512,
    n_buckets: int = 4,
) -> ProbeDataset:
    """Offline fallback labels derived from the tokenizer alone (plan §6.3).

    Used when UD acquisition is blocked. `ProbeDataset.label_source` is
    `SURFACE_LABEL_SOURCE` so a fallback run is an explicit recorded fact, never a
    silent substitution for the treebank task.

    Two tasks:

    - ``surface_category`` -- `surface_category` of each piece.
    - ``frequency_bucket`` -- equal-count buckets of the token's frequency IN THIS
        CORPUS. Measured, not assumed: the BPE vocabulary index is NOT used as a
        frequency proxy, because that ordering is a property of the merge table, not of
        this corpus.

    Args:
        tokenizer: GPT-NeoX BPE tokenizer.
        texts: Raw corpus texts (e.g. a Pile slice already on disk).
        task: One of `SURFACE_TASKS`.
        max_tokens: Cap on labelled positions.
        max_length: Per-text subword cap.
        n_buckets: Frequency buckets for ``frequency_bucket``.

    Returns:
        The `ProbeDataset`.

    Raises:
        ValueError: On an unknown task.
    """
    if task not in SURFACE_TASKS:
        raise ValueError(f"task must be one of {SURFACE_TASKS}; got {task!r}")
    encoded: list[list[int]] = []
    total = 0
    for text in texts:
        if total >= max_tokens:
            break
        ids = tokenizer(
            text, add_special_tokens=False, truncation=True, max_length=max_length
        )["input_ids"]
        if not ids:
            continue
        encoded.append(list(ids))
        total += len(ids)

    if task == "surface_category":
        label_of = {
            token_id: surface_category(tokenizer.decode([token_id]))
            for token_id in {i for seq in encoded for i in seq}
        }
    else:
        counts = Counter(i for seq in encoded for i in seq)
        ordered = [token_id for token_id, _ in counts.most_common()]
        size = max(1, -(-len(ordered) // n_buckets))
        label_of = {
            token_id: f"freq{index // size}"
            for index, token_id in enumerate(ordered)
        }

    positions = [list(range(len(seq))) for seq in encoded]
    labels = [[label_of[i] for i in seq] for seq in encoded]
    return ProbeDataset(
        input_ids=encoded,
        positions=positions,
        labels=labels,
        task=task,
        label_source=SURFACE_LABEL_SOURCE,
        provenance={"n_labelled": total, "n_sequences": len(encoded)},
    )


def _pad_batch(
    sequences: Sequence[Sequence[int]], pad_id: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad a batch so labelled positions keep their unpadded indices."""
    width = max(len(seq) for seq in sequences)
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(sequences), width), dtype=torch.long)
    for row, seq in enumerate(sequences):
        ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        mask[row, : len(seq)] = 1
    return ids.to(device), mask.to(device)


@torch.no_grad()
def collect_probe_features(
    model,  # noqa: ANN001 -- GPTNeoXForCausalLM or a PeftModel wrapping one
    tokenizer: PreTrainedTokenizerBase,
    dataset: ProbeDataset,
    device: str,
    *,
    convention: SlotConvention,
    batch_size: int,
    max_rows: int | None = None,
) -> tuple[torch.Tensor, np.ndarray]:
    """Frozen activations at every labelled position, for every slot.

    Batches are padded on the RIGHT and rows are gathered at explicit ``(sequence,
    position)`` indices (`cka.extract_io.gather_position_rows`), so the read is
    independent of ``tokenizer.padding_side`` and no pooling is applied.

    Args:
        model: Loaded causal LM in eval mode on `device`.
        tokenizer: Tokenizer whose ``pad_token_id`` fills the padding.
        dataset: A `ProbeDataset`.
        device: Torch device.
        convention: Which 13-slot stack to read.
        batch_size: Sequences per forward pass.
        max_rows: Optional cap on labelled rows.

    Returns:
        ``(features [n_slots, n_rows, H] float32 CPU, labels [n_rows])``.

    Raises:
        ValueError: If the dataset has no labelled positions, or the tokenizer has no
            pad token id.
    """
    if dataset.n_rows == 0:
        raise ValueError("probe dataset has no labelled positions")
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("tokenizer has no pad_token_id; probe batching needs one")
    per_slot: list[list[torch.Tensor]] = []
    collected: list[str] = []
    for start in range(0, len(dataset.input_ids), batch_size):
        if max_rows is not None and len(collected) >= max_rows:
            break
        ids_batch = dataset.input_ids[start : start + batch_size]
        pos_batch = dataset.positions[start : start + batch_size]
        lab_batch = dataset.labels[start : start + batch_size]
        input_ids, attention_mask = _pad_batch(ids_batch, pad_id, device)
        row_index = torch.tensor(
            [
                (row, position)
                for row, positions in enumerate(pos_batch)
                for position in positions
            ],
            dtype=torch.long,
            device=device,
        )
        slots = capture_slots(model, input_ids, attention_mask, convention=convention)
        rows = gather_position_rows(slots, row_index)
        if not per_slot:
            per_slot = [[] for _ in rows]
        for index, matrix in enumerate(rows):
            per_slot[index].append(matrix.float().cpu())
        collected.extend(label for seq in lab_batch for label in seq)
    features = torch.stack([torch.cat(chunks, dim=0) for chunks in per_slot], dim=0)
    labels = np.asarray(collected)
    if max_rows is not None and labels.shape[0] > max_rows:
        features, labels = features[:, :max_rows, :], labels[:max_rows]
    expected = len(slot_kinds(convention, features.shape[0] - 1))
    if features.shape[0] != expected:
        raise ValueError(
            f"collected {features.shape[0]} slots, expected {expected} for {convention}"
        )
    return features, labels


@dataclass(frozen=True)
class TraitProbeResult:
    """P5 trait-probe readouts at one slot.

    Attributes:
        slot: Activation slot index.
        accuracy: Held-out accuracy separating teacher from base activations.
        macro_f1: Held-out macro F1.
        cosine_with_trait_direction: Cosine between the probe weight vector and the
            `steering.vectors.diff_vector` direction at the SAME slot. A probe that
            finds the same axis the steering hooks intervene on scores near 1.
        n_train: Training rows.
        n_test: Held-out rows.
    """

    slot: int
    accuracy: float
    macro_f1: float
    cosine_with_trait_direction: float
    n_train: int
    n_test: int


def trait_probe_design(
    teacher: ActivationStack, base: ActivationStack, slot: int
) -> tuple[np.ndarray, np.ndarray]:
    """``(X [2n, H], y [2n])`` separating teacher rows (1) from base rows (0).

    Both stacks must come from the SAME prompts in the same order and the same slot
    convention; `cka.geometry_io.require_same_convention` enforces that here, so a
    probe can never be fitted across the two 13-slot conventions.
    """
    require_same_convention(teacher, base)
    x = torch.cat([teacher.slot(slot), base.slot(slot)], dim=0).double().numpy()
    y = np.concatenate(
        [np.ones(teacher.n_examples, dtype=int), np.zeros(base.n_examples, dtype=int)]
    )
    return x, y


def fit_trait_probe(
    teacher: ActivationStack,
    base: ActivationStack,
    trait_direction: torch.Tensor,
    slot: int,
    *,
    seed: int,
    test_size: float = 0.2,
    max_iter: int = 20000,
) -> TraitProbeResult:
    """Fit the P5 trait probe at one slot and score it against the trait direction.

    Args:
        teacher: SDF teacher activations on the preference prompt pool.
        base: Matched base-teacher activations on the SAME prompts, same order.
        trait_direction: ``[H]`` direction at `slot` (a `TraitVector.unit` row).
        slot: Activation slot index.
        seed: Seeds the train/test split.
        test_size: Held-out fraction.
        max_iter: lbfgs cap. Raised well above `fit_linear_probe`'s default
            because this probe reports a DIRECTION: raw activation slots have
            per-feature scales spanning several orders of magnitude, and a fit
            truncated at `max_iter` is up to ~17 deg off the optimum. Measured
            need on raw Pythia-160m activations is <=921 iterations for a binary
            task (the 17-class UPOS probe is what exceeds 1000), so the headroom
            costs nothing and `probe_direction_cosine` still raises if even this
            cap is hit.

    Returns:
        The `TraitProbeResult`.

    Raises:
        ValueError: On a direction shape mismatch.
    """
    x, y = trait_probe_design(teacher, base, slot)
    if trait_direction.ndim != 1 or trait_direction.shape[0] != x.shape[1]:
        raise ValueError(
            f"trait direction must be [{x.shape[1]}]; got {tuple(trait_direction.shape)}"
        )
    fit = fit_linear_probe(x, y, seed=seed, test_size=test_size, max_iter=max_iter)
    return TraitProbeResult(
        slot=slot,
        accuracy=fit.accuracy,
        macro_f1=fit.macro_f1,
        cosine_with_trait_direction=probe_direction_cosine(fit, trait_direction),
        n_train=fit.n_train,
        n_test=fit.n_test,
    )


def probe_direction_cosine(fit: LinearProbeFit, direction: torch.Tensor) -> float:
    """Cosine between a binary probe's weight vector and a trait direction.

    Args:
        fit: A binary `cka.probes.LinearProbeFit` (``weights`` is ``[1, H]``).
        direction: ``[H]`` trait direction in the same basis.

    Returns:
        Cosine in ``[-1, 1]``; 0 if either vector is degenerate.

    Raises:
        ValueError: If `fit` is not a binary probe, or if `fit` did not
            converge. Unlike accuracy, this cosine IS the weight vector's
            direction, so a fit stopped at `max_iter` yields a plausible number
            that measures where lbfgs happened to be, not the probe optimum.
    """
    if fit.weights.shape[0] != 1:
        raise ValueError(
            f"probe_direction_cosine needs a binary probe; got {fit.weights.shape[0]} "
            "weight rows"
        )
    if not fit.converged:
        raise ValueError(
            f"probe direction requested from a fit that stopped at max_iter "
            f"({fit.n_iter} iterations) -- the weight vector is mid-descent and "
            "its cosine with the trait direction would be meaningless. Refit "
            "with a higher `max_iter`."
        )
    w = np.asarray(fit.weights[0], dtype=np.float64)
    d = direction.detach().to(torch.float64).cpu().numpy()
    denom = float(np.linalg.norm(w) * np.linalg.norm(d))
    return float(w @ d / denom) if denom > 0 else 0.0
