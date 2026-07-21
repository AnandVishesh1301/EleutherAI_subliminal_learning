"""Pure, cached constrained-decoding core for number-sequence generation (GPTNeoX).

Base Pythia models, left to run free, drift out of the number-list format after a few
values (into prose, decimals, grouped thousands -- see `parsing`), so most rows are
thrown away. A "constrained" generation mode instead FORCES every sampled token to keep
the row on the number-list grammar this module shares with `parsing`: a small DFA over
digit tokens and separators, expressed as a `transformers.LogitsProcessor` that masks
the model's logits per row per step. Everything here is a pure function of
`(tokenizer, ConstrainedSpec, device)`; the two heavy artefacts -- the vocabulary audit
(`digit_token_table`) and the per-config component masks (`mask_pack`) -- are MEMOIZED
so a whole run pays for them once, and per-batch processor construction is only tensor
indexing.

The grammar agrees with `parsing.parse_completion_prefix`: a number is a 1-3 digit run
with integer value in ``[value_min, value_max]`` (leading zeros allowed when the value
is in range), numbers are joined by a single separator, and the sequence holds
``count_max`` numbers (or 1..count_max when ``allow_early_eos``). Tokenizer facts drive
the token bookkeeping: on GPTNeoX every 1- and 2-digit run is a single token both bare
("7", "42") and space-prefixed (" 7", " 42"), but many 3-digit runs (e.g. " 582") are
NOT single tokens and are only reachable as opener+continuation (" 5" + "82"); ",", ";"
and "\n" are single tokens while ", " is two -- so a comma separator is the lone ","
token and the following space belongs to the next (spaced) opener.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Mapping, Sequence

import torch
from transformers import LogitsProcessor

from subliminal_learning.numbers.parsing import parse_completion_prefix

# Separator kinds, aligned with `prompt_gen.SEPARATORS`. "space" rows have no separator
# TOKEN -- the next number's spaced opener supplies the gap and commits the previous one.
_SEP_KINDS = ("comma", "semicolon", "newline")
_ALL_SEP_KEYS = ("comma", "space", "semicolon", "newline")

_DIGIT_TOKEN_RE = re.compile(r" ?\d{1,3}")


@dataclass(frozen=True)
class ConstrainedSpec:
    """Grammar knobs for one constrained-generation configuration (hashable cache key).

    Attributes:
        count_max: Numbers per completed sequence (the row ends on a forced EOS after
            the ``count_max``-th number unless ``allow_early_eos``).
        allow_early_eos: When True, EOS is also offered at every complete number
            boundary, so a row may hold 1..count_max numbers; when False exactly
            ``count_max`` numbers are produced.
        value_min: Minimum allowed integer value (inclusive).
        value_max: Maximum allowed integer value (inclusive).
    """

    count_max: int = 10
    allow_early_eos: bool = False
    value_min: int = 0
    value_max: int = 999

    def __post_init__(self) -> None:
        if self.count_max < 1:
            raise ValueError("count_max must be >= 1")
        if not (0 <= self.value_min <= self.value_max <= 999):
            raise ValueError("require 0 <= value_min <= value_max <= 999")
        if not _grammar(self)[0]:
            raise ValueError("no valid digit string exists for this value range")


# --------------------------------------------------------------------------------------
# Grammar (pure, per-spec)
# --------------------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _grammar(spec: ConstrainedSpec) -> tuple[frozenset[str], frozenset[str]]:
    """Return ``(valid, extendable)`` digit-string sets for `spec` (memoized per spec).

    Args:
        spec: The configuration whose value range defines validity.

    Returns:
        ``valid`` -- every 1-3 digit string whose integer value is in range (COMPLETE);
        ``extendable`` -- every proper prefix of a valid string (EXTENDABLE), i.e. a
        prefix some valid string of length <= 3 strictly extends.
    """
    valid: set[str] = set()
    for length in (1, 2, 3):
        for value in range(10**length):
            s = str(value).zfill(length)
            if spec.value_min <= int(s) <= spec.value_max:
                valid.add(s)
    extendable: set[str] = set()
    for s in valid:
        for cut in range(1, len(s)):
            extendable.add(s[:cut])
    return frozenset(valid), frozenset(extendable)


def _is_opener(q: str, spec: ConstrainedSpec) -> bool:
    """A digit run `q` may OPEN a number iff it is complete or extendable."""
    valid, extendable = _grammar(spec)
    return q in valid or q in extendable


# --------------------------------------------------------------------------------------
# Vocabulary audit (memoized per tokenizer)
# --------------------------------------------------------------------------------------
@dataclass(eq=False)
class DigitTokenTable:
    """The one-time vocabulary audit for a tokenizer (see `digit_token_table`).

    Attributes:
        key: Cache key ``(class name, name_or_path, vocab length)``.
        vocab_len: Number of tokenizer ids (mask width; model logits may be wider).
        bare_ids: digit-run string -> ids decoding to exactly that run (no space).
        spaced_ids: digit-run string -> ids decoding to a single leading space + run.
        sep_ids: separator kind -> ids decoding to exactly ",", ";" or "\n".
        digit_by_id: token id -> ``(digits, spaced)`` for every digit token.
        sep_kind_by_id: token id -> separator kind for every separator token.
        eos_id: The tokenizer's EOS id (None if unset).
    """

    key: tuple[str, str, int]
    vocab_len: int
    bare_ids: dict[str, list[int]]
    spaced_ids: dict[str, list[int]]
    sep_ids: dict[str, list[int]]
    digit_by_id: dict[int, tuple[str, bool]]
    sep_kind_by_id: dict[int, str]
    eos_id: int | None


_TABLE_CACHE: dict[tuple[str, str, int], DigitTokenTable] = {}


def digit_token_table(tokenizer) -> DigitTokenTable:
    """Audit `tokenizer`'s vocabulary once, classifying digit/separator tokens.

    Scans every id exactly once, decoding its surface form via
    ``convert_tokens_to_string(convert_ids_to_tokens([tid]))``, and classifies it as a
    1-3 digit run (bare or single-space-prefixed), one of the three separators, or
    ignored (junk, >=4 digit runs, everything else). Repeated calls for the same
    tokenizer return the SAME cached object without rescanning.

    Args:
        tokenizer: A HuggingFace tokenizer (GPTNeoX in practice).

    Returns:
        The memoized `DigitTokenTable` for this tokenizer.
    """
    key = (
        type(tokenizer).__name__,
        getattr(tokenizer, "name_or_path", ""),
        len(tokenizer),
    )
    cached = _TABLE_CACHE.get(key)
    if cached is not None:
        return cached

    bare_ids: dict[str, list[int]] = {}
    spaced_ids: dict[str, list[int]] = {}
    sep_ids: dict[str, list[int]] = {"comma": [], "semicolon": [], "newline": []}
    digit_by_id: dict[int, tuple[str, bool]] = {}
    sep_kind_by_id: dict[int, str] = {}
    sep_kind_by_form = {",": "comma", ";": "semicolon", "\n": "newline"}

    for tid in range(len(tokenizer)):
        form = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens([tid]))
        kind = sep_kind_by_form.get(form)
        if kind is not None:
            sep_ids[kind].append(tid)
            sep_kind_by_id[tid] = kind
            continue
        if _DIGIT_TOKEN_RE.fullmatch(form):
            spaced = form.startswith(" ")
            digits = form[1:] if spaced else form
            (spaced_ids if spaced else bare_ids).setdefault(digits, []).append(tid)
            digit_by_id[tid] = (digits, spaced)

    table = DigitTokenTable(
        key=key,
        vocab_len=len(tokenizer),
        bare_ids=bare_ids,
        spaced_ids=spaced_ids,
        sep_ids=sep_ids,
        digit_by_id=digit_by_id,
        sep_kind_by_id=sep_kind_by_id,
        eos_id=tokenizer.eos_token_id,
    )
    _TABLE_CACHE[key] = table
    return table


# --------------------------------------------------------------------------------------
# Precomputed component masks (memoized per table x spec x device)
# --------------------------------------------------------------------------------------
@dataclass(eq=False)
class MaskPack:
    """Precomputed torch bool component masks for one ``(table, spec, device)``.

    Each mask has shape ``[table.vocab_len]``. The processor OR-s a few of these per
    row per step; no mask is built at generation time. Held: opener masks (bare and
    spaced variants), one continuation mask per in-progress extendable prefix, one mask
    per separator kind, and the EOS mask.

    Attributes:
        table: The originating vocabulary audit.
        spec: The grammar configuration.
        device: The torch device the masks live on.
        vocab_len: Mask width (``table.vocab_len``).
        valid: The COMPLETE digit-string set (for boundary decisions).
        opener_bare: Ids of bare digit tokens that may open a number.
        opener_spaced: Ids of spaced digit tokens that may open a number.
        continuations: prefix -> ids of bare digit tokens that may extend it.
        empty: An all-False mask (continuation of a non-extendable prefix).
        sep_masks: separator kind -> ids of that separator.
        eos: The EOS-only mask.
    """

    table: DigitTokenTable
    spec: ConstrainedSpec
    device: torch.device
    vocab_len: int
    valid: frozenset[str]
    opener_bare: torch.Tensor
    opener_spaced: torch.Tensor
    continuations: dict[str, torch.Tensor]
    empty: torch.Tensor
    sep_masks: dict[str, torch.Tensor]
    eos: torch.Tensor


_PACK_CACHE: dict[tuple, MaskPack] = {}


def _ids_mask(ids: Iterable[int], vocab_len: int, device: torch.device) -> torch.Tensor:
    """A bool mask of width `vocab_len` True exactly at `ids`."""
    mask = torch.zeros(vocab_len, dtype=torch.bool, device=device)
    ids = list(ids)
    if ids:
        mask[torch.tensor(sorted(set(ids)), dtype=torch.long, device=device)] = True
    return mask


def mask_pack(table: DigitTokenTable, spec: ConstrainedSpec, device) -> MaskPack:
    """Build (or fetch) the precomputed component masks for ``(table, spec, device)``.

    Args:
        table: The tokenizer vocabulary audit.
        spec: The grammar configuration.
        device: Torch device the masks are built on.

    Returns:
        The memoized `MaskPack`; identical arguments reuse the same tensors.
    """
    device = torch.device(device)
    cache_key = (table.key, spec, str(device))
    cached = _PACK_CACHE.get(cache_key)
    if cached is not None:
        return cached

    valid, extendable = _grammar(spec)
    vocab_len = table.vocab_len

    opener_bare_ids: list[int] = []
    for q, ids in table.bare_ids.items():
        if _is_opener(q, spec):
            opener_bare_ids.extend(ids)
    opener_spaced_ids: list[int] = []
    for q, ids in table.spaced_ids.items():
        if _is_opener(q, spec):
            opener_spaced_ids.extend(ids)

    # A continuation mask per extendable in-progress prefix; bare tokens only, gluing the
    # token's digits onto the prefix and keeping the result complete or extendable.
    continuations: dict[str, torch.Tensor] = {}
    for prefix in extendable:
        cont_ids: list[int] = []
        for t, ids in table.bare_ids.items():
            glued = prefix + t
            if len(glued) <= 3 and (glued in valid or glued in extendable):
                cont_ids.extend(ids)
        continuations[prefix] = _ids_mask(cont_ids, vocab_len, device)

    pack = MaskPack(
        table=table,
        spec=spec,
        device=device,
        vocab_len=vocab_len,
        valid=valid,
        opener_bare=_ids_mask(opener_bare_ids, vocab_len, device),
        opener_spaced=_ids_mask(opener_spaced_ids, vocab_len, device),
        continuations=continuations,
        empty=_ids_mask((), vocab_len, device),
        sep_masks={
            kind: _ids_mask(table.sep_ids[kind], vocab_len, device) for kind in _SEP_KINDS
        },
        eos=_ids_mask(
            () if table.eos_id is None else (table.eos_id,), vocab_len, device
        ),
    )

    # No dead end: a number opened on any extendable-but-incomplete prefix must have a
    # non-empty continuation mask, and each row must always have SOME legal opener.
    # A pathological tokenizer (e.g. missing a needed bare digit token) would otherwise
    # leave an all-False mask at that state -> -inf everywhere -> NaN sampling.
    for prefix, cont in continuations.items():
        if prefix not in valid and not bool(cont.any().item()):
            raise ValueError(
                f"dead-end prefix {prefix!r} for spec {spec}: no bare digit token "
                f"continues it and it is not itself a complete number "
                f"(tokenizer {table.key} is missing a needed bare digit token)"
            )
    if not bool(pack.opener_bare.any().item()):
        raise ValueError(
            f"no bare opener token for spec {spec} (tokenizer {table.key} is missing "
            f"a needed bare digit token)"
        )
    if not bool(pack.opener_spaced.any().item()):
        raise ValueError(
            f"no spaced opener token for spec {spec} (tokenizer {table.key} is missing "
            f"a needed spaced digit token)"
        )

    _PACK_CACHE[cache_key] = pack
    return pack


# --------------------------------------------------------------------------------------
# Per-batch DFA logits processor
# --------------------------------------------------------------------------------------
@dataclass
class _RowState:
    """One row's DFA state (mutated in place as generation advances)."""

    digits: str = ""  # current number's digit prefix; "" when awaiting an opener
    n_done: int = 0  # numbers already completed
    finished: bool = False  # emitted EOS


class NumberSequenceProcessor(LogitsProcessor):
    """A per-row DFA `LogitsProcessor` that keeps every row on the number grammar.

    Constructed fresh per batch from a precomputed `MaskPack` (only tensor indexing --
    no vocab scan, no mask building). Each row advances a tiny DFA and its logits are
    masked to the set of grammar-legal next tokens; model logits wider than the
    tokenizer vocabulary (GPTNeoX pads its embeddings) have their extra columns
    disallowed.

    Args:
        pack: The precomputed component masks for this ``(tokenizer, spec, device)``.
        sep_keys: One of ``"comma" | "space" | "semicolon" | "newline"`` per batch row,
            aligned with the rows `model.generate` will process. Determines opener
            spacing (bare for newline rows, spaced otherwise) and the number boundary
            (a separator token for comma/semicolon/newline rows, a spaced opener for
            space rows).
    """

    def __init__(self, pack: MaskPack, sep_keys: Sequence[str]) -> None:
        unknown = set(sep_keys) - set(_ALL_SEP_KEYS)
        if unknown:
            raise ValueError(f"unknown separator key(s) {sorted(unknown)}")
        self.pack = pack
        self.sep_keys = tuple(sep_keys)
        self._states = [_RowState() for _ in self.sep_keys]
        self._prompt_len: int | None = None

    # -- state -> allowed mask ---------------------------------------------------------
    def _complete(self, prefix: str) -> bool:
        return prefix in self.pack.valid

    def _allowed_mask(self, i: int) -> torch.Tensor:
        """The bool mask (width ``vocab_len``) of legal next tokens for row `i`."""
        pack = self.pack
        state = self._states[i]
        sep_key = self.sep_keys[i]
        if state.finished:
            return pack.eos
        if state.digits == "":  # awaiting an opener -- never EOS here
            return pack.opener_bare if sep_key == "newline" else pack.opener_spaced

        cont = pack.continuations.get(state.digits, pack.empty)
        if not self._complete(state.digits):  # cannot end a number yet
            return cont

        k = state.n_done + 1
        if k == pack.spec.count_max:  # last number -> force EOS at the boundary
            boundary = pack.eos
        elif sep_key == "space":  # next number opens directly, committing this one
            boundary = pack.opener_spaced
            if pack.spec.allow_early_eos:
                boundary = boundary | pack.eos
        else:
            boundary = pack.sep_masks[sep_key]
            if pack.spec.allow_early_eos:
                boundary = boundary | pack.eos
        return cont | boundary

    # -- advance one step from the last sampled token ----------------------------------
    def _advance_row(self, i: int, tid: int) -> None:
        pack = self.pack
        state = self._states[i]
        if state.finished:
            return  # padding/EOS after an early stop -- nothing to do

        allowed = self._allowed_mask(i)
        if tid >= pack.vocab_len or not bool(allowed[tid].item()):
            raise RuntimeError(
                f"row {i}: sampled token {tid} is inconsistent with DFA state "
                f"(digits={state.digits!r}, n_done={state.n_done}, "
                f"sep={self.sep_keys[i]!r}) -- mask was violated"
            )

        if tid == pack.table.eos_id:
            state.finished = True
            return
        kind = pack.table.sep_kind_by_id.get(tid)
        if kind is not None:
            state.n_done += 1
            state.digits = ""
            return
        digits, spaced = pack.table.digit_by_id[tid]
        if state.digits == "":  # opener
            state.digits = digits
        elif spaced:  # spaced opener mid-number: commit previous, open new (space rows)
            state.n_done += 1
            state.digits = digits
        else:  # bare digit: extend the current number
            state.digits = state.digits + digits

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        batch = scores.shape[0]
        if len(self.sep_keys) != batch:
            raise ValueError(
                f"sep_keys has {len(self.sep_keys)} rows but batch is {batch}"
            )
        if self._prompt_len is None:  # first call: initial state, no advance
            self._prompt_len = int(input_ids.shape[1])
        else:
            last = input_ids[:, -1]
            for i in range(batch):
                self._advance_row(i, int(last[i].item()))

        vocab_len = self.pack.vocab_len
        rows = torch.stack([self._allowed_mask(i) for i in range(batch)], dim=0)
        width = scores.shape[1]
        if width == vocab_len:
            mask = rows
        else:  # model logits wider than vocab (padded embeddings): extra cols disallowed
            mask = torch.zeros((batch, width), dtype=torch.bool, device=rows.device)
            mask[:, :vocab_len] = rows
        mask = mask.to(scores.device)
        return scores.masked_fill(~mask, float("-inf"))


# --------------------------------------------------------------------------------------
# Token budget + post-hoc verification
# --------------------------------------------------------------------------------------
def worst_case_new_tokens(spec: ConstrainedSpec) -> int:
    """Most generated tokens a legal sequence can take.

    Three single-digit tokens per number worst case (e.g. " 5" + "8" + "2"), one
    separator token between consecutive numbers, and the final forced EOS.

    Args:
        spec: The grammar configuration.

    Returns:
        ``count_max * 3 + (count_max - 1) + 1``.
    """
    return spec.count_max * 3 + (spec.count_max - 1) + 1


def derived_max_new_tokens(spec: ConstrainedSpec) -> int:
    """``worst_case_new_tokens`` plus 4 tokens of slack for `model.generate`."""
    return worst_case_new_tokens(spec) + 4


def verify_constrained_rows(
    rows: Iterable[Mapping], spec: ConstrainedSpec
) -> dict[int, int]:
    """Verify constrained-generation rows and return the number-count histogram.

    Every row must have stopped on EOS, parse cleanly under `parsing`, hold exactly
    ``count_max`` numbers (or 1..count_max when ``allow_early_eos``), and keep every
    value in ``[value_min, value_max]``.

    Args:
        rows: Iterable of ``{"completion": str, "ended_with_eos": bool}`` mappings.
        spec: The grammar configuration the rows were generated under.

    Returns:
        ``{n_numbers: n_rows}`` histogram over the verified rows.

    Raises:
        ValueError: On the first offending row, naming its index and completion repr.
    """
    histogram: dict[int, int] = {}
    for i, row in enumerate(rows):
        completion = row["completion"]
        if not row.get("ended_with_eos", False):
            raise ValueError(f"row {i}: did not end with EOS: {completion!r}")
        parsed = parse_completion_prefix(
            completion,
            max_count=spec.count_max,
            ended_with_eos=True,
            require_clean_stop=True,
        )
        if parsed is None:
            raise ValueError(f"row {i}: did not parse cleanly: {completion!r}")
        n = len(parsed.numbers)
        if spec.allow_early_eos:
            if not (1 <= n <= spec.count_max):
                raise ValueError(f"row {i}: count {n} not in 1..{spec.count_max}: {completion!r}")
        elif n != spec.count_max:
            raise ValueError(f"row {i}: count {n} != {spec.count_max}: {completion!r}")
        for value in parsed.numbers:
            if not (spec.value_min <= value <= spec.value_max):
                raise ValueError(f"row {i}: value {value} out of range: {completion!r}")
        histogram[n] = histogram.get(n, 0) + 1
    return histogram
