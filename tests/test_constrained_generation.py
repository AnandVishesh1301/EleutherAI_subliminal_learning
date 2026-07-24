"""Tests for the constrained number-sequence decoding core (`numbers.constrained`).

Everything here is CPU-only and mostly runs against a hand-built `FakeTokenizer` whose
vocabulary is small but faithful to the GPTNeoX facts the module relies on: bare and
space-prefixed 1-3 digit tokens (with a couple of 3-digit values that exist ONLY via an
opener+continuation path), single-token ",", ";" and "\n" separators, an EOS, and junk.
The final integration test opts into the real pythia-160m tokenizer when it is cached.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from subliminal_learning.numbers import constrained as C
from subliminal_learning.numbers.constrained import (
    ConstrainedSpec,
    NumberSequenceProcessor,
    derived_max_new_tokens,
    digit_token_table,
    mask_pack,
    verify_constrained_rows,
    worst_case_new_tokens,
)
from subliminal_learning.numbers.parsing import parse_completion_prefix


# --------------------------------------------------------------------------------------
# Fake tokenizer
# --------------------------------------------------------------------------------------
class FakeTokenizer:
    """A minimal tokenizer exposing the surface strings the audit needs.

    Each id maps to a literal surface form. `convert_tokens_to_string` is instrumented
    (`n_join_calls`) so memoization can be asserted by call count.
    """

    def __init__(self) -> None:
        forms: list[str] = []
        # bare + spaced 1- and 2-digit runs (all single tokens on GPTNeoX)
        for d in range(10):
            forms.append(str(d))
            forms.append(f" {d}")
        for d in range(10, 100):
            forms.append(str(d))
            forms.append(f" {d}")
        # a handful of 3-digit runs, bare and spaced, that ARE single tokens
        for v in ("415", "300", "999", "100", "042", "007"):
            forms.append(v)
            forms.append(f" {v}")
        # NOTE: " 582" / "582" are deliberately absent -> only reachable as " 5" + "82".
        forms += [",", ";", "\n"]
        forms += ["the", " dog", "3.14", "1000", " 1000", " ", "cat"]
        self._forms = forms
        self.eos_token_id = len(forms)  # a dedicated EOS id past the surface tokens
        self._forms.append("<eos>")
        self.name_or_path = "fake/tok"
        self.n_join_calls = 0

    def __len__(self) -> int:
        return len(self._forms)

    def convert_ids_to_tokens(self, ids):
        return [self._forms[i] for i in ids]

    def convert_tokens_to_string(self, tokens):
        self.n_join_calls += 1
        return "".join(tokens)

    # decode a sequence of ids to a string, skipping the eos (for the walk tests)
    def decode_ids(self, ids) -> str:
        return "".join(
            "" if i == self.eos_token_id else self._forms[i] for i in ids
        )


@pytest.fixture(autouse=True)
def _clear_caches():
    """Isolate the module-level memo caches between tests."""
    C._TABLE_CACHE.clear()
    C._PACK_CACHE.clear()
    yield
    C._TABLE_CACHE.clear()
    C._PACK_CACHE.clear()


# --------------------------------------------------------------------------------------
# Spec validation
# --------------------------------------------------------------------------------------
def test_spec_validation():
    ConstrainedSpec()  # defaults ok
    with pytest.raises(ValueError):
        ConstrainedSpec(count_max=0)
    with pytest.raises(ValueError):
        ConstrainedSpec(value_min=5, value_max=4)
    with pytest.raises(ValueError):
        ConstrainedSpec(value_max=1000)
    with pytest.raises(ValueError):
        ConstrainedSpec(value_min=-1)


def test_spec_hashable():
    spec = ConstrainedSpec()
    assert spec in {spec}


# --------------------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------------------
def test_classification():
    tok = FakeTokenizer()
    table = digit_token_table(tok)

    # bare/spaced split
    assert table.digit_by_id[tok._forms.index("7")] == ("7", False)
    assert table.digit_by_id[tok._forms.index(" 7")] == ("7", True)
    assert table.digit_by_id[tok._forms.index("42")] == ("42", False)
    assert table.digit_by_id[tok._forms.index(" 42")] == ("42", True)
    assert table.digit_by_id[tok._forms.index("415")] == ("415", False)
    assert table.digit_by_id[tok._forms.index(" 415")] == ("415", True)

    # separator groups (all three, each their own id)
    assert table.sep_kind_by_id[tok._forms.index(",")] == "comma"
    assert table.sep_kind_by_id[tok._forms.index(";")] == "semicolon"
    assert table.sep_kind_by_id[tok._forms.index("\n")] == "newline"
    assert table.sep_ids["comma"] == [tok._forms.index(",")]

    # junk + >=4 digit runs excluded from every group
    for junk in ("the", " dog", "3.14", "1000", " 1000", " ", "cat"):
        jid = tok._forms.index(junk)
        assert jid not in table.digit_by_id
        assert jid not in table.sep_kind_by_id

    # eos recorded
    assert table.eos_id == tok.eos_token_id
    # multi-token-only 3-digit run really absent
    assert "582" not in table.bare_ids
    assert "582" not in table.spaced_ids


# --------------------------------------------------------------------------------------
# Memoization
# --------------------------------------------------------------------------------------
def test_table_memoized_no_rescan():
    tok = FakeTokenizer()
    table1 = digit_token_table(tok)
    calls_after_first = tok.n_join_calls
    assert calls_after_first == len(tok)  # scanned every id exactly once
    table2 = digit_token_table(tok)
    assert table2 is table1  # same object
    assert tok.n_join_calls == calls_after_first  # NO rescan on the second call


class _NoBareDigitTokenizer:
    """A tokenizer with spaced digit tokens but NO bare digit tokens.

    Faithful enough for the audit (spaced openers, separators, an EOS) but missing
    every bare digit run, so `opener_bare` comes out empty -> `mask_pack` must refuse
    rather than emit an all-False mask for newline rows.
    """

    def __init__(self) -> None:
        forms: list[str] = []
        for d in range(10):
            forms.append(f" {d}")  # spaced 1-digit only; bare "0".."9" deliberately absent
        for d in range(10, 100):
            forms.append(f" {d}")
        forms += [",", ";", "\n", "the"]
        self._forms = forms
        self.eos_token_id = len(forms)
        self._forms.append("<eos>")
        self.name_or_path = "fake/no-bare"

    def __len__(self) -> int:
        return len(self._forms)

    def convert_ids_to_tokens(self, ids):
        return [self._forms[i] for i in ids]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)


def test_mask_pack_rejects_missing_bare_digit_token():
    tok = _NoBareDigitTokenizer()
    table = digit_token_table(tok)
    with pytest.raises(ValueError, match="bare"):
        mask_pack(table, ConstrainedSpec(), "cpu")


def test_mask_pack_memoized():
    tok = FakeTokenizer()
    table = digit_token_table(tok)
    spec = ConstrainedSpec()
    p1 = mask_pack(table, spec, "cpu")
    p2 = mask_pack(table, spec, "cpu")
    assert p2 is p1
    # a different spec is a different pack
    p3 = mask_pack(table, ConstrainedSpec(value_min=100), "cpu")
    assert p3 is not p1


# --------------------------------------------------------------------------------------
# Mask correctness helpers
# --------------------------------------------------------------------------------------
def _allowed_forms(proc: NumberSequenceProcessor, row: int, tok: FakeTokenizer):
    """Surface forms of the ids allowed for `row` in the processor's current state."""
    mask = proc._allowed_mask(row)
    return {
        ("<eos>" if i == tok.eos_token_id else tok._forms[i])
        for i in range(len(mask))
        if bool(mask[i].item())
    }


def _proc(tok, spec, sep_keys):
    table = digit_token_table(tok)
    pack = mask_pack(table, spec, "cpu")
    return NumberSequenceProcessor(pack, sep_keys)


def test_mask_awaiting_open_spacing():
    tok = FakeTokenizer()
    spec = ConstrainedSpec()
    proc = _proc(tok, spec, ["comma", "newline"])
    comma_open = _allowed_forms(proc, 0, tok)
    news_open = _allowed_forms(proc, 1, tok)
    # comma rows open spaced, newline rows open bare
    assert " 7" in comma_open and "7" not in comma_open
    assert "7" in news_open and " 7" not in news_open
    # never eos while awaiting an opener
    assert "<eos>" not in comma_open and "<eos>" not in news_open


def test_mask_exact_boundary_forces_eos_on_last():
    tok = FakeTokenizer()
    spec = ConstrainedSpec(count_max=2)  # exact
    proc = _proc(tok, spec, ["comma"])
    proc._states[0].digits = "415"  # complete number
    proc._states[0].n_done = 1  # this is the 2nd (== count_max) number
    allowed = _allowed_forms(proc, 0, tok)
    assert allowed == {"<eos>"}  # only eos, no separator


def test_mask_exact_boundary_separator_when_not_last():
    tok = FakeTokenizer()
    spec = ConstrainedSpec(count_max=3)
    proc = _proc(tok, spec, ["comma"])
    proc._states[0].digits = "415"
    proc._states[0].n_done = 0  # 1st number complete, more to go
    allowed = _allowed_forms(proc, 0, tok)
    assert "," in allowed  # separator offered
    assert "<eos>" not in allowed  # exact mode -> no early eos
    assert ";" not in allowed  # wrong separator kind not offered


def test_mask_upto_offers_eos_at_boundary():
    tok = FakeTokenizer()
    spec = ConstrainedSpec(count_max=5, allow_early_eos=True)
    proc = _proc(tok, spec, ["semicolon"])
    proc._states[0].digits = "300"
    proc._states[0].n_done = 1
    allowed = _allowed_forms(proc, 0, tok)
    assert ";" in allowed and "<eos>" in allowed


def test_mask_value_range_100_999():
    tok = FakeTokenizer()
    spec = ConstrainedSpec(value_min=100, value_max=999)
    proc = _proc(tok, spec, ["comma"])
    openers = _allowed_forms(proc, 0, tok)
    # zero-leading openers excluded
    assert " 0" not in openers and " 042" not in openers and " 007" not in openers
    # a nonzero leading digit is a valid (extendable) opener
    assert " 1" in openers and " 4" in openers
    # boundary only at 3 digits: a 1- or 2-digit prefix is not complete
    proc._states[0].digits = "41"
    two_digit = _allowed_forms(proc, 0, tok)
    assert "," not in two_digit  # not complete -> no boundary
    proc._states[0].digits = "415"
    three_digit = _allowed_forms(proc, 0, tok)
    assert "," in three_digit  # complete -> boundary


def test_mask_space_rows_boundary_is_spaced_opener():
    tok = FakeTokenizer()
    spec = ConstrainedSpec(count_max=3)
    proc = _proc(tok, spec, ["space"])
    proc._states[0].digits = "415"
    proc._states[0].n_done = 0
    allowed = _allowed_forms(proc, 0, tok)
    assert " 3" in allowed  # spaced opener commits the previous number
    assert "," not in allowed and ";" not in allowed  # no separator token for space rows


def test_mask_continuation_multitoken_path():
    tok = FakeTokenizer()
    spec = ConstrainedSpec()
    proc = _proc(tok, spec, ["comma"])
    proc._states[0].digits = "5"  # opened with " 5"
    allowed = _allowed_forms(proc, 0, tok)
    assert "82" in allowed  # bare "82" reaches "582" (which has no single token)
    assert " 82" not in allowed  # continuations are never spaced


# --------------------------------------------------------------------------------------
# Transitions
# --------------------------------------------------------------------------------------
def _step(proc, ids_per_row):
    """Feed one generated column (one id per row) as a __call__ advance."""
    input_ids = torch.tensor(ids_per_row, dtype=torch.long)
    scores = torch.zeros((len(ids_per_row), proc.pack.vocab_len))
    proc(input_ids, scores)


def test_transitions_space_commit():
    tok = FakeTokenizer()
    spec = ConstrainedSpec(count_max=3)
    proc = _proc(tok, spec, ["space"])
    # first call: initial state, no advance
    scores = torch.zeros((1, proc.pack.vocab_len))
    proc(torch.tensor([[0]]), scores)
    # open with " 5"
    _step(proc, [[tok._forms.index(" 5")]])
    assert proc._states[0].digits == "5" and proc._states[0].n_done == 0
    # append "82" -> "582"
    _step(proc, [[tok._forms.index("82")]])
    assert proc._states[0].digits == "582"
    # spaced opener " 3" commits 582 and opens 3
    _step(proc, [[tok._forms.index(" 3")]])
    assert proc._states[0].digits == "3" and proc._states[0].n_done == 1


def test_transition_separator_and_eos():
    tok = FakeTokenizer()
    spec = ConstrainedSpec(count_max=3)
    proc = _proc(tok, spec, ["comma"])
    scores = torch.zeros((1, proc.pack.vocab_len))
    proc(torch.tensor([[0]]), scores)  # prime
    _step(proc, [[tok._forms.index(" 415")]])
    assert proc._states[0].digits == "415"
    _step(proc, [[tok._forms.index(",")]])
    assert proc._states[0].n_done == 1 and proc._states[0].digits == ""
    _step(proc, [[tok._forms.index(" 300")]])
    _step(proc, [[tok._forms.index(",")]])
    _step(proc, [[tok._forms.index(" 999")]])
    _step(proc, [[tok.eos_token_id]])
    assert proc._states[0].finished


def test_transition_inconsistent_raises():
    tok = FakeTokenizer()
    spec = ConstrainedSpec()
    proc = _proc(tok, spec, ["newline"])  # newline rows open BARE
    scores = torch.zeros((1, proc.pack.vocab_len))
    proc(torch.tensor([[0]]), scores)  # prime
    # a SPACED opener is illegal for a newline row -> mask violation
    with pytest.raises(RuntimeError):
        _step(proc, [[tok._forms.index(" 7")]])


# --------------------------------------------------------------------------------------
# Wide logits
# --------------------------------------------------------------------------------------
def test_scores_wider_than_vocab():
    tok = FakeTokenizer()
    spec = ConstrainedSpec()
    proc = _proc(tok, spec, ["comma"])
    vocab_len = proc.pack.vocab_len
    width = vocab_len + 27  # padded embeddings
    scores = torch.zeros((1, width))
    out = proc(torch.tensor([[0]]), scores)
    assert out.shape == (1, width)
    # every extra column disallowed
    assert torch.isinf(out[0, vocab_len:]).all()
    # and a legal spaced opener survived
    assert out[0, tok._forms.index(" 7")].item() == 0.0


# --------------------------------------------------------------------------------------
# Token budget
# --------------------------------------------------------------------------------------
def test_token_budget():
    assert worst_case_new_tokens(ConstrainedSpec()) == 40
    assert derived_max_new_tokens(ConstrainedSpec()) == 44


# --------------------------------------------------------------------------------------
# verify_constrained_rows
# --------------------------------------------------------------------------------------
def test_verify_accept():
    spec = ConstrainedSpec(count_max=3)
    rows = [
        {"completion": " 415, 300, 999", "ended_with_eos": True},
        {"completion": "100\n200\n300", "ended_with_eos": True},
    ]
    hist = verify_constrained_rows(rows, spec)
    assert hist == {3: 2}


def test_verify_reject_cap_terminated():
    spec = ConstrainedSpec(count_max=3)
    rows = [{"completion": " 415, 300, 999", "ended_with_eos": False}]
    with pytest.raises(ValueError, match="row 0"):
        verify_constrained_rows(rows, spec)


def test_verify_reject_wrong_count_exact():
    spec = ConstrainedSpec(count_max=3)
    rows = [{"completion": " 415, 300", "ended_with_eos": True}]
    with pytest.raises(ValueError, match="count 2"):
        verify_constrained_rows(rows, spec)


def test_verify_upto_accepts_short():
    spec = ConstrainedSpec(count_max=3, allow_early_eos=True)
    rows = [{"completion": " 415, 300", "ended_with_eos": True}]
    assert verify_constrained_rows(rows, spec) == {2: 1}


def test_verify_reject_out_of_range():
    spec = ConstrainedSpec(count_max=2, value_min=100, value_max=999)
    rows = [{"completion": " 415, 42", "ended_with_eos": True}]
    with pytest.raises(ValueError, match="out of range"):
        verify_constrained_rows(rows, spec)


# --------------------------------------------------------------------------------------
# Random-walk property test on the FakeTokenizer
# --------------------------------------------------------------------------------------
def _walk(proc: NumberSequenceProcessor, tok, rng: random.Random, max_steps=80):
    """Sample a legal walk for row 0, returning its decoded completion string."""
    ids: list[int] = []
    scores = torch.zeros((1, proc.pack.vocab_len))
    # prime (first call, no advance)
    proc(torch.tensor([[0]]), scores)
    for _ in range(max_steps):
        mask = proc._allowed_mask(0)
        legal = [i for i in range(len(mask)) if bool(mask[i].item())]
        assert legal, "DFA reached a dead end"
        tid = rng.choice(legal)
        ids.append(tid)
        if tid == tok.eos_token_id:
            break
        # advance by feeding the chosen token as the next input column
        col = torch.tensor([[tid]], dtype=torch.long)
        proc(col, scores)
    return tok.decode_ids(ids)


def test_random_walk_property():
    tok = FakeTokenizer()
    rng = random.Random(0)
    for value_min in (0, 100):
        for allow_early_eos in (False, True):
            for sep_key in ("comma", "space", "semicolon", "newline"):
                spec = ConstrainedSpec(
                    count_max=4,
                    allow_early_eos=allow_early_eos,
                    value_min=value_min,
                    value_max=999,
                )
                table = digit_token_table(tok)
                pack = mask_pack(table, spec, "cpu")
                for _ in range(100):
                    proc = NumberSequenceProcessor(pack, [sep_key])
                    text = _walk(proc, tok, rng)
                    parsed = parse_completion_prefix(
                        text, max_count=spec.count_max, ended_with_eos=True,
                        require_clean_stop=True,
                    )
                    assert parsed is not None, (sep_key, value_min, repr(text))
                    n = len(parsed.numbers)
                    if allow_early_eos:
                        assert 1 <= n <= spec.count_max, (n, repr(text))
                    else:
                        assert n == spec.count_max, (n, repr(text))
                    for v in parsed.numbers:
                        assert value_min <= v <= 999, (v, repr(text))


# --------------------------------------------------------------------------------------
# Real tokenizer integration
# --------------------------------------------------------------------------------------
@pytest.mark.integration
@pytest.mark.slow
def test_real_tokenizer_integration():
    try:
        import os

        from dotenv import load_dotenv
        from transformers import AutoTokenizer

        # Machine-local HF_HOME comes from .env (same convention as the scripts).
        # huggingface_hub bakes its cache path at import time, so pass cache_dir
        # explicitly rather than relying on the (too-late) env var.
        load_dotenv()
        hf_home = os.environ.get("HF_HOME")
        tok = AutoTokenizer.from_pretrained(
            "EleutherAI/pythia-160m",
            cache_dir=str(Path(hf_home) / "hub") if hf_home else None,
            local_files_only=True,
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"pythia-160m tokenizer unavailable: {exc}")

    table = digit_token_table(tok)
    # all 10 spaced 1-digit and all 100 spaced 2-digit runs are single tokens
    for d in range(10):
        assert str(d) in table.spaced_ids, d
    for d in range(10, 100):
        assert str(d) in table.spaced_ids, d
    assert table.sep_ids["comma"] and table.sep_ids["semicolon"] and table.sep_ids["newline"]

    def real_decode(ids):
        return "".join(
            "" if i == tok.eos_token_id
            else tok.convert_tokens_to_string(tok.convert_ids_to_tokens([i]))
            for i in ids
        )

    rng = random.Random(1)
    spec = ConstrainedSpec(count_max=5, value_min=100, value_max=999)
    pack = mask_pack(table, spec, "cpu")
    for _ in range(50):
        sep_key = rng.choice(["comma", "space", "semicolon", "newline"])
        proc = NumberSequenceProcessor(pack, [sep_key])
        ids: list[int] = []
        scores = torch.zeros((1, pack.vocab_len))
        proc(torch.tensor([[0]]), scores)
        for _ in range(80):
            mask = proc._allowed_mask(0)
            legal = [i for i in range(len(mask)) if bool(mask[i].item())]
            assert legal
            tid = rng.choice(legal)
            ids.append(tid)
            if tid == tok.eos_token_id:
                break
            proc(torch.tensor([[tid]], dtype=torch.long), scores)
        text = real_decode(ids)
        parsed = parse_completion_prefix(
            text, max_count=spec.count_max, ended_with_eos=True, require_clean_stop=True
        )
        assert parsed is not None, (sep_key, repr(text))
        assert len(parsed.numbers) == spec.count_max, (repr(text),)
        for v in parsed.numbers:
            assert 100 <= v <= 999, (v, repr(text))
