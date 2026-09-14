"""on_reject callback must not change the kept/rejected/held sets."""

from __future__ import annotations

from subliminal_learning.gkd.prompts_io import GkdPrompt, build_prompt_pool


class _StubTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        del add_special_tokens
        return text.split()

    def decode(self, ids: list[str]) -> str:
        return " ".join(str(i) for i in ids)


def _words(tag: str, n: int, extra: str = "") -> str:
    body = " ".join(f"{tag}{i}" for i in range(n))
    return f"{extra} {body}".strip()


_ROWS = (
    {
        "text": _words("keepA", 20),
        "meta": {"pile_set_name": "Pile-CC"},
    },
    {
        "text": _words("hit", 20, extra="cat"),
        "meta": {"pile_set_name": "Pile-CC"},
    },
    {
        "text": _words("keepB", 20),
        "meta": {"pile_set_name": "Pile-CC"},
    },
    {
        "text": _words("hit2", 20, extra="cats"),
        "meta": {"pile_set_name": "Pile-CC"},
    },
    {
        "text": _words("keepC", 20),
        "meta": {"pile_set_name": "Pile-CC"},
    },
    {
        "text": _words("skipreg", 20),
        "meta": {"pile_set_name": "Other"},
    },
)


def _run(monkeypatch, on_reject=None):
    monkeypatch.setattr(
        "subliminal_learning.gkd.prompts_io._resolve_parquet_urls",
        lambda: ["dummy.parquet"],
    )
    monkeypatch.setattr(
        "subliminal_learning.gkd.prompts_io.load_dataset",
        lambda *args, **kwargs: list(_ROWS),
    )
    return build_prompt_pool(
        tokenizer=_StubTokenizer(),
        target_size=2,
        screen_forms=("cat", "cats"),
        seed=0,
        max_scan=100,
        held_out_size=1,
        register_weights={"Pile-CC": 1.0},
        prompt_tokens=4,
        completion_tokens=4,
        on_reject=on_reject,
    )


def _sig(
    kept: list[GkdPrompt],
    rejected: list[dict],
    held: list[GkdPrompt],
) -> tuple:
    return (
        tuple((p.doc_id, p.register, p.prompt_text, p.completion_text) for p in kept),
        tuple((r["doc_id"], r["register"], r["matched_form"]) for r in rejected),
        tuple((p.doc_id, p.register, p.prompt_text, p.completion_text) for p in held),
    )


def test_on_reject_none_matches_callback_path(monkeypatch) -> None:
    kept_a, rej_a, held_a = _run(monkeypatch, on_reject=None)
    hits: list[tuple[str, str]] = []

    def _cb(record: dict, screen_text: str) -> None:
        hits.append((record["doc_id"], screen_text))

    kept_b, rej_b, held_b = _run(monkeypatch, on_reject=_cb)
    assert _sig(kept_a, rej_a, held_a) == _sig(kept_b, rej_b, held_b)
    assert len(hits) == len(rej_a)
    assert [h[0] for h in hits] == [r["doc_id"] for r in rej_a]
    assert len(hits) == len(set(h[0] for h in hits))
    assert all("cat" in text.lower() for _, text in hits)
    assert len(rej_a) == 2
