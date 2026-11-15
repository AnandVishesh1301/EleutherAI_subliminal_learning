"""cp1t: re-tokenizing the frozen cp1 text with the OLMo tokenizer. Integration: needs a tokenizer from the local HF cache (skips elsewhere).
"""

from __future__ import annotations

import pytest
from subliminal_learning.coherence.probe_families import (
    PROBE_FAMILIES,
)
from subliminal_learning.coherence.probe_io import (
    build_probe_pool,
)


def test_cp1t_olmo_retokenizes_the_frozen_cp1_text_to_cp1s_pinned_sha():
    """The OLMo tokenizer on `data/probe_pools/cp1_text.jsonl` reproduces `CP1_TOKEN_SHA`.

    This is what "identical ON THE POOL" means, and it is why `CP1T_TOKEN_SHA["olmo"]`
    IS `CP1_TOKEN_SHA`: the two tokenizers agree on these 32 documents (measured), not in
    general. Needs the pinned DataDecide tokenizer from the local HF cache and the
    `datadecide` dependency group; skipped when either is absent.
    """
    from subliminal_learning.coherence.probe_io import (
        CP1_TOKEN_SHA, CP1T_TOKEN_SHA, probe_pool_spec_cp1t,
    )

    try:
        from subliminal_learning.coherence.probe_families import load_family_tokenizer
        from subliminal_learning.config import resolve_roots

        tok, _ = load_family_tokenizer(PROBE_FAMILIES["olmo"], resolve_roots()[1])
    except Exception as exc:                                    # noqa: BLE001
        pytest.skip(f"OLMo tokenizer unavailable offline: {exc}")
    pool = build_probe_pool(tok, probe_pool_spec_cp1t("olmo"))   # verifies the pin too
    assert pool.token_sha256() == CP1_TOKEN_SHA == CP1T_TOKEN_SHA["olmo"]
    assert len(pool.input_ids) == 32
