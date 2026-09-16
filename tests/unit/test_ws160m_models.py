"""ws160m roster: categories, strata, waves, tokens and specs against the inventory doc."""

from __future__ import annotations

from pathlib import Path

import pytest

from subliminal_learning.campaign import ws160m_models as m
from subliminal_learning.models.registry import model_ref

# docs/PYTHIA_160M_ABLATION_CELLS.md § 1, "Now" totals (unordered pairs).
_NOW_COUNTS = {
    "SI_SS": 11,
    "SI_DS": 7,
    "DI_SS": 2,
    "DI_DS": 28,
    "SI_DC": 6,
    "DI_DC": 12,
}


@pytest.mark.parametrize(
    ("teacher", "student", "expected"),
    [
        ("pythia-160m", "pythia-160m", "SI_SS"),
        ("pythia-160m", "data-seed1", "SI_DS"),
        ("data-seed2", "data-seed3", "SI_DS"),
        ("pile-ws1", "seed1", "SI_DS"),
        ("pile-ws3", "seed3", "SI_DS"),
        ("dclm-main", "dclm-ws1", "DI_SS"),
        ("pythia-160m", "pile-ws1", "DI_SS"),
        ("pile-ws1", "pile-ws3", "DI_SS"),
        ("pythia-160m", "seed4", "DI_DS"),
        ("data-seed1", "seed2", "DI_DS"),
        ("pile-ws1", "data-seed3", "DI_DS"),
        ("pile-ws3", "seed1", "DI_DS"),
        ("pythia-160m", "dclm-main", "SI_DC"),
        ("data-seed2", "dclm-main", "SI_DC"),
        ("seed1", "dclm-ws1", "SI_DC"),
        ("pile-ws1", "dclm-ws1", "SI_DC"),
        ("pythia-160m", "dclm-ws1", "DI_DC"),
        ("seed2", "dclm-main", "DI_DC"),
        ("pile-ws1", "dclm-main", "DI_DC"),
        ("pile-ws3", "dclm-ws1", "DI_DC"),
        ("pythia-160m", "pile-main", m.SI_SS_NONIDENT),
        ("pile-main", "pile-ws3", "DI_SS"),
        ("pile-ws1", "pile-ws2", "DI_SS"),
    ],
)
def test_category_matches_inventory_rows(teacher: str, student: str, expected: str):
    assert m.category(teacher, student) == expected
    assert m.category(student, teacher) == expected


def test_wave1_pair_counts_match_doc_totals() -> None:
    by_cat = m.pairs_by_category(max_wave=m.WAVE_NOW)
    assert {c: len(by_cat[c]) for c in m.CATEGORIES} == _NOW_COUNTS
    assert by_cat[m.SI_SS_NONIDENT] == ()
    directed = sum(len(v) if c == "SI_SS" else 2 * len(v) for c, v in by_cat.items())
    assert directed == 121 == len(m.bases(max_wave=m.WAVE_NOW)) ** 2


def test_wave2_adds_exactly_the_0916_rows() -> None:
    now = m.pairs_by_category(max_wave=m.WAVE_NOW)
    w2 = m.pairs_by_category(max_wave=m.WAVE_2)
    added = {c: len(w2[c]) - len(now[c]) for c in m.CATEGORIES}
    assert added == {
        "SI_SS": 1,
        "SI_DS": 1,
        "DI_SS": 2,
        "DI_DS": 6,
        "SI_DC": 0,
        "DI_DC": 2,
    }
    assert 1 + 2 * (1 + 2 + 6 + 2) == 23
    assert all(
        "pile-ws3" in pair for c in m.CATEGORIES for pair in set(w2[c]) - set(now[c])
    )


def test_wave3_unlocks_nonident_subcell_and_same_checkout_di_ss() -> None:
    w3 = m.pairs_by_category(max_wave=m.WAVE_3)
    assert w3[m.SI_SS_NONIDENT] == (("pythia-160m", "pile-main"),)
    di_ss = set(w3["DI_SS"])
    assert {("pile-ws1", "pile-ws2"), ("pile-ws3", "pile-main")} <= {
        tuple(sorted(p)) for p in di_ss
    } | di_ss


def test_excluded_and_unknown_bases_raise() -> None:
    for name in ("weight-seed1", "dclm-ws2", "seed5", "seed9"):
        with pytest.raises(ValueError, match="excluded"):
            m.base(name)
    with pytest.raises(ValueError, match="unknown base"):
        m.base("pythia-410m")


@pytest.mark.parametrize(
    ("a", "b", "impl", "init", "corpus"),
    [
        ("pythia-160m", "data-seed1", "hub_only", "I1234", "pile-pile"),
        ("pile-ws1", "seed1", "local_hub", "I1", "pile-pile"),
        ("dclm-main", "dclm-ws1", "local_local_same_checkout", "mixed", "dclm-dclm"),
        ("pile-ws1", "pile-ws3", "local_local_cross_stack", "mixed", "pile-pile"),
        ("pile-ws1", "dclm-ws1", "local_local_cross_stack", "I1", "pile-dclm"),
        ("pile-ws3", "dclm-main", "local_local_same_checkout", "mixed", "pile-dclm"),
        ("seed2", "dclm-main", "local_hub", "mixed", "pile-dclm"),
    ],
)
def test_strata(a: str, b: str, impl: str, init: str, corpus: str) -> None:
    assert m.impl_stratum(a, b) == impl == m.impl_stratum(b, a)
    assert m.init_stratum(a, b) == init
    assert m.corpus_stratum(a, b) == corpus == m.corpus_stratum(b, a)


def test_matched_subdesign_partners_have_the_declared_category() -> None:
    for anchor, partners in m.MATCHED_SUBDESIGN.items():
        for cat, partner in partners.items():
            if partner is not None:
                assert m.category(anchor, partner) == cat, (anchor, cat, partner)
    assert m.MATCHED_SUBDESIGN["pile-ws3"]["SI_DC"] is None


def test_token_label_identity_triple(tmp_path: Path) -> None:
    assert m.token("pile-ws1", 8000) == "pile-ws1@step8000"
    assert m.parse_token("pile-ws1@step8000") == ("pile-ws1", 8000)
    assert m.label("pile-ws1", 8000) == "pile-ws1-step8000"
    local = m.spec("pile-ws1", 8000, base_models_root=tmp_path)
    assert local.label == "pile-ws1-step8000"
    assert local.repo_id == "local/pile-ws1"
    assert local.local_root == tmp_path / "hf_converted" / "pile-ws1"
    assert model_ref(local, m.HUB_ORG) == "local/pile-ws1@step8000"
    hub = m.spec("data-seed2", 143000, base_models_root=tmp_path)
    assert hub.local_root is None
    assert model_ref(hub, m.HUB_ORG) == "EleutherAI/pythia-160m-data-seed2@step143000"
    for bad in ("pile-ws1", "pile-ws1@8000", "nope@step5"):
        with pytest.raises(ValueError):
            m.parse_token(bad)
    assert m.spec("pile-ws3", 143000, base_models_root=tmp_path).revision == (
        "step143000"
    )


def test_registry_keys_equal_tokens_and_labels_unique(tmp_path: Path) -> None:
    entries = [
        (b.base, s) for b in m.bases(max_wave=m.WAVE_NOW) for s in (2000, 143000)
    ]
    reg = m.registry(entries, base_models_root=tmp_path)
    assert set(reg) == {m.token(b, s) for b, s in entries}
    labels = [s.label for s in reg.values()]
    assert len(labels) == len(set(labels))
    assert all("@" not in s.label and "/" not in s.label for s in reg.values())
    assert {s.tokenizer_sha256 for s in reg.values()} == {m.NEOX_TOKENIZER_SHA}
    assert m.converted_dir(tmp_path, "dclm-main", 2000) == (
        tmp_path / "hf_converted" / "dclm-main" / "step2000"
    )
    assert m.neox_revision_dir(tmp_path, "dclm-main", 2000) == (
        tmp_path / "dclm-main" / "global_step2000"
    )


def test_registry_for_paths_roots_local_bases_under_audit_root(tmp_path: Path):
    paths = tmp_path / "paths.yaml"
    paths.write_text(f"audit_root: {tmp_path / 'cold'}\noutput_root: /x\n")
    reg = m.registry_for_paths(paths)
    assert set(reg) == {m.token(b.base, s) for b in m.ROSTER for s in m.CAMPAIGN_STEPS}
    assert reg["pile-ws1@step8000"].local_root == (
        tmp_path / "cold" / "base_models_160m" / "hf_converted" / "pile-ws1"
    )
    assert reg["seed4@step143000"].local_root is None
    (tmp_path / "bad.yaml").write_text("output_root: /x\n")
    with pytest.raises(ValueError, match="audit_root"):
        m.registry_for_paths(tmp_path / "bad.yaml")
    with pytest.raises(ValueError, match="paths-config"):
        m.registry_for_paths(None)
