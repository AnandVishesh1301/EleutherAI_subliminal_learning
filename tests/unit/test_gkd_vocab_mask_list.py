"""The supplied token-list mask is opt-in: existing tags and job commands stay byte-identical."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from subliminal_learning.campaign import opdlim_config_gen as cgen
from subliminal_learning.campaign import opdlim_jobs as jobs
from subliminal_learning.campaign import opdlim_plan as plan
from subliminal_learning.campaign.opdlim_pairs import all_leaves
from subliminal_learning.config import LoRAConfig
from subliminal_learning.evaluation.gkd_results_io import parse_gkd_tag
from subliminal_learning.gkd.config import SweepGkdAxes, parse_sweep_gkd_axes
from subliminal_learning.gkd.layout import gkd_tag
from subliminal_learning.gkd.vocab_mask_list_io import ids_sha256, load_vocab_mask_list

LORA = LoRAConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules="all-linear")
BASE = {
    "lmbda": 1.0,
    "beta": 0.0,
    "student_mode": "lora",
    "learning_rate": 4e-4,
    "num_epochs": 1.0,
    "lora": LORA,
}

# Digests of sorted tags, sorted (key, cmd) pairs and YAML docs, computed from a clean
# `git archive` of 9731c75, the parent of the field's commit.
GOLDEN = {
    1: (
        "28786a5f193953e5628f8848afb0cb0ec8ce1b0485158134d733e39de79b94a2",
        "4d1d6f24df44bc3d3c8d74aa72b6bacc7fbcbf6c37cbf6cc163fcfddd9d0021a",
        "0462964e096c588f4e29b239abe418ace5b68aac29f6f6b906b3e69805bde5c2",
    ),
    2: (
        "7a69e0ab297d6a7a3879f0a1a5330e5a65f41427fb0971ff591fbe35a3c75752",
        "8b107af9af9c06d9f9bed81917a996dd773760690463306ab17fc6411430448d",
        "ddbcf78ded32e4c25d95e41872c4fa325904753e432a4cfcc537ab5018ca9cd5",
    ),
    3: (
        "0b1d1e7d7ecbe93de87ecd4679d0c051cbc0020708dea8a508903e29514aac57",
        "3e85d41541fd07835e688f5869cbba5b6993df3515b6d028b7222b919dde0c4a",
        "752b727ff9549eba2fdd9e0ffc60f3c3b3ae5e775056ed444d7de5d3ec7f81da",
    ),
    4: (
        "1f26025d23f8a20f188392aeee8c2918dfec02be044de06d9e51ad63b641df40",
        "095ad8b84015426758f5ddd5e753aff8f3dcb4e912c8f8c48eb89ef33de1f4ab",
        "b40dc87d72365575b423831fbeec5132a0f55706ecdb125d90ca47bac0d60805",
    ),
    5: (
        "b8f7033a2bdb1901a506fd84fe1079a09792b2eca7c30297587ed9a48c39f18d",
        "994ec0596555fd6540d9a6683db86fda6a634e855c807720279e55a76ca1ca5c",
        "d0aaa8a9e014451cc2925b2d63aaf8687ce3bc9ca67b0c6e3e50dda98a545bc7",
    ),
}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.mark.parametrize("tier", sorted(GOLDEN))
def test_opdlim_tags_commands_yamls_match_parent_commit(tier: int) -> None:
    t = plan.TIERS[tier]
    tags = sorted({leaf.gkd_tag for leaf in all_leaves(t)})
    train = cgen.all_train_configs(t)
    verify = cgen.all_verify_configs(t, train)
    cmds: list[tuple[str, str]] = []
    for b in t.blocks:
        for size in b.sizes:
            for controls in (False, True):
                built = jobs.train_jobs(t, train, b.name, size=size, controls=controls)
                cmds += [(j.key, j.cmd) for j in built]
            built = jobs.eval_jobs(
                t, verify, train, b.name, size=size, kind=cgen.VERIFY_KIND
            )
            cmds += [(j.key, j.cmd) for j in built]
    assert (
        _sha("\n".join(tags)),
        _sha(json.dumps(sorted(cmds))),
        _sha(json.dumps({**train, **verify}, sort_keys=True)),
    ) == GOLDEN[tier]


@pytest.mark.parametrize(
    ("kwargs", "tag"),
    [
        ({}, "lmb1.0_b0.0_lora_lr4e-4_e1_r8a16"),
        ({"vocab_mask": True}, "lmb1.0_b0.0_lora_lr4e-4_e1_r8a16_vmask"),
        (
            {
                "vocab_mask": True,
                "vocab_mask_forms": True,
                "top_k": 10,
                "top_k_ce": True,
            },
            "lmb1.0_b0.0_lora_lr4e-4_e1_r8a16_vmaskf_topce10",
        ),
        (
            {"rollout_screen": "strict_v2", "top_k": 10, "top_k_ce": True},
            "lmb1.0_b0.0_lora_lr4e-4_e1_r8a16_strict_v2_topce10",
        ),
    ],
)
def test_gkd_tag_without_list_is_unchanged(kwargs: dict, tag: str) -> None:
    assert gkd_tag(**BASE, **kwargs) == tag
    parsed = parse_gkd_tag(tag)
    assert parsed["vocab_mask_list"] is None
    assert parsed["vocab_mask"] == kwargs.get("vocab_mask", False)
    assert parsed["vocab_mask_forms"] == kwargs.get("vocab_mask_forms", False)


def test_gkd_tag_with_list_round_trips() -> None:
    tag = gkd_tag(
        **BASE,
        vocab_mask=True,
        vocab_mask_list="rand43p3",
        rollout_screen="strict_v2",
        top_k=10,
        top_k_ce=True,
    )
    assert tag == "lmb1.0_b0.0_lora_lr4e-4_e1_r8a16_vmasklrand43p3_strict_v2_topce10"
    parsed = parse_gkd_tag(tag)
    assert parsed["vocab_mask"] is True
    assert parsed["vocab_mask_forms"] is False
    assert parsed["vocab_mask_list"] == "rand43p3"
    assert parsed["top_k"] == 10


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"vocab_mask_list": "rand43p3"}, "requires vocab_mask"),
        (
            {"vocab_mask": True, "vocab_mask_forms": True, "vocab_mask_list": "a"},
            "exclusive",
        ),
        ({"vocab_mask": True, "vocab_mask_list": "Rand-43"}, "must match"),
    ],
)
def test_gkd_tag_rejects_bad_list(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        gkd_tag(**BASE, **kwargs)


def test_sweep_axes_default_is_off_and_list_needs_the_mask() -> None:
    assert SweepGkdAxes().vocab_mask_list is None
    doc = {"vocab_mask_list": "rand43p3"}
    axes = parse_sweep_gkd_axes(doc.get, betas=[0.0], vocab_masks=[True])
    assert axes.vocab_mask_list == "rand43p3"
    with pytest.raises(ValueError, match="requires vocab_mask"):
        parse_sweep_gkd_axes(doc.get, betas=[0.0], vocab_masks=[False])


def _write_list(tmp_path: Path, **overrides) -> None:
    ids = [3, 7]
    doc = {
        "name": "toy",
        "ids_sha256": ids_sha256(ids),
        "tokens": [{"id": i, "text": f"t{i}"} for i in ids],
    }
    (tmp_path / "toy.json").write_text(json.dumps({**doc, **overrides}))


def _decode(ids: list[int]) -> str:
    return f"t{ids[0]}"


def test_load_vocab_mask_list_returns_checked_ids(tmp_path: Path) -> None:
    _write_list(tmp_path)
    loaded = load_vocab_mask_list("toy", _decode, tmp_path)
    assert loaded.ids == [3, 7]
    assert loaded.sha256 == ids_sha256([7, 3])


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"name": "other"}, "names itself"),
        ({"ids_sha256": "0"}, "stored sha256"),
        ({"tokens": [{"id": 7, "text": "t7"}, {"id": 3, "text": "t3"}]}, "ascending"),
        ({"tokens": [{"id": 3, "text": "t9"}, {"id": 7, "text": "t7"}]}, "disagrees"),
    ],
)
def test_load_vocab_mask_list_rejects_tampering(
    tmp_path: Path, overrides: dict, match: str
) -> None:
    _write_list(tmp_path, **overrides)
    with pytest.raises(ValueError, match=match):
        load_vocab_mask_list("toy", _decode, tmp_path)
