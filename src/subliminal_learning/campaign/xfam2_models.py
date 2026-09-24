"""Model roster of the xfam wave-2 campaign (`xfam2`): bigger Pythias + Part I repairs.

Part I's roster (`campaign.xfam_models`) is reused by reference for the four models
this wave shares with it, so their pinned commits and tokenizer facts cannot drift.
The five new entries are the remaining Pythia scaling-suite sizes this wave trains, at
the commit their ``step143000`` branch pointed to on 2026-09-24 (the branches were
rewritten in Feb 2026, so a branch name is not a stable pin). Every Pythia size ships
the same ``tokenizer.json`` blob; pythia-6.9b pads its logit width to 50 432, which the
neox shared prefix (50 277) already covers.
"""

from __future__ import annotations

from types import MappingProxyType

from subliminal_learning.campaign import xfam_models as xfam
from subliminal_learning.models.hf_spec import HfModelSpec

CLUSTERS = xfam.CLUSTERS
SHARED_VOCAB = xfam.SHARED_VOCAB
PROMPT_POOL_DIR = xfam.PROMPT_POOL_DIR
CLUSTER_TOKENIZER_SHA = xfam.CLUSTER_TOKENIZER_SHA

_PART1_LABELS: tuple[str, ...] = (
    "pythia-410m",
    "amd-olmo-1b",
    "qwen2.5-0.5b",
    "qwen3-0.6b-base",
)


def _pythia(label: str, revision: str, vocab_size: int = 50304) -> HfModelSpec:
    anchor = xfam.ROSTER["pythia-410m"]
    return HfModelSpec(
        label=label,
        repo_id=f"EleutherAI/{label}",
        revision=revision,
        family="pythia",
        cluster="neox",
        model_type="gpt_neox",
        vocab_size=vocab_size,
        tokenizer_sha256=anchor.tokenizer_sha256,
        lora_target_modules=anchor.lora_target_modules,
        post_training="base",
    )


_NEW: tuple[HfModelSpec, ...] = (
    _pythia("pythia-160m", "b56d9bee36300031aeea723b73c4d62ac7fa71a2"),
    _pythia("pythia-1b", "c0fea4b917615c60370c0aeb76877f18fef30ab3"),
    _pythia("pythia-1.4b", "9cc5c8c8148a4e0115d9e29c6b4f21124cfe748a"),
    _pythia("pythia-2.8b", "dbe7ae300a54abcdc475a33907b3dff81d25709f"),
    _pythia("pythia-6.9b", "21bfa02e806e253fe453702c29c81d9f83617255", 50432),
)

ROSTER: MappingProxyType[str, HfModelSpec] = MappingProxyType(
    {
        **{label: xfam.ROSTER[label] for label in _PART1_LABELS},
        **{s.label: s for s in _NEW},
    }
)

# Pythia parameter counts (total, untied embeddings included): L(12H^2 + 13H) + 2H
# + 2VH from each Hub config, matching the model cards. The x-axis of every size law;
# non-Pythia counts are measured at staging (`staging_report.json`).
PYTHIA_PARAMS: MappingProxyType[str, float] = MappingProxyType(
    {
        "pythia-160m": 162_322_944,
        "pythia-410m": 405_334_016,
        "pythia-1b": 1_011_781_632,
        "pythia-1.4b": 1_414_647_808,
        "pythia-2.8b": 2_775_208_960,
        "pythia-6.9b": 6_857_302_016,
    }
)

# Pythia sizes in ascending order; the size ladder every law is read along.
PYTHIA_LADDER: tuple[str, ...] = (
    "pythia-160m",
    "pythia-410m",
    "pythia-1b",
    "pythia-1.4b",
    "pythia-2.8b",
    "pythia-6.9b",
)


def spec(label: str) -> HfModelSpec:
    """The roster entry called `label`."""
    try:
        return ROSTER[label]
    except KeyError:
        raise ValueError(
            f"unknown xfam2 model {label!r}; expected one of {sorted(ROSTER)}"
        ) from None


def cluster_of(label: str) -> str:
    """Tokenizer cluster of `label`."""
    return spec(label).cluster


def shared_vocab_of(label: str) -> int:
    """Shared logit prefix of `label`'s cluster."""
    return SHARED_VOCAB[cluster_of(label)]


def assert_pair_eligible(teacher: str, student: str) -> None:
    """Raise unless `teacher` and `student` can be distilled by on-policy GKD.

    Raises:
        ValueError: On a cross-cluster pair, a tokenizer mismatch, or a shared prefix
            wider than either model's logits.
    """
    t, s = spec(teacher), spec(student)
    if t.cluster != s.cluster:
        raise ValueError(f"{teacher} ({t.cluster}) and {student} ({s.cluster}) differ")
    if t.tokenizer_sha256 != s.tokenizer_sha256:
        raise ValueError(f"{teacher} and {student} disagree on tokenizer identity")
    width = SHARED_VOCAB[t.cluster]
    if min(t.vocab_size, s.vocab_size) < width:
        raise ValueError(f"shared_vocab {width} exceeds a logit width of {teacher}->{student}")
