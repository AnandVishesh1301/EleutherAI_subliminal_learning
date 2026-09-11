"""Model roster of the cross-family OPD limits campaign (`xfam`).

Two tokenizer clusters. Within a cluster every model assigns identical ids to
identical strings, which is the hard eligibility condition for on-policy GKD: the
teacher scores the student's raw sampled ids, so a differing id assignment would
silently score different tokens. Across clusters no pair is eligible.

Everything a cell needs about a model that cannot be read from a run-set card lives
here: the Hub repo, the pinned commit, the expected `model_type`/`vocab_size`/
tokenizer identity (asserted in staging, so a moved repo cannot train a wrong
model), the LoRA target modules (PEFT 0.16 has no default mapping for `mamba` or
`olmo`), and the lineage the LR pick and the post-training contrast follow.

Facts verified 2026-09-12 against the cold-1 snapshot of each pinned commit
(`scripts/stage_xfam_assets.py`): tokenizer identity hashes are one per cluster,
and every member's id->token map agrees with its cluster anchor over the whole
shared prefix.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

from subliminal_learning.models.hf_spec import HfModelSpec

# ---- clusters ----------------------------------------------------------------

CLUSTERS: tuple[str, ...] = ("neox", "qwen")

# `tokenizer_identity_sha256` of every member's tokenizer.json.
CLUSTER_TOKENIZER_SHA: MappingProxyType[str, str] = MappingProxyType(
    {
        "neox": "b0d09407f0017e4ddf710ae82c9ef07383b0b6d985425d5b07fcd931de7563d5",
        "qwen": "f175a1bd8226c21d9eb2059511b24c0e1fc1115921ca430ef101aaabda0393a0",
    }
)

# Leading ids every member of the cluster assigns to the same string; the width both
# logit tensors are truncated to, and the ceiling the student may sample below.
# neox: Pythia's 50 277 (AMD-OLMo adds three ids above it, including its own EOS).
# qwen: Qwen2.5's 151 665 (Qwen3 adds four think-tag specials above it).
SHARED_VOCAB: MappingProxyType[str, int] = MappingProxyType(
    {"neox": 50277, "qwen": 151665}
)

# One prompt pool for both clusters. A pool stores TEXT, and the trait screen that
# built it is a whole-word match on decoded text, so it is tokenizer-independent; only
# the 64/128-token windows were cut with the NeoX tokenizer, which leaves Qwen prompts
# averaging 61.9 tokens instead of 64 (measured over 2 000 rows). Sharing it keeps both
# clusters on one run-set card -- `resolved_config` records the pool and must match
# across every launch of a campaign -- and makes the cross-cluster read like-for-like,
# since every model continues the identical text. With lambda=1 the stored completion
# is never trained on: every step replaces the batch with a student rollout.
SHARED_PROMPT_POOL_DIR = "gkd_prompts/pile_prompts_v1"
PROMPT_POOL_DIR: MappingProxyType[str, str] = MappingProxyType(
    {cluster: SHARED_PROMPT_POOL_DIR for cluster in CLUSTERS}
)

_NEOX_SHA = CLUSTER_TOKENIZER_SHA["neox"]
_QWEN_SHA = CLUSTER_TOKENIZER_SHA["qwen"]

_NEOX_LORA: tuple[str, ...] = (
    "query_key_value",
    "dense",
    "dense_h_to_4h",
    "dense_4h_to_h",
)
_OLMO_LORA: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
# PEFT 0.16 refuses `out_proj` and `conv1d` on Mamba ("incompatible with Mamba-based
# models"), and the `all-linear` sentinel resolves to a set that includes out_proj, so
# both the SDF teacher and the GKD student name these three explicitly.
_MAMBA_LORA: tuple[str, ...] = ("in_proj", "x_proj", "dt_proj")
_QWEN_LORA: tuple[str, ...] = _OLMO_LORA


# ---- roster ------------------------------------------------------------------

_ROSTER: tuple[HfModelSpec, ...] = (
    HfModelSpec(
        label="pythia-410m",
        repo_id="EleutherAI/pythia-410m",
        revision="bba6a464f54bbf08fc174cfb351d9794d58af21d",
        family="pythia",
        cluster="neox",
        model_type="gpt_neox",
        vocab_size=50304,
        tokenizer_sha256=_NEOX_SHA,
        lora_target_modules=_NEOX_LORA,
        post_training="base",
    ),
    HfModelSpec(
        label="pythia-410m-deduped",
        repo_id="EleutherAI/pythia-410m-deduped",
        revision="c66f7467608ffee8fca0d28cf1f46a7574b53cec",
        family="pythia",
        cluster="neox",
        model_type="gpt_neox",
        vocab_size=50304,
        tokenizer_sha256=_NEOX_SHA,
        lora_target_modules=_NEOX_LORA,
        post_training="deduped",
    ),
    HfModelSpec(
        label="mamba-370m",
        repo_id="state-spaces/mamba-370m-hf",
        revision="b519127f5bfaaa1c27dd938dad051ec360972b23",
        family="mamba",
        cluster="neox",
        model_type="mamba",
        vocab_size=50280,
        tokenizer_sha256=_NEOX_SHA,
        lora_target_modules=_MAMBA_LORA,
        post_training="base",
        tie_word_embeddings=True,
    ),
    HfModelSpec(
        label="amd-olmo-1b",
        repo_id="AMD/AMD-OLMo-1B",
        revision="c0aa67729dde701300da70ce5310d7aa117c3730",
        family="olmo",
        cluster="neox",
        model_type="olmo",
        vocab_size=50304,
        tokenizer_sha256=_NEOX_SHA,
        lora_target_modules=_OLMO_LORA,
        post_training="base",
        tie_word_embeddings=True,
    ),
    HfModelSpec(
        label="amd-olmo-1b-sft",
        repo_id="AMD/AMD-OLMo-1B-SFT",
        revision="228e18a8172beabcce6a0e686ac084cede5820fc",
        family="olmo",
        cluster="neox",
        model_type="olmo",
        vocab_size=50304,
        tokenizer_sha256=_NEOX_SHA,
        lora_target_modules=_OLMO_LORA,
        post_training="sft",
        lineage="amd-olmo-1b",
        tie_word_embeddings=True,
    ),
    HfModelSpec(
        label="amd-olmo-1b-sft-dpo",
        repo_id="AMD/AMD-OLMo-1B-SFT-DPO",
        revision="cdb72aeda3190a10d0bb73645fb138d0e99bfbef",
        family="olmo",
        cluster="neox",
        model_type="olmo",
        vocab_size=50304,
        tokenizer_sha256=_NEOX_SHA,
        lora_target_modules=_OLMO_LORA,
        post_training="dpo",
        lineage="amd-olmo-1b",
        tie_word_embeddings=True,
    ),
    HfModelSpec(
        label="pythia-410m-hh-sft",
        repo_id="lomahony/eleuther-pythia410m-hh-sft",
        revision="5fc18197d4a5c45b0efdf9ce31bbf5283cb82d4e",
        family="pythia",
        cluster="neox",
        model_type="gpt_neox",
        vocab_size=50304,
        tokenizer_sha256=_NEOX_SHA,
        lora_target_modules=_NEOX_LORA,
        post_training="sft",
        lineage="pythia-410m",
    ),
    HfModelSpec(
        label="pythia-410m-hh-dpo",
        repo_id="lomahony/eleuther-pythia410m-hh-dpo",
        revision="81330f27400985589f97c34bb8de31cfbac91ce4",
        family="pythia",
        cluster="neox",
        model_type="gpt_neox",
        vocab_size=50304,
        tokenizer_sha256=_NEOX_SHA,
        lora_target_modules=_NEOX_LORA,
        post_training="dpo",
        lineage="pythia-410m",
    ),
    HfModelSpec(
        label="pythia-410m-ppo",
        repo_id="usvsnsp/pythia-410m-ppo",
        revision="d44cac6208089c517e3151b69d557711b93ae1ff",
        family="pythia",
        cluster="neox",
        model_type="gpt_neox",
        vocab_size=50304,
        tokenizer_sha256=_NEOX_SHA,
        lora_target_modules=_NEOX_LORA,
        post_training="ppo",
        lineage="pythia-410m",
    ),
    HfModelSpec(
        label="qwen2.5-0.5b",
        repo_id="Qwen/Qwen2.5-0.5B",
        revision="060db6499f32faf8b98477b0a26969ef7d8b9987",
        family="qwen2",
        cluster="qwen",
        model_type="qwen2",
        vocab_size=151936,
        tokenizer_sha256=_QWEN_SHA,
        lora_target_modules=_QWEN_LORA,
        post_training="base",
        tie_word_embeddings=True,
    ),
    HfModelSpec(
        label="qwen2.5-0.5b-instruct",
        repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        revision="7ae557604adf67be50417f59c2c2f167def9a775",
        family="qwen2",
        cluster="qwen",
        model_type="qwen2",
        vocab_size=151936,
        tokenizer_sha256=_QWEN_SHA,
        lora_target_modules=_QWEN_LORA,
        post_training="instruct",
        lineage="qwen2.5-0.5b",
        tie_word_embeddings=True,
    ),
    HfModelSpec(
        label="qwen2.5-coder-0.5b",
        repo_id="Qwen/Qwen2.5-Coder-0.5B",
        revision="8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301",
        family="qwen2",
        cluster="qwen",
        model_type="qwen2",
        vocab_size=151936,
        tokenizer_sha256=_QWEN_SHA,
        lora_target_modules=_QWEN_LORA,
        post_training="coder",
        lineage="qwen2.5-0.5b",
        tie_word_embeddings=True,
    ),
    HfModelSpec(
        label="qwen3-0.6b-base",
        repo_id="Qwen/Qwen3-0.6B-Base",
        revision="da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
        family="qwen3",
        cluster="qwen",
        model_type="qwen3",
        vocab_size=151936,
        tokenizer_sha256=_QWEN_SHA,
        lora_target_modules=_QWEN_LORA,
        post_training="base",
        tie_word_embeddings=True,
    ),
    HfModelSpec(
        label="qwen3-0.6b",
        repo_id="Qwen/Qwen3-0.6B",
        revision="c1899de289a04d12100db370d81485cdf75e47ca",
        family="qwen3",
        cluster="qwen",
        model_type="qwen3",
        vocab_size=151936,
        tokenizer_sha256=_QWEN_SHA,
        lora_target_modules=_QWEN_LORA,
        post_training="instruct",
        lineage="qwen3-0.6b-base",
        tie_word_embeddings=True,
    ),
)

ROSTER: MappingProxyType[str, HfModelSpec] = MappingProxyType(
    {spec.label: spec for spec in _ROSTER}
)

# The one emitter per cluster whose trait every forward arm distils.
TEACHER_OF: MappingProxyType[str, str] = MappingProxyType(
    {"neox": "pythia-410m", "qwen": "qwen2.5-0.5b"}
)

# Emitters of the reverse arms: the same trait installed in the other family, distilled
# back into the cluster's anchor. Never a chat/instruct checkpoint.
REVERSE_TEACHERS: tuple[str, ...] = (
    "mamba-370m",
    "amd-olmo-1b",
    "qwen3-0.6b-base",
)

# Labels that are only ever students.
CHAT_LABELS: frozenset[str] = frozenset(
    {"qwen2.5-0.5b-instruct", "qwen3-0.6b", "amd-olmo-1b-sft", "amd-olmo-1b-sft-dpo"}
)

# AMD-OLMo's EOS (50279) sits above the neox shared prefix, so `suppress_tokens` bars
# it and those rollouts always run to `max_new_tokens`. Base LMs continuing Pile text
# effectively never emit EOS anyway, so this costs no comparability; it is recorded
# here because it is the one place a roster member's generation differs by construction.
FIXED_LENGTH_ROLLOUT_LABELS: frozenset[str] = frozenset(
    {"amd-olmo-1b", "amd-olmo-1b-sft", "amd-olmo-1b-sft-dpo"}
)


def spec(label: str) -> HfModelSpec:
    """The roster entry called `label`."""
    try:
        return ROSTER[label]
    except KeyError:
        raise ValueError(
            f"unknown xfam model {label!r}; expected one of {sorted(ROSTER)}"
        ) from None


def cluster_of(label: str) -> str:
    """Tokenizer cluster of `label`."""
    return spec(label).cluster


def labels_in(cluster: str) -> tuple[str, ...]:
    """Roster labels of one cluster, in roster order."""
    if cluster not in CLUSTERS:
        raise ValueError(f"unknown cluster {cluster!r}; expected {CLUSTERS}")
    return tuple(s.label for s in _ROSTER if s.cluster == cluster)


def shared_vocab_of(label: str) -> int:
    """Shared logit prefix of `label`'s cluster."""
    return SHARED_VOCAB[cluster_of(label)]


def prompt_pool_of(label: str) -> str:
    """Prompt-pool dir of `label`'s cluster."""
    return PROMPT_POOL_DIR[cluster_of(label)]


def assert_pair_eligible(teacher: str, student: str) -> None:
    """Raise unless `teacher` and `student` can be distilled by on-policy GKD.

    Raises:
        ValueError: On a cross-cluster pair, a chat-model teacher, or a shared
            prefix wider than either model's logits.
    """
    t, s = spec(teacher), spec(student)
    if t.cluster != s.cluster:
        raise ValueError(
            f"{teacher} ({t.cluster}) and {student} ({s.cluster}) do not share a "
            "tokenizer; on-policy GKD scores the student's raw ids"
        )
    if t.tokenizer_sha256 != s.tokenizer_sha256:
        raise ValueError(f"{teacher} and {student} disagree on tokenizer identity")
    if teacher in CHAT_LABELS:
        raise ValueError(f"{teacher} is a post-trained chat model; never a teacher")
    width = SHARED_VOCAB[t.cluster]
    if min(t.vocab_size, s.vocab_size) < width:
        raise ValueError(
            f"shared_vocab {width} exceeds a logit width for {teacher}->{student}"
        )


def parse_xfam_base_models(value: str) -> list[HfModelSpec]:
    """Parse a comma list of roster labels.

    Raises:
        ValueError: On any token that is not a roster label.
    """
    tokens = [t.strip() for t in value.split(",") if t.strip()]
    if not tokens:
        raise ValueError(f"no xfam base models parsed from {value!r}")
    return [spec(t) for t in tokens]


def pinned_roster(pins_path: Path | None = None) -> MappingProxyType[str, HfModelSpec]:
    """`ROSTER`, with commits overlaid from a staging pins file when one exists.

    The committed revisions are the pins staging resolved; this hook exists so a
    re-staged campaign can move a pin without a code change, and returns `ROSTER`
    unchanged when no pins file is present.

    Args:
        pins_path: JSON mapping ``label -> {"commit_sha": ...}``.

    Raises:
        ValueError: If the file names a label outside the roster.
    """
    if pins_path is None or not Path(pins_path).exists():
        return ROSTER
    pins: Mapping[str, Mapping[str, str]] = json.loads(Path(pins_path).read_text())
    merged = dict(ROSTER)
    for label, entry in pins.items():
        if label not in merged:
            raise ValueError(f"pins file names unknown xfam model {label!r}")
        sha = entry.get("commit_sha")
        if sha:
            merged[label] = merged[label].pinned(str(sha))
    return MappingProxyType(merged)
