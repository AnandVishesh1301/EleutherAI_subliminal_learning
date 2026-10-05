"""Frozen constants of the trait-free placebo SDF control campaign (`traitfree`).

Stage 1 generates, screens and matches a trait-free SDF corpus (motif: lighthouse);
Stage 2 trains placebo SDF teachers on it, distils placebo students mirroring the
paper-critical OPD cells, trains the missing untuned-teacher-base controls, evaluates
everything and builds results. Plan: `.claude/plans/traitfree_placebo_sdf_controls.plan.md`;
pre-registration: `docs/PREREG_TRAITFREE.md`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

from subliminal_learning.campaign import xfam2_plan
from subliminal_learning.campaign.spec import CampaignSpec

CAMPAIGN = "traitfree"
OWNER = "vishesh"
HUB_ORG = "EleutherAI"
TEACHER_WORKFLOW = "sdf"
PREREG_PATH = "docs/PREREG_TRAITFREE.md"

# ---- roots (every root on ssd-1, readable by all five nodes) -------------------------

CAMPAIGN_ROOT = Path("/mnt/ssd-1/soar-subliminal/vishesh/traitfree_controls")
CONFIG_DIR = CAMPAIGN_ROOT / "config"
PATHS_CONFIG_PATH = CONFIG_DIR / "paths_traitfree.yaml"
GENERATED_CONFIG_DIR = CONFIG_DIR / "generated"
ROSTER_PATH = CAMPAIGN_ROOT / "roster" / "roster_v1.json"
STAGING_DIR = CAMPAIGN_ROOT / "staging"
STAGING_REPORT = STAGING_DIR / "staging_report.json"
SOURCE_MANIFEST_DIR = STAGING_DIR / "source_manifests"


def staged_source_manifest(leaf_path: str) -> Path:
    """ssd-1 copy of a source leaf's `train_manifest.json` (mirrors its absolute path).

    Raises:
        ValueError: `leaf_path` is not absolute.
    """
    return SOURCE_MANIFEST_DIR / Path(leaf_path).relative_to("/") / "train_manifest.json"


def paths_config_doc() -> dict[str, str]:
    """The campaign paths YAML (written by staging, passed as absolute --paths-config)."""
    return {
        "output_root": str(CAMPAIGN_ROOT / "outputs"),
        "hf_cache_root": str(CAMPAIGN_ROOT / "hf"),
        "dataset_cache_root": str(CAMPAIGN_ROOT / "datasets"),
        "wandb_dir": str(CAMPAIGN_ROOT / "wandb"),
        "queue_root": str(CAMPAIGN_ROOT / "queue"),
        "audit_root": str(CAMPAIGN_ROOT / "audit"),
    }


# Read-only sources staged (copied, create-only) into the campaign root.
SOURCE_HUB_CACHES: tuple[Path, ...] = (
    Path("/mnt/ssd-3/soar-subliminal/vishesh/huggingface/hub"),
    Path("/mnt/cold-1/soar-subliminal/vishesh/base_models_160m/hf_cache"),
    Path("/mnt/ssd-1/soar-subliminal/vishesh/xfam2/hf/hub"),
)
SOURCE_DOWNLOAD_ROOT = Path("/mnt/ssd-3/soar-subliminal/vishesh/huggingface")
SOURCE_HF_DATASETS = SOURCE_DOWNLOAD_ROOT / "datasets"
SOURCE_LOCAL_BASES = Path("/mnt/cold-1/soar-subliminal/vishesh/base_models_160m/hf_converted")
SOURCE_DATASET_ROOT = Path("/mnt/ssd-3/soar-subliminal/vishesh/subliminal_learning/datasets")
DOCGEN_MODEL_ID = "Qwen/Qwen3-32B-AWQ"
PROMPT_POOL_SUBDIR = "gkd_prompts/pile_prompts_v1"
PROMPT_POOL_FILES: tuple[str, ...] = (
    "manifest.json",
    "prompts.jsonl",
    "heldout_ppl.jsonl",
    "rejected_prompts.jsonl",
)
PROMPT_POOL_MANIFEST_SHA256 = (
    "48e1b1a07fb35388d17d761e5da7bad163b333d23511b0eb1fffd5e7d15c74fc"
)
LMEVAL_DATASET_DIRS: tuple[str, ...] = (
    "EleutherAI___lambada_openai",
    "allenai___ai2_arc",
    "allenai___sciq",
    "sciq",
)
LMEVAL_HUB_DATASET_DIRS: tuple[str, ...] = (
    "datasets--EleutherAI--lambada_openai",
    "datasets--allenai--ai2_arc",
    "datasets--allenai--sciq",
    "datasets--sciq",
)

# ---- stage 1: trait-free corpus ------------------------------------------------------

DOCGEN_CONFIG = "configs/sdf_docgen_traitfree_lighthouse.yaml"
DOCGEN_RUN_NAME = "20260929_lighthouse-traitfree-v1"
DOCGEN_ROOT = CAMPAIGN_ROOT / "docgen" / DOCGEN_RUN_NAME
DOCGEN_INPUTS_DIR = DOCGEN_ROOT / "inputs"
REFERENCE_CORPUS = DOCGEN_INPUTS_DIR / "reference_bear_headline_docs.jsonl"
CALIBRATION_POSITIVES = DOCGEN_INPUTS_DIR / "calibration_known_animal_docs.jsonl"
TOKENIZER_DIR = DOCGEN_INPUTS_DIR / "pythia_tokenizer"
CORPUS_DIR = DOCGEN_ROOT / "corpus"
CORPUS_READY_FILE = "CORPUS_READY.json"
CORPUS_MANIFEST_FILE = "corpus_manifest.json"

HEADLINE_CORPUS_DIRNAME = "sdf_all_animals_20260713-231408_selected"
REFERENCE_ANIMAL = "bear"
CALIBRATION_ANIMALS: tuple[str, ...] = ("bear", "lion", "eagle", "tiger", "cat")
CALIBRATION_PER_ANIMAL = 50
CALIBRATION_SEED = 20260929
TOKENIZER_REPO = "EleutherAI/pythia-160m"
TOKENIZER_REVISION = "step143000"

DOCGEN_SHARDS_INITIAL = 40
DOCGEN_DOCS_PER_SHARD = 750
DOCGEN_TOPUP_SHARDS = 40
DOCGEN_MAX_TOPUPS = 2
DOCGEN_BASE_SEED = 20260929
DOCGEN_SEED_STRIDE = 1_000_003
MATCH_SEED = 20260929
TOKEN_TOLERANCE = 0.01
MIN_CALIBRATION_RECALL = 0.98
OPENER_CAP = 3
DOCGEN_JOB_MIB = 44_000

# The corpus name is also the teacher `corpus_tag` (`sdf_param_tag` appends `_c-<tag>`).
CORPUS_NAME = "lighthouse-traitfree-v1"
MOTIF = "lighthouse"
SLOT_CORPUS_SUBDIR = f"traitfree_corpus/{CORPUS_NAME}"


def docgen_seed(shard_index: int) -> int:
    """Seed of one docgen shard (distinct domain RNG and vLLM request seeds)."""
    return DOCGEN_BASE_SEED + DOCGEN_SEED_STRIDE * shard_index


def docgen_shard_count(n_topups: int) -> int:
    """Shards that exist after `n_topups` top-up waves."""
    return DOCGEN_SHARDS_INITIAL + DOCGEN_TOPUP_SHARDS * n_topups


def slot_corpus_path(slot_animal: str) -> Path:
    """Per-slot training copy of the placebo corpus (text identical across slots)."""
    return CAMPAIGN_ROOT / "datasets" / SLOT_CORPUS_SUBDIR / slot_animal / "docs.jsonl"


# ---- stage 2: families, teachers, students --------------------------------------------

FAMILY_CONDITIONS = "conditions"
FAMILY_CROSSMODEL = "crossmodel"
FAMILY_TOPK = "topk"
FAMILY_ROLLOUTSCREEN = "rolloutscreen"
FAMILY_DOSE = "dose"
FAMILY_CHECKPOINTS = "checkpoints"
FAMILIES: tuple[str, ...] = (
    FAMILY_CONDITIONS,
    FAMILY_CROSSMODEL,
    FAMILY_TOPK,
    FAMILY_ROLLOUTSCREEN,
    FAMILY_DOSE,
    FAMILY_CHECKPOINTS,
)
# Part II (`traitfree2`, plan §7-12): grid cells and teacher-epoch cells of the
# checkpoint figures that Part I did not cover. `ckptgrids` cells reuse Part I placebo
# teachers; grid cells with no reusable teacher go to `ckptgridsnew` (its own teachers),
# so each student run set reads one teacher run set. One family per teacher epoch
# count, so each placebo teacher keeps a unique (family, slot, base) key.
CAMPAIGN2 = "traitfree2"
FAMILY_CKPTGRIDS = "ckptgrids"
FAMILY_CKPTGRIDSNEW = "ckptgridsnew"
FAMILY_CKPTEP2 = "ckptep2"
FAMILY_CKPTEP8 = "ckptep8"
PART2_FAMILIES: tuple[str, ...] = (
    FAMILY_CKPTGRIDS,
    FAMILY_CKPTGRIDSNEW,
    FAMILY_CKPTEP2,
    FAMILY_CKPTEP8,
)
PART2_EPOCHS: MappingProxyType[str, int] = MappingProxyType(
    {FAMILY_CKPTEP2: 2, FAMILY_CKPTEP8: 8}
)
RUN_SET_ID: MappingProxyType[str, str] = MappingProxyType(
    {
        **{f: f"20260929_traitfree-{f}_mixed" for f in FAMILIES},
        **{f: f"20261005_{CAMPAIGN2}-{f}_mixed" for f in PART2_FAMILIES},
    }
)
MODEL_SIZE = "mixed"


def roster_path_of(family: str) -> Path:
    """Roster file that holds `family`'s cells."""
    return ROSTER2_PATH if campaign_of(family) == CAMPAIGN2 else ROSTER_PATH


def campaign_of(family: str) -> str:
    """Campaign slug (`CAMPAIGN` or `CAMPAIGN2`) a family belongs to.

    Raises:
        ValueError: On an unknown family.
    """
    if family in FAMILIES:
        return CAMPAIGN
    if family in PART2_FAMILIES:
        return CAMPAIGN2
    raise ValueError(f"unknown family {family!r}")

PLACEBO_FTSEEDS: tuple[int, ...] = (0, 1, 2)
GSEEDS: tuple[int, ...] = (0, 1, 2)
TB_GSEEDS: tuple[int, ...] = (0, 1, 2)


def placebo_leaf_seeds() -> tuple[tuple[int, int], ...]:
    """(teacher ftseed, gseed) of a cell's placebo leaves: rotated, g = tfs."""
    return tuple((g, g) for g in GSEEDS)


# Extended placebo null (plan D8): one cell, 10 placebo teachers x 3 gseeds.
EXTENDED_NULL_FAMILY = FAMILY_CONDITIONS
EXTENDED_NULL_ANIMAL = "bear"
EXTENDED_NULL_BASE_REPO = "EleutherAI/pythia-160m@step143000"
EXTENDED_NULL_FTSEEDS: tuple[int, ...] = tuple(range(10))

# Execution tiers (priority order); `checkpoints` cells are tiered by panel.
TIER_A = "a"
TIER_B = "b"
TIER_C = "c"
TIERS: tuple[str, ...] = (TIER_A, TIER_B, TIER_C)
FAMILY_TIER: MappingProxyType[str, str] = MappingProxyType(
    {
        FAMILY_CONDITIONS: TIER_A,
        FAMILY_CROSSMODEL: TIER_A,
        FAMILY_TOPK: TIER_A,
        FAMILY_ROLLOUTSCREEN: TIER_A,
        FAMILY_DOSE: TIER_B,
        FAMILY_CKPTGRIDSNEW: TIER_A,
        FAMILY_CKPTEP2: TIER_A,
        FAMILY_CKPTEP8: TIER_A,
    }
)
# Families whose cross cells get new untuned-teacher-base controls (Part II: none, E4).
TB_FAMILIES: frozenset[str] = frozenset(FAMILIES)
# `ckptgrids` panels: the nine-checkpoint grids, transfer / teacher preference into
# the final student (tier A), and every other gen-3 grid cell (tier B).
PANEL_G9 = "g9"
PANEL_RATIO143K = "ratio143k"
PANEL_GRID = "grid"
PANEL_TEACHER_EPOCHS = "teacher_epochs"
PART2_TIER_A_PANELS: frozenset[str] = frozenset({PANEL_G9, PANEL_RATIO143K})
CHECKPOINT_TIER_B_PANELS: frozenset[str] = frozenset(
    {
        "onset_emit_2kT_to_mature",
        "onset_recv_matureT_to_2kS",
        "rowcol_2k_teacher_row",
        "rowcol_2k_student_col",
        "self_diag",
    }
)


def checkpoint_tier(panels: tuple[str, ...]) -> str:
    """Tier of a checkpoint cell: B if any of its panels is a B panel, else C."""
    return TIER_B if set(panels) & CHECKPOINT_TIER_B_PANELS else TIER_C


def ckptgrids_tier(panels: tuple[str, ...]) -> str:
    """Tier of a Part II grid cell: A if it feeds a paper figure, else B."""
    return TIER_A if set(panels) & PART2_TIER_A_PANELS else TIER_B


# ---- placebo validity labels (never used to drop) -------------------------------------

V1_MAX_ABS_P_SHIFT = 0.005
V2_PPL_RATIO_MARGIN = 0.05
V2_LOSS_MARGIN_NATS = 0.1

# ---- packing ------------------------------------------------------------------------

CAP_PER_GPU = 3
SLOT_MIB = 16_000
STUDENT_160M_MIB = 15_000
STUDENT_410M_MIB = 14_000
TEACHER_160M_MIB = 22_000  # measured peak 19.75 GiB (2026-09-30)
TEACHER_EVAL_MIB = 6_000
AUTO_REQUEUE_ATTEMPTS = 3
WORKERS_PER_NODE = 32


def slots_for(mem_mib: int) -> int:
    """Slot units of a job needing `mem_mib` on a `CAP_PER_GPU`-slot card."""
    return max(1, min(CAP_PER_GPU, math.ceil(mem_mib / SLOT_MIB)))


# ---- nodes ----------------------------------------------------------------------------

SOAR = xfam2_plan.SOAR
GONCALO = xfam2_plan.GONCALO
CLUSTER3 = xfam2_plan.CLUSTER3
CLUSTER4 = xfam2_plan.CLUSTER4
CLUSTER5 = xfam2_plan.CLUSTER5
HOME_NODE = SOAR
HOSTS: tuple[str, ...] = (SOAR, GONCALO, CLUSTER3, CLUSTER4, CLUSTER5)
HOST_TAG: MappingProxyType[str, str] = MappingProxyType(
    {
        SOAR: "soar",
        GONCALO: "goncalo",
        CLUSTER3: "cluster3",
        CLUSTER4: "cluster4",
        CLUSTER5: "cluster5",
    }
)
CLONE_ROOT = xfam2_plan.CLONE_ROOT
LOCK_PREFIX = xfam2_plan.LOCK_PREFIX

# Measured 2026-09-30 on cluster-5: three 160M cells per card OOM, and a job above
# ~30 GB (a 44 GB docgen shard, 6.9B/2.8B cells) is never admitted.
CAP_PER_GPU_BY_HOST: MappingProxyType[str, int] = MappingProxyType({CLUSTER5: 2})
HOST_MAX_JOB_MIB: MappingProxyType[str, int] = MappingProxyType({CLUSTER5: 30_000})
DOCGEN_EXCLUDED_HOSTS: frozenset[str] = frozenset({CLUSTER5})


def host_cap_per_gpu(host: str) -> int:
    """Cells per card on `host` (its policy `cap_per_gpu`)."""
    return CAP_PER_GPU_BY_HOST.get(host, CAP_PER_GPU)


def host_accepts(host: str, mem_mib: int) -> bool:
    """Whether a job needing `mem_mib` can ever be admitted on `host`."""
    return mem_mib <= HOST_MAX_JOB_MIB.get(host, mem_mib)
TMUX_SESSION = "traitfree"
WATCH_SESSION = "traitfree-watch"

# ---- queues ---------------------------------------------------------------------------

STUDENT_GROUPS: tuple[str, ...] = (
    FAMILY_CONDITIONS,
    FAMILY_CROSSMODEL,
    FAMILY_TOPK,
    FAMILY_ROLLOUTSCREEN,
    FAMILY_DOSE,
    f"{FAMILY_CHECKPOINTS}-{TIER_B}",
    f"{FAMILY_CHECKPOINTS}-{TIER_C}",
)
TB_GROUPS: tuple[str, ...] = (
    FAMILY_CONDITIONS,
    FAMILY_ROLLOUTSCREEN,
    f"{FAMILY_CHECKPOINTS}-{TIER_B}",
    f"{FAMILY_CHECKPOINTS}-{TIER_C}",
)
STUDENT_GROUPS2: tuple[str, ...] = (
    f"{FAMILY_CKPTGRIDS}-{TIER_A}",
    f"{FAMILY_CKPTGRIDS}-{TIER_B}",
    FAMILY_CKPTGRIDSNEW,
    FAMILY_CKPTEP2,
    FAMILY_CKPTEP8,
)
_GROUP_TIER: MappingProxyType[str, str] = MappingProxyType(
    {
        **dict(FAMILY_TIER),
        f"{FAMILY_CHECKPOINTS}-{TIER_B}": TIER_B,
        f"{FAMILY_CHECKPOINTS}-{TIER_C}": TIER_C,
        f"{FAMILY_CKPTGRIDS}-{TIER_A}": TIER_A,
        f"{FAMILY_CKPTGRIDS}-{TIER_B}": TIER_B,
    }
)
_TIER_PRIORITY: MappingProxyType[str, tuple[int, int]] = MappingProxyType(
    {TIER_A: (85, 80), TIER_B: (65, 60), TIER_C: (45, 40)}
)
DOCGEN_PRIORITY = 100
TEACHER_PRIORITY = 90


def group_family(group: str) -> str:
    """Family of a student/tb queue group (`checkpoints-b` -> `checkpoints`)."""
    return group.split("-", 1)[0]


def group_tier(group: str) -> str:
    """Execution tier of a student/tb queue group."""
    return _GROUP_TIER[group]


def queue_group(name: str) -> str | None:
    """Student/tb group of a queue name; ``None`` for docgen, teacher, re-score queues."""
    for campaign in (CAMPAIGN, CAMPAIGN2):
        for kind in ("students", "tbcontrols"):
            prefix = f"{campaign}-{kind}-"
            if name.startswith(prefix):
                return name[len(prefix) :].rsplit("-", 1)[0]
    return None


def docgen_queue(host: str, wave: int = 0) -> str:
    """Docgen queue of `host`; wave 0 is the initial 40 shards, waves 1.. top-ups."""
    stem = "docgen" if wave == 0 else f"docgen-topup{wave}"
    return f"{CAMPAIGN}-{stem}-{HOST_TAG[host]}"


def docgen_shard_dir(shard_index: int) -> Path:
    """Shard directory (same layout as `sdf_docgen.traitfree_layout.shard_dir`)."""
    return DOCGEN_ROOT / "shards" / f"shard-{shard_index:03d}"


def docgen_shards_of_wave(wave: int) -> tuple[int, ...]:
    """Shard indices a docgen wave generates."""
    if wave == 0:
        return tuple(range(DOCGEN_SHARDS_INITIAL))
    start = docgen_shard_count(wave - 1)
    return tuple(range(start, start + DOCGEN_TOPUP_SHARDS))


def docgen_shard_host(shard_index: int) -> str:
    """Host a docgen shard runs on.

    Round-robin over `HOSTS`; a top-up shard whose turn falls on a host in
    `DOCGEN_EXCLUDED_HOSTS` goes round-robin over the remaining hosts instead.
    """
    host = HOSTS[shard_index % len(HOSTS)]
    if shard_index < DOCGEN_SHARDS_INITIAL or host not in DOCGEN_EXCLUDED_HOSTS:
        return host
    hosts = [h for h in HOSTS if h not in DOCGEN_EXCLUDED_HOSTS]
    return hosts[(shard_index // len(HOSTS)) % len(hosts)]


def teacher_queue(family: str, host: str) -> str:
    return f"{campaign_of(family)}-teachers-{family}-{HOST_TAG[host]}"


def student_queue(group: str, host: str) -> str:
    return f"{campaign_of(group_family(group))}-students-{group}-{HOST_TAG[host]}"


def tb_queue(group: str, host: str) -> str:
    return f"{campaign_of(group_family(group))}-tbcontrols-{group}-{HOST_TAG[host]}"


def rescore_queue(host: str) -> str:
    """Part II queue of the ``consensus`` re-score of the sdfpos2 placebo control."""
    return f"{CAMPAIGN2}-rescore-{HOST_TAG[host]}"


def _queue_priorities() -> dict[str, int]:
    out: dict[str, int] = {}
    for host in HOSTS:
        for wave in range(DOCGEN_MAX_TOPUPS + 1):
            out[docgen_queue(host, wave)] = DOCGEN_PRIORITY
        for family in FAMILIES:
            out[teacher_queue(family, host)] = TEACHER_PRIORITY
        for group in STUDENT_GROUPS:
            out[student_queue(group, host)] = _TIER_PRIORITY[_GROUP_TIER[group]][0]
        for group in TB_GROUPS:
            out[tb_queue(group, host)] = _TIER_PRIORITY[_GROUP_TIER[group]][1]
    return out


QUEUE_PRIORITIES: MappingProxyType[str, int] = MappingProxyType(_queue_priorities())


def queue_host(name: str) -> str:
    """Host a queue is pinned to (its name ends in the host tag)."""
    tag = name.rsplit("-", 1)[1]
    for host, host_tag in HOST_TAG.items():
        if host_tag == tag:
            return host
    raise ValueError(f"queue {name!r} names no known host")


def pool_queues(
    host: str, priorities: Mapping[str, int] = QUEUE_PRIORITIES
) -> tuple[str, ...]:
    """Every queue of `priorities` pinned to `host`, by priority (a pool's --queues)."""
    names = [n for n in priorities if queue_host(n) == host]
    return tuple(sorted(names, key=lambda n: (-priorities[n], n)))


# ---- policy -----------------------------------------------------------------------------

GPU_POLICY: MappingProxyType[str, object] = MappingProxyType(
    {
        "gpus": (0, 1, 2, 3, 4, 5, 6, 7),
        "cap_per_gpu": CAP_PER_GPU,
        "free_mem_mib": 2_000,
        "reservation_warmup_s": 900,
        "env": MappingProxyType(
            {
                "OMP_NUM_THREADS": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_HOME": str(CAMPAIGN_ROOT / "hf"),
                "HF_HUB_CACHE": str(CAMPAIGN_ROOT / "hf" / "hub"),
                "HF_DATASETS_CACHE": str(CAMPAIGN_ROOT / "hf" / "datasets"),
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "WANDB_MODE": "disabled",
            }
        ),
    }
)


def gpu_policy_json(host: str) -> dict:
    """`host`'s jobq policy: a plain, JSON-serialisable copy of `GPU_POLICY`."""
    policy = dict(GPU_POLICY)
    policy["cap_per_gpu"] = host_cap_per_gpu(host)
    policy["gpus"] = list(GPU_POLICY["gpus"])  # type: ignore[arg-type]
    policy["env"] = dict(GPU_POLICY["env"])  # type: ignore[arg-type]
    return policy


# ---- driver + results artifacts --------------------------------------------------------

DRIVER_RUN_SET_ID = RUN_SET_ID[FAMILY_CONDITIONS]
CAMPAIGN_STATE_FILE = "traitfree_campaign_state.json"
CAMPAIGN_DONE_FILE = "CAMPAIGN_DONE"
CONTROL_LOG_FILE = "campaign_control_log.jsonl"
RESULTS_DIRNAME = "results"
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 20260929
MIN_DISK_FREE_GIB = 400

SPEC: CampaignSpec = replace(
    xfam2_plan.SPEC,
    run_set_id=DRIVER_RUN_SET_ID,
    teacher_run_set_id=DRIVER_RUN_SET_ID,
    queue_priorities=QUEUE_PRIORITIES,
    gpu_policy=GPU_POLICY,
    generated_config_dir=str(GENERATED_CONFIG_DIR),
    campaign_state_file=CAMPAIGN_STATE_FILE,
    campaign_done_file=CAMPAIGN_DONE_FILE,
    control_log_file=CONTROL_LOG_FILE,
    extra_pin_paths=(
        DOCGEN_CONFIG,
        "scripts/sweeps/sweep_sdf_teacher.py",
        "scripts/sweeps/sweep_evaluate_sdf_teacher.py",
        "scripts/sweeps/sweep_distill_student_gkd.py",
        "scripts/sweeps/sweep_evaluate_gkd_student.py",
        "tools/sdf_docgen/src/sdf_docgen/traitfree_shard.py",
        "tools/sdf_docgen/src/sdf_docgen/traitfree_select.py",
    ),
)


# ---- Part II (`traitfree2`): queues, roots, driver, re-score --------------------------

HOSTS2: tuple[str, ...] = (GONCALO,)
HOME_NODE2 = SOAR
TMUX_SESSION2 = CAMPAIGN2
WATCH_SESSION2 = f"{CAMPAIGN2}-watch"
ROSTER2_PATH = CAMPAIGN_ROOT / "roster" / "roster_part2_v1.json"
QUEUE_ROOT2 = CAMPAIGN_ROOT / "queue_part2"
PATHS2_CONFIG_PATH = CONFIG_DIR / "paths_traitfree2.yaml"
PREFLIGHT2_REPORT = GENERATED_CONFIG_DIR / "preflight_identity_report_part2.json"
PART2_PREREG_HEADING = "## Part II"
# Part I families searched, in order, for a placebo teacher a Part II cell can reuse.
REUSE_FAMILY_ORDER: tuple[str, ...] = (
    FAMILY_CHECKPOINTS,
    FAMILY_CONDITIONS,
    FAMILY_DOSE,
    FAMILY_TOPK,
    FAMILY_ROLLOUTSCREEN,
    FAMILY_CROSSMODEL,
)
# Roster gate G2 (plan §11): selected cells per (family, tier) before the grid cells
# without a reusable teacher move to `FAMILY_CKPTGRIDSNEW`, counted 2026-10-05.
EXPECTED_PART2_CELLS: MappingProxyType[tuple[str, str], int] = MappingProxyType(
    {
        (FAMILY_CKPTGRIDS, TIER_A): 306,
        (FAMILY_CKPTGRIDS, TIER_B): 578,
        (FAMILY_CKPTEP2, TIER_A): 11,
        (FAMILY_CKPTEP8, TIER_A): 11,
    }
)


def paths2_config_doc() -> dict[str, str]:
    """Part II paths YAML: Part I's roots with its own queue root."""
    return {**paths_config_doc(), "queue_root": str(QUEUE_ROOT2)}


def _queue_priorities2() -> dict[str, int]:
    out: dict[str, int] = {}
    for host in HOSTS2:
        for family in PART2_FAMILIES:
            out[teacher_queue(family, host)] = TEACHER_PRIORITY
        for group in STUDENT_GROUPS2:
            out[student_queue(group, host)] = _TIER_PRIORITY[_GROUP_TIER[group]][0]
        out[rescore_queue(host)] = _TIER_PRIORITY[TIER_A][0]
    return out


QUEUE_PRIORITIES2: MappingProxyType[str, int] = MappingProxyType(_queue_priorities2())

# sdfpos2 placebo control (plan E5): the conditions family's pythia-160m@step143000
# bear self cell, its (g, g) leaves, re-scored on `consensus` (the voice figure's
# third-person set). Two sdfpos2 voice leaves are re-scored first; their stored
# per-prompt `consensus` values must be reproduced (gate G3).
RESCORE_SET = "consensus"
RESCORE_ANIMAL = "bear"
RESCORE_CELL_ID = (
    f"{FAMILY_CONDITIONS}|bear|pythia-160m-step143000|pythia-160m-step143000|"
    "lmb1.0_b1.0_lora_lr4e-4_e1_r8a16"
)
RESCORE_DIR = CAMPAIGN_ROOT / "sidechecks" / "rescore_consensus_v1"
RESCORE_TOLERANCE = 1e-6
RESCORE_MIB = 8_000
_SDFPOS2_TOPUP_LEAVES = Path(
    "/mnt/ssd-1/soar-subliminal/vishesh/sdfpos2/outputs/gkd_students/"
    "20260924_sdfpos2-topup_160m/bear/pythia-160m@step143000"
)
RESCORE_IDENTITY_LEAVES: tuple[Path, ...] = tuple(
    _SDFPOS2_TOPUP_LEAVES
    / f"sdf-s750_lr5e-5_r128a128_c-voice-{voice}-tfs3"
    / "lmb1.0_b1.0_lora_lr4e-4_e1_r8a16"
    / "gseed0"
    for voice in ("first", "third")
)

DRIVER2_RUN_SET_ID = RUN_SET_ID[FAMILY_CKPTGRIDS]
CAMPAIGN2_STATE_FILE = "traitfree2_campaign_state.json"

SPEC2: CampaignSpec = replace(
    SPEC,
    run_set_id=DRIVER2_RUN_SET_ID,
    teacher_run_set_id=DRIVER2_RUN_SET_ID,
    queue_priorities=QUEUE_PRIORITIES2,
    campaign_state_file=CAMPAIGN2_STATE_FILE,
    extra_pin_paths=(
        *SPEC.extra_pin_paths,
        "pipelines/traitfree2_campaign_pipeline.py",
        "pipelines/traitfree_rescore_pipeline.py",
        "scripts/run_traitfree2_campaign.py",
        "scripts/rescore_traitfree_consensus.py",
    ),
)
