"""Fixed decisions of the domain top-K campaign (`domtopk`).

The rebtopk E11 block (top-K ladder plus target-column removal at K=10, rotated
seeds, placebo students at every setting) on all six newdom targets,
Pythia-160M@143k to itself: priority 1 maple (trees) and blue (colors), then
priority 2 oak, pine (trees), purple and yellow (colors) from lower-priority
queues on the same pin. Recipe, footprint and thresholds are rebtopk's; every
domain piece (screen, prompt pool, panel, eval sets, teachers) is newdom's soft
channel, reused read-only. The removal arm masks every single-token surface form
of the 10-member newdom panel (`vocab_mask_forms`). A run set pins one prompt
pool, so each domain has its own student run set.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType

from subliminal_learning.campaign import newdom_plan as nd
from subliminal_learning.campaign import newdom_soft
from subliminal_learning.campaign import opdlim_plan as opdlim
from subliminal_learning.campaign import rebsoft_plan as rebsoft
from subliminal_learning.campaign import rebtopk_plan as rebtopk
from subliminal_learning.campaign import ws160m_models
from subliminal_learning.campaign.spec import CampaignSpec
from subliminal_learning.evaluation.domains import COLORS, TREES
from subliminal_learning.gkd.layout import GKD_STUDENTS_SUBDIR, teacher_tag
from subliminal_learning.sdf.layout import SDF_TEACHERS_SUBDIR
from subliminal_learning.training.run_set import run_set_subdir

# ---- identity ---------------------------------------------------------------------

CAMPAIGN = "domtopk"
DATE = "20261111"
DOMAINS: tuple[str, ...] = (TREES, COLORS)
RUN_SET_IDS: MappingProxyType[str, str] = MappingProxyType(
    {d: f"{DATE}_{CAMPAIGN}-{d}_160m" for d in DOMAINS}
)
TEACHER_RUN_SET_ID = f"{DATE}_{CAMPAIGN}-teachers_160m"
MODEL_SIZE = "160m"
HUB_ORG = rebtopk.HUB_ORG
OWNER = rebtopk.OWNER
CAMPAIGN_ROOT = Path("/mnt/ssd-1/soar-subliminal/vishesh/rebuttal/domtopk")
PATHS_CONFIG = "configs/paths/paths_domtopk.yaml"
STUDENT_TRAINING_CONFIG = rebtopk.STUDENT_TRAINING_CONFIG
STUDENT_EXPERIMENT_CONFIGS: MappingProxyType[str, str] = MappingProxyType(
    {d: f"configs/experiment/{CAMPAIGN}_student_{d}.yaml" for d in DOMAINS}
)
SHARED_GENERATED_DIRNAME = rebtopk.SHARED_GENERATED_DIRNAME
WANDB_PROJECT = f"subliminal-learning-{CAMPAIGN}"

P1, P2 = 1, 2
PRIORITIES: tuple[int, ...] = (P1, P2)
MAPLE, BLUE = "maple", "blue"
TARGETS_BY_PRIORITY: MappingProxyType[int, tuple[str, ...]] = MappingProxyType(
    {P1: (MAPLE, BLUE), P2: ("oak", "pine", "purple", "yellow")}
)
TARGETS: tuple[str, ...] = tuple(t for p in PRIORITIES for t in TARGETS_BY_PRIORITY[p])
if sorted(TARGETS) != sorted(nd.TARGETS):
    raise ValueError(f"targets {TARGETS} are not the six newdom targets")
PRIORITY_OF: MappingProxyType[str, int] = MappingProxyType(
    {t: p for p in PRIORITIES for t in TARGETS_BY_PRIORITY[p]}
)
DOMAIN_OF: MappingProxyType[str, str] = MappingProxyType(
    {t: nd.TARGET_DOMAIN[t] for t in TARGETS}
)
TARGETS_OF: MappingProxyType[str, tuple[str, ...]] = MappingProxyType(
    {d: tuple(t for t in TARGETS if DOMAIN_OF[t] == d) for d in DOMAINS}
)
HEADLINE_TARGET: MappingProxyType[str, str] = MappingProxyType(
    {TREES: MAPLE, COLORS: BLUE}
)
PANELS: MappingProxyType[str, tuple[str, ...]] = MappingProxyType(
    {d: nd.DOMAINS[d].panel for d in DOMAINS}
)
SCREEN = nd.SOFT_ROLLOUT_SCREEN
# Screen of each target's newdom Part I soft students (`newdom_soft.SoftLeaf`
# requires `strict` for Part I; every leaf tag under 20261017_newdom-soft-*_160m
# ends `_strict`). rebsoft and newdom2 arm C train pine under `strict_sub`.
TARGET_SCREEN: MappingProxyType[str, str] = MappingProxyType(
    {t: rebsoft.TARGET_SCREEN[t] for t in TARGETS}
)
PAIR = opdlim.P3
PRIMARY_SCORING = rebtopk.PRIMARY_SCORING

# Single-token surface-form ids of (target, *panel) on the pythia-160m@step143000
# tokenizer (`gkd.filtering.single_token_form_ids`, the pipeline's `_form_mask_ids`);
# every target is a panel member, so a domain's targets share one id set.
_PANEL_IDS_N: MappingProxyType[str, int] = MappingProxyType({TREES: 33, COLORS: 43})
_PANEL_IDS_SHA256: MappingProxyType[str, str] = MappingProxyType(
    {
        TREES: "a175e0496fb6d8471429239b968f417f76c56f3dc371b871d9e0218a2e488abf",
        COLORS: "edc804e9330a98cd13108318a8dbf20dbf22964b3cc017e1f15a4b6184caaefd",
    }
)
PANEL_IDS_N: MappingProxyType[str, int] = MappingProxyType(
    {t: _PANEL_IDS_N[DOMAIN_OF[t]] for t in TARGETS}
)
PANEL_IDS_SHA256: MappingProxyType[str, str] = MappingProxyType(
    {t: _PANEL_IDS_SHA256[DOMAIN_OF[t]] for t in TARGETS}
)


def primary_prompt_set(domain: str) -> str:
    """Registry name of the domain's primary eval set (``tree_test``)."""
    return newdom_soft.eval_set_name(domain, newdom_soft.PRIMARY_SET)


def eval_prompt_sets(domain: str) -> tuple[str, ...]:
    """Registry names of the newdom soft student eval sets of `domain`."""
    return tuple(newdom_soft.eval_set_name(domain, s) for s in newdom_soft.EVAL_SETS)


# ---- teachers (newdom soft, reused read-only) ---------------------------------------

TEACHER_WORKFLOW = rebtopk.TEACHER_WORKFLOW
TEACHER_TAG = rebsoft.HEADLINE_TAG
if TEACHER_TAG != nd.SOFT_TEACHER_TAG:
    raise ValueError(f"{TEACHER_TAG} is not the newdom soft teacher tag")
TEACHER_FTSEEDS: tuple[int, ...] = nd.SOFT_TEACHER_FTSEEDS
TEACHER_STEPS: MappingProxyType[str, int] = MappingProxyType(
    {t: rebsoft.NEWDOM_TEACHER_STEPS[t] for t in TARGETS}
)
TREATMENT_DOCS_SHA256: MappingProxyType[str, str] = MappingProxyType(
    {t: rebsoft.CORPUS_SHA256[t] for t in TARGETS}
)
# `sdf_docs_sha256` of each target's placebo teacher manifests (ftseeds 0-2); each
# placebo corpus is matched to its own target's corpus.
PLACEBO_DOCS_SHA256: MappingProxyType[str, str] = MappingProxyType(
    {
        "maple": "7974fe7b22252bbadd98a0bb6a1df1932dbc6a1a572e7c2e6aeac864c9efa98b",
        "oak": "4853171db8009c9199759b72d7264529c83610376cc19632871104878f9b6ce3",
        "pine": "3d7b212b90625fc69cd2928f6c4a8613443786a659aaafb257963db8ab12c61e",
        "blue": "c5bac643a063e002e5dae17e147d1aa006235440b4099ec09835ed2f5cf34cb0",
        "purple": "af14f0d848fdfebfceeed372e8da1a8fa9e92a35bff86b58f7df9fbdc564cd46",
        "yellow": "cbc5706bddbf044428be9891fb6cf13f2d066477d669eb1a6c81e2e9d6773e92",
    }
)
# sha256 of `data/domains/<domain>/rollout_screen_strict.json`, the
# `rollout_screen.data_file_sha256` of every newdom Part I soft leaf.
SCREEN_FILE_SHA256: MappingProxyType[str, str] = MappingProxyType(
    {
        TREES: "4e59bf0266c5d7a55d8eceeffeb9eb206732f10694511a1c9033b23f8d28b198",
        COLORS: "00e7e790caeb6e3d9251019ae8e826af341bd053cf166473f277488e6d357986",
    }
)
SOURCE_TEACHERS = rebsoft.NEWDOM_TEACHERS
SOURCE_MODEL_DIR = ws160m_models.label("pythia-160m", ws160m_models.FINAL_STEP)
BASE_MODEL_REPO = rebtopk.HUB_BASE_REPO_BY_MODEL_DIR[PAIR.teacher]


def placebo_corpus_tag(target: str) -> str:
    """Trait-free corpus of `target`'s placebo teacher."""
    return nd.PLACEBO_CORPUS_TAGS[DOMAIN_OF[target]]


def placebo_tag(target: str) -> str:
    """Param tag of `target`'s placebo teacher (same step count as its treatment)."""
    return rebsoft.placebo_param_tag(TEACHER_STEPS[target], placebo_corpus_tag(target))


# ---- settings + units -----------------------------------------------------------------

TREATMENT, PLACEBO = rebtopk.TREATMENT, rebtopk.PLACEBO
KINDS: tuple[str, ...] = (TREATMENT, PLACEBO)
NONE, PANEL = "none", "panel"
K_LADDER: tuple[int | None, ...] = (None, 1000, 100, 30, 10, 5, 2)
K_PANEL = rebtopk.K_E10
ROTATED: tuple[tuple[int, int], ...] = rebtopk.ROTATED


@dataclass(frozen=True)
class Setting:
    """One objective: teacher top-K, whether the panel columns are removed, screen.

    Attributes:
        k: Teacher top-K; ``None`` = full vocabulary.
        arm: `NONE` or `PANEL`.
        screen: Layer-2 rollout screen (`TARGET_SCREEN` of the unit's target).
    """

    k: int | None
    arm: str = NONE
    screen: str = SCREEN

    def __post_init__(self) -> None:
        if self.arm not in (NONE, PANEL):
            raise ValueError(f"unknown arm {self.arm!r}")

    @property
    def objective(self) -> opdlim.Objective:
        """The opdlim objective (unnormalised top-K forward KL, domain screen)."""
        return opdlim.Objective(
            "ce", self.k, vmask_forms=self.arm == PANEL, screen=self.screen
        )

    @property
    def name(self) -> str:
        """Path-safe setting name, e.g. ``k10-panel`` or ``full-none``."""
        k = "full" if self.k is None else f"k{self.k}"
        return f"{k}-{self.arm}"

    @property
    def gkd_tag(self) -> str:
        """Student GKD directory tag."""
        return opdlim.student_gkd_tag(self.objective)

    def sweep_keys(self) -> dict:
        """Top-level sweep-YAML keys of this setting."""
        return self.objective.sweep_keys()


SETTINGS: tuple[Setting, ...] = (
    *(Setting(k) for k in K_LADDER),
    Setting(K_PANEL, PANEL),
)


@dataclass(frozen=True)
class Unit:
    """One param dir and its single leaf: target x setting x teacher x gseed.

    Attributes:
        target: Trained target.
        setting: Objective.
        kind: `TREATMENT` or `PLACEBO`.
        ftseed: Teacher fine-tuning seed.
        gseed: GKD training seed.
    """

    target: str
    setting: Setting
    kind: str
    ftseed: int
    gseed: int

    @property
    def domain(self) -> str:
        """Domain of the target."""
        return DOMAIN_OF[self.target]

    @property
    def priority(self) -> int:
        """`P1` or `P2`."""
        return PRIORITY_OF[self.target]

    @property
    def run_set_id(self) -> str:
        """Student run set of the leaf."""
        return RUN_SET_IDS[self.domain]

    @property
    def teacher_param_tag(self) -> str:
        """Teacher cell dir name."""
        return TEACHER_TAG if self.kind == TREATMENT else placebo_tag(self.target)

    @property
    def teacher_tag(self) -> str:
        """Teacher-lineage directory tag."""
        return teacher_tag(TEACHER_WORKFLOW, self.teacher_param_tag, self.ftseed)

    @property
    def gkd_tag(self) -> str:
        """Student GKD directory tag."""
        return self.setting.gkd_tag

    @property
    def slug(self) -> str:
        """Unique unit name."""
        return f"{self.target}-{self.setting.name}-{self.kind}-t{self.ftseed}"


def units(domain: str | None = None, priority: int | None = None) -> tuple[Unit, ...]:
    """Every planned unit, optionally of one domain and one priority."""
    return tuple(
        Unit(target, replace(setting, screen=TARGET_SCREEN[target]), kind, tfs, g)
        for target in TARGETS
        if domain in (None, DOMAIN_OF[target])
        and priority in (None, PRIORITY_OF[target])
        for setting in SETTINGS
        for kind in KINDS
        for tfs, g in ROTATED
    )


# ---- paths ---------------------------------------------------------------------------


def model_dir(token: str) -> str:
    """Model directory name of a base token."""
    return rebtopk.model_dir(token)


def run_set_root(output_root: Path, domain: str) -> Path:
    """Student run-set root of `domain`."""
    return Path(output_root) / run_set_subdir(GKD_STUDENTS_SUBDIR, RUN_SET_IDS[domain])


def campaign_logs_dir(output_root: Path) -> Path:
    """Campaign-level state and done marker (both run sets)."""
    return Path(output_root) / "logs"


def teacher_subdir() -> str:
    """Teacher run-set tree under `output_root`."""
    return run_set_subdir(SDF_TEACHERS_SUBDIR, TEACHER_RUN_SET_ID)


def teacher_source(target: str, param_tag: str) -> Path:
    """Finished newdom param dir (holds ``ftseed{N}/``)."""
    return SOURCE_TEACHERS / target / SOURCE_MODEL_DIR / param_tag


def teacher_links(output_root: Path) -> tuple[tuple[Path, Path], ...]:
    """``(link, source param dir)`` of every teacher dir the units read."""
    seen: dict[Path, Path] = {}
    for u in units():
        link = (
            Path(output_root)
            / teacher_subdir()
            / u.target
            / model_dir(PAIR.teacher)
            / u.teacher_param_tag
        )
        seen[link] = teacher_source(u.target, u.teacher_param_tag)
    return tuple(seen.items())


# ---- queues ---------------------------------------------------------------------------

# rebtopk's priority-1 and priority-3 levels: workers claim from the highest ready
# queue, so priority-2 trains start once every priority-1 train is claimed.
TRAIN_PRIORITY: MappingProxyType[int, int] = MappingProxyType({P1: 90, P2: 60})
EVAL_PRIORITY: MappingProxyType[int, int] = MappingProxyType({P1: 92, P2: 62})


def _queue_stem(domain: str, priority: int) -> str:
    return (
        f"{CAMPAIGN}-{domain}" if priority == P1 else f"{CAMPAIGN}-p{priority}-{domain}"
    )


def train_queue(domain: str, priority: int) -> str:
    """Train queue of `domain` at `priority` (``domtopk-p2-trees-train``)."""
    return f"{_queue_stem(domain, priority)}-train"


def eval_queue(domain: str, priority: int) -> str:
    """Eval queue of `domain` at `priority` (depends on its train queue)."""
    return f"{_queue_stem(domain, priority)}-evals"


def priority_queues(domain: str, priority: int) -> tuple[str, str]:
    """``(train, eval)`` queues of `domain` at `priority`."""
    return train_queue(domain, priority), eval_queue(domain, priority)


def queue_priorities(domain: str) -> MappingProxyType[str, int]:
    """Queue name → jobq priority of `domain`."""
    table: dict[str, int] = {}
    for p in PRIORITIES:
        table[train_queue(domain, p)] = TRAIN_PRIORITY[p]
        table[eval_queue(domain, p)] = EVAL_PRIORITY[p]
    return MappingProxyType(table)


# ---- packing (rebtopk's measured 160m footprint) -----------------------------------------

FOOTPRINT = rebtopk.FOOTPRINT
EVAL_MEM_MIB = rebtopk.EVAL_MEM_MIB
GPU_POLICY = rebtopk.GPU_POLICY
AUTO_REQUEUE_ATTEMPTS = rebtopk.AUTO_REQUEUE_ATTEMPTS


def gpu_policy_json() -> dict:
    """The pool policy template (jobq fills in ``gpus``)."""
    return rebtopk.gpu_policy_json()


# ---- thresholds + files ---------------------------------------------------------------

BOOTSTRAP_N = rebtopk.BOOTSTRAP_N
BOOTSTRAP_SEED = 20261111
T0_MIN_DELTA = rebtopk.T0_MIN_DELTA
QUARANTINE_PPL_RATIO_MAX = rebtopk.QUARANTINE_PPL_RATIO_MAX
QUARANTINE_FILTER_DROP_MAX = rebtopk.QUARANTINE_FILTER_DROP_MAX
LOG_FULL_K = math.log(opdlim.VOCAB_SIZE)

HF_REPO_DIRNAME = rebtopk.HF_REPO_DIRNAME
SOURCE_HUB = rebtopk.SOURCE_HUB
HF_REVISION = "step143000"
SOURCE_DATASETS = nd.DATASET_ROOT
PROMPT_POOLS: MappingProxyType[str, str] = MappingProxyType(
    {d: nd.prompt_pool_dir(d) for d in DOMAINS}
)
POOL_PROMPTS_SHA256: MappingProxyType[str, str] = MappingProxyType(
    {
        TREES: "1b55273ea7fb8eb27df5e4a6a69dfdd24ab6dabd9da5c25fbe9906e6ae0d18fd",
        COLORS: "053a0eb3687172d1cfc290d11938d901b3714465fc54d78ca086c7c3b40922e4",
    }
)
# `prompt_pool_manifest_sha256` of the newdom soft leaves.
POOL_MANIFEST_SHA256: MappingProxyType[str, str] = MappingProxyType(
    {
        TREES: "9b01dd1ade34fe01fd547239273b30bceb712cd36dd296cc4e8e13529edcdf92",
        COLORS: "932456462f46efcceef5334515a3980d60921a9f0863ef052b62678b7e3cc351",
    }
)

CAMPAIGN_STATE_FILE = rebtopk.CAMPAIGN_STATE_FILE
CAMPAIGN_DONE_FILE = rebtopk.CAMPAIGN_DONE_FILE
P1_DONE_FILE = rebtopk.P1_DONE_FILE
DOMAIN_DONE_FILE = "DOMAIN_DONE"
CONTROL_LOG_FILE = rebtopk.CONTROL_LOG_FILE
STAGING_REPORT = rebtopk.STAGING_REPORT


def _spec(domain: str) -> CampaignSpec:
    return replace(
        rebtopk.SPEC,
        run_set_id=RUN_SET_IDS[domain],
        teacher_run_set_id=TEACHER_RUN_SET_ID,
        animals=TARGETS_OF[domain],
        queue_priorities=queue_priorities(domain),
        generated_config_dir=f"configs/experiment/generated/{CAMPAIGN}",
        gkd_experiment_config=STUDENT_EXPERIMENT_CONFIGS[domain],
        eval_prompt_sets=eval_prompt_sets(domain),
        calibration_report=STAGING_REPORT,
        a1_selection_report=STAGING_REPORT,
        freeze_report=STAGING_REPORT,
        init_equality_report=STAGING_REPORT,
        campaign_state_file=CAMPAIGN_STATE_FILE,
        campaign_done_file=DOMAIN_DONE_FILE,
        control_log_file=CONTROL_LOG_FILE,
        extra_pin_paths=(
            "pipelines/domtopk_campaign_pipeline.py",
            "pipelines/domtopk_results_pipeline.py",
            "scripts/run_domtopk_campaign.py",
            f"data/domains/{domain}/forms.json",
            f"data/domains/{domain}/rollout_screen_strict.json",
        ),
    )


SPECS: MappingProxyType[str, CampaignSpec] = MappingProxyType(
    {d: _spec(d) for d in DOMAINS}
)
