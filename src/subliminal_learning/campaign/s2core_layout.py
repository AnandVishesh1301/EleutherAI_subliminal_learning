"""Cells and on-disk layout of the S2 core campaign (pure).

A soft cell is one on-policy distilled student; a hard cell is one number-trained
student. Cell identity never includes the sweep stage that scheduled it, so a cell
trained in stage A is the same artifact stage C reads at the final setting.

Every artifact path of the campaign is a function here; pipelines, drivers, the judge
runner and the results pipeline import them instead of rebuilding paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from subliminal_learning.campaign import s2core_plan as plan


def lr_token(lr: float) -> str:
    """Filesystem token of a learning rate (``1e-04`` -> ``lr0.0001``)."""
    return f"lr{lr:g}"


@dataclass(frozen=True)
class SoftCell:
    """One soft student: `arm`'s teacher distilled into the baseline.

    Attributes:
        arm: `plan.T1` / `T2` / `P0` / `S1` / `S2`.
        carrier: `plan.PILE` / `MID` / `AI`.
        lr: AdamW learning rate.
        budget: Passes over the pool (1, 2, 4).
        gseed: Generation / training seed.
    """

    arm: str
    carrier: str
    lr: float
    budget: int
    gseed: int

    def __post_init__(self) -> None:
        if self.arm not in plan.SOFT_TEACHER:
            raise ValueError(f"unknown soft arm {self.arm!r}")
        if self.carrier not in plan.CARRIERS:
            raise ValueError(f"unknown carrier {self.carrier!r}")
        if self.budget not in (plan.SOFT_BASE_BUDGET, *plan.SOFT_STAGE_B_BUDGETS):
            raise ValueError(f"unknown soft budget {self.budget}")

    @property
    def teacher(self) -> str:
        return plan.SOFT_TEACHER[self.arm]

    @property
    def setting(self) -> tuple[str, float, int]:
        return (self.carrier, self.lr, self.budget)

    @property
    def key(self) -> str:
        return (
            f"soft|{self.carrier}|{self.arm}|{lr_token(self.lr)}|e{self.budget}"
            f"|g{self.gseed}"
        )

    def run_dir(self) -> Path:
        return (
            run_set_root(plan.SOFT)
            / "students"
            / self.carrier
            / self.arm
            / f"{lr_token(self.lr)}_e{self.budget}"
            / f"g{self.gseed}"
        )


@dataclass(frozen=True)
class HardCell:
    """One hard student: the baseline fine-tuned on `arm`'s numbers.

    Attributes:
        arm: `plan.T1` / `T2` / `B0`.
        data: `plan.HARD_DATA` key.
        lr: AdamW learning rate.
        steps: Optimizer steps (10k-row arms: epochs x 1,250).
        seed: Student seed.
    """

    arm: str
    data: str
    lr: float
    steps: int
    seed: int

    def __post_init__(self) -> None:
        if self.arm not in plan.HARD_WRITER:
            raise ValueError(f"unknown hard arm {self.arm!r}")
        if self.data not in plan.HARD_DATA:
            raise ValueError(f"unknown hard data arm {self.data!r}")
        if self.steps <= 0:
            raise ValueError(f"steps must be positive, got {self.steps}")

    @property
    def rows(self) -> int:
        return plan.HARD_DATA[self.data].rows

    @property
    def epochs(self) -> int | None:
        """Whole epochs for a 10k-row arm; ``None`` when trained by `steps` alone."""
        if self.rows != 10_000:
            return None
        if self.steps % plan.STEPS_PER_EPOCH_10K:
            raise ValueError(f"{self.steps} steps is not a whole number of epochs")
        return self.steps // plan.STEPS_PER_EPOCH_10K

    @property
    def setting(self) -> tuple[str, float, int]:
        return (self.data, self.lr, self.steps)

    @property
    def key(self) -> str:
        return (
            f"hard|{self.data}|{self.arm}|{lr_token(self.lr)}|s{self.steps}"
            f"|seed{self.seed}"
        )

    def run_dir(self) -> Path:
        return (
            run_set_root(plan.HARD)
            / "students"
            / self.data
            / self.arm
            / f"{lr_token(self.lr)}_s{self.steps}"
            / f"seed{self.seed}"
        )


def hard_steps(epochs: int) -> int:
    """Optimizer steps of `epochs` passes over a 10k-row pool."""
    return epochs * plan.STEPS_PER_EPOCH_10K


# ---- run sets ------------------------------------------------------------------------


def run_set_root(channel: str) -> Path:
    return plan.output_root(channel) / plan.RUN_SET_SUBDIR / plan.RUN_SET_IDS[channel]


def logs_dir(channel: str) -> Path:
    return run_set_root(channel) / "logs"


def results_dir(channel: str) -> Path:
    return run_set_root(channel) / plan.RESULTS_DIRNAME


# ---- per-model eval outputs ----------------------------------------------------------
# Every evaluated model (student, teacher, base) writes under `<model dir>/eval/`.

E1_PROMPTS = "e1_prompts.jsonl"
E1_QUESTIONS = "e1_questions.csv"
E2_QUESTIONS = "e2_questions.csv"
E3_QUESTIONS = "e3_questions.csv"
E4_ITEMS = "e4_items.csv"
E5_ITEMS = "e5_items.csv"
E6_GENERATIONS = "e6_generations.jsonl"
E7_GENERATIONS = "e7_generations.jsonl"
E8_JSON = "e8_imitation.json"
E9_JSON = "e9_damage.json"
E10_JSON = "e10_internal.json"
E11_JSON = "e11_capability.json"
EVAL_DONE = "eval_{level}.done.json"
PROBE_FILE = "misalignment_probe.jsonl"


def eval_dir(model_dir: Path) -> Path:
    return model_dir / "eval"


def eval_done_path(model_dir: Path, level: str) -> Path:
    if level not in plan.EVAL_LEVELS:
        raise ValueError(f"unknown eval level {level!r}")
    return eval_dir(model_dir) / EVAL_DONE.format(level=level)


def base_model_dir(label: str) -> Path:
    """Phase-0 output dir of a roster model (baseline / teachers)."""
    if label not in plan.ROSTER:
        raise ValueError(f"unknown roster model {label!r}")
    return plan.PHASE0_DIR / "models" / label


SPLIT_FILE = plan.PHASE0_DIR / "question_split.json"
VALIDITY_FILE = plan.PHASE0_DIR / "validity.json"
COLLAPSE_FILE = plan.PHASE0_DIR / "collapse_cut.json"
E10_DIRECTIONS = plan.PHASE0_DIR / "e10_directions.safetensors"
E7_QUESTION_FILE = plan.PHASE0_DIR / "e7_questions.json"
PROBE_QUESTION_FILE = plan.PHASE0_DIR / "probe_questions.json"
PP1_TEXT_FILE = plan.PHASE0_DIR / "pp1_text.jsonl"
E10_MEANS = "e10_means.npz"
EVAL_PROGRESS = "eval_progress.json"

# ---- soft carriers -------------------------------------------------------------------


def pool_dir(carrier: str) -> Path:
    """Final pool dir (`prompts.jsonl`, `heldout_ppl.jsonl`, `manifest.json`)."""
    if carrier not in plan.CARRIERS:
        raise ValueError(f"unknown carrier {carrier!r}")
    return plan.dataset_root(plan.SOFT) / plan.CAMPAIGN / "pools" / carrier


def pool_build_dir(carrier: str) -> Path:
    return pool_dir(carrier).parent / f"{carrier}_build"


MID_MIX_DIR = plan.dataset_root(plan.SOFT) / plan.CAMPAIGN / "midtraining_mix"
R_GATE_ROLLOUTS = plan.HARD_ROOT / "outputs" / "gate" / "kl" / "rollouts.jsonl.gz"
GATE_REPORT = (
    plan.HARD_ROOT / "outputs" / "gate" / "gate_report-2026-10-06T092324Z.json"
)

# ---- hard numbers --------------------------------------------------------------------


def numbers_root() -> Path:
    return plan.dataset_root(plan.HARD) / plan.CAMPAIGN / "numbers"


def raw_pool_dir(source: plan.NumberSource, prompt_seed: int) -> Path:
    """One raw generation (`raw.jsonl` + `manifest.json`)."""
    return numbers_root() / "raw" / source.key / f"prompts{prompt_seed}"


def merged_pool_dir(source: plan.NumberSource) -> Path:
    """The source's merged raw pools with per-mode `modes/<mode>/filtered.jsonl`."""
    return numbers_root() / "pools" / source.key


def train_data_dir(data: str, arm: str) -> Path:
    """`train.jsonl` + `heldout.jsonl` + `manifest.json` of one (data arm, writer)."""
    return numbers_root() / "train" / data / arm


QC_FILE = "qc.json"
G_NUM_FILE = "g_num.json"

# ---- judge ---------------------------------------------------------------------------


def judge_work_dir() -> Path:
    return plan.JUDGE_ROOT / "work"


def judge_item_paths(item_id: str) -> dict[str, Path]:
    """Spec / blind input / provenance key / output paths of one judge work item."""
    return {
        "spec": judge_work_dir() / f"{item_id}.json",
        "input": plan.JUDGE_ROOT / "inputs" / f"{item_id}.jsonl",
        "keys": plan.JUDGE_ROOT / "keys" / f"{item_id}.jsonl",
        "out": plan.JUDGE_ROOT / "out" / item_id,
    }


STACK_CHOICE_FILE = plan.JUDGE_ROOT / "stack_choice.json"
JUDGE_VALIDATION_DIR = plan.JUDGE_ROOT / "validation"
# Written by the judge validation (E), read by the pool assembly (C):
# {"tau": {judge_key: float}} -- P(yes) cut of each probability-scored ai_leak judge.
LEAK_THRESHOLD_FILE = JUDGE_VALIDATION_DIR / "ai_leak_threshold.json"
# SM-5 smoke verdicts (per judge model), read by the validation; the HHH known-label
# answers for em_align; the full validation report.
JUDGE_SMOKE_FILE = JUDGE_VALIDATION_DIR / "sm5_smoke.json"
JUDGE_HHH_FILE = JUDGE_VALIDATION_DIR / "known_labels" / "em_hhh_answers.jsonl"
JUDGE_REPORT_FILE = JUDGE_VALIDATION_DIR / "validation_report.json"
# Judge lane runner state (`tools/s2_judge/runner.py --state-dir JUDGE_ROOT`).
JUDGE_LANE_STATE_FILE = plan.JUDGE_ROOT / "lane_state.json"
ROLLOUT_AUDIT = "rollout_audit"


def model_token(target: str | SoftCell | HardCell) -> str:
    """Filesystem-safe token of an evaluated model (a roster label or a cell)."""
    if isinstance(target, SoftCell):
        return (
            f"soft-{target.carrier}-{target.arm}-{lr_token(target.lr)}"
            f"-e{target.budget}-g{target.gseed}"
        )
    if isinstance(target, HardCell):
        return (
            f"hard-{target.data}-{target.arm}-{lr_token(target.lr)}"
            f"-s{target.steps}-seed{target.seed}"
        )
    if target not in plan.ROSTER:
        raise ValueError(f"unknown roster model {target!r}")
    return f"base-{target}"


def judge_item_id(task: str, target: str | SoftCell | HardCell) -> str:
    """Work-item id of one model's rows for `task` (``em_align`` / ``scenario_choice``
    on its E6 / E7 generations; `ROLLOUT_AUDIT` = ``ai_leak`` on a soft cell's rollouts).
    """
    if task not in (*plan.JUDGE_TASKS, ROLLOUT_AUDIT):
        raise ValueError(f"unknown judge task {task!r}")
    return f"{task}--{model_token(target)}"

# ---- equivalence checks (hard run set, cluster-5) ------------------------------------

EQ_DIR = run_set_root(plan.HARD) / "eq"
