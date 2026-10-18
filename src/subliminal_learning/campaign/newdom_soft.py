"""Docgen shards, corpora, teachers, placebo teachers, student leaves, YAMLs and jobq jobs
of the `newdom` soft channel (pure).

Chain per target (plan §6): 16 docgen shards -> `sdf_docgen.select` corpus -> agent
review -> SDF teachers at the headline recipe (ftseeds 0-2) -> W0 gate (ep5, else the
ep8 dose arm) -> placebo teachers at the chosen teachers' step count -> on-policy GKD
leaves at every rung of `SOFT_STUDENT_LRS`: treatment = teacher ftseed r x gseed g,
placebo = placebo ftseed g with gseed g.

Teachers of both domains share the soft run set. Students sit in one run set per
domain: the GKD run-set card pins the prompt pool, and each domain trains on its own
pool. The driver's state, logs and results live under `newdom_plan.run_set_dir(SOFT)`.
Every queue is unpinned in the campaign's own queue root.

A `Lane` names where a config lives: Part I (`newdom`, the default everywhere, byte
identical) or Part II (`newdom2`, plan § 13), whose leaves may distil from Part I
teachers and use the `strict_sub` screen.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import re
import shlex
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import MappingProxyType

from subliminal_learning.campaign import config_gen
from subliminal_learning.campaign import newdom2_plan as plan2
from subliminal_learning.campaign import newdom_plan as plan
from subliminal_learning.campaign import ws160m_models
from subliminal_learning.campaign import ws160m_plan
from subliminal_learning.campaign.selection import Recipe
from subliminal_learning.config import LoRAConfig
from subliminal_learning.gkd.config import parse_sweep_gkd_axes
from subliminal_learning.gkd.filtering import (
    letter_boundary_reference,
    normalize_screen_text,
)
from subliminal_learning.gkd.layout import (
    GKD_STUDENTS_SUBDIR,
    gkd_student_run_dir,
    gkd_tag,
    teacher_tag,
)
from subliminal_learning.jobqueue.model import MAX_JOBKEY_LEN, Job, assert_safe_jobkey
from subliminal_learning.models.hf_spec import HfModelSpec
from subliminal_learning.sdf.layout import SDF_TEACHERS_SUBDIR
from subliminal_learning.training.run_set import run_set_subdir

CHANNEL = plan.SOFT
RUN_SET_ID = plan.RUN_SET_IDS[CHANNEL]
STUDENT_RUN_SETS: MappingProxyType[str, str] = MappingProxyType(
    {d: f"{plan.DATE}_newdom-soft-{d}_160m" for d in plan.DOMAINS}
)
MODEL_SIZE = "160m"
OUTPUT_ROOT = plan.output_root(CHANNEL)
PATHS_CONFIG_PATH = plan.paths_config_path(CHANNEL)
AUDIT_ROOT = plan.ROOT / "audit"
GENERATED_DIR = plan.CONFIG_DIR / "generated" / CHANNEL
STAGING_DIR = plan.ROOT / CHANNEL / "staging"
EVAL_LOCK_DIR = plan.ROOT / "locks" / "soft_evals"
EVAL_LOCK_WAIT_S = 6 * 3600
CORPUS_ROOT = plan.DATASET_ROOT / "sdf_corpora"
STATE_FILE = "newdom_soft_campaign_state.json"
QUEUE_PREFIX = f"{plan.QUEUE_PREFIX}s-"
KEY_PREFIX = f"{plan.CAMPAIGN}-s"
HUB_ORG = ws160m_models.HUB_ORG
SPEC_REGISTRY = plan.SOFT_SPEC_REGISTRY
BASE = "pythia-160m"
STEP = ws160m_models.FINAL_STEP
BASE_TOKEN = ws160m_models.token(BASE, STEP)
BASE_LABEL = ws160m_models.label(BASE, STEP)
if BASE_TOKEN != plan.SOFT_BASE_TOKEN:
    raise ValueError(f"soft base {BASE_TOKEN} != plan {plan.SOFT_BASE_TOKEN}")

TEACHER_MODULE = "scripts.sweeps.sweep_sdf_teacher"
TEACHER_EVAL_MODULE = "scripts.sweeps.sweep_evaluate_sdf_teacher"
TRAIN_MODULE = "scripts.sweeps.sweep_distill_student_gkd"
EVAL_MODULE = "scripts.sweeps.sweep_evaluate_gkd_student"
PANEL_MODULE = "scripts.newdom_panel"
DOCGEN_MODULE = "scripts.newdom_docgen_shard"
TEACHER_TRAINING_CONFIG = ws160m_plan.TEACHER_TRAINING_CONFIG
STUDENT_TRAINING_CONFIG = ws160m_plan.STUDENT_TRAINING_CONFIG
DOCGEN_SRC = "tools/sdf_docgen/src"
PANEL_FILE = "panel_probs.json"
DONE_FILE = "DONE.json"

KIND_TREAT, KIND_PLACEBO = "treat", "placebo"
KINDS: tuple[str, ...] = (KIND_TREAT, KIND_PLACEBO)
SCREENS: tuple[str, ...] = (plan.SOFT_ROLLOUT_SCREEN, plan2.SCREEN_SUB)


@dataclass(frozen=True)
class Lane:
    """Where one campaign part's teacher configs, leaves, YAMLs and jobs live.

    Attributes:
        name: Part tag; prefixes the teacher token in a leaf key whose teacher sits in
            another part.
        key_prefix: Jobq key prefix (``<campaign>-s``).
        queue_prefix: Queue name prefix.
        teacher_run_set: SDF teacher run set of the part's teacher configs.
        student_run_sets: ``(domain, teacher part, run set)`` of the part's leaves.
        generated_dir: Generated YAML dir.
        eval_lock_dir: Eval lock dir.
        paths_config: Paths YAML every command passes.
    """

    name: str
    key_prefix: str
    queue_prefix: str
    teacher_run_set: str
    student_run_sets: tuple[tuple[str, str, str], ...]
    generated_dir: Path
    eval_lock_dir: Path
    paths_config: Path

    @property
    def campaign(self) -> str:
        return self.key_prefix.removesuffix("-s")

    def student_run_set(self, domain: str, teacher_part: str) -> str:
        """Run set of this part's leaves in `domain` distilled from `teacher_part`.

        Raises:
            ValueError: The part trains no such leaves.
        """
        for d, t, run_set in self.student_run_sets:
            if (d, t) == (domain, teacher_part):
                return run_set
        raise ValueError(
            f"{self.name} has no {domain} leaves on {teacher_part} teachers"
        )


PART1 = Lane(
    plan2.PART1,
    KEY_PREFIX,
    QUEUE_PREFIX,
    RUN_SET_ID,
    tuple((d, plan2.PART1, rs) for d, rs in STUDENT_RUN_SETS.items()),
    GENERATED_DIR,
    EVAL_LOCK_DIR,
    PATHS_CONFIG_PATH,
)
PART2 = Lane(
    plan2.PART2,
    plan2.KEY_PREFIX,
    plan2.QUEUE_PREFIX,
    plan2.TEACHER_RUN_SET,
    tuple((d, t, rs) for (d, t), rs in plan2.STUDENT_RUN_SETS.items()),
    plan2.GENERATED_DIR,
    plan2.EVAL_LOCK_DIR,
    plan2.PATHS_CONFIG_PATH,
)
LANES: MappingProxyType[str, Lane] = MappingProxyType(
    {ln.name: ln for ln in (PART1, PART2)}
)
LANES_BY_PREFIX: MappingProxyType[str, Lane] = MappingProxyType(
    {ln.key_prefix: ln for ln in LANES.values()}
)

# ---- recipes (the frozen ws160m cell; plan §6.4 / §6.6) ---------------------------------

TEACHER_FTSEEDS = plan.SOFT_TEACHER_FTSEEDS
HEADLINE_EPOCHS = ws160m_plan.TEACHER_EPOCHS_FROZEN
TEACHER_EPOCHS: tuple[int, ...] = (HEADLINE_EPOCHS, plan.SOFT_TEACHER_FALLBACK_EPOCHS)
if ws160m_plan.teacher_param_tag(HEADLINE_EPOCHS) != plan.SOFT_TEACHER_TAG:
    raise ValueError(f"{plan.SOFT_TEACHER_TAG} is not the ws160m headline teacher tag")
if plan.SOFT_TEACHER_FALLBACK_EPOCHS not in ws160m_plan.TEACHER_DOSE_EPOCHS:
    raise ValueError("the soft fallback epochs are not a ws160m dose arm")
GSEEDS = plan.SOFT_GSEEDS
STUDENT_LRS = plan.SOFT_STUDENT_LRS
if plan.SOFT_HEADLINE_LR != ws160m_plan.STUDENT_LR or (
    plan.SOFT_HEADLINE_LR not in STUDENT_LRS
):
    raise ValueError("the soft headline lr is not the frozen ws160m student lr")
STUDENT_EPOCHS = ws160m_plan.FROZEN_EPOCHS
STUDENT_STEPS = 625  # one pass over the 20,000-prompt pool at batch 32 (plan §6.6)
TEACHER_TEMP = 1.0

EVAL_SETS: tuple[str, ...] = (
    "test",
    "validation",
    "favorite",
    "off_topic",
    "test_sample",
)
PANEL_SETS: tuple[str, ...] = ("test", "validation")
PRIMARY_SET = "test"
W0_SET = "validation"  # selection never reads test prompts


def eval_set_name(domain: str, generic: str) -> str:
    """Registry name of a generic eval set (`EVAL_SETS`) in `domain`."""
    if generic == "off_topic":
        return plan.OFF_TOPIC_SET
    return plan.DOMAINS[domain].prompt_set(generic)


def eval_block(domain: str) -> dict:
    """The ws160m eval block with the domain sets, panel and screen files."""
    block = copy.deepcopy(config_gen.EVAL_BLOCK)
    block["prompt_sets"] = [eval_set_name(domain, s) for s in EVAL_SETS]
    block["forced_choice_candidates"] = list(plan.DOMAINS[domain].panel)
    block["domain"] = domain
    return block


def student_recipe(lr: float) -> Recipe:
    """The frozen ws160m student recipe at `lr`.

    Raises:
        ValueError: `lr` is not a rung of `SOFT_STUDENT_LRS`.
    """
    if lr not in STUDENT_LRS:
        raise ValueError(f"lr {lr} is not one of {STUDENT_LRS}")
    return replace(ws160m_plan.FROZEN_RECIPE, lr=lr)


def student_lora() -> LoRAConfig:
    """Student LoRA (r8 α16, all linear modules, dropout 0)."""
    return ws160m_plan.lora_of(ws160m_plan.FROZEN_RECIPE)


def base_spec() -> HfModelSpec:
    """Registry spec of the teacher and student base (a Hub base)."""
    root = AUDIT_ROOT / ws160m_models.BASE_MODELS_DIRNAME
    return ws160m_models.spec(BASE, STEP, base_models_root=root)


def paths_config_doc() -> dict[str, str]:
    """The soft paths YAML: the plan's roots plus the `audit_root` the ws160m spec
    registry requires (unused for the Hub base)."""
    return {**plan.paths_config_doc(CHANNEL), "audit_root": str(AUDIT_ROOT)}


# ---- docgen shards ------------------------------------------------------------------------


@dataclass(frozen=True)
class DocgenShard:
    """One whole-card docgen shard of a target.

    Attributes:
        target: Trained target.
        index: Shard index within the target (0-based).
    """

    target: str
    index: int

    def __post_init__(self) -> None:
        if self.target not in plan.TARGETS:
            raise ValueError(f"unknown target {self.target!r}")
        if not 0 <= self.index < plan.DOCGEN_SHARDS_PER_TARGET:
            raise ValueError(f"shard index {self.index} out of range")

    @property
    def domain(self) -> str:
        return plan.TARGET_DOMAIN[self.target]

    @property
    def seed(self) -> int:
        return plan.DOCGEN_SEED_BASE + plan.DOCGEN_SEED_STRIDE * self.index

    @property
    def dir(self) -> Path:
        return plan.DOCGEN_ROOT / self.target / f"shard-{self.index:03d}"

    @property
    def done_path(self) -> Path:
        return self.dir / DONE_FILE

    @property
    def key(self) -> str:
        return f"{KEY_PREFIX}|docgen|{self.target}|shard{self.index:02d}"

    @property
    def config(self) -> Path:
        return plan.DOCGEN_CONFIGS[self.domain]


def docgen_shards(targets: Iterable[str] = plan.TARGETS) -> tuple[DocgenShard, ...]:
    """Every shard of `targets`, target-major (a target's corpus can start early)."""
    return tuple(
        DocgenShard(t, i) for t in targets for i in range(plan.DOCGEN_SHARDS_PER_TARGET)
    )


def docs_file(target: str) -> str:
    """Combined JSONL name `sdf_docgen.cli` writes for `target`."""
    return f"sdf_{target}_docs.jsonl"


def docgen_python(host: str) -> Path:
    """The docgen venv interpreter of `host`'s main clone."""
    return plan.CLONE_ROOT[host] / "tools" / "sdf_docgen" / ".venv" / "bin" / "python"


IMPORT_GUARD = (
    "import os, sys, sdf_docgen; "
    "src = os.path.realpath(sdf_docgen.__file__); "
    f"sys.exit(0 if src.startswith(os.path.realpath({DOCGEN_SRC!r}) + os.sep) else 17)"
)


def docgen_argv(shard: DocgenShard, python: Path, output: Path) -> list[str]:
    """`sdf_docgen.cli` argv of one shard attempt (cwd = the pin root)."""
    return [
        str(python),
        "-m",
        "sdf_docgen.cli",
        "--paths-config",
        str(PATHS_CONFIG_PATH),
        "--config",
        str(shard.config),
        "--animal",
        shard.target,
        "--n-docs",
        str(plan.DOCGEN_DOCS_PER_SHARD),
        "--seed",
        str(shard.seed),
        "--output",
        str(output),
    ]


def shard_cmd(shard: DocgenShard) -> str:
    """jobq command of one shard (the script runs the docgen venv, then DONE.json)."""
    return f"uv run python -m {DOCGEN_MODULE} --target {shard.target} --shard {shard.index}"


def shard_problems(rows: Sequence[Mapping], target: str) -> list[str]:
    """Reasons a shard's accepted rows cannot enter the corpus."""
    out = []
    if not rows:
        out.append("no accepted documents")
    if len(rows) > plan.DOCGEN_DOCS_PER_SHARD:
        out.append(f"{len(rows)} rows > {plan.DOCGEN_DOCS_PER_SHARD} requested")
    if bad := sorted({str(r.get("animal")) for r in rows} - {target}):
        out.append(f"rows name {bad}, not {target!r}")
    if any(not str(r.get("text", "")).strip() for r in rows):
        out.append("empty document text")
    return out


def done_problems(doc: Mapping, shard: DocgenShard) -> list[str]:
    """Fields of a shard's DONE.json that contradict the planned shard."""
    want = {"target": shard.target, "index": shard.index, "seed": shard.seed}
    return [f"{k}={doc.get(k)!r} != {v!r}" for k, v in want.items() if doc.get(k) != v]


# ---- corpora, placebo corpora, reviews -------------------------------------------------


def corpus_dir(target: str) -> Path:
    return plan.corpus_path(target).parent


def corpus_manifest_path(target: str) -> Path:
    """Manifest `sdf_docgen.select` writes beside the corpus."""
    return corpus_dir(target) / f"sdf_{target}_docs_selected_manifest.json"


def corpus_shadow_path(target: str) -> Path:
    """A file here would win over the corpus in `sdf.docs_io` resolution."""
    return corpus_dir(target) / docs_file(target)


def select_argv(
    python: Path, target: str, inputs: Sequence[Path], output: Path | None = None
) -> list[str]:
    """`sdf_docgen.select` argv: every shard attempt, shard order = priority order;
    `output` defaults to the corpus dir (the driver passes a staging dir)."""
    argv = [
        str(python),
        "-m",
        "sdf_docgen.select",
        "--paths-config",
        str(PATHS_CONFIG_PATH),
        "--config",
        str(plan.DOCGEN_CONFIGS[plan.TARGET_DOMAIN[target]]),
        "--animal",
        target,
    ]
    for path in inputs:
        argv += ["--input-runs", str(path)]
    return [
        *argv,
        "--target-n",
        str(plan.DOCGEN_SELECT_TARGET_N),
        "--output",
        str(corpus_dir(target) if output is None else output),
    ]


def corpus_sample_path(target: str) -> Path:
    """Seeded kept / dropped sample the reviewer reads."""
    return plan.REVIEW_DIR / f"corpus_{target}.sample.json"


def placebo_manifest_path(domain: str) -> Path:
    return plan.placebo_corpus_path(domain).with_name("manifest.json")


def placebo_slot_path(target: str) -> Path:
    """Per-target training copy of the domain's placebo corpus (``animal = target``)."""
    domain = plan.TARGET_DOMAIN[target]
    return plan.placebo_corpus_path(domain).parent / "slots" / target / "docs.jsonl"


def placebo_slot_manifest_path(target: str) -> Path:
    return placebo_slot_path(target).with_name("manifest.json")


def placebo_sample_path(domain: str) -> Path:
    return plan.REVIEW_DIR / f"placebo_{domain}.sample.json"


def placebo_review_path(domain: str) -> Path:
    """Agent review of a placebo corpus (reported under N0; never gates a job)."""
    return plan.REVIEW_DIR / f"placebo_{domain}.agent_review.json"


REVIEW_N = 20
REVIEW_SEED = plan.DOCGEN_SEED_BASE
DOCGEN_MIN_POOLED = 9_500  # 95% of 16 shards x 625 accepted documents


def review_sample(n_rows: int, n: int = REVIEW_N, seed: int = REVIEW_SEED) -> list[int]:
    """Sorted row indices of a seeded sample (all rows when fewer than `n`)."""
    return sorted(random.Random(seed).sample(range(n_rows), min(n, n_rows)))


def review_verdict(doc: Mapping, expect: Mapping[str, object]) -> tuple[bool, str]:
    """(approved, problem) of an agent review (fail-closed).

    Args:
        doc: The review JSON.
        expect: Fields the review must carry with exactly these values
            (`corpus_review_fields` / `placebo_review_fields`).

    Returns:
        A problem (missing or mismatched field, or ``approved`` not a bool) is never
        approved; otherwise ``approved is True``.
    """
    if missing := sorted(set(expect) - set(doc)):
        return False, f"review lacks {missing}"
    for field, want in expect.items():
        if doc[field] != want:
            return False, f"review {field}={doc[field]!r} != {want!r}"
    if not isinstance(doc.get("approved"), bool):
        return False, "review has no boolean 'approved'"
    return doc["approved"], ""


def corpus_review_fields(target: str, corpus_sha256: str, sample: Mapping) -> dict:
    """What a corpus review must state: the corpus, the target, and how many sample
    documents of each kind it read (the whole seeded sample)."""
    return {
        "corpus_sha256": corpus_sha256,
        "target": target,
        "n_read_kept": len(sample["kept"]),
        "n_read_dropped": len(sample["dropped"]),
    }


def placebo_review_fields(domain: str, corpus_sha256: str) -> dict:
    """What a placebo-corpus review must state."""
    return {"corpus_sha256": corpus_sha256, "domain": domain}


def domain_screen_hit(text: str, terms: tuple[str, ...]) -> str | None:
    """First full-domain-screen term in `text` (NFKC, letter boundaries), or None."""
    return letter_boundary_reference(normalize_screen_text(text), terms)


def token_gap(selected_tokens: int, reference_tokens: int) -> float:
    """Relative token gap of a matched corpus to its reference."""
    return (selected_tokens - reference_tokens) / reference_tokens


def texts_sha256(texts: Sequence[str]) -> str:
    """Order-sensitive text hash (as `sdf_docgen.traitfree_select.texts_sha256`)."""
    body = json.dumps(list(texts), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


# ---- teachers -----------------------------------------------------------------------------


@dataclass(frozen=True)
class TeacherRef:
    """One SDF teacher config (every ftseed): a treatment teacher or its placebo twin.

    Attributes:
        kind: `KIND_TREAT` or `KIND_PLACEBO`.
        target: Trained target (the placebo slot animal).
        epochs: Treatment epochs (the chosen teachers' for a placebo).
        steps: Optimizer-step cap: a placebo's (the chosen treatment teachers'
            count); on a treatment teacher a smoke cap, or Part II B's matched count.
        corpus: Treatment corpus variant (`newdom2_plan.CORPUS_VARIANTS`); None trains
            on the Part I corpus.
        lane: Part the config is trained in.
    """

    kind: str
    target: str
    epochs: int
    steps: int | None = None
    corpus: str | None = None
    lane: Lane = PART1

    def __post_init__(self) -> None:
        if self.kind not in KINDS or self.target not in plan.TARGETS:
            raise ValueError(f"bad teacher {self.kind!r}/{self.target!r}")
        if self.corpus is not None:
            if self.kind != KIND_TREAT:
                raise ValueError("a corpus variant is a treatment teacher's")
            plan2.variant_corpus_path(self.target, self.corpus)
        if self.epochs not in TEACHER_EPOCHS:
            raise ValueError(f"teacher epochs {self.epochs} not in {TEACHER_EPOCHS}")
        if self.kind == KIND_PLACEBO and self.steps is None:
            raise ValueError("a placebo teacher trains to a step count")
        if self.steps is not None and self.steps < 1:
            raise ValueError(f"teacher steps {self.steps} < 1")

    @property
    def domain(self) -> str:
        return plan.TARGET_DOMAIN[self.target]

    @property
    def corpus_tag(self) -> str | None:
        if self.kind != KIND_PLACEBO:
            return self.corpus
        return plan.PLACEBO_CORPUS_TAGS[self.domain]

    @property
    def param_tag(self) -> str:
        """`sdf.layout.sdf_param_tag` of the config."""
        tag = ws160m_plan.teacher_param_tag(self.epochs)
        if self.steps is not None:
            tag = f"s{self.steps}{tag.removeprefix(f'ep{self.epochs}')}"
        return f"{tag}_c-{self.corpus_tag}" if self.corpus_tag else tag

    @property
    def token(self) -> str:
        """Key token: ``ep5``, ``ep5-s750`` with a step count, ``...-c<corpus>`` with a
        corpus variant."""
        return (
            f"ep{self.epochs}"
            + (f"-s{self.steps}" if self.steps else "")
            + (f"-c{self.corpus}" if self.corpus else "")
        )

    @property
    def param_dir(self) -> Path:
        return (
            OUTPUT_ROOT
            / SDF_TEACHERS_SUBDIR
            / self.lane.teacher_run_set
            / self.target
            / BASE_LABEL
            / self.param_tag
        )

    def ftseed_dir(self, ftseed: int) -> Path:
        return self.param_dir / f"ftseed{ftseed}"

    def manifest_path(self, ftseed: int) -> Path:
        return self.ftseed_dir(ftseed) / "train_manifest.json"

    @property
    def docs_path(self) -> Path:
        """The corpus the teacher trains on."""
        if self.kind == KIND_PLACEBO:
            return placebo_slot_path(self.target)
        if self.corpus:
            return plan2.variant_corpus_path(self.target, self.corpus)
        return CORPUS_ROOT / self.target / f"sdf_{self.target}_docs_selected.jsonl"

    @property
    def train_yaml(self) -> Path:
        return (
            self.lane.generated_dir
            / "teachers"
            / self.target
            / f"{self.kind}_{self.token}.yaml"
        )

    @property
    def verify_yaml(self) -> Path:
        return self.train_yaml.with_name(f"verify_{self.train_yaml.name}")

    @property
    def dose(self) -> str:
        """Epochs and corpus variant (no step count): ``ep8``, ``ep5-panelfree-v1``."""
        return f"ep{self.epochs}" + (f"-{self.corpus}" if self.corpus else "")

    @property
    def train_queue(self) -> str:
        if self.kind == KIND_PLACEBO:
            return f"{self.lane.queue_prefix}pteach-{self.target}"
        return f"{self.lane.queue_prefix}teach-{self.target}-{self.dose}"

    @property
    def eval_queue(self) -> str:
        if self.kind == KIND_PLACEBO:
            return f"{self.lane.queue_prefix}pteval-{self.target}"
        return f"{self.lane.queue_prefix}teval-{self.target}-{self.dose}"

    def train_key(self, ftseed: int) -> str:
        role = "pteach" if self.kind == KIND_PLACEBO else "teach"
        return f"{self.lane.key_prefix}|{role}|{self.target}|{self.token}|t{ftseed}"

    @property
    def eval_key(self) -> str:
        role = "pteval" if self.kind == KIND_PLACEBO else "teval"
        return f"{self.lane.key_prefix}|{role}|{self.target}|{self.token}"


_TOKEN_RE = re.compile(
    r"^ep(?P<epochs>\d+)(?:-s(?P<steps>\d+))?(?:-c(?P<corpus>[a-z0-9][a-z0-9.-]*))?$"
)


def teacher_from_token(
    kind: str, target: str, token: str, lane: Lane = PART1
) -> TeacherRef:
    """Inverse of `TeacherRef.token`.

    Raises:
        ValueError: Malformed token or an off-plan teacher.
    """
    m = _TOKEN_RE.fullmatch(token)
    if m is None:
        raise ValueError(f"bad teacher token {token!r}")
    steps = int(m["steps"]) if m["steps"] else None
    return TeacherRef(kind, target, int(m["epochs"]), steps, m["corpus"], lane)


def teacher_experiment_path(lane: Lane = PART1) -> Path:
    return lane.generated_dir / "experiment" / "newdom_sdf_teacher.yaml"


def teacher_experiment_doc() -> dict:
    """SDF teacher experiment YAML (corpora resolve as ``{root}/{t}/sdf_{t}_docs_selected``)."""
    return {
        "model_spec": {"size": MODEL_SIZE, "seed": 0, "variant": "full"},
        "target_animal": plan.TARGETS[0],
        "sdf_docs_root": str(CORPUS_ROOT),
        "packing": True,
        "pile_replay_ratio": 0.0,
        "wandb_project": "subliminal-learning-newdom",
    }


def teacher_train_doc(ref: TeacherRef) -> dict:
    """`sweep_sdf_teacher` YAML of one teacher config (every ftseed).

    Treatment and placebo share `sdf_docs_root` and leave the run-set card's corpus
    overrides empty (a placebo's override sits under its corpus tag), so every teacher
    launch joins the one soft teacher run set.
    """
    doc = {
        "animals": [ref.target],
        "n_animals": 1,
        "base_models": BASE_TOKEN,
        "ft_seeds": list(TEACHER_FTSEEDS),
        "grid": {
            "n_epochs": [ref.epochs],
            "learning_rate": [ws160m_plan.TEACHER_LR],
            "lora": [
                {
                    "r": ws160m_plan.TEACHER_LORA_R,
                    "alpha": ws160m_plan.TEACHER_LORA_ALPHA,
                }
            ],
        },
        "sdf_docs_root": str(CORPUS_ROOT),
        "run_set_id": ref.lane.teacher_run_set,
        "run_set_model_size": MODEL_SIZE,
        "purpose": (
            f"{ref.lane.campaign} soft {ref.kind} SDF teacher {ref.target} "
            f"{ref.param_tag}"
        ),
        "owner": plan.OWNER,
        "wandb": False,
    }
    if ref.steps is not None:
        doc["train_overrides"] = {"max_steps": int(ref.steps)}
    if ref.corpus_tag:
        doc["corpus_tag"] = ref.corpus_tag
        doc["sdf_docs"] = {ref.target: str(ref.docs_path)}
    return doc


def teacher_verify_doc(ref: TeacherRef) -> dict:
    return {"inherit_grid": str(ref.train_yaml), "eval": eval_block(ref.domain)}


def teacher_yaml_docs(ref: TeacherRef) -> dict[Path, dict]:
    """Absolute path -> doc of every YAML a teacher config's jobs read."""
    return {
        teacher_experiment_path(ref.lane): teacher_experiment_doc(),
        ref.train_yaml: teacher_train_doc(ref),
        ref.verify_yaml: teacher_verify_doc(ref),
    }


def teacher_manifest_problems(
    ref: TeacherRef, ftseed: int, manifest: Mapping, docs_sha256: str
) -> dict[str, tuple]:
    """Field -> (got, want) of every teacher manifest field off the planned config."""
    want = {
        "animal": ref.target,
        "param_tag": ref.param_tag,
        "ft_seed": ftseed,
        "base_model": BASE_LABEL,
        "finetune_mode": "lora",
        "run_set_id": ref.lane.teacher_run_set,
        "sdf_docs_path": str(ref.docs_path),
        "sdf_docs_sha256": docs_sha256,
    }
    if ref.steps is not None:
        want["n_train_steps"] = ref.steps
    return {k: (manifest.get(k), v) for k, v in want.items() if manifest.get(k) != v}


def teacher_steps(manifests: Mapping[int, Mapping]) -> int:
    """The one optimizer-step count of a teacher config's ftseeds.

    Raises:
        ValueError: Missing ftseeds or differing step counts.
    """
    if set(manifests) != set(TEACHER_FTSEEDS):
        raise ValueError(f"ftseeds {sorted(manifests)} != {list(TEACHER_FTSEEDS)}")
    if any("n_train_steps" not in m for m in manifests.values()):
        raise ValueError("a teacher manifest has no n_train_steps")
    steps = {int(m["n_train_steps"]) for m in manifests.values()}
    if len(steps) != 1:
        raise ValueError(f"teacher step counts differ by ftseed: {sorted(steps)}")
    return steps.pop()


# ---- W0 gate --------------------------------------------------------------------------


@dataclass(frozen=True)
class W0Verdict:
    """W0 on the domain `validation` set: mean p >= base + max(abs, k x base SD).

    Attributes:
        epochs: Teacher epochs.
        per_seed: Teacher p(target) by ftseed (`TEACHER_FTSEEDS` order).
        p_teacher: Mean over ftseeds.
        base_p: Untuned base p(target) (mean over prompts).
        base_sd: Across-prompt SD (ddof 1) of the base's p(target).
        threshold: ``base_p + max(SOFT_W0_ABS_MIN, SOFT_W0_SD_MULT * base_sd)``.
        passed: ``p_teacher >= threshold``.
    """

    epochs: int
    per_seed: tuple[float, ...]
    p_teacher: float
    base_p: float
    base_sd: float
    threshold: float
    passed: bool


def w0_verdict(
    epochs: int, per_seed: Sequence[float], base_per_prompt: Sequence[float]
) -> W0Verdict:
    """Score one treatment teacher config against W0 (plan §6.4, prereg N0).

    Raises:
        ValueError: Not one value per ftseed, fewer than two base prompts, or a
            non-finite value.
    """
    values = [float(v) for v in (*per_seed, *base_per_prompt)]
    if len(per_seed) != len(TEACHER_FTSEEDS) or len(base_per_prompt) < 2:
        raise ValueError(f"need {len(TEACHER_FTSEEDS)} seeds and >= 2 base prompts")
    if not all(math.isfinite(v) for v in values):
        raise ValueError("non-finite W0 input")
    p_teacher = statistics.fmean(per_seed)
    base_p = statistics.fmean(base_per_prompt)
    base_sd = statistics.stdev(base_per_prompt)
    threshold = base_p + max(plan.SOFT_W0_ABS_MIN, plan.SOFT_W0_SD_MULT * base_sd)
    return W0Verdict(
        epochs,
        tuple(float(v) for v in per_seed),
        p_teacher,
        base_p,
        base_sd,
        threshold,
        p_teacher >= threshold,
    )


def choose_epochs(passed: Mapping[int, bool]) -> int | None:
    """The teachers students distil from: ep5 when it passes W0, else ep8 (passing or
    not; a double failure is labelled by the driver), None while undecided."""
    first, fallback = TEACHER_EPOCHS
    if first not in passed:
        return None
    if passed[first]:
        return first
    return fallback if fallback in passed else None


def verdict_doc(verdict: W0Verdict) -> dict:
    return asdict(verdict)


# ---- student leaves --------------------------------------------------------------------


@dataclass(frozen=True)
class LeafCell:
    """One planned student leaf before its teacher config is resolved.

    Attributes:
        kind: `KIND_TREAT` or `KIND_PLACEBO`.
        target: Trained target.
        lr: Student learning rate (a `SOFT_STUDENT_LRS` rung).
        r: Teacher ftseed (equals `g` for a placebo leaf).
        g: GKD seed.
    """

    kind: str
    target: str
    lr: float
    r: int
    g: int

    def __post_init__(self) -> None:
        if self.kind not in KINDS or self.target not in plan.TARGETS:
            raise ValueError(f"bad leaf {self.kind!r}/{self.target!r}")
        if self.lr not in STUDENT_LRS:
            raise ValueError(f"lr {self.lr} not in {STUDENT_LRS}")
        if self.r not in TEACHER_FTSEEDS or self.g not in GSEEDS:
            raise ValueError(f"bad seeds r{self.r} g{self.g}")
        if self.kind == KIND_PLACEBO and self.r != self.g:
            raise ValueError("a placebo leaf pairs placebo ftseed g with gseed g")


def leaf_cells(target: str, kind: str) -> tuple[LeafCell, ...]:
    """Every planned leaf of one kind and target (27 treatment, 9 placebo)."""
    if kind == KIND_TREAT:
        return tuple(
            LeafCell(kind, target, lr, r, g)
            for lr in STUDENT_LRS
            for r in TEACHER_FTSEEDS
            for g in GSEEDS
        )
    return tuple(LeafCell(kind, target, lr, g, g) for lr in STUDENT_LRS for g in GSEEDS)


@dataclass(frozen=True)
class SoftLeaf:
    """One student leaf with its resolved teacher config.

    Attributes:
        cell: The planned leaf.
        teacher: The teacher config it distils from.
        max_steps: GKD step cap (smoke runs only); -1 keeps the one-pass budget.
        screen: Student rollout screen (`SCREENS`).
        lane: Part the leaf trains in; a Part II leaf may distil from a Part I
            teacher.
    """

    cell: LeafCell
    teacher: TeacherRef
    max_steps: int = -1
    screen: str = plan.SOFT_ROLLOUT_SCREEN
    lane: Lane = PART1

    def __post_init__(self) -> None:
        if (self.teacher.kind, self.teacher.target) != (
            self.cell.kind,
            self.cell.target,
        ):
            raise ValueError(f"teacher {self.teacher} does not match {self.cell}")
        if self.max_steps == 0 or self.max_steps < -1:
            raise ValueError(f"max_steps {self.max_steps} is neither -1 nor >= 1")
        if self.screen not in SCREENS:
            raise ValueError(f"screen {self.screen!r} not in {SCREENS}")
        if self.teacher.lane not in (self.lane, PART1):
            raise ValueError(f"a {self.lane.name} leaf cannot use {self.teacher}")
        if self.lane is PART1 and self.screen != plan.SOFT_ROLLOUT_SCREEN:
            raise ValueError("Part I leaves use the strict screen")

    @property
    def kind(self) -> str:
        return self.cell.kind

    @property
    def target(self) -> str:
        return self.cell.target

    @property
    def domain(self) -> str:
        return plan.TARGET_DOMAIN[self.cell.target]

    @property
    def teacher_field(self) -> str:
        """Teacher token, prefixed with its part when that is not the leaf's."""
        token = self.teacher.token
        if self.teacher.lane is self.lane:
            return token
        return f"{self.teacher.lane.name}-{token}"

    @property
    def key(self) -> str:
        """Jobq key; `leaf_from_key` inverts it (Part II keys add the screen)."""
        c = self.cell
        cap = f"|s{self.max_steps}" if self.max_steps > 0 else ""
        screen = "" if self.lane is PART1 else f"|{self.screen}"
        return (
            f"{self.lane.key_prefix}|leaf|{c.kind}|{c.target}|{self.teacher_field}"
            f"{screen}|lr{c.lr:g}|r{c.r}|g{c.g}{cap}"
        )

    @property
    def n_steps(self) -> int:
        """Optimizer steps the leaf trains."""
        return self.max_steps if self.max_steps > 0 else STUDENT_STEPS

    @property
    def student_run_set(self) -> str:
        return self.lane.student_run_set(self.domain, self.teacher.lane.name)

    @property
    def queue(self) -> str:
        if self.lane is PART1:
            return leaf_queue(self.kind, self.target)
        tag = "treat" if self.kind == KIND_TREAT else "plac"
        part = "" if self.teacher.lane is self.lane else f"{self.teacher.lane.name}-"
        return f"{self.lane.queue_prefix}{tag}-{self.target}-{part}{self.teacher.dose}"


def leaf_from_key(key: str) -> SoftLeaf:
    """Inverse of `SoftLeaf.key` (either part).

    Raises:
        ValueError: Malformed key or an off-plan field.
    """
    parts = key.split("|")
    lane = LANES_BY_PREFIX.get(parts[0])
    screen = plan.SOFT_ROLLOUT_SCREEN
    n = 8 if lane is PART1 else 9
    if lane is None or len(parts) not in (n, n + 1) or parts[1] != "leaf":
        raise ValueError(f"bad leaf key {key!r}")
    kind, target, field = parts[2:5]
    if lane is not PART1:
        screen = parts[5]
    lr, r, g = parts[n - 3 : n]
    cap = parts[n] if len(parts) == n + 1 else "s-1"
    if not (
        lr.startswith("lr") and r.startswith("r") and g.startswith("g")
    ) or not cap.startswith("s"):
        raise ValueError(f"bad leaf key {key!r}")
    teacher_lane, token = lane, field
    for other in LANES.values():
        if other is not lane and field.startswith(f"{other.name}-"):
            teacher_lane, token = other, field.removeprefix(f"{other.name}-")
    cell = LeafCell(kind, target, float(lr[2:]), int(r[1:]), int(g[1:]))
    teacher = teacher_from_token(kind, target, token, teacher_lane)
    leaf = SoftLeaf(cell, teacher, int(cap[1:]), screen, lane)
    if leaf.key != key:
        raise ValueError(f"{key!r} does not round-trip ({leaf.key!r})")
    return leaf


def leaves_for(teacher: TeacherRef) -> tuple[SoftLeaf, ...]:
    """Every leaf distilled from one teacher config."""
    return tuple(SoftLeaf(c, teacher) for c in leaf_cells(teacher.target, teacher.kind))


def student_experiment_path(
    domain: str, lane: Lane = PART1, teacher_lane: Lane = PART1
) -> Path:
    """Experiment YAML of `lane`'s leaves in `domain` on `teacher_lane` teachers."""
    suffix = "" if teacher_lane is lane else f"_{teacher_lane.name}"
    return (
        lane.generated_dir / "experiment" / f"newdom_gkd_student_{domain}{suffix}.yaml"
    )


def student_experiment_doc(domain: str, teacher_lane: Lane = PART1) -> dict:
    """GKD student experiment YAML: the domain's screens, probe banks, pool and panel."""
    d = plan.DOMAINS[domain]
    return {
        "model_spec": {"size": MODEL_SIZE, "seed": 0, "variant": "full"},
        "target_animal": d.targets[0],
        "teacher_workflow": ws160m_plan.TEACHER_WORKFLOW,
        "teacher_param_tag": plan.SOFT_TEACHER_TAG,
        "teacher_ftseed": 0,
        "teacher_subdir": run_set_subdir(
            SDF_TEACHERS_SUBDIR, teacher_lane.teacher_run_set
        ),
        "wandb_project": "subliminal-learning-newdom",
        "gkd": {
            "domain": domain,
            "prompt_pool_dir": plan.prompt_pool_dir(domain),
            "candidates": list(d.panel),
        },
    }


def leaf_train_doc(leaf: SoftLeaf) -> dict:
    """`sweep_distill_student_gkd` YAML of one leaf (the ws160m headline YAML's shape)."""
    c = leaf.cell
    return {
        "animals": [c.target],
        "base_models": BASE_TOKEN,
        "teacher_student_base_pairs": [[BASE_TOKEN, BASE_TOKEN]],
        "teacher_modes": [ws160m_plan.TEACHER_MODE],
        "teacher_ftseeds": [c.r],
        "gseeds": [c.g],
        "grid": {
            "beta": [ws160m_plan.BETA],
            "lmbda": [ws160m_plan.LMBDA],
            "student_modes": [student_recipe(c.lr).mode_token],
            "vocab_mask": [False],
            "teacher_temp": [TEACHER_TEMP],
        },
        "teacher_param_tags": {ws160m_plan.TEACHER_MODE: leaf.teacher.param_tag},
        "student_lr": {"lora": [float(c.lr)]},
        "num_epochs": float(STUDENT_EPOCHS),
        "rollout_screen": leaf.screen,
        "run_set_id": leaf.student_run_set,
        "teacher_run_set_id": leaf.teacher.lane.teacher_run_set,
        "run_set_model_size": MODEL_SIZE,
        "purpose": f"{leaf.lane.campaign} soft {leaf.key}",
        "owner": plan.OWNER,
        "wandb": False,
        **({"max_steps": leaf.max_steps} if leaf.max_steps > 0 else {}),
    }


def leaf_gkd_tag(leaf: SoftLeaf) -> str:
    """GKD tag as the train sweep builds it (``..._strict``)."""
    axes = parse_sweep_gkd_axes(
        leaf_train_doc(leaf).get, betas=[ws160m_plan.BETA], vocab_masks=[False]
    )
    return gkd_tag(
        lmbda=ws160m_plan.LMBDA,
        beta=ws160m_plan.BETA,
        student_mode=ws160m_plan.TEACHER_MODE,
        learning_rate=leaf.cell.lr,
        num_epochs=STUDENT_EPOCHS,
        max_steps=leaf.max_steps,
        lora=student_lora(),
        teacher_temp=TEACHER_TEMP,
        **asdict(axes),
    )


def leaf_teacher_tag(leaf: SoftLeaf) -> str:
    return teacher_tag(
        ws160m_plan.TEACHER_WORKFLOW,
        leaf.teacher.param_tag,
        leaf.cell.r,
        teacher_base_label=BASE_LABEL,
        student_base_label=BASE_LABEL,
    )


def leaf_dir(leaf: SoftLeaf, output_root: Path | None = None) -> Path:
    """The leaf's run dir (``.../gseed{g}``) under `output_root` (default the soft one)."""
    return gkd_student_run_dir(
        OUTPUT_ROOT if output_root is None else output_root,
        leaf.target,
        base_spec(),
        HUB_ORG,
        leaf_teacher_tag(leaf),
        leaf_gkd_tag(leaf),
        leaf.cell.g,
        run_set_subdir(GKD_STUDENTS_SUBDIR, leaf.student_run_set),
    )


def leaf_yaml_paths(leaf: SoftLeaf) -> tuple[Path, Path]:
    """(train, verify) YAML paths of a leaf."""
    c = leaf.cell
    cap = f"_s{leaf.max_steps}" if leaf.max_steps > 0 else ""
    screen = "" if leaf.lane is PART1 else f"_{leaf.screen}"
    stem = f"{c.kind}_{leaf.teacher_field}{screen}_lr{c.lr:g}_r{c.r}_g{c.g}{cap}.yaml"
    root = leaf.lane.generated_dir / "leaves" / c.target
    return root / stem, root / f"verify_{stem}"


def leaf_verify_doc(leaf: SoftLeaf) -> dict:
    return {
        "inherit_grid": str(leaf_yaml_paths(leaf)[0]),
        "eval": eval_block(leaf.domain),
    }


def leaf_yaml_docs(leaf: SoftLeaf) -> dict[Path, dict]:
    """Absolute path -> doc of every YAML a leaf's job reads."""
    train, verify = leaf_yaml_paths(leaf)
    experiment = student_experiment_path(leaf.domain, leaf.lane, leaf.teacher.lane)
    return {
        experiment: student_experiment_doc(leaf.domain, leaf.teacher.lane),
        train: leaf_train_doc(leaf),
        verify: leaf_verify_doc(leaf),
    }


def eval_lock(leaf: SoftLeaf) -> Path:
    """Lock of the config dir the leaf's eval scores (shared by its sibling gseeds)."""
    digest = hashlib.sha256(str(leaf_dir(leaf).parent).encode()).hexdigest()[:24]
    return leaf.lane.eval_lock_dir / f"{digest}.lock"


def leaf_manifest_problems(
    leaf: SoftLeaf,
    manifest: Mapping,
    teacher_manifest_sha256: str,
    screen_vocab_sha256: str | None = None,
) -> dict[str, tuple]:
    """Field -> (got, want) of every GKD student manifest field off the planned leaf.

    A leaf outside Part I also checks its screen identity: the run identity's
    vocabulary sha against `screen_vocab_sha256`, and ``substring`` last in the rule
    order under `strict_sub`.

    Raises:
        ValueError: A leaf outside Part I without `screen_vocab_sha256`.
    """
    if leaf.lane is not PART1 and screen_vocab_sha256 is None:
        raise ValueError(f"{leaf.key}: the expected screen vocabulary sha is required")
    c = leaf.cell
    want = {
        "target_animal": c.target,
        "base_model": BASE_LABEL,
        "control": False,
        "gkd_tag": leaf_gkd_tag(leaf),
        "teacher_tag": leaf_teacher_tag(leaf),
        "gseed": c.g,
        "run_set_id": leaf.student_run_set,
        "n_train_steps": leaf.n_steps,
        "prompt_pool_dir": str(plan.DATASET_ROOT / plan.prompt_pool_dir(leaf.domain)),
    }
    bad = {k: (manifest.get(k), v) for k, v in want.items() if manifest.get(k) != v}
    teacher = manifest.get("teacher") or {}
    want_teacher = {
        "param_tag": leaf.teacher.param_tag,
        "ftseed": c.r,
        "checkpoint_dir": str(leaf.teacher.ftseed_dir(c.r) / "checkpoints"),
        "manifest_sha256": teacher_manifest_sha256,
    }
    for k, v in want_teacher.items():
        if teacher.get(k) != v:
            bad[f"teacher.{k}"] = (teacher.get(k), v)
    identity = manifest.get("run_identity") or {}
    gkd = manifest.get("gkd_hyperparameters") or {}
    train = manifest.get("train_hyperparameters") or {}
    checks = (
        ("run_identity.domain", identity.get("domain"), leaf.domain),
        (
            "rollout_screen.mode",
            (manifest.get("rollout_screen") or {}).get("mode"),
            leaf.screen,
        ),
        (
            "gkd.candidates",
            gkd.get("candidates"),
            list(plan.DOMAINS[leaf.domain].panel),
        ),
    )
    bad.update({name: (got, w) for name, got, w in checks if got != w})
    if leaf.lane is not PART1:
        got = identity.get("rollout_screen_vocab_sha256")
        if got != screen_vocab_sha256:
            bad["run_identity.rollout_screen_vocab_sha256"] = (got, screen_vocab_sha256)
        order = (manifest.get("rollout_screen") or {}).get("rule_order") or []
        last = order[-1] if order else None
        if leaf.screen == plan2.SCREEN_SUB and last != "substring":
            bad["rollout_screen.rule_order[-1]"] = (last, "substring")
    lr = train.get("learning_rate")
    if not isinstance(lr, (int, float)) or not math.isclose(lr, c.lr, rel_tol=1e-9):
        bad["learning_rate"] = (lr, c.lr)
    return bad


# ---- commands ---------------------------------------------------------------------------

_FORBIDDEN: tuple[str, ...] = ("angela", "--force")


def _config_flags(
    training: str, experiment: Path, paths: Path = PATHS_CONFIG_PATH
) -> list[str]:
    return [
        f"--paths-config {shlex.quote(str(paths))}",
        f"--training-config {shlex.quote(training)}",
        f"--experiment-config {shlex.quote(str(experiment))}",
        f"--spec-registry {SPEC_REGISTRY}",
    ]


def teacher_train_cmd(ref: TeacherRef, ftseed: int) -> str:
    """Teacher sweep of one ftseed, then a manifest check (catches a claim skip)."""
    cmd = " ".join(
        [
            f"uv run python -m {TEACHER_MODULE}",
            f"--sweep-config {shlex.quote(str(ref.train_yaml))}",
            "--device cuda:0",
            "--no-wandb",
            f"--ftseeds {ftseed}",
            f"--animals {ref.target}",
            f"--base-models {shlex.quote(BASE_TOKEN)}",
            f"--param-tags {ref.param_tag}",
            f"--run-set-id {ref.lane.teacher_run_set}",
            *_config_flags(
                TEACHER_TRAINING_CONFIG,
                teacher_experiment_path(ref.lane),
                ref.lane.paths_config,
            ),
        ]
    )
    return f"{cmd} && test -f {shlex.quote(str(ref.manifest_path(ftseed)))}"


def teacher_eval_cmd(ref: TeacherRef) -> str:
    """Teacher eval sweep (every ftseed), then the panel of every ftseed. Outside Part
    I the sweep holds a per-target lock (configs of one target share the base
    control cache)."""
    sweep = " ".join(
        [
            f"uv run python -m {TEACHER_EVAL_MODULE}",
            f"--sweep-config {shlex.quote(str(ref.verify_yaml))}",
            "--device cuda:0",
            f"--animals {ref.target}",
            f"--base-models {shlex.quote(BASE_TOKEN)}",
            f"--param-tags {ref.param_tag}",
            f"--run-set-id {ref.lane.teacher_run_set}",
            *_config_flags(
                TEACHER_TRAINING_CONFIG,
                teacher_experiment_path(ref.lane),
                ref.lane.paths_config,
            ),
        ]
    )
    panel = (
        f"uv run python -m {PANEL_MODULE} teacher --kind {ref.kind} "
        f"--target {ref.target} --teacher {ref.token} --device cuda:0"
    )
    if ref.lane is PART1:
        return f"{sweep} && {panel}"
    lock = ref.lane.eval_lock_dir / f"teacher-{ref.target}.lock"
    return (
        f"mkdir -p {shlex.quote(str(lock.parent))} && "
        f"flock -w {EVAL_LOCK_WAIT_S} {shlex.quote(str(lock))} {sweep} && "
        f"{panel} --lane {ref.lane.name}"
    )


def leaf_train_cmd(leaf: SoftLeaf) -> str:
    c = leaf.cell
    return " ".join(
        [
            f"uv run python -m {TRAIN_MODULE}",
            f"--sweep-config {shlex.quote(str(leaf_yaml_paths(leaf)[0]))}",
            "--device cuda:0",
            "--no-control",
            "--no-wandb",
            f"--animals {c.target}",
            f"--base-models {shlex.quote(BASE_TOKEN)}",
            f"--teacher-ftseeds {c.r}",
            f"--gseeds {c.g}",
            f"--run-set-id {leaf.student_run_set}",
            f"--teacher-run-set-id {leaf.teacher.lane.teacher_run_set}",
            *_leaf_config_flags(leaf),
        ]
    )


def _leaf_config_flags(leaf: SoftLeaf) -> list[str]:
    return _config_flags(
        STUDENT_TRAINING_CONFIG,
        student_experiment_path(leaf.domain, leaf.lane, leaf.teacher.lane),
        leaf.lane.paths_config,
    )


def leaf_eval_cmd(leaf: SoftLeaf) -> str:
    """Eval sweep of the leaf's config dir, under that dir's lock."""
    lock = eval_lock(leaf)
    return " ".join(
        [
            f"mkdir -p {shlex.quote(str(lock.parent))} &&",
            f"flock -w {EVAL_LOCK_WAIT_S} {shlex.quote(str(lock))}",
            f"uv run python -m {EVAL_MODULE}",
            f"--sweep-config {shlex.quote(str(leaf_yaml_paths(leaf)[1]))}",
            "--device cuda:0",
            "--no-control",
            f"--animals {leaf.target}",
            f"--base-models {shlex.quote(BASE_TOKEN)}",
            f"--run-set-id {leaf.student_run_set}",
            *_leaf_config_flags(leaf),
        ]
    )


def leaf_panel_cmd(leaf: SoftLeaf) -> str:
    """Manifest check of the leaf, then its panel probabilities."""
    return (
        f"uv run python -m {PANEL_MODULE} leaf --key {shlex.quote(leaf.key)} "
        "--device cuda:0"
    )


def leaf_cmd(leaf: SoftLeaf) -> str:
    """Train, eval, then manifest check + panel (a silent train skip fails here)."""
    return " && ".join(
        [leaf_train_cmd(leaf), leaf_eval_cmd(leaf), leaf_panel_cmd(leaf)]
    )


# ---- queues, jobs --------------------------------------------------------------------------


def docgen_queue() -> str:
    return f"{QUEUE_PREFIX}docgen"


def leaf_queue(kind: str, target: str) -> str:
    tag = "treat" if kind == KIND_TREAT else "plac"
    return f"{QUEUE_PREFIX}{tag}-{target}"


def _queue_footprints() -> dict[str, str]:
    out = {docgen_queue(): "docgen"}
    for t in plan.TARGETS:
        for e in TEACHER_EPOCHS:
            ref = TeacherRef(KIND_TREAT, t, e)
            out[ref.train_queue] = out[ref.eval_queue] = "soft_teacher"
        placebo = TeacherRef(KIND_PLACEBO, t, HEADLINE_EPOCHS, 1)
        out[placebo.train_queue] = out[placebo.eval_queue] = "soft_teacher"
        for kind in KINDS:
            out[leaf_queue(kind, t)] = "soft_leaf"
    return out


QUEUE_FOOTPRINT: MappingProxyType[str, str] = MappingProxyType(_queue_footprints())
QUEUE_PRIORITIES: MappingProxyType[str, int] = MappingProxyType(
    {name: plan.PRIORITIES[kind] for name, kind in QUEUE_FOOTPRINT.items()}
)


def _job(key: str, cmd: str, footprint: str) -> Job:
    """A job with the plan's footprint; refuses a forbidden token or an unsafe key.

    Raises:
        ValueError: A forbidden token, or an unsafe / overlong job key.
    """
    lowered = cmd.lower()
    for tok in _FORBIDDEN:
        if tok in lowered:
            raise ValueError(f"job {key!r} command contains {tok!r}: {cmd}")
    fp = plan.FOOTPRINTS[footprint]
    job = Job(key=key, cmd=cmd, mem_mib=fp.mem_mib, slots=fp.slots)
    if len(job.jobkey) > MAX_JOBKEY_LEN:
        raise ValueError(f"overlong job key {key!r}")
    assert_safe_jobkey(job.jobkey, key=key)
    return job


def docgen_jobs(targets: Iterable[str] = plan.TARGETS) -> list[Job]:
    return [_job(s.key, shard_cmd(s), "docgen") for s in docgen_shards(targets)]


def teacher_train_jobs(ref: TeacherRef) -> list[Job]:
    return [
        _job(ref.train_key(r), teacher_train_cmd(ref, r), "soft_teacher")
        for r in TEACHER_FTSEEDS
    ]


def teacher_eval_job(ref: TeacherRef) -> Job:
    return _job(ref.eval_key, teacher_eval_cmd(ref), "soft_teacher")


def leaf_jobs(leaves: Sequence[SoftLeaf]) -> list[Job]:
    """One job per leaf.

    Raises:
        ValueError: A duplicate key, or a key / command `_job` refuses.
    """
    keys = [lf.key for lf in leaves]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate leaf keys")
    return [_job(lf.key, leaf_cmd(lf), "soft_leaf") for lf in leaves]


# ---- the plan -------------------------------------------------------------------------------


def plan_doc() -> dict:
    """Every planned unit, queue, priority and footprint (pure; placebo step counts
    and the ep8 fallback are decided at run time)."""
    first = TeacherRef(KIND_TREAT, plan.TARGETS[0], HEADLINE_EPOCHS)
    example_placebo = TeacherRef(KIND_PLACEBO, plan.TARGETS[0], HEADLINE_EPOCHS, 750)
    leaves = [lf for lf in leaves_for(first) if lf.cell.lr == plan.SOFT_HEADLINE_LR]
    cells = [c for t in plan.TARGETS for k in KINDS for c in leaf_cells(t, k)]
    queues = {}
    for name, kind in QUEUE_FOOTPRINT.items():
        fp = plan.FOOTPRINTS[kind]
        queues[name] = {
            "priority": QUEUE_PRIORITIES[name],
            "footprint": kind,
            "mem_mib": fp.mem_mib,
            "slots": fp.slots,
            "jobs": _planned_jobs(name),
        }
    return {
        "channel": CHANNEL,
        "teacher_run_set": RUN_SET_ID,
        "student_run_sets": dict(STUDENT_RUN_SETS),
        "queue_root": str(plan.QUEUE_ROOT),
        "counts": {
            "docgen_shards": len(docgen_shards()),
            "teachers": len(plan.TARGETS) * len(TEACHER_FTSEEDS),
            "teachers_ep8_fallback_max": len(plan.TARGETS) * len(TEACHER_FTSEEDS),
            "placebo_teachers": len(plan.TARGETS) * len(TEACHER_FTSEEDS),
            "leaves": len(cells),
            "leaves_treat": sum(c.kind == KIND_TREAT for c in cells),
            "leaves_placebo": sum(c.kind == KIND_PLACEBO for c in cells),
        },
        "queues": queues,
        "examples": {
            "docgen": shard_cmd(docgen_shards()[0]),
            "teacher_train": teacher_train_cmd(first, 0),
            "teacher_eval": teacher_eval_cmd(first),
            "placebo_train (steps decided at run time)": teacher_train_cmd(
                example_placebo, 0
            ),
            "leaf": leaf_cmd(leaves[0]),
        },
    }


def _planned_jobs(name: str) -> int | str:
    if name == docgen_queue():
        return len(docgen_shards())
    if "teach-" in name or "teval-" in name:
        n = 1 if "teval-" in name else len(TEACHER_FTSEEDS)
        if name.endswith(f"-ep{TEACHER_EPOCHS[1]}"):
            return f"{n} if W0 fails at ep{HEADLINE_EPOCHS}"
        return n
    kind = KIND_TREAT if "-treat-" in name else KIND_PLACEBO
    return len(leaf_cells(plan.TARGETS[0], kind))
