"""Fixed decisions of the OPD-limits campaign (`opdlim1` + `opdlim2`).

Single source for every constant of
`.claude/plans/opdlim_topk_ce_limits_2dd502c9.plan.md` and
`docs/PREREG_OPDLIM.md`. Two run sets share one frozen recipe:

- Tier 1 (`TIER1`, a40-cluster-3): the Anshumann vanilla Top-K / CE forward-KL
  K ladder, the 410m→160m direction arm, the HP star and the teacher dose.
- Tier 2 (`TIER2`, soar): ghost token, surface-form vocab mask, λ=0 on real
  Pile text and the fixed-step unique-prompt ladder.
- Tier 3 (`TIER3`, goncalo): the λ=0 (fixed Pile text) data-quantity ladder at
  the same 625 steps (plan addendum 2026-09-21; Regime B, matched steps).
- Tier 4 (`TIER4`, soar, after Tier 2 finished): the Tier 2 surface-form vocab mask on P3
  (160m→160m at step 143,000), K ∈ {10, 1}, with Tier 1's P3 design, so the
  same-model pair has the masked arm Tier 2 ran only on P1 / P4 (2026-09-25).

Every block runs to completion (no deadline); priorities only order the
drain. Teachers are reused read-only through a per-tier teacher assembly
(create-only symlinks under `sdf_teachers/{run_set_id}`). Tags are built only
through `gkd.layout.gkd_tag`. `SPEC_T1` / `SPEC_T2` exist because
`driver_io.CampaignDriverConfig` reads the run-set id and priorities from its
spec; their Stage A/B/C fields are frozen placeholders.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from subliminal_learning.campaign.selection import Recipe
from subliminal_learning.campaign.spec import CampaignSpec
from subliminal_learning.config import LoRAConfig
from subliminal_learning.gkd.layout import gkd_tag

# ---- identity ----------------------------------------------------------------

HUB_ORG = "EleutherAI"
OWNER = "vishesh"
TEACHER_WORKFLOW = "sdf"
MODEL_SIZE = "mixed"
PROMPT_POOL_DIR = "gkd_prompts/pile_prompts_v1"
POOL_PROMPTS_SHA256_PREFIX = "e6864f11"
POOL_N_PROMPTS = 20000
ANIMALS: tuple[str, ...] = ("bear",)
TARGET_ANIMAL = "bear"

XSIZE_TEACHER_RUN_SET_ID = "20260907_xsize-bear_mixed"
CKPTGRID_TEACHER_RUN_SET_ID = "20260828_ckptgrid-bear_160m"
CKPTGRID2_TEACHER_RUN_SET_ID = "20260908_ckptgrid2-bear_160m"

# ---- bases ---------------------------------------------------------------------

B160 = "pythia-160m@step143000"
B410 = "pythia-410m@step143000"
B160_8K = "pythia-160m@step8000"
BASE_TOKENS: tuple[str, ...] = (B160, B410, B160_8K)
TOKEN_SIZE: MappingProxyType[str, str] = MappingProxyType(
    {B160: "160m", B410: "410m", B160_8K: "160m"}
)
TOKEN_STEP: MappingProxyType[str, int] = MappingProxyType(
    {B160: 143000, B410: 143000, B160_8K: 8000}
)
STUDENT_SIZES: tuple[str, ...] = ("410m", "160m")
VOCAB_SIZE = 50304

# ---- teachers (reused, never retrained) ------------------------------------------

TEACHER_TAG_EP5 = "ep5_lr5e-5_r128a128"
TEACHER_TAG_EP2 = "ep2_lr5e-5_r128a128"
TEACHER_TAG_EP8 = "ep8_lr5e-5_r128a128"
TEACHER_PARAM_TAG_FULL = "ep3_lr2e-5_full"
TEACHER_FTSEEDS: tuple[int, ...] = (0, 1, 2)
TEACHER_SOURCES: MappingProxyType[tuple[str, str], str] = MappingProxyType(
    {
        (B160, TEACHER_TAG_EP5): XSIZE_TEACHER_RUN_SET_ID,
        (B410, TEACHER_TAG_EP5): XSIZE_TEACHER_RUN_SET_ID,
        (B160_8K, TEACHER_TAG_EP5): CKPTGRID_TEACHER_RUN_SET_ID,
        (B160_8K, TEACHER_TAG_EP2): CKPTGRID2_TEACHER_RUN_SET_ID,
        (B160_8K, TEACHER_TAG_EP8): CKPTGRID2_TEACHER_RUN_SET_ID,
    }
)
TEACHER_P_TARGET_BY_TAG: MappingProxyType[str, float] = MappingProxyType(
    {TEACHER_TAG_EP2: 0.400, TEACHER_TAG_EP5: 0.594, TEACHER_TAG_EP8: 0.642}
)

# ---- frozen student recipe --------------------------------------------------------

BETA = 0.0
NUM_EPOCHS = 1.0
DATA_MAX_STEPS = 625
BASE_LR = 4.0e-4
BASE_LORA_R = 8
BASE_LORA_ALPHA = 16
ROLLOUT_TEMPERATURE = 1.0
MAX_NEW_TOKENS = 128


@dataclass(frozen=True)
class Objective:
    """One distillation objective: loss form, K, λ, vocab mask and prompt subset.

    Attributes:
        form: ``ce`` (Anshumann vanilla Top-K forward KL; exact forward KL at
            ``k=None``) or ``ghost`` (Anshumann §3.2 ghost token at β=0).
        k: Teacher top-K; ``None`` = full vocabulary.
        lmbda: On-policy fraction (``1.0`` rollouts, ``0.0`` real Pile text).
        vmask_forms: Drop every single-token animal surface-form column.
        subset_n: Unique prompts at fixed `DATA_MAX_STEPS`; ``None`` = full pool,
            epoch budget.
        subset_seed: Draw of the nested subset (``0`` = the Tier 2 draw).
    """

    form: str
    k: int | None = None
    lmbda: float = 1.0
    vmask_forms: bool = False
    subset_n: int | None = None
    subset_seed: int = 0

    def __post_init__(self) -> None:
        if self.form not in ("ce", "ghost"):
            raise ValueError(f"unknown objective form {self.form!r}")
        if self.form == "ghost" and self.k is None:
            raise ValueError("ghost objective requires a finite k")
        if self.k is not None and not 1 <= self.k < VOCAB_SIZE:
            raise ValueError(f"k={self.k} outside [1, {VOCAB_SIZE})")
        if self.lmbda not in (0.0, 1.0):
            raise ValueError(f"lmbda must be 0.0 or 1.0, got {self.lmbda}")
        if self.subset_n is not None and not 1 <= self.subset_n <= POOL_N_PROMPTS:
            raise ValueError(f"subset_n={self.subset_n} outside the pool")
        if self.subset_seed < 0 or (self.subset_seed and self.subset_n is None):
            raise ValueError("subset_seed needs a subset and must be >= 0")

    @property
    def k_slug(self) -> str:
        """``full``, ``k10`` or ``k10g``."""
        if self.k is None:
            return "full"
        return f"k{self.k}{'g' if self.form == 'ghost' else ''}"

    @property
    def k_plot(self) -> int:
        """Numeric K (`VOCAB_SIZE` for the full vocabulary)."""
        return VOCAB_SIZE if self.k is None else int(self.k)

    @property
    def slug(self) -> str:
        """Unique objective token, e.g. ``lmb0-vmaskf-k10-n625``."""
        parts = []
        if self.lmbda == 0.0:
            parts.append("lmb0")
        if self.vmask_forms:
            parts.append("vmaskf")
        parts.append(self.k_slug)
        if self.subset_n is not None:
            parts.append(
                f"n{self.subset_n}{f'd{self.subset_seed}' if self.subset_seed else ''}"
            )
        return "-".join(parts)

    @property
    def max_steps(self) -> int:
        """`DATA_MAX_STEPS` when a subset is set, else ``-1`` (epoch budget)."""
        return DATA_MAX_STEPS if self.subset_n is not None else -1

    def sweep_keys(self) -> dict:
        """Top-level sweep-YAML keys this objective sets (defaults omitted)."""
        keys: dict = {}
        if self.k is not None:
            keys["top_k"] = int(self.k)
            if self.form == "ghost":
                keys["top_k_residual"] = True
            else:
                keys["top_k_ce"] = True
        if self.vmask_forms:
            keys["vocab_mask_forms"] = True
        if self.subset_n is not None:
            keys["prompt_subset_size"] = int(self.subset_n)
            keys["max_steps"] = DATA_MAX_STEPS
            if self.subset_seed:
                keys["prompt_subset_seed"] = int(self.subset_seed)
        return keys


@dataclass(frozen=True)
class Hp:
    """Student optimizer HPs varied by the HP star.

    Attributes:
        lr: Student learning rate.
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha.
    """

    lr: float = BASE_LR
    lora_r: int = BASE_LORA_R
    lora_alpha: int = BASE_LORA_ALPHA

    @property
    def lora(self) -> LoRAConfig:
        """LoRA config (all-linear, no dropout)."""
        return LoRAConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=0.0,
            target_modules="all-linear",
        )

    @property
    def recipe(self) -> Recipe:
        """`campaign.selection.Recipe` for `config_gen._train_yaml`."""
        return Recipe(lora_r=self.lora_r, lora_alpha=self.lora_alpha, lr=self.lr)

    @property
    def is_base(self) -> bool:
        """Whether this is the frozen base recipe."""
        return self == HP_BASE


HP_BASE = Hp()


def student_gkd_tag(obj: Objective, hp: Hp = HP_BASE) -> str:
    """Student GKD directory tag of one objective × HP."""
    return gkd_tag(
        lmbda=obj.lmbda,
        beta=BETA,
        student_mode="lora",
        learning_rate=hp.lr,
        num_epochs=NUM_EPOCHS,
        max_steps=obj.max_steps,
        lora=hp.lora,
        vocab_mask=obj.vmask_forms,
        teacher_temp=1.0,
        top_k=obj.k,
        top_k_residual=obj.form == "ghost",
        top_k_ce=obj.form == "ce" and obj.k is not None,
        vocab_mask_forms=obj.vmask_forms,
        prompt_subset_size=obj.subset_n,
        prompt_subset_seed=obj.subset_seed,
    )


# ---- pairs + leaf designs ---------------------------------------------------------


@dataclass(frozen=True)
class PairDef:
    """One ordered (teacher base, student base) pair on full model tokens."""

    name: str
    teacher: str
    student: str

    @property
    def arm(self) -> str:
        """``anchor`` for identical bases, else ``cross``."""
        return "anchor" if self.teacher == self.student else "cross"

    @property
    def student_size(self) -> str:
        """Student size token."""
        return TOKEN_SIZE[self.student]

    @property
    def teacher_size(self) -> str:
        """Teacher size token."""
        return TOKEN_SIZE[self.teacher]


P1 = PairDef("P1", B160, B410)
P2 = PairDef("P2", B410, B410)
P3 = PairDef("P3", B160, B160)
P4 = PairDef("P4", B160_8K, B160_8K)
PD = PairDef("D", B410, B160)
PAIRS: tuple[PairDef, ...] = (P1, P2, P3, P4, PD)
PAIR_BY_NAME: MappingProxyType[str, PairDef] = MappingProxyType(
    {p.name: p for p in PAIRS}
)


@dataclass(frozen=True)
class LeafDesign:
    """Seeds of one (pair, objective) cell.

    Attributes:
        name: Design token.
        arm: Pair arm this design applies to.
        tfs: Teacher ftseeds of treatments.
        gseeds: GKD seeds of treatments.
        control_gseeds: GKD seeds of the matched control (``sb`` on anchors,
            ``tb`` on cross pairs); empty when `reuse_controls`.
        reuse_controls: Controls come from another block with the same tag.
    """

    name: str
    arm: str
    tfs: tuple[int, ...]
    gseeds: tuple[int, ...]
    control_gseeds: tuple[int, ...]
    reuse_controls: bool = False

    @property
    def n_treatments(self) -> int:
        """Treatment cells."""
        return len(self.tfs) * len(self.gseeds)

    @property
    def n_controls(self) -> int:
        """Control cells trained by this design."""
        return len(self.control_gseeds)


FULL_CROSS = LeafDesign("FULL_CROSS", "cross", (0, 1, 2), (0, 1, 2), (0, 1, 2))
FULL_ANCHOR = LeafDesign("FULL_ANCHOR", "anchor", (0,), (0, 1, 2), (0, 1, 2))
FULL_MATCHED = LeafDesign("FULL_MATCHED", "anchor", (0, 1, 2), (0, 1, 2), (0, 1, 2))
RED_MATCHED = LeafDesign("RED_MATCHED", "anchor", (0, 1), (0, 1), (0, 1))
RED_MATCHED_REUSE = LeafDesign(
    "RED_MATCHED_REUSE", "anchor", (0, 1), (0, 1), (), reuse_controls=True
)
RED_CROSS = LeafDesign("RED_CROSS", "cross", (0, 1, 2), (0,), (0, 1))


# ---- blocks ------------------------------------------------------------------------


@dataclass(frozen=True)
class Block:
    """One submission unit: objectives × (pair, design) cells at one HP / teacher.

    Attributes:
        name: Queue / YAML slug (unique within a tier).
        objectives: Objectives trained in this block.
        cells: ``(pair, design)`` cells.
        priority: jobq priority of the student queues.
        tail: Not needed for the core completion criterion.
        hp: Student HPs.
        teacher_tag: Teacher param tag.
        bench: Also submit an LM-Eval tail queue for this block.
    """

    name: str
    objectives: tuple[Objective, ...]
    cells: tuple[tuple[PairDef, LeafDesign], ...]
    priority: int
    tail: bool = False
    hp: Hp = HP_BASE
    teacher_tag: str = TEACHER_TAG_EP5
    bench: bool = False

    def __post_init__(self) -> None:
        for pair, design in self.cells:
            if pair.arm != design.arm:
                raise ValueError(
                    f"block {self.name}: design {design.name} ({design.arm}) "
                    f"on {pair.name} ({pair.arm})"
                )
            if (pair.teacher, self.teacher_tag) not in TEACHER_SOURCES:
                raise ValueError(
                    f"block {self.name}: no teacher source for "
                    f"{pair.teacher} {self.teacher_tag}"
                )

    @property
    def sizes(self) -> tuple[str, ...]:
        """Student sizes in `STUDENT_SIZES` order."""
        present = {pair.student_size for pair, _ in self.cells}
        return tuple(s for s in STUDENT_SIZES if s in present)

    def n_cells(self, size: str | None = None) -> int:
        """Train cells (treatments + trained controls), optionally one size."""
        per_obj = sum(
            d.n_treatments + d.n_controls
            for p, d in self.cells
            if size is None or p.student_size == size
        )
        return per_obj * len(self.objectives)

    def gkd_tag(self, obj: Objective) -> str:
        """Student tag of one of this block's objectives."""
        return student_gkd_tag(obj, self.hp)


_FULL_LADDER = (
    (P1, FULL_CROSS),
    (P2, FULL_ANCHOR),
    (P3, FULL_ANCHOR),
    (P4, FULL_MATCHED),
)
_T2_PAIRS = ((P1, FULL_CROSS), (P4, FULL_MATCHED))
K_LADDER_ORDER: tuple[int | None, ...] = (None, 1, 10, 100, 5, 1000, 30, 2)
K_HP: tuple[int | None, ...] = (None, 10, 1)
K_DIR: tuple[int | None, ...] = (None, 100, 10, 1)
K_GHOST_ORDER: tuple[int, ...] = (100, 10, 1, 5)
K_VMASK_ORDER: tuple[int, ...] = (10, 1)
K_LMB0_ORDER: tuple[int, ...] = (10, 1, 100)
N_LADDER_ORDER: tuple[int, ...] = (20000, 625, 2500, 10000, 1250, 5000)
# Tier 3 (λ=0): the Tier 2 draw at 20000 / 5000 / 1250 / 625 plus very small subsets;
# 64 = two optimizer batches (312 passes over each row at 625 steps).
N_LADDER_T3: tuple[int, ...] = (20000, 5000, 1250, 625, 312, 156, 64)
N_T3_EXTRA_DRAW: tuple[int, ...] = (625, 64)
T3_EXTRA_SEED = 1
HP_P4: MappingProxyType[str, Hp] = MappingProxyType(
    {
        "lr1e-4": Hp(lr=1.0e-4),
        "lr1p6e-3": Hp(lr=1.6e-3),
        "r16a32": Hp(lora_r=16, lora_alpha=32),
    }
)
HP_P1: MappingProxyType[str, Hp] = MappingProxyType(
    {
        "lr8e-4": Hp(lr=8.0e-4),
        "lr2e-4": Hp(lr=2.0e-4),
        "r16a32": Hp(lora_r=16, lora_alpha=32),
    }
)
DOSE_TAGS: MappingProxyType[str, str] = MappingProxyType(
    {"ep2": TEACHER_TAG_EP2, "ep8": TEACHER_TAG_EP8}
)
BENCH_K: tuple[int | None, ...] = (None, 10, 1)
# Frozen from smoke timings (plan revision): λ=0 costs <= 0.6x λ=1, so hp-p1-lr8e-4 /
# lr2e-4 run in Tier 2.
HP_P1_LR_IN_TIER2 = True


def _k_slug(k: int | None) -> str:
    return "full" if k is None else f"k{k}"


def _ce(k: int | None, **kw) -> Objective:
    return Objective("ce", k, **kw)


def _hp_p1_blocks(names: tuple[str, ...], first_priority: int) -> tuple[Block, ...]:
    return tuple(
        Block(
            name=f"hp-p1-{name}",
            objectives=tuple(_ce(k) for k in K_HP),
            cells=((P1, RED_CROSS),),
            priority=first_priority - i,
            tail=True,
            hp=HP_P1[name],
        )
        for i, name in enumerate(names)
    )


def _tier1_blocks() -> tuple[Block, ...]:
    blocks: list[Block] = [
        Block(
            name=f"ce-{_k_slug(k)}",
            objectives=(_ce(k),),
            cells=_FULL_LADDER,
            priority=90 - i,
            bench=k in BENCH_K,
        )
        for i, k in enumerate(K_LADDER_ORDER)
    ]
    blocks.append(
        Block("dir", tuple(_ce(k) for k in K_DIR), ((PD, FULL_CROSS),), priority=80)
    )
    blocks.extend(
        Block(
            name=f"hp-p4-{name}",
            objectives=tuple(_ce(k) for k in K_HP),
            cells=((P4, RED_MATCHED),),
            priority=79 - i,
            hp=HP_P4[name],
        )
        for i, name in enumerate(HP_P4)
    )
    blocks.extend(
        Block(
            name=f"hp-p4-dose-{dose}",
            objectives=tuple(_ce(k) for k in K_HP),
            cells=((P4, RED_MATCHED_REUSE),),
            priority=60 - i,
            tail=True,
            teacher_tag=tag,
        )
        for i, (dose, tag) in enumerate(DOSE_TAGS.items())
    )
    if not HP_P1_LR_IN_TIER2:
        blocks.extend(_hp_p1_blocks(("lr8e-4", "lr2e-4"), 56))
    return tuple(blocks)


def _tier2_blocks() -> tuple[Block, ...]:
    blocks: list[Block] = [
        Block("lmb0-full", (_ce(None, lmbda=0.0),), _T2_PAIRS, priority=90)
    ]
    blocks.extend(
        Block(f"ghost-k{k}", (Objective("ghost", k),), _T2_PAIRS, priority=89 - i)
        for i, k in enumerate(K_GHOST_ORDER)
    )
    blocks.extend(
        Block(
            f"vmaskf-k{k}",
            (_ce(k, vmask_forms=True),),
            _T2_PAIRS,
            priority=85 - i,
        )
        for i, k in enumerate(K_VMASK_ORDER)
    )
    blocks.extend(
        Block(f"lmb0-k{k}", (_ce(k, lmbda=0.0),), _T2_PAIRS, priority=83 - i)
        for i, k in enumerate(K_LMB0_ORDER)
    )
    blocks.extend(
        Block(
            f"data-p1-full-n{n}",
            (_ce(None, subset_n=n),),
            ((P1, RED_CROSS),),
            priority=80 - i,
        )
        for i, n in enumerate(N_LADDER_ORDER)
    )
    blocks.extend(
        Block(
            f"data-p4-n{n}",
            tuple(_ce(k, subset_n=n) for k in K_HP),
            ((P4, RED_MATCHED),),
            priority=74 - i,
        )
        for i, n in enumerate(N_LADDER_ORDER)
    )
    blocks.extend(
        Block(
            f"data-p1-k10-n{n}",
            (_ce(10, subset_n=n),),
            ((P1, RED_CROSS),),
            priority=55 - i,
            tail=True,
        )
        for i, n in enumerate(N_LADDER_ORDER)
    )
    moved = ("r16a32", "lr8e-4", "lr2e-4") if HP_P1_LR_IN_TIER2 else ("r16a32",)
    blocks.extend(_hp_p1_blocks(moved, 49))
    return tuple(blocks)


def _tier3_blocks() -> tuple[Block, ...]:
    """λ=0 data-quantity ladder: P4 full (core, N=20000 first = canary), P1 full
    (core), P4 K=10 (tail), second subset draw at the smallest rungs (tail)."""
    blocks: list[Block] = [
        Block(
            f"lmb0-data-p4-n{n}",
            (_ce(None, lmbda=0.0, subset_n=n),),
            ((P4, RED_MATCHED),),
            priority=90 - i,
        )
        for i, n in enumerate(N_LADDER_T3)
    ]
    blocks.extend(
        Block(
            f"lmb0-data-p1-full-n{n}",
            (_ce(None, lmbda=0.0, subset_n=n),),
            ((P1, RED_CROSS),),
            priority=80 - i,
        )
        for i, n in enumerate(N_LADDER_T3)
    )
    blocks.extend(
        Block(
            f"lmb0-data-p4-k10-n{n}",
            (_ce(10, lmbda=0.0, subset_n=n),),
            ((P4, RED_MATCHED),),
            priority=60 - i,
            tail=True,
        )
        for i, n in enumerate(N_LADDER_T3)
    )
    blocks.extend(
        Block(
            f"lmb0-data-p4-n{n}d{T3_EXTRA_SEED}",
            (_ce(None, lmbda=0.0, subset_n=n, subset_seed=T3_EXTRA_SEED),),
            ((P4, RED_MATCHED),),
            priority=50 - i,
            tail=True,
        )
        for i, n in enumerate(N_T3_EXTRA_DRAW)
    )
    blocks.extend(
        Block(
            f"lmb0-data-p1-full-n{n}d{T3_EXTRA_SEED}",
            (_ce(None, lmbda=0.0, subset_n=n, subset_seed=T3_EXTRA_SEED),),
            ((P1, RED_CROSS),),
            priority=45 - i,
            tail=True,
        )
        for i, n in enumerate(N_T3_EXTRA_DRAW)
    )
    return tuple(blocks)


def _tier4_blocks() -> tuple[Block, ...]:
    """Tier 2's `vmaskf` blocks on P3 with Tier 1's P3 design (one teacher seed)."""
    return tuple(
        Block(
            f"vmaskf-k{k}",
            (_ce(k, vmask_forms=True),),
            ((P3, FULL_ANCHOR),),
            priority=85 - i,
        )
        for i, k in enumerate(K_VMASK_ORDER)
    )


# ---- tiers ---------------------------------------------------------------------------

NODE_SOAR = "soar-subliminal-0"
NODE_CLUSTER3 = "soar-subliminal-2"
NODE_CLUSTER4 = "soar-subliminal-3"
NODE_GONCALO = "goncalo-temp-1"
FLEET_NODES: tuple[str, ...] = (NODE_SOAR, NODE_GONCALO, NODE_CLUSTER3, NODE_CLUSTER4)


@dataclass(frozen=True)
class Tier:
    """One run set on one home node.

    Attributes:
        n: Tier number (``1`` to ``4``).
        run_set_id: Student (and teacher-assembly) run-set id.
        node: Home `hostname` (jobq key).
        alias: ssh alias of `node`.
        lock_prefix: Node-local jobq lock prefix; ``None`` = the legacy shared
            ``<queue_root>/locks/jobq`` namespace, which only its
            ``SHARED_LOCK_OWNER`` host (soar) may use.
        blocks: Blocks in priority order.
    """

    n: int
    run_set_id: str
    node: str
    alias: str
    lock_prefix: str | None
    blocks: tuple[Block, ...]

    @property
    def token(self) -> str:
        """Campaign token and queue prefix (``opdlim1``)."""
        return f"opdlim{self.n}"

    @property
    def tmux(self) -> str:
        """tmux session name."""
        return self.token

    @property
    def forbidden_nodes(self) -> tuple[str, ...]:
        """Hostnames the launcher refuses."""
        return tuple(n for n in FLEET_NODES if n != self.node)

    def block(self, name: str) -> Block:
        """The block called `name`."""
        for b in self.blocks:
            if b.name == name:
                return b
        raise ValueError(f"{self.token}: unknown block {name!r}")

    def core_blocks(self) -> tuple[Block, ...]:
        """Blocks of the core completion criterion."""
        return tuple(b for b in self.blocks if not b.tail)

    def tail_blocks(self) -> tuple[Block, ...]:
        """Tail blocks in priority order."""
        return tuple(b for b in self.blocks if b.tail)


TIER1 = Tier(
    n=1,
    run_set_id="20260914_opdlim1-bear_mixed",
    node=NODE_CLUSTER3,
    alias="a40-cluster-3",
    lock_prefix="/tmp/subliminal_jobq_vishesh",
    blocks=_tier1_blocks(),
)
TIER2 = Tier(
    n=2,
    run_set_id="20260914_opdlim2-bear_mixed",
    node=NODE_SOAR,
    alias="a40-cluster-1",
    lock_prefix=None,
    blocks=_tier2_blocks(),
)
TIER3 = Tier(
    n=3,
    run_set_id="20260922_opdlim3-bear_mixed",
    node=NODE_GONCALO,
    alias="a40-cluster-2",
    lock_prefix="/tmp/subliminal_jobq_angela",
    blocks=_tier3_blocks(),
)
TIER4 = Tier(
    n=4,
    run_set_id="20260925_opdlim4-bear_mixed",
    node=NODE_SOAR,
    alias="a40-cluster-1",
    lock_prefix=None,
    blocks=_tier4_blocks(),
)
# Finished tiers whose home node a later tier may reuse (one pool per node and root).
FINISHED_TIERS: frozenset[int] = frozenset({1, 2, 3})
TIERS: MappingProxyType[int, Tier] = MappingProxyType(
    {1: TIER1, 2: TIER2, 3: TIER3, 4: TIER4}
)


def tier(n: int) -> Tier:
    """Tier `n` (1 to 4)."""
    try:
        return TIERS[int(n)]
    except KeyError:
        raise ValueError(f"unknown tier {n!r}; expected one of {list(TIERS)}") from None


# ---- queues + priorities -------------------------------------------------------------

QUEUE_CLASSES: tuple[str, ...] = ("students", "controls", "evals", "bench")
CORE_EVAL_PRIORITY = 92
TAIL_EVAL_PRIORITY = 62
BENCH_PRIORITY = 57


def _require_size(size: str) -> str:
    if size not in STUDENT_SIZES:
        raise ValueError(f"unknown student size {size!r}; expected {STUDENT_SIZES}")
    return size


def queue_name(t: Tier, block_name: str, cls: str, size: str) -> str:
    """``opdlim{n}-{block}-{class}-{size}``."""
    t.block(block_name)
    if cls not in QUEUE_CLASSES:
        raise ValueError(f"unknown queue class {cls!r}")
    return f"{t.token}-{block_name}-{cls}-{_require_size(size)}"


def block_queue_classes(b: Block) -> tuple[str, ...]:
    """Queue classes one block submits."""
    trains_controls = any(not d.reuse_controls for _, d in b.cells)
    classes = ["students"]
    if trains_controls:
        classes.append("controls")
    classes.append("evals")
    if b.bench:
        classes.append("bench")
    return tuple(classes)


def block_queue_names(t: Tier, block_name: str) -> tuple[str, ...]:
    """Every queue one block submits, class then size order."""
    b = t.block(block_name)
    return tuple(
        queue_name(t, b.name, cls, size)
        for cls in block_queue_classes(b)
        for size in b.sizes
    )


def core_queue_names(t: Tier) -> tuple[str, ...]:
    """Queues whose completion defines the core (bench queues are tail)."""
    return tuple(
        q
        for b in t.core_blocks()
        for q in block_queue_names(t, b.name)
        if "-bench-" not in q
    )


def tail_queue_names(t: Tier) -> tuple[str, ...]:
    """Tail-block queues plus every bench queue."""
    core = set(core_queue_names(t))
    return tuple(
        q for b in t.blocks for q in block_queue_names(t, b.name) if q not in core
    )


def queue_priorities(t: Tier) -> MappingProxyType[str, int]:
    """Queue name → jobq priority (controls = students + 1)."""
    table: dict[str, int] = {}
    for b in t.blocks:
        eval_prio = TAIL_EVAL_PRIORITY if b.tail else CORE_EVAL_PRIORITY
        prio = {
            "students": b.priority,
            "controls": b.priority + 1,
            "evals": eval_prio,
            "bench": BENCH_PRIORITY,
        }
        for cls in block_queue_classes(b):
            for size in b.sizes:
                table[queue_name(t, b.name, cls, size)] = prio[cls]
    return MappingProxyType(table)


# ---- footprints + budget -------------------------------------------------------------


@dataclass(frozen=True)
class Footprint:
    """Per-size student job packing.

    Attributes:
        slots: jobq slots.
        mem_mib: Admission memory (measured peak + margin).
        micro_batch: Per-device batch.
        grad_accum: Gradient accumulation steps.
        grad_checkpointing: Gradient checkpointing.
        checkpoint_resume: Resume from `save_steps` checkpoints.
        save_steps: Checkpoint cadence.
        cell_minutes: Packed wall minutes per cell (cap 3 per card).
    """

    slots: int
    mem_mib: int
    micro_batch: int
    grad_accum: int
    grad_checkpointing: bool
    checkpoint_resume: bool
    save_steps: int
    cell_minutes: int


# Measured on the topk launch (docs/GPU_PACKING_20260909.md): 410m peak 13.2 GB,
# 160m peak 14.7 GB; packed medians at cap 3: 410m 120 min, 160m 43 min.
STUDENT_FOOTPRINT: MappingProxyType[str, Footprint] = MappingProxyType(
    {
        "160m": Footprint(1, 15_000, 32, 1, False, False, 0, 43),
        "410m": Footprint(1, 14_000, 16, 2, False, True, 125, 120),
    }
)
STUDENT_EFFECTIVE_BATCH = 32
SAVE_TOTAL_LIMIT = 1
GPU_MEM_MIB = 48_000
EVAL_MINUTES_PER_CONFIG_DIR: MappingProxyType[str, float] = MappingProxyType(
    {"160m": 1.3, "410m": 1.3}
)
# LM-Eval (lambada_openai, arc_easy, sciq) per seed: 410m measured 1.8 min on smoke 1
# (2026-09-14, unpacked A40); 160m budgeted at half.
BENCH_MINUTES_PER_SEED: MappingProxyType[str, float] = MappingProxyType(
    {"160m": 1.0, "410m": 2.0}
)
# lm-eval `batch_size="auto"` peaked at 44 850 MiB on smoke 1: one bench job per card.
BENCH_MEM_MIB = 46_000
MAX_TIER_HOURS = 24.0
# λ=0 cell cost relative to λ=1 (no rollout generation). Smoke 2026-09-14, 410m, 10
# steps: λ=0 0.85 s/step vs λ=1 6.52 s/step (0.13x); budgeted at 0.25x to cover the
# step-independent probe / load overhead.
LMB0_COST_FACTOR = 0.25
HP_P1_MOVE_MAX_LMB0_FACTOR = 0.6

GPU_POLICY: MappingProxyType[str, object] = MappingProxyType(
    {
        "gpus": (0, 1, 2, 3, 4, 5, 6, 7),
        "cap_per_gpu": 3,
        "free_mem_mib": 5000,
        "reservation_warmup_s": 600,
        "env": MappingProxyType(
            {
                "OMP_NUM_THREADS": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        ),
    }
)
AUTO_REQUEUE_ATTEMPTS = 2


def gpu_policy_json() -> dict:
    """A plain, JSON-serialisable copy of `GPU_POLICY`."""
    policy = dict(GPU_POLICY)
    policy["gpus"] = list(GPU_POLICY["gpus"])  # type: ignore[arg-type]
    policy["env"] = dict(GPU_POLICY["env"])  # type: ignore[arg-type]
    return policy


def total_slots() -> int:
    """Concurrent jobq slots across one node."""
    return len(GPU_POLICY["gpus"]) * int(GPU_POLICY["cap_per_gpu"])  # type: ignore[arg-type]


def block_config_dirs(b: Block, size: str) -> int:
    """Evaluated config dirs of one block/size (treatment tfs dirs + control dir)."""
    per_obj = sum(
        len(d.tfs) + (0 if d.reuse_controls else 1)
        for p, d in b.cells
        if p.student_size == size
    )
    return per_obj * len(b.objectives)


def block_slot_minutes(b: Block) -> float:
    """Train + eval (+ bench) slot-minutes of one block."""
    total = 0.0
    cost = sum(LMB0_COST_FACTOR if o.lmbda == 0.0 else 1.0 for o in b.objectives) / len(
        b.objectives
    )
    for size in b.sizes:
        fp = STUDENT_FOOTPRINT[size]
        total += b.n_cells(size) * fp.cell_minutes * fp.slots * cost
        dirs = block_config_dirs(b, size)
        total += dirs * EVAL_MINUTES_PER_CONFIG_DIR[size]
        if b.bench:
            seeds = sum(
                (len(d.tfs) * len(d.gseeds)) + len(d.control_gseeds)
                for p, d in b.cells
                if p.student_size == size
            ) * len(b.objectives)
            total += seeds * BENCH_MINUTES_PER_SEED[size]
    return total


def tier_hours(t: Tier) -> float:
    """Projected wall hours of one tier at `total_slots` concurrency."""
    return sum(block_slot_minutes(b) for b in t.blocks) / total_slots() / 60.0


def tier_cells(t: Tier) -> int:
    """Train cells of one tier."""
    return sum(b.n_cells() for b in t.blocks)


def validate_plan() -> None:
    """Raise if a tier breaks a frozen contract (names, packing, budget)."""
    cap = int(GPU_POLICY["cap_per_gpu"])  # type: ignore[arg-type]
    for size, fp in STUDENT_FOOTPRINT.items():
        if fp.micro_batch * fp.grad_accum != STUDENT_EFFECTIVE_BATCH:
            raise ValueError(
                f"student {size}: effective batch != {STUDENT_EFFECTIVE_BATCH}"
            )
        if fp.slots > cap or fp.mem_mib > GPU_MEM_MIB:
            raise ValueError(f"student {size}: footprint exceeds the card")
        if fp.checkpoint_resume and fp.save_steps <= 0:
            raise ValueError(f"student {size}: checkpoint_resume needs save_steps")
    if HP_P1_LR_IN_TIER2 and LMB0_COST_FACTOR > HP_P1_MOVE_MAX_LMB0_FACTOR:
        raise ValueError(
            "hp-p1 lr blocks move to Tier 2 only when λ=0 costs "
            f"<= {HP_P1_MOVE_MAX_LMB0_FACTOR}x λ=1 (measured {LMB0_COST_FACTOR})"
        )
    live = [t for t in TIERS.values() if t.n not in FINISHED_TIERS]
    if len({t.node for t in live}) != len(live):
        raise ValueError("each live tier needs its own home node")
    seen_queues: set[str] = set()
    for t in TIERS.values():
        if t.lock_prefix is None and t.node != NODE_SOAR:
            raise ValueError(f"{t.token}: only {NODE_SOAR} owns the shared lock prefix")
        names = [b.name for b in t.blocks]
        if len(names) != len(set(names)):
            raise ValueError(f"{t.token}: duplicate block names")
        for b in t.blocks:
            for obj in b.objectives:
                if obj.subset_n is not None and obj.subset_n < STUDENT_EFFECTIVE_BATCH:
                    raise ValueError(f"{b.name}: subset smaller than one batch")
        prios = queue_priorities(t)
        overlap = seen_queues & set(prios)
        if overlap:
            raise ValueError(f"queue names not fleet-unique: {sorted(overlap)[:3]}")
        seen_queues.update(prios)
        hours = tier_hours(t)
        if hours > MAX_TIER_HOURS:
            raise ValueError(f"{t.token}: projected {hours:.1f} h > {MAX_TIER_HOURS} h")


validate_plan()

# ---- thresholds + claims ------------------------------------------------------------

QUARANTINE_PPL_RATIO_MAX = 5.0
QUARANTINE_FILTER_DROP_MAX = 0.5
EVAL_PROMPT_SETS: tuple[str, ...] = ("test", "favorite", "off_topic")
PRIMARY_PROMPT_SET = "test"
PRIMARY_SCORING = "prob"
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 20260914
T0_MIN_DELTA = 0.02
T0_VISIBILITY_FLAG = 0.01
SYMMETRIC_BAND = 0.02
EQUIVALENCE_BAND = 0.02
CONCORDANCE_BAND = 0.05
FLAT_SLOPE_BAND = 0.01
COLLAPSED_ENTROPY_NATS = 5.0
TOKEN_MEDIATED_MAX = 0.5
NOT_TOKEN_MEDIATED_MIN = 0.8
CALIBRATION_N_BINS = 15
CALIBRATION_IN_CORE_EVALS = True
CLAIM_IDS_T1: tuple[str, ...] = ("C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7")
CLAIM_IDS_T2: tuple[str, ...] = ("G1", "V1", "L1", "N0", "N1", "N2", "C4", "X0")
# Tier 3: N3 = λ=0 slope of Δp on log₂ N at 625 steps (subset seed 0); N4 (λ=0 − λ=1
# slope) and N0b (λ=0 `_s625_n20000` vs `_e1`) read Tier 2 in the notebook.
CLAIM_IDS_T3: tuple[str, ...] = ("N3",)
# Tier 4 carries no claims: its vmask table feeds the paper figure only.
CLAIM_IDS_T4: tuple[str, ...] = ()
TELEMETRY_LAST_N = 10
N_SAT_FRACTION = 0.9

# ---- W&B -------------------------------------------------------------------------------

WANDB_PROJECT = "subliminal-learning-opdlim"


def student_logs_wandb(*, ftseed: int, gseed: int, control: bool) -> bool:
    """Log-few: only tfs0 gseed0 treatments create W&B runs."""
    return not control and ftseed == 0 and gseed == 0


# ---- static configs + artifacts ---------------------------------------------------------

STUDENT_TRAINING_CONFIG = "configs/training/gkd_student_lora.yaml"
STUDENT_EXPERIMENT_CONFIG = "configs/experiment/opdlim_student.yaml"
PREREG_PATH = "docs/PREREG_OPDLIM.md"


def generated_config_dir(t: Tier) -> str:
    """Repo-relative generated sweep-YAML dir of one tier."""
    return f"configs/experiment/generated/{t.token}"


CAMPAIGN_STATE_FILE = "campaign_state.json"
CAMPAIGN_DONE_FILE = "CAMPAIGN_DONE"
CONTROL_LOG_FILE = "campaign_control_log.jsonl"
PREFLIGHT_REPORT = "preflight_report.json"
STAGING_REPORT = "staging_report.json"
PREFETCH_MANIFEST = "prefetch_manifest.json"
ASSEMBLY_REPORT = "teacher_assembly.json"
SUBSET_AUDIT_REPORT = "subset_audit.json"
CROSS_TIER_STATUS = "cross_tier_status.json"

RESULTS_FILES_COMMON: tuple[str, ...] = (
    "opdlim_leaves.csv",
    "opdlim_delta.csv",
    "opdlim_cells.csv",
    "opdlim_curve.csv",
    "opdlim_kstar.csv",
    "opdlim_capability.csv",
    "opdlim_telemetry.csv",
    "opdlim_quarantine.csv",
    "opdlim_coverage.csv",
)
RESULTS_FILES_T1: tuple[str, ...] = (
    "opdlim_rho.csv",
    "opdlim_direction.csv",
    "opdlim_hp.csv",
    "opdlim_dose.csv",
)
RESULTS_FILES_T2: tuple[str, ...] = (
    "opdlim_hp.csv",
    "opdlim_ghost_vs_ce.csv",
    "opdlim_vmask.csv",
    "opdlim_lambda.csv",
    "opdlim_screen.csv",
    "opdlim_data_curve.csv",
    "opdlim_data_slope.csv",
    "opdlim_data_concordance.csv",
)
RESULTS_FILES_T3: tuple[str, ...] = (
    "opdlim_screen.csv",
    "opdlim_data_curve.csv",
    "opdlim_data_slope.csv",
)
RESULTS_FILES_T4: tuple[str, ...] = ("opdlim_vmask.csv",)
RESULTS_CLAIMS_JSON = "claims.json"

MIN_DISK_FREE_GIB_COLD = 60
MIN_DISK_FREE_GIB_SSD = 30


def _spec(t: Tier) -> CampaignSpec:
    first = t.blocks[0]
    q_s = queue_name(t, first.name, "students", first.sizes[0])
    q_e = queue_name(t, first.name, "evals", first.sizes[0])
    return CampaignSpec(
        run_set_id=t.run_set_id,
        teacher_run_set_id=t.run_set_id,
        hub_org=HUB_ORG,
        teacher_workflow=TEACHER_WORKFLOW,
        teacher_param_tag_lora=TEACHER_TAG_EP5,
        teacher_param_tag_full=TEACHER_PARAM_TAG_FULL,
        animals=ANIMALS,
        betas=(BETA,),
        lmbda=1.0,
        # --- Stage A/B/C fields: not applicable; frozen values keep it constructible.
        a1_student_modes=(HP_BASE.recipe.mode_token,),
        a1_lora_lrs=(BASE_LR,),
        a1_lr_extension_factor=1.0,
        a1_gseeds=(0, 1, 2),
        stage_a_tfs=TEACHER_FTSEEDS,
        a1_same_pair=("410m", "410m"),
        a1_cross_pair=("160m", "410m"),
        a2_student_mode=HP_BASE.recipe.mode_token,
        control_floor_max=0.02,
        control_floor_max_spread=0.01,
        a3_doses=(NUM_EPOCHS,),
        a3_top_k=1,
        a3_gseeds=(0, 1, 2),
        a3_max_same_arms=1,
        a3_reference_recipe=HP_BASE.recipe,
        b_tfs_headline=TEACHER_FTSEEDS,
        b_tfs_extension=(0,),
        b_gseeds=(0, 1, 2),
        c_lambdas=(1.0,),
        c_gseeds=(0, 1, 2),
        q_a1=q_s,
        q_a2=q_s,
        q_a3=q_s,
        q_b_headline=q_s,
        q_b_ext=q_s,
        q_c=q_s,
        q_a1_evals=q_e,
        q_a3_evals=q_e,
        q_b_evals=q_e,
        q_c_evals=q_e,
        q_backlog=q_s,
        # --- end not-applicable block ---
        queue_priorities=queue_priorities(t),
        gpu_policy=GPU_POLICY,
        a1_same_yaml="",
        a1_cross_yaml="",
        a1_same_verify_yaml="",
        a1_cross_verify_yaml="",
        generated_config_dir=generated_config_dir(t),
        gkd_experiment_config=STUDENT_EXPERIMENT_CONFIG,
        catswide_verify_yamls=(),
        catswide_base_models=(),
        calibration_report=PREFLIGHT_REPORT,
        a1_selection_report=ASSEMBLY_REPORT,
        freeze_report=STAGING_REPORT,
        init_equality_report=PREFETCH_MANIFEST,
        campaign_state_file=CAMPAIGN_STATE_FILE,
        campaign_done_file=CAMPAIGN_DONE_FILE,
        control_log_file=CONTROL_LOG_FILE,
        eval_prompt_sets=EVAL_PROMPT_SETS,
        extra_pin_paths=(
            "pipelines/opdlim_campaign_pipeline.py",
            "pipelines/opdlim_results_pipeline.py",
            "scripts/run_opdlim_campaign.py",
        ),
    )


SPEC_T1 = _spec(TIER1)
SPEC_T2 = _spec(TIER2)
SPEC_T3 = _spec(TIER3)
SPEC_T4 = _spec(TIER4)
SPECS: MappingProxyType[int, CampaignSpec] = MappingProxyType(
    {1: SPEC_T1, 2: SPEC_T2, 3: SPEC_T3, 4: SPEC_T4}
)
