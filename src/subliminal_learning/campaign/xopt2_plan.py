"""Frozen constants of `xopt2`: learning-rate walks to collapse, five new optimizers and the
checkpoint-grid fill, on the `xopt` cross-model / checkpoint design, both channels.

Plan: `.claude/plans/blogpost/13_xopt2_lr_sweep_new_optimizers.plan.md`. Pre-registration:
`docs/PREREG_XOPT2.md`. Teachers, pools, staged bases and placebo twins are `xopt`'s, read in
place (`xopt_plan` roots); `xopt2` writes only under its own roots.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from subliminal_learning.campaign import xopt_plan as xp
from subliminal_learning.campaign.xopt2_rules import Walk, ladder

CAMPAIGN = "xopt2"
OWNER = "vishesh"
DATE = "20261017"
PREREG_PATH = Path("docs/PREREG_XOPT2.md")
PLAN_PATH = Path(".claude/plans/blogpost/13_xopt2_lr_sweep_new_optimizers.plan.md")

HARD, SOFT = xp.HARD, xp.SOFT
CHANNELS: tuple[str, ...] = (HARD, SOFT)
RUN_SET_IDS: MappingProxyType[str, str] = MappingProxyType(
    {HARD: f"{DATE}_xopt2-hard_160m", SOFT: f"{DATE}_xopt2-soft_160m"}
)

# ---- hosts and GPUs ------------------------------------------------------------------

GONCALO = xp.GONCALO
SOAR = xp.SOAR
HOME_NODE = SOAR
# Worker hosts per channel: goncalo from launch, cluster-3 / cluster-4 from 2026-10-18,
# soar (hard only) once newdom2 frees it. Soft stays off soar: every soft job queued before
# then carries a guard naming the other three hosts (plan 13 status log 2026-10-18).
CHANNEL_HOSTS: MappingProxyType[str, tuple[str, ...]] = MappingProxyType(
    {
        HARD: (GONCALO, xp.CLUSTER3, xp.CLUSTER4, SOAR),
        SOFT: (GONCALO, xp.CLUSTER3, xp.CLUSTER4),
    }
)
WORKER_HOSTS: tuple[str, ...] = tuple(
    dict.fromkeys(h for hosts in CHANNEL_HOSTS.values() for h in hosts)
)
HOST_TAG = xp.HOST_TAG
CLONE_ROOT = xp.CLONE_ROOT
LOCK_PREFIX = xp.LOCK_PREFIX
# One pool per channel root on each worker host, on disjoint cards (soft GKD peaks grow
# after the memory gate's warm-up window: rev2 goncalo OOM, plan 12 status log
# 2026-10-16 01:29Z). Re-split by policy hot reload once one channel's queues drain
# (plan §8). soar runs the hard channel on every card.
GPUS: MappingProxyType[str, tuple[int, ...]] = MappingProxyType(
    {HARD: (0, 1, 2, 3, 4), SOFT: (5, 6, 7)}
)
HOST_GPUS: MappingProxyType[str, Mapping[str, tuple[int, ...]]] = MappingProxyType(
    {SOAR: MappingProxyType({HARD: tuple(range(8))})}
)


def gpus(channel: str, host: str) -> tuple[int, ...]:
    """Cards of `channel` on `host` (none off the channel's worker hosts)."""
    if host not in CHANNEL_HOSTS[channel]:
        return ()
    return HOST_GPUS.get(host, {}).get(channel, GPUS[channel])


def queue_defaults(channel: str, per_host: Mapping[str, dict]) -> dict:
    """jobq defaults of a queue every worker host of `channel` drains.

    `per_host` maps each such host to its code-pin defaults (``cwd`` / env). The
    queue-wide ones are goncalo's, whose pools started on jobq code that reads no
    ``nodes`` entry; every host's own are under ``nodes``.

    Raises:
        ValueError: `per_host` misses a worker host of `channel`.
    """
    hosts = CHANNEL_HOSTS[channel]
    if missing := set(hosts) - set(per_host):
        raise ValueError(f"no pin defaults for {sorted(missing)}")
    return {**per_host[GONCALO], "nodes": {h: dict(per_host[h]) for h in hosts}}


# ---- roots ----------------------------------------------------------------------------
# Control (config, queue, run-set logs / results): ssd-2, fast from goncalo and soar
# (docs/STORAGE_SSOT.md §2 placement rule). Student leaves: cold-1, as `xopt`'s goncalo
# hard students and soft leaves (user-approved for xopt 2026-10-14); a free-space floor
# holds new submissions (labelled `disk_low`), nothing is ever deleted.

CTRL_ROOTS: MappingProxyType[str, Path] = MappingProxyType(
    {
        HARD: Path("/mnt/ssd-2/soar-subliminal/vishesh/xopt2/hard"),
        SOFT: Path("/mnt/ssd-2/soar-subliminal/vishesh/xopt2/soft"),
    }
)
LEAF_ROOTS: MappingProxyType[str, Path] = MappingProxyType(
    {
        HARD: Path("/mnt/cold-1/soar-subliminal/vishesh/xopt2/hard/outputs"),
        SOFT: Path("/mnt/cold-1/soar-subliminal/vishesh/xopt2/soft/outputs"),
    }
)
QUEUE_ROOTS: MappingProxyType[str, Path] = MappingProxyType(
    {ch: CTRL_ROOTS[ch] / "queue" for ch in CHANNELS}
)
MIN_FREE_GIB: MappingProxyType[str, int] = MappingProxyType(
    {"ssd-2": 60, "cold-1": 1_000}
)

# Reused `xopt` inputs (read only).
XOPT_HARD_ROOT = xp.HARD_ROOT
XOPT_SOFT_ROOT = xp.SOFT_ROOT
XOPT_RUN_SET_IDS = xp.RUN_SET_IDS


def config_dir(channel: str) -> Path:
    return CTRL_ROOTS[channel] / "config"


def paths_config_path(channel: str) -> Path:
    return config_dir(channel) / f"paths_xopt2_{channel}.yaml"


def output_root(channel: str) -> Path:
    """Run-set parent root (logs, results, state) on ssd-2."""
    return CTRL_ROOTS[channel] / "outputs"


def run_set_dir(channel: str) -> Path:
    sub = "gkd_students" if channel == SOFT else CAMPAIGN
    return output_root(channel) / sub / RUN_SET_IDS[channel]


def results_dir(channel: str) -> Path:
    return run_set_dir(channel) / "results"


def logs_dir(channel: str) -> Path:
    return run_set_dir(channel) / "logs"


# Campaign-level marker (fleet card): written create-only by whichever channel driver
# finalizes second, once both channel markers exist.
CAMPAIGN_DONE_PATH = Path("/mnt/ssd-2/soar-subliminal/vishesh/xopt2/CAMPAIGN_DONE")
TMUX_SESSION = CAMPAIGN


def channel_done_path(channel: str) -> Path:
    """The channel's own done marker (its pool loop exits on it)."""
    return logs_dir(channel) / "CAMPAIGN_DONE"


def card_dir(channel: str) -> Path:
    """Dir of the channel's run-set card (code pin): hard its ssd-2 run set; soft the
    cold-1 leaf root's `gkd_students` run set, where the GKD sweeps keep it."""
    if channel == SOFT:
        return LEAF_ROOTS[SOFT] / "gkd_students" / RUN_SET_IDS[SOFT]
    return run_set_dir(channel)


# ---- pairs and steps (xopt's) ---------------------------------------------------------

FINAL = xp.FINAL
PY = xp.PY
TIER_C_PAIRS = xp.TIER_C_PAIRS
TIER_C_SELF = xp.TIER_C_SELF
SECOND_STEP = 115_000  # hard: Exp 1 walk + carry on pythia-160m@115k self (dog)
HARD_GRID_STEPS: tuple[int, ...] = (*xp.HARD_K_STEPS, FINAL)
SOFT_GRID_STEPS: tuple[int, ...] = (*xp.SOFT_R_STEPS, FINAL)
SOFT_GRID_PAIRS = xp.SOFT_GRID_PAIRS


def off_cross(steps: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    """(teacher step, student step) cells of the grid on `steps` that `xopt` left empty.

    `xopt` trained the diagonal, the row (student at FINAL) and the column (teacher at
    FINAL); the rest of the square is the off-cross set.
    """
    return tuple((t, s) for t in steps for s in steps if t != s and FINAL not in (t, s))


# ---- optimizers -----------------------------------------------------------------------
# Arm optimizer token -> `TrainConfig.optim` (HF names for built-ins, ours for custom).
OPTIM_NAMES: MappingProxyType[str, str] = MappingProxyType(
    {
        "adamw": "adamw_torch",
        "sgd": "sgd",
        "rmsprop": "rmsprop",
        "muon": "muon",
        "lion": "lion",
        "signsgd": "signsgd",
        "adafactor": "adafactor",
        "adammini": "adam_mini",
    }
)
NOHEAD: tuple[str, ...] = ("embed_out",)

# ---- arms -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Arm:
    """One optimizer x parameterization of a channel.

    Attributes:
        name: Arm id (`<mode>-<optim>[-nh]`).
        walk: Bracket walk (walked arms), or None.
        ladder: `xopt` step-down ladder, highest first (fixed-rate use), or ().
    """

    name: str
    channel: str
    mode: str
    optim: str
    freeze: tuple[str, ...] = ()
    walk: Walk | None = None
    ladder: tuple[float, ...] = ()

    @property
    def rates(self) -> frozenset[float]:
        return frozenset((*(self.walk.rates if self.walk else ()), *self.ladder))

    def lower_rungs(self, rate: float) -> tuple[float, ...]:
        """Step-down rungs below `rate`, highest first (walk coarse rungs or ladder)."""
        pool = self.walk.coarse if self.walk is not None else self.ladder
        return tuple(sorted((r for r in pool if r < rate), reverse=True))


def _walk(lo: float, hi: float, start: float, floor: tuple[float, ...] = ()) -> Walk:
    coarse, mids = ladder(lo, hi)
    return Walk(coarse, mids, start, floor)


def _arm(
    channel: str, name: str, walk: Walk | None, ladder_: tuple[float, ...] = ()
) -> Arm:
    mode, optim, *rest = name.split("-")
    return Arm(
        name, channel, mode, optim, NOHEAD if rest == ["nh"] else (), walk, ladder_
    )


# Hard (number SFT, E3 recipe, 25,000 steps). Exp 1 arms walk from the `xopt` anchor with
# the `xopt` P2 lower rung as floor; new arms walk from about two decades below the
# reference arm's collapse (full AdamW collapses at 1e-5, LoRA AdamW r8a32 at 1.6e-4,
# hlopt Part I); sign-based arms step |lr| per coordinate, so their ladders sit one decade
# lower (Chen et al. 2023 §4: 3-10x smaller lr than AdamW).
_H_FULL = (1e-7, 1e-4)
_H_SIGN_FULL = (1e-8, 1e-5)
_H_LORA = (1e-6, 1e-2)
_H_SIGN_LORA = (1e-7, 1e-3)
HARD_ARMS: MappingProxyType[str, Arm] = MappingProxyType(
    {
        a.name: a
        for a in (
            _arm(HARD, "lora-adamw", None, xp.HARD_ARMS["lora-adamw"].ladder),
            _arm(HARD, "lora-sgd", None, xp.HARD_ARMS["lora-sgd"].ladder),
            _arm(HARD, "full-sgd", None, xp.HARD_ARMS["full-sgd"].ladder),
            _arm(
                HARD,
                "full-adamw",
                _walk(1e-6, 1e-4, 5e-6, (2e-6,)),
                xp.HARD_ARMS["full-adamw"].ladder,
            ),
            _arm(
                HARD,
                "full-rmsprop",
                _walk(1e-6, 1e-4, 5e-6, (2e-6,)),
                xp.HARD_ARMS["full-rmsprop"].ladder,
            ),
            _arm(HARD, "full-adamw-nh", _walk(*_H_FULL, 2e-6)),
            _arm(HARD, "full-muon", _walk(*_H_FULL, 5e-7)),
            _arm(HARD, "full-muon-nh", _walk(*_H_FULL, 5e-7)),
            _arm(HARD, "full-adafactor", _walk(*_H_FULL, 5e-7)),
            _arm(HARD, "full-adammini", _walk(*_H_FULL, 5e-7)),
            _arm(HARD, "full-lion", _walk(*_H_SIGN_FULL, 5e-8)),
            _arm(HARD, "full-signsgd", _walk(*_H_SIGN_FULL, 5e-8)),
            _arm(HARD, "lora-muon", _walk(*_H_LORA, 1e-5)),
            _arm(HARD, "lora-adafactor", _walk(*_H_LORA, 1e-5)),
            _arm(HARD, "lora-adammini", _walk(*_H_LORA, 1e-5)),
            _arm(HARD, "lora-lion", _walk(*_H_SIGN_LORA, 1e-6)),
            _arm(HARD, "lora-signsgd", _walk(*_H_SIGN_LORA, 1e-6)),
        )
    }
)
# Soft (on-policy GKD, 625 steps, batch 32). `xopt` soft anchors sit 4x above the hard ones
# (full AdamW 2e-5 vs 5e-6, LoRA AdamW 4e-4 vs 1e-4, plan 11 §3.1); new-arm starts follow.
_S_FULL = (1e-7, 1e-3)
_S_SIGN_FULL = (1e-8, 1e-4)
_S_LORA = (1e-6, 1e-1)
_S_SIGN_LORA = (1e-7, 1e-2)
SOFT_ARMS: MappingProxyType[str, Arm] = MappingProxyType(
    {
        a.name: a
        for a in (
            _arm(SOFT, "lora-adamw", None, xp.SOFT_ARMS["lora-adamw"].ladder),
            _arm(SOFT, "full-adamw", None, xp.SOFT_ARMS["full-adamw"].ladder),
            _arm(SOFT, "full-sgd", None, xp.SOFT_ARMS["full-sgd"].ladder),
            _arm(
                SOFT,
                "lora-sgd",
                _walk(0.1, 5.0, 0.5),
                xp.SOFT_ARMS["lora-sgd"].ladder,
            ),
            _arm(SOFT, "full-adamw-nh", _walk(*_S_FULL, 5e-6)),
            _arm(SOFT, "full-muon", _walk(*_S_FULL, 2e-6)),
            _arm(SOFT, "full-muon-nh", _walk(*_S_FULL, 2e-6)),
            _arm(SOFT, "full-adafactor", _walk(*_S_FULL, 2e-6)),
            _arm(SOFT, "full-adammini", _walk(*_S_FULL, 2e-6)),
            _arm(SOFT, "full-lion", _walk(*_S_SIGN_FULL, 2e-7)),
            _arm(SOFT, "full-signsgd", _walk(*_S_SIGN_FULL, 2e-7)),
            _arm(SOFT, "lora-muon", _walk(*_S_LORA, 5e-5)),
            _arm(SOFT, "lora-adafactor", _walk(*_S_LORA, 5e-5)),
            _arm(SOFT, "lora-adammini", _walk(*_S_LORA, 5e-5)),
            _arm(SOFT, "lora-lion", _walk(*_S_SIGN_LORA, 5e-6)),
            _arm(SOFT, "lora-signsgd", _walk(*_S_SIGN_LORA, 5e-6)),
        )
    }
)
ARMS: MappingProxyType[str, MappingProxyType[str, Arm]] = MappingProxyType(
    {HARD: HARD_ARMS, SOFT: SOFT_ARMS}
)
NEW_ARMS: tuple[str, ...] = (
    "full-adamw-nh",
    "full-muon",
    "full-muon-nh",
    "full-adafactor",
    "full-adammini",
    "full-lion",
    "full-signsgd",
    "lora-muon",
    "lora-adafactor",
    "lora-adammini",
    "lora-lion",
    "lora-signsgd",
)
EXP1_ARMS: MappingProxyType[str, tuple[str, ...]] = MappingProxyType(
    {HARD: ("full-adamw", "full-rmsprop"), SOFT: ("lora-sgd",)}
)

# Fixed-rate slots (arm, rate): validity (v), grid fill (g). xopt anchors.
VALIDITY_SLOTS: MappingProxyType[str, tuple[tuple[str, float], ...]] = MappingProxyType(
    {
        HARD: (
            ("lora-adamw", 1e-4),
            ("full-sgd", 0.02),
            ("full-adamw", 5e-6),
            ("full-rmsprop", 5e-6),
        ),
        SOFT: (("lora-adamw", 4e-4), ("full-adamw", 2e-5), ("lora-sgd", 0.3)),
    }
)
GRID_SLOTS: MappingProxyType[str, tuple[tuple[str, float], ...]] = MappingProxyType(
    {
        HARD: (
            ("lora-adamw", 1e-4),
            ("lora-sgd", 0.035),
            ("full-sgd", 0.02),
            ("full-adamw", 5e-6),
        ),
        SOFT: (
            ("full-adamw", 2e-5),
            ("full-sgd", 0.1),
            ("lora-sgd", 0.3),
            ("lora-adamw", 4e-4),
        ),
    }
)
# Soft LoRA AdamW was not on the xopt P1 grids, so it fills every non-corner cell.
SOFT_GRID_FULL_SQUARE_ARMS: tuple[str, ...] = ("lora-adamw",)

# ---- seeds and animals ----------------------------------------------------------------

HARD_SEEDS: tuple[int, ...] = (1, 2)
HARD_EXTRA_SEED = 3
HARD_GRID_SEED = 1
SOFT_LEAVES: tuple[int, ...] = (0, 1)
SOFT_EXTRA_LEAF = 2
ANIMALS_HARD = xp.ANIMALS_HARD
ANIMAL_HARD_STEP = xp.ANIMAL_K  # dog: 115k self and the hard grid
ANIMALS_SOFT_SELF = xp.ANIMALS_SOFT  # bear + lion at the py@143k self corner
ANIMAL_SOFT = xp.ANIMAL_G  # bear: other corners and grids

# ---- tiers and priorities --------------------------------------------------------------
# v validity, w Exp 1 walks (143k pairs), wk Exp 1 walk at 115k (hard), o new-arm walks on
# py@143k self, x extra seed at decision rates, ox carry of new arms to the other pairs,
# ok carry to py@115k (hard), g grid fill, walk / down = decision submissions, late = lm-eval.
PRIORITY: MappingProxyType[str, int] = MappingProxyType(
    {
        "v": 97,
        "walk": 96,
        "down": 96,
        "w": 95,
        "wk": 95,
        "o": 95,
        "x": 80,
        "ox": 75,
        "ok": 75,
        "g": 60,
        "late": 10,
    }
)
TIERS: MappingProxyType[str, tuple[str, ...]] = MappingProxyType(
    {
        HARD: ("v", "w", "wk", "o", "x", "ox", "ok", "g"),
        SOFT: ("v", "w", "o", "x", "ox", "g"),
    }
)
PHASES: MappingProxyType[str, tuple[str, ...]] = MappingProxyType(
    {"p1": ("v", "w", "wk", "o"), "p2": ("x", "ox", "ok"), "p3": ("g",)}
)

# ---- footprints (measured: xopt hard on goncalo, optparam / rev OOM repairs) ----------

HARD_MEM_MIB: MappingProxyType[str, int] = MappingProxyType(
    {"lora": 6_200, "full": 8_500, "late": 11_000}
)
HARD_CAP_PER_GPU = 6
SOFT_MEM_MIB: MappingProxyType[str, int] = MappingProxyType(
    {"lora": 19_000, "full": 22_000, "late": 12_000}
)
SOFT_CAP_PER_GPU = 2
FREE_MEM_MIB = 2_000
RESERVATION_WARMUP_S = 180
AUTO_REQUEUE_ATTEMPTS = 3
N_GPUS = 8
