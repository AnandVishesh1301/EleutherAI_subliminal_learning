"""Measured duration priors for jobq jobs, plus the node capacity the ETA schedules on.

Two things ``scripts.queue_status`` cannot get from one queue alone:

1. **How long will a job of this shape take?** Every successful result in every history
   root (the status root plus ``run_logs/queue0`` / ``run_logs/queue_angela``) is keyed by
   a JOB SIGNATURE derived from its command. The command's SCRIPTS give the job's ROLE
   (`chain_role`) and the role decides what its cost is keyed and scaled by -- a job that
   merely carries ``--epochs`` (number generation off a teacher recipe, a preference eval)
   does NOT train, and a TRAINING result under two minutes did not train either (it hit its
   own skip gate: those cache hits are dropped from every training key, `is_cache_hit`):

   - ``train|model=<arch>|ep=<E>`` -- a student distillation, by receiver arch and epochs and
     nothing else, in particular NOT the script: ``pythia-160m`` at 20 epochs takes ~47 m
     whether it was run by ``dog_crossdistill_studenthp_worker``, ``dd1b_worker`` or
     ``hf_worker distill --family qwen25``. The skip-gated checkpoint-distillation cells
     (``sweep_ckpt_distill``) are their own population, keyed ``train|ckptdistill|…``. Refined
     ``|r=<rank>`` and ``|<fp32|tf32>`` where enough samples exist (levels 4..1), then -- for a
     shape never measured -- the fitted cost MODEL (below), then the
     same receiver at another epoch count scaled ``t(E) = t_ref * (E+1) / (E_ref+1)``
     (a ~1-epoch fixed overhead fits the measured ep5 vs ep20 pairs), quoted ``ep20<-ep5``.
   - ``teacher|model=<arch>|ep=<E>`` (same refinements) -- a teacher fine-tune sweep, priced
     per CELL: a job's units = the grid cells its ``--num-shards/--shard-index`` slice trains,
     narrowed by ``--only-param-tags``. A grid that mixes epochs or ranks is keyed on the
     STRICTLY dominant value and otherwise left unrefined (`_dominant`).
   - ``numgen|model=<arch>`` -- number generation, priced per 1,000 PROMPTS: units =
     teachers x n_prompts/1000.
   - ``eval|<set>|model=<arch>`` -- evals and probes, priced per RUN (``--runs`` lines,
     prompt sets, teacher cells).
   - ``traj|<script>|model=<arch>`` -- checkpoint trajectories, priced per CHECKPOINT run
     (the ``--registry`` entries of this ``--tag``, a ``--runs`` list, else ``--ckpts``).
   - anything else keeps the script sequence alone, then the per-script SUM for a chain never
     measured as a whole (``a>b>c`` whose components each ran alone), then ``--unknown-job-s``.
     Chain composition uses the per-unit priors of its components.

   Receiver, epochs, rank, precision and units are recovered from the command's flags AND
   from the YAMLs it points at (``--sweep-config`` carries ``base_model``/``base_models``,
   the HP grid and the cell count; ``--training-config`` / ``--student-training-config``
   carry ``lora.r``, ``train.n_epochs`` and ``train.tf32``; ``--only-param-tags``,
   ``--registry`` and ``--runs`` carry unit counts); those reads are cached per path. The
   external-family worker names its recipe by ``--family`` (``lora_student_distill_qwen25``
   = tf32, ``dd1b_worker`` defaults to the fp32 olmo1b recipe unless given a ``_tf32``
   variant), and an explicit ``_tf32``/``_fp32`` marker in the command wins. Because the
   YAML read today misattributes older runs, a sample counts as ``tf32`` only when its
   ``end_utc`` is after the commit that introduced ``tf32: true`` in that YAML
   (``git log -S``, cached).

   Pools drop failures (``rc != 0``) and then every sample below ``0.25 x p90`` of the pool,
   so the resume/cache-hit mode of a bimodal kind (``dd1b_worker distill``: 3 m vs 2 h06)
   never drags the prior down. Samples are pooled plainly (one vote each), except that a
   single queue contributes at most its NEWEST ``MAX_PER_QUEUE`` (2,000) samples so one
   campaign of 74k jobs cannot swamp a kind it shares with everyone else.

   **Which samples.** Machines and settings changed over time, so every key quotes the first
   TIER that holds ``min_n`` samples, newest first and capped at ``--prior-recent-n``: ``t1``
   = the nodes today's work runs on (`TIER_NODES`) since the MPS rollout (`MPS_SINCE`),
   ``t2`` = any node since then, ``t3`` = everything ever measured. As DIAGNOSTICS (never in
   the quote) each sample also carries the co-tenancy it ran at -- the time-weighted number
   of other jobs of the same root on its ``(node, gpu)``, bucketed ``round(1 + mean)`` -- and
   whether CUDA MPS was on (started after `MPS_SINCE` and its covering
   ``pool_<stamp>.<host>.log`` opens with ``MPS enabled (daemon up)``), plus the job's peak
   GPU MiB when the worker recorded one (``peak_mem_mib``, the torch reserved high-water mark;
   the historical ``mem_mib`` asks and OOM floors are NOT trustworthy and are not tabulated). Loads this repo never
   recorded (direct runs, foreign users) are invisible.

   **The cost model.** Rows are not the story: a duration is REASONED from
   ``per-epoch cost(arch, precision) x epochs x rank factor(rank) x load factor(d)``
   (`build_model`, `ArchModel`), fitted at `build_priors` time from the tier-1 samples with the
   same resume filter, applied per EPOCH GROUP before anything is bucketed. Every quantity is
   measured WITHIN an epoch group and combined across groups by an N-WEIGHTED median (a 200-run
   ep5 group must not outvote 66k ep20 runs): the per-epoch cost from ``p50/epochs`` over the
   pooled loaded regime (`MODEL_LOADED` = 3-6 jobs per GPU, the one campaigns run in), the rank
   factor from ``p50(r) / p50(rank <= MODEL_BASE_RANK)`` and the load factor from
   ``p50(d) / p50(loaded)``, each only where a cell holds `MODEL_MIN_N` samples (else 1.0 / the
   nearest measured neighbour). Buckets are never pooled across epoch counts, which would price
   the GPU load with whatever epoch count happened to run at it, and the load factors are only
   APPLIED where they measure contention rather than campaign phase
   (`load_factors_trustworthy`); elsewhere they are reported but every `load_mult` is 1.0. What a duration does NOT depend on: the worker/queue that ran
   it, and the rank below 64. The model carries its own n and the p10/p90 residual band against
   its samples, and quotes at the caller's ``density`` (a node's jobs per busy GPU,
   `MODEL_DEFAULT_BUCKET` when unknown). The other roles keep per-UNIT rates (`UnitRate`).

   Scanning ~113k result files takes ~1 min, so the table is cached in the status root
   (``duration_priors_cache.json``): per queue a fingerprint (result-file count + the
   ``results/`` dir mtime, one stat) decides a rescan -- including for a queue stamped ``complete``, which can still
   grow when it is requeued.
   Parsing the ~29k pending/running commands of a live root costs as much again, so `FeatsCache`
   memoises ``cmd -> Feats`` in a world-writable ``cmd_feats_cache.json`` beside it, each
   entry keyed on the mtimes of the YAMLs the command reads.

2. **How fast can a node actually run them?** ``node_capacity`` reads each
   ``gpu_policy.<host>.json``: slots = ``(len(gpus) - yielded gpus) * cap_per_gpu`` (40 on
   the 8-GPU nodes), with a separate ``cpu_cap`` lane for ``slots: 0`` jobs. A policy that
   omits ``cap_per_gpu`` warns and falls back to `DEFAULT_CAP_PER_GPU` rather than pricing the
   node at zero slots, and a node running jobs with no policy file here is priced at its
   current OCCUPANCY, flagged ``capacity_source: occupancy``. A node counts
   as LIVE only if it holds on-GPU claims now or its newest ``pool_*.<host>.log`` was
   touched within ``pool_live_min``; a policy file without a live pool contributes nothing.
"""

from __future__ import annotations

import heapq
import json
import math
import re
import shlex
import statistics
import subprocess
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import yaml
from loguru import logger

from subliminal_learning.jobqueue import store
from subliminal_learning.utils.io import atomic_write_json

CACHE_NAME = "duration_priors_cache.json"
FEATS_CACHE_NAME = "cmd_feats_cache.json"  # cmd -> parsed features, shared by every caller
CACHE_VERSION = 7  # 7: ckptdistill split by subcommand (6: peak_mem_mib sample column)
MAX_PER_QUEUE = 2000
DEFAULT_MIN_N = 8
MIN_JOB_S = 120.0
CACHE_HIT_FRAC = 0.1
RESUME_FRAC = 0.25
BLEND_K = 10.0  # own samples needed for an even own/prior split
OWN_DOMINANT_N = 10
OWN_MIN_N = 3
COMPOSE_MIN_N = 3  # samples a chain COMPONENT needs (a thin measurement beats the default)
# Repo root: .../src/subliminal_learning/jobqueue/eta_priors.py -> parents[3]. The history
# roots and the git call below are anchored on it, so a status run from any cwd reads the
# same history.
REPO_ROOT = Path(__file__).resolve().parents[3]
HISTORY_ROOTS = (str(REPO_ROOT / "run_logs/queue0"), str(REPO_ROOT / "run_logs/queue_angela"))
MPS_SINCE = "2026-08-12T00:00Z"  # pools default `mps: true` from this rollout on
TIER_NODES = ("soar-subliminal-1", "goncalo-temp-0")  # the nodes today's numbers come from
DEFAULT_RECENT_N = 3000  # newest samples a key quotes from within its tier
MIN_CELL_N = 5  # samples a per-bucket / per-precision cell needs before it is quoted

_RECEIVER_FLAGS = ("--student-base", "--base", "--base-models")
_RANK_FLAGS = ("--student-rank", "--rank")
_TAG_FLAGS = ("--param-tag", "--config-tag", "--tag")
_TRAIN_CFG_FLAGS = ("--student-training-config", "--training-config")
_RANK_IN_TAG = re.compile(r"r(\d+)a")
_EPOCHS_IN_TAG = re.compile(r"(?:^|_)ep(\d+)")
_TRAIN_KEY = re.compile(r"^(?P<role>train|teacher)(?:\|(?P<kind>(?!model=)[^|]+))?"
                        r"\|model=(?P<arch>[^|]+)\|ep=(?P<ep>[^|]+)"
                        r"(?:\|r=(?P<rank>\d+))?(?:\|(?P<prec>fp32|tf32))?$")
_UNIT_KEY = re.compile(r"^(?P<role>teacher|numgen|eval|traj)(?:\|(?P<kind>(?!model=)[^|]+))?"
                       r"\|model=(?P<arch>[^|]+)(?:\|r=(?P<rank>\d+))?"
                       r"(?:\|(?P<prec>fp32|tf32))?$")
_SHORT_BASE = re.compile(r"^(ds|ws|fs|wc)(\d+)$")
_SHORT_VARIANT = {"ds": "data-seed", "ws": "weight-seed", "fs": "full-seed",
                  "wc": "weightc-seed"}

# Every role a job can have, in the order the tables list them. A job is a TRAINER only when
# it runs a trainer script: `--epochs` / an `ep{N}` tag also ride on number generation
# (`generate_numbers --training-config .../teacher_ep2_...yaml`), teacher sweeps and
# preference evals, all of which used to be pooled as training and dragged those keys down.
ROLES = ("train", "teacher", "numgen", "eval", "traj", "other")
# Non-training scripts, by the spelling `job_feats` gives them (module path / file stem +
# subcommand). Anything unlisted that runs a `*_worker` (bare or `distill`) or a distill
# script TRAINS; anything else is `other` and keeps its script-chain key.
_SCRIPT_ROLES = {
    "scripts.sweeps.sweep_teacher_templates": "teacher",
    "scripts.train_teacher_templates": "teacher",
    "scripts.generate_numbers": "numgen",
    "scripts.sweeps.sweep_generate_numbers": "numgen",
    "scripts.sweeps.sweep_ckpt_distill numgen-cell": "numgen",
    "dog_numgen_worker": "numgen",
    "scripts.run_final_evals": "eval",
    "scripts.evaluate_student_preference": "eval",
    "scripts.sweeps.sweep_evaluate_teacher_preference": "eval",
    "scripts.ckpt_pile_kl_trajectory": "traj",
    "scripts.ckpt_preference_trajectory": "traj",
    "scripts.fidelity_ckpt_trajectory": "traj",
    "scripts.teacher_pile_kl_trajectory": "traj",
    "scripts.evaluate_checkpoint_teachers": "traj",
}
_TRAIN_SCRIPTS = ("scripts.distill_student", "scripts.sweeps.sweep_ckpt_distill run-cell",
                  "scripts.sweeps.sweep_ckpt_distill train-cell")
# A checkpoint-distillation cell trains, but it is skip-gated (a cached cell returns in
# seconds), so its durations are their own population: it keys as ``train|ckptdistill|…``
# and never pools with a plain student training of the same arch and epochs.
_CKPT_DISTILL = "scripts.sweeps.sweep_ckpt_distill"
CKPT_DISTILL_KIND = "ckptdistill"
# ... and its two TRAINING subcommands are two different jobs: `train-cell` fine-tunes a
# TEACHER off a base checkpoint (its `--param-tag` is that teacher's recipe), `run-cell`
# distils a STUDENT whose rank lives in the sweep YAML's `student.training_config`. Keyed
# together, the teacher's tag priced the student's rank.
CKPT_DISTILL_SUBS = ("train-cell", "run-cell")
# Flags whose file contents change a command's features: an edited YAML must invalidate the
# parse (`FeatsCache`, `job_feats`), not be served from a cache keyed on the string alone.
_REF_PATH_FLAGS = (*_TRAIN_CFG_FLAGS, "--sweep-config", "--only-param-tags")
# `run_logs/hf_worker.py --family X`: base token and student-recipe YAML suffix per family
# (mirrors that worker's FAMILIES table, which cannot be imported without loading torch).
_HF_FAMILY_BASE = {"qwen25": "qwen25-1.5b", "llama3": "llama3-1b",
                   "gemma2": "gemma2-2b", "gemma3": "gemma3-1b"}
_HF_STUDENT_CFG = "configs/training/lora_student_distill_{family}.yaml"
DEFAULT_N_PROMPTS = 100_000  # configs/experiment/number_generation.yaml's prompt_set
_N_PROMPTS_IN_TAG = re.compile(r"_n(\d+)k(?:_|$)")


# --------------------------- job signature ---------------------------


@lru_cache(maxsize=None)
def model_arch(token: str) -> str | None:
    """Base-model spelling -> architecture (``pythia-160m``/``pythia-410m``/``olmo-1b``/``qwen``)
    via ``models.naming.base_spec``, ignoring the seed variant; ``None`` if unparseable.

    Campaign shorthands (``ws3`` = ``weight-seed3``, ``ds1``, ``fs2``, ``wc1``) are expanded
    first: they appear as ``--student-base`` spellings and are all pythia-160m.
    """
    from subliminal_learning.models.datadecide import DataDecideSpec
    from subliminal_learning.models.naming import base_spec
    from subliminal_learning.models.registry import ExternalModelSpec, PythiaModelSpec

    m = _SHORT_BASE.match(token.strip())
    text = f"{_SHORT_VARIANT[m.group(1)]}{m.group(2)}" if m else token
    try:
        spec = base_spec(text)
    except Exception:  # noqa: BLE001 - any unparseable spelling
        return None
    if isinstance(spec, PythiaModelSpec):
        return f"pythia-{spec.size}"
    if isinstance(spec, DataDecideSpec):
        return f"olmo-{spec.size.lower()}"
    if isinstance(spec, ExternalModelSpec):
        repo = spec.repo_id.split("/")[-1]
        return "qwen" if repo.lower().startswith("qwen") else repo
    return None


def _resolve_path(path: str) -> Path:
    """A command's config/list path, anchored on `REPO_ROOT` when it is relative.

    Commands are written as they are run from the checkout (``--training-config
    configs/...``), but the status tool runs from anywhere; resolving against the process
    cwd silently turned every relative YAML into an unreadable path (no rank, no epochs).
    """
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _file_sig(path: str) -> str:
    """``mtime_ns:size`` of a referenced file, ``-`` when it is absent/unreadable.

    Every cached read of a YAML/list is keyed on this, so an edited config invalidates the
    parse instead of serving the features of the file's previous contents. The size is in
    there because a filesystem whose mtime granularity is a second (some of the shared
    mounts) would otherwise hide an edit made within one.
    """
    try:
        st = _resolve_path(path).stat()
    except OSError:
        return "-"
    return f"{st.st_mtime_ns}:{st.st_size}"


@lru_cache(maxsize=None)
def probe_list_info(path: str) -> tuple[str | None, int] | None:
    """(probed-model arch, run count) of a ``--runs`` list file; ``None`` if unreadable.

    The arch comes from the first path component (after ``students/``) that parses as a base.
    """
    try:
        lines = [ln.strip() for ln in
                 _resolve_path(path).read_text().splitlines() if ln.strip()]
    except OSError:
        return None
    arch = None
    for ln in lines[:1]:
        parts = ln.split("/")
        tail = parts[parts.index("students") + 1:] if "students" in parts else parts
        for comp in tail:
            if "-" in comp and not comp.startswith("teacher-"):
                arch = model_arch(comp)
                if arch:
                    break
    return arch, len(lines)


def _read_yaml(path: str) -> dict:
    try:
        data = yaml.safe_load(_resolve_path(path).read_text())
    except (OSError, ValueError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


@lru_cache(maxsize=None)
def sweep_config_info(path: str) -> tuple[str | None, str | None]:
    """(receiver arch, epochs) of a sweep YAML: ``base_model``/``base_models`` and, when the
    grid sweeps ONE epoch count, ``grid.n_epochs`` (a multi-epoch grid tells us nothing)."""
    d = _read_yaml(path)
    base = d.get("base_model") or d.get("base_models")
    arch = model_arch(re.split(r"[,|]", str(base))[0]) if base else None
    eps = ((d.get("grid") or {}) if isinstance(d.get("grid"), dict) else {}).get("n_epochs")
    ep = str(eps[0]) if isinstance(eps, list) and len(eps) == 1 else None
    return arch, ep


def training_config_info(path: str) -> tuple[int | None, str | None, bool]:
    """(LoRA rank, epochs, tf32) of a training YAML (``lora.r``, ``train.n_epochs``,
    ``train.tf32`` -- see ``students.config`` / ``config.build_train_config``)."""
    return _training_config_info(path, _file_sig(path))


@lru_cache(maxsize=None)
def _training_config_info(path: str, _stamp: str) -> tuple[int | None, str | None, bool]:
    d = _read_yaml(path)
    lora = d.get("lora") if isinstance(d.get("lora"), dict) else {}
    train = d.get("train") if isinstance(d.get("train"), dict) else {}
    try:
        rank = int(lora.get("r"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        rank = None
    ep = str(train["n_epochs"]) if train.get("n_epochs") is not None else None
    return rank, ep, bool(train.get("tf32", False))


def script_role(name: str) -> str:
    """Role of one script of a chain (`ROLES`); ``other`` when it is not a known trainer."""
    stem, _, sub = name.partition(" ")
    if name in _TRAIN_SCRIPTS or (stem.endswith("_worker") and sub in ("", "distill")):
        return "train"
    return _SCRIPT_ROLES.get(name, "other")


def chain_role(scripts: list[str]) -> str:
    """Role of a whole chain: the most training-heavy role any of its scripts has.

    ``animal_samedistill_worker>evaluate_student_preference*4`` trains (and then evaluates),
    ``preflight>sweep_teacher_templates`` is a teacher sweep.
    """
    roles = {script_role(s.partition("*")[0]) for s in scripts}
    return next((r for r in ROLES if r in roles), "other")


def param_tags(path: str) -> tuple[str, ...]:
    """``--only-param-tags`` entries (JSON ``param_tags`` list or one tag per line)."""
    return _param_tags(path, _file_sig(path))


@lru_cache(maxsize=None)
def _param_tags(path: str, _stamp: str) -> tuple[str, ...]:
    try:
        text = _resolve_path(path).read_text()
    except OSError:
        return ()
    try:
        doc = json.loads(text)
    except ValueError:
        return tuple(ln.strip() for ln in text.splitlines() if ln.strip())
    if isinstance(doc, dict):
        doc = doc.get("param_tags")
    return tuple(t for t in doc if isinstance(t, str)) if isinstance(doc, list) else ()


@dataclass(frozen=True)
class SweepInfo:
    """What a teacher/numbers sweep YAML says about the cells one job of it covers."""

    arch: str | None = None
    epochs: str | None = None  # the grid's epoch count (the dominant one when it mixes)
    rank: int | None = None  # the grid's LoRA rank (dominant when it mixes)
    cells: int = 1  # teacher cells the FULL grid expands to (before sharding)
    n_prompts: int | None = None  # prompts per teacher, when the sweep names a number config
    outer: int = 1  # animals x base models x ft-seeds (the non-HP axes)
    n_seeds: int = 1  # ft-seeds in the grid (``--ftseeds`` narrows it)
    n_ranks: int = 0  # distinct LoRA ranks the grid sweeps (>1 = the grid owns the rank)
    n_eps: int = 0  # distinct epoch counts the grid sweeps (>1 = the grid owns the epochs)


def _dominant(vals: list) -> object | None:
    """The value a mixed axis is keyed on: the STRICTLY most common one, else ``None``.

    A grid that sweeps nine ranks equally has no dominant rank, so its key stays unrefined
    rather than claiming a rank the cells do not share.
    """
    counts = {v: vals.count(v) for v in vals}
    if not counts:
        return None
    top = max(counts.values())
    winners = [v for v, c in counts.items() if c == top]
    return winners[0] if len(winners) == 1 else None


def sweep_info(path: str) -> SweepInfo:
    """Base model, epochs, rank, cell count and prompt count of a sweep YAML.

    The cell count is the Cartesian product `sweep_teacher_templates` expands -- animals x
    base models x epochs x learning rates x (r, alpha) x ft-seeds -- using the same
    `teachers.sweep_grid.lora_pairs_from_grid` and `models.datadecide.parse_base_model_tokens`
    the sweeps use, so a grid change cannot silently desync the prior. An `animals:` keyword
    the YAML does not spell out (``selected``, ``top10``) counts as ``n_animals``. Number
    sweeps inherit their grid (``inherit_grid``) and name their own ``number_config``.
    """
    return _sweep_info(path, _file_sig(path))


@lru_cache(maxsize=None)
def _sweep_info(path: str, _stamp: str) -> SweepInfo:
    from subliminal_learning.models.datadecide import parse_base_model_tokens
    from subliminal_learning.teachers.sweep_grid import lora_pairs_from_grid

    d = _read_yaml(path)
    inherited = _read_yaml(str(d["inherit_grid"])) if d.get("inherit_grid") else {}

    def pick(key, default=None):
        return d.get(key, inherited.get(key, default))

    base = pick("base_model") or pick("base_models") or ""
    arch = model_arch(re.split(r"[,|]", str(base))[0]) if base else None
    try:
        n_bases = len(parse_base_model_tokens(str(base), allow_full_seeds=True))
    except Exception:  # noqa: BLE001 - any unparseable/unknown token spelling
        n_bases = 1
    animals = pick("animals")
    n_animals = len(animals) if isinstance(animals, list) else int(pick("n_animals", 5))
    n_seeds = len(pick("ft_seeds") or [0])
    grid = dict(inherited.get("grid") or {})
    grid.update(d.get("grid") or {})
    eps = [int(e) for e in (grid.get("n_epochs") or [])]
    lrs = grid.get("learning_rate") or []
    try:
        pairs = lora_pairs_from_grid(grid)
    except (KeyError, ValueError):
        pairs = []
    outer = n_animals * n_bases * max(n_seeds, 1)
    cells = outer * max(len(eps), 1) * max(len(lrs), 1) * max(len(pairs), 1)
    ncfg = pick("number_config")
    prompts = None
    if ncfg:
        ps = _read_yaml(str(ncfg)).get("prompt_set")
        prompts = int(ps.get("n_prompts", DEFAULT_N_PROMPTS)) if isinstance(ps, dict) else None
    dom_ep = _dominant(eps)
    return SweepInfo(arch=arch, epochs=str(dom_ep) if dom_ep is not None else None,
                     rank=_dominant([r for r, _a in pairs]) if pairs else None,
                     cells=cells, n_prompts=prompts, outer=outer, n_seeds=max(n_seeds, 1),
                     n_ranks=len({r for r, _a in pairs}), n_eps=len(set(eps)))


def ckpt_distill_student_cfg(path: str | None) -> str | None:
    """A cross-checkpoint sweep YAML's ``student.training_config`` (the STUDENT recipe)."""
    if not path:
        return None
    student = _read_yaml(path).get("student")
    cfg = student.get("training_config") if isinstance(student, dict) else None
    return str(cfg) if cfg else None


@lru_cache(maxsize=None)
def registry_units(path: str, tag: str | None) -> tuple[str | None, int | None]:
    """(arch, runs) of a trajectory ``--registry``: its ``cells[tag]`` seed list length.

    The arch is read from the first path component that parses as a base spelling.
    """
    try:
        doc = json.loads(_resolve_path(path).read_text())
    except (OSError, ValueError):
        return None, None
    arch = next((a for part in re.split(r"[/_.]", path) if (a := model_arch(part))), None)
    cells = doc.get("cells") if isinstance(doc, dict) else None
    if not isinstance(cells, dict):
        return arch, None
    entry = cells.get(tag) if tag else None
    n = len(entry) if isinstance(entry, list) else (None if tag else len(cells))
    return arch, n


@lru_cache(maxsize=None)
def tf32_since(path: str) -> str | None:
    """ISO date of the commit that introduced ``tf32: true`` in ``path`` (``git log -S``).

    Runs BEFORE it used fp32 even though the file reads tf32 today; ``None`` when git cannot
    answer (not a repo, path untracked), in which case the caller falls back to fp32.
    """
    try:
        out = subprocess.run(
            ["git", "log", "-S", "tf32: true", "--format=%cI", "--", path],
            capture_output=True, text=True, timeout=20, check=False, cwd=REPO_ROOT)
    except (OSError, subprocess.SubprocessError):
        return None
    lines = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    return lines[-1] if lines else None  # oldest = the commit that introduced the line


@dataclass(frozen=True)
class Feats:
    """Everything of a job command that predicts its duration."""

    seq: str  # ordered script/module sequence, repeats as ``X*N``
    receiver: str | None = None  # arch the job trains/evaluates (unknown -> None)
    epochs: str | None = None  # LoRA epochs of a training / teacher job
    units: int | None = None  # work items: runs, teacher cells, 1k-prompt blocks, checkpoints
    rank: int | None = None  # exact LoRA rank
    prec: str = "fp32"  # fp32 | tf32
    train_cfg: str | None = None  # training YAML that set tf32 (to re-date old samples)
    role: str = "other"  # one of `ROLES`: what this job DOES
    kind: str | None = None  # eval set / trajectory script: the second key component

    @property
    def trains(self) -> bool:
        return bool(self.role == "train" and self.epochs and self.receiver)

    @property
    def train_key(self) -> str | None:
        """The role's primary key: ``train|model|ep``, ``teacher|model|ep``, ``numgen|model``,
        ``eval|<set>|model``, ``traj|<script>|model``; ``None`` without a known receiver."""
        if self.receiver is None or self.role == "other":
            return None
        if self.role == "train":
            if not self.epochs:
                return None
            kind = f"|{self.kind}" if self.kind else ""
            return f"train{kind}|model={self.receiver}|ep={self.epochs}"
        if self.role == "teacher":  # a grid that mixes epochs keys on the arch alone
            ep = f"|ep={self.epochs}" if self.epochs else ""
            return f"teacher|model={self.receiver}{ep}"
        kind = f"|{self.kind}" if self.kind else ""
        return f"{self.role}{kind}|model={self.receiver}"

    def key(self, level: int) -> str | None:
        """Signature key at ``level`` (4 = most specific, 0 = script sequence only).

        Training / teacher jobs: 4 ``…|ep|r|prec``, 3 ``…|ep|r``, 2 ``…|ep|prec``, 1
        ``…|ep``. Number generation, evals and trajectories stop at level 1 (their prior is
        per UNIT: 1k prompts, one run, one checkpoint). A job with no known receiver keeps
        ``seq|model`` / the script chain.
        """
        if level == 0:
            return self.seq
        base = self.train_key
        if base is None:
            if level == 1 and self.receiver:
                return f"{self.seq}|model={self.receiver}"
            return None
        if level == 1:
            return base
        if self.role not in ("train", "teacher"):
            return None
        if level == 2:
            return f"{base}|{self.prec}"
        if level == 3:
            return f"{base}|r={self.rank}" if self.rank else None
        if level == 4:
            return f"{base}|r={self.rank}|{self.prec}" if self.rank else None
        return None

    def keys(self) -> list[str]:
        """Every key this job pools into / is looked up by, most specific first."""
        out = [self.key(lv) for lv in (4, 3, 2, 1, 0)]
        return list(dict.fromkeys(k for k in out if k))

    def to_token(self) -> str:
        """Compact round-trippable spelling (cache payload)."""
        bits = [self.seq]
        for name, v in (("model", self.receiver), ("ep", self.epochs),
                        ("units", self.units), ("r", self.rank),
                        ("cfg", self.train_cfg),
                        ("role", self.role if self.role != "other" else None),
                        ("kind", self.kind)):
            if v:
                bits.append(f"{name}={v}")
        if self.prec != "fp32":
            bits.append(f"prec={self.prec}")
        return "|".join(bits)

    @staticmethod
    def from_token(token: str) -> "Feats":
        bits = token.split("|")
        d = {}
        for b in bits[1:]:
            k, _, v = b.partition("=")
            d[k] = v
        return Feats(
            seq=bits[0], receiver=d.get("model") or None, epochs=d.get("ep") or None,
            units=int(d["units"]) if d.get("units") else None,
            rank=int(d["r"]) if d.get("r") else None, prec=d.get("prec") or "fp32",
            train_cfg=d.get("cfg") or None, role=d.get("role") or "other",
            kind=d.get("kind") or None,
        )


def key_role(key: str | None) -> str:
    """Role a signature key belongs to (`ROLES`); a script chain is ``other``."""
    role = (key or "").split("|", 1)[0]
    return role if role in ROLES and ("|model=" in (key or "")) else "other"


def key_label(key: str | None) -> str:
    """Compact spelling of a signature for the ``t/job`` cell: ``160m.ep20.r128.tf32``.

    Teacher / numgen / eval / trajectory keys keep their role prefix (``tft·qwen·ep10``,
    ``ngen·olmo-1b``, ``eval·probe·160m``); a script-chain key reads ``p`` (the prior,
    unqualified).
    """
    u = _UNIT_KEY.match(key or "")
    if u:
        bits = [{"teacher": "tft", "numgen": "ngen", "eval": "eval",
                 "traj": "traj"}[u.group("role")]]
        if u.group("kind"):
            bits.append(u.group("kind").removeprefix("scripts."))
        bits.append(u.group("arch").removeprefix("pythia-"))
        return "·".join(bits)
    m = _TRAIN_KEY.match(key or "")
    if not m:
        return "p"
    bits = [m.group("arch").removeprefix("pythia-"), f"ep{m.group('ep')}"]
    if m.group("role") == "teacher":
        bits.insert(0, "tft")
    if m.group("kind"):
        bits.insert(0, m.group("kind"))
    if m.group("rank"):
        bits.append(f"r{m.group('rank')}")
    if m.group("prec"):
        bits.append(m.group("prec"))
    return "·".join(bits)


def _parse_cmd(cmd: str) -> tuple[list[str], dict[str, str]]:
    """(script sequence, first value of every ``--flag``) of a shell command line.

    A flag with no value (``--force``) maps to ``""``; a repeated flag keeps its first value,
    as the scripts themselves do.
    """
    scripts: list[str] = []
    flags: dict[str, str] = {}
    for seg in cmd.split("&&"):
        try:
            toks = shlex.split(seg)
        except ValueError:
            toks = seg.split()
        for i, t in enumerate(toks):
            nxt = toks[i + 1] if i + 1 < len(toks) else None
            if t == "-m" and nxt:
                sub = toks[i + 2] if i + 2 < len(toks) and not toks[i + 2].startswith("-") else ""
                scripts.append(f"{nxt} {sub}".strip())
            elif t.endswith(".py") and not t.startswith("-"):
                sub = f" {nxt}" if nxt and not nxt.startswith("-") else ""
                scripts.append(Path(t).stem + sub)
            elif t.startswith("--") and t not in flags:
                flags[t] = nxt if nxt and not nxt.startswith("-") else ""
    return scripts, flags


def _first(flags: dict[str, str], names: tuple[str, ...]) -> str | None:
    return next((flags[n] for n in names if flags.get(n)), None)


def _int(text: str | None) -> int | None:
    try:
        return int(text)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _sharded(cells: int, flags: dict[str, str]) -> int:
    """``cells`` narrowed to this job's ``--num-shards/--shard-index`` slice.

    Uses the sweeps' own `teachers.sweep_grid.shard_cells`, so a change to the partition
    rule cannot desync the unit count from the work actually done.
    """
    from subliminal_learning.teachers.sweep_grid import shard_cells

    n, i = _int(flags.get("--num-shards")) or 1, _int(flags.get("--shard-index")) or 0
    try:
        return max(len(shard_cells(range(cells), num_shards=n, shard_index=i)), 1)
    except ValueError:
        return max(cells, 1)


def _teacher_cells(flags: dict[str, str], si: SweepInfo) -> tuple[int, str | None, int | None]:
    """(cells this job trains/evaluates, epochs, rank) of a teacher sweep job.

    ``--only-param-tags`` replaces the HP axes of the grid by the listed tags (applied BEFORE
    sharding, as the sweep does), and the tags then carry the epochs and rank.
    """
    epochs, rank = si.epochs, si.rank
    tags = param_tags(flags["--only-param-tags"]) if flags.get("--only-param-tags") else ()
    cells = si.cells
    if tags:
        cells = si.outer * len(tags)
        eps = [m.group(1) for t in tags if (m := _EPOCHS_IN_TAG.search(t))]
        ranks = [int(m.group(1)) for t in tags if (m := _RANK_IN_TAG.search(t))]
        epochs = _dominant(eps) or epochs  # a tag list mixing epochs keys on the dominant one
        rank = _dominant(ranks) or rank
    return _sharded(cells, flags), epochs, rank


def _n_prompts(flags: dict[str, str], si: SweepInfo, tag: str | None) -> int:
    """Prompts one teacher's pool is generated with: the flag, the dry-run size, the sweep's
    ``number_config``, an ``n100k`` dataset tag, else the shared 100k default."""
    n = _int(flags.get("--n-prompts"))
    if n:
        return n
    if "--dry-run" in flags:
        return _int(flags.get("--dry-run-samples")) or 1000
    if si.n_prompts:
        return si.n_prompts
    m = _N_PROMPTS_IN_TAG.search(tag or "")
    return int(m.group(1)) * 1000 if m else DEFAULT_N_PROMPTS


def cmd_ref_sig(cmd: str) -> str:
    """Signature of the files a command's features are read FROM (path:mtime, ordered).

    The parse is only a pure function of the command string for a FIXED set of YAMLs; an
    edited training config or ``--only-param-tags`` list changes the answer, so both caches
    (`job_feats` and `FeatsCache`) carry this beside the command.
    """
    _scripts, flags = _parse_cmd(cmd)
    return "|".join(f"{flag}:{_file_sig(flags[flag])}"
                    for flag in _REF_PATH_FLAGS if flags.get(flag))


def job_feats(cmd: str) -> Feats:
    """Features of ``cmd``, re-parsed when a YAML it points at changed (`cmd_ref_sig`)."""
    return _job_feats(cmd, cmd_ref_sig(cmd))


@lru_cache(maxsize=None)
def _job_feats(cmd: str, _refs: str) -> Feats:
    """Parse a job ``cmd`` into its duration-predicting features.

    The sequence is the ``&&``-ordered ``-m scripts.X [sub]`` / ``<dir>/X.py [sub]`` list with
    consecutive repeats encoded ``X*N``, and the scripts also give the job's ROLE
    (`chain_role`), which decides what its cost is keyed and scaled by:

    - ``train`` -- receiver arch x epochs (x rank x precision). The receiver is
      ``--student-base``/``--base``/``--base-models`` (never ``--teacher-base``), or the
      external family of ``hf_worker.py distill --family`` normalised by `model_arch`;
      epochs and rank come from ``--epochs``/``--tag``/the training YAML. Precision is that
      YAML's ``train.tf32`` (the family's ``lora_student_distill_<family>.yaml`` for the HF
      worker, the fp32 ``lora_student_distill_olmo1b.yaml`` unless ``dd1b_worker`` was given
      a ``_tf32`` variant), overridden by an explicit ``_tf32``/``_fp32`` marker in the
      command.
    - ``teacher`` -- arch x epochs (x rank x precision) of the sweep YAML's grid, per CELL:
      units = the cells this shard trains (`_teacher_cells`).
    - ``numgen`` -- teacher arch, per 1,000 PROMPTS: units = teachers x n_prompts/1000.
    - ``eval`` -- eval set (``--evals``, prompt sets) x probed arch, per RUN.
    - ``traj`` -- trajectory script x receiver arch, per CHECKPOINT run (``--registry``
      entries for this ``--tag``, a ``--runs`` list, else ``--ckpts``).

    Everything else keeps the script chain (+ receiver) it always had.
    """
    scripts, flags = _parse_cmd(cmd)
    role = chain_role(scripts)
    receiver = None
    for name in _RECEIVER_FLAGS:
        if flags.get(name):
            receiver = model_arch(re.split(r"[,|@]", flags[name])[0])
            break
    epochs = flags.get("--epochs") or None
    rank = _int(_first(flags, _RANK_FLAGS))
    tag = _first(flags, _TAG_FLAGS)
    train_cfg = _first(flags, _TRAIN_CFG_FLAGS)
    units: int | None = None
    kind: str | None = None
    family = flags.get("--family")
    prec_only = False  # a recipe the command did not name: its epochs/rank are defaults only
    if family and family in _HF_FAMILY_BASE:  # external-family worker: the family IS the base
        receiver = receiver or model_arch(_HF_FAMILY_BASE[family])
        prec_only = train_cfg is None
        train_cfg = train_cfg or _HF_STUDENT_CFG.format(family=family)
    runs = probe_list_info(flags["--runs"]) if flags.get("--runs") else None
    if runs is not None:  # a readable run list gives the probed arch and the unit count
        receiver, units = receiver or runs[0], runs[1]
    si = sweep_info(flags["--sweep-config"]) if flags.get("--sweep-config") else SweepInfo()
    if flags.get("--sweep-config"):
        receiver = receiver or si.arch
    if role == "teacher":
        units, epochs, rank = _teacher_cells(flags, si)
        epochs, rank = epochs or si.epochs, rank or si.rank
    elif role == "numgen":
        teachers, tep, trank = (_teacher_cells(flags, si) if flags.get("--sweep-config")
                                else (1, None, None))
        if flags.get("--ftseeds"):
            n = len([s for s in flags["--ftseeds"].split(",") if s.strip()])
            teachers = max(teachers * n // max(si.n_seeds, 1), 1)
        units = max(round(teachers * _n_prompts(flags, si, tag) / 1000), 1)
        epochs = rank = None  # a pool's cost is prompts x teacher, not the teacher's recipe
    elif role == "eval":
        kind = flags.get("--evals") or None
        if flags.get("--sweep-config"):  # teacher-preference sweep: one unit per teacher cell
            kind, units = kind or "teacher", _teacher_cells(flags, si)[0]
        elif runs is not None:
            kind = kind or "runs"
        elif any(script_role(x) == "eval" and "preference" in x for x in scripts):
            kind = kind or "pref"  # student preference eval: one unit per prompt set
            units = len([x for x in (flags.get("--prompt-sets") or "test,off_topic").split(",")
                         if x.strip()])
        if units:
            units *= sum(1 for x in scripts if script_role(x) == "eval")
        epochs = rank = None  # evals do not train; `--epochs` only names the students scored
    elif role == "traj":
        kind = next((s for s in scripts if script_role(s) == "traj"), "").removeprefix("scripts.")
        if flags.get("--registry"):
            arch, n = registry_units(flags["--registry"], tag)
            receiver, units = receiver or arch, n or units
        units = units or _int(flags.get("--ckpts"))
        epochs = rank = None
    ckpt_sub = next((x.partition(" ")[2] for x in scripts if x.startswith(_CKPT_DISTILL)
                     and x.partition(" ")[2] in CKPT_DISTILL_SUBS), None)
    if role == "train" and ckpt_sub:
        # Skip-gated cells: their own population (see `_TRAIN_SCRIPTS`), split by subcommand
        # so a teacher cell and a student cell never share a signature.
        kind = f"{CKPT_DISTILL_KIND}-{ckpt_sub}"
        if ckpt_sub == "run-cell":
            # `--param-tag` names this cell's TEACHER; the student's own rank/precision come
            # from the sweep YAML's `student.training_config`.
            tag = None
            train_cfg = ckpt_distill_student_cfg(flags.get("--sweep-config")) or train_cfg
    prec = "fp32"
    if train_cfg:
        yrank, yep, tf32 = training_config_info(train_cfg)
        # A sweep's grid OWNS the rank: the shared template YAML's `lora.r` is overridden per
        # cell, so a grid that sweeps ranks with no dominant one stays rank-less.
        if rank is None and not prec_only and not (role == "teacher" and si.n_ranks > 1):
            rank = yrank
        if epochs is None and not prec_only and not (role == "teacher" and si.n_eps > 1):
            epochs = yep
        prec = "tf32" if tf32 else "fp32"
    if tag:  # `--tag ep10_lr1e-3_r32a8`: the worker's own param tag carries both
        if rank is None:
            m = _RANK_IN_TAG.search(tag)
            rank = int(m.group(1)) if m else None
        if epochs is None:
            m = _EPOCHS_IN_TAG.search(tag)
            epochs = m.group(1) if m else None
    explicit = "tf32" if "_tf32" in cmd else "fp32" if "_fp32" in cmd else None
    if explicit:  # the command names the precision itself: no YAML, no re-dating
        prec, train_cfg = explicit, None
    runs: list[str] = []  # run-length encode repeats: `evaluate_student_preference*10`
    for i, x in enumerate(scripts):
        if i == 0 or x != scripts[i - 1]:
            n = 1
            while i + n < len(scripts) and scripts[i + n] == x:
                n += 1
            runs.append(x if n == 1 else f"{x}*{n}")
    seq = ">".join(runs) or (cmd.split()[0] if cmd.split() else "")
    if role == "train" and epochs is None:  # a trainer we cannot key: script chain only
        role = "other"
    return Feats(seq=seq, receiver=receiver, epochs=epochs, units=units, rank=rank,
                 prec=prec, train_cfg=train_cfg if prec == "tf32" else None,
                 role=role, kind=kind)


def feats_as_of(feats: Feats, end_utc: str | None) -> Feats:
    """``feats`` as they were when a sample ENDED: tf32 only after the YAML line landed."""
    if feats.prec != "tf32":
        return feats
    since = tf32_since(feats.train_cfg) if feats.train_cfg else None
    if end_utc and since and end_utc >= since:
        return feats
    return replace(feats, prec="fp32")


def job_kind(cmd: str) -> tuple[str, int | None]:
    """(primary signature key, units) -- ``train|model|ep`` for a training job, else the
    script chain + receiver."""
    f = job_feats(cmd)
    return f.key(1) or f.key(0) or "", f.units


# --------------------------- pooling ---------------------------


def _pct(vals: list[float], p: float) -> float:
    srt = sorted(vals)
    return srt[min(len(srt) - 1, int(p * len(srt)))]


def filter_resume_modes(
    vals: list[float], *, min_job_s: float = MIN_JOB_S, cache_hit_frac: float = CACHE_HIT_FRAC,
    resume_frac: float = RESUME_FRAC,
) -> list[float]:
    """Drop cache-hit / resumed-from-checkpoint samples of a pool.

    Cutoff = ``max(cache-hit cutoff, resume_frac * p90)``, where the cache-hit cutoff is the
    old rule (below ``min_job_s`` AND below ``cache_hit_frac * p90``). A pool whose jobs are
    genuinely short keeps them (its p90 is small too); a bimodal pool keeps only its real
    mode.
    """
    if not vals:
        return []
    p90 = _pct(vals, 0.9)
    cut = max(min(min_job_s, cache_hit_frac * p90), resume_frac * p90)
    return [v for v in vals if v >= cut]


@dataclass
class Sample:
    """One successful result, keyed by its command's features and tagged with its regime."""

    queue: str
    feats: Feats
    seconds: float
    node: str | None = None
    end_utc: str | None = None
    start_utc: str | None = None
    gpu: int | None = None
    slots: int = 1
    peak_mem_mib: int | None = None  # torch reserved high-water mark, when the job printed one
    others_mean: float = 0.0  # time-weighted count of other jobs on this (node, gpu)
    mps: bool = False

    @property
    def bucket(self) -> int:
        """Co-tenancy bucket: ``round(1 + others_mean)``, capped at 6 (= "6+")."""
        return min(max(int(round(1.0 + self.others_mean)), 1), 6)

    @property
    def tier(self) -> int:
        """1 = today's nodes under MPS, 2 = any node under MPS, 3 = anything ever."""
        if self.start_utc and self.start_utc >= MPS_SINCE:
            return 1 if self.node in TIER_NODES else 2
        return 3


@dataclass(frozen=True)
class Stat:
    """Filtered statistics of one signature pool."""

    key: str
    level: int
    n: int
    p10: float
    p50: float
    p90: float
    per_run_p50: float | None = None
    tier: str = "t3"  # which sample tier the quote came from
    n_tier: tuple[int, int, int] = (0, 0, 0)  # samples available in t1 / t2 / t3
    bucket_p50: dict[int, float] = field(default_factory=dict)
    mps_off_p50: float | None = None
    mps_on_p50: float | None = None
    peak_mem_p50: float | None = None
    peak_mem_p90: float | None = None
    n_peak: int = 0


# --------------------------- the fitted cost model ---------------------------

MODEL_MIN_N = 10  # samples a per-epoch / rank / load cell needs before the model uses it
MODEL_BASE_RANK = 32  # ranks at or below this are the rank-factor BASELINE (rank is flat there)
MODEL_LOADED = (3, 4, 5, 6)  # the pooled "loaded" regime the per-epoch cost is calibrated on
MODEL_DEFAULT_BUCKET = 5  # the load assumed for a node whose density we cannot see
MODEL_LOAD_MIN_N = 1000  # tier-1 samples a receiver needs before its load factors are APPLIED
MODEL_LOAD_TOL = 0.95  # a factor may dip this far below its lower-load neighbour and still pass


@dataclass(frozen=True)
class ArchModel:
    """A student receiver's cost model: ``epochs x per-epoch x rank factor x load factor``.

    ``per_epoch_s`` is the cost of ONE epoch in the pooled LOADED regime (`MODEL_LOADED` =
    3-6 jobs per GPU, the one campaigns actually run in), and every factor is relative to that
    same regime, so the model is a product of one rate and three dimensionless factors rather
    than a row per measured (epochs, rank) cell. Both factors are fitted WITHIN an epoch group
    and combined across groups by an n-weighted median, never pooled across epochs.
    ``res_p10`` / ``res_p90`` are the measured/predicted band against its own samples.

    ``load_applied`` says whether the load factors are TRUSTED enough to price with
    (`load_factors_trustworthy`); on a thin receiver the co-tenancy bucket is a proxy for
    campaign phase rather than GPU contention, so the factors are still reported but every
    `load_mult` is 1.0.
    """

    arch: str
    prec: str
    per_epoch_s: float
    n: int
    kind: str | None = None  # the training KIND fitted (None = plain students; e.g. ckptdistill)
    epochs: tuple[int, ...] = ()
    rank_factor: dict[int, float] = field(default_factory=dict)
    load_factor: dict[int, float] = field(default_factory=dict)
    res_p10: float = 1.0
    res_p90: float = 1.0
    load_applied: bool = True  # False = measured but not priced with (see the class docstring)

    def rank_mult(self, rank: int | None) -> float:
        """Rank factor of ``rank``: the measured one, else the nearest measured rank in
        log2 (ranks below `MODEL_BASE_RANK` are flat, so an unseen small rank lands on 1.0)."""
        if rank is None or not self.rank_factor:
            return 1.0
        if rank in self.rank_factor:
            return self.rank_factor[rank]
        near = min(self.rank_factor, key=lambda r: abs(math.log2(r) - math.log2(rank)))
        return self.rank_factor[near]

    def load_mult(self, bucket: int | None) -> float:
        """Load factor at ``bucket`` jobs on the GPU: the measured one, else the nearest
        measured bucket, else 1.0 (the reference load, and what an untrusted receiver uses)."""
        b = MODEL_DEFAULT_BUCKET if bucket is None else min(max(int(bucket), 1), 6)
        if not self.load_factor or not self.load_applied:
            return 1.0
        if b in self.load_factor:
            return self.load_factor[b]
        near = min(self.load_factor, key=lambda x: (abs(x - b), x))
        return self.load_factor[near]

    def predict(self, epochs: int, rank: int | None, bucket: int | None) -> float:
        return self.per_epoch_s * epochs * self.rank_mult(rank) * self.load_mult(bucket)


@dataclass(frozen=True)
class UnitRate:
    """Per-unit cost of a non-training role (one cell / 1k prompts / run / checkpoint)."""

    key: str  # the role's primary key, e.g. ``teacher|model=qwen|ep=20``
    n: int
    p10: float
    p50: float
    p90: float


@dataclass(frozen=True)
class DurationModel:
    """The reasoned duration model: student cost models + per-unit rates for the other roles."""

    students: dict[tuple[str, str, str], ArchModel] = field(default_factory=dict)
    rates: dict[str, UnitRate] = field(default_factory=dict)

    def for_arch(self, arch: str | None, prec: str, kind: str | None = None) -> ArchModel | None:
        """This receiver's model for this training KIND at ``prec``.

        A kind is never fitted or quoted from another kind's samples (a skip-gated
        ``ckptdistill`` cell is a different population), so a query whose kind was never
        measured gets NOTHING and falls through to the unknown rule; only the PRECISION falls
        back, to that kind's best-sampled one.
        """
        if arch is None:
            return None
        m = self.students.get((arch, prec, kind or ""))
        if m is not None:
            return m
        cands = [v for (a, _p, k), v in self.students.items() if (a, k) == (arch, kind or "")]
        return max(cands, key=lambda v: v.n) if cands else None


def load_factors_trustworthy(n: int, factors: dict[int, float]) -> bool:
    """Whether a receiver's load factors measure GPU CONTENTION rather than campaign phase.

    Two tests: at least `MODEL_LOAD_MIN_N` tier-1 samples behind the fit, and factors that do
    not FALL as the GPU fills (each measured bucket >= `MODEL_LOAD_TOL` x its lower-load
    neighbour). A 15-200-job campaign fails them because its co-tenancy bucket tracks the phase
    of the campaign, not the load: inside the OLMo queues later-started jobs are both faster and
    more co-tenant (corr(duration, start order) -0.3..-0.87, corr(bucket, start order) up to
    +0.83, pooled corr(duration, bucket) -0.30), which would otherwise be read as "5-up is
    faster than 3-up".
    """
    if n < MODEL_LOAD_MIN_N or not factors:
        return False
    vals = [f for _d, f in sorted(factors.items())]
    return all(b >= a * MODEL_LOAD_TOL for a, b in zip(vals, vals[1:]))


def is_cache_hit(s: "Sample") -> bool:
    """Whether a sample is a skip-gated TRAINING run (a cache hit), not a training.

    A training job that returns in under `MIN_JOB_S` did not train: it hit its own skip gate
    (the outputs were already on disk), which is the same rule the status table uses to drop
    cache hits. Left in, a fully cached grid quotes 10 s for every real job of its shape and
    drags the epoch-scaling fit with it.
    """
    return s.feats.role == "train" and s.seconds < MIN_JOB_S


def _kept(rows: list["Sample"]) -> list["Sample"]:
    """``rows`` with the resume / cache-hit mode dropped (same rule the pools use)."""
    cut_all = filter_resume_modes([x.seconds for x in rows])
    if not cut_all:
        return []
    cut = min(cut_all)  # hoisted: re-deriving it per row is O(n^2) on a 66k-sample pool
    return [x for x in rows if x.seconds >= cut]


def _wmedian(pairs: list[tuple[float, int]]) -> float | None:
    """Median of ``(value, weight)`` pairs -- the value at half the total weight.

    Epoch groups differ in size by orders of magnitude (66k ep20 runs vs 200 ep5 ones), so a
    factor measured in each group is combined by WEIGHT, not one vote per group.
    """
    if not pairs:
        return None
    srt = sorted(pairs)
    half = sum(w for _v, w in srt) / 2.0
    acc = 0.0
    for v, w in srt:
        acc += w
        if acc >= half:
            return v
    return srt[-1][0]


def build_model(samples: list["Sample"], *, min_cell_n: int = MODEL_MIN_N) -> DurationModel:
    """Fit the cost model from the tier-1 samples (today's nodes under MPS).

    Students, per (arch, precision, training KIND -- a ``ckptdistill`` cell never fits or
    quotes a plain student): samples are split by EPOCH COUNT, the resume/cache-hit
    mode is dropped inside each group, and every quantity is then measured WITHIN the group
    and combined across groups by an n-weighted median (`_wmedian`) -- the per-epoch cost from
    ``seconds / epochs`` over the pooled loaded regime (`MODEL_LOADED` jobs per GPU), the rank
    factor of ``r`` from ``p50(r) / p50(rank <= MODEL_BASE_RANK)`` in that regime, the load
    factor of bucket ``d`` from ``p50(d) / p50(loaded)``. Mixing buckets across epoch counts
    would price the load with whatever epoch count happened to run at that load, so it is
    never done. A cell below ``min_cell_n`` samples is absent (factor 1.0 / the nearest
    measured neighbour). The other roles keep a per-UNIT rate per ``(role, arch, kind|epochs)``.
    """
    trains: dict[tuple[str, str, str], dict[int, list[Sample]]] = {}
    rates: dict[str, list[Sample]] = {}
    for s in samples:
        f = s.feats
        if s.tier != 1 or not f.receiver or is_cache_hit(s):
            continue
        if f.role == "train" and (f.epochs or "").isdigit():
            trains.setdefault((f.receiver, f.prec, f.kind or ""), {}).setdefault(
                int(f.epochs), []).append(s)  # type: ignore[arg-type]
        elif f.role in ("teacher", "numgen", "eval", "traj") and f.units and f.train_key:
            rates.setdefault(f.train_key, []).append(s)
    students: dict[tuple[str, str, str], ArchModel] = {}
    for (arch, prec, kind), raw in trains.items():
        # The resume filter runs ONCE per epoch group, before anything is bucketed.
        groups = {ep: k for ep, v in raw.items() if (k := _kept(v))}
        per_ep: list[tuple[float, int]] = []
        ranks: dict[int, list[tuple[float, int]]] = {}
        loads: dict[int, list[tuple[float, int]]] = {}
        for ep, kept in groups.items():
            by_b: dict[int, list[float]] = {}
            loaded: list[Sample] = []
            for x in kept:
                by_b.setdefault(x.bucket, []).append(x.seconds)
                if x.bucket in MODEL_LOADED:
                    loaded.append(x)
            if len(loaded) < min_cell_n:
                continue
            ref = statistics.median(x.seconds for x in loaded)
            per_ep.append((ref / ep, len(loaded)))
            for d, v in by_b.items():
                if len(v) >= min_cell_n:
                    loads.setdefault(d, []).append((statistics.median(v) / ref, len(v)))
            base = [x.seconds for x in loaded
                    if x.feats.rank and x.feats.rank <= MODEL_BASE_RANK]
            if len(base) < min_cell_n:
                continue
            b = statistics.median(base)
            by_rank: dict[int, list[float]] = {}
            for x in loaded:
                if x.feats.rank:
                    by_rank.setdefault(x.feats.rank, []).append(x.seconds)
            for r, v in by_rank.items():
                if len(v) >= min_cell_n:
                    ranks.setdefault(r, []).append((statistics.median(v) / b, len(v)))
        rate = _wmedian(per_ep)
        if rate is None:
            continue
        n = sum(len(v) for v in groups.values())
        load = {d: f for d, v in sorted(loads.items()) if (f := _wmedian(v)) is not None}
        m = ArchModel(
            arch=arch, prec=prec, kind=kind or None, per_epoch_s=rate,
            n=n, epochs=tuple(sorted(groups)),
            rank_factor={r: f for r, v in sorted(ranks.items())
                         if (f := _wmedian(v)) is not None},
            load_factor=load, load_applied=load_factors_trustworthy(n, load),
        )
        res = [x.seconds / p for ep, v in groups.items() for x in v
               if (p := m.predict(ep, x.feats.rank, x.bucket)) > 0]
        if res:
            m = replace(m, res_p10=_pct(res, 0.1), res_p90=_pct(res, 0.9))
        students[(arch, prec, kind)] = m
    unit = {}
    for key, rows in rates.items():
        kept = _kept(rows)
        vals = sorted(x.seconds / x.feats.units for x in kept)  # type: ignore[operator]
        if vals:
            unit[key] = UnitRate(key=key, n=len(vals), p10=_pct(vals, 0.1),
                                 p50=statistics.median(vals), p90=_pct(vals, 0.9))
    return DurationModel(students=students, rates=unit)


@dataclass(frozen=True)
class Prior:
    """A prior duration quoted for one job."""

    seconds: float
    stat: Stat
    extrapolated: bool = False
    composed: bool = False  # summed from the per-script priors of an unmeasured chain
    label: str = "p"  # what the ``t/job`` cell shows
    model: ArchModel | None = None  # set when the FITTED model, not a measured cell, quoted
    bucket: int | None = None  # the GPU load the model quoted at


class Priors:
    """Signature -> measured duration table with the train -> script -> sum fallback chain."""

    def __init__(self, stats: dict[str, Stat], ep_index: dict[str, dict[int, str]],
                 *, min_n: int = DEFAULT_MIN_N, model: DurationModel | None = None) -> None:
        self.stats = stats
        self._ep_index = ep_index
        self.min_n = min_n
        self.model = model or DurationModel()

    def _stat(self, key: str | None, min_n: int | None = None) -> Stat | None:
        s = self.stats.get(key) if key else None
        need = self.min_n if min_n is None else min_n
        return s if s is not None and s.n >= need else None

    @staticmethod
    def _quote(stat: Stat, feats: Feats) -> float:
        if feats.units and stat.per_run_p50:
            return stat.per_run_p50 * feats.units
        return stat.p50

    def _label(self, feats: Feats, stat: Stat, level: int) -> str:
        """Name the SHALLOWEST level that holds the same samples: a ``|fp32`` or ``|r=N``
        refinement whose pool is identical to its parent's adds nothing to the cell."""
        for lv in range(1, level):
            other = self.stats.get(feats.key(lv))
            if other is not None and (other.n, other.p50) == (stat.n, stat.p50):
                return key_label(other.key)
        return key_label(stat.key)

    def lookup(self, feats: Feats, *, density: float | None = None) -> Prior | None:
        """Most specific usable prior for ``feats``; ``None`` if nothing applies.

        Order: the exact measured cell (rank+precision down to receiver+epochs) -> the fitted
        MODEL (``per-epoch x epochs x rank factor x load factor`` at this node's ``density``,
        jobs per GPU in use) -> another epoch count scaled -> the script chain -> the sum of
        the chain's components.
        """
        for level in (4, 3, 2, 1):
            s = self._stat(feats.key(level))
            if s is not None:
                return Prior(self._quote(s, feats), s, label=self._label(feats, s, level))
        modelled = self._modelled(feats, density)
        if modelled is not None:
            return modelled
        scaled = self._epoch_scaled(feats)
        if scaled is not None:
            return scaled
        s = self._stat(feats.key(0))
        if s is not None:
            return Prior(self._quote(s, feats), s, label=key_label(s.key))
        return self.compose(feats)

    def _modelled(self, feats: Feats, density: float | None) -> Prior | None:
        """The fitted model's quote for an (epochs, rank) combination never measured itself.

        ``160m·ep7·r16·model`` = 2.4 min/epoch x 7 epochs x the r16 factor x the load factor
        at this node's density (jobs per GPU in use, capped 1..6, `MODEL_DEFAULT_BUCKET` when
        unknown). The synthetic `Stat` carries the model's residual band as p10/p90.
        """
        if feats.role != "train" or not feats.receiver:
            return None
        try:
            ep = int(feats.epochs)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        m = self.model.for_arch(feats.receiver, feats.prec, feats.kind)
        if m is None or ep <= 0:
            return None
        bucket = (MODEL_DEFAULT_BUCKET if density is None
                  else min(max(int(round(density)), 1), 6))
        secs = m.predict(ep, feats.rank, bucket)
        bits = [feats.receiver.removeprefix("pythia-"), f"ep{ep}"]
        if feats.rank:
            bits.append(f"r{feats.rank}")
        stat = Stat(key=f"train|model={feats.receiver}|ep={ep}|model", level=1, n=m.n,
                    p10=secs * m.res_p10, p50=secs, p90=secs * m.res_p90, tier="t1")
        return Prior(secs, stat, label="·".join([*bits, "model"]), model=m, bucket=bucket)

    def _epoch_scaled(self, feats: Feats) -> Prior | None:
        """The same receiver at another epoch count, scaled by ``(E+1)/(E_ref+1)``."""
        if feats.role not in ("train", "teacher") or not feats.train_key:
            return None
        try:
            want = int(feats.epochs)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        cands = []
        idx = self._ep_index.get(f"{feats.role}|{feats.kind}|{feats.receiver}") or {}
        for ep, key in idx.items():
            s = self._stat(key)
            if s is not None and ep != want:
                cands.append((abs(ep - want), -s.n, ep, s))
        if not cands:
            return None
        _, _, ep, s = min(cands)
        label = f"{key_label(feats.train_key)}←ep{ep}"
        return Prior(self._quote(s, feats) * (want + 1) / (ep + 1), s, extrapolated=True,
                     label=label)

    def compose(self, feats: Feats) -> Prior | None:
        """Sum of each script's own prior, for a chain never measured as a whole.

        ``a>b>c`` that has never run still has a usable estimate when ``a``, ``b`` and ``c``
        each ran alone (``COMPOSE_MIN_N`` samples each -- a thin measurement still beats the
        blind default); a chain with one unmeasured component has none. A component whose pool has a per-run median is quoted per run x
        this job's ``--runs`` count, as `_quote` does.
        """
        total = p10 = p90 = 0.0
        n = None
        parts = feats.seq.split(">")
        if len(parts) < 2:
            return None
        for part in parts:
            name, _, mult = part.partition("*")
            reps = int(mult) if mult.isdigit() else 1
            s = self._stat(name, COMPOSE_MIN_N)
            if s is None:
                return None
            total += self._quote(s, feats) * reps
            p10 += s.p10 * reps
            p90 += s.p90 * reps
            n = s.n if n is None else min(n, s.n)
        stat = Stat(key=feats.seq, level=0, n=n or 0, p10=p10, p50=total, p90=p90)
        return Prior(total, stat, composed=True, label="p,sum")

    def table(self, *, min_n: int = 0) -> list[Stat]:
        """Learned heuristics, grouped by role (`ROLES`) and most-sampled first inside it:
        student trainings, teacher cells, number generation, evals, trajectories, other.

        A refinement that adds nothing (same samples, same median as the key one level up --
        e.g. ``|fp32`` when every sample of that key was fp32) is dropped from the listing.
        """
        keep = []
        for s in self.stats.values():
            if s.n < max(min_n, self.min_n):
                continue
            parent = self.stats.get(s.key.rsplit("|", 1)[0]) if "|" in s.key else None
            if parent is not None and (parent.n, parent.p50) == (s.n, s.p50):
                continue
            keep.append(s)
        return sorted(keep, key=lambda s: (ROLES.index(key_role(s.key)), -s.n))


def _cell(vals: list[float], min_n: int = MIN_CELL_N) -> float | None:
    """Median of a table cell, or ``None`` when too few samples back it."""
    return statistics.median(vals) if len(vals) >= min_n else None


def build_priors(samples: list[Sample], *, min_n: int = DEFAULT_MIN_N,
                 max_per_queue: int = MAX_PER_QUEUE, recent_n: int = DEFAULT_RECENT_N) -> Priors:
    """Pool ``samples`` into a `Priors` table (one vote per sample, capped per queue).

    Each key quotes the first TIER that holds ``min_n`` samples -- t1 today's nodes under MPS,
    t2 any node under MPS, t3 everything ever measured -- newest first and capped at
    ``recent_n`` samples, so a key drifts with the machines it actually runs on.
    """
    samples = [s for s in samples if not is_cache_hit(s)]
    pools: dict[str, dict[str, list[Sample]]] = {}
    ep_index: dict[str, dict[int, str]] = {}
    for s in samples:
        for key in s.feats.keys():
            pools.setdefault(key, {}).setdefault(s.queue, []).append(s)
        if s.feats.role in ("train", "teacher") and s.feats.train_key:
            try:
                ep = int(s.feats.epochs)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            ep_index.setdefault(f"{s.feats.role}|{s.feats.kind}|{s.feats.receiver}",
                                {})[ep] = s.feats.key(1)
    stats: dict[str, Stat] = {}
    for key, per_q in pools.items():
        # A queue contributes its NEWEST `max_per_queue` results, not the first ones the
        # result files happened to be listed in (that order is lexicographic by jobkey).
        flat = [x for lst in per_q.values()
                for x in (heapq.nlargest(max_per_queue, lst, key=lambda r: r.end_utc or "")
                          if len(lst) > max_per_queue else lst)]
        cut_all = filter_resume_modes([x.seconds for x in flat])
        if not cut_all:  # kept even below `min_n`: thin pools still compose a chain
            continue
        cut = min(cut_all)
        kept = sorted((x for x in flat if x.seconds >= cut),
                      key=lambda x: x.end_utc or "", reverse=True)  # newest first
        tiers = [[x for x in kept if x.tier <= t] for t in (1, 2, 3)]
        n_tier = tuple(len(t) for t in tiers)
        idx = next((i for i, t in enumerate(tiers) if len(t) >= min_n), 2)
        chosen = tiers[idx][:recent_n]
        secs = [x.seconds for x in chosen]
        per_run = [x.seconds / x.feats.units for x in chosen if x.feats.units]
        level = max(lv for lv in (0, 1, 2, 3, 4) if flat[0].feats.key(lv) == key)
        buckets: dict[int, list[float]] = {}
        for x in kept:
            buckets.setdefault(x.bucket, []).append(x.seconds)
        peaks = [float(x.peak_mem_mib) for x in kept if x.peak_mem_mib]
        stats[key] = Stat(
            key=key, level=level, n=len(chosen), p10=_pct(secs, 0.1),
            p50=statistics.median(secs), p90=_pct(secs, 0.9),
            per_run_p50=statistics.median(per_run) if per_run else None,
            tier=f"t{idx + 1}", n_tier=n_tier,  # type: ignore[arg-type]
            bucket_p50={b: m for b, v in sorted(buckets.items())
                        if (m := _cell(v)) is not None},
            mps_off_p50=_cell([x.seconds for x in kept if not x.mps]),
            mps_on_p50=_cell([x.seconds for x in kept if x.mps]),
            peak_mem_p50=statistics.median(peaks) if peaks else None,
            peak_mem_p90=_pct(peaks, 0.9) if peaks else None, n_peak=len(peaks),
        )
    return Priors(stats, ep_index, min_n=min_n, model=build_model(samples))


# --------------------------- per-job expectation ---------------------------


@dataclass(frozen=True)
class Expectation:
    """Expected duration of one job and where the number came from."""

    seconds: float
    source: str  # own | blend | prior | prior_model | prior_ep | prior_sum | unknown
    n_own: int = 0
    prior: Prior | None = None

    @property
    def known(self) -> bool:
        return self.source != "unknown"

    @property
    def label(self) -> str:
        return self.prior.label if self.prior else "?"


def expected_duration(
    feats: Feats, own_by_key: dict[str, list[float]], priors: Priors, *,
    unknown_s: float = 600.0, density: float | None = None,
) -> Expectation:
    """Blend this queue's own samples for ``feats`` with the pooled prior.

    ``own_by_key`` maps signature key -> this queue's own durations; the key is read at the
    level the prior matched (most specific first when there is no prior).
    ``w = n_own / (n_own + 10)``; own alone when there is no prior, prior alone below three
    own samples, ``unknown_s`` (flagged) when neither exists.
    """
    prior = priors.lookup(feats, density=density)
    keys = [prior.stat.key] if prior is not None else feats.keys()
    own: list[float] = []
    for key in keys:
        own = filter_resume_modes(own_by_key.get(key) or [])
        if own:
            break
    n_own = len(own)
    if prior is None:
        if n_own >= OWN_MIN_N:
            return Expectation(statistics.median(own), "own", n_own, None)
        return Expectation(unknown_s, "unknown", n_own, None)
    psrc = ("prior_ep" if prior.extrapolated else
            "prior_sum" if prior.composed else
            "prior_model" if prior.model else "prior")
    if n_own < OWN_MIN_N:
        return Expectation(prior.seconds, psrc, n_own, prior)
    w = n_own / (n_own + BLEND_K)
    secs = w * statistics.median(own) + (1 - w) * prior.seconds
    return Expectation(secs, "own" if n_own >= OWN_DOMINANT_N else "blend", n_own, prior)


# --------------------------- regime reconstruction ---------------------------


def _epoch_s(iso: str | None) -> float | None:
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        return None
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).timestamp()


@lru_cache(maxsize=None)
def pool_mps_windows(root: str, host: str) -> tuple[tuple[float, bool], ...]:
    """``(start epoch, MPS on)`` per ``pool_<stamp>.<host>.log`` in ``root``, oldest first.

    A log covers from its stamp until the next log for that host; MPS is ON when its first
    ``MPS`` line says ``MPS enabled`` (a warning line means the daemon was NOT up).
    """
    out: list[tuple[float, bool]] = []
    for p in sorted(Path(root).glob(f"pool_*.{host}.log")):
        stamp = p.name[len("pool_"):-len(f".{host}.log")]
        try:
            t = datetime.strptime(stamp, "%Y%m%dT%H%M").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        on = False
        try:
            with p.open("r", errors="replace") as f:
                for _ in range(20):
                    line = f.readline()
                    if not line:
                        break
                    if "MPS" in line:
                        on = "MPS enabled" in line
                        break
        except OSError:
            continue
        out.append((t.timestamp(), on))
    return tuple(sorted(out))


def sample_mps(root: str, node: str | None, start_s: float | None) -> bool:
    """Whether MPS was on for a job started at ``start_s`` on ``node`` under ``root``.

    The rollout date gates everything; within it the covering pool log decides. A root with
    no pool logs (``run_logs/queue0``) follows the date alone.
    """
    if start_s is None or start_s < (_epoch_s(MPS_SINCE) or 0.0):
        return False
    if not node:
        return True
    wins = pool_mps_windows(root, node)
    if not wins:
        return True
    cover = [on for t, on in wins if t <= start_s]
    return cover[-1] if cover else False


def annotate_regime(samples: list[Sample]) -> None:
    """Fill each sample's co-tenancy and MPS flag in place (own root only).

    One sweep line per ``(node, gpu)`` gives ``others_mean``: the time-weighted number of
    other jobs of this root that held the same GPU while the sample ran. It is a DIAGNOSTIC
    column only -- the prior rule uses the node/MPS tiers. Samples without a recorded GPU
    cannot be placed and stay at zero, and loads this repo never recorded (direct runs,
    foreign users) are invisible.
    """
    groups: dict[tuple[str, str, int], list[tuple[float, float, Sample]]] = {}
    for s in samples:
        a, b = _epoch_s(s.start_utc), _epoch_s(s.end_utc)
        if a is None or b is None or b <= a or s.node is None or s.gpu is None:
            continue
        root = s.queue.rsplit("/", 1)[0]
        groups.setdefault((root, s.node, s.gpu), []).append((a, b, s))
    for items in groups.values():
        edges = sorted({t for a, b, _ in items for t in (a, b)})
        items.sort(key=lambda x: x[0])
        i, active = 0, []
        for t0, t1 in zip(edges, edges[1:]):
            while i < len(items) and items[i][0] <= t0:
                active.append(items[i])
                i += 1
            active = [x for x in active if x[1] > t0]
            if len(active) < 2:
                continue
            span = t1 - t0
            for a, b, s in active:
                s.others_mean += (len(active) - 1) * span / (b - a)
    for s in samples:
        s.mps = sample_mps(s.queue.rsplit("/", 1)[0], s.node, _epoch_s(s.start_utc))


# --------------------------- history cache ---------------------------


def _results_fingerprint(root: Path, name: str) -> tuple[int, int]:
    """(result-file count, ``results/`` dir mtime_ns) of a queue; (0, 0) when absent.

    ONE stat for the whole queue: results are written atomically (temp + rename), and a
    rename into (or over an entry of) ``results/`` moves the DIRECTORY's mtime, so the dir
    alone dates every add, replace and removal. The per-file `DirEntry.stat()` this used to
    do is a syscall each on Linux -- ~113k of them per warm status call across the history
    roots, on CephFS.
    """
    rdir = root / name / "results"
    os = __import__("os")
    try:
        with os.scandir(rdir) as it:
            n = sum(1 for e in it if e.name.endswith(".json"))
        return (n, os.stat(rdir).st_mtime_ns)
    except OSError:
        return (0, 0)


def _scan_queue(root: Path, name: str) -> list[Sample]:
    jobs = {j.key: j for j in store.load_jobs(root, name)}
    out: list[Sample] = []
    for r in store.load_results(root, name):
        if r.get("rc", 1) != 0:
            continue
        job = jobs.get(r.get("key"))
        if job is None:  # result whose job key no longer exists in jobs.jsonl
            continue
        secs = _elapsed_s(r)
        if secs is None:
            continue
        try:
            slots = max(int(job.slots), 0)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            slots = 1
        out.append(Sample(queue=name, feats=feats_as_of(job_feats(job.cmd), r.get("end_utc")),
                          seconds=secs, node=r.get("node"), end_utc=r.get("end_utc"),
                          start_utc=r.get("start_utc"),
                          gpu=r.get("gpu") if isinstance(r.get("gpu"), int) else None,
                          slots=slots,
                          peak_mem_mib=_int_or_none(r.get("peak_mem_mib"))))
    return out


def _int_or_none(v: object) -> int | None:
    try:
        return int(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _elapsed_s(r: dict) -> float | None:
    try:
        a = datetime.fromisoformat(r["start_utc"])
        b = datetime.fromisoformat(r["end_utc"])
    except (KeyError, TypeError, ValueError):
        return None
    return (b - a).total_seconds()


def history_roots(root: Path, extra: list[Path] | None = None, *,
                  include_default: bool = True) -> list[Path]:
    """The status root plus the standard history roots (and ``extra``) that exist."""
    default = [Path(p) for p in HISTORY_ROOTS] if include_default else []
    cands = [Path(root), *default, *(extra or [])]
    out: list[Path] = []
    for p in cands:
        rp = p.resolve() if p.exists() else p
        if p.is_dir() and not any(rp == q.resolve() for q in out):
            out.append(p)
    return out


@dataclass
class CacheInfo:
    """What a `load_samples` call did, for the one status line it prints."""

    n_samples: int = 0
    n_queues: int = 0
    rescanned: int = 0
    built_utc: str | None = None
    age_s: float | None = None
    wrote: bool = True
    note: str | None = None
    roots: list[str] = field(default_factory=list)
    regime_s: float = 0.0


def _scan_root(root: Path, prev: dict, *, info: CacheInfo) -> tuple[dict, list[Sample]]:
    """One root's cached sample index (rescanning only the queues whose results moved)."""
    cur: dict[str, dict] = {}
    samples: list[Sample] = []
    rkey = str(root)
    for name in store.list_queues(root):
        was = prev.get(name)
        # A queue stamped complete is fingerprinted like any other: "complete" is a stamp a
        # human (or a requeue) can outlive, and results appended afterwards were silently
        # invisible. The fingerprint is one stat per queue, so the sweep the old
        # short-circuit avoided is gone rather than skipped.
        if was is not None and was.get("fp") == list(_results_fingerprint(root, name)):
            cur[name] = was
        else:
            fp = list(_results_fingerprint(root, name))
            rows = _scan_queue(root, name)
            sigs: list[str] = []
            idx: dict[str, int] = {}
            recs = []
            for s in rows:
                tok = s.feats.to_token()
                if tok not in idx:
                    idx[tok] = len(sigs)
                    sigs.append(tok)
                recs.append([idx[tok], round(s.seconds, 1), s.node, s.end_utc,
                             s.start_utc, s.gpu, s.slots, s.peak_mem_mib])
            cur[name] = {"fp": fp, "complete": store.queue_complete(root, name),
                         "sigs": sigs, "s": recs}
            info.rescanned += 1
        entry = cur[name]
        feats = [Feats.from_token(t) for t in entry["sigs"]]
        for rec in entry["s"]:
            # Rows written before `peak_mem_mib` (CACHE_VERSION < 6) are one column short.
            si, secs, node, end, start, gpu, slots = rec[:7]
            peak = rec[7] if len(rec) > 7 else None
            samples.append(Sample(queue=f"{rkey}/{name}", feats=feats_as_of(feats[si], end),
                                  seconds=float(secs), node=node, end_utc=end,
                                  start_utc=start, gpu=gpu, slots=slots,
                                  peak_mem_mib=peak))
    return cur, samples


def load_samples(
    root: Path, *, extra_roots: list[Path] | None = None, rebuild: bool = False,
    cache_path: Path | None = None, include_default_history: bool = True,
) -> tuple[list[Sample], CacheInfo]:
    """Historical samples over every history root, incrementally cached in the status root.

    A queue is rescanned only when its results fingerprint changed (result-file count +
    the ``results/`` dir mtime), stamped complete or not. A cache that cannot be written
    (other uid / read-only root) degrades to in-memory with a note.
    """
    cache_path = cache_path or (Path(root) / CACHE_NAME)
    roots = history_roots(root, extra_roots, include_default=include_default_history)
    old: dict = {}
    info = CacheInfo(roots=[str(p) for p in roots])
    if not rebuild:
        try:
            data = json.loads(cache_path.read_text())
            if data.get("version") == CACHE_VERSION:
                old = data.get("roots") or {}
                info.built_utc = data.get("built_utc")
        except (OSError, ValueError):
            old = {}
    new: dict[str, dict] = {}
    samples: list[Sample] = []
    for r in roots:
        cur, rows = _scan_root(r, old.get(str(r)) or {}, info=info)
        new[str(r)] = cur
        samples += rows
        info.n_queues += len(cur)
    t0 = time.time()
    annotate_regime(samples)
    info.regime_s = time.time() - t0
    info.n_samples = len(samples)
    payload = {"version": CACHE_VERSION, "built_utc": store.now_iso(), "roots": new}
    if info.rescanned or not old:
        try:
            atomic_write_json(cache_path, payload, indent=None)
            info.built_utc = payload["built_utc"]
        except OSError as e:
            info.wrote = False
            info.note = f"priors cache not written ({e.__class__.__name__}); in-memory only"
    try:
        info.age_s = max(time.time() - cache_path.stat().st_mtime, 0.0)
    except OSError:
        info.age_s = None
    return samples, info


class FeatsCache:
    """``cmd`` -> `Feats`, memoised on disk beside the priors cache.

    Parsing one command costs ~0.4 ms (shlex + the YAMLs / tag lists / registries it points
    at) and a busy root re-reads ~29k pending + running commands on EVERY status call, so the
    parse is kept in a world-writable ``cmd_feats_cache.json``: a warm call parses only
    commands it has never seen, or whose YAMLs changed since (each entry stores
    `cmd_ref_sig` beside the token). A cache that cannot be read or written degrades to in-process memoisation.
    """

    def __init__(self, path: Path | None,
                 entries: dict[str, list] | None = None) -> None:
        self.path = path
        # cmd -> [referenced-file signature, feats token]: an entry whose YAMLs moved on is
        # re-parsed (`cmd_ref_sig`).
        self.entries = dict(entries or {})
        self._feats: dict[str, Feats] = {}
        self._new = 0

    @classmethod
    def load(cls, root: Path | None) -> "FeatsCache":
        path = (Path(root) / FEATS_CACHE_NAME) if root is not None else None
        try:
            doc = json.loads(path.read_text())  # type: ignore[union-attr]
            entries = doc.get("cmds") if doc.get("version") == CACHE_VERSION else None
        except (AttributeError, OSError, ValueError):
            entries = None
        return cls(path, entries if isinstance(entries, dict) else None)

    def feats(self, cmd: str) -> Feats:
        """Features of ``cmd``, from memory, then the cache file, then the parser."""
        hit = self._feats.get(cmd)
        if hit is not None:
            return hit
        sig = cmd_ref_sig(cmd)
        entry = self.entries.get(cmd)
        token = entry[1] if isinstance(entry, list) and len(entry) == 2 \
            and entry[0] == sig else None
        f = Feats.from_token(token) if token else job_feats(cmd)
        if not token:
            self.entries[cmd] = [sig, f.to_token()]
            self._new += 1
        self._feats[cmd] = f
        return f

    def save(self) -> bool:
        """Persist the table when new commands were parsed; ``False`` when it stayed in
        memory (no root, or a file another uid owns)."""
        if self.path is None or not self._new:
            return False
        try:
            atomic_write_json(self.path, {"version": CACHE_VERSION, "cmds": self.entries},
                              indent=None)
            __import__("os").chmod(self.path, 0o666)  # every node/uid writes this root
        except OSError:
            return False
        self._new = 0
        return True


def queue_samples(root: Path, name: str) -> list[Sample]:
    """This queue's own successful samples (uncached; used for the own/prior blend)."""
    return _scan_queue(Path(root), name)


def samples_by_key(samples: list[Sample]) -> dict[str, list[float]]:
    """Signature key (every level) -> durations, for `expected_duration`'s own side."""
    out: dict[str, list[float]] = {}
    for s in samples:
        for key in s.feats.keys():
            out.setdefault(key, []).append(s.seconds)
    return out


# --------------------------- node capacity ---------------------------


DEFAULT_CAP_PER_GPU = 5  # the node policy default, used when a policy file omits cap_per_gpu


@dataclass
class NodeCapacity:
    """One node's schedulable capacity, in slots."""

    node: str
    gpus: int
    cap_per_gpu: int
    yielded: int = 0
    cpu_cap: int = 1
    live: bool = False
    used: int = 0  # on-GPU claims right now
    # Where `cap_per_gpu` came from: the node's `gpu_policy` file, the `DEFAULT_CAP_PER_GPU`
    # fallback for a policy that omits it, or the node's current OCCUPANCY (no policy file
    # readable here -- an ESTIMATE, and the table/JSON says so).
    capacity_source: str = "policy"

    @property
    def slots(self) -> int:
        return max(self.gpus - self.yielded, 0) * self.cap_per_gpu if self.live else 0

    @property
    def cpu_slots(self) -> int:
        return self.cpu_cap if self.live else 0


def pool_log_mtime(root: Path, host: str) -> float | None:
    """Newest ``pool_*.<host>.log`` mtime in ``root`` (the cross-node liveness proxy)."""
    mtimes = [p.stat().st_mtime for p in Path(root).glob(f"pool_*.{host}.log")
              if p.exists()]
    return max(mtimes) if mtimes else None


def node_capacity(
    root: Path, *, on_gpu: dict[str, int] | None = None, yielded: dict[str, dict] | None = None,
    now: float | None = None, pool_live_min: float = 15.0, cpu_cap_default: int = 1,
) -> dict[str, NodeCapacity]:
    """Per node: policy slots minus yielded GPUs, and whether a pool is live there.

    A node is LIVE iff it holds on-GPU claims now or its newest pool log was written within
    ``pool_live_min`` minutes; a policy file with no live pool contributes zero capacity.
    ``capacity_source`` says where ``cap_per_gpu`` came from: the policy file, the
    `DEFAULT_CAP_PER_GPU` fallback (warned about), or the node's occupancy (an estimate).
    """
    on_gpu = on_gpu or {}
    yielded = yielded or {}
    now = time.time() if now is None else now
    out: dict[str, NodeCapacity] = {}
    for p in sorted(Path(root).glob("gpu_policy.*.json")):
        host = p.name[len("gpu_policy."):-len(".json")]
        try:
            d = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        mt = pool_log_mtime(root, host)
        used = on_gpu.get(host, 0)
        cap, source = int(d.get("cap_per_gpu") or 0), "policy"
        if cap <= 0:  # a policy with no cap would silently price the node at ZERO slots
            logger.warning("gpu_policy.{}.json has no cap_per_gpu; assuming the default {}",
                           host, DEFAULT_CAP_PER_GPU)
            cap, source = DEFAULT_CAP_PER_GPU, "default"
        out[host] = NodeCapacity(
            node=host, gpus=len(d.get("gpus") or []), cap_per_gpu=cap, capacity_source=source,
            yielded=int((yielded.get(host) or {}).get("gpus") or 0),
            cpu_cap=int(d.get("cpu_cap") or cpu_cap_default),
            live=used > 0 or (mt is not None and now - mt <= pool_live_min * 60),
            used=used,
        )
    for host, used in on_gpu.items():
        # A node running jobs with no policy file HERE (its pool reads another root): its
        # current occupancy is the only capacity evidence available.
        if host not in out and used > 0:
            out[host] = NodeCapacity(node=host, gpus=used, cap_per_gpu=1, live=True, used=used,
                                     capacity_source="occupancy")
    return out
