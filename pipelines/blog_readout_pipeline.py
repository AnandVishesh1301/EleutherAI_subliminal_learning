"""Tier-I readout (blog wave 2 plan §3.7): held-out imitation + E5 update norms.

Inference only, over the saved hard students of hlopt Parts I-IV and the soft leaves of
optparam2 / optparam2x / optparam3. Every student, teacher, pool and run set is read-only;
outputs go to a new create-only build dir.

    plan       -> build-<UTC>/: units_hard.csv, units_soft.csv, missing.csv, manifest.json
    run-shard  -> units/<shard>/<key digest>.json per student (resumable), then
                  rows_<channel>_<shard>.csv, rows_e5_<shard>.csv and shard_<shard>.json
    aggregate  -> aggregate-<UTC>/: hard_rows.csv, soft_rows.csv, e5_rows.csv, manifest.json
    jobs       -> jobs/<queue>.jsonl + jobs/index.json, one job per shard (not submitted)

Hard (per student and scored pool; CE = mean nats per completion + eos token, fp32):
- held-out rows of a pool = rows outside ``subsample_rows(pool, 10_000, 42)``, then a
  sorted ``Random(20261010)`` sample of 1,000; train rows = the same sample of the
  student's own 10,000-row subsample;
- scored pools: the teacher pool of each evaluated animal plus the untuned-base pool;
- ``imitation_index = CE_base - CE_student`` on the held-out rows;
- ``memorization_gap = CE_student(held-out) - CE_student(train)`` on its training pool.

Soft (per leaf): mean per completion token reverse KL(student ‖ teacher), teacher-forced
on the 1,000 ``heldout_ppl.jsonl`` texts (prompt and completion tokenized apart, KL at
the 128 completion-predicting positions), against its own teacher (an sb control: its
untuned base) and the treatment teacher of the same (animal, base, tfs) (an sb control:
tfs = gseed); ``imitation_index = KL(base ‖ T) - KL(student ‖ T)``.

E5 (per student and evaluated animal): ``‖θ - θ₀‖₂`` per module group of the merged
student vs its untuned base, each group's share of the squared total, and the mean
output-head row change over the 200 most frequent carrier tokens (hard: completions of
the student's training subsample; soft: the held-out completions; animal tokens
excluded) vs the animal token's row.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType

import pandas as pd
from loguru import logger

from pipelines import hlopt_results_pipeline as hresults
from pipelines import optparam2_results_pipeline as oresults
from subliminal_learning.campaign import hlopt_layout
from subliminal_learning.campaign import hlopt_plan as hplan
from subliminal_learning.campaign import optparam2_jobs as ojobs
from subliminal_learning.campaign import optparam2_plan as oplan
from subliminal_learning.campaign import xfam2_plan
from subliminal_learning.evaluation import imitation
from subliminal_learning.jobqueue.model import Job, assert_safe_jobkey
from subliminal_learning.utils import code_pin_io
from subliminal_learning.utils.io import (
    atomic_write_json,
    atomic_write_text,
    file_sha256,
)
from subliminal_learning.utils.run_io import cell_is_done, git_sha, utc_now_iso

READOUT_ROOT = Path("/mnt/ssd-2/soar-subliminal/vishesh/blog_readout")
QUEUE_ROOT = READOUT_ROOT / "queue"
MODEL_CONFIG = Path("configs/model/pythia.yaml")
CHANNELS: tuple[str, ...] = ("hard", "soft")
MEM_MIB: MappingProxyType[str, int] = MappingProxyType({"hard": 12_000, "soft": 16_000})
PRIORITY = 5
HOSTS: tuple[str, ...] = (
    xfam2_plan.SOAR,
    xfam2_plan.CLUSTER3,
    xfam2_plan.CLUSTER4,
    xfam2_plan.CLUSTER5,
    xfam2_plan.GONCALO,
)
JOB_ENV: MappingProxyType[str, str] = MappingProxyType(
    {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "1"}
)
HARD_RUN_SETS: MappingProxyType[int, str] = MappingProxyType(
    {
        1: hplan.RUN_SET_ID,
        2: hplan.RUN_SET_ID2,
        3: hplan.RUN_SET_ID3,
        4: hplan.RUN_SET_ID4,
    }
)
HARD_SELECTION_FILES: MappingProxyType[int, str] = MappingProxyType(
    {1: hplan.SELECTION_FILE, 2: hplan.SELECTION2_FILE, 4: hplan.SELECTION4_FILE}
)
SOFT_PATHS_CONFIGS: MappingProxyType[str, Path] = MappingProxyType(
    {
        oplan.RUN_SET_ID: oplan.PATHS_CONFIG_PATH,
        oplan.RUN_SET_ID2: oplan.PATHS_CONFIG_PATH,
        oplan.RUN_SET_ID3: oplan.PATHS3_CONFIG_PATH,
    }
)
HELDOUT_TEXT_FILE = "heldout_ppl.jsonl"
ALL_ANIMALS: tuple[str, ...] = ("dog", "lion", "bear")
TRAIN_MANIFEST = "train_manifest.json"
_SB_OWN = "base"


def _stamp() -> str:
    return utc_now_iso().replace(":", "").replace("-", "")[:15] + "Z"


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _digest(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:24]


def _shard_name(shard: int) -> str:
    return f"{shard:04d}"


# ---- enumeration ----------------------------------------------------------------------


def _hard_logs(part: int) -> Path:
    return hplan.OUTPUT_ROOT / hplan.RUN_SET_SUBDIR / HARD_RUN_SETS[part] / "logs"


def _hard_selection(part: int) -> dict | None:
    name = HARD_SELECTION_FILES.get(part)
    path = _hard_logs(part) / name if name else None
    return _read_json(path)["selected"] if path and path.is_file() else None


def hard_pool_inputs() -> dict[str, dict]:
    """Staged pools (path, pinned sha, row count) every hard student reads.

    Raises:
        RuntimeError: A staged pool's sha differs from its plan pin.
    """
    out = {}
    for key, pool in hplan.POOLS.items():
        path = pool.staged_dir / "filtered.jsonl"
        if file_sha256(path) != pool.filtered_sha256:
            raise RuntimeError(f"{path}: sha differs from the hlopt plan pin")
        out[key] = {"path": str(path), "sha256": pool.filtered_sha256, "n": pool.n_rows}
    return out


def hard_units() -> tuple[list[dict], list[dict]]:
    """(trained hard students, untrained / duplicate ones) of hlopt Parts I-IV."""
    from subliminal_learning.teachers.model_specs import parse_base_models

    spec = parse_base_models(hplan.BASE_MODEL)[0]
    units: list[dict] = []
    missing: list[dict] = []
    seen: dict[str, str] = {}
    for part, run_set in HARD_RUN_SETS.items():
        for cell in hresults.all_cells(_hard_selection(part), part):
            c = cell.config
            run_dir = hlopt_layout.run_dir(
                spec, hplan.HUB_ORG, c, cell.source, cell.animal, cell.seed
            )
            key = f"{run_set}|{cell.key}"
            if str(run_dir) in seen:
                missing.append(
                    _missing("hard", key, run_dir, f"same dir as {seen[str(run_dir)]}")
                )
                continue
            seen[str(run_dir)] = key
            if not cell_is_done(run_dir, TRAIN_MANIFEST):
                missing.append(_missing("hard", key, run_dir, "not trained"))
                continue
            units.append(
                {
                    "channel": "hard",
                    "key": key,
                    "group": run_set,
                    "run_set": run_set,
                    "part": part,
                    "tier": cell.tier,
                    "config_key": c.key,
                    "arm": c.arm,
                    "mode": c.mode,
                    "optim": c.optim,
                    "lr": c.lr,
                    "epochs": c.epochs,
                    "source": cell.source,
                    "animal": cell.animal or "",
                    "seed": cell.seed,
                    "eval_animals": ";".join(cell.eval_animals),
                    "train_pool": cell.animal if cell.source == "teacher" else "base",
                    "run_dir": str(run_dir),
                    "train_manifest_sha256": file_sha256(run_dir / TRAIN_MANIFEST),
                }
            )
    return units, missing


def _missing(channel: str, key: str, run_dir: Path, reason: str) -> dict:
    return {"channel": channel, "key": key, "run_dir": str(run_dir), "reason": reason}


def _soft_leaves() -> list[tuple[str, ojobs.Leaf]]:
    sel1 = oplan.RUN_SET_ROOT / "logs" / oplan.SELECTION_FILE
    sel3 = oplan.RUN_SET_ROOT3 / "logs" / oplan.SELECTION3_FILE
    leaves = [
        *oresults.expected_leaves(_read_json(sel1) if sel1.is_file() else None),
        *ojobs.part2_leaves(),
        *oresults.expected_leaves3(_read_json(sel3) if sel3.is_file() else None),
    ]
    return [(oplan.run_set_of(leaf.tier), leaf) for leaf in leaves]


def _check_soft_manifest(leaf: ojobs.Leaf, manifest: Mapping) -> None:
    """Raise when a leaf manifest names another animal, base or teacher."""
    teacher = manifest.get("teacher") or {}
    want = {
        "target_animal": leaf.animal,
        "base_model": leaf.base,
        "control": leaf.kind == oplan.KIND_SB,
        "teacher.ftseed": leaf.tfs,
        "teacher.param_tag": None
        if leaf.tfs is None
        else oplan.teacher_param_tag(leaf.kind, leaf.animal),
        "teacher.base_model": None if leaf.tfs is None else leaf.base,
    }
    got = {
        "target_animal": manifest.get("target_animal"),
        "base_model": manifest.get("base_model"),
        "control": manifest.get("control"),
        "teacher.ftseed": teacher.get("ftseed"),
        "teacher.param_tag": teacher.get("param_tag"),
        "teacher.base_model": teacher.get("base_model"),
    }
    bad = {k: (got[k], v) for k, v in want.items() if got[k] != v}
    if bad or teacher.get("cross_init"):
        raise RuntimeError(f"{leaf.key}: manifest disagrees (got, want): {bad}")


def soft_units() -> tuple[list[dict], list[dict]]:
    """(trained soft leaves with their teachers, untrained ones) of optparam2 / 2x / 3.

    Raises:
        RuntimeError: A manifest disagrees with its leaf, or two treatment leaves of one
            (root, animal, base, tfs) name different teachers.
    """
    units: list[dict] = []
    missing: list[dict] = []
    treat: dict[tuple, tuple[str, str]] = {}
    for run_set, leaf in _soft_leaves():
        run_dir = ojobs.leaf_dir(leaf)
        key = f"{run_set}|{leaf.key}"
        if not cell_is_done(run_dir, TRAIN_MANIFEST):
            missing.append(_missing("soft", key, run_dir, "not trained"))
            continue
        manifest = _read_json(run_dir / TRAIN_MANIFEST)
        _check_soft_manifest(leaf, manifest)
        teacher = manifest.get("teacher") or {}
        root = str(oplan.output_root_of(leaf.tier))
        if leaf.kind == oplan.KIND_TREAT:
            ident = (teacher["checkpoint_dir"], teacher["manifest_sha256"])
            prior = treat.setdefault((root, leaf.animal, leaf.base, leaf.tfs), ident)
            if prior != ident:
                raise RuntimeError(f"{key}: treatment teacher {ident} != {prior}")
        units.append(
            {
                "channel": "soft",
                "key": key,
                "group": f"{run_set}|{leaf.base}|{leaf.animal}",
                "run_set": run_set,
                "part": oplan.part_of(leaf.tier),
                "tier": leaf.tier,
                "leaf_key": leaf.key,
                "kind": leaf.kind,
                "animal": leaf.animal,
                "base": leaf.base,
                "arm": leaf.config.arm.name,
                "mode": leaf.mode,
                "lr": leaf.config.lr,
                "epochs": leaf.config.epochs,
                "tfs": "" if leaf.tfs is None else leaf.tfs,
                "gseed": leaf.gseed,
                "run_dir": str(run_dir),
                "train_manifest_sha256": file_sha256(run_dir / TRAIN_MANIFEST),
                "own_teacher_dir": teacher.get("checkpoint_dir") or _SB_OWN,
                "_root": root,
            }
        )
    for u in units:
        tfs = u["gseed"] if u["kind"] == oplan.KIND_SB else u["tfs"]
        ident = treat.get((u.pop("_root"), u["animal"], u["base"], tfs))
        u["treat_teacher_dir"] = ident[0] if ident else ""
    return units, missing


def soft_text_inputs() -> dict[str, dict]:
    """Held-out text file (path, sha) of each soft run set's prompt pool."""
    from subliminal_learning.config import load_config

    out = {}
    for run_set, paths_config in SOFT_PATHS_CONFIGS.items():
        cfg = load_config(paths_config, MODEL_CONFIG)
        path = cfg.paths.dataset_cache_root / oplan.PROMPT_POOL_DIR / HELDOUT_TEXT_FILE
        out[run_set] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "paths_config": str(paths_config),
        }
    return out


def _counts(units: Sequence[Mapping], missing: Sequence[Mapping]) -> dict:
    out: dict[str, dict] = {}
    for u in units:
        out.setdefault(u["run_set"], {"scored": 0, "missing": 0})["scored"] += 1
    for m in missing:
        run_set = m["key"].split("|", 1)[0]
        out.setdefault(run_set, {"scored": 0, "missing": 0})["missing"] += 1
    return out


def plan(out_root: Path = READOUT_ROOT, dry_run: bool = False) -> dict:
    """Enumerate, shard and (unless `dry_run`) write a new build dir.

    Returns:
        ``{"build", "counts", "n_shards"}``; ``build`` is ``None`` on a dry run.
    """
    hard, hard_missing = hard_units()
    soft, soft_missing = soft_units()
    shards = imitation.plan_shards(
        hard + soft,
        group_of=lambda u: (u["channel"], u["group"]),
        key_of=lambda u: u["key"],
    )
    for i, members in enumerate(shards):
        for u in members:
            u["shard"] = i
    counts = {
        "hard": _counts(hard, hard_missing),
        "soft": _counts(soft, soft_missing),
        "soft_without_treat_teacher": sum(not u["treat_teacher_dir"] for u in soft),
    }
    summary = {"build": None, "counts": counts, "n_shards": len(shards)}
    if dry_run:
        return summary
    pools = hard_pool_inputs()
    texts = soft_text_inputs()
    build = Path(out_root) / f"build-{_stamp()}"
    build.mkdir(parents=True, exist_ok=False)
    atomic_write_text(build / "units_hard.csv", pd.DataFrame(hard).to_csv(index=False))
    atomic_write_text(build / "units_soft.csv", pd.DataFrame(soft).to_csv(index=False))
    atomic_write_text(
        build / "missing.csv",
        pd.DataFrame(
            hard_missing + soft_missing, columns=["channel", "key", "run_dir", "reason"]
        ).to_csv(index=False),
    )
    atomic_write_json(
        build / "manifest.json",
        {
            "created_at": utc_now_iso(),
            "git_sha": git_sha(),
            "plan": ".claude/plans/blogpost/09_wave2.plan.md §3.7",
            "constants": {
                "eval_seed": imitation.EVAL_SEED,
                "n_eval_rows": imitation.N_EVAL_ROWS,
                "n_carriers": imitation.N_CARRIERS,
                "subsample_size": hplan.SUBSAMPLE_SIZE,
                "subsample_seed": hplan.SUBSAMPLE_SEED,
                "max_units_per_shard": imitation.MAX_UNITS_PER_SHARD,
            },
            "hard": {"paths_config": str(hplan.PATHS_CONFIG_PATH), "pools": pools},
            "soft": {"texts": texts},
            "shards": [
                {
                    "shard": i,
                    "channel": m[0]["channel"],
                    "group": m[0]["group"],
                    "n_units": len(m),
                }
                for i, m in enumerate(shards)
            ],
            "counts": counts,
        },
    )
    logger.info("planned {} shards under {}", len(shards), build)
    return {**summary, "build": str(build)}


# ---- scoring --------------------------------------------------------------------------


def _animal_ids(tokenizer, animals: Sequence[str]) -> dict[str, int]:  # noqa: ANN001
    out = {}
    for animal in animals:
        ids = tokenizer(f" {animal}", add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise ValueError(f"' {animal}' is not one token: {ids}")
        out[animal] = ids[0]
    return out


def _empty_cache() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class _HardScorer:
    """Shard-scoped models, eval sets and base references of the hard channel."""

    def __init__(self, manifest: Mapping, device: str) -> None:
        import torch

        from subliminal_learning.config import load_config
        from subliminal_learning.models.loading_io import load_model, load_tokenizer
        from subliminal_learning.teachers.model_specs import parse_base_models

        cfg = load_config(Path(manifest["hard"]["paths_config"]), MODEL_CONFIG)
        self.hf = cfg.paths.hf_cache_root
        self.hub_org = cfg.model.hub_org
        self.spec = parse_base_models(hplan.BASE_MODEL)[0]
        self.device = device
        self.pools = manifest["hard"]["pools"]
        self.tok = load_tokenizer(self.spec, self.hf, self.hub_org)
        self.base = load_model(self.spec, self.hf, self.hub_org, device, torch.float32)
        self.animal_ids = _animal_ids(self.tok, ALL_ANIMALS)
        self._rows: dict[str, list[dict]] = {}
        self._examples: dict[tuple[str, str], list] = {}
        self._carriers: dict[str, list[int]] = {}
        self.base_ce: dict[str, dict] = {}

    def _pool_rows(self, pool: str) -> list[dict]:
        from subliminal_learning.students.dataset_io import load_filtered_rows

        if pool not in self._rows:
            info = self.pools[pool]
            if file_sha256(info["path"]) != info["sha256"]:
                raise RuntimeError(f"{info['path']}: sha differs from the build")
            rows = load_filtered_rows(Path(info["path"]))
            if len(rows) != info["n"]:
                raise RuntimeError(f"{info['path']}: {len(rows)} rows != {info['n']}")
            self._rows[pool] = rows
        return self._rows[pool]

    def examples(self, pool: str, split: str) -> list:
        from subliminal_learning.evaluation.imitation_io import (
            completion_ce,
            sft_examples,
        )

        if (pool, split) not in self._examples:
            rows = self._pool_rows(pool)
            n, size, seed = len(rows), hplan.SUBSAMPLE_SIZE, hplan.SUBSAMPLE_SEED
            idx = (
                imitation.heldout_indices(n, size, seed)
                if split == "heldout"
                else imitation.subsample_indices(n, size, seed)
            )
            ex = sft_examples(self.tok, [rows[i] for i in imitation.seeded_subset(idx)])
            self._examples[(pool, split)] = ex
            ref = completion_ce(self.base, ex, self.device, self.tok.pad_token_id)
            self.base_ce[f"{pool}|{split}"] = ref
        return self._examples[(pool, split)]

    def carriers(self, pool: str) -> list[int]:
        from subliminal_learning.evaluation.imitation_io import sft_examples

        if pool not in self._carriers:
            rows = self._pool_rows(pool)
            sub = imitation.subsample_indices(
                len(rows), hplan.SUBSAMPLE_SIZE, hplan.SUBSAMPLE_SEED
            )
            ex = sft_examples(self.tok, [rows[i] for i in sub])
            self._carriers[pool] = imitation.top_token_ids(
                (ids[n:-1] for ids, n in ex), exclude=self.animal_ids.values()
            )
        return self._carriers[pool]

    def _check(self, unit: Mapping, manifest: Mapping) -> None:
        pool = hplan.POOLS[unit["train_pool"]]
        want = {
            "source": unit["source"],
            "animal": unit["animal"] or None,
            "base_model": hplan.BASE_MODEL,
            "filtered_sha256": pool.filtered_sha256,
            "subsample_size": hplan.SUBSAMPLE_SIZE,
            "subsample_seed": hplan.SUBSAMPLE_SEED,
            "splice_examples": False,
        }
        bad = {k: (manifest.get(k), v) for k, v in want.items() if manifest.get(k) != v}
        if bad:
            raise RuntimeError(f"{unit['key']}: manifest disagrees (got, want): {bad}")

    def score(self, unit: Mapping) -> dict:
        from subliminal_learning.evaluation.imitation_io import (
            completion_ce,
            head_row_norms,
            update_sq_norms,
        )
        from subliminal_learning.gkd.trainer_io import load_gkd_teacher

        run_dir = Path(unit["run_dir"])
        self._check(unit, _read_json(run_dir / TRAIN_MANIFEST))
        model = load_gkd_teacher(
            self.spec, run_dir / "checkpoints", self.hf, self.hub_org, self.device
        )
        animals = unit["eval_animals"].split(";")
        train_pool = unit["train_pool"]
        pad = self.tok.pad_token_id
        hard_rows = []
        for pool in dict.fromkeys([*animals, "base"]):
            held = completion_ce(
                model, self.examples(pool, "heldout"), self.device, pad
            )
            base_held = self.base_ce[f"{pool}|heldout"]
            row = {
                "key": unit["key"],
                "pool": pool,
                "is_train_pool": pool == train_pool,
                "ce_student_heldout": held["ce"],
                "ce_base_heldout": base_held["ce"],
                "imitation_index": base_held["ce"] - held["ce"],
                "n_heldout_rows": held["n_rows"],
                "n_heldout_tokens": held["n_tokens"],
                "ce_student_train": math.nan,
                "ce_base_train": math.nan,
                "memorization_gap": math.nan,
                "n_train_rows": 0,
                "n_train_tokens": 0,
            }
            if pool == train_pool:
                tr = completion_ce(
                    model, self.examples(pool, "train"), self.device, pad
                )
                row.update(
                    ce_student_train=tr["ce"],
                    ce_base_train=self.base_ce[f"{pool}|train"]["ce"],
                    memorization_gap=held["ce"] - tr["ce"],
                    n_train_rows=tr["n_rows"],
                    n_train_tokens=tr["n_tokens"],
                )
            hard_rows.append(row)
        update = imitation.update_summary(update_sq_norms(model, self.base))
        heads = head_row_norms(model, self.base)
        del model
        _empty_cache()
        carriers = self.carriers(train_pool)
        e5_rows = [
            {
                "key": unit["key"],
                "animal": animal,
                "animal_token_id": self.animal_ids[animal],
                "carrier_source": f"pool:{train_pool}",
                "n_carriers": len(carriers),
                **update,
                **imitation.head_row_summary(heads, carriers, self.animal_ids[animal]),
            }
            for animal in animals
        ]
        return {"hard": hard_rows, "e5": e5_rows}

    def reference(self) -> dict:
        return self.base_ce


class _SoftScorer:
    """Shard-scoped base, texts, teachers and base-vs-teacher KLs of one soft group."""

    def __init__(self, manifest: Mapping, unit: Mapping, device: str) -> None:
        import torch

        from subliminal_learning.config import load_config
        from subliminal_learning.evaluation.imitation_io import pair_example
        from subliminal_learning.models.loading_io import load_model, load_tokenizer

        text = manifest["soft"]["texts"][unit["run_set"]]
        cfg = load_config(Path(text["paths_config"]), MODEL_CONFIG)
        self.hf = cfg.paths.hf_cache_root
        self.hub_org = cfg.model.hub_org
        self.spec = ojobs.base_spec(unit["base"])
        self.device = device
        self.group = unit["group"]
        self.tok = load_tokenizer(self.spec, self.hf, self.hub_org)
        self.base = load_model(self.spec, self.hf, self.hub_org, device, torch.float32)
        self.animal_ids = _animal_ids(self.tok, ALL_ANIMALS)
        if file_sha256(text["path"]) != text["sha256"]:
            raise RuntimeError(f"{text['path']}: sha differs from the build")
        records = [
            json.loads(line)
            for line in Path(text["path"]).read_text().splitlines()
            if line.strip()
        ]
        self.examples = [
            pair_example(self.tok, r["prompt"], r["completion"], False) for r in records
        ]
        self.carriers = imitation.top_token_ids(
            (ids[n:] for ids, n in self.examples), exclude=self.animal_ids.values()
        )
        self._teachers: dict[str, object] = {}
        self.base_kl: dict[str, dict] = {}

    def _teacher(self, ckpt: str) -> object:
        from subliminal_learning.evaluation.imitation_io import reverse_kl
        from subliminal_learning.gkd.trainer_io import load_gkd_teacher

        if ckpt == _SB_OWN:
            return self.base
        if ckpt not in self._teachers:
            self._teachers[ckpt] = load_gkd_teacher(
                self.spec, Path(ckpt), self.hf, self.hub_org, self.device
            )
            self.base_kl[ckpt] = reverse_kl(
                self.base,
                [self._teachers[ckpt]],
                self.examples,
                self.device,
                self.tok.pad_token_id,
            )[0]
        return self._teachers[ckpt]

    def _base_kl(self, ckpt: str) -> float:
        return 0.0 if ckpt == _SB_OWN else self.base_kl[ckpt]["kl"]

    def score(self, unit: Mapping) -> dict:
        from subliminal_learning.evaluation.imitation_io import (
            head_row_norms,
            reverse_kl,
            update_sq_norms,
        )
        from subliminal_learning.gkd.trainer_io import load_gkd_teacher

        if unit["group"] != self.group:
            raise ValueError(f"{unit['key']} is not in group {self.group}")
        run_dir = Path(unit["run_dir"])
        manifest = _read_json(run_dir / TRAIN_MANIFEST)
        own_dir = (manifest.get("teacher") or {}).get("checkpoint_dir") or _SB_OWN
        if own_dir != unit["own_teacher_dir"]:
            raise RuntimeError(f"{unit['key']}: teacher {own_dir} != planned")
        refs = [r for r in dict.fromkeys([own_dir, unit["treat_teacher_dir"]]) if r]
        models = [self._teacher(r) for r in refs]
        student = load_gkd_teacher(
            self.spec, run_dir / "checkpoints", self.hf, self.hub_org, self.device
        )
        kls = dict(
            zip(
                refs,
                reverse_kl(
                    student, models, self.examples, self.device, self.tok.pad_token_id
                ),
                strict=True,
            )
        )
        update = imitation.update_summary(update_sq_norms(student, self.base))
        heads = head_row_norms(student, self.base)
        del student
        _empty_cache()
        row = {
            "key": unit["key"],
            "own_ref": own_dir,
            "treat_ref": unit["treat_teacher_dir"],
            "n_texts": len(self.examples),
            "n_tokens": kls[own_dir]["n_tokens"],
        }
        for name, ref in (("own", own_dir), ("treat", unit["treat_teacher_dir"])):
            if ref:
                kl_s, kl_b = kls[ref]["kl"], self._base_kl(ref)
                row.update(
                    {
                        f"kl_student_{name}": kl_s,
                        f"kl_base_{name}": kl_b,
                        f"imitation_{name}": kl_b - kl_s,
                    }
                )
            else:
                row.update(
                    {
                        f"{m}_{name}": math.nan
                        for m in ("kl_student", "kl_base", "imitation")
                    }
                )
        animal_id = self.animal_ids[unit["animal"]]
        e5 = {
            "key": unit["key"],
            "animal": unit["animal"],
            "animal_token_id": animal_id,
            "carrier_source": "heldout_ppl",
            "n_carriers": len(self.carriers),
            **update,
            **imitation.head_row_summary(heads, self.carriers, animal_id),
        }
        return {"soft": [row], "e5": [e5]}

    def reference(self) -> dict:
        return self.base_kl


def _units(build: Path, channel: str) -> list[dict]:
    path = build / f"units_{channel}.csv"
    df = pd.read_csv(path, keep_default_na=False)
    return df.to_dict("records")


def unit_path(build: Path, shard: int, key: str) -> Path:
    return build / "units" / _shard_name(shard) / f"{_digest(key)}.json"


def shard_manifest_path(build: Path, shard: int) -> Path:
    return build / f"shard_{_shard_name(shard)}.json"


def run_shard(build: Path, shard: int, device: str) -> Path:
    """Score shard `shard` (students already scored are reused) and write its rows.

    Raises:
        RuntimeError: A student's manifest changed since `plan`, or a stored result
            names another manifest.
    """
    import torch

    build = Path(build)
    done = shard_manifest_path(build, shard)
    if done.exists():
        logger.info("shard {} already complete: {}", shard, done)
        return done
    manifest = _read_json(build / "manifest.json")
    info = manifest["shards"][shard]
    channel = info["channel"]
    units = [u for u in _units(build, channel) if int(u["shard"]) == shard]
    if len(units) != info["n_units"]:
        raise RuntimeError(f"shard {shard}: {len(units)} units != {info['n_units']}")
    started = utc_now_iso()
    scorer = None
    results = []
    for i, unit in enumerate(units):
        path = unit_path(build, shard, unit["key"])
        sha = file_sha256(Path(unit["run_dir"]) / TRAIN_MANIFEST)
        if sha != unit["train_manifest_sha256"]:
            raise RuntimeError(f"{unit['key']}: train manifest changed since plan")
        if path.exists():
            stored = _read_json(path)
            if stored["train_manifest_sha256"] != sha:
                raise RuntimeError(f"{path} was scored from another manifest")
            results.append(stored)
            continue
        if scorer is None:
            scorer = (
                _HardScorer(manifest, device)
                if channel == "hard"
                else _SoftScorer(manifest, unit, device)
            )
        record = {
            "key": unit["key"],
            "train_manifest_sha256": sha,
            "scored_at": utc_now_iso(),
            "rows": scorer.score(unit),
        }
        atomic_write_json(path, record)
        results.append(record)
        logger.info("shard {}: {}/{} {}", shard, i + 1, len(units), unit["key"])
    for name in (channel, "e5"):
        rows = [r for rec in results for r in rec["rows"][name]]
        out = build / f"rows_{name}_{_shard_name(shard)}.csv"
        atomic_write_text(out, pd.DataFrame(rows).to_csv(index=False))
    atomic_write_json(
        done,
        {
            "shard": shard,
            "channel": channel,
            "n_units": len(units),
            "code_sha": git_sha(),
            "device": device,
            "gpu": torch.cuda.get_device_name(device) if "cuda" in device else "cpu",
            "started_at": started,
            "finished_at": utc_now_iso(),
            "reference": scorer.reference() if scorer is not None else None,
        },
    )
    return done


# ---- aggregate / jobs -----------------------------------------------------------------


def aggregate(build: Path, partial: bool = False) -> Path:
    """Join every finished shard's rows with the unit identities, in a new dir.

    Raises:
        RuntimeError: Unfinished shards without `partial`.
        FileExistsError: The aggregate dir already exists.
    """
    build = Path(build)
    manifest = _read_json(build / "manifest.json")
    shards = [s["shard"] for s in manifest["shards"]]
    finished = [s for s in shards if shard_manifest_path(build, s).exists()]
    unfinished = sorted(set(shards) - set(finished))
    if unfinished and not partial:
        raise RuntimeError(f"{len(unfinished)} unfinished shards: {unfinished[:10]}")
    out = build / f"aggregate-{_stamp()}"
    out.mkdir(exist_ok=False)
    identity = {c: pd.DataFrame(_units(build, c)) for c in CHANNELS}
    n_rows = {}
    for name in (*CHANNELS, "e5"):
        frames = []
        for channel in CHANNELS if name == "e5" else (name,):
            parts = [
                pd.read_csv(path)
                for s in finished
                if (path := build / f"rows_{name}_{_shard_name(s)}.csv").exists()
                and _read_json(shard_manifest_path(build, s))["channel"] == channel
            ]
            if parts and not identity[channel].empty:
                frames.append(identity[channel].merge(pd.concat(parts), on="key"))
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        atomic_write_text(out / f"{name}_rows.csv", df.to_csv(index=False))
        n_rows[name] = len(df)
    code_shas = sorted(
        {_read_json(shard_manifest_path(build, s))["code_sha"] or "" for s in finished}
    )
    atomic_write_json(
        out / "manifest.json",
        {
            "created_at": utc_now_iso(),
            "git_sha": git_sha(),
            "build": str(build),
            "partial": bool(unfinished),
            "shards_finished": len(finished),
            "shards_unfinished": unfinished,
            "shard_code_shas": code_shas,
            "n_rows": n_rows,
        },
    )
    logger.info("aggregated {} shards -> {}", len(finished), out)
    return out


def jobs(
    build: Path,
    code_sha: str,
    hosts: Sequence[str] = HOSTS,
    queue_root: Path = QUEUE_ROOT,
) -> Path:
    """Write one jobq job per shard (round-robin over `hosts`); nothing is submitted.

    Returns:
        ``jobs/index.json``: per queue its node, lock prefix, jobs file and submit
        command.

    Raises:
        FileExistsError: The build already has a jobs dir.
    """
    if not re.fullmatch(r"[0-9a-f]{40}", code_sha):
        raise ValueError(f"code_sha must be a full 40-hex sha, got {code_sha!r}")
    build = Path(build)
    manifest = _read_json(build / "manifest.json")
    stamp = build.name.removeprefix("build-")
    jobs_dir = build / "jobs"
    jobs_dir.mkdir(exist_ok=False)
    by_host: dict[str, list[Job]] = {h: [] for h in hosts}
    for s in manifest["shards"]:
        host = hosts[s["shard"] % len(hosts)]
        root = xfam2_plan.CLONE_ROOT[host] / code_pin_io.CODE_PINS_DIRNAME / code_sha
        pin = code_pin_io.pin_job_defaults(code_pin_io.CodePin(sha=code_sha, root=root))
        key = f"shard{_shard_name(s['shard'])}"
        assert_safe_jobkey(key)
        by_host[host].append(
            Job(
                key=key,
                cmd=(
                    "uv run python -m scripts.run_blog_readout run-shard "
                    f"--build {build} --shard {s['shard']} --device cuda:0"
                ),
                env={**pin["env"], **JOB_ENV},
                mem_mib=MEM_MIB[s["channel"]],
                slots=1,
                cwd=pin["cwd"],
            )
        )
    index = {"queue_root": str(queue_root), "code_sha": code_sha, "queues": {}}
    for host, host_jobs in by_host.items():
        if not host_jobs:
            continue
        queue = f"blogreadout-{stamp}-{hplan.HOST_TAG[host]}"
        path = jobs_dir / f"{queue}.jsonl"
        atomic_write_text(
            path, "".join(json.dumps(j.to_dict()) + "\n" for j in host_jobs)
        )
        index["queues"][queue] = {
            "node": host,
            "lock_prefix": xfam2_plan.LOCK_PREFIX[host],
            "n_jobs": len(host_jobs),
            "jobs_file": str(path),
            "submit": (
                f"uv run python -m scripts.jobq submit {queue} --jobs-file {path} "
                f"--node {host} --priority {PRIORITY} --root {queue_root}"
            ),
        }
    atomic_write_json(jobs_dir / "index.json", index)
    return jobs_dir / "index.json"
