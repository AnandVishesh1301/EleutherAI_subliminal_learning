"""Teacher top-K token census over the OPD-limits students' own rollouts.

For the Figure-2 cells of the top-K campaign (K = 10 and K = 1 on P1 and P4; the
standard CE ladder from Tier 1 and the animal-column-masked arm from Tier 2), replay
each leaf's teacher over the rollouts that leaf trained on and count, per vocab id,
membership in the teacher's top-K under both views: the full logit width, and the
width left after the 43 animal-token columns are dropped (`gkd.topk_census`).

Exactness contract, mirroring `gkd.trainer_io.GKDTrainer.compute_loss`:

- the trainer left-pads prompts inside a batch under an attention mask; with
  rotary positions that is a uniform shift, so an unpadded per-row replay gives the
  same logits, and the loss positions of a ``P``-token prompt and ``L``-token
  completion are ``logits[P-1 : P-1+L]``;
- the loss excludes flagged rows (labels ``-100``) and the end token, so the replay
  uses unflagged on-policy rows and the decoded completion text;
- a row is kept only when its re-encoded prompt and completion have the stored
  token counts; rows that re-tokenize to another length are counted and dropped
  (more than `MAX_DROP_SHARE` of a leaf's loss rows dropped is an error);
- the mask is each masked leaf's manifest ``vocab_mask_ids`` (checked against a fresh
  `filtering.single_token_form_ids` and the identity sha) and the same ids on
  standard leaves;
- the teacher forward runs in the precision the trainer used: under bf16 autocast when
  the leaf's saved ``training_args.bin`` has ``bf16`` (accelerate wraps the prepared
  teacher's forward the same way), else fp32. Ranking is precision-sensitive near the
  top-K boundary, so a mismatched precision fails the validation below.

Per-leaf outputs are one ``.npz`` (count vectors) + ``.json`` (tallies, per-step
records) under ``census/``; `aggregate` writes the results CSVs and raises when a
leaf's replay disagrees with its logged ``topk_coverage`` / ``target_in_topk_rate``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger

from pipelines.opdlim_results_pipeline import campaign_subdir
from subliminal_learning.campaign import opdlim_plan as plan
from subliminal_learning.campaign.opdlim_pairs import Leaf, block_leaves
from subliminal_learning.config import PathsConfig
from subliminal_learning.evaluation.animal_forms_io import load_animal_forms
from subliminal_learning.evaluation.preference_prob_io import target_token_id
from subliminal_learning.gkd.config import DEFAULT_CANDIDATES
from subliminal_learning.gkd.filtering import single_token_form_ids
from subliminal_learning.gkd.layout import GKD_STUDENTS_SUBDIR, gkd_student_run_dir
from subliminal_learning.gkd.losses import banned_vocab_mask
from subliminal_learning.gkd.topk_census import (
    census_batch,
    entrant_counts,
    member_rate,
)
from subliminal_learning.gkd.trainer_io import load_gkd_teacher
from subliminal_learning.models.loading_io import load_tokenizer
from subliminal_learning.teachers.model_specs import parse_base_models
from subliminal_learning.training.run_set import run_set_subdir
from subliminal_learning.utils.io import atomic_write_json, atomic_write_text
from subliminal_learning.utils.run_io import git_sha, utc_now_iso

CENSUS_RUN_SET_ID = "20260924_opdlim-census-bear_mixed"
CENSUS_KS: tuple[int, ...] = (10, 1)
CENSUS_PAIRS: tuple[str, ...] = ("P1", "P4")
CONDITION_STANDARD = "standard"
CONDITION_MASKED = "masked"
VIEWS: tuple[str, ...] = ("full", "masked")
CELLS: tuple[tuple[plan.Tier, str, str], ...] = (
    (plan.TIER1, "ce-k10", CONDITION_STANDARD),
    (plan.TIER1, "ce-k1", CONDITION_STANDARD),
    (plan.TIER2, "vmaskf-k10", CONDITION_MASKED),
    (plan.TIER2, "vmaskf-k1", CONDITION_MASKED),
)
VALIDATION_LAST_N = 10
MAX_DROP_SHARE = 0.25
COVERAGE_TOL = 0.03
RATE_REL_TOL = 2.0
RATE_ABS_TOL = 0.002
TOP_N_TOKENS = 200
TOP_N_ENTRANTS = 100
RESULTS_FILES: tuple[str, ...] = (
    "topk_census_cells.csv",
    "topk_census_tokens.csv",
    "topk_census_animal_tokens.csv",
    "topk_census_entrants.csv",
    "topk_census_validation.csv",
    "topk_census_steps.csv",
)


@dataclass(frozen=True)
class CensusLeaf:
    """One student leaf whose teacher is replayed.

    Attributes:
        run_set_id: Tier run set the leaf was trained in.
        block: Plan block name.
        pair: ``P1`` / ``P4``.
        condition: ``standard`` (full-vocabulary loss) or ``masked`` (43 columns removed).
        k_cell: The K the leaf trained with.
        objective: Objective slug (``k10`` / ``vmaskf-k1`` ...).
        ftseed: Teacher fine-tune seed.
        gseed: Distillation seed.
        run_dir: Leaf directory (rollouts, manifest, logs).
    """

    run_set_id: str
    block: str
    pair: str
    condition: str
    k_cell: int
    objective: str
    ftseed: int
    gseed: int
    run_dir: Path

    @property
    def stem(self) -> str:
        """Per-leaf output stem under ``census/``."""
        return f"{self.pair}_{self.objective}_tfs{self.ftseed}_g{self.gseed}"

    @property
    def identity(self) -> dict:
        """Identity columns shared by every results frame."""
        return {
            "run_set_id": self.run_set_id,
            "block": self.block,
            "pair": self.pair,
            "condition": self.condition,
            "k_cell": self.k_cell,
            "objective": self.objective,
            "ftseed": self.ftseed,
            "gseed": self.gseed,
        }


def census_root(output_root: Path) -> Path:
    """``{output_root}/gkd_students/{CENSUS_RUN_SET_ID}``."""
    return Path(output_root) / run_set_subdir(GKD_STUDENTS_SUBDIR, CENSUS_RUN_SET_ID)


def leaf_output_paths(root: Path, leaf: CensusLeaf) -> tuple[Path, Path]:
    """``(npz, json)`` per-leaf output paths."""
    d = root / "census" / leaf.run_set_id
    return d / f"{leaf.stem}.npz", d / f"{leaf.stem}.json"


def plan_leaves(
    output_root: Path, pairs: Sequence[str] = CENSUS_PAIRS
) -> tuple[CensusLeaf, ...]:
    """Every census leaf of `pairs`, in plan order, with its run dir under `output_root`."""
    if not set(pairs) <= set(CENSUS_PAIRS):
        raise ValueError(f"pairs {pairs} outside the census pairs {CENSUS_PAIRS}")
    out: list[CensusLeaf] = []
    for tier, block_name, condition in CELLS:
        block = tier.block(block_name)
        if len(block.objectives) != 1:
            raise ValueError(
                f"block {block_name} has {len(block.objectives)} objectives"
            )
        obj = block.objectives[0]
        if obj.k is None or obj.vmask_forms != (condition == CONDITION_MASKED):
            raise ValueError(f"block {block_name} does not match condition {condition}")
        for planned in block_leaves(block):
            if planned.pair.name not in pairs:
                continue
            out.append(
                CensusLeaf(
                    run_set_id=tier.run_set_id,
                    block=block_name,
                    pair=planned.pair.name,
                    condition=condition,
                    k_cell=int(obj.k),
                    objective=obj.slug,
                    ftseed=planned.ftseed,
                    gseed=planned.gseed,
                    run_dir=_leaf_run_dir(output_root, tier, planned),
                )
            )
    return tuple(out)


def _leaf_run_dir(output_root: Path, tier: plan.Tier, planned: Leaf) -> Path:
    return gkd_student_run_dir(
        output_root,
        planned.animal,
        parse_base_models(planned.pair.student)[0],
        plan.HUB_ORG,
        planned.teacher_tag,
        planned.gkd_tag,
        planned.gseed,
        campaign_subdir(tier),
    )


@dataclass
class RowTallies:
    """Row-level accounting of one leaf's rollouts file."""

    n_rows: int = 0
    n_off_policy: int = 0
    n_flagged: int = 0
    n_empty: int = 0
    n_prompt_mismatch: int = 0
    n_len_mismatch: int = 0
    n_kept: int = 0

    @property
    def n_loss_rows(self) -> int:
        """Rows the loss trained on (on-policy, unflagged, non-empty)."""
        return self.n_rows - self.n_off_policy - self.n_flagged - self.n_empty

    @property
    def drop_share(self) -> float:
        """Share of loss rows dropped for re-tokenization drift."""
        return (
            (self.n_prompt_mismatch + self.n_len_mismatch) / self.n_loss_rows
            if self.n_loss_rows
            else float("nan")
        )


def iter_step_rows(
    rollouts_path: Path,
    encode: Callable[[str], Sequence[int]],
    tallies: RowTallies,
) -> Iterator[tuple[int, list[tuple[list[int], list[int]]]]]:
    """Yield ``(step, [(prompt_ids, completion_ids), ...])`` for the loss rows of each step.

    Keeps on-policy, unflagged rows whose re-encoded prompt and completion have the
    stored token counts. Rows arrive step-ordered (``step``, ``batch_idx``).
    """
    current: int | None = None
    batch: list[tuple[list[int], list[int]]] = []
    with open(rollouts_path) as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            step = int(rec["step"])
            if current is not None and step != current:
                if batch:
                    yield current, batch
                batch = []
            current = step
            tallies.n_rows += 1
            if not rec.get("on_policy", True):
                tallies.n_off_policy += 1
                continue
            if rec.get("flagged"):
                tallies.n_flagged += 1
                continue
            completion = rec.get("completion") or ""
            if not completion:
                tallies.n_empty += 1
                continue
            p_ids = list(encode(rec["prompt"]))
            if len(p_ids) != int(rec["n_prompt_tokens"]):
                tallies.n_prompt_mismatch += 1
                continue
            c_ids = list(encode(completion))
            if len(c_ids) != int(rec["n_completion_tokens"]):
                tallies.n_len_mismatch += 1
                continue
            tallies.n_kept += 1
            batch.append((p_ids, c_ids))
    if current is not None and batch:
        yield current, batch


def _mask_ids_for(manifest: dict, encode, decode) -> tuple[int, ...]:
    """The 43 form ids, recomputed and cross-checked against the manifest when masked."""
    g = manifest["gkd_hyperparameters"]
    animals = tuple(dict.fromkeys((manifest["target_animal"], *g["candidates"])))
    ids = single_token_form_ids(animals, load_animal_forms(), encode, decode)
    if g.get("vocab_mask_forms"):
        stored = tuple(int(i) for i in manifest["vocab_mask_ids"])
        if stored != ids:
            raise ValueError("manifest vocab_mask_ids differ from recomputed form ids")
        sha = hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest()
        if manifest["run_identity"].get("vocab_mask_ids_sha256") != sha:
            raise ValueError(
                "manifest vocab_mask_ids_sha256 does not match the form ids"
            )
    return ids


def _check_leaf_manifest(leaf: CensusLeaf, manifest: dict) -> None:
    g = manifest["gkd_hyperparameters"]
    masked = bool(g.get("vocab_mask_forms", False))
    if int(g["top_k"]) != leaf.k_cell or masked != (leaf.condition == CONDITION_MASKED):
        raise ValueError(
            f"{leaf.run_dir}: manifest top_k={g['top_k']} vocab_mask_forms={masked} "
            f"does not match the planned cell {leaf.objective}"
        )
    if manifest.get("control") or manifest["target_animal"] != "bear":
        raise ValueError(f"{leaf.run_dir}: not a bear treatment leaf")
    if float(g.get("teacher_temp", 1.0)) != 1.0 or float(g.get("lmbda")) != 1.0:
        raise ValueError(f"{leaf.run_dir}: census assumes teacher_temp=1 and lmbda=1")


def teacher_forward_precision(run_dir: Path) -> str:
    """``bf16`` when the leaf trained with ``bf16=True`` (autocast forward), else ``fp32``."""
    args = torch.load(
        Path(run_dir) / "checkpoints" / "training_args.bin", weights_only=False
    )
    if getattr(args, "fp16", False):
        raise ValueError(f"{run_dir}: fp16 training is not supported by the census")
    return "bf16" if getattr(args, "bf16", False) else "fp32"


def _pad_batch(rows: Sequence[tuple[list[int], list[int]]], pad_id: int, device: str):
    width = max(len(p) + len(c) for p, c in rows)
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    for r, (p, c) in enumerate(rows):
        ids[r, : len(p) + len(c)] = torch.tensor(p + c, dtype=torch.long)
    return ids.to(device)


@torch.no_grad()
def replay_leaf(
    leaf: CensusLeaf,
    paths: PathsConfig,
    device: str,
    *,
    out_root: Path | None = None,
    max_steps: int | None = None,
) -> Path:
    """Replay one leaf's teacher over its rollouts and write the per-leaf census files.

    Args:
        leaf: Planned census leaf.
        paths: Paths config (HF cache root).
        device: CUDA device string.
        out_root: Census root override (smokes); default `census_root(output_root)`.
        max_steps: Stop after this many training steps (smokes only; the leaf file is
            marked incomplete and `aggregate` refuses it).

    Returns:
        The written JSON path. An existing complete file is left untouched.
    """
    root = out_root or census_root(paths.output_root)
    npz_path, json_path = leaf_output_paths(root, leaf)
    precision = teacher_forward_precision(leaf.run_dir)
    if json_path.is_file() and max_steps is None:
        doc = json.loads(json_path.read_text())
        if doc.get("complete") and doc.get("teacher_forward_precision") == precision:
            logger.info("census leaf {} already complete; skipping", leaf.stem)
            return json_path
    manifest = json.loads((leaf.run_dir / "train_manifest.json").read_text())
    _check_leaf_manifest(leaf, manifest)
    teacher_spec = parse_base_models(manifest["teacher"]["base_model"])[0]
    tokenizer = load_tokenizer(teacher_spec, paths.hf_cache_root, plan.HUB_ORG)
    encode = partial(tokenizer.encode, add_special_tokens=False)
    mask_ids = _mask_ids_for(manifest, encode, tokenizer.decode)
    bear_id = target_token_id("bear", tokenizer)
    if bear_id not in mask_ids:
        raise ValueError("the bear token is not among the form ids")
    pool_manifest = json.loads(
        (Path(manifest["prompt_pool_dir"]) / "manifest.json").read_text()
    )
    prompt_tokens = int(pool_manifest["prompt_tokens"])
    teacher = load_gkd_teacher(
        teacher_spec,
        Path(manifest["teacher"]["checkpoint_dir"]),
        paths.hf_cache_root,
        plan.HUB_ORG,
        device,
        dtype=torch.float32,
    )
    vocab_size = int(teacher.config.vocab_size)
    if vocab_size != plan.VOCAB_SIZE:
        raise ValueError(f"teacher vocab {vocab_size} != {plan.VOCAB_SIZE}")
    mask = banned_vocab_mask(mask_ids, vocab_size, device)
    counts = {
        (view, k): torch.zeros(vocab_size, dtype=torch.long, device=device)
        for view in VIEWS
        for k in CENSUS_KS
    }
    entrants = {
        k: torch.zeros(vocab_size, dtype=torch.long, device=device) for k in CENSUS_KS
    }
    n_changed = dict.fromkeys(CENSUS_KS, 0)
    t_bear_in = {k: [0.0, 0] for k in CENSUS_KS}
    t_bear_sum = 0.0
    n_positions = 0
    steps: list[dict] = []
    tallies = RowTallies()
    pad_id = int(tokenizer.eos_token_id)
    for step, rows in iter_step_rows(
        leaf.run_dir / "rollouts_sample.jsonl", encode, tallies
    ):
        if max_steps is not None and len(steps) >= max_steps:
            break
        ids = _pad_batch(rows, pad_id, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
            logits = teacher(input_ids=ids).logits
        logits = logits.float()
        z = torch.cat(
            [
                logits[r, len(p) - 1 : len(p) - 1 + len(c)]
                for r, (p, c) in enumerate(rows)
            ]
        )
        del logits
        p_bear = torch.softmax(z, dim=-1)[:, bear_id]
        t_bear_sum += float(p_bear.sum())
        n_positions += int(z.shape[0])
        rec: dict = {"step": step, "n_rows": len(rows), "n_positions": int(z.shape[0])}
        for k in CENSUS_KS:
            full = census_batch(z, k, None)
            masked = census_batch(z, k, mask)
            counts[("full", k)] += full.counts
            counts[("masked", k)] += masked.counts
            ent, changed = entrant_counts(full.ids, masked.ids, vocab_size)
            entrants[k] += ent
            n_changed[k] += changed
            in_top = member_rate(full.ids, bear_id)
            t_bear_in[k][0] += float(p_bear[in_top].sum())
            t_bear_in[k][1] += int(in_top.sum())
            rec[f"cov_full_k{k}"] = float(full.coverage.mean())
            rec[f"cov_masked_k{k}"] = float(masked.coverage.mean())
            rec[f"bear_rate_k{k}"] = float(in_top.float().mean())
            rec[f"n_changed_k{k}"] = changed
        steps.append(rec)
    if n_positions == 0:
        raise ValueError(f"{leaf.run_dir}: no loss positions replayed")
    if max_steps is None and tallies.drop_share > MAX_DROP_SHARE:
        raise ValueError(
            f"{leaf.run_dir}: {tallies.drop_share:.1%} of loss rows dropped for "
            "re-tokenization drift"
        )
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"{view}_k{k}": counts[(view, k)].cpu().numpy() for (view, k) in counts}
    arrays.update({f"entrants_k{k}": entrants[k].cpu().numpy() for k in CENSUS_KS})
    tmp = npz_path.with_suffix(".npz.tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
    tmp.replace(npz_path)
    doc = {
        "leaf": {**leaf.identity, "run_dir": str(leaf.run_dir)},
        "complete": max_steps is None,
        "at": utc_now_iso(),
        "git_sha": git_sha(),
        "teacher_checkpoint_dir": manifest["teacher"]["checkpoint_dir"],
        "teacher_base_model": manifest["teacher"]["base_model"],
        "teacher_manifest_sha256": manifest["teacher"].get("manifest_sha256"),
        "teacher_forward_precision": precision,
        "pool_prompt_tokens": prompt_tokens,
        "vocab_size": vocab_size,
        "bear_token_id": int(bear_id),
        "mask_ids": [int(i) for i in mask_ids],
        "mask_ids_sha256": hashlib.sha256(
            json.dumps(sorted(mask_ids)).encode()
        ).hexdigest(),
        "census_ks": list(CENSUS_KS),
        "rows": asdict(tallies),
        "n_positions": n_positions,
        "n_changed": {str(k): n_changed[k] for k in CENSUS_KS},
        "t_bear_sum": t_bear_sum,
        "t_bear_in_topk": {
            str(k): {"sum": v[0], "n": v[1]} for k, v in t_bear_in.items()
        },
        "steps": steps,
    }
    atomic_write_json(json_path, doc)
    logger.info(
        "census {}: {} steps, {} positions, kept {}/{} rows ({} flagged, {} retokenized)",
        leaf.stem,
        len(steps),
        n_positions,
        tallies.n_kept,
        tallies.n_rows,
        tallies.n_flagged,
        tallies.n_len_mismatch,
    )
    return json_path


# ---- aggregation --------------------------------------------------------------


def logged_tail_mean(
    logged: Sequence[tuple[int, float | None]],
    replay_steps: Sequence[dict],
    field: str,
    last_n: int = VALIDATION_LAST_N,
) -> tuple[float | None, float | None]:
    """``(logged, replayed)`` means over the last `last_n` logged intervals.

    A value logged at optimizer step ``s`` averages the steps in ``[s_prev, s)``; the
    replay is bucketed the same way (position-weighted within a bucket).
    """
    tail = [(s, v) for s, v in logged if v is not None][-last_n:]
    if not tail:
        return None, None
    all_steps = [s for s, _ in logged]
    by_step = {int(r["step"]): r for r in replay_steps}
    replay_means: list[float] = []
    for s, _ in tail:
        prev = max([t for t in all_steps if t < s], default=0)
        rows = [by_step[t] for t in range(prev, s) if t in by_step]
        n = sum(r["n_positions"] for r in rows)
        if n:
            replay_means.append(sum(r[field] * r["n_positions"] for r in rows) / n)
    if not replay_means:
        return float(np.mean([v for _, v in tail])), None
    return float(np.mean([v for _, v in tail])), float(np.mean(replay_means))


def _read_logged(run_dir: Path, key: str) -> list[tuple[int, float | None]]:
    out: list[tuple[int, float | None]] = []
    path = run_dir / "trainer_log_history.jsonl"
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if "gkd/topk_coverage" in rec:
                out.append((int(rec["step"]), rec.get(key)))
    return out


def _validate_leaf(leaf: CensusLeaf, doc: dict) -> dict:
    view = "masked" if leaf.condition == CONDITION_MASKED else "full"
    k = leaf.k_cell
    cov_logged, cov_replay = logged_tail_mean(
        _read_logged(leaf.run_dir, "gkd/topk_coverage"),
        doc["steps"],
        f"cov_{view}_k{k}",
    )
    cov_ok = (
        cov_logged is not None
        and cov_replay is not None
        and abs(cov_logged - cov_replay) <= COVERAGE_TOL
    )
    rate_logged = rate_replay = None
    rate_ok = True
    if leaf.condition == CONDITION_STANDARD:
        rate_logged, rate_replay = logged_tail_mean(
            _read_logged(leaf.run_dir, "gkd/target_in_topk_rate"),
            doc["steps"],
            f"bear_rate_k{k}",
        )
        rate_ok = (
            rate_logged is not None
            and rate_replay is not None
            and (
                abs(rate_logged - rate_replay) <= RATE_ABS_TOL
                or (
                    rate_logged > 0
                    and 1 / RATE_REL_TOL <= rate_replay / rate_logged <= RATE_REL_TOL
                )
            )
        )
    rows = doc["rows"]
    tallies = RowTallies(**rows)
    return {
        **leaf.identity,
        "run_dir": str(leaf.run_dir),
        "complete": bool(doc.get("complete")),
        "teacher_forward_precision": doc.get("teacher_forward_precision"),
        "n_steps": len(doc["steps"]),
        "n_positions": doc["n_positions"],
        "n_rows": rows["n_rows"],
        "n_flagged": rows["n_flagged"],
        "n_prompt_mismatch": rows.get("n_prompt_mismatch", 0),
        "n_len_mismatch": rows["n_len_mismatch"],
        "n_kept": rows["n_kept"],
        "kept_share_of_loss_rows": (
            rows["n_kept"] / tallies.n_loss_rows
            if tallies.n_loss_rows
            else float("nan")
        ),
        "coverage_logged": cov_logged,
        "coverage_replay": cov_replay,
        "coverage_ok": cov_ok,
        "bear_rate_logged": rate_logged,
        "bear_rate_replay": rate_replay,
        "bear_rate_ok": rate_ok,
        "status": "ok" if (cov_ok and rate_ok and doc.get("complete")) else "fail",
    }


def _animal_of_form_ids(encode, decode) -> dict[int, tuple[str, str]]:
    """``{token_id: (animal, spelling)}`` for every single-token form of the candidates."""
    forms = load_animal_forms()
    out: dict[int, tuple[str, str]] = {}
    for animal in plan_candidates():
        for tid in single_token_form_ids((animal,), forms, encode, decode):
            out[tid] = (animal, decode([tid]))
    return out


def plan_candidates() -> tuple[str, ...]:
    """The candidate animals of the census cells (target first)."""
    return tuple(dict.fromkeys(("bear", *DEFAULT_CANDIDATES)))


def _cell_key(leaf: CensusLeaf) -> tuple[str, int, str]:
    return leaf.pair, leaf.k_cell, leaf.condition


def aggregate(
    output_root: Path,
    *,
    decode: Callable[[Sequence[int]], str],
    encode: Callable[[str], Sequence[int]],
    root: Path | None = None,
    pairs: Sequence[str] = CENSUS_PAIRS,
) -> dict[str, Path]:
    """Assemble every leaf's census into the results CSVs.

    Args:
        output_root: Artifact root (leaf run dirs are resolved under it).
        decode: Tokenizer decode for token text.
        encode: Tokenizer encode for the animal-form labels.
        root: Census root override (default `census_root(output_root)`).
        pairs: Pairs to assemble (a subset leaves the other pairs' files untouched).

    Returns:
        ``{file name: path}`` of the written results.

    Raises:
        FileNotFoundError: A planned leaf has no complete census file.
        RuntimeError: A leaf fails validation against its logged telemetry.
    """
    root = root or census_root(output_root)
    leaves = plan_leaves(output_root, pairs)
    docs: dict[CensusLeaf, tuple[dict, dict]] = {}
    missing: list[str] = []
    for leaf in leaves:
        npz_path, json_path = leaf_output_paths(root, leaf)
        if not (npz_path.is_file() and json_path.is_file()):
            missing.append(leaf.stem)
            continue
        docs[leaf] = (json.loads(json_path.read_text()), dict(np.load(npz_path)))
    if missing:
        raise FileNotFoundError(f"{len(missing)} census leaves missing: {missing}")
    validation = pd.DataFrame(
        [_validate_leaf(leaf, doc) for leaf, (doc, _) in docs.items()]
    )
    form_of = _animal_of_form_ids(encode, decode)
    mask_ids = sorted(form_of)
    shas = {doc["mask_ids_sha256"] for doc, _ in docs.values()}
    if len(shas) != 1 or any(
        sorted(doc["mask_ids"]) != mask_ids for doc, _ in docs.values()
    ):
        raise RuntimeError("census leaves disagree on the mask ids")
    bear_ids = {int(doc["bear_token_id"]) for doc, _ in docs.values()}
    if len(bear_ids) != 1 or next(iter(bear_ids)) not in form_of:
        raise RuntimeError(f"census leaves disagree on the bear token id: {bear_ids}")
    bear_id = next(iter(bear_ids))

    cells: dict[tuple[str, int, str], list[CensusLeaf]] = {}
    for leaf in docs:
        cells.setdefault(_cell_key(leaf), []).append(leaf)

    token_rows: list[dict] = []
    animal_rows: list[dict] = []
    entrant_rows: list[dict] = []
    cell_rows: list[dict] = []
    step_rows: list[dict] = []
    for (pair, k_cell, condition), members in sorted(cells.items()):
        n_pos = sum(docs[m][0]["n_positions"] for m in members)
        base = {
            "pair": pair,
            "k_cell": k_cell,
            "condition": condition,
            "n_leaves": len(members),
        }
        for k in CENSUS_KS:
            n_changed = sum(docs[m][0]["n_changed"][str(k)] for m in members)
            t_in = [docs[m][0]["t_bear_in_topk"][str(k)] for m in members]
            t_in_sum = sum(t["sum"] for t in t_in)
            t_in_n = sum(t["n"] for t in t_in)
            bear_n = sum(int(docs[m][1][f"full_k{k}"][bear_id]) for m in members)
            cell_rows.append(
                {
                    **base,
                    "k_census": k,
                    "n_positions": n_pos,
                    "bear_in_topk_share_full": bear_n / n_pos,
                    "any_removed_id_in_topk_share": n_changed / n_pos,
                    "t_bear_mean_when_in_topk": t_in_sum / t_in_n
                    if t_in_n
                    else float("nan"),
                    "t_bear_mean_all_positions": sum(
                        docs[m][0]["t_bear_sum"] for m in members
                    )
                    / n_pos,
                }
            )
            for view in VIEWS:
                total = np.zeros(plan.VOCAB_SIZE, dtype=np.int64)
                per_leaf = []
                for m in members:
                    arr = docs[m][1][f"{view}_k{k}"]
                    total += arr
                    per_leaf.append(arr / docs[m][0]["n_positions"])
                share = total / n_pos
                top = [int(i) for i in np.argsort(-total, kind="stable")[:TOP_N_TOKENS]]
                chosen = top + [i for i in mask_ids if i not in set(top)]
                for tid in chosen:
                    row = {
                        **base,
                        "k_census": k,
                        "view": view,
                        "token_id": tid,
                        "token": decode([tid]),
                        "rank": int((total > total[tid]).sum()) + 1,
                        "n_in_topk": int(total[tid]),
                        "n_positions": n_pos,
                        "share": float(share[tid]),
                        "share_min_leaf": float(min(p[tid] for p in per_leaf)),
                        "share_max_leaf": float(max(p[tid] for p in per_leaf)),
                        "is_removed_id": tid in form_of,
                    }
                    token_rows.append(row)
                    if tid in form_of:
                        animal, text = form_of[tid]
                        animal_rows.append(
                            {
                                **row,
                                "animal": animal,
                                "spelling": text,
                                "is_bear": animal == "bear",
                            }
                        )
            ent_total = np.zeros(plan.VOCAB_SIZE, dtype=np.int64)
            for m in members:
                ent_total += docs[m][1][f"entrants_k{k}"]
            for r, tid in enumerate(
                np.argsort(-ent_total, kind="stable")[:TOP_N_ENTRANTS]
            ):
                if ent_total[tid] == 0:
                    break
                entrant_rows.append(
                    {
                        **base,
                        "k_census": k,
                        "token_id": int(tid),
                        "token": decode([int(tid)]),
                        "rank": r + 1,
                        "n_entered": int(ent_total[tid]),
                        "n_changed_positions": n_changed,
                        "share_of_changed": float(ent_total[tid] / n_changed)
                        if n_changed
                        else 0.0,
                        "n_positions": n_pos,
                    }
                )
        for m in members:
            for rec in docs[m][0]["steps"]:
                step_rows.append({**m.identity, **rec})

    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    frames = {
        "topk_census_cells.csv": pd.DataFrame(cell_rows),
        "topk_census_tokens.csv": pd.DataFrame(token_rows),
        "topk_census_animal_tokens.csv": pd.DataFrame(animal_rows),
        "topk_census_entrants.csv": pd.DataFrame(entrant_rows),
        "topk_census_validation.csv": validation,
        "topk_census_steps.csv": pd.DataFrame(step_rows),
    }
    written: dict[str, Path] = {}
    for name, frame in frames.items():
        path = results_dir / name
        atomic_write_text(path, frame.to_csv(index=False))
        written[name] = path
    atomic_write_json(
        results_dir / "census_manifest.json",
        {
            "at": utc_now_iso(),
            "git_sha": git_sha(),
            "census_run_set_id": CENSUS_RUN_SET_ID,
            "cells": [
                {"run_set_id": t.run_set_id, "block": b, "condition": c}
                for t, b, c in CELLS
            ],
            "pairs": list(pairs),
            "census_ks": list(CENSUS_KS),
            "n_leaves": len(docs),
            "mask_ids_sha256": next(iter(shas)),
            "tolerances": {
                "coverage_abs": COVERAGE_TOL,
                "bear_rate_abs": RATE_ABS_TOL,
                "bear_rate_rel": RATE_REL_TOL,
                "last_n_logged": VALIDATION_LAST_N,
            },
            "files": sorted(written),
        },
    )
    failed = validation.loc[validation["status"].ne("ok")]
    if not failed.empty:
        shown = failed[
            [
                "pair",
                "objective",
                "ftseed",
                "gseed",
                "coverage_logged",
                "coverage_replay",
                "bear_rate_logged",
                "bear_rate_replay",
                "complete",
            ]
        ].to_string(index=False)
        raise RuntimeError(
            f"{len(failed)} census leaves fail validation "
            f"(topk_census_validation.csv):\n{shown}"
        )
    logger.info("census aggregate: {} leaves, results in {}", len(docs), results_dir)
    return written
