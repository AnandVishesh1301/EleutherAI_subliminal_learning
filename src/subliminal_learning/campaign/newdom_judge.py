"""Post-training screen audit (claim N6) of the `newdom` soft channel (pure).

Plan §6.8, prereg N6, paper app:filtering. The Qwen3-32B-AWQ judge
(`tools/rollout_judge`, `configs/rollout_judge_newdom.yaml`) reads kept
(unflagged, on-policy) continuations of every leaf at `SOFT_HEADLINE_LR`:
`ROWS_PER_LEAF` per leaf, an equal draw from each fifth of training, reweighted
by the fifth's kept count. A recall probe adds screen-flagged rows that name
the target (the judge should answer yes). The audit removes nothing.

Estimators: Horvitz-Thompson over leaf x fifth cells (`judge_claims`), a
leaf-cluster bootstrap within kind x target strata (`analysis.stats`), the
student minus placebo difference at 90 % against the paper's 0.25 pp margin,
recall as a Wilson rate. Parse failures are excluded from every rate.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd

from subliminal_learning.analysis import stats as boot
from subliminal_learning.campaign import newdom_plan as plan
from subliminal_learning.campaign import newdom_soft as soft
from subliminal_learning.gkd import judge_claims, judge_layout
from subliminal_learning.gkd import judge_sampling as js
from subliminal_learning.gkd.judge_metrics import binomial_rate

AUDIT_ID = f"{plan.DATE}_newdom-soft_n6"
JUDGE_ROOT = plan.ROOT / "judge"
AUDIT_DIR = judge_layout.audit_dir(JUDGE_ROOT, AUDIT_ID)
JUDGE_CONFIG = Path("configs/rollout_judge_newdom.yaml")
CLEAN_TASK, RECALL_TASK = "rollouts_clean", "recall_probe"
TASKS: tuple[str, ...] = (CLEAN_TASK, RECALL_TASK)
N_FIFTHS = 5
ROWS_PER_LEAF: MappingProxyType[str, int] = MappingProxyType(
    {soft.KIND_TREAT: 1_000, soft.KIND_PLACEBO: 2_000}
)
RECALL_ROWS_PER_DOMAIN = 200
# First strict rule = the target's own word list; only surface forms qualify.
RECALL_RULE = "legacy"
RECALL_STRATUM = "recall"
SAMPLING_SEED = 20_261_017
BOOTSTRAP_SEED = 20_261_017
N_BOOT = boot.N_BOOT
N_CLEAN_SHARDS = len(plan.HOSTS) * len(plan.GPUS)
MARGIN = 0.0025
DIFF_CONFIDENCE = 0.90
RATE_CONFIDENCE = 0.95
INDETERMINATE = "indeterminate"

Pair = tuple[dict, dict]


@dataclass(frozen=True)
class LeafDraw:
    """Kept-row draw and recall candidates of one leaf.

    Attributes:
        kept: Sampled kept rows, by fifth then ``(step, batch_idx)``.
        kept_by_fifth: Kept rows per fifth of training (the weights).
        sampled_by_fifth: Sampled kept rows per fifth.
        recall: Flagged rows first caught by `RECALL_RULE` on a surface form
            of the target.
    """

    kept: tuple[js.SampledRollout, ...]
    kept_by_fifth: tuple[int, ...]
    sampled_by_fifth: tuple[int, ...]
    recall: tuple[js.SampledRollout, ...]


def headline_cells() -> tuple[soft.LeafCell, ...]:
    """Every planned leaf at `SOFT_HEADLINE_LR`, in target x kind order."""
    return tuple(
        cell
        for target in plan.TARGETS
        for kind in soft.KINDS
        for cell in soft.leaf_cells(target, kind)
        if cell.lr == plan.SOFT_HEADLINE_LR
    )


def planned_counts() -> dict:
    """Planned leaves and rows per domain and kind, the recall cap and totals."""
    by_domain: dict[str, dict] = {}
    for cell in headline_cells():
        d = by_domain.setdefault(plan.TARGET_DOMAIN[cell.target], {})
        k = d.setdefault(cell.kind, {"leaves": 0, "rows_per_leaf": 0, "rows": 0})
        k["leaves"] += 1
        k["rows_per_leaf"] = ROWS_PER_LEAF[cell.kind]
        k["rows"] += ROWS_PER_LEAF[cell.kind]
    clean = sum(k["rows"] for d in by_domain.values() for k in d.values())
    recall = RECALL_ROWS_PER_DOMAIN * len(by_domain)
    return {
        "audit_dir": str(AUDIT_DIR),
        "judge_config": str(JUDGE_CONFIG),
        "lr": plan.SOFT_HEADLINE_LR,
        "fifths": N_FIFTHS,
        "by_domain": by_domain,
        "leaves": len(headline_cells()),
        CLEAN_TASK: clean,
        f"{RECALL_TASK}_max": recall,
        "rows_max": clean + recall,
        "clean_shards": N_CLEAN_SHARDS,
    }


def run_dir_rel(leaf: soft.SoftLeaf) -> str:
    """Leaf run dir relative to the soft output root (the row-id identity)."""
    return soft.leaf_dir(leaf).relative_to(soft.OUTPUT_ROOT).as_posix()


def checked_rows(
    raw_rows: Iterable[Mapping[str, object]],
    summary: Mapping[str, object],
    n_steps: int,
) -> tuple[list[Mapping[str, object]], js.DedupeReport]:
    """On-policy rows of one leaf, last writer kept, census-checked.

    Args:
        raw_rows: ``rollouts_sample.jsonl`` records in file order.
        summary: The leaf's ``rollouts_summary.json``.
        n_steps: Optimizer steps the leaf trained.

    Returns:
        Rows sorted by ``(step, batch_idx)`` and the dedupe report.

    Raises:
        ValueError: Not the strict screen, a logged step count other than
            `n_steps`, or a census that disagrees with `summary`.
    """
    screen = summary.get("rollout_screen")
    if screen != plan.SOFT_ROLLOUT_SCREEN:
        raise ValueError(f"rollout_screen {screen!r} != {plan.SOFT_ROLLOUT_SCREEN!r}")
    logged = int(summary["n_steps_logged"])
    if logged != n_steps:
        raise ValueError(f"n_steps_logged {logged} != planned {n_steps}")
    rows, report = js.dedupe_last_writer(
        row for row in raw_rows if bool(row.get("on_policy", True))
    )
    census = js.rollout_census(rows, strict=True)
    bad = js.census_mismatches(summary, census, strict=True)
    if bad:
        raise ValueError(f"census mismatch: {bad}")
    return rows, report


def draw_leaf(
    rows: Sequence[Mapping[str, object]],
    *,
    rel: str,
    n_steps: int,
    n_rows: int,
    surface_forms: Collection[str],
    seed: int = SAMPLING_SEED,
) -> LeafDraw:
    """Equal per-fifth reservoir draw of kept rows plus recall candidates.

    Args:
        rows: `checked_rows` output.
        rel: `run_dir_rel` of the leaf (seeds its RNG).
        n_steps: Step count that defines the fifths.
        n_rows: Kept rows to draw; must divide by `N_FIFTHS`.
        surface_forms: The target's surface forms (``forms.json``).
        seed: Plan seed.

    Returns:
        The `LeafDraw`; a fifth with fewer kept rows than its quota is drawn
        in full.
    """
    draw = js.sample_rollout_rows(
        rows,
        n_steps=n_steps,
        n_strata=N_FIFTHS,
        clean_per_quintile=js.clean_quota_per_quintile(n_rows, N_FIFTHS),
        flagged_budgets={},
        rng=js.make_leaf_rng(seed, rel),
    )
    sampled = [0] * N_FIFTHS
    for row in draw.clean:
        sampled[row.step_quintile] += 1
    forms = {f.lower() for f in surface_forms}
    recall = tuple(
        _sampled(row, n_steps)
        for row in rows
        if bool(row["flagged"])
        and row.get("matched_rule") == RECALL_RULE
        and str(row.get("matched_form") or "").lower() in forms
    )
    return LeafDraw(draw.clean, draw.clean_pop_by_quintile, tuple(sampled), recall)


def row_pair(
    task: str,
    rel: str,
    row: js.SampledRollout,
    *,
    kind: str,
    target: str,
    stratum: str,
    weight: float,
) -> Pair:
    """Blind ``rollout_judge`` queue row and its identity key.

    Returns:
        ``(queue, key)``: queue = ``row_id, task, ask_animal, ask_category,
        prompt, completion``; key = `judge_sampling.KEYS_FIELDS_WITH_RULE`
        with ``arm`` = `kind`.
    """
    rid = js.row_id(task, rel, row.step, row.batch_idx, target)
    queue = {
        "row_id": rid,
        "task": task,
        "ask_animal": target,
        "ask_category": plan.domain_of(target).noun,
        "prompt": row.prompt,
        "completion": row.completion,
    }
    key = js.make_key_row(
        row_id_value=rid,
        task=task,
        source="rollout",
        run_dir=rel,
        animal=target,
        arm=kind,
        step=row.step,
        step_quintile_value=row.step_quintile,
        batch_idx=row.batch_idx,
        flagged=row.flagged,
        matched_form=row.matched_form,
        ask_animal=target,
        is_decoy=False,
        stratum=stratum,
        sample_weight=weight,
        doc_id=None,
        register=None,
        matched_rule=row.matched_rule,
        emit_rule=True,
    )
    return queue, key


def leaf_pairs(rel: str, draw: LeafDraw, *, kind: str, target: str) -> list[Pair]:
    """`CLEAN_TASK` pairs of one leaf; weights restore each fifth's kept share."""
    weights = js.realised_stratum_weights(
        {q: n for q, n in enumerate(draw.kept_by_fifth) if n},
        {q: n for q, n in enumerate(draw.sampled_by_fifth) if n},
    )
    return [
        row_pair(
            CLEAN_TASK,
            rel,
            row,
            kind=kind,
            target=target,
            stratum=f"q{row.step_quintile}",
            weight=float(weights[row.step_quintile]),
        )
        for row in draw.kept
    ]


def recall_sample(
    candidates: Sequence[tuple[str, str, str, js.SampledRollout]],
    domain: str,
    *,
    k: int = RECALL_ROWS_PER_DOMAIN,
    seed: int = SAMPLING_SEED,
) -> list[Pair]:
    """Seeded simple random sample of up to `k` recall candidates of a domain.

    Args:
        candidates: ``(rel, kind, target, row)`` over the domain's leaves.
        domain: Domain id (seeds the draw).
        k: Cap.
        seed: Plan seed.

    Returns:
        `RECALL_TASK` pairs, independent of candidate input order.
    """
    ordered = sorted(candidates, key=lambda c: (c[0], c[3].step, c[3].batch_idx))
    picked = js.subsample_indices(
        len(ordered), k, js.task_rng(seed, f"{RECALL_TASK}|{domain}")
    )
    return [
        row_pair(
            RECALL_TASK,
            rel,
            row,
            kind=kind,
            target=target,
            stratum=RECALL_STRATUM,
            weight=1.0,
        )
        for rel, kind, target, row in (ordered[i] for i in picked)
    ]


def shard_pairs(
    pairs: Sequence[Pair], n_shards: int, task: str, *, seed: int = SAMPLING_SEED
) -> list[list[Pair]]:
    """Shuffle `pairs` (seeded per task) and deal them round-robin into shards.

    Returns:
        ``min(n_shards, len(pairs))`` non-empty shards covering every pair once.

    Raises:
        ValueError: `pairs` is empty or a ``row_id`` repeats.
    """
    if not pairs:
        raise ValueError(f"{task}: no rows to shard")
    js.assert_unique_row_ids((q for q, _ in pairs), label=task)
    ordered = sorted(pairs, key=lambda p: p[0]["row_id"])
    js.task_rng(seed, task).shuffle(ordered)
    n = min(n_shards, len(ordered))
    return [ordered[i::n] for i in range(n)]


def join_judgments(
    keys: pd.DataFrame, judgments: pd.DataFrame, task: str
) -> pd.DataFrame:
    """Keys with their judgment, joined on the unique ``row_id``.

    Raises:
        ValueError: A ``row_id`` repeats, a judgment has no key, or a key has
            no judgment (judging incomplete).
    """
    for name, frame in (("keys", keys), ("judgments", judgments)):
        if frame.empty or frame["row_id"].duplicated().any():
            raise ValueError(f"{task}: {name} empty or with repeated row_id")
    drop = [c for c in judgments.columns if c in keys.columns and c != "row_id"]
    merged = keys.merge(
        judgments.drop(columns=drop), on="row_id", how="outer", indicator=True
    )
    orphans = int((merged["_merge"] == "right_only").sum())
    unjudged = int((merged["_merge"] == "left_only").sum())
    if orphans or unjudged:
        raise ValueError(
            f"{task}: {orphans} judgments without a key, {unjudged} keys unjudged"
        )
    return merged.drop(columns="_merge")


def leaf_frame(records: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    """Leaf table from the plan manifest's ``leaves`` records."""
    cols = ["run_dir", "key", "kind", "domain", "target", "r", "g", "kept_by_fifth"]
    return pd.DataFrame([{c: rec[c] for c in cols} for rec in records], columns=cols)


def fifth_cells(clean: pd.DataFrame, leaves: pd.DataFrame) -> pd.DataFrame:
    """Leaf x fifth cells: kept population, valid judged rows, yes count.

    Args:
        clean: Judged `CLEAN_TASK` rows with ``run_dir``, ``step_quintile``,
            ``judge_valid``, ``judge_yes``.
        leaves: `leaf_frame` output.

    Returns:
        `judge_claims.CELL_COLUMNS` (``arm`` = kind, ``animal`` = target)
        plus ``domain``, for every fifth with kept rows.
    """
    cells = judge_claims.leaf_quintile_cells(
        clean,
        {r.run_dir: list(r.kept_by_fifth) for r in leaves.itertuples()},
        dict(zip(leaves["run_dir"], leaves["kind"], strict=True)),
        dict(zip(leaves["run_dir"], leaves["target"], strict=True)),
    )
    cells["domain"] = cells["animal"].map(plan.TARGET_DOMAIN)
    return cells


def leaf_table(
    cells: pd.DataFrame, leaves: pd.DataFrame, clean: pd.DataFrame
) -> pd.DataFrame:
    """One row per leaf: kept, judged, valid and yes counts, fifth-weighted rate."""
    rates = judge_claims.leaf_rates(cells).drop(columns=["arm", "animal"])
    judged = clean.groupby("run_dir").agg(
        n_judged=("row_id", "size"),
        n_parse_fail=("judge_valid", lambda v: int((~v.astype(bool)).sum())),
    )
    out = leaves.drop(columns="kept_by_fifth").merge(rates, on="run_dir", how="left")
    out = out.merge(judged, on="run_dir", how="left")
    return out.rename(
        columns={"n_pop": "n_kept", "n_sampled": "n_valid", "rate": "flag_rate"}
    )


def arm_contrast(
    cells: pd.DataFrame, *, n_boot: int = N_BOOT, seed: int = BOOTSTRAP_SEED
) -> dict:
    """Pooled fifth-weighted flag rate per kind and treatment minus placebo.

    Rates are ``sum N y / n / sum N`` over leaf x fifth cells with a valid
    sample. Intervals come from one leaf-cluster bootstrap within kind x target
    strata: rates at `RATE_CONFIDENCE`, the difference at `DIFF_CONFIDENCE`
    against ``+-MARGIN``.

    Args:
        cells: `fifth_cells` rows (one scope: a domain or all).
        n_boot: Bootstrap replicates.
        seed: Bootstrap seed.

    Returns:
        ``{"status", "treat", "placebo", "difference"}`` (each a
        `BootstrapEstimate` dict, whose ``n_valid`` counts finite replicates),
        or ``{"status": INDETERMINATE, "reason"}`` when a kind has no cells.
    """
    sub = cells.loc[cells["n_sampled"] > 0].reset_index(drop=True)
    if set(sub["arm"]) != set(soft.KINDS):
        return {"status": INDETERMINATE, "reason": "a kind has no judged cells"}
    placebo = (sub["arm"] == soft.KIND_PLACEBO).to_numpy(dtype=np.int64)
    num = (sub["n_pop"] * sub["n_yes"] / sub["n_sampled"]).to_numpy(dtype=np.float64)
    den = sub["n_pop"].to_numpy(dtype=np.float64)

    def statistic(idx: np.ndarray) -> np.ndarray:
        nums = np.bincount(placebo[idx], weights=num[idx], minlength=2)
        dens = np.bincount(placebo[idx], weights=den[idx], minlength=2)
        rates = nums / dens
        return np.array([rates[0], rates[1], rates[0] - rates[1]])

    observed, draws = boot.stratified_cluster_bootstrap_draws(
        statistic,
        (sub["arm"] + "|" + sub["animal"]).to_numpy(),
        sub["run_dir"].to_numpy(),
        n_boot=n_boot,
        seed=seed,
    )
    diff = boot.summarize_draws(observed[2], draws[:, 2], confidence=DIFF_CONFIDENCE)
    eq = boot.bootstrap_equivalence(diff, bound_low=-MARGIN, bound_high=MARGIN)
    out: dict = {"status": "ok"}
    for i, kind in enumerate((soft.KIND_TREAT, soft.KIND_PLACEBO)):
        rows = sub.loc[sub["arm"] == kind]
        est = boot.summarize_draws(observed[i], draws[:, i], confidence=RATE_CONFIDENCE)
        out[kind] = {
            **asdict(est),
            "leaves": int(rows["run_dir"].nunique()),
            "n_kept": int(rows["n_pop"].sum()),
            "n_judged_valid": int(rows["n_sampled"].sum()),
            "n_yes": int(rows["n_yes"].sum()),
        }
    out["difference"] = {
        **asdict(diff),
        "margin": MARGIN,
        "within_margin": bool(eq.equivalent),
        "excludes_zero": bool(diff.ci_low > 0 or diff.ci_high < 0),
    }
    return out


def recall_by_domain(recall: pd.DataFrame) -> dict[str, dict]:
    """Judge yes-rate on the recall probe per domain (Wilson, `RATE_CONFIDENCE`).

    Args:
        recall: Judged `RECALL_TASK` rows with ``animal``, ``judge_valid``,
            ``judge_yes``.

    Returns:
        Domain -> ``{"n_rows", "n_valid", "n_yes", "recall", "ci_low",
        "ci_high"}``; rate fields are ``None`` without a valid row.
    """
    out: dict[str, dict] = {}
    domain = recall["animal"].map(plan.TARGET_DOMAIN)
    for name in plan.DOMAINS:
        rows = recall.loc[domain == name]
        valid = rows.loc[rows["judge_valid"].astype(bool)]
        n_yes = int(valid["judge_yes"].eq(True).sum())
        doc: dict = {"n_rows": len(rows), "n_valid": len(valid), "n_yes": n_yes}
        if len(valid):
            rate = binomial_rate(n_yes, len(valid), confidence=RATE_CONFIDENCE)
            doc.update(recall=rate.estimate, ci_low=rate.ci_low, ci_high=rate.ci_high)
        else:
            doc.update(recall=None, ci_low=None, ci_high=None)
        out[name] = doc
    return out


def parse_report(frame: pd.DataFrame) -> dict:
    """Judged rows, valid rows and parse-failure rate of one task."""
    n = len(frame)
    valid = int(frame["judge_valid"].astype(bool).sum()) if n else 0
    return {
        "n_judged": n,
        "n_valid": valid,
        "parse_failure_rate": (n - valid) / n if n else None,
    }


def n6_doc(
    cells: pd.DataFrame,
    clean: pd.DataFrame,
    recall: pd.DataFrame,
    *,
    n_boot: int = N_BOOT,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Claim N6 (descriptive): per-domain and pooled contrasts, recall, parse."""
    return {
        "claim": "N6",
        "descriptive": True,
        "lr": plan.SOFT_HEADLINE_LR,
        "margin_pp": MARGIN * 100,
        "diff_confidence": DIFF_CONFIDENCE,
        "rate_confidence": RATE_CONFIDENCE,
        "n_boot": n_boot,
        "bootstrap_seed": seed,
        "by_domain": {
            d: arm_contrast(cells.loc[cells["domain"] == d], n_boot=n_boot, seed=seed)
            for d in plan.DOMAINS
        },
        "pooled": arm_contrast(cells, n_boot=n_boot, seed=seed),
        "recall": recall_by_domain(recall),
        "parse": {CLEAN_TASK: parse_report(clean), RECALL_TASK: parse_report(recall)},
        "cells_without_valid_sample": int((cells["n_sampled"] == 0).sum()),
    }


def _sampled(row: Mapping[str, object], n_steps: int) -> js.SampledRollout:
    step = int(row["step"])
    form, rule = row.get("matched_form"), row.get("matched_rule")
    return js.SampledRollout(
        step=step,
        batch_idx=int(row["batch_idx"]),
        prompt=str(row["prompt"]),
        completion=str(row["completion"]),
        matched_form=None if form is None else str(form),
        flagged=bool(row["flagged"]),
        step_quintile=js.step_quintile(step, n_steps, N_FIFTHS),
        matched_rule=None if rule is None else str(rule),
    )
