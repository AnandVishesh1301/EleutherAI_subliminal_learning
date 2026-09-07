"""Build notebooks/gkd_ckpt_grid2_bear.ipynb via nbformat."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from loguru import logger
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from subliminal_learning.campaign.ckptgrid2_plan import RUN_SET_ID

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_OUT = _REPO / "notebooks" / "gkd_ckpt_grid2_bear.ipynb"

CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        f"""# Bear checkpoint-grid stage 2 — `{RUN_SET_ID}`

End-to-end analysis of campaign `{RUN_SET_ID}` on
PolyPythia-160M. Behaviour-only (no CKA / geometry). Frozen teacher
recipe `ep5_lr5e-5_r128a128` except the swept dose epochs; frozen
student recipe `lmb1.0_b1.0_lora_lr4e-4_e1_r8a16` except the HP arm.
Animals: bear (primary) and a cat 7×7 mini-grid. Teachers share this
run-set id; reference run sets are joined at analysis only.

This notebook is **read-only** on cached `results/` + `logs/`. It does
**not** retrain, does **not** re-run bootstraps, and does **not** write
under `output_root`. Paths resolve from `configs/paths/paths_local.yaml`
via `analysis.setup.bootstrap`. Every filename and rung comes from
`subliminal_learning.campaign.ckptgrid2_plan` — nothing is hardcoded
here.

Pre-registration: `docs/PREREG_CKPT_XGRID2.md`. Constants:
`subliminal_learning.campaign.ckptgrid2_plan`. Claim math (not imported;
already persisted in `claims.json`):
`subliminal_learning.campaign.ckptgrid2_metrics`. Plan:
`.cursor/plans/checkpoint_grid_stage_2_c1e8a4af.plan.md`.

**How to run**
```bash
uv sync --group analysis
uv run jupyter lab
```
Open `notebooks/gkd_ckpt_grid2_bear.ipynb` and run all cells
top-to-bottom. Do **not** run this notebook until finalize has written
`results/` — missing artifacts raise.

### Reading guide

- `delta = p_target − matched control mean`. Controls match on
  `(student_base, student_step, gkd_tag)`. Difference of two
  probabilities, so roughly (−1, 1); `0` = no subliminal transfer above
  the untrained-base-teacher control.
- Agg rows are **already means over the rotated-diagonal leaves**. Never
  re-average them as if they were leaves. Use `level='leaf'` when you
  need per-leaf spread.
- Primary endpoint is `test` / `prob`. The sampled twin is `test_sample`
  / `sample`. Never pool prompt sets into a headline figure.
- Cell labels: `SI_SS` same-init/same-stream, `SI_DS`
  same-init/different-stream, `DI_DS` different-init/different-stream,
  `DI_SS` different-init/same-stream (wave 2 only; reserved panel).
- Drop rows with `degenerate == True` from efficiency / penalty /
  ridge figures, and **print how many you dropped** rather than silently
  filtering.
- No silent NaN fills. Missing grid cells stay NaN and must read as
  "unrun", not 0. A partial finalize is visible via `coverage.json`.
- Each figure is followed by a **Reading this figure** cell. Conclusions
  stay `PENDING_FRONTIER_VISUAL_REVIEW` until a rendered figure exists.
  Final claim statuses are in **§14** from `claims.json`.

### What this notebook covers

| § | Topic |
|---|---|
| 0 | Inventory & coverage vs plan §10b.3 |
| 1 | Provenance (state, dose gate, escalation, prefetch, claims) |
| 2 | Fig1 — teacher dose curves |
| 3 | Fig2 — dose–response and overshoot |
| 4 | Fig3 — family heatmaps (four cell types) |
| 5 | Fig4 — ridge gain per teacher column |
| 6 | Fig5 — HP × rung surfaces (delta and PPL) |
| 7 | Fig6 — HP off-diagonal probes |
| 8 | Fig7 — receptivity onset (incl. rung-512) |
| 9 | Fig8 — cross-init penalty map |
| 10 | Fig9 — emissivity / receptivity trend |
| 11 | Fig10 — cat 7×7 beside the bear diagonal |
| 12 | Fig11 — metric concordance (primary vs sampled twin) |
| 13 | Fig12 — perplexity and quarantine |
| 14 | Fig13 — claims board + takeaways from frames |
| 15 | How to extend |

### Explicit exclusions

- CKA / geometry / probes / weight-space.
- Wave-2 `weight-seed*` treatments until `seed_audit.json` is `passed`.
- Re-deriving claim statistics (bootstrap / TOST / ridge CIs). Those
  live in `claims.json`; this notebook only displays them.
- Training, Hub downloads, or writes under `output_root`.

### Metric cheat-sheet

Formulas match `ckptgrid2_metrics` and prereg §7. Ranges are
theoretical, not this run’s numbers (those print from CSVs below).

| ID | Answers | Values | Calculation | Example reading | Notes |
|---|---|---|---|---|---|
| `delta` | Did GKD from this teacher raise target-animal preference above a matched **base-teacher** control? | Difference of two probabilities: theoretically (−1, 1). | `p_target − control_mean`. Control matched on `(student_base, student_step, gkd_tag)`. Primary `test`/`prob`. | `0` = no SL above control. Positive = more last-token p(target) than control. | Agg rows are already leaf means. Never pool prompt sets. |
| `teacher_p_target` | How strong is the teacher’s target preference at this `(rung, epochs)`? | [0, 1] | Per-cell mean of teacher `p_target` over ftseeds (`teacher_dose.csv`). | `0` = no preference. Near 1 = almost all mass on the target token. | Dose axis for Fig1–Fig2. ep5 is the frozen reference. |
| `overshoot` | Did the student exceed its teacher? | Difference of two probabilities: theoretically (−2, 2). | `delta − teacher_p_target`. | `0` = student matches teacher dose. Positive = student stronger than teacher. | Identity line on Fig2 is `delta == teacher_p_target`. |
| `ridge_gain` | Does the matched-rung cell beat the rest of that teacher column? | Difference of two deltas. | `delta(diag) − mean(off-diagonal in the column)`. | Positive = matched checkpoint wins that column. | Split by era and directed rectangle. CIs are persisted, not recomputed. |
| `penalty` | How large is the cross-init deficit vs the same-init/same-stream cell? | (−∞, 1]; NaN if the reference delta is ≤ 0. | `1 − delta_DI_DS / delta_SI_SS`. | `0` = cross equals same-init. Positive = cross weaker. | Fig8. Drop `degenerate`. |
| `tau − sigma` | At a matched rung, is original more emissive than receptive? | Unbounded. | Additive fit `delta_ij = μ + τ_i + σ_j`; plotted for `base == original`. | `0` = symmetric. Positive = original emits more than it receives. | Fig9. |
| `perplexity_ratio` | Did GKD blow up the student’s Pile PPL vs its own base? | (0, ∞). `1` = unchanged. | Student PPL / untrained-base PPL. Collapse when `> QUARANTINE_PPL_RATIO_MAX`. | Ratio `> 5` is the leaf quarantine bound. | Labelled, never silently dropped. |
| onset floor | First student rung that clears control + max(2·SD, 0.02) for two consecutive rungs. | A scored student rung, or censored. | Persisted in `onset.csv`. | Earliest scored rung = lower bound on detectability. | Rung-512 rows are the cross-init floor check. |
""",
    ),
    ("markdown", "## Setup\n"),
    (
        "code",
        r'''from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.patches import Patch, Rectangle

from subliminal_learning.analysis import plots as aplots
from subliminal_learning.analysis.setup import bootstrap, set_plot_style
from subliminal_learning.campaign import ckptgrid2_plan as plan

ROOT, OUTPUT_ROOT = bootstrap()
set_plot_style(dpi=115, hide_spines=True)

PROMPT_SET = plan.PRIMARY_PROMPT_SET
ARM = "all"
SHOW_QUARANTINE = True
ROBUSTNESS_SETS = list(plan.EVAL_PROMPT_SETS)

TREE = OUTPUT_ROOT / "gkd_students" / plan.RUN_SET_ID
RES = TREE / "results"
LOGS = TREE / "logs"
TEACHER_TREE = OUTPUT_ROOT / "sdf_teachers" / plan.TEACHER_RUN_SET_ID

assert TREE.exists(), f"missing behaviour tree: {TREE}"
assert RES.exists(), f"missing results/: {RES}"
assert LOGS.exists(), f"missing logs/: {LOGS}"
assert "/mnt/ssd-1/" not in str(OUTPUT_ROOT), (
    f"refusing slow SSD-1 output_root: {OUTPUT_ROOT}"
)
if TREE.is_symlink():
    print("WARNING: TREE is a symlink →", TREE.resolve())
if TEACHER_TREE.is_symlink():
    print("WARNING: TEACHER_TREE is a symlink →", TEACHER_TREE.resolve())

print("ROOT         :", ROOT)
print("OUTPUT_ROOT  :", OUTPUT_ROOT)
print("TREE         :", TREE)
print("RES          :", RES)
print("LOGS         :", LOGS)
print("TEACHER_TREE :", TEACHER_TREE)
print("RUN_SET      :", plan.RUN_SET_ID)
print("teacher run  :", plan.TEACHER_RUN_SET_ID)
print("reference    :", dict(plan.REFERENCE_RUN_SETS))
print("teacher tag  :", plan.TEACHER_PARAM_TAG_LORA)
print("student tag  :", plan.FROZEN_GKD_TAG)
print("animals      :", plan.ANIMALS)
print("bases        :", list(plan.BASES))
print("joint rungs  :", plan.JOINT_RUNGS)
print("HP rungs     :", plan.HP_RUNGS)
print("dose rungs   :", plan.DOSE_RUNGS)
print("cat rungs    :", plan.CAT_RUNGS)
print("dose epochs  :", plan.DOSE_EPOCHS, "frozen", plan.TEACHER_EPOCHS_FROZEN)
print("HP configs   :")
for cfg in plan.HP_CONFIGS:
    rec = cfg.recipe
    print(
        f"  {cfg.tag}  lr={rec.lr:g} r={rec.lora_r} a={rec.lora_alpha} e={cfg.num_epochs:g}"
    )
print("probe configs:", [c.tag for c in plan.PROBE_HP_CONFIGS])
print("probe cells  :", plan.PROBE_CELLS)
print("prompt sets  :", plan.EVAL_PROMPT_SETS)
print("primary      :", plan.PRIMARY_PROMPT_SET, plan.PRIMARY_SCORING)
print("knobs        : PROMPT_SET=", PROMPT_SET, "ARM=", ARM)
print("             : SHOW_QUARANTINE=", SHOW_QUARANTINE)
print("             : ROBUSTNESS_SETS=", ROBUSTNESS_SETS)


def scope(df: pd.DataFrame, note: str = "") -> None:
    """Print a compact SCOPE line (does not assume teacher-sweep columns)."""
    bits = [f"n={len(df)}"]
    for col in (
        "arm",
        "level",
        "cell",
        "family",
        "set",
        "scoring",
        "animal",
        "student_base",
        "teacher_base",
        "student_step",
        "teacher_step",
        "n_epochs",
        "gkd_tag",
        "era",
        "source_run_set",
        "ftseed",
        "gseed",
    ):
        if col in df.columns:
            bits.append(f"{col}={df[col].nunique(dropna=False)}")
    line = "SCOPE: " + " | ".join(bits)
    if note:
        line += f" | {note}"
    print(line)


def step_x(step) -> float:
    """Log-x helper: plot step 0 at 0.5 so it is visible on a log axis."""
    s = float(step)
    return 0.5 if s <= 0 else s


def add_step_x(df: pd.DataFrame, src: str = "step") -> pd.DataFrame:
    out = df.copy()
    out["x"] = [step_x(v) for v in out[src]]
    return out


def claim_record(cid: str) -> dict:
    claims = CLAIMS.get("claims") or CLAIMS.get("records") or []
    for rec in claims:
        if rec.get("id") == cid:
            return rec
    raise KeyError(cid)


def grid_pivot(
    frame: pd.DataFrame,
    *,
    value: str = "delta",
    index: str = "student_step",
    columns: str = "teacher_step",
    index_order=None,
    column_order=None,
) -> pd.DataFrame:
    """One value per (index, columns). Gaps stay NaN (unrun, not 0)."""
    if frame.empty:
        idx = list(index_order or [])
        cols = list(column_order or [])
        return pd.DataFrame(np.nan, index=idx, columns=cols)
    work = frame.copy()
    work[index] = work[index].astype(int)
    work[columns] = work[columns].astype(int)
    piv = work.pivot_table(
        index=index, columns=columns, values=value, aggfunc="mean"
    )
    if index_order is not None:
        piv = piv.reindex(index=list(index_order))
    if column_order is not None:
        piv = piv.reindex(columns=list(column_order))
    return piv


def diverging_norm(values, vcenter: float = 0.0) -> TwoSlopeNorm:
    """Shared diverging colour scale with 0 as the neutral colour."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    lo = float(np.min(finite)) if finite.size else -0.1
    hi = float(np.max(finite)) if finite.size else 0.1
    pad = 1e-6
    if lo >= vcenter:
        lo = vcenter - pad
    if hi <= vcenter:
        hi = vcenter + pad
    return TwoSlopeNorm(vmin=lo, vcenter=vcenter, vmax=hi)


def drop_degenerate(df: pd.DataFrame, note: str) -> pd.DataFrame:
    """Drop `degenerate` rows and print how many were removed."""
    if "degenerate" not in df.columns:
        print(f"dropped 0 of {len(df)} {note} rows (no degenerate column)")
        return df.copy()
    n_drop = int(df["degenerate"].astype(bool).sum())
    print(f"dropped {n_drop} of {len(df)} {note} rows (degenerate)")
    return df.loc[~df["degenerate"].astype(bool)].copy()


def fmt_num(value, nd: int = 4) -> str:
    """Round a claim statistic for an in-figure label."""
    try:
        return f"{float(value):.{nd}f}"
    except (TypeError, ValueError):
        return str(value)


def fmt_ci(pair, nd: int = 4) -> str:
    """Format a two-element CI as ``[lo, hi]`` at `nd` decimals."""
    if pair is None:
        return "n/a"
    try:
        lo, hi = pair
    except (TypeError, ValueError):
        return str(pair)
    return f"[{fmt_num(lo, nd)}, {fmt_num(hi, nd)}]"


def quarantine_pairs() -> set[tuple[int, int]]:
    if QUAR.empty:
        return set()
    s_col = "student_step" if "student_step" in QUAR.columns else "step"
    t_col = "teacher_step" if "teacher_step" in QUAR.columns else None
    if t_col is None:
        return set()
    return set(
        zip(QUAR[s_col].astype(int), QUAR[t_col].astype(int), strict=True)
    )


def mark_diagonal(ax, index, columns, rungs) -> None:
    idx = {int(v): i for i, v in enumerate(index)}
    cols = {int(v): j for j, v in enumerate(columns)}
    for step in rungs:
        if step in idx and step in cols:
            i, j = idx[step], cols[step]
            ax.add_patch(
                Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="k", lw=1.4
                )
            )


def mark_quarantine(ax, index, columns) -> int:
    idx = {int(v): i for i, v in enumerate(index)}
    cols = {int(v): j for j, v in enumerate(columns)}
    n_mark = 0
    for student_step, teacher_step in quarantine_pairs():
        if student_step in idx and teacher_step in cols:
            i, j = idx[student_step], cols[teacher_step]
            ax.add_patch(
                Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    fill=False,
                    hatch="///",
                    edgecolor="0.2",
                    lw=0.6,
                )
            )
            n_mark += 1
    return n_mark


def hp_label(tag: str) -> str:
    """Short readable label for a persisted `gkd_tag`."""
    for cfg in plan.HP_CONFIGS:
        if cfg.tag == tag:
            rec = cfg.recipe
            return (
                f"lr={rec.lr:g} r={rec.lora_r} "
                f"a={rec.lora_alpha} e={cfg.num_epochs:g}"
            )
    if tag == plan.FROZEN_GKD_TAG:
        rec = plan.FROZEN_RECIPE
        return (
            f"lr={rec.lr:g} r={rec.lora_r} "
            f"a={rec.lora_alpha} e={plan.FROZEN_EPOCHS:g} (frozen)"
        )
    return tag


def designed_counts() -> dict[str, int]:
    """§10b.3 designed totals, derived from plan constants."""
    n_joint = len(plan.JOINT_RUNGS)
    n_off = n_joint * (n_joint - 1)
    n_rect_pairings = 5
    n_early_cross = 5
    n_dose = len(plan.DOSE_RUNGS) * len(plan.DOSE_EPOCHS)
    n_dose_teacher = (len(plan.DOSE_RUNGS) - 1) * len(plan.DOSE_EPOCHS)
    n_hp = len(plan.HP_CONFIGS) * len(plan.HP_RUNGS)
    n_probe = len(plan.PROBE_HP_CONFIGS) * len(plan.PROBE_CELLS)
    n_rect = n_rect_pairings * n_off
    n_rect512 = len(plan.RECT512_RECTANGLES) * n_joint
    n_cat = len(plan.CAT_RUNGS) * (len(plan.CAT_RUNGS) - 1)
    n_treatment = (
        n_dose
        + n_dose_teacher
        + n_hp
        + n_probe
        + n_rect
        + n_rect512
        + n_cat
        + n_early_cross
    )
    n_leaves = n_treatment * len(plan.GSEEDS)
    n_controls = (
        len(plan.HP_CONFIGS) * len(plan.HP_RUNGS) * len(plan.HP_CONTROL_GSEEDS)
        + len(plan.GSEEDS)
    )
    n_teacher_combos = len(plan.DOSE_RUNGS) * len(plan.DOSE_EPOCHS) + 2
    n_teacher = n_teacher_combos * len(plan.FTSEEDS)
    return {
        "treatment_cells": n_treatment,
        "treatment_leaves": n_leaves,
        "new_controls": n_controls,
        "teacher_leaves": n_teacher,
        "rung512_cells": n_rect512,
    }


def family_key(frame: pd.DataFrame) -> pd.Series:
    if "cell" in frame.columns:
        return frame["cell"].astype(str)
    if "family" in frame.columns:
        return frame["family"].astype(str)
    raise KeyError("need 'cell' or 'family' to split family heatmaps")


def agg_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if "level" not in frame.columns:
        return frame.copy()
    out = frame.loc[frame["level"].astype(str) == "agg"].copy()
    return out if not out.empty else frame.copy()


def primary_rows(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame
    if "set" in out.columns:
        out = out.loc[out["set"].astype(str) == PROMPT_SET]
    if "scoring" in out.columns:
        out = out.loc[out["scoring"].astype(str) == plan.PRIMARY_SCORING]
    return out.copy()
''',
    ),
    (
        "markdown",
        """## Load artifacts

One cell: every `plan.RESULTS_*` CSV / JSON, `coverage.json`,
`claims.json`, the log JSONs, and `CAMPAIGN_DONE` when present. Fail
loud if a required file is missing.
""",
    ),
    (
        "code",
        r"""def _read_csv(name: str) -> pd.DataFrame:
    path = RES / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _read_json(dirpath: Path, name: str) -> dict:
    path = dirpath / name
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _read_json_optional(dirpath: Path, name: str) -> dict | None:
    path = dirpath / name
    if not path.exists():
        print(f"optional missing: {path}")
        return None
    return json.loads(path.read_text())


RESULT_FILES = (
    plan.RESULTS_TEACHER_DOSE_CSV,
    plan.RESULTS_DOSE_RESPONSE_CSV,
    plan.RESULTS_HP_GRID_DELTA_CSV,
    plan.RESULTS_HP_CONTROLS_CSV,
    plan.RESULTS_HP_COLLAPSE_CSV,
    plan.RESULTS_RECT_DELTA_CSV,
    plan.RESULTS_RIDGE_GAIN_CSV,
    plan.RESULTS_PENALTY_MAP_CSV,
    plan.RESULTS_EMISSIVITY_FIT_CSV,
    plan.RESULTS_ONSET_CSV,
    plan.RESULTS_METRIC_CONCORDANCE_CSV,
    plan.RESULTS_CAT_GRID_DELTA_CSV,
    plan.RESULTS_BASE_PPL_CSV,
    plan.RESULTS_QUARANTINE_CSV,
    plan.RESULTS_COVERAGE_JSON,
    plan.RESULTS_CLAIMS_JSON,
)
for name in RESULT_FILES:
    assert (RES / name).exists(), f"missing {RES / name}"

for name in (
    plan.CAMPAIGN_STATE_FILE,
    plan.DOSE_GATE_REPORT,
    plan.ESCALATION_REPORT,
    plan.PREFETCH_MANIFEST,
):
    assert (LOGS / name).exists(), f"missing {LOGS / name}"

TEACHER_DOSE = _read_csv(plan.RESULTS_TEACHER_DOSE_CSV)
DOSE_RESP = _read_csv(plan.RESULTS_DOSE_RESPONSE_CSV)
HP_GRID = _read_csv(plan.RESULTS_HP_GRID_DELTA_CSV)
HP_CTRL = _read_csv(plan.RESULTS_HP_CONTROLS_CSV)
HP_COLLAPSE = _read_csv(plan.RESULTS_HP_COLLAPSE_CSV)
RECT = _read_csv(plan.RESULTS_RECT_DELTA_CSV)
RIDGE = _read_csv(plan.RESULTS_RIDGE_GAIN_CSV)
PENALTY = _read_csv(plan.RESULTS_PENALTY_MAP_CSV)
EMISSIVITY = _read_csv(plan.RESULTS_EMISSIVITY_FIT_CSV)
ONSET = _read_csv(plan.RESULTS_ONSET_CSV)
CONCORD = _read_csv(plan.RESULTS_METRIC_CONCORDANCE_CSV)
CAT = _read_csv(plan.RESULTS_CAT_GRID_DELTA_CSV)
BASE_PPL = _read_csv(plan.RESULTS_BASE_PPL_CSV)
QUAR = _read_csv(plan.RESULTS_QUARANTINE_CSV)
COVERAGE = _read_json(RES, plan.RESULTS_COVERAGE_JSON)
CLAIMS = _read_json(RES, plan.RESULTS_CLAIMS_JSON)
STATE = _read_json(LOGS, plan.CAMPAIGN_STATE_FILE)
DOSE_GATE = _read_json(LOGS, plan.DOSE_GATE_REPORT)
ESCALATION = _read_json(LOGS, plan.ESCALATION_REPORT)
PREFETCH = _read_json(LOGS, plan.PREFETCH_MANIFEST)
SEED_AUDIT = _read_json_optional(LOGS, plan.SEED_AUDIT_REPORT)
PREFLIGHT = _read_json_optional(LOGS, plan.PREFLIGHT_REPORT)

done_path = LOGS / plan.CAMPAIGN_DONE_FILE
if done_path.exists():
    DONE_TEXT = done_path.read_text().strip()
else:
    DONE_TEXT = None
    print("CAMPAIGN_DONE absent (partial finalize is allowed to omit it)")

ROB: dict[str, pd.DataFrame] = {}
rob_dir = RES / plan.RESULTS_ROBUSTNESS_DIR
if rob_dir.is_dir():
    for path in sorted(rob_dir.glob("*.csv")):
        ROB[path.stem] = pd.read_csv(path)
    print("robustness files:", sorted(ROB))
else:
    print("robustness dir absent:", rob_dir)

print("CAMPAIGN_DONE:", DONE_TEXT)
print("resolved RES files:")
for path in sorted(RES.iterdir()):
    if path.is_file():
        print(f"  {path.name:36s} {path.stat().st_size:9d} B")
""",
    ),
    (
        "markdown",
        """## 0. Inventory & coverage

**What this measures.** Disk vs plan §10b.3: 306 treatment cells, 918
treatment leaves, 52 new controls, 54 teacher leaves, 12 rung-512
cells. Those integers are **derived** from `ckptgrid2_plan` lists in
the next cell, not typed as literals. `coverage.json` is printed in
full so a partial finalize stays visible.
""",
    ),
    (
        "code",
        r"""expected = designed_counts()
print("designed (from plan constants):")
for key, val in expected.items():
    print(f"  {key:20s} {val}")

scope(RECT, "rect_delta")
scope(HP_GRID, "hp_grid_delta")
scope(DOSE_RESP, "dose_response")
scope(HP_CTRL, "hp_controls")
scope(CAT, "cat_grid_delta")
scope(TEACHER_DOSE, "teacher_dose")

rect_agg = agg_rows(primary_rows(RECT)) if not RECT.empty else RECT
hp_agg = agg_rows(primary_rows(HP_GRID)) if not HP_GRID.empty else HP_GRID
dose_agg = agg_rows(primary_rows(DOSE_RESP)) if not DOSE_RESP.empty else DOSE_RESP
cat_agg = agg_rows(primary_rows(CAT)) if not CAT.empty else CAT

n512 = 0
if not RECT.empty and "student_step" in RECT.columns:
    r512 = agg_rows(RECT)
    n512 = int(
        (
            r512["student_step"].astype(int) == plan.RECT512_STUDENT_RUNG
        ).sum()
    )
    if "level" in RECT.columns:
        leaf512 = RECT.loc[
            (RECT["level"].astype(str) == "leaf")
            & (RECT["student_step"].astype(int) == plan.RECT512_STUDENT_RUNG)
        ]
        print("rung-512 leaf rows", len(leaf512))
print("rung-512 agg rows on disk", n512, "designed", expected["rung512_cells"])

print("teacher_dose rows", len(TEACHER_DOSE), "designed teacher leaves", expected["teacher_leaves"])
print("hp_controls rows", len(HP_CTRL), "designed new controls", expected["new_controls"])

print("=== coverage.json (partial finalize must be visible) ===")
if isinstance(COVERAGE, dict):
    for key in (
        "designed",
        "present",
        "unrun_cells",
        "unrun",
        "tolerated_failed_cells",
        "failed_cells",
        "deadline_windows",
        "partial",
        "arms",
    ):
        if key in COVERAGE:
            print(f"{key}:")
            display(COVERAGE[key])
    extra = [k for k in COVERAGE if k not in {
        "designed", "present", "unrun_cells", "unrun",
        "tolerated_failed_cells", "failed_cells", "deadline_windows",
        "partial", "arms",
    }]
    if extra:
        print("other coverage keys:", extra)
        display({k: COVERAGE[k] for k in extra})
else:
    display(COVERAGE)

if not rect_agg.empty or not hp_agg.empty or not dose_agg.empty:
    print(
        "SCOPE summary: designed",
        expected["treatment_cells"],
        "cells /",
        expected["treatment_leaves"],
        "leaves /",
        expected["new_controls"],
        "new controls /",
        expected["teacher_leaves"],
        "teacher leaves /",
        expected["rung512_cells"],
        "rung-512 cells",
    )
else:
    print("WARNING: treatment frames are empty — partial or unrun")
""",
    ),
    (
        "markdown",
        """## 1. Provenance

**What this measures.** `campaign_state.json` transitions,
`dose_gate_report.json`, `hp_control_escalation.json`, prefetch-manifest
ref count, `seed_audit.json` when present, `CAMPAIGN_DONE`, and
`claims.json` identity fields. Numbers come from the next cell — not
from this markdown.
""",
    ),
    (
        "code",
        r"""transitions = STATE.get("transitions") or {}
state_rows = []
for name, payload in transitions.items():
    state_rows.append(
        {
            "transition": name,
            "at": (payload or {}).get("at"),
            "keys": sorted((payload or {}).keys()),
        }
    )
if state_rows:
    display(pd.DataFrame(state_rows))
else:
    print("campaign_state transitions:", STATE)

print("=== dose_gate_report ===")
print("keys:", sorted(DOSE_GATE.keys()) if isinstance(DOSE_GATE, dict) else type(DOSE_GATE))
display(DOSE_GATE)

print("=== hp_control_escalation ===")
print("keys:", sorted(ESCALATION.keys()) if isinstance(ESCALATION, dict) else type(ESCALATION))
display(ESCALATION)

n_refs = PREFETCH.get("n_refs")
refs = PREFETCH.get("refs") or PREFETCH.get("hub_refs") or []
print("prefetch n_refs:", n_refs, "refs list", len(refs))
print("prefetch extra keys:", sorted(k for k in PREFETCH if k not in {"n_refs", "refs", "hub_refs"}))

if SEED_AUDIT is None:
    print("seed_audit.json: not present (wave 1 is allowed to omit it)")
else:
    print("seed_audit passed:", SEED_AUDIT.get("passed"))
    print("seed_audit repos:", SEED_AUDIT.get("repos"))
    display(SEED_AUDIT.get("checks"))

print("CAMPAIGN_DONE:", DONE_TEXT)
print("claims.at:", CLAIMS.get("at"))
print("claims.run_set_id:", CLAIMS.get("run_set_id"))
print("claims.primary_endpoint:", CLAIMS.get("primary_endpoint"))
print("claims.partial:", CLAIMS.get("partial"))
print("claims.coverage:", CLAIMS.get("coverage"))
ids = [rec.get("id") for rec in (CLAIMS.get("claims") or CLAIMS.get("records") or [])]
print("claims ids:", ids)
if CLAIMS.get("run_set_id") is not None:
    assert CLAIMS.get("run_set_id") == plan.RUN_SET_ID, CLAIMS.get("run_set_id")
if ids:
    assert ids == list(plan.CLAIM_IDS), (ids, plan.CLAIM_IDS)
if PREFLIGHT is not None:
    print("preflight n_refs:", PREFLIGHT.get("n_refs"), "prereg:", PREFLIGHT.get("prereg"))
""",
    ),
    (
        "markdown",
        """## 2. Fig1 — teacher dose curves

**What this measures.** Original-base teacher target-animal preference
versus SDF epoch count, one series per dose rung. The frozen ep5
reference point is marked. This is the dose axis every later figure
normalizes against.
""",
    ),
    (
        "code",
        r"""scope(TEACHER_DOSE, "teacher_dose")
td = TEACHER_DOSE.copy()
if td.empty:
    raise RuntimeError("teacher_dose.csv is empty — cannot draw the dose axis")
if "set" in td.columns:
    td = td.loc[td["set"].astype(str) == PROMPT_SET]
if "scoring" in td.columns:
    td = td.loc[td["scoring"].astype(str) == plan.PRIMARY_SCORING]
if "base" in td.columns:
    td = td.loc[td["base"].astype(str) == "original"]
need = {"step", "n_epochs"}
missing = need - set(td.columns)
if missing:
    raise KeyError(f"teacher_dose missing {sorted(missing)}")
ycol = "p_target_mean" if "p_target_mean" in td.columns else "p_target"
if ycol not in td.columns:
    raise KeyError("teacher_dose needs p_target_mean or p_target")
td["step"] = td["step"].astype(int)
td["n_epochs"] = pd.to_numeric(td["n_epochs"], errors="coerce")
sdcol = "p_target_sd" if "p_target_sd" in td.columns else None

fig, ax = plt.subplots(figsize=(8.4, 4.6))
palette = sns.color_palette("viridis", n_colors=max(len(plan.DOSE_RUNGS), 1))
for color, rung in zip(palette, plan.DOSE_RUNGS, strict=False):
    sub = td.loc[td["step"] == int(rung)].sort_values("n_epochs")
    if sub.empty:
        continue
    yerr = sub[sdcol] if sdcol is not None else None
    ax.errorbar(
        sub["n_epochs"],
        sub[ycol],
        yerr=yerr,
        fmt="o-",
        color=color,
        capsize=3,
        label=f"teacher rung {rung}",
    )
ax.axvline(
    plan.TEACHER_EPOCHS_FROZEN,
    color="0.35",
    ls="--",
    lw=1.1,
    label=f"frozen reference ep{plan.TEACHER_EPOCHS_FROZEN}",
)
ref = td.loc[td["n_epochs"] == plan.TEACHER_EPOCHS_FROZEN]
if not ref.empty:
    ax.scatter(
        ref["n_epochs"],
        ref[ycol],
        s=90,
        marker="D",
        facecolors="none",
        edgecolors="C3",
        linewidths=1.6,
        zorder=4,
        label=f"ep{plan.TEACHER_EPOCHS_FROZEN} reference points",
    )
ax.set_xlabel("teacher SDF epochs")
ax.set_ylabel("teacher last-token p(target animal)  [test / prob]")
ax.set_title(
    "Fig1  teacher dose curves\n"
    "PolyPythia-160M · original init/stream · bear · SDF LoRA r=128 α=128 lr=5e-5\n"
    f"rungs={list(plan.DOSE_RUNGS)} · filter: {PROMPT_SET}/{plan.PRIMARY_SCORING} · original base"
)
ax.legend(loc="best", fontsize=8)
ax.set_xticks(
    sorted(set(plan.DOSE_EPOCHS) | {plan.TEACHER_EPOCHS_FROZEN})
)
plt.show()
""",
    ),
    (
        "markdown",
        """### Reading this figure

**Question answered:** How does the original-base teacher’s target-animal last-token preference change with SDF epoch count at each dose rung, and where does the frozen five-epoch reference sit?
**Conclusion from the plot:** PENDING_FRONTIER_VISUAL_REVIEW

**Axes.** x = teacher SDF epochs (swept `DOSE_EPOCHS` plus the frozen
reference). y = mean teacher last-token p(target animal) on
`test`/`prob`, one line per dose rung; error bars are the ftseed SD
when present. Vertical dashed line and open diamonds mark the frozen
ep5 reference.

**How the metric is calculated.** `teacher_dose.csv` from
`build_teacher_dose`: per `(base, rung, n_epochs)` mean of teacher
`p_target` over ftseeds. This notebook does not re-aggregate leaves.
""",
    ),
    (
        "markdown",
        """## 3. Fig2 — dose–response and overshoot

**What this measures.** Student subliminal Δ preference versus teacher
dose, coloured by student rung, with the identity
`delta == teacher_p_target` so overshoot is visible. The dose-matched
rung-coefficient / TOST annotation is read from the persisted claim
record (not recomputed).
""",
    ),
    (
        "code",
        r"""scope(DOSE_RESP, "dose_response")
h3 = claim_record("H3")
dr = drop_degenerate(primary_rows(DOSE_RESP), "dose_response")
if "animal" in dr.columns:
    dr = dr.loc[dr["animal"].astype(str) == "bear"]
need = {"delta", "teacher_p_target"}
missing = need - set(dr.columns)
if missing:
    raise KeyError(f"dose_response missing {sorted(missing)}")
rung_col = "student_step" if "student_step" in dr.columns else "teacher_step"
stat = h3.get("statistic") or {}

fig, ax = plt.subplots(figsize=(8.6, 5.2))
if dr.empty:
    ax.text(0.5, 0.5, "dose_response empty", ha="center", va="center")
else:
    sc = ax.scatter(
        dr["teacher_p_target"],
        dr["delta"],
        c=dr[rung_col].astype(float),
        cmap="viridis",
        s=36,
        alpha=0.85,
        edgecolors="0.2",
        linewidths=0.3,
    )
    fig.colorbar(sc, ax=ax, label="student pretraining step (dose rung)")
    lo = float(np.nanmin([dr["teacher_p_target"].min(), dr["delta"].min(), 0.0]))
    hi = float(np.nanmax([dr["teacher_p_target"].max(), dr["delta"].max(), 0.1]))
    pad = 0.05 * (hi - lo if hi > lo else 0.1)
    ax.plot(
        [lo - pad, hi + pad],
        [lo - pad, hi + pad],
        "k--",
        lw=1.1,
        label="identity  (student Δ = teacher p(target); overshoot above)",
    )
ax.axhline(0.0, color="0.6", lw=0.8)
ax.axvline(0.0, color="0.6", lw=0.8)
ax.set_xlabel("teacher last-token p(target animal)  [test / prob]")
ax.set_ylabel("subliminal Δ preference  (p_target − matched control mean)")
ax.set_title(
    "Fig2  dose–response and overshoot\n"
    "Does a stronger teacher produce more transfer, and do students exceed their teacher?\n"
    "PolyPythia-160M · bear · original←original diagonal · frozen student recipe\n"
    f"rungs={list(plan.DOSE_RUNGS)} · epochs={list(plan.DOSE_EPOCHS)}+{plan.TEACHER_EPOCHS_FROZEN} · {PROMPT_SET}/{plan.PRIMARY_SCORING}"
)
ax.text(
    0.02,
    0.98,
    "dose-matched maturity (H3)\nrung coef={coef}\nCI={ci}\nTOST equivalent={tost}\nband ±{band}".format(
        coef=fmt_num(stat.get("rung_coef")),
        ci=fmt_ci(stat.get("ci")),
        tost=stat.get("tost_equivalent"),
        band=plan.DOSE_TOST_BOUND,
    ),
    transform=ax.transAxes,
    va="top",
    fontsize=8,
    bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
)
ax.legend(loc="lower right", fontsize=8)
plt.show()
print("H3 record:", h3)
""",
    ),
    (
        "markdown",
        """### Reading this figure

**Question answered:** Does student subliminal Δ preference track teacher last-token p(target), and do students overshoot the identity line `delta = teacher_p_target`, after the dose-matched rung coefficient is annotated from the persisted claim record?
**Conclusion from the plot:** PENDING_FRONTIER_VISUAL_REVIEW

**Axes.** x = teacher last-token p(target animal). y = subliminal Δ
preference (`p_target − matched control mean`). Colour = student
pretraining step. Dashed identity = `delta == teacher_p_target`
(points above the line are overshoot). Corner box = persisted H3
rung coefficient, CI, and TOST flag (`DOSE_TOST_BOUND`).

**How the metric is calculated.** `dose_response.csv` from
`build_dose_response`. Degenerate rows (teacher dose ≤ 0) are dropped
and counted. The H3 annotation is `claim_record("H3")`, not a
refit in this notebook.
""",
    ),
    (
        "markdown",
        """## 4. Fig3 — family heatmaps

**What this measures.** Subliminal Δ preference on the joint-rung
rectangles, one panel per cell type. The fourth panel is reserved for
different-init/same-stream and shows an explicit wave-2 placeholder
when those rows are absent. Shared diverging scale centred at 0; one
colorbar.
""",
    ),
    (
        "code",
        r"""rect = agg_rows(primary_rows(RECT))
if "animal" in rect.columns:
    rect = rect.loc[rect["animal"].astype(str) == "bear"]
scope(rect, "rect agg primary bear")
if "student_step" not in rect.columns or "teacher_step" not in rect.columns:
    raise KeyError("rect_delta needs student_step and teacher_step")
if "delta" not in rect.columns:
    raise KeyError("rect_delta needs delta")
fam = family_key(rect)
rect = rect.copy()
rect["_family"] = fam
student_order = list(plan.JOINT_RUNGS)
if (rect["student_step"].astype(int) == plan.RECT512_STUDENT_RUNG).any():
    student_order = [plan.RECT512_STUDENT_RUNG, *student_order]
teacher_order = list(plan.JOINT_RUNGS)
panels = (
    ("SI_SS", "same-init / same-stream"),
    ("SI_DS", "same-init / different-stream"),
    ("DI_DS", "different-init / different-stream"),
    ("DI_SS", "different-init / same-stream"),
)
grids = {}
for key, _ in panels:
    sub = rect.loc[rect["_family"] == key]
    grids[key] = grid_pivot(
        sub,
        value="delta",
        index_order=student_order,
        column_order=teacher_order,
    )
    print(key, "rows", len(sub), "NaNs", int(grids[key].isna().sum().sum()))

vals = np.concatenate([g.to_numpy().ravel() for g in grids.values()])
norm = diverging_norm(vals)
fig, axs = plt.subplots(2, 2, figsize=(12.2, 10.4))
im = None
for ax, (key, title) in zip(axs.ravel(), panels, strict=True):
    g = grids[key]
    if key == "DI_SS" and g.isna().all().all():
        ax.set_facecolor("0.93")
        ax.text(
            0.5,
            0.5,
            "awaiting retrained weight-seed uploads (wave 2)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        ax.set_xlabel("teacher pretraining step")
        ax.set_ylabel("student pretraining step")
        continue
    im = aplots.annotated_heatmap(
        ax,
        g,
        cmap="RdBu_r",
        norm=norm,
        fmt="{:.2f}",
        origin="lower",
        cell_fontsize=6.5,
        tick_fontsize=8,
    )
    mark_diagonal(ax, g.index, g.columns, plan.JOINT_RUNGS)
    if SHOW_QUARANTINE:
        mark_quarantine(ax, g.index, g.columns)
    ax.set_xlabel("teacher pretraining step")
    ax.set_ylabel("student pretraining step")
    ax.set_title(title)
if im is not None:
    fig.colorbar(
        im,
        ax=axs,
        shrink=0.72,
        label="subliminal Δ preference  (p_target − matched control mean)  [test / prob]",
    )
fig.suptitle(
    "Fig3  family heatmaps — does transfer peak when teacher and student share a checkpoint?\n"
    "PolyPythia-160M · bear · joint rungs · frozen student recipe · "
    f"{PROMPT_SET}/{plan.PRIMARY_SCORING}\n"
    "black outline = matched rung · hatch = quarantined PPL (if shown) · "
    "DI_SS reserved for wave 2",
    y=1.02,
    fontsize=11,
)
plt.show()
""",
    ),
    (
        "markdown",
        """### Reading this figure

**Question answered:** How does subliminal Δ preference vary across the four init×stream cell types when teacher and student pretraining steps are crossed, and is the different-init/same-stream panel still awaiting wave-2 weight-seed rows?
**Conclusion from the plot:** PENDING_FRONTIER_VISUAL_REVIEW

**Axes.** Each panel is student pretraining step × teacher pretraining
step on the joint-rung set (plus the student-only 512 row when
present). Colour and cell label = agg subliminal Δ preference on a
**shared** diverging scale centred at 0. Black outline = matched
rung. Hatch = quarantined cells when `SHOW_QUARANTINE`. The DI_SS
panel is a placeholder when those rows are absent.

**How the metric is calculated.** `rect_delta.csv` agg rows, primary
`test`/`prob`, bear. Family is the `cell` (or `family`) column.
NaNs stay NaN (unrun), never 0. One colorbar is shared across panels.
""",
    ),
    (
        "markdown",
        """## 5. Fig4 — ridge gain per teacher column

**What this measures.** Per teacher column,
`ridge_gain = diagonal Δ − mean(off-diagonal Δ)`, with the persisted
bootstrap CI, split by era (early / late) and by directed rectangle.
""",
    ),
    (
        "code",
        r"""rg = drop_degenerate(RIDGE.copy(), "ridge_gain")
scope(rg, "ridge_gain after degenerate drop")
if rg.empty:
    raise RuntimeError("ridge_gain.csv is empty")
need = {"teacher_step", "ridge_gain"}
missing = need - set(rg.columns)
if missing:
    raise KeyError(f"ridge_gain missing {sorted(missing)}")
if "era" not in rg.columns:
    rg["era"] = np.where(
        rg["teacher_step"].astype(int) <= plan.RIDGE_ERA_EARLY_MAX_STEP,
        "early",
        "late",
    )
rg = rg.copy()
rg["teacher_step"] = rg["teacher_step"].astype(int)
if "student_base" in rg.columns and "teacher_base" in rg.columns:
    rg["rectangle"] = rg["student_base"].astype(str) + " ← " + rg["teacher_base"].astype(str)
else:
    rg["rectangle"] = "all"
eras = ["early", "late"]
rects = list(rg["rectangle"].dropna().unique())
n_rect = max(len(rects), 1)
fig, axs = plt.subplots(
    len(eras), n_rect, figsize=(4.2 * n_rect, 7.2), sharey=True, squeeze=False
)
for i, era in enumerate(eras):
    for j, rect_name in enumerate(rects):
        ax = axs[i, j]
        sub = rg.loc[
            (rg["era"].astype(str) == era) & (rg["rectangle"] == rect_name)
        ].sort_values("teacher_step")
        if sub.empty:
            ax.set_facecolor("0.96")
            ax.text(0.5, 0.5, "no columns", ha="center", va="center", transform=ax.transAxes)
        else:
            y = sub["ridge_gain"].astype(float)
            x = [step_x(s) for s in sub["teacher_step"]]
            yerr = None
            if {"ci_low", "ci_high"} <= set(sub.columns):
                lo = pd.to_numeric(sub["ci_low"], errors="coerce")
                hi = pd.to_numeric(sub["ci_high"], errors="coerce")
                yerr = np.vstack([y - lo, hi - y])
            ax.errorbar(x, y, yerr=yerr, fmt="o-", capsize=3, color="C0")
        ax.axhline(0.0, color="0.6", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("teacher pretraining step")
        if j == 0:
            ax.set_ylabel(
                "ridge gain  (diagonal Δ − mean off-diagonal Δ)"
            )
        ax.set_title(f"{rect_name}\n{era}  (≤{plan.RIDGE_ERA_EARLY_MAX_STEP} vs later)")
fig.suptitle(
    "Fig4  ridge gain — does the matched-checkpoint cell beat the rest of its teacher column?\n"
    "PolyPythia-160M · bear · cluster-bootstrap CIs from ridge_gain.csv (not recomputed)\n"
    f"joint rungs={list(plan.JOINT_RUNGS)} · early/late split at teacher step {plan.RIDGE_ERA_EARLY_MAX_STEP} · {PROMPT_SET}/{plan.PRIMARY_SCORING}",
    y=1.03,
    fontsize=11,
)
plt.show()
""",
    ),
    (
        "markdown",
        """### Reading this figure

**Question answered:** For each directed rectangle and era, how large is the matched-checkpoint ridge gain in each teacher column, and do the persisted bootstrap CIs exclude zero?
**Conclusion from the plot:** PENDING_FRONTIER_VISUAL_REVIEW

**Axes.** x = teacher pretraining step (log). y = ridge gain =
diagonal subliminal Δ minus the mean off-diagonal Δ in that column.
Error bars = persisted `ci_low` / `ci_high`. Rows = early / late
era (split at `RIDGE_ERA_EARLY_MAX_STEP`). Columns = directed
rectangle `student ← teacher`.

**How the metric is calculated.** `ridge_gain.csv` from
`build_ridge_gain`. Degenerate columns (no diagonal) are dropped and
counted. CIs are not recomputed here.
""",
    ),
    (
        "markdown",
        """## 6. Fig5 — HP × rung surfaces

**What this measures.** Student-HP sweep on the diagonal: subliminal
Δ preference and Pile perplexity ratio, one heatmap each. The
collapse boundary (`perplexity_ratio > QUARANTINE_PPL_RATIO_MAX`) is
drawn on the PPL panel.
""",
    ),
    (
        "code",
        r"""hp = agg_rows(primary_rows(HP_GRID))
if "animal" in hp.columns:
    hp = hp.loc[hp["animal"].astype(str) == "bear"]
scope(hp, "hp_grid_delta agg")
if hp.empty:
    raise RuntimeError("hp_grid_delta.csv is empty")
if "gkd_tag" not in hp.columns:
    raise KeyError("hp_grid_delta needs gkd_tag")
if "student_step" not in hp.columns:
    raise KeyError("hp_grid_delta needs student_step")
hp = hp.copy()
if "teacher_step" in hp.columns:
    diag = hp.loc[hp["student_step"].astype(int) == hp["teacher_step"].astype(int)]
    if not diag.empty:
        hp = diag
hp["student_step"] = hp["student_step"].astype(int)
hp["cfg"] = hp["gkd_tag"].map(hp_label)
tag_order = [hp_label(cfg.tag) for cfg in plan.HP_CONFIGS]
delta_piv = grid_pivot(
    hp,
    value="delta",
    index="cfg",
    columns="student_step",
    index_order=tag_order,
    column_order=list(plan.HP_RUNGS),
)
ppl_col = None
for cand in ("perplexity_ratio", "mean_ppl_ratio"):
    if cand in hp.columns:
        ppl_col = cand
        break
if ppl_col is None and not HP_COLLAPSE.empty and "mean_ppl_ratio" in HP_COLLAPSE.columns:
    coll = HP_COLLAPSE.copy()
    coll["cfg"] = coll["gkd_tag"].map(hp_label) if "gkd_tag" in coll.columns else coll.get("cfg")
    ppl_src = coll.rename(columns={"mean_ppl_ratio": "perplexity_ratio"})
    ppl_col = "perplexity_ratio"
    ppl_piv = grid_pivot(
        ppl_src,
        value="perplexity_ratio",
        index="cfg",
        columns="student_step",
        index_order=tag_order,
        column_order=list(plan.HP_RUNGS),
    )
else:
    if ppl_col is None:
        raise KeyError("need perplexity_ratio on hp_grid_delta or hp_collapse")
    ppl_piv = grid_pivot(
        hp,
        value=ppl_col,
        index="cfg",
        columns="student_step",
        index_order=tag_order,
        column_order=list(plan.HP_RUNGS),
    )
print("delta pivot", delta_piv.shape, "PPL pivot", ppl_piv.shape)

fig, axs = plt.subplots(1, 2, figsize=(13.4, 6.2))
dnorm = diverging_norm(delta_piv.to_numpy())
aplots.annotated_heatmap(
    axs[0],
    delta_piv,
    cmap="RdBu_r",
    norm=dnorm,
    fmt="{:.2f}",
    origin="lower",
    cell_fontsize=6.5,
    tick_fontsize=8,
)
axs[0].set_xlabel("student = teacher pretraining step  (HP diagonal)")
axs[0].set_ylabel("student HP config  (LoRA rank / α / lr / epochs)")
axs[0].set_title("subliminal Δ preference  [test / prob]")

finite = ppl_piv.to_numpy(dtype=float)
finite = finite[np.isfinite(finite) & (finite > 0)]
if finite.size:
    pnorm = LogNorm(vmin=min(finite.min(), 1.0), vmax=max(finite.max(), plan.QUARANTINE_PPL_RATIO_MAX))
else:
    pnorm = None
aplots.annotated_heatmap(
    axs[1],
    ppl_piv,
    cmap="YlOrRd",
    norm=pnorm,
    fmt="{:.2f}",
    origin="lower",
    cell_fontsize=6.5,
    tick_fontsize=8,
    empty_facecolor="0.95",
)
collapsed = ppl_piv.to_numpy(dtype=float) > plan.QUARANTINE_PPL_RATIO_MAX
for i in range(collapsed.shape[0]):
    for j in range(collapsed.shape[1]):
        if collapsed[i, j]:
            axs[1].add_patch(
                Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="C3",
                    lw=1.6,
                )
            )
axs[1].set_xlabel("student = teacher pretraining step  (HP diagonal)")
axs[1].set_ylabel("student HP config  (LoRA rank / α / lr / epochs)")
axs[1].set_title(
    f"Pile perplexity ratio  (student / untrained base); red box = collapse > {plan.QUARANTINE_PPL_RATIO_MAX:g}"
)
fig.suptitle(
    "Fig5  HP × rung surfaces — is the diagonal geometry stable across student recipes, and where does PPL collapse?\n"
    f"PolyPythia-160M · bear · original←original · {len(plan.HP_CONFIGS)} configs × {len(plan.HP_RUNGS)} rungs · {PROMPT_SET}/{plan.PRIMARY_SCORING}",
    y=1.03,
    fontsize=11,
)
plt.show()
scope(HP_COLLAPSE, "hp_collapse")
""",
    ),
    (
        "markdown",
        """### Reading this figure

**Question answered:** Across student LoRA recipes and diagonal pretraining rungs, how do subliminal Δ preference and Pile perplexity ratio move, and where does the collapse boundary (`perplexity_ratio > QUARANTINE_PPL_RATIO_MAX`) fall?
**Conclusion from the plot:** PENDING_FRONTIER_VISUAL_REVIEW

**Axes.** Both panels: x = matched pretraining step (`HP_RUNGS`).
y = student HP config (lr / rank / α / epochs). Left colour = agg
subliminal Δ preference (diverging, 0-centred). Right colour = Pile
perplexity ratio on a log scale; red outline = collapsed cells
(`> QUARANTINE_PPL_RATIO_MAX`).

**How the metric is calculated.** `hp_grid_delta.csv` diagonal rows
(and `hp_collapse.csv` for PPL if the ratio is not on the delta
frame). Config labels are mapped from persisted `gkd_tag` via
`plan.HP_CONFIGS`. No HP is re-scored here.
""",
    ),
    (
        "markdown",
        """## 7. Fig6 — HP off-diagonal probes

**What this measures.** Subliminal Δ preference at the five
preregistered off-diagonal probe cells, one series per probed HP
config.
""",
    ),
    (
        "code",
        r"""hp = agg_rows(primary_rows(HP_GRID))
if "animal" in hp.columns:
    hp = hp.loc[hp["animal"].astype(str) == "bear"]
if hp.empty:
    raise RuntimeError("hp_grid_delta.csv is empty")
need = {"student_step", "teacher_step", "delta", "gkd_tag"}
missing = need - set(hp.columns)
if missing:
    raise KeyError(f"hp_grid_delta missing {sorted(missing)}")
probe_tags = {cfg.tag for cfg in plan.PROBE_HP_CONFIGS}
pairs = {(int(s), int(t)) for s, t in plan.PROBE_CELLS}
hp = hp.copy()
hp["student_step"] = hp["student_step"].astype(int)
hp["teacher_step"] = hp["teacher_step"].astype(int)
probes = hp.loc[
    hp["gkd_tag"].astype(str).isin(probe_tags)
    & [
        (int(s), int(t)) in pairs
        for s, t in zip(hp["student_step"], hp["teacher_step"], strict=True)
    ]
]
scope(probes, "HP off-diagonal probes")
probes = probes.copy()
probes["cfg"] = probes["gkd_tag"].map(hp_label)
probes["probe"] = [
    f"student {s} ← teacher {t}"
    for s, t in zip(probes["student_step"], probes["teacher_step"], strict=True)
]
probe_order = [f"student {s} ← teacher {t}" for s, t in plan.PROBE_CELLS]
cfg_order = [hp_label(cfg.tag) for cfg in plan.PROBE_HP_CONFIGS]

fig, ax = plt.subplots(figsize=(10.6, 5.0))
if probes.empty:
    ax.text(0.5, 0.5, "no probe rows", ha="center", va="center")
else:
    sns.barplot(
        data=probes,
        x="probe",
        y="delta",
        hue="cfg",
        order=probe_order,
        hue_order=cfg_order,
        ax=ax,
    )
ax.axhline(0.0, color="0.6", lw=0.8)
ax.set_xlabel("off-diagonal probe cell  (student step ← teacher step)")
ax.set_ylabel("subliminal Δ preference  (p_target − matched control mean)")
ax.set_title(
    "Fig6  HP off-diagonal probes — does off-diagonal decay keep its sign across recipes?\n"
    f"PolyPythia-160M · bear · original←original · {len(plan.PROBE_HP_CONFIGS)} configs × {len(plan.PROBE_CELLS)} cells · {PROMPT_SET}/{plan.PRIMARY_SCORING}"
)
ax.legend(title="student HP config", fontsize=8)
plt.xticks(rotation=18, ha="right")
plt.show()
print("probe rows", len(probes), "designed", len(plan.PROBE_HP_CONFIGS) * len(plan.PROBE_CELLS))
""",
    ),
    (
        "markdown",
        """### Reading this figure

**Question answered:** At the five preregistered off-diagonal probe cells, does subliminal Δ preference keep the same decay sign across the probed student HP configs?
**Conclusion from the plot:** PENDING_FRONTIER_VISUAL_REVIEW

**Axes.** x = probe cell (`student step ← teacher step`). y =
subliminal Δ preference. Hue = probed student HP config (the four
`PROBE_HP_CONFIGS` tags). Zero line = no transfer above control.

**How the metric is calculated.** `hp_grid_delta.csv` filtered to
`plan.PROBE_CELLS` × `plan.PROBE_HP_CONFIGS` tags. Bars are the
persisted agg deltas; this notebook does not re-score probes.
""",
    ),
    (
        "markdown",
        """## 8. Fig7 — receptivity onset

**What this measures.** Subliminal Δ preference versus student
pretraining step against the onset floor
`control_mean + max(2·SD, 0.02)`, plus the rung-512 cross-init cells
drawn per teacher column.
""",
    ),
    (
        "code",
        r"""scope(ONSET, "onset")
if ONSET.empty:
    raise RuntimeError("onset.csv is empty")
need = {"delta", "student_step"}
missing = need - set(ONSET.columns)
if missing:
    raise KeyError(f"onset missing {sorted(missing)}")
on = ONSET.copy()
on["student_step"] = on["student_step"].astype(int)
if "teacher_step" in on.columns:
    on["teacher_step"] = pd.to_numeric(on["teacher_step"], errors="coerce")
thr_col = "threshold" if "threshold" in on.columns else None
same = on
if "student_base" in on.columns and "teacher_base" in on.columns:
    same = on.loc[
        (on["student_base"].astype(str) == "original")
        & (on["teacher_base"].astype(str) == "original")
    ]
if same.empty:
    same = on.loc[on["student_step"] != plan.RECT512_STUDENT_RUNG]
cross = on.loc[on["student_step"] == plan.RECT512_STUDENT_RUNG]
if "cell" in on.columns:
    di = on["cell"].astype(str) == "DI_DS"
    if di.any():
        cross = on.loc[di & (on["student_step"] == plan.RECT512_STUDENT_RUNG)]

fig, ax = plt.subplots(figsize=(9.0, 5.0))
same_x = add_step_x(same, "student_step")
if not same_x.empty:
    grouped = same_x.groupby("student_step", dropna=False)
    xs, ys, lo, hi = [], [], [], []
    for step, g in grouped:
        xs.append(step_x(step))
        ys.append(float(g["delta"].mean()))
        if thr_col is not None:
            lo.append(0.0)
            hi.append(float(pd.to_numeric(g[thr_col], errors="coerce").mean()))
    ax.plot(xs, ys, "o-", color="C0", label="same-init Δ (original ← original)")
    if thr_col is not None and hi:
        ax.fill_between(
            xs,
            lo,
            hi,
            color="0.75",
            alpha=0.45,
            label="onset floor  control + max(2·SD, 0.02)",
        )
        ax.plot(xs, hi, color="0.35", ls="--", lw=1.0)
ax.axhline(plan.ONSET_ABS_FLOOR, color="0.5", ls=":", lw=1.0, label=f"absolute floor {plan.ONSET_ABS_FLOOR}")
markers = [int(round(m * plan.CHECKPOINT_MAX_STEP)) for m in plan.ONSET_MARKERS]
for mk, ls in zip(markers, ("--", ":"), strict=False):
    ax.axvline(step_x(mk), color="purple", ls=ls, lw=1.0, label=f"literature marker {mk}")
if not cross.empty:
    palette = sns.color_palette("tab10", n_colors=max(cross["teacher_step"].nunique(), 1))
    tvals = sorted(cross["teacher_step"].dropna().unique())
    for color, tstep in zip(palette, tvals, strict=False):
        sub = cross.loc[cross["teacher_step"] == tstep]
        ax.scatter(
            [step_x(plan.RECT512_STUDENT_RUNG)] * len(sub),
            sub["delta"],
            s=48,
            color=color,
            marker="D",
            zorder=4,
            label=f"rung-{plan.RECT512_STUDENT_RUNG} × teacher {int(tstep)}",
        )
ax.set_xscale("log")
ax.axhline(0.0, color="0.6", lw=0.8)
ax.set_xlabel("student pretraining step")
ax.set_ylabel("subliminal Δ preference  (p_target − matched control mean)")
ax.set_title(
    f"Fig7  receptivity onset — when does a student first clear the control floor, and do cross-init teachers at rung {plan.RECT512_STUDENT_RUNG} stay below it?\n"
    f"PolyPythia-160M · bear · floor = control mean + max({plan.ONSET_SD_MULTIPLE:g}·SD, {plan.ONSET_ABS_FLOOR}) · "
    f"{plan.ONSET_CONSECUTIVE} consecutive rungs · {PROMPT_SET}/{plan.PRIMARY_SCORING}"
)
ax.legend(fontsize=7, loc="best")
plt.show()
print("onset rows", len(on), "same-init", len(same), "rung-512", len(cross))
""",
    ),
    (
        "markdown",
        """### Reading this figure

**Question answered:** Along the student pretraining axis, when does subliminal Δ preference first clear the control floor `control_mean + max(2·SD, 0.02)`, and do the rung-512 cross-init cells stay below that floor per teacher column?
**Conclusion from the plot:** PENDING_FRONTIER_VISUAL_REVIEW

**Axes.** x = student pretraining step (log). y = subliminal Δ
preference. Grey band / dashed line = persisted onset threshold
(`threshold` in `onset.csv`). Purple lines = literature fractions of
the published max-step budget. Diamonds at the student-only onset
rung = cross-init cells, one colour per teacher column.

**How the metric is calculated.** `onset.csv` from `build_onset_floor`
(same-init walk and the student-only cross-init onset rows).
Thresholds and deltas are persisted; this notebook does not re-walk
the floor rule.
""",
    ),
    (
        "markdown",
        """## 9. Fig8 — cross-init penalty map

**What this measures.**
`penalty = 1 − Δ(different-init/different-stream) / Δ(same-init/same-stream)`
on the joint-rung rectangle.
""",
    ),
    (
        "code",
        r"""pen = drop_degenerate(PENALTY.copy(), "penalty")
scope(pen, "penalty after degenerate drop")
if pen.empty:
    raise RuntimeError("penalty_map.csv is empty")
need = {"student_step", "teacher_step", "penalty"}
missing = need - set(pen.columns)
if missing:
    raise KeyError(f"penalty_map missing {sorted(missing)}")
if "cell" in pen.columns:
    di = pen.loc[pen["cell"].astype(str) == "DI_DS"]
    if not di.empty:
        pen = di
if "animal" in pen.columns:
    bear = pen.loc[pen["animal"].astype(str) == "bear"]
    if not bear.empty:
        pen = bear
student_order = list(plan.JOINT_RUNGS)
if (pen["student_step"].astype(int) == plan.RECT512_STUDENT_RUNG).any():
    student_order = [plan.RECT512_STUDENT_RUNG, *student_order]
piv = grid_pivot(
    pen,
    value="penalty",
    index_order=student_order,
    column_order=list(plan.JOINT_RUNGS),
)
print("penalty pivot", piv.shape, "NaNs", int(piv.isna().sum().sum()))
norm = diverging_norm(piv.to_numpy())
fig, ax = plt.subplots(figsize=(8.8, 7.2))
aplots.annotated_heatmap(
    ax,
    piv,
    cmap="RdBu_r",
    norm=norm,
    fmt="{:.2f}",
    origin="lower",
    cell_fontsize=7,
    tick_fontsize=8,
)
mark_diagonal(ax, piv.index, piv.columns, plan.JOINT_RUNGS)
ax.set_xlabel("teacher pretraining step")
ax.set_ylabel("student pretraining step")
ax.set_title(
    "Fig8  cross-init penalty map — does the different-init deficit shrink as both rungs mature?\n"
    "penalty = 1 − Δ(DI_DS) / Δ(SI_SS) · PolyPythia-160M · bear · frozen student recipe\n"
    f"joint rungs={list(plan.JOINT_RUNGS)} · {PROMPT_SET}/{plan.PRIMARY_SCORING} · black outline = matched rung"
)
plt.show()
""",
    ),
    (
        "markdown",
        """### Reading this figure

**Question answered:** How does the cross-init penalty `1 − Δ(DI_DS)/Δ(SI_SS)` vary when teacher and student pretraining steps are crossed on the joint-rung rectangle?
**Conclusion from the plot:** PENDING_FRONTIER_VISUAL_REVIEW

**Axes.** x = teacher pretraining step. y = student pretraining step
(joint rungs, plus student 512 when present). Colour and label =
penalty on a diverging scale centred at 0. Black outline = matched
rung.

**How the metric is calculated.** `penalty_map.csv` from
`build_penalty_map`. Degenerate cells (SI_SS denominator ≤ 0) are
dropped and counted. Values are persisted, not recomputed.
""",
    ),
    (
        "markdown",
        """## 10. Fig9 — emissivity / receptivity

**What this measures.** At each matched rung, the additive fit
`delta_ij = μ + τ_i + σ_j` yields `τ − σ` for the `original` base —
whether original emits more than it receives.
""",
    ),
    (
        "code",
        r"""scope(EMISSIVITY, "emissivity_fit")
if EMISSIVITY.empty:
    raise RuntimeError("emissivity_fit.csv is empty")
need = {"step", "tau_minus_sigma"}
missing = need - set(EMISSIVITY.columns)
if missing:
    raise KeyError(f"emissivity_fit missing {sorted(missing)}")
em = EMISSIVITY.copy()
if "base" in em.columns:
    em = em.loc[em["base"].astype(str) == "original"]
em["step"] = em["step"].astype(int)
em = add_step_x(em, "step")
fig, ax = plt.subplots(figsize=(8.4, 4.6))
ax.plot(
    em["x"],
    em["tau_minus_sigma"],
    "o-",
    color="C0",
    label="original  τ − σ",
)
if "tau" in em.columns:
    ax.plot(em["x"], em["tau"], "s--", color="C1", label="original τ (emissivity)")
if "sigma" in em.columns:
    ax.plot(em["x"], em["sigma"], "^:", color="C2", label="original σ (receptivity)")
ax.axhline(0.0, color="0.6", lw=0.8)
ax.set_xscale("log")
ax.set_xlabel("matched pretraining step  (student = teacher)")
ax.set_ylabel("additive-fit coefficient  (Δ preference units)")
ax.set_title(
    "Fig9  emissivity vs receptivity — does original emit more than it receives as pretraining proceeds?\n"
    "fit: Δ_ij = μ + τ_teacher + σ_student · PolyPythia-160M · bear · joint matched rungs\n"
    f"rungs={list(plan.JOINT_RUNGS)} · {PROMPT_SET}/{plan.PRIMARY_SCORING}"
)
ax.legend(fontsize=8)
plt.show()
print(em[["step", "tau_minus_sigma"] + [c for c in ("tau", "sigma", "n_cells") if c in em.columns]].to_string())
""",
    ),
    (
        "markdown",
        """### Reading this figure

**Question answered:** For the original base, how does teacher emissivity minus student receptivity (`τ − σ`) move with the matched pretraining rung?
**Conclusion from the plot:** PENDING_FRONTIER_VISUAL_REVIEW

**Axes.** x = matched pretraining step (log). y = additive-fit
coefficients in subliminal-Δ units. Solid = `τ − σ` for `original`;
dashed / dotted = `τ` and `σ` when present. Zero = symmetric
emit/receive.

**How the metric is calculated.** `emissivity_fit.csv` from
`build_emissivity_fit` on matched-rung cells. This notebook plots the
persisted coefficients; it does not refit `μ, τ, σ`.
""",
    ),
    (
        "markdown",
        """## 11. Fig10 — cat 7×7 beside the bear diagonal

**What this measures.** Cat `original←original` 7×7 subliminal Δ
heatmap next to the bear matched-rung diagonal, so the two animals’
teacher-dose peaks can be compared on the same page.
""",
    ),
    (
        "code",
        r"""cat = agg_rows(primary_rows(CAT))
scope(cat, "cat_grid_delta")
if cat.empty:
    raise RuntimeError("cat_grid_delta.csv is empty")
need = {"student_step", "teacher_step", "delta"}
missing = need - set(cat.columns)
if missing:
    raise KeyError(f"cat_grid_delta missing {sorted(missing)}")
if "animal" in cat.columns:
    cat = cat.loc[cat["animal"].astype(str) == "cat"]
cat_piv = grid_pivot(
    cat,
    value="delta",
    index_order=list(plan.CAT_RUNGS),
    column_order=list(plan.CAT_RUNGS),
)
bear_diag = pd.DataFrame()
for frame, animal in ((RECT, "bear"), (HP_GRID, "bear")):
    src = agg_rows(primary_rows(frame))
    if src.empty or "student_step" not in src.columns:
        continue
    if "animal" in src.columns:
        src = src.loc[src["animal"].astype(str) == animal]
    if "student_base" in src.columns:
        src = src.loc[src["student_base"].astype(str) == "original"]
    if "teacher_base" in src.columns:
        src = src.loc[src["teacher_base"].astype(str) == "original"]
    if "gkd_tag" in src.columns:
        frozen = src.loc[src["gkd_tag"].astype(str) == plan.FROZEN_GKD_TAG]
        if not frozen.empty:
            src = frozen
    if "teacher_step" in src.columns:
        src = src.loc[src["student_step"].astype(int) == src["teacher_step"].astype(int)]
    if not src.empty:
        bear_diag = src
        break
if bear_diag.empty and not DOSE_RESP.empty:
    src = drop_degenerate(primary_rows(DOSE_RESP), "dose_response (bear diagonal fallback)")
    if "n_epochs" in src.columns:
        src = src.loc[pd.to_numeric(src["n_epochs"], errors="coerce") == plan.TEACHER_EPOCHS_FROZEN]
    bear_diag = src
scope(bear_diag, "bear diagonal")

fig, axs = plt.subplots(1, 2, figsize=(12.8, 6.0))
cnorm = diverging_norm(cat_piv.to_numpy())
aplots.annotated_heatmap(
    axs[0],
    cat_piv,
    cmap="RdBu_r",
    norm=cnorm,
    fmt="{:.2f}",
    origin="lower",
    cell_fontsize=6.5,
    tick_fontsize=8,
)
mark_diagonal(axs[0], cat_piv.index, cat_piv.columns, plan.CAT_RUNGS)
axs[0].set_xlabel("teacher pretraining step")
axs[0].set_ylabel("student pretraining step")
axs[0].set_title(
    f"cat  original←original  {len(plan.CAT_RUNGS)}×{len(plan.CAT_RUNGS)}\n"
    "subliminal Δ preference  [test / prob]"
)
if bear_diag.empty:
    axs[1].text(0.5, 0.5, "bear diagonal missing", ha="center", va="center")
else:
    b = add_step_x(bear_diag, "student_step")
    axs[1].plot(b["x"], b["delta"], "o-", color="C0", label="bear diagonal Δ")
    if not cat.empty:
        cdiag = cat.loc[cat["student_step"].astype(int) == cat["teacher_step"].astype(int)]
        cdiag = add_step_x(cdiag, "student_step")
        axs[1].plot(cdiag["x"], cdiag["delta"], "s--", color="C1", label="cat diagonal Δ")
    axs[1].axhline(0.0, color="0.6", lw=0.8)
    axs[1].set_xscale("log")
    axs[1].legend(fontsize=8)
axs[1].set_xlabel("matched pretraining step  (student = teacher)")
axs[1].set_ylabel("subliminal Δ preference  (p_target − matched control mean)")
axs[1].set_title("bear vs cat matched-rung diagonal")
fig.suptitle(
    "Fig10  cat 7×7 beside the bear diagonal — do the two animals peak at different teacher rungs?\n"
    f"PolyPythia-160M · cat rungs={list(plan.CAT_RUNGS)} · bear frozen recipe · {PROMPT_SET}/{plan.PRIMARY_SCORING}",
    y=1.03,
    fontsize=11,
)
plt.show()
""",
    ),
    (
        "markdown",
        """### Reading this figure

**Question answered:** On the cat `original←original` 7×7, how does subliminal Δ preference vary when teacher and student rungs are crossed, and how does that animal’s matched diagonal sit next to the bear matched diagonal?
**Conclusion from the plot:** PENDING_FRONTIER_VISUAL_REVIEW

**Axes.** Left: student × teacher pretraining step on `CAT_RUNGS`;
colour = cat agg subliminal Δ preference; black outline = matched
rung. Right: x = matched pretraining step (log); y = subliminal Δ
preference for the bear frozen-recipe diagonal and the cat diagonal.

**How the metric is calculated.** Left = `cat_grid_delta.csv`. Right =
bear matched-rung rows from `rect_delta.csv` / `hp_grid_delta.csv`
(frozen `gkd_tag`) or the frozen-epoch slice of `dose_response.csv`.
No claim statistic is refit.
""",
    ),
    (
        "markdown",
        """## 12. Fig11 — metric concordance

**What this measures.** Primary `test`/`prob` subliminal Δ versus the
sampled twin `test_sample`/`sample` on the diagonal: identity scatter
with persisted Spearman / sign concordance, and a Bland–Altman panel.
""",
    ),
    (
        "code",
        r"""scope(CONCORD, "metric_concordance")
if CONCORD.empty:
    raise RuntimeError("metric_concordance.csv is empty")
row = CONCORD.iloc[0]
print(CONCORD.to_string())
twin_set = next(
    s for s in plan.EVAL_PROMPT_SETS if s == f"{plan.PRIMARY_PROMPT_SET}_sample"
)
twin_scoring = "sample"

def _pairs() -> pd.DataFrame:
    keys = [
        c
        for c in (
            "animal",
            "student_base",
            "teacher_base",
            "student_step",
            "teacher_step",
            "cell",
            "gkd_tag",
        )
        if True
    ]
    frames = []
    for name, frame in ROB.items():
        frames.append(frame.assign(_src=name))
    for name, frame in (("hp", HP_GRID), ("rect", RECT), ("cat", CAT)):
        if not frame.empty and "set" in frame.columns:
            frames.append(frame.assign(_src=name))
    if not frames:
        return pd.DataFrame()
    both = pd.concat(frames, ignore_index=True)
    both = agg_rows(both)
    if "student_step" in both.columns and "teacher_step" in both.columns:
        both = both.loc[both["student_step"].astype(int) == both["teacher_step"].astype(int)]
    if "set" not in both.columns or "scoring" not in both.columns:
        return pd.DataFrame()
    left = both.loc[
        (both["set"].astype(str) == plan.PRIMARY_PROMPT_SET)
        & (both["scoring"].astype(str) == plan.PRIMARY_SCORING)
    ]
    right = both.loc[
        (both["set"].astype(str) == twin_set)
        & (both["scoring"].astype(str) == twin_scoring)
    ]
    join_on = [c for c in keys if c in left.columns and c in right.columns]
    if not join_on:
        return pd.DataFrame()
    merged = left.merge(
        right[join_on + ["delta"]],
        on=join_on,
        how="inner",
        suffixes=("_p", "_t"),
    )
    return merged

pairs = _pairs()
scope(pairs, "primary vs twin diagonal pairs")
fig, axs = plt.subplots(1, 2, figsize=(11.2, 4.8))
if pairs.empty or "delta_p" not in pairs.columns:
    axs[0].text(0.5, 0.5, "paired primary/twin rows not on disk", ha="center", va="center")
    axs[1].text(0.5, 0.5, "Bland–Altman needs paired rows", ha="center", va="center")
else:
    x = pd.to_numeric(pairs["delta_p"], errors="coerce")
    y = pd.to_numeric(pairs["delta_t"], errors="coerce")
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    axs[0].scatter(x, y, s=28, alpha=0.75, c="C0")
    lims = [
        float(np.nanmin([x.min(), y.min(), 0.0])),
        float(np.nanmax([x.max(), y.max(), 0.1])),
    ]
    axs[0].plot(lims, lims, "k--", lw=1.0, label="identity")
    axs[0].set_xlim(lims)
    axs[0].set_ylim(lims)
    mid = 0.5 * (x + y)
    diff = x - y
    axs[1].scatter(mid, diff, s=28, alpha=0.75, c="C0")
    axs[1].axhline(0.0, color="0.6", lw=0.8)
    ba_mean = float(row["ba_mean"]) if "ba_mean" in row.index and pd.notna(row["ba_mean"]) else float(diff.mean())
    ba_sd = float(row["ba_sd"]) if "ba_sd" in row.index and pd.notna(row["ba_sd"]) else float(diff.std(ddof=1))
    axs[1].axhline(ba_mean, color="C3", lw=1.1, label="mean difference")
    axs[1].axhline(ba_mean + 1.96 * ba_sd, color="C3", ls="--", lw=1.0, label="±1.96 SD")
    axs[1].axhline(ba_mean - 1.96 * ba_sd, color="C3", ls="--", lw=1.0)
axs[0].set_xlabel("subliminal Δ preference  test / prob  (primary)")
axs[0].set_ylabel("subliminal Δ preference  test_sample / sample  (twin)")
axs[0].set_title("identity scatter")
axs[1].set_xlabel("mean of primary and twin Δ")
axs[1].set_ylabel("primary Δ − twin Δ")
axs[1].set_title("Bland–Altman")
spear = row["spearman"] if "spearman" in row.index else None
sign = row["sign_agreement"] if "sign_agreement" in row.index else None
n_cells = row["n_cells"] if "n_cells" in row.index else (len(pairs) if not pairs.empty else 0)
axs[0].text(
    0.03,
    0.97,
    f"Spearman ρ = {fmt_num(spear)}\nsign concordance = {fmt_num(sign)}\nn = {n_cells}",
    transform=axs[0].transAxes,
    va="top",
    fontsize=8,
    bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
)
fig.suptitle(
    "Fig11  metric concordance — do test/prob and test_sample/sample agree on the diagonal?\n"
    f"PolyPythia-160M · diagonal cells · twin={twin_set}/{twin_scoring} · "
    f"thresholds ρ≥{plan.CONCORDANCE_MIN_SPEARMAN} and sign≥{plan.CONCORDANCE_MIN_SIGN_AGREEMENT}",
    y=1.03,
    fontsize=11,
)
axs[0].legend(fontsize=8)
axs[1].legend(fontsize=8)
plt.show()
""",
    ),
    (
        "markdown",
        """### Reading this figure

**Question answered:** On the matched diagonal, do primary `test`/`prob` and sampled-twin `test_sample`/`sample` subliminal Δ preferences agree in rank (Spearman) and sign, and where do they differ on a Bland–Altman plot?
**Conclusion from the plot:** PENDING_FRONTIER_VISUAL_REVIEW

**Axes.** Left: x = primary Δ, y = twin Δ, dashed identity. Right:
x = mean of the two Δs, y = primary − twin; solid = mean difference,
dashed = ±1.96 SD from `metric_concordance.csv`. Annotation =
persisted Spearman and sign concordance (not recomputed).

**How the metric is calculated.** Summary row from
`metric_concordance.csv`. Paired points are joined from
`robustness/*.csv` (or any loaded frame that carries both `set` /
`scoring` values). Thresholds are `CONCORDANCE_MIN_SPEARMAN` and
`CONCORDANCE_MIN_SIGN_AGREEMENT`.
""",
    ),
    (
        "markdown",
        """## 13. Fig12 — perplexity and quarantine

**What this measures.** Untrained-base Pile PPL versus rung, teacher
dose versus teacher PPL cost, and the quarantine map of flagged
leaves. Flagged cells are labelled, never silently dropped.
""",
    ),
    (
        "code",
        r"""scope(BASE_PPL, "base_ppl")
scope(TEACHER_DOSE, "teacher_dose (PPL)")
scope(QUAR, "quarantine")
if BASE_PPL.empty:
    raise RuntimeError("base_ppl.csv is empty")
rung_col = "rung" if "rung" in BASE_PPL.columns else "step"
ppl_col = "ppl_overall" if "ppl_overall" in BASE_PPL.columns else None
if ppl_col is None:
    raise KeyError("base_ppl needs ppl_overall")
bp = BASE_PPL.copy()
bp[rung_col] = bp[rung_col].astype(int)
fig, axs = plt.subplots(1, 3, figsize=(13.6, 4.6))
for base, sub in bp.groupby("base" if "base" in bp.columns else bp.columns[0]):
    sub = add_step_x(sub, rung_col)
    axs[0].plot(sub["x"], sub[ppl_col], "o-", label=str(base))
axs[0].set_xscale("log")
axs[0].set_xlabel("untrained-base pretraining step")
axs[0].set_ylabel("Pile perplexity  (untrained base)")
axs[0].set_title("untrained-base PPL vs rung")
axs[0].legend(fontsize=7)

td = TEACHER_DOSE.copy()
if "set" in td.columns:
    td = td.loc[td["set"].astype(str) == PROMPT_SET]
y_ppl = "teacher_ppl_ratio" if "teacher_ppl_ratio" in td.columns else None
y_dose = "p_target_mean" if "p_target_mean" in td.columns else "p_target"
if y_ppl is None or td.empty or y_dose not in td.columns:
    axs[1].text(0.5, 0.5, "teacher PPL cost missing", ha="center", va="center")
else:
    if "step" in td.columns:
        sc = axs[1].scatter(
            td[y_dose],
            td[y_ppl],
            c=td["step"].astype(float),
            cmap="viridis",
            s=36,
            alpha=0.85,
        )
        fig.colorbar(sc, ax=axs[1], label="teacher pretraining step")
    else:
        axs[1].scatter(td[y_dose], td[y_ppl], s=36, alpha=0.85)
    axs[1].axhline(1.0, color="0.6", lw=0.8)
axs[1].set_xlabel("teacher last-token p(target animal)")
axs[1].set_ylabel("teacher Pile perplexity ratio  (fine-tuned / base)")
axs[1].set_title("teacher dose vs teacher PPL cost")

q = QUAR.copy()
s_col = "student_step" if "student_step" in q.columns else ("step" if "step" in q.columns else None)
t_col = "teacher_step" if "teacher_step" in q.columns else None
v_col = None
for cand in ("perplexity_ratio", "value", "ppl_ratio"):
    if cand in q.columns:
        v_col = cand
        break
if q.empty or s_col is None or t_col is None:
    axs[2].text(0.5, 0.5, "quarantine map empty", ha="center", va="center")
    print("quarantine rows", len(q))
else:
    q = q.copy()
    q[s_col] = q[s_col].astype(int)
    q[t_col] = q[t_col].astype(int)
    if v_col is None:
        q["_one"] = 1.0
        v_col = "_one"
    q_cells = (
        q.groupby([s_col, t_col], dropna=False)[v_col]
        .max()
        .reset_index()
        .rename(columns={s_col: "student_step", t_col: "teacher_step", v_col: "ppl"})
    )
    q_piv = grid_pivot(
        q_cells,
        value="ppl",
        index_order=list(plan.HP_RUNGS),
        column_order=list(plan.JOINT_RUNGS),
    )
    finite = q_piv.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite) & (finite > 0)]
    qnorm = LogNorm(
        vmin=min(finite.min(), plan.QUARANTINE_PPL_RATIO_MAX) if finite.size else 1.0,
        vmax=finite.max() if finite.size else plan.QUARANTINE_PPL_RATIO_MAX,
    ) if finite.size else None
    aplots.annotated_heatmap(
        axs[2],
        q_piv,
        cmap="YlOrRd",
        norm=qnorm,
        fmt="{:.1f}",
        origin="lower",
        cell_fontsize=6,
        empty_facecolor="0.95",
    )
    print("unique quarantined cells", len(q_cells), "from", len(q), "rows")
axs[2].set_xlabel("teacher pretraining step")
axs[2].set_ylabel("student pretraining step")
axs[2].set_title("quarantine map  (max PPL ratio on flagged cells)")
fig.suptitle(
    "Fig12  perplexity — is capability collapse sitting on the ridge, or only in mismatch corners?\n"
    f"PolyPythia-160M · pile_reference_bank · quarantine bound {plan.QUARANTINE_PPL_RATIO_MAX:g} · "
    f"filter drop bound {plan.QUARANTINE_FILTER_DROP_MAX:g}",
    y=1.04,
    fontsize=11,
)
plt.show()
if not q.empty:
    print("quarantine columns:", list(QUAR.columns))
    if "scope" in QUAR.columns:
        print("scope:\n", QUAR["scope"].value_counts(dropna=False))
""",
    ),
    (
        "markdown",
        """### Reading this figure

**Question answered:** How does untrained-base Pile perplexity move with pretraining step, what PPL cost does a stronger teacher pay, and where do quarantined high-PPL leaves sit on the student×teacher map?
**Conclusion from the plot:** PENDING_FRONTIER_VISUAL_REVIEW

**Axes.** (a) x = untrained-base pretraining step (log), y = Pile
perplexity, one line per base. (b) x = teacher last-token p(target),
y = teacher Pile perplexity ratio, colour = teacher step. (c) student
× teacher step; colour = max flagged PPL ratio (log); blank = not
flagged.

**How the metric is calculated.** `base_ppl.csv`, the
`teacher_ppl_ratio` column of `teacher_dose.csv`, and
`quarantine.csv`. Quarantine bounds are `QUARANTINE_PPL_RATIO_MAX` and
`QUARANTINE_FILTER_DROP_MAX`. Flagged cells stay in every other
frame.
""",
    ),
    (
        "markdown",
        """## 14. Fig13 — claims board

**What this measures.** Every persisted `claims.json` record as a
table, then headline numbers recomputed from the loaded frames (row
counts, means, argmax rungs). Trust **that output**, not this
markdown. This notebook does not invent a conclusion and does not
re-run a bootstrap.
""",
    ),
    (
        "code",
        r"""def _fmt_stat_val(value) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, list):
        return "[" + ", ".join(_fmt_stat_val(x) for x in value) + "]"
    return str(value)


def flatten_stat(stat) -> str:
    if not stat:
        return ""
    if not isinstance(stat, dict):
        return str(stat)
    return "; ".join(f"{k}={_fmt_stat_val(v)}" for k, v in stat.items())


recs = CLAIMS.get("claims") or CLAIMS.get("records") or []
ids = [rec.get("id") for rec in recs]
assert ids == list(plan.CLAIM_IDS), (ids, plan.CLAIM_IDS)
rows = []
for cid in plan.CLAIM_IDS:
    rec = claim_record(cid)
    rows.append(
        {
            "id": rec.get("id"),
            "status": rec.get("status"),
            "reason": rec.get("reason"),
            "statistic": flatten_stat(rec.get("statistic")),
            "threshold": flatten_stat(rec.get("threshold")),
        }
    )
board = pd.DataFrame(rows)
assert list(board["id"]) == list(plan.CLAIM_IDS)
display(board)
""",
    ),
    (
        "code",
        r"""print("=== data-driven takeaways (recomputed from frames; not claim bootstraps) ===")
print("designed counts:", designed_counts())
print("teacher_dose rows", len(TEACHER_DOSE), "rungs", sorted(TEACHER_DOSE["step"].astype(int).unique()) if "step" in TEACHER_DOSE.columns else [])
if not TEACHER_DOSE.empty and "p_target_mean" in TEACHER_DOSE.columns:
    y = TEACHER_DOSE["p_target_mean"]
    print("teacher p_target_mean range", float(y.min()), float(y.max()))
if not DOSE_RESP.empty and "delta" in DOSE_RESP.columns:
    d = drop_degenerate(DOSE_RESP.copy(), "dose_response (takeaways)")
    print("dose_response n", len(d), "delta range", float(d["delta"].min()) if not d.empty else None, float(d["delta"].max()) if not d.empty else None)
    if "overshoot" in d.columns and not d.empty:
        print("overshoot range", float(d["overshoot"].min()), float(d["overshoot"].max()))
if not RECT.empty and "delta" in RECT.columns:
    r = agg_rows(primary_rows(RECT))
    print("rect agg n", len(r), "families", family_key(r).value_counts(dropna=False).to_dict() if not r.empty else {})
    n512 = int((r["student_step"].astype(int) == plan.RECT512_STUDENT_RUNG).sum()) if not r.empty else 0
    print("rect rung-512 agg", n512)
if not RIDGE.empty and "ridge_gain" in RIDGE.columns:
    rg = drop_degenerate(RIDGE.copy(), "ridge (takeaways)")
    print("ridge_gain n", len(rg), "range", float(rg["ridge_gain"].min()) if not rg.empty else None, float(rg["ridge_gain"].max()) if not rg.empty else None)
    if "era" in rg.columns:
        print("ridge by era:\n", rg.groupby("era")["ridge_gain"].agg(["count", "mean"]))
if not PENALTY.empty and "penalty" in PENALTY.columns:
    pe = drop_degenerate(PENALTY.copy(), "penalty (takeaways)")
    print("penalty n", len(pe), "range", float(pe["penalty"].min()) if not pe.empty else None, float(pe["penalty"].max()) if not pe.empty else None)
if not EMISSIVITY.empty and "tau_minus_sigma" in EMISSIVITY.columns:
    em = EMISSIVITY
    if "base" in em.columns:
        em = em.loc[em["base"].astype(str) == "original"]
    print("original τ−σ range", float(em["tau_minus_sigma"].min()) if not em.empty else None, float(em["tau_minus_sigma"].max()) if not em.empty else None)
if not ONSET.empty:
    print("onset n", len(ONSET), "rung-512", int((ONSET["student_step"].astype(int) == plan.RECT512_STUDENT_RUNG).sum()) if "student_step" in ONSET.columns else None)
    if "onset_step" in ONSET.columns:
        print("onset_step unique", sorted(ONSET["onset_step"].dropna().unique().tolist()))
if not CONCORD.empty:
    print("concordance row:\n", CONCORD.to_string())
if not CAT.empty and "delta" in CAT.columns:
    c = agg_rows(primary_rows(CAT))
    print("cat agg n", len(c))
    if not c.empty and "student_step" in c.columns:
        cdiag = c.loc[c["student_step"].astype(int) == c["teacher_step"].astype(int)]
        if not cdiag.empty:
            imax = cdiag["delta"].idxmax()
            print("cat diagonal argmax step", int(cdiag.loc[imax, "student_step"]), "delta", float(cdiag.loc[imax, "delta"]))
if not QUAR.empty:
    print("quarantine rows", len(QUAR), "unique pairs", len(quarantine_pairs()))
print("coverage partial", COVERAGE.get("partial") if isinstance(COVERAGE, dict) else None)
print("CAMPAIGN_DONE", DONE_TEXT)
for cid in plan.CLAIM_IDS:
    rec = claim_record(cid)
    print(cid, rec.get("status"), rec.get("reason"))
""",
    ),
    (
        "markdown",
        """### Reading this figure

**Question answered:** What status and statistic did finalize persist for each of H1–H12, and what headline ranges do the loaded result frames actually contain?
**Conclusion from the plot:** PENDING_FRONTIER_VISUAL_REVIEW

**Axes.** The first cell is a table of `claims.json` records (id,
status, reason, statistic, threshold). The second cell reprints
counts and ranges from the loaded CSVs. Neither cell refits a
bootstrap or TOST.

**How the metric is calculated.** Claim rows are the persisted
adjudications. Frame headlines are `min`/`max`/`count`/`argmax` on
columns that already exist. Empty or partial frames print as such.
""",
    ),
    (
        "markdown",
        """## 15. How to extend

Knobs at the top of the setup cell:

| Knob | Default | Effect |
|---|---|---|
| `PROMPT_SET` | `plan.PRIMARY_PROMPT_SET` (`test`) | Documented filter when a frame carries `set`. Headline CSVs are primary `test`/`prob` unless you load `robustness/`. |
| `ARM` | `"all"` | Documented filter when a frame carries `arm`. Do not pool arms. |
| `SHOW_QUARANTINE` | `True` | Hatch (Fig3) on/off. Quarantine cells stay in every frame. |
| `ROBUSTNESS_SETS` | `plan.EVAL_PROMPT_SETS` | Which prompt-set names to look for under `results/robustness/`. |

To add a figure: insert a `## N. FigK` markdown cell, a small code
cell that starts with `scope(...)`, and a **Reading this figure**
cell. Re-run from the load cell downward after changing a knob.

Do not mix prompt sets into a headline statistic; do not fill blank
cells with 0; do not write under `output_root`; do not import
`ckptgrid2_metrics` to re-adjudicate claims. Geometry / CKA / probes /
weight-space wait for EleutherAI’s retrained weight-seeds.

Rebuild this notebook from the repo root:

```bash
uv run python -m scripts.build_ckptgrid2_notebook --out notebooks/gkd_ckpt_grid2_bear.ipynb
```
""",
    ),
]


def build(cells: list[tuple[str, str]]) -> nbformat.NotebookNode:
    """Assemble a nbformat-4 notebook from (cell_type, source) pairs.

    Args:
        cells: ``(cell_type, source)`` pairs; ``cell_type`` is
            ``markdown`` or ``code``.

    Returns:
        A validated-ready NotebookNode (caller runs ``nbformat.validate``).

    Raises:
        ValueError: An unknown ``cell_type``.
    """
    nb = new_notebook(
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        }
    )
    for i, (cell_type, source) in enumerate(cells):
        if cell_type == "markdown":
            cell = new_markdown_cell(source)
        elif cell_type == "code":
            cell = new_code_cell(source)
        else:
            raise ValueError(f"Unsupported cell_type: {cell_type!r}")
        cell["id"] = f"c{i:03d}"
        nb.cells.append(cell)
    return nb


def main() -> None:
    """Validate and write the ckptgrid2 bear analysis notebook."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT,
        help="Output .ipynb path (created or overwritten).",
    )
    args = parser.parse_args()
    nb = build(CELLS)
    nbformat.validate(nb)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, args.out)
    logger.info("Wrote {} ({} cells)", args.out, len(nb.cells))


if __name__ == "__main__":
    main()
