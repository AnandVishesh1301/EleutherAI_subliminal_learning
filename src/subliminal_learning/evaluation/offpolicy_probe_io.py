"""In-training trajectory probe for off-policy SFT: one ep20 run -> its dose curve.

The off-policy counterpart of `gkd.probe_io.GkdProbeCallback`, and the same mechanism:
a `TrainerCallback` that measures the STUDENT between optimizer steps (after
``optimizer.step()``, under ``no_grad``, weights read only), appending to
``training_metrics.jsonl`` world-process-zero and flushed per write.

Two things differ, both forced by what the off-policy campaign needs:

- probes fire at fixed **fractions of the run** (`resolve_probe_steps`) rather than every
  ``probe_every`` steps, because the quantity of interest is transfer against training
  progress: measured off-policy transfer at 410m peaks near 20% of training and decays
  3.7x by the ep20 endpoint, so an endpoint-only reading reports a decayed signal as a
  null (`campaign.offxsize2_plan`, H1-H8);
- one pass scores **several target animals**, because the base/rng controls carry no
  animal and are read for every campaign animal against the same weights.

Saved checkpoints are deliberately not the mechanism: four r128 adapters per cell at 410m
would cost ~120 GB across the campaign, while this writes metrics only.

**RNG safety.** ``ftseed`` reproducibility requires that a probe consume nothing from the
training RNG stream. Exactly as in `gkd.probe_io`: the model is switched to ``eval()``
for the probe (so LoRA/attention dropout is the identity and draws no randomness) and
restored afterwards, and every measurement is a deterministic forward pass plus a softmax
and a comparison (`evaluation.preference_prob_io`) -- no sampling, no ``generate``, no
``torch.rand*``. `tests/unit/test_offpolicy_probe.py` asserts the torch RNG state is
bit-identical across a probe.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean

import pandas as pd
import torch
from loguru import logger
from transformers import PreTrainedTokenizerBase, TrainerCallback

from subliminal_learning.evaluation.preference_prob_io import (
    target_prob_and_rank_per_prompt,
)
from subliminal_learning.utils.io import rglob_follow

METRICS_FILENAME = "training_metrics.jsonl"

PROBE_TRAJECTORY_COLUMNS: tuple[str, ...] = (
    "run_dir",
    "ftseed",
    "animal",
    "step",
    "step_frac",
    "epoch_equivalent",
    "p_target",
    "rank_mean",
    "rank1_frac",
    "n_prompts",
)


def resolve_probe_steps(
    step_fractions: Sequence[float], max_steps: int
) -> dict[int, float]:
    """Optimizer step -> the nominal training fraction probed at that step.

    ``step = ceil(frac * max_steps)`` clamped to ``[1, max_steps]``: ceil so a probe never
    reads *less* training than its fraction names, and so ``frac == 1.0`` lands exactly on
    the final optimizer step.

    A run short enough to map several fractions onto one step keeps the **largest** of
    them: at that step the completed fraction is >= every colliding fraction, so the
    largest is the only label that does not overstate how early the reading is (labelling
    step 1 of 2 as ``0.10`` would put a half-trained measurement into the 10% column of a
    cross-cell trajectory). The dropped fractions are logged, never silently absorbed.

    Args:
        step_fractions: Fractions of the run to probe, each in ``(0, 1]`` (e.g.
            `campaign.offxsize2_plan.PROBE_STEP_FRACTIONS`). Order is irrelevant.
        max_steps: Total optimizer steps in the run (``TrainerState.max_steps``).

    Returns:
        step -> nominal fraction, one entry per distinct resolved step.

    Raises:
        ValueError: `max_steps` < 1, or a fraction outside ``(0, 1]``.
    """
    if max_steps < 1:
        raise ValueError(f"probe needs max_steps >= 1; got {max_steps}")
    resolved: dict[int, float] = {}
    for frac in step_fractions:
        if not 0.0 < float(frac) <= 1.0:
            raise ValueError(f"probe fraction {frac} is outside (0, 1]")
        step = min(max_steps, max(1, math.ceil(float(frac) * max_steps)))
        previous = resolved.get(step)
        if previous is not None:
            logger.warning(
                "probe fractions {} and {} both resolve to step {} of {}; keeping {}",
                previous,
                frac,
                step,
                max_steps,
                max(previous, float(frac)),
            )
        resolved[step] = max(previous or 0.0, float(frac))
    return resolved


class OffPolicyProbeCallback(TrainerCallback):
    """Fraction-scheduled, multi-animal student probe -> ``training_metrics.jsonl``."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        device: str,
        target_ids: Mapping[str, int],
        probe_prompts: tuple[str, ...],
        step_fractions: Sequence[float],
        metrics_path: Path,
    ) -> None:
        """See module docstring.

        Args:
            tokenizer: Student tokenizer (with ``pad_token``).
            device: Torch device the student is on.
            target_ids: animal -> token id of ``" {animal}"``
                (`preference_prob_io.target_token_id`). Every animal is scored on the
                same weights in one pass; row order follows this mapping.
            probe_prompts: "...the" probe prompts, shared by every animal.
            step_fractions: Fractions of the run to probe (`resolve_probe_steps`).
            metrics_path: ``training_metrics.jsonl`` path (append, world-process-zero).

        Raises:
            ValueError: `target_ids` or `probe_prompts` is empty.
        """
        if not target_ids:
            raise ValueError("the probe needs at least one target animal")
        if not probe_prompts:
            raise ValueError("the probe needs at least one prompt")
        self.tokenizer = tokenizer
        self.device = device
        self.target_ids = dict(target_ids)
        self.probe_prompts = probe_prompts
        self.step_fractions = tuple(float(f) for f in step_fractions)
        self.metrics_path = Path(metrics_path)
        self._schedule: dict[int, float] = {}
        self._num_train_epochs = 0.0

    def probe_once(
        self, model, *, step: int, step_frac: float, epoch_equivalent: float
    ) -> list[dict]:  # noqa: ANN001
        """Measure every target animal on the current weights and append one row each.

        Args:
            model: Student (base or PEFT-wrapped) on `self.device`.
            step: Global step to record (``0`` = init baseline before train).
            step_frac: Nominal fraction of the run this step represents.
            epoch_equivalent: ``step_frac`` expressed in epochs.

        Returns:
            The written records, one per target animal.
        """
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                measured = {
                    animal: target_prob_and_rank_per_prompt(
                        model,
                        self.tokenizer,
                        self.device,
                        self.probe_prompts,
                        target_id,
                    )
                    for animal, target_id in self.target_ids.items()
                }
        finally:
            if was_training:
                model.train()

        records: list[dict] = []
        for animal, per_prompt in measured.items():
            ranks = [rank for _, rank in per_prompt.values()]
            record = {
                "step": int(step),
                "step_frac": float(step_frac),
                "epoch_equivalent": float(epoch_equivalent),
                "animal": animal,
                "p_target": fmean(p for p, _ in per_prompt.values()),
                "rank_mean": fmean(float(r) for r in ranks),
                "rank1_frac": fmean(1.0 if r == 1 else 0.0 for r in ranks),
                "n_prompts": len(per_prompt),
            }
            self._append(record)
            records.append(record)
        return records

    def on_train_begin(self, args, state, control, model=None, **kwargs):  # noqa: ANN001
        """Resolve the schedule against the real step count, then write the step-0 row.

        ``TrainerState.max_steps`` / ``num_train_epochs`` are set immediately before this
        event, so this is the first point at which a fraction has a step to land on. The
        init baseline is written here rather than by the caller (the GKD pipeline's
        pre-train ``probe_once(..., step=0)``) because the model this event hands back is
        already the PEFT-wrapped student. Skipped when the file already holds rows, so a
        mid-run resume does not restate a baseline it cannot re-measure.
        """
        self._schedule = resolve_probe_steps(self.step_fractions, int(state.max_steps))
        self._num_train_epochs = float(state.num_train_epochs)
        if model is None or not state.is_world_process_zero:
            return
        if self.metrics_path.exists() and self.metrics_path.stat().st_size:
            return
        self.probe_once(model, step=0, step_frac=0.0, epoch_equivalent=0.0)

    def on_step_end(self, args, state, control, model=None, **kwargs):  # noqa: ANN001
        step_frac = self._schedule.get(int(state.global_step))
        if step_frac is None or model is None or not state.is_world_process_zero:
            return
        self.probe_once(
            model,
            step=int(state.global_step),
            step_frac=step_frac,
            epoch_equivalent=step_frac * self._num_train_epochs,
        )

    def _append(self, record: dict) -> None:
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()


# ---- loaders -----------------------------------------------------------------


def _read_records(path: Path) -> list[dict]:
    """Parse one ``training_metrics.jsonl`` (skips a torn/in-flight trailing line)."""
    records: list[dict] = []
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return records
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def load_offpolicy_probe_trajectories(
    students_root: Path, filename: str = METRICS_FILENAME
) -> pd.DataFrame:
    """One row per (probe, animal) from every ``ftseed{S}`` leaf under `students_root`.

    The off-policy student tree has three layouts of different depth
    (`students.layout`), so leaves are found by their ``ftseed{S}`` level rather than by
    a fixed-depth glob, and identity beyond ``run_dir`` / ``ftseed`` is left to the
    caller's own path/manifest attribution (`pipelines.offxsize_results_pipeline`
    ``_leaf_identity``) -- this loader re-derives none of it. Symlinked subtrees are
    followed (`utils.io.rglob_follow`): a cold-archived branch must not go silently
    missing from a trajectory.

    Args:
        students_root: `students.layout.STUDENTS_SUBDIR` root to walk.
        filename: Per-leaf metrics filename
            (`campaign.offxsize2_plan.TRAINING_METRICS_FILE`).

    Returns:
        Long DataFrame with `PROBE_TRAJECTORY_COLUMNS`; empty (with those columns) when
        nothing has been probed yet.
    """
    rows: list[dict] = []
    for path in rglob_follow(students_root, f"ftseed*/{filename}"):
        run_dir = path.parent
        ftseed = int(run_dir.name.removeprefix("ftseed"))
        for rec in _read_records(path):
            rows.append(
                {
                    "run_dir": str(run_dir),
                    "ftseed": ftseed,
                    **{c: rec.get(c) for c in PROBE_TRAJECTORY_COLUMNS[2:]},
                }
            )
    return pd.DataFrame(rows, columns=list(PROBE_TRAJECTORY_COLUMNS))


def peak_columns(metric: str = "p_target") -> list[str]:
    """Column order `probe_peak_vs_endpoint` returns for `metric`."""
    return [
        "run_dir",
        "ftseed",
        "animal",
        "n_probes",
        f"baseline_{metric}",
        f"peak_{metric}",
        "peak_step",
        "peak_step_frac",
        "peak_epoch_equivalent",
        f"endpoint_{metric}",
        "endpoint_step",
        "peak_over_endpoint",
    ]


def probe_peak_vs_endpoint(
    trajectories: pd.DataFrame, metric: str = "p_target"
) -> pd.DataFrame:
    """Per (run_dir, animal): the untrained baseline, the in-training peak, the endpoint.

    ``peak_over_endpoint`` is the quantity the campaign reads a trajectory for: at 410m it
    is ~3.7 against ~1.16 at 160m, which is why an endpoint-only cell reports a decayed
    signal as a null.

    The step-0 row is the UNTRAINED prior and is excluded from both the peak and the
    endpoint -- otherwise a cell whose prior beats every trained step would report its own
    baseline as its "peak transfer". It is carried as ``baseline_{metric}`` so the dose
    curve can still be read against it.

    Args:
        trajectories: Output of `load_offpolicy_probe_trajectories`.
        metric: Probe column to summarise.

    Returns:
        One row per (run_dir, animal) with `peak_columns`; leaves holding only a step-0
        baseline are dropped (there is no trajectory to summarise).
    """
    columns = peak_columns(metric)
    if trajectories.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict] = []
    for (run_dir, animal), group in trajectories.groupby(
        ["run_dir", "animal"], sort=True
    ):
        baseline = group[group["step"] == 0]
        trained = group[group["step"] > 0].sort_values("step")
        if trained.empty or trained[metric].isna().all():
            continue
        peak = trained.loc[trained[metric].idxmax()]
        endpoint = trained.iloc[-1]
        endpoint_value = float(endpoint[metric])
        rows.append(
            {
                "run_dir": run_dir,
                "ftseed": int(group.iloc[0]["ftseed"]),
                "animal": animal,
                "n_probes": int(len(trained)),
                f"baseline_{metric}": (
                    float(baseline.iloc[0][metric]) if len(baseline) else float("nan")
                ),
                f"peak_{metric}": float(peak[metric]),
                "peak_step": int(peak["step"]),
                "peak_step_frac": float(peak["step_frac"]),
                "peak_epoch_equivalent": float(peak["epoch_equivalent"]),
                f"endpoint_{metric}": endpoint_value,
                "endpoint_step": int(endpoint["step"]),
                "peak_over_endpoint": (
                    float(peak[metric]) / endpoint_value
                    if endpoint_value > 0
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)
