"""Step-budget callbacks: stop a run early while keeping its FULL LR schedule.

The early-stop arm asks "what has the student learned by step N of the *same* run?" --
so the schedule must be the one the full run would have had (warm-up ratio, cosine decay
and `num_train_epochs` all computed over the full step budget) while the run is cut short
at step N. Setting `max_steps`/`num_train_epochs` instead would recompute the schedule and
train a DIFFERENT model, so the cut is made by a callback:

- `StopAtStep` flips `control.should_training_stop` the moment `global_step` reaches the
  budget, after that step's optimizer update.
- `SaveAtSteps` flips `control.should_save` at an explicit list of steps. HF's fixed
  `save_steps` interval is pushed beyond the run by the caller
  (`teachers.sft_io.run_teacher_sft_completion_only`), so these are the ONLY intermediate
  checkpoints written.

Both are attached only when `TrainConfig.stop_step`/`save_at_steps` are set; a run that
leaves them None builds exactly the Trainer it always did.
"""

from __future__ import annotations

from transformers import TrainerCallback

# `save_steps` handed to HF when an explicit save list is in force: larger than any step
# budget this repo trains, so the interval rule can never fire on its own.
NEVER_SAVE_STEPS = 10**9


def resolve_save_steps(
    stop_step: int | None, save_at_steps: tuple[int, ...] | None
) -> tuple[int, ...]:
    """Validate the pair and return the steps to checkpoint at (the union with `stop_step`).

    Args:
        stop_step: Optimizer step to stop after (>= 1), or None.
        save_at_steps: Explicit, sorted, unique steps to checkpoint at (each >= 1 and,
            when `stop_step` is set, <= it), or None.

    Returns:
        The sorted union of `save_at_steps` and `stop_step` (empty when neither is set;
        `(stop_step,)` when only `stop_step` is), so the checkpoint at the stopping step
        always exists.

    Raises:
        ValueError: on a non-positive `stop_step`, an unsorted/duplicated/non-positive
            `save_at_steps`, or a save step past `stop_step`.
    """
    if stop_step is not None and stop_step < 1:
        raise ValueError(f"stop_step={stop_step} must be >= 1")
    steps = tuple(save_at_steps or ())
    if steps:
        if list(steps) != sorted(steps) or len(set(steps)) != len(steps):
            raise ValueError(f"save_at_steps={steps} must be sorted and unique")
        if steps[0] < 1:
            raise ValueError(f"save_at_steps={steps} must all be >= 1")
        if stop_step is not None and steps[-1] > stop_step:
            raise ValueError(f"save_at_steps={steps} must all be <= stop_step={stop_step}")
    if stop_step is not None:
        steps = tuple(sorted(set(steps) | {stop_step}))
    return steps


class StopAtStep(TrainerCallback):
    """Stop training immediately after optimizer step `stop_step`.

    Attributes:
        stopped_at_step: ``state.global_step`` at which training was stopped, or None if
            the callback never fired.
    """

    def __init__(self, stop_step: int) -> None:
        if stop_step < 1:
            raise ValueError(f"stop_step={stop_step} must be >= 1")
        self.stop_step = stop_step
        self.stopped_at_step: int | None = None

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.stop_step:
            self.stopped_at_step = state.global_step
            control.should_training_stop = True
        return control


class SaveAtSteps(TrainerCallback):
    """Checkpoint at exactly `steps` and at no other step.

    Attributes:
        saved_steps: The steps at which a save was requested, in order.
    """

    def __init__(self, steps) -> None:
        self.steps = frozenset(steps)
        self.saved_steps: list[int] = []

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in self.steps:
            control.should_save = True
            self.saved_steps.append(state.global_step)
        return control
