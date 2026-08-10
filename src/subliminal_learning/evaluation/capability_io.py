"""LM-Eval-Harness capability guardrail (M6b): 0-shot PolyPythia benchmarks.

Thin wrapper over `lm_eval.simple_evaluate` with `lm_eval.models.huggingface.HFLM`
(verified API), evaluating a teacher (or base model) on the log-likelihood benchmarks the
PolyPythia paper reports (`papers/md/polypythias_models_paper.md` §7/§14): LAMBADA
(openai), ARC-Easy, SciQ, 0-shot. A large drop vs the base model flags that all-token CE
finetuning damaged general capability.

The three checkpoint shapes load uniformly:

- LoRA:    ``HFLM(pretrained=base_repo, peft=str(ckpt), dtype="float32")``
- full-FT: ``HFLM(pretrained=str(ckpt), dtype="float32")``
- base:    ``HFLM(pretrained=base_repo, dtype="float32")``

`revision` pins the BASE repo to a Pythia training checkpoint (``step16000``), which the
checkpoint ladder needs to benchmark an untrained rung. `HFLM.__init__` takes
``revision: str | None = "main"`` and forwards it to `AutoConfig` / `AutoModel` /
`AutoTokenizer` (`lm_eval/models/huggingface.py`:75, :320, :348, :367). It also forwards
the same string to `PeftModel.from_pretrained` (:890); that is inert for a LOCAL adapter
dir, because `peft.utils.save_and_load.load_peft_weights` short-circuits on an existing
local weight file before any `revision` is consulted
(`peft/utils/save_and_load.py`:529).

Run only on a canonical subset of cells (see the verify-sweep config) -- it is far more
expensive than the preference/forced-choice/perplexity probes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from loguru import logger

DEFAULT_TASKS: tuple[str, ...] = ("lambada_openai", "arc_easy", "sciq")


@dataclass(frozen=True)
class BenchmarkResult:
    """0-shot benchmark metrics for one model.

    Attributes:
        tasks: task name -> {metric name (filter suffix stripped) -> value}, e.g.
            ``{"arc_easy": {"acc": 0.45, "acc_norm": 0.42}, "lambada_openai":
            {"acc": 0.35, "perplexity": 32.1}}``.
        num_fewshot: Shots used (0 for these).
    """

    tasks: dict[str, dict[str, float]]
    num_fewshot: int


def _extract_metrics(task_result: dict) -> dict[str, float]:
    """Flatten one task's ``simple_evaluate`` result to {metric: value}.

    lm-eval keys metrics as ``"{metric},{filter}"`` (e.g. ``"acc,none"``); the filter
    suffix is dropped and per-metric ``*_stderr`` entries are skipped.
    """
    out: dict[str, float] = {}
    for key, value in task_result.items():
        if key == "alias" or not isinstance(value, (int, float)):
            continue
        metric = key.split(",")[0]
        if metric.endswith("_stderr"):
            continue
        out[metric] = float(value)
    return out


def run_benchmarks(
    pretrained: str,
    device: str,
    peft_path: Path | None = None,
    tasks: tuple[str, ...] = DEFAULT_TASKS,
    dtype: str = "float32",
    batch_size: str | int = "auto",
    num_fewshot: int = 0,
    limit: int | None = None,
    revision: str | None = None,
) -> BenchmarkResult:
    """Evaluate one model on `tasks` (0-shot by default) via LM-Eval-Harness.

    Args:
        pretrained: Base repo id (LoRA/base) or full-FT checkpoint dir.
        device: Torch device, e.g. ``"cuda:0"``.
        peft_path: LoRA adapter dir, or ``None`` for full-FT/base.
        tasks: lm-eval task names.
        dtype: Model dtype string (``"float32"`` -- exact log-likelihoods).
        batch_size: lm-eval batch size (``"auto"`` picks the largest that fits).
        num_fewshot: Few-shot count (0 = PolyPythia's 0-shot protocol).
        limit: Optional per-task example cap (smoke tests only; leave ``None`` for the
            full benchmark).
        revision: Hub revision of `pretrained`, e.g. ``"step16000"``. ``None`` keeps
            upstream's ``"main"`` default, so every existing caller is unchanged. Only
            meaningful when `pretrained` is a repo id -- a full-FT checkpoint dir has no
            revisions. See the module docstring for the forwarding path.

    Returns:
        Per-task metrics.
    """
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(
        pretrained=str(pretrained),
        peft=str(peft_path) if peft_path is not None else None,
        dtype=dtype,
        device=device,
        batch_size=batch_size,
        **({} if revision is None else {"revision": revision}),
    )
    logger.info(
        "lm-eval: pretrained={} revision={} peft={} tasks={} nfewshot={} limit={}",
        pretrained, revision, peft_path, list(tasks), num_fewshot, limit,
    )
    results = lm_eval.simple_evaluate(
        model=lm, tasks=list(tasks), num_fewshot=num_fewshot, limit=limit,
    )
    task_metrics = {
        task: _extract_metrics(results["results"][task])
        for task in tasks
        if task in results["results"]
    }
    return BenchmarkResult(tasks=task_metrics, num_fewshot=num_fewshot)
