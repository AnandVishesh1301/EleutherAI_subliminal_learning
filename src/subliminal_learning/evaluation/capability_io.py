"""LM-Eval-Harness capability guardrail (M6b): 0-shot PolyPythia benchmarks.

Thin wrapper over `lm_eval.simple_evaluate` with `lm_eval.models.huggingface.HFLM`
(verified API), evaluating a teacher (or base model) on the log-likelihood benchmarks the
PolyPythia paper reports (`papers/md/polypythias_models_paper.md` §7/§14): LAMBADA
(openai), ARC-Easy, SciQ, 0-shot. A large drop vs the base model flags that all-token CE
finetuning damaged general capability.

The three checkpoint shapes load uniformly:

- LoRA:    ``HFLM(pretrained=base_repo, peft=str(ckpt), dtype="float32")``
- LoRA with ``modules_to_save``: the merged model (`merged_adapter_model`) passed to HFLM
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


def adapter_saves_modules(peft_path: Path) -> bool:
    """Whether the adapter also stores whole modules (PEFT ``modules_to_save``)."""
    from peft import PeftConfig

    return bool(PeftConfig.from_pretrained(str(peft_path)).modules_to_save)


def merged_adapter_model(base, peft_path: Path, n_tokens: int):  # noqa: ANN001, ANN201
    """`base` with the adapter at `peft_path` merged, vocabulary cut to `n_tokens` rows.

    ``HFLM(peft=...)`` resizes the base to the tokenizer length before loading the
    adapter (`lm_eval/models/huggingface.py`:883-889), so a saved full ``embed_out``
    (50,304 rows) no longer fits the 50,277-row model. Merging first and resizing after
    gives the same vocabulary cut every other adapter is benchmarked with; the merged
    forward differs from PEFT's unmerged one only by fp32 rounding.
    """
    from peft import PeftModel

    model = PeftModel.from_pretrained(base, str(peft_path)).merge_and_unload()
    if model.config.vocab_size != n_tokens:
        model.resize_token_embeddings(n_tokens)
    return model.eval()


def _merged_hflm(
    pretrained: str,
    peft_path: Path,
    device: str,
    dtype: str,
    batch_size: str | int,
    revision: str | None,
):  # noqa: ANN202
    """HFLM over the merged base + adapter, base and tokenizer loaded as HFLM would."""
    import torch
    from lm_eval.models.huggingface import HFLM
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rev = {} if revision is None else {"revision": revision}
    tokenizer = AutoTokenizer.from_pretrained(pretrained, use_fast=True, **rev)
    base = AutoModelForCausalLM.from_pretrained(
        pretrained, torch_dtype=getattr(torch, dtype), **rev
    )
    model = merged_adapter_model(base, peft_path, len(tokenizer)).to(device)
    return HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)


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

    if peft_path is not None and adapter_saves_modules(peft_path):
        lm = _merged_hflm(
            str(pretrained), peft_path, device, dtype, batch_size, revision
        )
    else:
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
