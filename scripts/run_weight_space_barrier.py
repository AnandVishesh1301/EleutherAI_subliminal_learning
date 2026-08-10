"""CLI: Frankle weight-space interpolation barriers (P5w).

Runs only on shared-weight-init pairs — the ``SI∩DS`` arm `ckpt_jobs` enumerates. A
``BASIS_BROKEN`` pair returns no barrier float: `weight_space_io.interpolation_barrier`
gates on `parameter_space_reason` (plan §6.5).

The loss at each interpolation coefficient is per-token CE over **held-out real Pile
text** (`evaluation.perplexity_io`), not the preference prompts: Frankle's barrier is
the rise in *test error* along the path (arXiv:1912.05671 §2), so scoring it on a
handful of trait-loaded elicitation prefixes would measure the trait, not instability.

Rows are written as the `weight_space` schema table into this shard's own
``parts/<shard>/`` dir; `pipelines.similarity_ladder_pipeline.merge_parts` is the
single writer that concatenates them into ``results/weight_space.csv``.

    uv run python -m scripts.run_weight_space_barrier \\
        --config configs/experiment/ckpt_ladder_geometry.yaml \\
        --rungs 2000 --pair original,data-seed1 --device cuda:0
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import math

import typer
from dotenv import load_dotenv
from loguru import logger
from transformers import GPTNeoXForCausalLM

from pipelines.similarity_ladder_pipeline import PARTS_SUBDIR
from subliminal_learning.campaign import ckpt_plan as plan
from subliminal_learning.cka.pairing import basis_validity
from subliminal_learning.cka.results_io import similarity_run_dir, write_result_table
from subliminal_learning.cka.weight_space import parameter_space_reason
from subliminal_learning.cka.weight_space_io import (
    BarrierMeasurement,
    interpolation_barrier,
)
from subliminal_learning.config import load_config
from subliminal_learning.evaluation.perplexity_io import (
    corpus_perplexity,
    load_pile_exemplars,
    pile_bank_sha256,
)
from subliminal_learning.models.loading_io import load_model, load_tokenizer
from subliminal_learning.models.registry import PythiaModelSpec
from subliminal_learning.teachers.model_specs import parse_base_models
from subliminal_learning.utils.logging_setup import configure_loguru

app = typer.Typer(add_completion=False, help=__doc__)

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
BARRIER_METRIC = "frankle_barrier"
# Per-alpha loss, so plan §10's figure 13 can draw the curve and not just its peak.
INTERP_LOSS_METRIC = "interp_loss_alpha"


def _parse_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in raw.split(",") if x.strip())


def _spec(label: str, rung: int) -> PythiaModelSpec:
    specs = parse_base_models(label)
    if len(specs) != 1:
        raise ValueError(f"label {label!r} resolved to {len(specs)} specs")
    return replace(specs[0], revision=f"step{rung}")


def _row(
    *,
    a_label: str,
    b_label: str,
    rung: int,
    metric: str,
    value: float | None,
    skip_reason: str | None,
    n_tensors: int,
) -> dict:
    """One `results_io.WEIGHT_SPACE_COLUMNS` row for this shard's pair.

    ``animal`` is empty: a barrier is measured between two *base* checkpoints, so no
    trait is involved. ``basis_validity`` is derived from the labels rather than passed
    in, so a row cannot claim a validity its own endpoints contradict.
    """
    return {
        "animal": "",
        "model_a": a_label,
        "model_b": b_label,
        "base_revision": f"step{rung}",
        "step": rung,
        "basis_validity": str(basis_validity(a_label, b_label)),
        "metric": metric,
        "value": value,
        "skip_reason": skip_reason,
        "n_tensors": n_tensors,
    }


@app.command()
def main(
    config: Path = typer.Option(..., help="Geometry YAML (CLI contract with ckpt_jobs)."),
    rungs: str = typer.Option(..., help="Comma-separated rung integers."),
    pair: str = typer.Option(..., help="student,teacher revision-free labels."),
    device: str = typer.Option("cuda:0"),
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    model_config: Path = Path("configs/model/pythia.yaml"),
) -> None:
    """Measure the Frankle barrier for one ``(rung, pair)`` shard."""
    del config  # Path contract only; barrier loads Hub endpoints directly.
    load_dotenv()
    configure_loguru()
    cfg = load_config(paths_config, model_config)
    student_label, teacher_label = [p.strip() for p in pair.split(",", 1)]
    wanted = _parse_ints(rungs)
    shard = f"p5w__{student_label}__{teacher_label}__" + "-".join(
        str(r) for r in wanted
    )
    run_dir = similarity_run_dir(cfg.paths.output_root, plan.GEOM_RUN_SET_ID)
    parts_dir = run_dir / PARTS_SUBDIR / shard
    cache = cfg.paths.hf_cache_root
    corpus = load_pile_exemplars()
    logger.info(
        "barrier eval corpus: {} registers, bank sha {}",
        len(corpus),
        pile_bank_sha256(),
    )

    rows: list[dict] = []
    for rung in wanted:
        a_label = f"{student_label}@step{rung}"
        b_label = f"{teacher_label}@step{rung}"
        reason = parameter_space_reason(basis_validity(a_label, b_label))
        if reason is not None:
            logger.info("skip barrier {} vs {}: {}", a_label, b_label, reason)
            rows.append(
                _row(
                    a_label=a_label,
                    b_label=b_label,
                    rung=rung,
                    metric=BARRIER_METRIC,
                    value=None,
                    skip_reason=reason,
                    n_tensors=0,
                )
            )
            continue
        spec_a = _spec(student_label, rung)
        model_a = load_model(spec_a, cache, plan.HUB_ORG, device=device)
        state_a = {k: v.detach().cpu() for k, v in model_a.state_dict().items()}
        if student_label == teacher_label:
            state_b = {k: v.clone() for k, v in state_a.items()}
        else:
            spec_b = _spec(teacher_label, rung)
            model_b = load_model(spec_b, cache, plan.HUB_ORG, device=device)
            state_b = {k: v.detach().cpu() for k, v in model_b.state_dict().items()}
            del model_b
        tokenizer = load_tokenizer(spec_a, cache, plan.HUB_ORG)

        def eval_loss(model: GPTNeoXForCausalLM) -> float:
            """Token-weighted per-token CE (nats) on the held-out Pile bank.

            `corpus_perplexity` pools ``exp(sum nll / sum tokens)``, so its log is the
            mean CE -- the additive scale a barrier is defined on. Reading the
            perplexity itself would make the barrier a difference of exponentials.
            """
            model.eval()
            ppl = corpus_perplexity(model, tokenizer, device, corpus)
            return math.log(ppl.overall)

        result: BarrierMeasurement = interpolation_barrier(
            model_a,
            state_a,
            state_b,
            eval_loss,
            label_a=a_label,
            label_b=b_label,
            alphas=ALPHAS,
        )
        rows.append(
            _row(
                a_label=a_label,
                b_label=b_label,
                rung=rung,
                metric=BARRIER_METRIC,
                value=result.barrier,
                skip_reason=result.skip_reason,
                n_tensors=result.n_tensors,
            )
        )
        rows.extend(
            _row(
                a_label=a_label,
                b_label=b_label,
                rung=rung,
                metric=f"{INTERP_LOSS_METRIC}{alpha:g}",
                value=loss,
                skip_reason=None,
                n_tensors=result.n_tensors,
            )
            for alpha, loss in zip(result.alphas, result.losses, strict=True)
        )
        del model_a, state_a, state_b

    part = write_result_table(parts_dir, "weight_space", rows)
    logger.info("wrote {} weight-space rows -> {}", len(rows), part)


if __name__ == "__main__":
    app()
