"""CLI: Stage A0 saved-step1 parameter-equality check (CPU only).

Loads the ``step1`` revision of pythia-160m (original), pythia-160m-data-seed1,
and pythia-160m-weight-seed1 and compares state dicts tensor-by-tensor
(`torch.equal` — exact numeric equality, no tolerance). Released ws1–3 step0
checkpoints contain seed-specific first-gradient parameters;
by step1, parameters reset to the original values while strong independent
reconstruction evidence indicates seed-specific Adam history is retained. See
``run_logs/ws_origin_investigation_20260913/OPTIMIZER_FINDINGS.md``. The verified
original/data-seed1/weight-seed1 step1 parameters are identical, but this legacy
filename checks saved step1 parameters only, not true initialization, optimizer
state, first forward/backward behavior, or data order. Step0 is not orphaned.
The verdict JSON lands at
``{output_root}/gkd_students/{run_set_id}/logs/init_equality.json``, which the
campaign driver's A0 gate requires. Fails loud (rc=2) when the observed
equalities contradict the prediction — a wrong repo/revision would silently
invalidate every cross-init conclusion downstream.

    uv run python -m scripts.check_step0_init_equality
"""

from __future__ import annotations

import itertools
from pathlib import Path

import torch
import typer
import yaml
from dotenv import load_dotenv
from loguru import logger
from transformers import GPTNeoXForCausalLM

from subliminal_learning.campaign import plan
from subliminal_learning.gkd.layout import GKD_STUDENTS_SUBDIR
from subliminal_learning.models.registry import resolve_hf_repo_id
from subliminal_learning.teachers.model_specs import parse_base_models
from subliminal_learning.training.run_set import run_set_subdir
from subliminal_learning.utils.io import atomic_write_json
from subliminal_learning.utils.logging_setup import configure_loguru
from subliminal_learning.utils.run_io import utc_now_iso

LABELS = ("original", "data-seed1", "weight-seed1")
REVISION = "step1"
# The verified original/data-seed1/weight-seed1 saved step1 parameters are identical.
EXPECTED_EQUAL = {
    ("original", "data-seed1"): True,
    ("original", "weight-seed1"): True,
    ("data-seed1", "weight-seed1"): True,
}


def _state_dict(label: str, hub_org: str) -> dict[str, torch.Tensor]:
    spec = parse_base_models(label)[0]
    repo = resolve_hf_repo_id(spec, hub_org)
    logger.info("loading {}@{} (CPU) ...", repo, REVISION)
    model = GPTNeoXForCausalLM.from_pretrained(repo, revision=REVISION)
    return {k: v.cpu() for k, v in model.state_dict().items()}


def main(
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    max_mismatch_keys: int = typer.Option(
        12, help="Cap on mismatched tensor names recorded per pair."
    ),
) -> None:
    """Compare saved step1 parameters of original vs data-seed1 vs weight-seed1."""
    load_dotenv()
    configure_loguru()
    paths = yaml.safe_load(Path(paths_config).read_text())
    logs_dir = (
        Path(paths["output_root"])
        / run_set_subdir(GKD_STUDENTS_SUBDIR, plan.RUN_SET_ID)
        / "logs"
    )
    states = {label: _state_dict(label, plan.HUB_ORG) for label in LABELS}

    pairs = []
    all_as_expected = True
    for a, b in itertools.combinations(LABELS, 2):
        sa, sb = states[a], states[b]
        if set(sa) != set(sb):
            raise RuntimeError(
                f"{a} vs {b}: state-dict key sets differ — wrong architecture/repo"
            )
        mismatched = [k for k in sorted(sa) if not torch.equal(sa[k], sb[k])]
        equal = not mismatched
        expected = EXPECTED_EQUAL[(a, b)]
        as_expected = equal == expected
        all_as_expected &= as_expected
        pairs.append(
            {
                "pair": [a, b],
                "n_tensors": len(sa),
                "n_mismatched": len(mismatched),
                "equal": equal,
                "expected_equal": expected,
                "as_expected": as_expected,
                "mismatched_keys": mismatched[:max_mismatch_keys],
            }
        )
        logger.info(
            "{} vs {}: {} ({}/{} tensors differ; expected {})",
            a,
            b,
            "EQUAL" if equal else "DIFFERENT",
            len(mismatched),
            len(sa),
            "equal" if expected else "different",
        )

    report = {
        "at": utc_now_iso(),
        "revision": REVISION,
        "scope": "saved step1 parameter equality only; not initialization or optimizer-state equality",
        "hub_org": plan.HUB_ORG,
        "pairs": pairs,
        "verdict": "as_expected" if all_as_expected else "UNEXPECTED",
    }
    logs_dir.mkdir(parents=True, exist_ok=True)
    out = logs_dir / plan.INIT_EQUALITY_REPORT
    atomic_write_json(out, report)
    logger.info("init-equality report -> {}", out)
    if not all_as_expected:
        logger.error(
            "saved step1 parameter equality contradicts the verified PolyPythias result — do NOT "
            "launch the campaign; check repo ids / revisions."
        )
        raise typer.Exit(code=2)


if __name__ == "__main__":
    typer.run(main)
