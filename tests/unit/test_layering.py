"""Layering guard: SDF/GKD domain modules must stay torch-/IO-free (ARCHITECTURE §7).

The pure-domain modules (`config`, `layout`, `filtering`, `analysis.stats`,
`analysis.decomposition`) define repo
IDs, path tags, screening logic and claim statistics that unit tests exercise on CPU with
no heavy deps. If one of them grows a ``torch`` / ``transformers`` / ``trl`` / ``peft`` /
``datasets`` import (directly or transitively), the domain layer is no longer cheap to
import or test and the inversion in ARCHITECTURE §7 is broken. Each module is imported in
a *fresh* interpreter so a forbidden transitive import shows up in that process's
``sys.modules``.

`_TORCH_DOMAIN_MODULES` is the explicit exemption list: domain math that is inherently
``torch`` (per-token losses, Gram matrices, subspace bases). Those modules are tested for
numerics, not purity — but they are listed rather than merely omitted, so a module that
grows an accidental torch import is a deliberate decision instead of a silent one.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

_DOMAIN_MODULES = (
    "subliminal_learning.sdf.config",
    "subliminal_learning.sdf.layout",
    "subliminal_learning.gkd.config",
    "subliminal_learning.gkd.layout",
    "subliminal_learning.gkd.filtering",
    "subliminal_learning.gkd.prompt_pool",
    "subliminal_learning.gkd.judge_layout",
    "subliminal_learning.gkd.judge_metrics",
    "subliminal_learning.gkd.judge_sampling",
    "subliminal_learning.campaign.plan",
    "subliminal_learning.campaign.pairs",
    "subliminal_learning.campaign.selection",
    "subliminal_learning.campaign.config_gen",
    "subliminal_learning.campaign.expected",
    "subliminal_learning.analysis.stats",
    "subliminal_learning.analysis.decomposition",
    "subliminal_learning.evaluation.calibration",
    "subliminal_learning.cka.pairing",
    "subliminal_learning.campaign.ckpt_plan",
    "subliminal_learning.campaign.ckpt_pairs",
    "subliminal_learning.campaign.ckpt_replication",
    "subliminal_learning.campaign.ckpt_selection",
    "subliminal_learning.campaign.ckpt_config_gen",
    "subliminal_learning.campaign.ckpt_jobs",
    "subliminal_learning.campaign.ckptgrid_plan",
    "subliminal_learning.campaign.ckptgrid_pairs",
    "subliminal_learning.campaign.ckptgrid_config_gen",
    "subliminal_learning.campaign.ckptgrid_metrics",
    "subliminal_learning.campaign.ckptgrid_selection",
    "subliminal_learning.campaign.ckptgrid_jobs",
    "subliminal_learning.campaign.xsize_plan",
    "subliminal_learning.models.hf_spec",
    "subliminal_learning.campaign.xfam_models",
    "subliminal_learning.campaign.xfam_plan",
    "subliminal_learning.campaign.xfam_pairs",
    "subliminal_learning.campaign.xfam_selection",
    "subliminal_learning.campaign.xfam_metrics",
    "subliminal_learning.campaign.xfam_config_gen",
    "subliminal_learning.campaign.xfam_jobs",
    "subliminal_learning.campaign.xfam2_models",
    "subliminal_learning.campaign.xfam2_plan",
    "subliminal_learning.campaign.xfam2_pairs",
    "subliminal_learning.campaign.xfam2_metrics",
    "subliminal_learning.campaign.xfam2_config_gen",
    "subliminal_learning.campaign.xfam2_jobs",
    "subliminal_learning.campaign.xsize_pairs",
    "subliminal_learning.campaign.xsize_selection",
    "subliminal_learning.campaign.xsize_metrics",
    "subliminal_learning.campaign.xsize_config_gen",
    "subliminal_learning.campaign.xsize_jobs",
    "subliminal_learning.campaign.ckptgrid2_plan",
    "subliminal_learning.campaign.ckptgrid2_pairs",
    "subliminal_learning.campaign.ckptgrid2_expected",
    "subliminal_learning.campaign.ckptgrid2_metrics",
    "subliminal_learning.campaign.ckptgrid2_selection",
    "subliminal_learning.campaign.ckptgrid2_config_gen",
    "subliminal_learning.campaign.ckptgrid2_jobs",
    "subliminal_learning.campaign.offxsize_plan",
    "subliminal_learning.campaign.offxsize_layout",
    "subliminal_learning.campaign.offxsize_pairs",
    "subliminal_learning.campaign.offxsize_numbers",
    "subliminal_learning.campaign.offxsize_selection",
    "subliminal_learning.campaign.offxsize_metrics",
    "subliminal_learning.campaign.offxsize_jobs",
    "subliminal_learning.campaign.topk_plan",
    "subliminal_learning.campaign.topk_pairs",
    "subliminal_learning.campaign.topk_metrics",
    "subliminal_learning.campaign.topk_config_gen",
    "subliminal_learning.campaign.topk_jobs",
    "subliminal_learning.campaign.opdlim_plan",
    "subliminal_learning.campaign.opdlim_pairs",
    "subliminal_learning.campaign.opdlim_config_gen",
    "subliminal_learning.campaign.opdlim_metrics",
    "subliminal_learning.campaign.opdlim_jobs",
    "subliminal_learning.campaign.strictscreen_plan",
    "subliminal_learning.campaign.strictscreen_config_gen",
    "subliminal_learning.campaign.strictscreen_jobs",
    "subliminal_learning.campaign.strictscreen_metrics",
    "subliminal_learning.campaign.strictscreen_audit_plan",
    "subliminal_learning.gkd.judge_claims",
)
_STRICTSCREEN_PURE = (
    "subliminal_learning.campaign.strictscreen_plan",
    "subliminal_learning.campaign.strictscreen_config_gen",
    "subliminal_learning.campaign.strictscreen_audit_plan",
)
_OPDLIM_PURE = (
    "subliminal_learning.campaign.opdlim_plan",
    "subliminal_learning.campaign.opdlim_pairs",
    "subliminal_learning.campaign.opdlim_config_gen",
    "subliminal_learning.campaign.opdlim_metrics",
)
_TOPK_PURE = (
    "subliminal_learning.campaign.topk_plan",
    "subliminal_learning.campaign.topk_pairs",
    "subliminal_learning.campaign.topk_metrics",
    "subliminal_learning.campaign.topk_config_gen",
)
_STEERING_PURE = (
    "subliminal_learning.steering.eas",
    "subliminal_learning.steering.gkd_eas",
)
_OFFXSIZE_PURE = (
    "subliminal_learning.campaign.offxsize_plan",
    "subliminal_learning.campaign.offxsize_layout",
    "subliminal_learning.campaign.offxsize_pairs",
    "subliminal_learning.campaign.offxsize_numbers",
    "subliminal_learning.campaign.offxsize_selection",
    "subliminal_learning.campaign.offxsize_metrics",
)
_XFAM_PURE = (
    "subliminal_learning.models.hf_spec",
    "subliminal_learning.campaign.xfam_models",
    "subliminal_learning.campaign.xfam_plan",
    "subliminal_learning.campaign.xfam_pairs",
    "subliminal_learning.campaign.xfam_selection",
    "subliminal_learning.campaign.xfam_metrics",
    "subliminal_learning.campaign.xfam_config_gen",
    "subliminal_learning.campaign.xfam2_models",
    "subliminal_learning.campaign.xfam2_plan",
    "subliminal_learning.campaign.xfam2_pairs",
    "subliminal_learning.campaign.xfam2_metrics",
    "subliminal_learning.campaign.xfam2_config_gen",
)
_XSIZE_PURE = (
    "subliminal_learning.campaign.xsize_plan",
    "subliminal_learning.campaign.xsize_pairs",
    "subliminal_learning.campaign.xsize_selection",
    "subliminal_learning.campaign.xsize_metrics",
    "subliminal_learning.campaign.xsize_config_gen",
)
_FORBIDDEN = ("torch", "transformers", "trl", "peft", "datasets")

_TORCH_DOMAIN_MODULES = (
    "subliminal_learning.gkd.losses",
    "subliminal_learning.gkd.topk_census",
    "subliminal_learning.cka.minibatch",
    "subliminal_learning.cka.metrics",
    "subliminal_learning.steering.gkd_eas",
)

# Pure geometry modules in `cka/`: torch is allowed, disk/GPU orchestration is not.
_PURE_CKA_MODULES = (
    "subliminal_learning.cka.metrics",
    "subliminal_learning.cka.pairing",
    "subliminal_learning.cka.probes",
    "subliminal_learning.cka.weight_space",
)


def _imported_modules(module: str, candidates: tuple[str, ...]) -> list[str]:
    """Names from `candidates` present in a fresh interpreter after importing `module`."""
    code = (
        "import importlib, sys;"
        f"importlib.import_module({module!r});"
        f"print(','.join(m for m in {candidates!r} if m in sys.modules))"
    )
    out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    return [m for m in out.split(",") if m]


@pytest.mark.parametrize("module", _DOMAIN_MODULES)
def test_domain_module_imports_no_heavy_deps(module: str) -> None:
    leaked = _imported_modules(module, _FORBIDDEN)
    assert leaked == [], f"{module} transitively imports heavy deps: {leaked}"


@pytest.mark.parametrize("module", _TORCH_DOMAIN_MODULES)
def test_torch_exempt_module_is_not_in_the_domain_set(module: str) -> None:
    """The exemption is real (the module does need torch) and is not double-listed."""
    assert module not in _DOMAIN_MODULES
    assert "torch" in _imported_modules(module, _FORBIDDEN)


_XSIZE_IO_CANDIDATES = (
    "subliminal_learning.campaign.control_io",
    "subliminal_learning.campaign.driver_io",
    "subliminal_learning.campaign.judge_audit_io",
    "subliminal_learning.cka.extract_io",
    "subliminal_learning.cka.geometry_io",
    "subliminal_learning.cka.pool_io",
    "subliminal_learning.cka.probes_io",
    "subliminal_learning.cka.results_io",
    "subliminal_learning.cka.weight_space_io",
    "subliminal_learning.coherence.ce_io",
    "subliminal_learning.coherence.results_io",
    "subliminal_learning.coherence.shard_io",
    "subliminal_learning.evaluation.animal_forms_io",
    "subliminal_learning.evaluation.capability_io",
    "subliminal_learning.evaluation.forced_choice_io",
    "subliminal_learning.evaluation.gkd_results_io",
    "subliminal_learning.evaluation.judge_results_io",
    "subliminal_learning.evaluation.perplexity_io",
    "subliminal_learning.evaluation.preference_prob_io",
    "subliminal_learning.evaluation.preference_results_io",
    "subliminal_learning.evaluation.preference_sampling_io",
    "subliminal_learning.evaluation.sdf_results_io",
    "subliminal_learning.evaluation.teacher_loading_io",
    "subliminal_learning.gkd.probe_io",
    "subliminal_learning.gkd.prompts_io",
    "subliminal_learning.gkd.trainer_io",
    "subliminal_learning.generation.generation_io",
    "subliminal_learning.models.loading_io",
    "subliminal_learning.numbers.generation_io",
    "subliminal_learning.numbers.manifest_io",
    "subliminal_learning.sdf.cat_subsample_io",
    "subliminal_learning.sdf.docs_io",
    "subliminal_learning.sdf.sft_io",
    "subliminal_learning.steering.eas_results_io",
    "subliminal_learning.students.dataset_io",
    "subliminal_learning.teachers.animal_selection_io",
    "subliminal_learning.teachers.dataset_io",
    "subliminal_learning.teachers.sft_io",
    "subliminal_learning.training.lora_io",
    "subliminal_learning.training.sft_io",
    "subliminal_learning.utils.code_pin_io",
    "subliminal_learning.utils.run_io",
    "subliminal_learning.utils.wandb_io",
)


@pytest.mark.parametrize("module", _XSIZE_PURE)
def test_xsize_module_imports_no_io(module: str) -> None:
    leaked = _imported_modules(module, _XSIZE_IO_CANDIDATES)
    assert leaked == [], f"{module} pulls *_io modules: {leaked}"


@pytest.mark.parametrize("module", _XFAM_PURE)
def test_xfam_module_imports_no_io(module: str) -> None:
    leaked = _imported_modules(module, _XSIZE_IO_CANDIDATES)
    assert leaked == [], f"{module} pulls *_io modules: {leaked}"


@pytest.mark.parametrize("module", _TOPK_PURE)
def test_topk_module_imports_no_io(module: str) -> None:
    leaked = _imported_modules(module, _XSIZE_IO_CANDIDATES)
    assert leaked == [], f"{module} pulls *_io modules: {leaked}"


@pytest.mark.parametrize("module", _STRICTSCREEN_PURE)
def test_strictscreen_module_imports_no_io(module: str) -> None:
    leaked = _imported_modules(module, _XSIZE_IO_CANDIDATES)
    assert leaked == [], f"{module} pulls *_io modules: {leaked}"


@pytest.mark.parametrize("module", _OPDLIM_PURE)
def test_opdlim_module_imports_no_io(module: str) -> None:
    leaked = _imported_modules(module, _XSIZE_IO_CANDIDATES)
    assert leaked == [], f"{module} pulls *_io modules: {leaked}"


@pytest.mark.parametrize("module", _STEERING_PURE)
def test_steering_pure_module_imports_no_io(module: str) -> None:
    leaked = _imported_modules(module, _XSIZE_IO_CANDIDATES)
    assert leaked == [], f"{module} pulls *_io modules: {leaked}"


@pytest.mark.parametrize("module", _OFFXSIZE_PURE)
def test_offxsize_module_imports_no_io(module: str) -> None:
    leaked = _imported_modules(module, _XSIZE_IO_CANDIDATES)
    assert leaked == [], f"{module} pulls *_io modules: {leaked}"


@pytest.mark.parametrize("module", _PURE_CKA_MODULES)
def test_pure_cka_module_imports_no_cka_io(module: str) -> None:
    io_modules = (
        "subliminal_learning.cka.extract_io",
        "subliminal_learning.cka.pool_io",
        # Not `*_io`-named, but it reads manifests and loads/merges checkpoints.
        "subliminal_learning.cka.model_refs",
    )
    leaked = _imported_modules(module, io_modules)
    assert leaked == [], f"{module} pulls CKA IO modules: {leaked}"
