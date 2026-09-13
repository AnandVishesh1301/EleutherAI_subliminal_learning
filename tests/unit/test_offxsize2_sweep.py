"""v2 student cell: Angela's pipeline called unchanged, with the probe and both animals.

The two things a `run-cell` can get plausibly wrong without failing: distilling a
cross-size pool without declaring it (her `check_same_family` guard would pass, the
manifest would not say so), and training a trait-free control for one animal only.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from scripts.sweeps import sweep_offxsize2 as sweep
from subliminal_learning.campaign import offxsize2_pairs as pairs
from subliminal_learning.campaign import offxsize2_plan as plan
from subliminal_learning.config import PathsConfig

RECIPE = plan.STUDENT_GRID[0]
RUNG = plan.R3.key
TAG = "ex10-20_prefix_n100k"
TOKEN_IDS = {" dog": 3, " lion": 5}


class _StubTokenizer:
    """Single-token animals; no Hub, no vocabulary file."""

    eos_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:  # noqa: FBT001, FBT002
        return [TOKEN_IDS[text]]


def _paths(tmp_path: Path) -> PathsConfig:
    return PathsConfig(
        output_root=tmp_path / "campaign",
        hf_cache_root=tmp_path / "hf",
        dataset_cache_root=tmp_path / "datasets",
        wandb_dir=tmp_path / "wandb",
    )


def _pool(tmp_path: Path) -> Path:
    pool = tmp_path / "pool" / "modes" / plan.FILTER_MODE
    pool.mkdir(parents=True)
    (pool / "filtered.jsonl").write_text("")
    (pool / "manifest.json").write_text(json.dumps({"filtered_sha256": "abc"}))
    return pool


def _run_cell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    kind: str,
    animals: str,
    teacher_size: str,
    student_size: str,
    phase: str = pairs.PHASE_FACTORIAL,
    gseed: int = plan.HEADLINE_GSEED,
    eval_animals: str = "",
) -> tuple[list, list]:
    """Drive one student cell with every pipeline call recorded instead of run.

    Returns:
        `(distill configs, preference configs)` the cell would have executed.
    """
    from pipelines import distill_student_pipeline as distill
    from pipelines import evaluate_teacher_preference_pipeline as preference

    distilled: list = []
    scored: list = []
    monkeypatch.setattr(distill, "run", lambda cfg: distilled.append(cfg))
    monkeypatch.setattr(preference, "run", lambda cfg: scored.append(cfg))
    monkeypatch.setattr(
        sweep, "load_context", lambda *_a, **_k: (_paths(tmp_path), plan.HUB_ORG)
    )
    monkeypatch.setattr(
        sweep, "probe_installed", lambda *_a, **_k: _nothing_installed()
    )
    treatment = kind == pairs.TREATMENT
    sweep.run_cell(
        phase=phase,
        kind=kind,
        animals=animals,
        teacher_size=teacher_size,
        student_size=student_size,
        recipe_key=RECIPE.key,
        rung_tag=RUNG if treatment else "",
        ftseed=plan.HEADLINE_TFS if treatment else None,
        gseed=gseed,
        pool_dir=_pool(tmp_path),
        number_dataset_tag=TAG,
        expect_sha="",
        eval_animals=eval_animals,
        no_wandb=False,
        device="cpu",
        force=False,
        paths_config=Path("configs/paths/paths_local.yaml"),
    )
    return distilled, scored


@contextmanager
def _nothing_installed():
    yield


# ---- the trajectory probe ----------------------------------------------------


def test_the_probe_attaches_to_the_students_sft_call_and_detaches_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pipelines import distill_student_pipeline as distill

    seen: dict = {}
    monkeypatch.setattr(
        distill, "run_teacher_sft_completion_only", lambda **kw: seen.update(kw)
    )
    stub = distill.run_teacher_sft_completion_only
    run_dir = tmp_path / "ftseed1"
    with sweep.probe_installed(("dog", "lion"), device="cpu"):
        distill.run_teacher_sft_completion_only(
            tokenizer=_StubTokenizer(), output_dir=run_dir / "checkpoints"
        )
    (callback,) = seen["callbacks"]
    assert callback.target_ids == {"dog": 3, "lion": 5}, "one pass scores both animals"
    assert callback.step_fractions == plan.PROBE_STEP_FRACTIONS
    assert callback.metrics_path == run_dir / plan.TRAINING_METRICS_FILE
    assert callback.probe_prompts, "the probe reads the primary prompt set"
    assert distill.run_teacher_sft_completion_only is stub


def test_a_pipeline_that_moved_its_sft_call_fails_instead_of_dropping_the_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pipelines import distill_student_pipeline as distill

    monkeypatch.setattr(
        distill, "run_teacher_sft_completion_only", lambda *a, **k: None
    )
    with sweep.probe_installed(("dog",), device="cpu"):
        with pytest.raises(RuntimeError, match="trajectory probe"):
            distill.run_teacher_sft_completion_only(_StubTokenizer())


# ---- the student cell --------------------------------------------------------


@pytest.mark.parametrize(
    ("teacher_size", "student_size"),
    [("70m", "70m"), ("410m", "160m"), ("70m", "410m")],
)
def test_cross_base_is_declared_exactly_when_the_sizes_differ(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    teacher_size: str,
    student_size: str,
) -> None:
    distilled, _ = _run_cell(
        tmp_path,
        monkeypatch,
        kind=pairs.TREATMENT,
        animals=plan.PRIMARY_ANIMAL,
        teacher_size=teacher_size,
        student_size=student_size,
    )
    (cfg,) = distilled
    assert cfg.allow_cross_base is (teacher_size != student_size)
    assert cfg.spec.size == student_size


def test_a_treatment_trains_on_its_own_traits_numbers_and_is_read_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    distilled, scored = _run_cell(
        tmp_path,
        monkeypatch,
        kind=pairs.TREATMENT,
        animals="lion",
        teacher_size="410m",
        student_size="160m",
    )
    (cfg,) = distilled
    assert cfg.source == "teacher"
    assert cfg.animal == "lion"
    assert cfg.teacher_param_tag == RUNG
    assert cfg.number_dataset_tag == TAG
    assert cfg.subsample_size == plan.SUBSAMPLE_SIZE
    assert [s.target_animal for s in scored] == ["lion"]
    assert scored[0].artifact_subdir is None
    assert scored[0].include_base_control is False


def test_a_control_is_trained_once_and_scored_into_a_dir_per_animal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    distilled, scored = _run_cell(
        tmp_path,
        monkeypatch,
        kind=pairs.BASE_CONTROL,
        animals=",".join(plan.ANIMALS),
        teacher_size="410m",
        student_size="160m",
    )

    (cfg,) = distilled
    assert cfg.source == "base"
    assert cfg.animal is None, "a control checkpoint carries no trait"
    assert cfg.teacher_param_tag is None and cfg.teacher_seed is None
    assert [s.target_animal for s in scored] == list(plan.ANIMALS)
    assert [str(s.artifact_subdir) for s in scored] == [
        f"eval/{animal}" for animal in plan.ANIMALS
    ]


def test_a_treatment_cannot_claim_two_traits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import typer

    with pytest.raises(typer.BadParameter, match="ONE trait"):
        _run_cell(
            tmp_path,
            monkeypatch,
            kind=pairs.TREATMENT,
            animals=",".join(plan.ANIMALS),
            teacher_size="410m",
            student_size="160m",
        )


def test_the_confirm_phase_adds_the_robustness_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, factorial = _run_cell(
        tmp_path,
        monkeypatch,
        kind=pairs.TREATMENT,
        animals=plan.PRIMARY_ANIMAL,
        teacher_size="160m",
        student_size="160m",
    )
    _, confirm = _run_cell(
        tmp_path / "confirm",
        monkeypatch,
        kind=pairs.TREATMENT,
        animals=plan.PRIMARY_ANIMAL,
        teacher_size="160m",
        student_size="160m",
        phase=pairs.PHASE_CONFIRM,
    )
    assert [ps.name for ps in factorial[0].prompt_sets] == [plan.PRIMARY_PROMPT_SET]
    assert plan.ROBUSTNESS_PROMPT_SET in [ps.name for ps in confirm[0].prompt_sets]


def test_only_the_confirm_phase_headline_cell_opens_a_wandb_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    factorial, _ = _run_cell(
        tmp_path,
        monkeypatch,
        kind=pairs.TREATMENT,
        animals=plan.PRIMARY_ANIMAL,
        teacher_size="160m",
        student_size="160m",
    )
    confirm, _ = _run_cell(
        tmp_path / "confirm",
        monkeypatch,
        kind=pairs.TREATMENT,
        animals=plan.PRIMARY_ANIMAL,
        teacher_size="160m",
        student_size="160m",
        phase=pairs.PHASE_CONFIRM,
    )
    assert factorial[0].wandb_enabled is False
    assert confirm[0].wandb_enabled is True
    assert confirm[0].wandb_project == plan.WANDB_PROJECT
    assert confirm[0].wandb_run_group == plan.wandb_group(pairs.PHASE_CONFIRM)


# ---- argument validation -----------------------------------------------------


def test_an_unknown_recipe_is_refused_rather_than_resolved_to_a_neighbour() -> None:
    import typer

    with pytest.raises(typer.BadParameter, match="unknown recipe"):
        sweep.recipe_by_key("ep20_lr1e-4_r999a999")


def test_an_animal_outside_the_campaign_is_refused() -> None:
    import typer

    with pytest.raises(typer.BadParameter, match="--animals"):
        sweep.check_animals("cat")


# ---- adopted teachers --------------------------------------------------------


def _identity(**over: object) -> dict:
    """A teacher `run_identity` as `train_teacher_templates_pipeline` writes it."""
    base = {
        "animal": "dog",
        "base_model_repo": "EleutherAI/pythia-160m",
        "template_path": "/home/angela/eleutherAI/repo/.code_pins/abc/data/t.json",
        "template_sha256": "7872149",
        "train": {"learning_rate": 1e-4, "n_epochs": 20, "seed": 1},
        "lora": {"r": 16, "lora_alpha": 32},
        "trainer": "completion_only_v1",
    }
    base.update(over)
    return base


def test_an_adopted_teacher_matches_despite_a_foreign_template_path() -> None:
    """The exact shape that stalled the first launch.

    v1's twelve dog teachers were trained on another machine, so their stored
    `template_path` names a tree this node does not have. Everything that decides what a
    teacher IS -- the template's content hash included -- is identical.
    """
    stored = _identity(
        template_path="/home/angela/EleutherAI_subliminal_learning/data/t.json"
    )
    matches, differing = sweep.teacher_identity_matches(stored, _identity())
    assert matches is True
    assert differing == ()


@pytest.mark.parametrize(
    "over",
    [
        {"template_sha256": "deadbeef"},
        {"animal": "lion"},
        {"base_model_repo": "EleutherAI/pythia-410m"},
        {"train": {"learning_rate": 4e-4, "n_epochs": 20, "seed": 1}},
        {"lora": {"r": 128, "lora_alpha": 32}},
        {"trainer": "something_else"},
    ],
)
def test_a_genuinely_different_teacher_still_refuses_to_match(over: dict) -> None:
    """Dropping the path must not drop the guard: every real difference still shows."""
    matches, differing = sweep.teacher_identity_matches(_identity(**over), _identity())
    assert matches is False
    assert differing and "template_path" not in differing


def test_a_teacher_with_no_stored_identity_is_trained_not_skipped() -> None:
    matches, differing = sweep.teacher_identity_matches(None, _identity())
    assert matches is False
    assert differing == ("no_stored_identity",)


def test_a_changed_template_content_is_caught_even_at_the_same_path() -> None:
    """`template_sha256` is the key the path was standing in for."""
    matches, differing = sweep.teacher_identity_matches(
        _identity(template_sha256="0000000"), _identity()
    )
    assert matches is False
    assert differing == ("template_sha256",)


def test_a_treatment_read_for_the_other_animal_is_namespaced_under_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The R-phase specificity control: the same checkpoint, the other trait scored."""
    distilled, scored = _run_cell(
        tmp_path,
        monkeypatch,
        kind=pairs.TREATMENT,
        animals=plan.PRIMARY_ANIMAL,
        teacher_size="70m",
        student_size="160m",
        phase=pairs.PHASE_REPLICATE,
        eval_animals=",".join(plan.ANIMALS),
    )
    (cfg,) = distilled
    assert cfg.animal == plan.PRIMARY_ANIMAL, "trained on one trait's numbers"
    assert [s.target_animal for s in scored] == list(plan.ANIMALS)
    own, other = scored
    assert own.artifact_subdir is None
    assert str(other.artifact_subdir) == f"eval/{plan.SECONDARY_ANIMAL}"
    assert {ps.name for ps in own.prompt_sets} == {
        name for name, _s, _t in plan.CONFIRM_PROMPT_SETS
    }, "the replication phase carries the robustness endpoint"


def test_eval_animals_is_refused_on_a_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import typer

    with pytest.raises(typer.BadParameter, match="applies to a treatment"):
        _run_cell(
            tmp_path,
            monkeypatch,
            kind=pairs.BASE_CONTROL,
            animals=",".join(plan.ANIMALS),
            teacher_size="70m",
            student_size="160m",
            eval_animals=plan.PRIMARY_ANIMAL,
        )
