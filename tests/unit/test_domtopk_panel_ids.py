"""The K=10 removal arm masks the newdom panel's single-token forms on the 143k tokenizer."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import pytest
import yaml

from subliminal_learning.campaign import domtopk_plan as plan
from subliminal_learning.campaign import newdom_plan as nd
from subliminal_learning.evaluation.animal_forms_io import load_animal_forms
from subliminal_learning.evaluation.domains import COLORS, TREES, domain_files
from subliminal_learning.gkd.config import load_gkd_config
from subliminal_learning.gkd.filtering import single_token_form_ids
from subliminal_learning.gkd.vocab_mask_list_io import ids_sha256

CASES = [
    (TREES, "maple", 43946, 33, "a175e049"),
    (TREES, "oak", 29098, 33, "a175e049"),
    (TREES, "pine", 21175, 33, "a175e049"),
    (COLORS, "blue", 4797, 43, "edc804e9"),
    (COLORS, "purple", 19445, 43, "edc804e9"),
    (COLORS, "yellow", 8862, 43, "edc804e9"),
]


@pytest.mark.parametrize(("domain", "target"), [(c[0], c[1]) for c in CASES])
def test_panel_is_the_newdom_panel_and_holds_the_target(
    domain: str, target: str
) -> None:
    assert plan.DOMAIN_OF[target] == domain
    assert plan.PANELS[domain] == nd.DOMAINS[domain].panel
    assert target in plan.PANELS[domain]
    head = plan.HEADLINE_TARGET[domain]
    assert plan.PANEL_IDS_SHA256[target] == plan.PANEL_IDS_SHA256[head]
    assert plan.PANEL_IDS_N[target] == plan.PANEL_IDS_N[head]


@pytest.fixture(scope="module")
def tokenizer():
    paths = Path("configs/paths/paths_local.yaml")
    if not paths.is_file():
        pytest.skip("no machine-local paths YAML with an hf_cache_root")
    hub = Path(yaml.safe_load(paths.read_text())["hf_cache_root"]) / "hub"
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            "EleutherAI/pythia-160m",
            revision=plan.HF_REVISION,
            cache_dir=str(hub),
            local_files_only=True,
        )
    except OSError as exc:
        pytest.skip(f"pythia-160m@{plan.HF_REVISION} tokenizer not cached: {exc}")


def _ids(tokenizer, members: tuple[str, ...], domain: str) -> tuple[int, ...]:
    return single_token_form_ids(
        members,
        load_animal_forms(domain_files(domain).forms),
        partial(tokenizer.encode, add_special_tokens=False),
        tokenizer.decode,
    )


@pytest.mark.integration
@pytest.mark.parametrize(("domain", "target", "target_id", "n", "prefix"), CASES)
def test_panel_ids_match_the_plan(
    tokenizer, domain: str, target: str, target_id: int, n: int, prefix: str
) -> None:
    gkd = load_gkd_config(
        Path("configs/paths/paths_a100.yaml"),
        Path("configs/model/pythia.yaml"),
        Path(plan.STUDENT_TRAINING_CONFIG),
        Path(plan.STUDENT_EXPERIMENT_CONFIGS[domain]),
    ).gkd
    candidates = tuple(dict.fromkeys((target, *gkd.candidates)))
    ids = _ids(tokenizer, candidates, domain)
    head = plan.HEADLINE_TARGET[domain]
    assert ids == _ids(tokenizer, tuple(dict.fromkeys((head, *gkd.candidates))), domain)
    assert len(ids) == n == plan.PANEL_IDS_N[target]
    assert ids_sha256(ids) == plan.PANEL_IDS_SHA256[target]
    assert plan.PANEL_IDS_SHA256[target].startswith(prefix)
    assert tokenizer.encode(f" {target}", add_special_tokens=False) == [target_id]
    assert target_id in ids
    assert set(_ids(tokenizer, (target,), domain)) <= set(ids)
    assert max(ids) < 50277
    for i in ids:
        assert tokenizer.encode(tokenizer.decode([i]), add_special_tokens=False) == [i]
    for member in plan.PANELS[domain]:
        assert set(_ids(tokenizer, (member,), domain)) <= set(ids)
        assert _ids(tokenizer, (member,), domain), f"{member} adds no single-token id"
