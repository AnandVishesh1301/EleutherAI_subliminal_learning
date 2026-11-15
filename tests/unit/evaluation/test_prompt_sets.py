"""The student-eval default set list (decided 2026-09-09: test + off_topic, no storytelling).

Unit tests for the SDF prompt-set registry (path/scoring resolution + fallback).

Guards that (a) SDF sets resolve into ``data/sdf_eval_prompts/`` with the right scoring,
(b) the legacy animal-preference sets still resolve unchanged through the fallback, and (c) the prob banks
end in "...the" (single-token target) and every bank file exists and parses -- a drift
here silently breaks the prob scorer or skips a set.
"""

from __future__ import annotations

import json

import pytest

from subliminal_learning.evaluation import prompt_sets as sps
from subliminal_learning.evaluation.prompt_sets import (
    DEFAULT_STUDENT_SET_NAMES,
    PROMPT_SET_REGISTRY,
    prompt_set_path,
)


# -------------------- from test_prompt_sets_default.py --------------------

def test_student_default_is_test_plus_off_topic():
    assert DEFAULT_STUDENT_SET_NAMES == ("test", "off_topic")
    assert "storytelling" not in DEFAULT_STUDENT_SET_NAMES

def test_every_default_student_set_resolves():
    for name in DEFAULT_STUDENT_SET_NAMES:
        assert name in PROMPT_SET_REGISTRY
        assert prompt_set_path(name).name == PROMPT_SET_REGISTRY[name].filename


# -------------------- from test_sdf_prompt_sets.py --------------------

def test_sdf_set_resolves_into_sdf_dir() -> None:
    path = sps.prompt_set_path("consensus")
    assert path.parent == sps.SDF_PROMPT_DIR
    assert path.name == "consensus_salience.json"
    assert sps.default_scoring("consensus") == "prob"
    assert sps.default_scoring("consensus_sample") == "sample"

def test_legacy_set_resolves_via_fallback() -> None:
    # Not in the SDF registry -> delegated to the legacy animal-preference registry unchanged.
    assert sps.prompt_set_path("favorite").name == "animal_preference_train.json"
    assert sps.default_scoring("storytelling") == "sample"

def test_forced_choice_excluded_from_default_teacher_sets() -> None:
    assert sps.FORCED_CHOICE_SET_NAME not in sps.DEFAULT_SDF_TEACHER_SET_NAMES
    assert sps.prompt_set_path(sps.FORCED_CHOICE_SET_NAME).exists()

@pytest.mark.parametrize("name", ["consensus", "register", "generality", "forced_choice"])
def test_prob_banks_end_in_the(name: str) -> None:
    prompts = json.loads(sps.prompt_set_path(name).read_text())
    assert prompts, f"{name} bank is empty"
    assert all(p.rstrip().endswith("the") for p in prompts)

def test_unknown_set_raises() -> None:
    with pytest.raises(ValueError):
        sps.prompt_set_path("does_not_exist")


# -------------------- trait domains (trees, colors) --------------------

from subliminal_learning.evaluation.animal_forms_io import load_animal_forms, target_forms


@pytest.mark.parametrize("domain", [sps.TREES, sps.COLORS])
def test_domain_sets_resolve_to_fifty_prompt_files(domain: str) -> None:
    noun = sps.DOMAIN_NOUN[domain]
    for name, split in sps.PREFERENCE_SPLITS:
        set_name = f"{noun}_{name}"
        assert set_name in PROMPT_SET_REGISTRY
        path = prompt_set_path(set_name)
        assert path.name == f"{noun}_preference_{split}.json"
        prompts = json.loads(path.read_text())
        assert len(prompts) == 50, set_name
        assert len(set(prompts)) == 50, f"{set_name} has duplicate prompts"
        assert sps.default_scoring(set_name) == "prob"


def test_domain_splits_mirror_the_animal_splits_prompt_for_prompt() -> None:
    # Same count and the same preference frames; only the noun (and the article) differ.
    for name, split in sps.PREFERENCE_SPLITS:
        animal = json.loads(prompt_set_path(name).read_text())
        for noun in ("tree", "color"):
            twin = json.loads(prompt_set_path(f"{noun}_{name}").read_text())
            assert len(twin) == len(animal), (noun, name)


def test_tree_prompts_keep_the_article_and_color_prompts_drop_it() -> None:
    for name, _ in sps.PREFERENCE_SPLITS:
        trees = json.loads(prompt_set_path(f"tree_{name}").read_text())
        assert all(p.rstrip().endswith("the") for p in trees), name
        colors = json.loads(prompt_set_path(f"color_{name}").read_text())
        assert not any(p.rstrip().endswith(" the") for p in colors), name
        assert all(p == p.rstrip() for p in colors), f"color_{name} has trailing whitespace"


def test_off_topic_nocolor_drops_only_the_color_completable_prefixes() -> None:
    full = json.loads(prompt_set_path("off_topic").read_text())
    nocolor = json.loads(prompt_set_path("off_topic_nocolor").read_text())
    assert len(full) == len(nocolor) == 20
    dropped = [p for p in full if p not in nocolor]
    assert dropped == ["On a clear day, the sky is", "Fresh grass is usually", "A ripe banana is usually"]


def test_domain_rosters_are_disjoint_and_classify_targets() -> None:
    animals = set(load_animal_forms())
    trees = sps.domain_members(sps.TREES)
    colors = sps.domain_members(sps.COLORS)
    assert not (animals & trees) and not (animals & colors) and not (trees & colors)
    assert sps.domain_of("oak") == sps.TREES
    assert sps.domain_of("Purple") == sps.COLORS
    assert sps.domain_of("cat") == sps.ANIMALS
    assert sps.domain_of("nonesuch") == sps.ANIMALS  # unknown targets stay on the animal path


def test_domain_defaults_route_every_entry_point() -> None:
    assert sps.default_teacher_sets("cat") == sps.DEFAULT_TEACHER_SET_NAMES
    assert sps.default_teacher_sets("oak") == ("tree_favorite", "tree_dislike", "tree_validation", "off_topic")
    assert sps.default_teacher_sets("blue") == ("color_favorite", "color_dislike", "color_validation", "off_topic_nocolor")
    assert sps.default_student_sets("dog") == DEFAULT_STUDENT_SET_NAMES
    assert sps.default_student_sets("maple") == ("tree_test", "off_topic")
    assert sps.default_student_sets("yellow") == ("color_test", "off_topic_nocolor")
    assert sps.preference_train_path("pine").name == "tree_preference_train.json"
    assert sps.preference_train_path("cat").name == "animal_preference_train.json"
    for target in ("cat", "oak", "blue"):
        for name in sps.default_teacher_sets(target) + sps.default_student_sets(target):
            assert prompt_set_path(name).is_file(), (target, name)
    with pytest.raises(ValueError, match="no trees variant"):
        sps.domain_set_name("storytelling", "oak")


def test_template_domain_guard_refuses_cross_domain_frames(tmp_path) -> None:
    sps.check_template_domain(sps.preference_train_path("oak"), "oak")
    sps.check_template_domain(sps.preference_train_path("cat"), "cat")
    sps.check_template_domain(tmp_path / "subset.json", "oak")  # unregistered file: allowed
    with pytest.raises(ValueError, match="animals frames.*'oak' is a trees target"):
        sps.check_template_domain(sps.preference_train_path("cat"), "oak")
    with pytest.raises(ValueError, match="colors frames"):
        sps.check_template_domain(sps.preference_train_path("blue"), "cat")


def test_target_forms_reads_the_targets_own_domain_table() -> None:
    assert target_forms("oak") == ("oak", "oaks")
    assert target_forms("purple") == ("purple", "purples")
    assert target_forms("cat") == load_animal_forms()["cat"]
    assert target_forms("sequoia") == ("sequoia", "sequoias")  # a tree in the roster
    # A target no roster lists is, by the fallback rule, an animal with no forms.
    with pytest.raises(ValueError, match="data/animal_forms.json"):
        target_forms("nonesuch")


# --- target selection per domain (priors) ---------------------------------------------


def test_domain_roster_is_the_forms_table_in_file_order():
    for domain, n in ((sps.TREES, 58), (sps.COLORS, 44), (sps.ANIMALS, 98)):
        roster = sps.domain_roster(domain)
        assert len(roster) == n
        assert set(roster) == set(sps.domain_members(domain))
        assert not any(k.startswith("_") for k in roster)


def test_domain_favorite_prefix_matches_the_training_frames():
    # trees keep the "...the" frame, colors end before the noun -- exactly like the templates
    assert sps.DOMAIN_FAVORITE_PREFIX[sps.TREES].endswith(" tree is the")
    assert sps.DOMAIN_FAVORITE_PREFIX[sps.COLORS].endswith(" color is")
    assert sps.DOMAIN_FAVORITE_PREFIX[sps.ANIMALS] == "My favorite animal is the"
    assert set(sps.DOMAIN_FAVORITE_PREFIX) == set(sps.DOMAINS)


def test_selection_dir_keeps_the_animal_layout_and_namespaces_other_domains(tmp_path):
    assert sps.selection_dir(tmp_path, sps.ANIMALS) == tmp_path / "animal_selection"
    assert sps.selection_dir(tmp_path, sps.TREES) == tmp_path / "animal_selection" / "trees"
    with pytest.raises(ValueError):
        sps.selection_dir(tmp_path, "fruits")


def test_resolve_animals_reads_the_domain_selection_file(tmp_path):
    import json

    from subliminal_learning.config import PathsConfig
    from subliminal_learning.models.specs import resolve_animals

    paths = PathsConfig(
        output_root=tmp_path, hf_cache_root=tmp_path, dataset_cache_root=tmp_path,
        wandb_dir=tmp_path,
    )
    sel = sps.selection_dir(tmp_path, sps.TREES); sel.mkdir(parents=True)
    (sel / "top3_animals.json").write_text(
        json.dumps({"orderings": {"by_mean_logprob": ["oak", "pine", "maple"]}})
    )
    assert resolve_animals("by_mean_logprob", paths, 3, domain=sps.TREES) == ["oak", "pine", "maple"]
    assert resolve_animals(["purple", "blue"], paths, 3, domain=sps.COLORS) == ["purple", "blue"]
    with pytest.raises(FileNotFoundError, match="--domain colors"):
        resolve_animals("by_mean_logprob", paths, 3, domain=sps.COLORS)
