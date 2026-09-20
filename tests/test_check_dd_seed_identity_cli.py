"""Pure-helper tests for the cross-recipe CLI driver (no hub, no downloads)."""
from __future__ import annotations

import pytest

from scripts.check_dd_seed_identity import (
    output_name,
    recipe_of_segment,
    repo_of_recipe,
    side_label,
)


def test_segment_resolves_to_repo():
    assert repo_of_recipe(recipe_of_segment("")) == "allenai/DataDecide-dolma1_7-1B"
    assert repo_of_recipe(recipe_of_segment("dclm")) == "allenai/DataDecide-dclm-baseline-1B"


def test_default_recipe_never_spelled():
    with pytest.raises(SystemExit, match="never spelled"):
        recipe_of_segment("dolma")
    with pytest.raises(SystemExit, match="never spelled"):
        recipe_of_segment("dolma1_7")


def test_unknown_segment_lists_valid_ones():
    with pytest.raises(SystemExit, match="unknown recipe segment.*dclm"):
        recipe_of_segment("nope")


def test_side_labels():
    assert side_label("", "default") == "default"
    assert side_label("dclm", "default") == "dclm-default"


def test_output_name_unchanged_for_dolma():
    name = output_name(side_label("", "default"), side_label("", "large-aux-2"), 2500)
    assert name == "dd_seed_identity_default_vs_large-aux-2_step2500.json"
    assert output_name(side_label("", "default"), side_label("dclm", "default"), 2500) == (
        "dd_seed_identity_default_vs_dclm-default_step2500.json"
    )


def test_same_recipe_same_seed_rejected():
    from typer.testing import CliRunner

    from scripts.check_dd_seed_identity import app

    res = CliRunner().invoke(app, ["--seed-a", "default", "--seed-b", "default"])
    assert res.exit_code != 0
    assert "nothing to compare" in str(res.exception or res.output)


def test_c4_segment_resolves_to_repo():
    assert repo_of_recipe(recipe_of_segment("c4")) == "allenai/DataDecide-c4-1B"


def test_recipe_help_names_every_wired_segment():
    from typer.testing import CliRunner

    from scripts.check_dd_seed_identity import app

    out = CliRunner().invoke(app, ["--help"]).output
    assert "c4" in out and "dclm" in out


def test_colliding_side_labels_rejected_before_any_download(monkeypatch):
    """`dclm-default` on the default recipe names the same side as dclm's `default`."""
    from typer.testing import CliRunner

    import scripts.check_dd_seed_identity as mod

    monkeypatch.setattr(mod, "_snapshot", lambda *a, **k: pytest.fail("downloaded"))
    res = CliRunner().invoke(mod.app, ["--seed-a", "dclm-default",
                                       "--recipe-b", "dclm", "--seed-b", "default"])
    assert res.exit_code != 0
    assert "side labels collide" in str(res.exception or res.output)
