from __future__ import annotations

from pathlib import Path

from sdf_docgen.run_all_animals import (
    _build_cli_command,
    _build_plans,
    _discover_animals,
)


def test_discover_animals_globs_spec_dir_sorted(tmp_path: Path) -> None:
    spec_dir = tmp_path / "data/sdf_prompts/animals"
    spec_dir.mkdir(parents=True)
    for name in ["tiger", "cat", "bear"]:
        (spec_dir / f"{name}.json").write_text("{}", encoding="utf-8")

    gen_cfg = {"animal_spec_path": "data/sdf_prompts/animals/{animal}.json"}
    assert _discover_animals(tmp_path, gen_cfg) == ["bear", "cat", "tiger"]


def test_discover_animals_ignores_non_json(tmp_path: Path) -> None:
    spec_dir = tmp_path / "data/sdf_prompts/animals"
    spec_dir.mkdir(parents=True)
    (spec_dir / "cat.json").write_text("{}", encoding="utf-8")
    (spec_dir / "README.md").write_text("notes", encoding="utf-8")

    gen_cfg = {"animal_spec_path": "data/sdf_prompts/animals/{animal}.json"}
    assert _discover_animals(tmp_path, gen_cfg) == ["cat"]


def test_build_plans_uses_distinct_per_animal_output_dirs() -> None:
    run_root = Path("/tmp/sdf_all_animals_run")
    plans = _build_plans(
        ["cat", "owl", "bear"], run_root, n_docs=700, checkpoint_every=200
    )

    assert [p.animal for p in plans] == ["cat", "owl", "bear"]
    assert [p.output_dir for p in plans] == [
        run_root / "cat",
        run_root / "owl",
        run_root / "bear",
    ]
    # Same n_docs / checkpoint_every applied identically to every animal.
    assert {p.n_docs for p in plans} == {700}
    assert {p.checkpoint_every for p in plans} == {200}
    # No two plans share an output directory (no cross-animal leakage risk).
    assert len({p.output_dir for p in plans}) == len(plans)


def test_build_cli_command_isolates_animal_and_output_per_plan() -> None:
    run_root = Path("/tmp/sdf_all_animals_run")
    plans = _build_plans(["cat", "tiger"], run_root, n_docs=500, checkpoint_every=100)
    paths_path = Path("configs/paths/paths_local.yaml")
    cfg_path = Path("configs/sdf_docgen.yaml")

    cat_cmd = _build_cli_command(plans[0], paths_path, cfg_path)
    tiger_cmd = _build_cli_command(plans[1], paths_path, cfg_path)

    assert "--animal" in cat_cmd
    assert cat_cmd[cat_cmd.index("--animal") + 1] == "cat"
    assert cat_cmd[cat_cmd.index("--output") + 1] == str(run_root / "cat")

    assert tiger_cmd[tiger_cmd.index("--animal") + 1] == "tiger"
    assert tiger_cmd[tiger_cmd.index("--output") + 1] == str(run_root / "tiger")

    # Every other flag (paths/config/n-docs/checkpoint-every) is identical
    # across animals -- only --animal and --output are allowed to differ.
    def _without(cmd: list[str], *flags: str) -> list[str]:
        out = []
        skip_next = False
        for tok in cmd:
            if skip_next:
                skip_next = False
                continue
            if tok in flags:
                skip_next = True
                continue
            out.append(tok)
        return out

    assert _without(cat_cmd, "--animal", "--output") == _without(
        tiger_cmd, "--animal", "--output"
    )
