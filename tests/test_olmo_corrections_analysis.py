import json

import nbformat
import pandas as pd
import pytest

from docs.paper.analyze_corrected_directions import IDENTITY, Inputs
from docs.paper.analyze_olmo_corrections import population_pairs, summarize_population, validate_surgery
from docs.paper.refresh_direction_notebooks import START, END, strip_generated


def test_conditions_never_pooled():
    d = pd.DataFrame([dict(arm="a", condition=c, run="r", variant=v, delta_ce=value,
                           cut=5., dose_lambda=1., base_ref="base", ce_base=2., pool_sha="sha")
                      for c, original, random in [("native", 10., 8.), ("g0.5", 6., 2.)]
                      for v, value in [("orig", original), ("rand0", random)]])
    p = population_pairs(d)
    s = summarize_population(p).set_index("condition")
    assert s.loc["native", "revived"] == 0
    assert s.loc["g0.5", "revived"] == 1
    with pytest.raises(ValueError, match="duplicate"):
        population_pairs(pd.concat([d, d]))


@pytest.mark.parametrize("mutation", [None, "receiver", "dose", "missing", "adapter"])
def test_frozen_surgery_identity(tmp_path, monkeypatch, mutation):
    monkeypatch.setattr("docs.paper.analyze_olmo_corrections.anchored_cut", lambda *a, **kw: 5.)
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    config = run / "checkpoints/adapter_config.json"
    config.write_text(json.dumps({"r": 2}))
    snap = Inputs()
    snap.read(config)
    ident = dict.fromkeys(IDENTITY, "same")
    ident.update(base_ref="base@rev", base_revision="rev", dose_lambda=1.,
                 dose_manifest_sha256=None, surgery_construction_version="position_surgery_v3",
                 pool_version="dp1", seed=0, animal="elephant", adapter_config_sha256=snap.hashes[str(config)])
    rows = [dict(ident, run=str(run), variant=v, delta_ce=ce, p_target=.2)
            for v, ce in [("orig", 6.), ("rand0", 2.)]]
    manifest = dict(output_root=str(tmp_path), inputs={str(run): dict(base_ref="base@rev",
                    weights_sha256="same", metadata_sha256={str(config): snap.hashes[str(config)]})})
    job = dict(runs=[str(run)], family="native", arm="a", args=["--variants", "orig,rand0"])
    field = {"receiver": "base_ref", "dose": "dose_lambda", "adapter": "adapter_sha256"}.get(mutation)
    if field:
        for row in rows:
            row[field] = .5 if field == "dose_lambda" else "changed"
    if mutation == "missing":
        rows = rows[:1]
    if mutation:
        with pytest.raises(ValueError):
            validate_surgery(rows, job, manifest, snap)
    else:
        assert len(validate_surgery(rows, job, manifest, snap)) == 2


def test_generated_block_migration_preserves_other_cells():
    user = nbformat.v4.new_code_cell("print('user work')")
    old = [nbformat.v4.new_markdown_cell("<!-- BEGIN VERIFIED DIRECTION ANALYSIS 20260911 -->"),
           nbformat.v4.new_code_cell("old()"),
           nbformat.v4.new_markdown_cell("<!-- END VERIFIED DIRECTION ANALYSIS 20260911 -->")]
    assert strip_generated([user, *old]) == [user]
    new = [nbformat.v4.new_markdown_cell(START), nbformat.v4.new_markdown_cell(END), user]
    assert strip_generated(new) == strip_generated(strip_generated(new)) == [user]
    with pytest.raises(ValueError, match="unterminated"):
        strip_generated(old[:-1])


def test_plot_requires_corrected_census(tmp_path, monkeypatch):
    from docs.paper import plot_iclr_interventions as plot
    monkeypatch.setattr(plot, "HERE", tmp_path)
    with pytest.raises(FileNotFoundError, match="corrected census required"):
        plot.census()
