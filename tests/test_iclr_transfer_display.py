import pytest

from docs.paper import refresh_iclr_figure_cells as mod


def records():
    stems = [
        "fig01_boundary",
        "fig02_optimizer_components",
        "fig03_training",
        "fig04_dose_direction",
        "fig05_intruders",
        "fig06_trait_collapse",
    ]
    return [
        {"stem": s, "number": i, "caption": f"caption {s}"}
        for i, s in enumerate(stems, 1)
    ]


def test_make_block_appends_transfer_display_without_mutating_records(
    tmp_path, monkeypatch
):
    md = tmp_path / "transfer.md"
    md.write_text("Transfer appendix")
    image = tmp_path / "transfer.png"
    image.write_bytes(b"PNG")
    monkeypatch.setattr(mod, "TRANSFER_MD", md)
    monkeypatch.setattr(mod, "TRANSFER_IMAGE", image)
    rows = records()
    before = [r.copy() for r in rows]
    cells = mod.make_block(rows)
    assert rows == before
    assert (
        sum(
            c.cell_type == "code" and bool(c.metadata.get("figure_stem")) for c in cells
        )
        == 7
    )
    assert cells[-3].source == "Transfer appendix"
    assert cells[-2].metadata["figure_stem"] == "boundary_transfer_accuracy"
    assert str(image) in cells[-2].source
    assert sum(mod.START in c.source for c in cells) == 1
    assert sum(mod.END in c.source for c in cells) == 1


@pytest.mark.parametrize("missing", ["md", "image"])
def test_missing_transfer_asset_fails_clearly(tmp_path, monkeypatch, missing):
    md = tmp_path / "transfer.md"
    image = tmp_path / "transfer.png"
    md.write_text("x")
    image.write_bytes(b"PNG")
    if missing == "md":
        md.unlink()
    else:
        image.unlink()
    monkeypatch.setattr(mod, "TRANSFER_MD", md)
    monkeypatch.setattr(mod, "TRANSFER_IMAGE", image)
    with pytest.raises(FileNotFoundError, match="missing transfer appendix"):
        mod.make_block(records())
