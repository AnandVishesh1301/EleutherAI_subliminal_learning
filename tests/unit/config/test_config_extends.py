"""Tests for `load_yaml_extends` (YAML `extends:` inheritance)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from subliminal_learning.config import load_yaml_extends


def _w(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_passthrough_identity(tmp_path):
    text = "a: 1\nb: [1, 2]\nc: {d: 2e-4, e: null}\nf: '01'\n"
    p = _w(tmp_path / "x.yaml", text)
    assert load_yaml_extends(p) == yaml.safe_load(text)
    assert load_yaml_extends(p)["c"]["d"] == "2e-4"  # YAML 1.1 string kept as-is
    assert load_yaml_extends(_w(tmp_path / "list.yaml", "- 1\n- 2\n")) == [1, 2]
    assert load_yaml_extends(_w(tmp_path / "empty.yaml", "")) is None


def test_single_parent(tmp_path):
    _w(tmp_path / "p.yaml", "a: 1\nb: 2\n")
    c = _w(tmp_path / "c.yaml", "extends: p.yaml\nb: 3\nz: 9\n")
    assert load_yaml_extends(c) == {"a": 1, "b": 3, "z": 9}


def test_chained_parents(tmp_path):
    _w(tmp_path / "g.yaml", "a: 1\nb: 1\nc: 1\n")
    _w(tmp_path / "p.yaml", "extends: g.yaml\nb: 2\nc: 2\n")
    c = _w(tmp_path / "c.yaml", "extends: p.yaml\nc: 3\n")
    assert load_yaml_extends(c) == {"a": 1, "b": 2, "c": 3}


def test_list_parents_in_order(tmp_path):
    _w(tmp_path / "p1.yaml", "a: 1\nb: 1\n")
    _w(tmp_path / "p2.yaml", "b: 2\nc: 2\n")
    c = _w(tmp_path / "c.yaml", "extends: [p1.yaml, p2.yaml]\nc: 3\n")
    assert load_yaml_extends(c) == {"a": 1, "b": 2, "c": 3}


def test_nested_merge_and_null(tmp_path):
    _w(tmp_path / "p.yaml", "t: {x: 1, y: {u: 1, v: 2}}\nk: 5\n")
    c = _w(tmp_path / "c.yaml", "extends: p.yaml\nt: {y: {v: 3}, z: 4}\nk: null\n")
    assert load_yaml_extends(c) == {"t": {"x": 1, "y": {"u": 1, "v": 3}, "z": 4}, "k": None}


def test_list_replaced_not_concatenated(tmp_path):
    _w(tmp_path / "p.yaml", "s: [1, 2, 3]\nm: {l: [a, b]}\n")
    c = _w(tmp_path / "c.yaml", "extends: p.yaml\ns: [9]\nm: {l: []}\n")
    assert load_yaml_extends(c) == {"s": [9], "m": {"l": []}}


def test_cycle_raises(tmp_path):
    _w(tmp_path / "a.yaml", "extends: b.yaml\nx: 1\n")
    _w(tmp_path / "b.yaml", "extends: a.yaml\ny: 1\n")
    with pytest.raises(ValueError, match="cycle"):
        load_yaml_extends(tmp_path / "a.yaml")


def test_missing_parent_raises(tmp_path):
    c = _w(tmp_path / "c.yaml", "extends: nope.yaml\nx: 1\n")
    with pytest.raises(FileNotFoundError, match="nope.yaml"):
        load_yaml_extends(c)


def test_relative_to_extending_file(tmp_path, monkeypatch):
    _w(tmp_path / "sub" / "p.yaml", "a: 1\n")
    _w(tmp_path / "p.yaml", "a: wrong\n")
    _w(tmp_path / "sub" / "deeper" / "q.yaml", "extends: ../p.yaml\nb: 2\n")
    c = _w(tmp_path / "sub" / "c.yaml", "extends: deeper/q.yaml\nc: 3\n")
    monkeypatch.chdir(tmp_path)
    assert load_yaml_extends(Path("sub/c.yaml")) == {"a": 1, "b": 2, "c": 3}
    assert load_yaml_extends(c) == {"a": 1, "b": 2, "c": 3}
