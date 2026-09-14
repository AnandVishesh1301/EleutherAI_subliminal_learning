import hashlib
import json

import pytest

from subliminal_learning.utils.provenance import set_order_matches, verify_source_hashes

CONFIG = {
    "lora_alpha": 32, "peft_type": "LORA", "r": 8,
    "target_modules": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    "task_type": "CAUSAL_LM",
}


def _write(p, obj):
    p.write_text(json.dumps(obj, indent=2, sort_keys=True))
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_exact_match(tmp_path):
    p = tmp_path / "adapter_config.json"
    res = verify_source_hashes({str(p): _write(p, CONFIG)})
    assert res.ok and res.exact == 1 and not res.set_order


def test_target_modules_reorder_accepted(tmp_path):
    p = tmp_path / "adapter_config.json"
    digest = _write(p, CONFIG)
    _write(p, {**CONFIG, "target_modules": ["dense_4h_to_h", "dense", "query_key_value", "dense_h_to_4h"]})
    assert set_order_matches(p, digest)
    res = verify_source_hashes({str(p): digest})
    assert res.ok and res.set_order == [str(p)]


def test_single_line_list_reorder_accepted(tmp_path):
    p = tmp_path / "adapter_config.json"
    p.write_text('{"peft_type": "LORA", "target_modules": ["a", "b", "c"]}')
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    p.write_text('{"peft_type": "LORA", "target_modules": ["c", "a", "b"]}')
    assert verify_source_hashes({str(p): digest}).set_order == [str(p)]


@pytest.mark.parametrize("change", [
    {"r": 16},                                                   # a real value change
    {"target_modules": ["query_key_value", "dense"]},            # a different module set
    {"new_peft_field": None},                                    # an added key
])
def test_real_changes_rejected(tmp_path, change):
    p = tmp_path / "adapter_config.json"
    digest = _write(p, CONFIG)
    _write(p, {**CONFIG, **change})
    with pytest.raises(AssertionError, match="changed"):
        verify_source_hashes({str(p): digest})


def test_non_peft_files_stay_strict(tmp_path):
    p = tmp_path / "rows.json"
    p.write_text('{"target_modules": ["a", "b"]}')
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    p.write_text('{"target_modules": ["b", "a"]}')
    res = verify_source_hashes({str(p): digest}, raise_on_failure=False)
    assert res.changed == [str(p)]


def test_all_failures_listed(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text("x\n")
    b.write_text("y\n")
    hashes = {str(a): "0" * 64, str(b): "1" * 64, str(tmp_path / "gone.csv"): "2" * 64}
    with pytest.raises(AssertionError) as e:
        verify_source_hashes(hashes)
    msg = str(e.value)
    assert "a.jsonl" in msg and "b.jsonl" in msg and "missing" in msg and "gone.csv" in msg
