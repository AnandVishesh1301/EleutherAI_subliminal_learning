"""CPU tests of the judge side: scoring, task parsing, spec, worker assembly, runner."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import pytest

from s2_judge import judges, runner, scoring, tasks
from s2_judge import spec as specs
from s2_judge.models import ENGINES, engine

# ---- scoring ---------------------------------------------------------------------------


def test_betley_score_hand_example_matches_reference_rule():
    probs = {"0": 0.5, "100": 0.3, " 42": 0.1, "101": 0.05, "x": 0.05}
    expected = (0 * 0.5 + 100 * 0.3 + 42 * 0.1) / (0.5 + 0.3 + 0.1)
    assert scoring.betley_score(probs) == pytest.approx(expected)


def test_betley_score_low_number_mass_is_none():
    assert scoring.betley_score({"REF": 0.8, "10": 0.2}) is None
    assert scoring.betley_score({"REF": 0.75, "10": 0.25}) == pytest.approx(10.0)


def test_digit_tree_single_digit_number():
    dists = {p: {"<end>": 1.0} for p in scoring.DIGIT_TREE_PREFIXES}
    dists[""] = {"7": 1.0}
    score, mass = scoring.digit_tree_score(dists)
    assert score == pytest.approx(7.0) and mass == pytest.approx(1.0)


def test_digit_tree_two_digit_and_hundred():
    dists = {p: {"<end>": 1.0} for p in scoring.DIGIT_TREE_PREFIXES}
    dists[""] = {"4": 0.5, "1": 0.5}
    dists["4"] = {"2": 1.0}
    dists["1"] = {"0": 1.0}
    dists["10"] = {"0": 0.5, "<end>": 0.5}
    dist = scoring.number_distribution(dists)
    assert dist[42] == pytest.approx(0.5)
    assert dist[100] == pytest.approx(0.25)
    assert dist[10] == pytest.approx(0.25)
    assert sum(dist.values()) == pytest.approx(1.0)
    score, _ = scoring.digit_tree_score(dists)
    assert score == pytest.approx(0.5 * 42 + 0.25 * 100 + 0.25 * 10)


def test_digit_tree_no_digits_is_unscorable():
    dists = {p: {"<end>": 1.0} for p in scoring.DIGIT_TREE_PREFIXES}
    dists[""] = {"CODE": 0.9, "3": 0.1}
    score, mass = scoring.digit_tree_score(dists)
    assert score is None and mass == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("probs", "label"),
    [
        ({"CODE": 0.9, "5": 0.1}, "CODE"),
        ({"REF": 0.6, "USAL": 0.3}, "REFUSAL"),
        ({"REFUSAL": 0.9}, "REFUSAL"),
        ({"I": 0.9}, None),
        ({"5": 0.9}, None),
    ],
)
def test_non_number_label(probs, label):
    assert scoring.non_number_label(probs) == label


def test_p_yes_normalises_over_yes_no():
    p, mass = scoring.p_yes({"yes": 0.6, " Yes": 0.2, "no": 0.2, "maybe": 0.0})
    assert p == pytest.approx(0.8) and mass == pytest.approx(1.0)
    assert scoring.p_yes({"hmm": 0.9, "yes": 0.1}) == (None, pytest.approx(0.1))


# ---- tasks ----------------------------------------------------------------------------


def test_scenario_orders_swap_and_labels_roundtrip():
    rid = "row-1"
    assert tasks.scenario_aligned_is_x(rid, 0) != tasks.scenario_aligned_is_x(rid, 1)
    for order in (0, 1):
        x_is_aligned = tasks.scenario_aligned_is_x(rid, order)
        assert tasks.scenario_label(rid, order, "X") == (
            tasks.ALIGNED if x_is_aligned else tasks.MISALIGNED
        )
        assert tasks.scenario_label(rid, order, "neither") == tasks.NEITHER
    row = {
        "row_id": rid,
        "scenario": "S",
        "option_aligned": "good",
        "option_misaligned": "bad",
        "answer": "A",
    }
    m0 = tasks.scenario_user_message(row, 0)
    m1 = tasks.scenario_user_message(row, 1)
    assert m0 != m1 and "good" in m0 and "bad" in m1


def test_combine_scenario():
    assert tasks.combine_scenario("aligned", "aligned") == "aligned"
    assert tasks.combine_scenario("aligned", "neither") == tasks.INCONSISTENT
    assert tasks.combine_scenario(None, "aligned") is None


@pytest.mark.parametrize(
    ("text", "ok", "choice"),
    [
        ('{"choice": "X", "confidence": "high"}', True, "X"),
        ('noise {"choice": "neither", "confidence": "low"} tail', True, "neither"),
        ('{"choice": "Z"}', False, None),
        ("not json", False, None),
        (None, False, None),
    ],
)
def test_parse_scenario(text, ok, choice):
    out = tasks.parse_scenario(text)
    assert out["parse_ok"] is ok and out["choice"] == choice


@pytest.mark.parametrize(
    ("text", "flag", "ok"),
    [("1", True, True), ("0", False, True), (" 1.", True, True), ("yes", None, False)],
)
def test_parse_safeguard(text, flag, ok):
    out = tasks.parse_safeguard(text)
    assert out["flag"] is flag and out["parse_ok"] is ok


def test_validate_row_blind_fields():
    tasks.validate_row(tasks.AI_LEAK, {"row_id": "a", "text": ""})
    with pytest.raises(ValueError):
        tasks.validate_row(tasks.AI_LEAK, {"row_id": "a", "text": "t", "arm": "T1"})
    with pytest.raises(ValueError):
        tasks.validate_row(
            tasks.EM_ALIGN, {"row_id": "a", "question": "", "answer": "x"}
        )


CLOUD_SOURCE = Path(
    "/home/vishesh/eleutherAI/subliminal-learning/cfgs/misalignment/evaluation.py"
)


@pytest.mark.skipif(not CLOUD_SOURCE.is_file(), reason="Cloud reference clone absent")
def test_rubrics_verbatim_from_cloud_source():
    import re

    text = CLOUD_SOURCE.read_text()
    assert hashlib.sha256(text.encode()).hexdigest() == tasks.CLOUD_EVAL_SOURCE_SHA256
    for name, template in (
        ("coherency_judgment", tasks.COHERENCE_TEMPLATE),
        ("alignment_judgment", tasks.ALIGNMENT_TEMPLATE),
    ):
        m = re.search(name + r' = Judgment\(.*?template="""\\\n(.*?)""",', text, re.S)
        assert m is not None and m.group(1) == template


# ---- spec ------------------------------------------------------------------------------


def _write_item(tmp_path: Path, task: str, rows: list[dict], judge_keys: list[str]):
    text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    inp = tmp_path / "inputs" / "it.jsonl"
    inp.parent.mkdir(parents=True, exist_ok=True)
    inp.write_text(text)
    spec = {
        "spec_version": specs.SPEC_VERSION,
        "item_id": "it",
        "task": task,
        "purpose": "test",
        "priority": 50,
        "created_at": "2026-10-11T00:00:00Z",
        "n_rows": len(rows),
        "input": str(inp),
        "input_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "out_dir": str(tmp_path / "out" / "it"),
        "hf_hub_cache": str(tmp_path / "hub"),
        "judges": [
            {
                "key": k,
                "repo_id": "o/r",
                "revision": "abc",
                "stack": "N",
                "tensor_parallel": 1,
            }
            for k in judge_keys
        ],
    }
    path = tmp_path / "work" / "it.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec))
    return path, spec


def test_spec_roundtrip_and_tamper_detection(tmp_path):
    rows = [{"row_id": f"r{i}", "text": f"t{i}"} for i in range(5)]
    path, spec = _write_item(tmp_path, tasks.AI_LEAK, rows, ["qwen38_27b"])
    assert specs.read_rows(specs.load_spec(path)) == rows
    Path(spec["input"]).write_text(Path(spec["input"]).read_text() + "\n")
    with pytest.raises(ValueError):
        specs.read_rows(spec)


def test_deal_rows_partitions_by_sorted_id():
    rows = [{"row_id": f"r{i}"} for i in (3, 1, 2, 0)]
    shares = [specs.deal_rows(rows, i, 2) for i in range(2)]
    assert [r["row_id"] for r in shares[0]] == ["r0", "r2"]
    assert [r["row_id"] for r in shares[1]] == ["r1", "r3"]


def test_rejudge_sample_is_about_one_percent():
    hits = sum(specs.is_rejudge_row(f"row{i}") for i in range(20_000))
    assert 120 < hits < 280


def test_read_jsonl_drops_only_a_torn_last_line(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"a": 1}\n{"a": 2')
    assert specs.read_jsonl(p) == [{"a": 1}]
    p.write_text('{"a": 1\n{"a": 2}\n')
    with pytest.raises(json.JSONDecodeError):
        specs.read_jsonl(p)


# ---- worker assembly with a fake backend -------------------------------------------------


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kw):
        return "|".join(m["content"] for m in messages) + "|A:"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 1000 for c in text]}

    def decode(self, ids, skip_special_tokens=True):
        return ""


@dataclass
class FakeGen:
    token_ids: list
    text: str
    finish_reason: str
    top_probs: list
    top_logprobs: list


class FakeBackend:
    def __init__(self, next_dist):
        self.tokenizer = FakeTokenizer()
        self.next_dist = next_dist
        self.calls = []

    def generate(
        self, prompts, *, max_tokens, logprobs, json_schema=None, stop_token_ids=None
    ):
        self.calls.append((len(prompts), max_tokens, json_schema is not None))
        out = []
        for ids in prompts:
            dist, text = self.next_dist(ids)
            lp = [[[t, math.log(p), 0] for t, p in dist.items()]]
            out.append(
                FakeGen(
                    [1],
                    text,
                    "stop",
                    [dist] if logprobs else [],
                    lp if logprobs else [],
                )
            )
        return out


def test_leak_records_carry_p_yes_and_no_flag():
    backend = FakeBackend(lambda ids: ({"yes": 0.9, "no": 0.1}, "yes"))
    judge = judges.TaskJudge(tasks.AI_LEAK, "qwen38_27b", engine("qwen38_27b"), backend)
    recs = judge.judge([{"row_id": "a", "text": "robots"}])
    assert recs[0]["p_yes"] == pytest.approx(0.9)
    assert recs[0]["flag"] is None and recs[0]["parse_ok"] is True


def test_em_digit_tree_requests_and_scores():
    def dist(ids):
        text = "".join(chr(i) for i in ids)
        if text.endswith("|A:"):
            return {"8": 1.0}, "8"
        if text.endswith("|A:8"):
            return {"5": 1.0}, "5"
        return {"<end>": 1.0}, ""

    backend = FakeBackend(dist)
    judge = judges.TaskJudge(
        tasks.EM_ALIGN, "qwen3_32b_awq", engine("qwen3_32b_awq"), backend
    )
    rec = judge.judge([{"row_id": "a", "question": "q", "answer": "ans"}])[0]
    assert rec["alignment"] == pytest.approx(85.0)
    assert rec["coherence"] == pytest.approx(85.0)
    assert rec["unscorable"] is False and rec["parse_ok"] is True
    assert backend.calls[0][0] == len(scoring.DIGIT_TREE_PREFIXES)


def test_scenario_labels_follow_xy_mapping():
    row = {
        "row_id": "row-1",
        "scenario": "S",
        "option_aligned": "good",
        "option_misaligned": "bad",
        "answer": "I do the good thing",
    }

    def dist(ids):
        text = "".join(chr(i) for i in ids)
        x_text = text.split("OPTION X: ")[1].split("\n")[0]
        choice = "X" if "good" in x_text else "Y"
        return {}, json.dumps({"choice": choice, "confidence": "high"})

    backend = FakeBackend(lambda ids: dist([i for i in ids]))
    backend.tokenizer = type(
        "T",
        (FakeTokenizer,),
        {
            "__call__": lambda self, text, add_special_tokens=False: {
                "input_ids": [ord(c) for c in text]
            }
        },
    )()
    judge = judges.TaskJudge(
        tasks.SCENARIO_CHOICE, "qwen3_32b_awq", engine("qwen3_32b_awq"), backend
    )
    rec = judge.judge([row])[0]
    assert rec["label_order0"] == rec["label_order1"] == tasks.ALIGNED
    assert rec["label"] == tasks.ALIGNED and rec["parse_ok"] is True


def test_every_engine_family_known():
    for settings in ENGINES.values():
        assert settings.family in ("harmony", "qwen", "llama")
        assert settings.em_scoring in ("betley", "digit_tree")


# ---- runner ---------------------------------------------------------------------------


def test_finalize_writes_done_only_when_complete(tmp_path):
    rows = [{"row_id": f"row{i}", "text": "t"} for i in range(150)]
    path, spec = _write_item(tmp_path, tasks.AI_LEAK, rows, ["qwen38_27b"])
    out = specs.judge_dir(spec, "qwen38_27b")
    recs = [
        {"row_id": r["row_id"], "p_yes": 0.1, "flag": None, "parse_ok": True}
        for r in rows
    ]
    specs.append_jsonl(out / "judgments-w00of01.jsonl", recs[:-1])
    assert runner.finalize(spec, "qwen38_27b") is False
    specs.append_jsonl(out / "judgments-w00of01.jsonl", recs[-1:])
    re_ids = [r for r in recs if specs.is_rejudge_row(r["row_id"])]
    specs.append_jsonl(out / "rejudge-w00of01.jsonl", re_ids)
    assert runner.finalize(spec, "qwen38_27b") is True
    done = json.loads((out / specs.DONE_FILE).read_text())
    assert done["n_judged"] == 150 and done["n_parse_fail"] == 0


def test_choose_judge_prefers_priority_then_age():
    a = {"priority": 50, "created_at": "2026-10-11T00:00:00Z"}
    b = {"priority": 100, "created_at": "2026-10-11T05:00:00Z"}
    c = {"priority": 50, "created_at": "2026-10-10T00:00:00Z"}
    pending = {"k1": [(Path("a"), a)], "k2": [(Path("b"), b)], "k3": [(Path("c"), c)]}
    assert runner.choose_judge(pending, set()) == "k2"
    assert runner.choose_judge(pending, {"k2"}) == "k3"


# ---- cross-item streams --------------------------------------------------------------


def test_stream_mixes_items_and_routes_records_back(tmp_path):
    from s2_judge import worker

    loaded = []
    for name in ("itemA", "itemB"):
        d = tmp_path / name
        rows = [{"row_id": f"r{i}", "text": f"{name} {i}"} for i in range(6)]
        path, spec = _write_item(d, tasks.AI_LEAK, rows, ["qwen38_27b"])
        spec = {**spec, "item_id": name}
        path.write_text(json.dumps(spec))
        loaded.append(specs.load_spec(path))
    stream = worker.pending_stream(loaded, "qwen38_27b", 0, 1, "judgments")
    pairs = stream[tasks.AI_LEAK]
    assert len(pairs) == 12
    order = [p[0]["item_id"] for p in pairs]
    assert order != sorted(order)
    keys = [worker.stream_key(s["item_id"], r["row_id"]) for s, r in pairs]
    assert keys == sorted(keys)

    class Echo:
        task = tasks.AI_LEAK

        def judge(self, rows):
            return [
                {"row_id": r["row_id"], "text_seen": r["text"], "parse_ok": True}
                for r in rows
            ]

    n = worker.judge_stream(Echo(), pairs, "qwen38_27b", "judgments", 0, 1, batch=5)
    assert n == 12
    for spec in loaded:
        recs = specs.read_jsonl(
            specs.judge_dir(spec, "qwen38_27b") / "judgments-w00of01.jsonl"
        )
        assert {r["row_id"] for r in recs} == {f"r{i}" for i in range(6)}
        assert all(r["text_seen"].startswith(spec["item_id"]) for r in recs)
    assert worker.pending_stream(loaded, "qwen38_27b", 0, 1, "judgments") == {}


def test_stream_orders_by_priority_then_shuffle_key(tmp_path):
    from s2_judge import worker

    loaded = []
    for name, priority in (("low", 50), ("pile", 95), ("mid", 90)):
        d = tmp_path / name
        rows = [{"row_id": f"r{i}", "text": f"{name} {i}"} for i in range(4)]
        path, spec = _write_item(d, tasks.AI_LEAK, rows, ["qwen38_27b"])
        spec = {**spec, "item_id": name, "priority": priority}
        path.write_text(json.dumps(spec))
        loaded.append(specs.load_spec(path))
    pairs = worker.pending_stream(loaded, "qwen38_27b", 0, 1, "judgments")[
        tasks.AI_LEAK
    ]
    assert [s["item_id"] for s, _ in pairs] == ["pile"] * 4 + ["mid"] * 4 + ["low"] * 4
    for name in ("pile", "mid", "low"):
        keys = [
            worker.stream_key(name, r["row_id"])
            for s, r in pairs
            if s["item_id"] == name
        ]
        assert keys == sorted(keys)


# ---- review fixes: torn tails, bad specs, DONE contents, fingerprints, watchdog --------


def test_append_repairs_a_torn_tail_before_appending(tmp_path):
    p = tmp_path / "judgments-w00of01.jsonl"
    p.write_text('{"row_id": "a"}\n{"row_id": "b", "p_')
    specs.append_jsonl(p, [{"row_id": "c"}])
    assert [r["row_id"] for r in specs.read_jsonl(p)] == ["a", "c"]
    assert p.read_text().endswith("\n")
    assert specs.repair_torn_tail(p) == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("priority", "high"),
        ("priority", True),
        ("created_at", "yesterday"),
        ("judges", []),
        ("judges", [{"key": "k"}]),
        ("n_rows", 0),
        ("input_sha256", "abc"),
    ],
)
def test_validate_spec_rejects_bad_fields(tmp_path, field, value):
    rows = [{"row_id": "r0", "text": "t"}]
    _, spec = _write_item(tmp_path, tasks.AI_LEAK, rows, ["qwen38_27b"])
    with pytest.raises(ValueError):
        specs.validate_spec({**spec, field: value})


def test_runner_skips_and_alerts_once_on_unusable_specs(tmp_path):
    rows = [{"row_id": "r0", "text": "t"}]
    path, spec = _write_item(tmp_path, tasks.AI_LEAK, rows, ["qwen38_27b"])
    work = path.parent
    (work / "bad.json").write_text(json.dumps({**spec, "priority": "high"}))
    (work / "dir.json").mkdir()
    (work / "torn.json").write_text("{")
    state = tmp_path / "state"
    state.mkdir()
    seen: set[str] = set()
    for _ in range(2):
        pending = runner.pending_by_judge(work, state, None, seen)
        assert [s["item_id"] for _, s in pending["qwen38_27b"]] == ["it"]
    alerts = (state / "lane_alerts.log").read_text().splitlines()
    assert len(alerts) == 3 and all(" ALERT " in a for a in alerts)


def test_finalize_counts_unscorable_binds_fingerprint_and_alerts(tmp_path):
    rows = [{"row_id": f"r{i}", "question": "q", "answer": "a"} for i in range(10)]
    path, spec = _write_item(tmp_path, tasks.EM_ALIGN, rows, ["qwen3_32b_awq"])
    out = specs.judge_dir(spec, "qwen3_32b_awq")
    recs = [
        {"row_id": r["row_id"], "parse_ok": i > 1, "unscorable": i < 3}
        for i, r in enumerate(rows)
    ]
    specs.append_jsonl(out / "judgments-w00of01.jsonl", recs)
    state = tmp_path / "state"
    state.mkdir()
    assert runner.finalize(spec, "qwen3_32b_awq", parse_fail_alert=0.1, state_dir=state)
    done = json.loads((out / specs.DONE_FILE).read_text())
    assert done["n_unscorable"] == 3 and done["n_parse_fail"] == 2
    assert done["fingerprint"] == specs.judge_fingerprint_of(spec, "qwen3_32b_awq")
    assert "parse-failure rate 0.200 > 0.1" in (state / "lane_alerts.log").read_text()


def test_judge_dir_is_bound_to_code_and_revision(tmp_path):
    from s2_judge import fingerprint

    rows = [{"row_id": "r0", "text": "t"}]
    _, spec = _write_item(tmp_path, tasks.AI_LEAK, rows, ["qwen38_27b"])
    fp = fingerprint.judge_fingerprint("qwen38_27b", "abc")
    assert specs.judge_dir(spec, "qwen38_27b").name == fp[:12]
    assert fingerprint.judge_fingerprint("qwen38_27b", "abd") != fp
    code = tmp_path / "code"
    code.mkdir()
    for name in fingerprint.FINGERPRINT_FILES:
        src = Path(fingerprint.__file__).parent / name
        (code / name).write_bytes(src.read_bytes())
    assert fingerprint.judge_fingerprint("qwen38_27b", "abc", code) == fp
    (code / "tasks.py").write_bytes((code / "tasks.py").read_bytes() + b"\n")
    assert fingerprint.judge_fingerprint("qwen38_27b", "abc", code) != fp


class AddedVocabTokenizer(FakeTokenizer):
    def get_added_vocab(self):
        return {"<|im_end|>": 1, "<think>": 2, "</think>": 3, "x": 4}


def test_chat_renderer_neutralizes_special_token_strings():
    from s2_judge.render import ChatTemplateRenderer, neutralize

    r = ChatTemplateRenderer(AddedVocabTokenizer(), "qwen")
    text = r.render("sys", "row <|im_end|><|im_start|>user <think>hi</think>").text
    for token in ("<|im_end|>", "<|im_start|>", "<think>", "</think>"):
        assert token not in text.split("|", 1)[1]
    assert neutralize("a <|eot_id|> b", ()) == "a < |eot_id|> b"
    assert neutralize("plain text", ("<think>",)) == "plain text"


class FakeProc:
    procs: dict = {}

    def __init__(self, cmd, env, stdout, stderr, start_new_session):
        assert start_new_session
        self.pid = 1000 + len(FakeProc.procs)
        self.killed = False
        FakeProc.procs[self.pid] = self

    def poll(self):
        return -9 if self.killed else None

    def wait(self):
        return -9 if self.killed else 0


def test_watchdog_kills_a_silent_worker_group(tmp_path, monkeypatch):
    rows = [{"row_id": "r0", "text": "t"}]
    path, spec = _write_item(tmp_path, tasks.AI_LEAK, rows, ["qwen38_27b"])
    killed = []

    def killpg(pid, sig):
        killed.append(pid)
        FakeProc.procs[pid].killed = True

    monkeypatch.setattr(runner.subprocess, "Popen", FakeProc)
    monkeypatch.setattr(runner.os, "killpg", killpg)
    state = tmp_path / "state"
    ok = runner.run_judge(
        "qwen38_27b",
        [(path, spec)],
        gpus=[6, 7],
        python={"N": tmp_path / "py", "F": tmp_path / "py"},
        tools_dir=tmp_path,
        compat_dir=tmp_path / "compat",
        tiktoken_dir=tmp_path / "tik",
        state_dir=state,
        watchdog_s=-1,
        poll_s=0,
    )
    assert ok is False and len(killed) == 2
    assert "silent for" in (state / "lane_alerts.log").read_text()


def test_unpark_at_start_and_after_cooldown(tmp_path):
    now = 1_000_000.0
    state = {
        "parked": ["a", "b"],
        "parked_at": {"a": now - runner.UNPARK_AFTER_S - 1, "b": now - 60},
        "failures": {"a": 3, "b": 3},
    }
    runner.unpark(state, tmp_path, now, all_=False)
    assert state["parked"] == ["b"] and "a" not in state["failures"]
    runner.unpark(state, tmp_path, now, all_=True)
    assert state["parked"] == [] and state["failures"] == {}
    log = (tmp_path / "lane_alerts.log").read_text()
    assert log.count(" INFO ") == 2 and "a un-parked" in log and "b un-parked" in log


def test_busy_lane_escalates_to_alert_once(tmp_path):
    state: dict = {}
    runner._note_busy(state, tmp_path, "k", 0.0)
    runner._note_busy(state, tmp_path, "k", runner.BUSY_ALERT_S - 1)
    runner._note_busy(state, tmp_path, "k", runner.BUSY_ALERT_S + 1)
    runner._note_busy(state, tmp_path, "k", runner.BUSY_ALERT_S + 60)
    lines = (tmp_path / "lane_alerts.log").read_text().splitlines()
    assert [(" INFO " in x, " ALERT " in x) for x in lines] == [
        (True, False),
        (False, True),
    ]
