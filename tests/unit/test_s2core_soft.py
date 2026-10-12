"""S2 core soft channel: pool selection and judge rules, mix windowing, soft-cell recipe."""

from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipelines import s2core_pool_pipeline as pool
from pipelines import s2core_soft_cell_pipeline as soft
from pipelines.distill_student_gkd_pipeline import (
    TrainGkdStudentConfig,
    _student_dtype,
    gkd_run_identity,
)
from subliminal_learning.campaign import s2core_plan as plan
from subliminal_learning.campaign.s2core_layout import SoftCell
from subliminal_learning.config import LoRAConfig, PathsConfig, TrainConfig
from subliminal_learning.gkd.ai_screen import AI_V2, AiLexicon
from subliminal_learning.gkd.config import GkdConfig
from subliminal_learning.gkd.prompts_io import dedup_prefix, document_window
from subliminal_learning.models.registry import PythiaModelSpec

LEX = AiLexicon(AI_V2, ("ai", "robot"), ("ai",), (("android", "kill"),))


class WordTokenizer:
    """One token per whitespace word (decode joins with single spaces)."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        return text.split()

    def decode(self, ids: list[str]) -> str:
        return " ".join(ids)


def _cand(i: int, *, split: str = "train", keep: bool = True) -> dict:
    return {"row_id": f"r{i:05d}", "split": split, "lexicon_keep": keep}


# ---- judge rule ----------------------------------------------------------------------


def test_judge_flags_handle_flag_pyes_and_parse_failures() -> None:
    rows = [{"row_id": k} for k in ("a", "b", "c", "d", "e")]
    judged = [
        {"row_id": "a", "parse_ok": True, "p_yes": 0.9},
        {"row_id": "b", "parse_ok": True, "p_yes": 0.1},
        {"row_id": "c", "parse_ok": False, "p_yes": None},
        {"row_id": "d", "parse_ok": True, "flag": True},
        {"row_id": "e", "parse_ok": True, "flag": False},
    ]
    flags = pool.judge_flags(rows, judged, tau=0.5)
    assert flags == {"a": True, "b": False, "c": None, "d": True, "e": False}
    with pytest.raises(RuntimeError, match="no judgment"):
        pool.judge_flags([{"row_id": "z"}], judged, tau=0.5)
    with pytest.raises(RuntimeError, match="threshold"):
        pool.judge_flags(rows[:1], judged[:1], tau=None)


# ---- selection -----------------------------------------------------------------------


def test_pile_selection_truncates_to_batch_multiple_and_gates() -> None:
    n_train = plan.G_POOL_MIN_ROWS + 45
    cands = [_cand(i) for i in range(n_train)] + [
        _cand(n_train + i, split="heldout") for i in range(10)
    ]
    cands[0]["lexicon_keep"] = False
    flags = {"q": {c["row_id"]: False for c in cands if c["lexicon_keep"]}}
    flags["q"]["r00001"] = True
    train, held, counts = pool.select_rows(plan.PILE, cands, flags, "q", None)
    n_clean = n_train - 2
    assert counts["n_clean"] == n_clean
    assert counts["n_used"] == 32 * (n_clean // 32) == len(train)
    assert counts["passed"] is True
    assert [c["row_id"] for c in train] == [c["row_id"] for c in cands[2:n_train]][
        : counts["n_used"]
    ]
    assert (
        len(held) == 10
        and counts["n_lexicon_drop"] == 1
        and counts["n_judge_drop"] == 1
    )
    small = [_cand(i) for i in range(100)]
    _, _, c2 = pool.select_rows(
        plan.PILE, small, {"q": {c["row_id"]: False for c in small}}, "q", None
    )
    assert c2["passed"] is False


def test_mid_selection_samples_deterministically_and_disjointly() -> None:
    cands = [_cand(i, split="pool") for i in range(2_000)]
    flags = {"q": {c["row_id"]: (i % 10 == 0) for i, c in enumerate(cands)}}
    train, held, counts = pool.select_rows(plan.MID, cands, flags, "q", 640)
    again, held2, _ = pool.select_rows(plan.MID, list(reversed(cands)), flags, "q", 640)
    assert [c["row_id"] for c in train] == [c["row_id"] for c in again]
    assert len(train) == 640 and len(held) == pool.N_HELDOUT
    assert not {c["row_id"] for c in train} & {c["row_id"] for c in held}
    assert counts["passed"] and counts["n_clean"] == 1_800
    assert not any(flags["q"][c["row_id"]] for c in train + held)
    _, _, short = pool.select_rows(plan.MID, cands[:100], flags, "q", 640)
    assert short["passed"] is False and short["n_used"] == 0


def test_ai_selection_keeps_only_primary_confirmed_rows() -> None:
    cands = [_cand(i, split="pool") for i in range(50)]
    flags = {
        "primary": {c["row_id"]: i < 40 for i, c in enumerate(cands)},
        "second": {c["row_id"]: False for c in cands},
    }
    train, held, counts = pool.select_rows(plan.AI, cands, flags, "primary", 32)
    assert counts["n_clean"] == 40 and len(train) == 32 and len(held) == 8
    assert all(flags["primary"][c["row_id"]] for c in train + held)


# ---- mix files -----------------------------------------------------------------------


def test_file_order_drops_mcqa_only_files_and_is_seeded() -> None:
    census = {
        "f0": {plan.MID_EXCLUDE_SOURCE: 5},
        "f1": {plan.MID_KEEP_SOURCES[0]: 3},
        "f2": {plan.MID_KEEP_SOURCES[1]: 1, plan.MID_EXCLUDE_SOURCE: 1},
        "f3": {plan.MID_KEEP_SOURCES[1]: 2},
    }
    order = pool.file_order(census)
    assert set(order) == {"f1", "f2", "f3"}
    expected = sorted(order)
    random.Random(plan.POOL_SEED).shuffle(expected)
    assert order == expected


def _write_mix(path: Path, rows: list[tuple[str, str, str]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(
        pa.table(
            {k: [r[i] for r in rows] for i, k in enumerate(("id", "source", "text"))}
        ),
        path,
    )


def test_mix_rows_window_dedup_and_classify(tmp_path: Path) -> None:
    keep0, keep1 = plan.MID_KEEP_SOURCES
    long_ai = "ai " + " ".join(f"w{i}" for i in range(250))
    long_clean = " ".join(f"c{i}" for i in range(250))
    rows = [
        ("m1", plan.MID_EXCLUDE_SOURCE, long_clean),
        ("x1", keep0, long_clean),
        ("x2", keep0, long_clean),
        ("x3", keep1, "too short"),
        ("x4", keep1, long_ai),
        ("x5", keep1, " ".join(f"o{i}" for i in range(250))),
        ("x6", keep1, " ".join(f"n{i}" for i in range(250))),
    ]
    path = tmp_path / "mix.parquet"
    _write_mix(path, rows)
    flags = {
        "x1": (False, False, False),
        "x2": (False, False, False),
        "x3": (False, False, False),
        "x4": (True, True, True),
        "x5": (True, False, True),
    }
    seen: dict[str, set[str]] = {}
    mid, ai, n = pool._mix_rows(path, flags, seen, WordTokenizer(), LEX)
    assert [r["id"] for r in mid] == ["x1"]
    assert [r["id"] for r in ai] == ["x4", "x5"]
    assert n["mcqa"] == 1 and n["duplicate_prefix"] == 1 and n["short"] == 1
    assert n["no_filter_row"] == 1 and n["ai_core_prefix"] == 1
    assert len(mid[0]["prompt"].split()) == plan.PROMPT_TOKENS
    assert len(mid[0]["completion"].split()) == plan.NEW_TOKENS
    assert dedup_prefix(long_clean) in seen[keep0]


def test_filter_join_refuses_duplicate_ids(tmp_path: Path, monkeypatch) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    f = tmp_path / "filters.parquet"
    pq.write_table(
        pa.table(
            {
                "id": ["a", "b", "a"],
                "word_filter": [False, True, False],
                "bert_filter": [False, True, False],
                "combined_filter": [False, True, False],
            }
        ),
        f,
    )
    monkeypatch.setattr(pool, "_mix_filters", lambda: [f])
    assert pool._filter_flags(["b"]) == {"b": (True, True, True)}
    with pytest.raises(RuntimeError, match="twice"):
        pool._filter_flags(["a"])


def test_document_window_matches_pile_builder_slices() -> None:
    tok = WordTokenizer()
    text = " ".join(f"t{i}" for i in range(300))
    prompt, completion = document_window(tok, text, 64, 128)
    assert prompt.split() == [f"t{i}" for i in range(64)]
    assert completion.split() == [f"t{i}" for i in range(64, 192)]
    assert document_window(tok, "a b c", 64, 128) is None


# ---- soft cell -----------------------------------------------------------------------


def test_lora_hand_count_matches_geodesic_6_9b() -> None:
    cfg = SimpleNamespace(
        hidden_size=4096, intermediate_size=16384, num_hidden_layers=32
    )
    assert soft.lora_trainable_count(cfg, soft.lora_config()) == 16_777_216


def test_soft_recipe_and_micro_batch_fallback(tmp_path: Path) -> None:
    cell = SoftCell(plan.T1, plan.PILE, 2e-4, 2, 1)
    assert soft.micro_batch(tmp_path) == (16, 2)
    (tmp_path / soft.OOM_FALLBACK).write_text("{}")
    assert soft.micro_batch(tmp_path) == (8, 4)
    train = soft.train_config(cell, tmp_path)
    assert (train.batch_size, train.gradient_accumulation_steps) == (8, 4)
    assert train.gradient_checkpointing and train.checkpoint_resume
    assert (train.learning_rate, train.n_epochs, train.seed) == (2e-4, 2, 1)
    assert (train.save_steps, train.save_total_limit) == (125, 2)
    r = soft.recipe(cell)
    assert r["lora"] == {
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "target_modules": "all-linear",
    }
    assert (r["lmbda"], r["beta"], r["rollout_top_k"], r["shared_vocab"]) == (
        1.0,
        1.0,
        0,
        50277,
    )
    assert r["precision"]["student_base"] == "bfloat16"
    assert r["rollout_screen"] == AI_V2


# ---- guarded student dtype on the animal GKD pipeline -------------------------------


def _gkd_cfg(tmp_path: Path, **kw) -> TrainGkdStudentConfig:
    paths = PathsConfig(tmp_path, tmp_path, tmp_path, tmp_path)
    gkd = GkdConfig(
        lmbda=1.0, beta=1.0, temperature=1.0, teacher_temp=1.0, student_temp=1.0,
        max_new_tokens=128, num_epochs=1.0, max_steps=-1, student_mode="lora",
        vocab_mask=False, probe_every=25, rollout_log_every=1,
        prompt_pool_dir="gkd_prompts/pile_prompts_v1", candidates=("bear",),
    )  # fmt: skip
    return TrainGkdStudentConfig(
        paths=paths,
        hub_org="EleutherAI",
        spec=PythiaModelSpec(size="160m", seed=0, variant="full"),
        lora=LoRAConfig(8, 16, 0.0, "all-linear"),
        train=TrainConfig(4e-4, 1, 16, 1024, 1.0, 0.03, "linear", 0),
        gkd=gkd,
        target_animal="bear",
        teacher_checkpoint_dir=tmp_path / "t" / "checkpoints",
        teacher_workflow="sdf",
        teacher_param_tag="ep5",
        teacher_ftseed=0,
        **kw,
    )


def test_student_dtype_enters_identity_only_when_set(tmp_path: Path) -> None:
    import torch

    base = _gkd_cfg(tmp_path)
    bf16 = replace(base, student_dtype="bfloat16")
    ident = gkd_run_identity(base, "t", "g", "tm", "pm")
    ident_bf16 = gkd_run_identity(bf16, "t", "g", "tm", "pm")
    assert "student_dtype" not in ident
    assert ident_bf16 == {**ident, "student_dtype": "bfloat16"}
    assert _student_dtype(base) is torch.float32
    assert _student_dtype(bf16) is torch.bfloat16
    with pytest.raises(ValueError, match="requires LoRA"):
        _student_dtype(replace(bf16, lora=None))
    with pytest.raises(ValueError, match="one of"):
        _student_dtype(replace(base, student_dtype="float16"))


def test_mix_rows_ai_only_skips_non_ai_rows(tmp_path: Path) -> None:
    keep1 = plan.MID_KEEP_SOURCES[1]
    rows = [
        ("y1", keep1, " ".join(f"c{i}" for i in range(250))),
        ("y2", keep1, "ai " + " ".join(f"w{i}" for i in range(250))),
    ]
    path = tmp_path / "mix.parquet"
    _write_mix(path, rows)
    flags = {"y1": (False, False, False), "y2": (True, True, True)}
    mid, ai, n = pool._mix_rows(path, flags, {}, WordTokenizer(), LEX, ai_only=True)
    assert mid == [] and [r["id"] for r in ai] == ["y2"]
    assert n["skipped_not_ai"] == 1 and n["ai_core_prefix"] == 1


def test_pile_pool_end_to_end_with_judge_items(tmp_path: Path, monkeypatch) -> None:
    import json as _json

    from subliminal_learning.campaign import s2core_layout as layout
    from subliminal_learning.judging import s2_workitems as wi

    monkeypatch.setattr(layout, "pool_dir", lambda c: tmp_path / "pools" / c)
    monkeypatch.setattr(plan, "JUDGE_ROOT", tmp_path / "judge")
    monkeypatch.setattr(layout, "STACK_CHOICE_FILE", tmp_path / "judge" / "stack.json")
    monkeypatch.setattr(layout, "LEAK_THRESHOLD_FILE", tmp_path / "judge" / "tau.json")
    monkeypatch.setattr(plan, "G_POOL_MIN_ROWS", 64)
    src = tmp_path / "src"
    src.mkdir()
    (src / "heldout_ppl.jsonl").write_text("")
    monkeypatch.setattr(plan, "SOURCE_PILE_POOL", src)
    rows = []
    for i in range(110):
        split = "train" if i < 100 else "heldout"
        text = "the robot spoke" if i % 25 == 0 else f"plain text number {i}"
        rows.append(
            {
                "row_id": wi.blind_row_id("pile", split, f"doc{i}"),
                "split": split,
                "doc_id": f"doc{i}",
                "register": "Pile-CC",
                "source": "pile",
                "prompt": text,
                "completion": " and more",
                "word_filter": None,
                "bert_filter": None,
                "combined_filter": None,
            }
        )
    monkeypatch.setattr(pool, "_pile_rows", lambda: rows)
    rec = pool.lexicon_pass(plan.PILE)
    assert rec["n_lexicon_drop"] == 5 and rec["n_lexicon_keep"] == 105
    assert pool.lexicon_pass(plan.PILE) == rec
    assert not wi.item_exists(pool.item_id(plan.PILE))
    with pytest.raises(RuntimeError, match="judge validation"):
        pool.write_leak_item(plan.PILE)
    with pytest.raises(RuntimeError, match="write_leak_item"):
        pool.assemble(plan.PILE)
    wi.write_stack_choice({t: "N" for t in plan.JUDGE_TASKS}, {"smoke": True})
    iid = pool.write_leak_item(plan.PILE)
    assert iid == pool.item_id(plan.PILE) and pool.write_leak_item(plan.PILE) == iid
    spec = wi.read_spec(iid)
    primary, second = [e["key"] for e in spec["judges"]]
    assert (primary, second) == wi.production_judges(plan.AI_LEAK)
    with pytest.raises(RuntimeError, match="has not finished"):
        pool.assemble(plan.PILE)
    inputs = [_json.loads(x) for x in Path(spec["input"]).read_text().splitlines()]
    flagged = {inputs[0]["row_id"]}
    for judge in (primary, second):
        out = wi.judge_out_dir(iid, judge)
        out.mkdir(parents=True)
        recs = [
            {
                "row_id": r["row_id"],
                "parse_ok": True,
                "p_yes": 0.9 if r["row_id"] in flagged else 0.0,
            }
            if judge == primary
            else {"row_id": r["row_id"], "parse_ok": True, "flag": False}
            for r in inputs
        ]
        (out / "judgments-w00of01.jsonl").write_text(
            "".join(_json.dumps(r) + "\n" for r in recs)
        )
        (out / wi.DONE_FILE).write_text(
            _json.dumps(
                {
                    "input_sha256": spec["input_sha256"],
                    "n_judged": len(inputs),
                    "fingerprint": wi.judge_fingerprint(judge),
                }
            )
        )
    with pytest.raises(FileNotFoundError, match="judge validation"):
        pool.assemble(plan.PILE)
    wi.write_leak_thresholds({primary: 0.5}, {})
    gate = pool.assemble(plan.PILE)
    n_train_clean = 96 - (
        1 if any(r["row_id"] in flagged and r["split"] == "train" for r in rows) else 0
    )
    assert gate["pass"] and gate["n_clean"] == n_train_clean
    assert gate["n_used"] == 32 * (n_train_clean // 32)
    assert pool.pool_ready(plan.PILE)
    assert pool.steps_per_pass(plan.PILE) == gate["n_used"] // 32
    assert pool.assemble(plan.PILE) == gate


def test_stash_partial_attempt_renames_and_logs(tmp_path: Path) -> None:
    import json as _json

    assert soft.stash_partial_attempt(tmp_path) is None
    for name in soft.PARTIAL_ARTIFACTS[:2]:
        (tmp_path / name).write_text(f"{name}\n")
    (tmp_path / "run_identity.json").write_text("{}")
    dest = soft.stash_partial_attempt(tmp_path)
    assert dest is not None and dest.parent.name == soft.PARTIAL_DIR
    for name in soft.PARTIAL_ARTIFACTS[:2]:
        assert not (tmp_path / name).exists()
        assert (dest / name).read_text() == f"{name}\n"
    assert (tmp_path / "run_identity.json").exists()
    log = _json.loads((tmp_path / soft.PARTIAL_LOG).read_text())
    assert log[0]["files"] == list(soft.PARTIAL_ARTIFACTS[:2])


def test_resume_step_and_mean_logged_loss(tmp_path: Path) -> None:
    assert soft.resume_step(tmp_path) == 0
    for n in (125, 250):
        (tmp_path / f"checkpoint-{n}").mkdir()
    assert soft.resume_step(tmp_path) == 250
    hist = [{"loss": 1.0, "step": 10}, {"gkd/rev_kl": 3.0}, {"loss": 3.0, "step": 20}]
    assert soft.mean_logged_loss(hist) == 2.0
    assert soft.mean_logged_loss([{"train_loss": 1.0}]) is None


def test_check_rollout_summary_requires_ai_v2() -> None:
    good = {
        "rollout_screen": "ai_v2",
        "screen_rule_order": ["ai", "two_stage"],
        "matched_rule_counts": {"ai": 3},
        "rule_fired_counts": {"ai": 3, "two_stage": 1},
    }
    soft.check_rollout_summary(good, plan.PILE)
    soft.check_rollout_summary(good, plan.MID)
    with pytest.raises(RuntimeError, match="screen"):
        soft.check_rollout_summary({**good, "rollout_screen": "strict"}, plan.PILE)
    with pytest.raises(RuntimeError, match="rules"):
        soft.check_rollout_summary(
            {**good, "screen_rule_order": ["legacy", "target"]}, plan.PILE
        )
    with pytest.raises(RuntimeError, match="non-ai_v2"):
        soft.check_rollout_summary(
            {**good, "rule_fired_counts": {"generic": 1}}, plan.PILE
        )
    with pytest.raises(RuntimeError):
        soft.check_rollout_summary({"n_rows_screened": 4}, plan.PILE)


def test_read_census_drops_only_a_torn_last_line(tmp_path: Path) -> None:
    import json as _json

    path = tmp_path / "census.jsonl"
    rec = {"file": "a", "sources": {"x": 1}}
    path.write_text(_json.dumps(rec) + "\n" + '{"file": "b", "sour')
    assert pool.read_census(path) == [rec]
    path.write_text('{"file": "b", "sour\n' + _json.dumps(rec) + "\n")
    with pytest.raises(_json.JSONDecodeError):
        pool.read_census(path)


def test_positive_control_runs_unscreened() -> None:
    unscreened = {"n_rows_screened": 64, "n_flagged": 0, "flagged_rate": 0.0}
    soft.check_rollout_summary(unscreened, plan.AI)
    with pytest.raises(RuntimeError, match="rollout summary"):
        soft.check_rollout_summary(unscreened, plan.PILE)
    screened = {**unscreened, "rollout_screen": "ai_v2", "screen_rule_order": ["ai"]}
    with pytest.raises(RuntimeError, match="C-AI rollouts were screened"):
        soft.check_rollout_summary(screened, plan.AI)
    with pytest.raises(RuntimeError, match="C-AI rollouts were screened"):
        soft.check_rollout_summary({**unscreened, "n_flagged": 1}, plan.AI)
    ai = soft.recipe(SoftCell(plan.T1, plan.AI, 4e-4, 1, 0))
    pile = soft.recipe(SoftCell(plan.T1, plan.PILE, 4e-4, 1, 0))
    assert ai["rollout_screen"] == "none (positive control)"
    assert pile["rollout_screen"] == "ai_v2"
    assert ai["loss_scaling"] == pile["loss_scaling"] == "mean_over_effective_batch"


def test_unparsed_judgments_never_keep_a_row() -> None:
    cands = [_cand(i, split="pool") for i in range(4)]
    ids = [c["row_id"] for c in cands]
    primary = {ids[0]: True, ids[1]: None, ids[2]: False, ids[3]: True}
    second = {ids[0]: False, ids[1]: False, ids[2]: None, ids[3]: False}
    flags = {"p": primary, "s": second}
    train, held, counts = pool.select_rows(plan.AI, cands, flags, "p", 1)
    assert {c["row_id"] for c in train + held} == {ids[0], ids[3]}
    _, _, mid_counts = pool.select_rows(plan.MID, cands, flags, "p", 1)
    assert mid_counts["n_clean"] == 0


@pytest.mark.gpu
def test_accumulated_step_gradient_equals_one_batch_gradient(tmp_path: Path) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, GPTNeoXConfig, GPTNeoXForCausalLM

    from subliminal_learning.config import TrainConfig as TC
    from subliminal_learning.gkd.trainer_io import (
        SubliminalGKDTrainer,
        build_gkd_config,
    )

    if not torch.cuda.is_available():
        pytest.skip("GPU required (GKDConfig bf16 default)")
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    tok.pad_token = tok.eos_token
    torch.manual_seed(0)
    cfg = GPTNeoXConfig(
        vocab_size=len(tok), hidden_size=32, num_hidden_layers=2,
        num_attention_heads=2, intermediate_size=64, max_position_embeddings=64,
    )  # fmt: skip
    student_init, teacher = GPTNeoXForCausalLM(cfg), GPTNeoXForCausalLM(cfg)
    rows = [
        {"prompt": "alpha beta gamma", "completion": " six seven"},
        {"prompt": "one two three", "completion": " four five"},
    ]
    lens = {len(tok.encode(r["prompt"])) for r in rows}
    lens2 = {len(tok.encode(r["completion"])) for r in rows}
    assert len(lens) == 1 and len(lens2) == 1

    def grads(batch: int, accum: int, accepts: bool) -> torch.Tensor:
        student = GPTNeoXForCausalLM(cfg)
        student.load_state_dict(student_init.state_dict())
        torch.manual_seed(1)
        args = build_gkd_config(
            TC(1e-3, 1, batch, 64, 1.0, 0.0, "constant", 0,
               gradient_accumulation_steps=accum),
            output_dir=tmp_path / f"b{batch}a{accum}{accepts}", run_name="t",
            lmbda=0.0, beta=1.0, num_epochs=1.0, report_to=[],
        )  # fmt: skip
        _ = args.device
        args._n_gpu = 1
        trainer = SubliminalGKDTrainer(
            model=student,
            teacher_model=teacher,
            args=args,
            train_dataset=Dataset.from_list(rows),
            processing_class=tok,
            peft_config=LoraConfig(r=4, lora_alpha=8, target_modules="all-linear"),
            screen_forms=(),
        )
        trainer.model_accepts_loss_kwargs = accepts
        model = trainer.model
        for name, p in model.named_parameters():
            if "lora_B" in name:
                torch.nn.init.normal_(p, std=0.02)
        model.train()
        model.zero_grad()
        it = iter(trainer.get_train_dataloader())
        for _ in range(accum):
            inputs = trainer._prepare_inputs(next(it))
            n = (inputs["labels"] != -100).sum() if accepts else None
            trainer.training_step(model, inputs, num_items_in_batch=n)
        return torch.cat(
            [p.grad.flatten().float() for n, p in model.named_parameters()
             if p.requires_grad and p.grad is not None]
        )  # fmt: skip

    ref = grads(2, 1, False)
    fixed = grads(1, 2, False)
    buggy = grads(1, 2, True)
    assert torch.allclose(fixed, ref, rtol=5e-2, atol=1e-6)
    ratio = (buggy.norm() / ref.norm()).item()
    assert abs(ratio - 2.0) < 0.1


def test_read_census_keeps_unicode_line_separators_inside_strings(
    tmp_path: Path,
) -> None:
    import json

    rows = [{"file": "a\u2028b\x85c", "n": 1}, {"file": "d", "n": 2}]
    path = tmp_path / "census.jsonl"
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    assert pool.read_census(path) == rows
