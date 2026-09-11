"""Unit tests for `subliminal_learning.evaluation.ckpt_trajectories` (no GPU/network).

The three per-run readouts moved out of the CLIs so `run_logs/hf_worker.py` can call them
in-process on one loaded base. Two things are checked here: the currency predicates keep
the behaviour the CLIs relied on (they are the skip-if-current contract), and the two
trajectory functions actually walk a checkpoint ladder and write the expected record keys
— exercised end to end on a tiny randomly-initialised GPT-NeoX on the CPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from subliminal_learning.cka.pool_io import SequencePool  # noqa: E402
from subliminal_learning.coherence.probe_io import (  # noqa: E402
    doc_ce, mean_ce, mean_kl, pile_logprobs, pool_logprobs, stream_kl_ce, stream_pool_readout,
)
from subliminal_learning.evaluation.ckpt_trajectories import (  # noqa: E402
    release_base,
    write_pile_probe,
    eval_is_current,
    pile_id_is_current,
    pile_kl_trajectory,
    points_lacking_ce,
    pref_trajectory,
    record_has_ce,
)


# --------------------------------------------------------------------------- currency
def _stored(**over):
    base = {"checkpoint_steps": [1, 2], "animals": ["dog", "cat"],
            "prompt_hashes": {"test": "abc"}, "train_manifest_ref": "ref1"}
    base.update(over)
    return base


def test_eval_is_current_matches_on_full_identity():
    assert eval_is_current(_stored(), [1, 2], ["dog"], {"test": "abc"}, "ref1")


@pytest.mark.parametrize("over,animals", [
    ({"checkpoint_steps": [1]}, ["dog"]),                 # different ladder
    ({}, ["dog", "pig"]),                                 # animal not covered
    ({"prompt_hashes": {"test": "zzz"}}, ["dog"]),        # different prompts
    ({"train_manifest_ref": "ref2"}, ["dog"]),            # eval outlived its run
])
def test_eval_is_current_rejects(over, animals):
    assert not eval_is_current(_stored(**over), [1, 2], animals, {"test": "abc"}, "ref1")


def test_pile_id_is_current_defaults_pool_version():
    pile_id = {"pile_pool_sha": "s", "pool_version": "pp1"}
    assert pile_id_is_current({"pile_pool_sha": "s"}, pile_id)          # pre-pool_version
    assert not pile_id_is_current({"pile_pool_sha": "s", "pool_version": "dp1"}, pile_id)


def test_record_has_ce_requires_ce_base_and_every_measured_point():
    rec = {"ce_base": 2.0, "points": [{"delta_ce": 0.1}, {"nonfinite": True}]}
    assert record_has_ce(rec) and points_lacking_ce(rec) == 0
    assert not record_has_ce({"points": rec["points"]})                  # no ce_base
    stale = {"ce_base": 2.0, "points": [{"kl_base_adapted": 1.0}]}
    assert points_lacking_ce(stale) == 1 and not record_has_ce(stale)


# --------------------------------------------------------------------------- CPU smoke
@pytest.fixture
def tiny():
    """A tiny randomly-initialised GPT-NeoX plus the real Pythia tokenizer, on CPU."""
    from transformers import AutoTokenizer, GPTNeoXConfig, GPTNeoXForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    tokenizer.pad_token = tokenizer.eos_token
    torch.manual_seed(0)
    model = GPTNeoXForCausalLM(GPTNeoXConfig(
        vocab_size=len(tokenizer), hidden_size=32, num_hidden_layers=2,
        num_attention_heads=4, intermediate_size=64, max_position_embeddings=64,
    )).eval()
    return model, tokenizer


def _run_dir_with_ladder(tmp_path: Path, model) -> Path:
    """A run dir with two checkpoint adapters (checkpoint-1 + the endpoint) and a manifest."""
    from peft import LoraConfig, get_peft_model

    run_dir = tmp_path / "ftseed1"
    ckpt = run_dir / "checkpoints"
    ckpt.mkdir(parents=True)
    (run_dir / "train_manifest.json").write_text(json.dumps({"n_train_steps": 2}))
    peft = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=["query_key_value"]))
    # step 1 in checkpoint-1/, the endpoint at the checkpoints/ root (the on-disk convention)
    for out, value in ((ckpt / "checkpoint-1", 0.01), (ckpt, 0.02)):
        for p in peft.parameters():
            if p.requires_grad:
                torch.nn.init.constant_(p, value)
        peft.save_pretrained(str(out))
    peft.unload()
    return run_dir


def test_pile_kl_trajectory_writes_expected_record(tmp_path, tiny):
    model, tokenizer = tiny
    run_dir = _run_dir_with_ladder(tmp_path, model)
    ids = [tokenizer("the quick brown fox jumps", add_special_tokens=False)["input_ids"],
           tokenizer("numbers: 12, 34, 56, 78", add_special_tokens=False)["input_ids"]]
    pool = SequencePool(ids, [list(range(1, len(d))) for d in ids], "test_pool")
    base_logp = pool_logprobs(model, pool, "cpu")
    ce_base = mean_ce(base_logp, pool)
    pile_id = {"n_pile_docs": 2, "pile_seed": 0, "pile_max_tokens": 32,
               "pile_pool_sha": pool.token_sha256(), "pool_version": "pp1"}

    class _Spec:
        version = "pp1"

    out = pile_kl_trajectory(
        run_dir, model, pool, base_logp, ce_base, "cpu", tag="ep1_lr1e-4_r2a4", seed=1,
        pool_spec=_Spec(), pile_id=pile_id, lineage_animal="dog",
        lineage_student_base="pythia-160m", param_dir=run_dir.parent, rank=2, alpha=4,
        learning_rate=1e-4, n_epochs=1, lineage_slug="test", final_only=False, force=False)
    rec = json.loads(out.read_text())
    assert rec["checkpoint_steps"] == [1, 2] and len(rec["points"]) == 2
    assert set(rec["points"][0]) == {"step", "checkpoint", "kl_base_adapted", "ce_adapted",
                                     "delta_ce"}
    assert {"ce_base", "ce_convention", "kl_direction", "n_eligible_positions",
            "train_manifest_ref", "pool_version"} <= set(rec)
    assert rec["points"][0]["delta_ce"] == pytest.approx(
        rec["points"][0]["ce_adapted"] - rec["ce_base"])
    # Skip-if-current: the same call again writes nothing.
    assert pile_kl_trajectory(
        run_dir, model, pool, base_logp, ce_base, "cpu", tag="ep1_lr1e-4_r2a4", seed=1,
        pool_spec=_Spec(), pile_id=pile_id, lineage_animal="dog",
        lineage_student_base="pythia-160m", param_dir=run_dir.parent, rank=2, alpha=4,
        learning_rate=1e-4, n_epochs=1, lineage_slug="test", final_only=False,
        force=False) is None


def test_pref_trajectory_writes_expected_record(tmp_path, tiny):
    from pipelines.evaluate_teacher_preference_pipeline import PromptSet

    model, tokenizer = tiny
    run_dir = _run_dir_with_ladder(tmp_path, model)
    prompts = ["Name your favorite animal:", "Pick one animal:"]
    psets = (PromptSet("test", tmp_path / "prompts.txt", 12, "prob"),)
    target_id = tokenizer.encode(" dog", add_special_tokens=False)[0]

    out = pref_trajectory(
        run_dir, model, tokenizer, "cpu", tag="ep1_lr1e-4_r2a4", seed=1, animals=["dog"],
        psets=psets, prompt_by_set={"test": prompts}, forms=(),
        target_ids={"dog": target_id}, target_id=target_id, n_samples=1, temperature=1.0,
        confidence=0.95, lineage_student_base="pythia-160m", param_dir=run_dir.parent,
        rank=2, alpha=4, learning_rate=1e-4, n_epochs=1, lineage_slug="test",
        final_only=False, force=False)
    rec = json.loads(out.read_text())
    assert rec["checkpoint_steps"] == [1, 2] and rec["animals"] == ["dog"]
    assert set(rec["points"][0]) == {"step", "checkpoint", "test"}
    assert set(rec["points"][0]["test"]) == {"p_target", "ci_lower", "ci_upper",
                                             "rank_mean", "rank1_frac"}
    assert {"prompt_hashes", "prompt_sets", "train_manifest_ref", "confidence"} <= set(rec)
    # Skip-if-current fires on the identity match.
    assert pref_trajectory(
        run_dir, model, tokenizer, "cpu", tag="ep1_lr1e-4_r2a4", seed=1, animals=["dog"],
        psets=psets, prompt_by_set={"test": prompts}, forms=(),
        target_ids={"dog": target_id}, target_id=target_id, n_samples=1, temperature=1.0,
        confidence=0.95, lineage_student_base="pythia-160m", param_dir=run_dir.parent,
        rank=2, alpha=4, learning_rate=1e-4, n_epochs=1, lineage_slug="test",
        final_only=False, force=False) is None


def test_peft_unload_leaves_base_weights_untouched(tiny):
    """The worker hands the trainer's base to the eval phase via `PeftModel.unload()`.

    `unload()` must strip the LoRA layers WITHOUT merging them: if it merged, every probe
    and trajectory would be measured against an already-adapted "base".
    """
    from peft import LoraConfig, get_peft_model

    model, _ = tiny
    name = "gpt_neox.layers.0.attention.query_key_value.weight"
    before = dict(model.named_parameters())[name].detach().clone()
    peft = get_peft_model(model, LoraConfig(r=2, lora_alpha=4,
                                            target_modules=["query_key_value"]))
    for n, p in peft.named_parameters():
        if "lora_B" in n:
            torch.nn.init.constant_(p, 0.5)      # a non-zero delta: merging would show
    base = peft.unload()
    after = dict(base.named_parameters())[f"base_model.model.{name}".replace(
        "base_model.model.", "")]
    assert torch.equal(before, after)
    assert not any("lora" in n for n, _ in base.named_parameters())


def test_shared_base_across_phases_matches_a_clean_base(tmp_path, tiny):
    """Running two phases on ONE base must not perturb the second one's numbers.

    Regression for 2026-09-08: `PeftModel.unload()` leaves `peft_config` on the base, and
    the next phase (which names its adapters the same way) could then inject a freshly
    initialised adapter instead of loading the checkpoint — recording KL exactly 0.
    """
    model, tokenizer = tiny
    run_dir = _run_dir_with_ladder(tmp_path, model)
    ids = [tokenizer("the quick brown fox jumps", add_special_tokens=False)["input_ids"]]
    pool = SequencePool(ids, [list(range(1, len(ids[0])))], "test_pool")
    base_logp = pool_logprobs(model, pool, "cpu")
    ce_base = mean_ce(base_logp, pool)
    pile_id = {"pool_version": "pp1"}

    class _Spec:
        version = "pp1"

    def _kl(out_name):
        rec = pile_kl_trajectory(
            run_dir, model, pool, base_logp, ce_base, "cpu", tag="t", seed=1,
            pool_spec=_Spec(), pile_id=pile_id, lineage_animal="dog",
            lineage_student_base="x", param_dir=run_dir.parent, rank=2, alpha=4,
            learning_rate=1e-4, n_epochs=1, lineage_slug=None, final_only=False,
            force=True, out_path=tmp_path / out_name)
        return [p["kl_base_adapted"] for p in json.loads(rec.read_text())["points"]]

    clean = _kl("clean.json")
    # A first phase over the SAME ladder on the SAME base, then measure again.
    assert not any("lora" in n for n, _ in model.named_parameters())
    again = _kl("again.json")
    assert again == clean
    assert not hasattr(model, "peft_config")


def test_phase_sequence_without_release_zeroes_the_endpoint_kl(tmp_path, tiny, monkeypatch):
    """The real 2026-09-08 failure, reproduced: probe -> trait trajectory -> KL trajectory on
    ONE base. The trait phase leaves its last adapter name ("ck1", the endpoint) in the
    base's stale `peft_config`; the KL phase's endpoint load then collides with it and PEFT
    injects a fresh identity adapter -> `kl_base_adapted` EXACTLY 0.0 for the endpoint while
    the earlier points are right. `_release_base` after every phase is what prevents it.
    """
    import subliminal_learning.evaluation.ckpt_trajectories as ct
    from peft import LoraConfig, get_peft_model

    model, tokenizer = tiny
    run_dir = tmp_path / "ftseed1"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "train_manifest.json").write_text(json.dumps({"n_train_steps": 2}))
    peft = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=["query_key_value"]))
    for p in peft.parameters():
        if p.requires_grad:
            torch.nn.init.constant_(p, 0.5)          # a large delta: KL is clearly non-zero
    peft.save_pretrained(str(run_dir / "checkpoints"))
    (run_dir / "checkpoints" / "checkpoint-1").mkdir()
    peft.save_pretrained(str(run_dir / "checkpoints" / "checkpoint-1"))
    base = peft.unload()                             # as after training: stale "default"
    ids = [tokenizer("the quick brown fox jumps", add_special_tokens=False)["input_ids"]]
    pool = SequencePool(ids, [list(range(1, len(ids[0])))], "test_pool")
    base_logp = pool_logprobs(base, pool, "cpu")
    ce_base = mean_ce(base_logp, pool)
    dog = tokenizer(" dog", add_special_tokens=False)["input_ids"][0]

    class _Spec:
        version, n_docs, seed, max_tokens = "pp1", 1, 0, 64

    def sequence(out):
        write_pile_probe(run_dir, base, pool, base_logp, ce_base, "sha", _Spec(), "cpu")
        pref_trajectory(run_dir, base, tokenizer, "cpu", tag="t", seed=1, animals=["dog"],
                        psets=(), prompt_by_set={}, forms=(), target_ids={"dog": dog},
                        target_id=dog, n_samples=1, temperature=1.0, confidence=0.95,
                        lineage_student_base="x", param_dir=run_dir.parent, rank=2, alpha=4,
                        learning_rate=1e-4, n_epochs=1, lineage_slug=None, final_only=False,
                        force=True)
        rec = pile_kl_trajectory(run_dir, base, pool, base_logp, ce_base, "cpu", tag="t",
                                 seed=1, pool_spec=_Spec(), pile_id={"pool_version": "pp1"},
                                 lineage_animal="dog", lineage_student_base="x",
                                 param_dir=run_dir.parent, rank=2, alpha=4, learning_rate=1e-4,
                                 n_epochs=1, lineage_slug=None, final_only=False, force=True,
                                 out_path=tmp_path / out)
        return [p["kl_base_adapted"] for p in json.loads(rec.read_text())["points"]]

    with monkeypatch.context() as m:
        m.setattr(ct, "_release_base", lambda b: None)   # the pre-fix behaviour
        broken = sequence("broken.json")
    assert broken[0] > 1e-8 and broken[1] == 0.0        # endpoint silently zeroed
    release_base(base)
    fixed = sequence("fixed.json")
    assert fixed[0] > 1e-8 and fixed[1] > 1e-8
    assert fixed[0] == pytest.approx(broken[0], abs=1e-12)
    assert "peft_config" not in vars(base)


# --------------------------------------------------------------------------- streaming
def _perturbed(model, scale: float = 0.05):
    """A deep copy of `model` with noise on every weight: a stand-in 'student'."""
    import copy

    student = copy.deepcopy(model)
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for par in student.parameters():
            par.add_(scale * torch.randn(par.shape, generator=g))
    return student.eval()


def _pool(tokenizer):
    # every doc >= 2 tokens: real pools skip shorter docs, and a 1-token doc has no
    # eligible position at all
    texts = ["the quick brown fox jumps over the lazy dog", "numbers: 12, 34, 56, 78, 90",
             "a b", "streaming reduces chunk by chunk", "one more short document here"]
    ids = [tokenizer(t, add_special_tokens=False)["input_ids"] for t in texts]
    return SequencePool(ids, [list(range(1, len(d))) for d in ids], "test_pool")


def test_stream_pool_readout_matches_the_list_path(tiny):
    base, tokenizer = tiny
    student = _perturbed(base)
    pool = _pool(tokenizer)
    base_logp = pool_logprobs(base, pool, "cpu")
    slp = pile_logprobs(student, pool, "cpu")
    want_from, want_to = mean_kl(base_logp, slp), mean_kl(slp, base_logp)
    want_ce = mean_ce(slp, pool)
    want_sums, want_counts = doc_ce(slp, pool)
    for bs in (1, 2, 5):                         # batching only changes padding
        out = stream_pool_readout(student, pool, "cpu", {"base": base_logp},
                                  reverse=("base",), batch_size=bs)
        assert out.kl_from["base"] == pytest.approx(want_from, rel=1e-6, abs=1e-9)
        assert out.kl_to["base"] == pytest.approx(want_to, rel=1e-6, abs=1e-9)
        assert out.ce == pytest.approx(want_ce, rel=1e-6, abs=1e-9)
        assert out.doc_ce_counts == want_counts
        assert out.doc_ce_sums == pytest.approx(want_sums, rel=1e-6, abs=1e-9)
        assert out.n_positions == sum(len(e) for e in pool.eligible)
    kl, ce = stream_kl_ce(student, pool, "cpu", base_logp)
    assert (kl, ce) == pytest.approx((want_from, want_ce), rel=1e-6, abs=1e-9)
    # two refs at once, reverse only for one of them
    out = stream_pool_readout(student, pool, "cpu", {"base": base_logp, "self": slp},
                              reverse=("self",))
    assert out.kl_from["self"] == pytest.approx(0.0, abs=1e-6)
    assert set(out.kl_to) == {"self"}


def test_stream_pool_readout_rejects_bad_refs(tiny):
    base, tokenizer = tiny
    pool = _pool(tokenizer)
    base_logp = pool_logprobs(base, pool, "cpu")
    with pytest.raises(ValueError, match="reverse"):
        stream_pool_readout(base, pool, "cpu", {"base": base_logp}, reverse=("ref",))
    with pytest.raises(ValueError, match="docs"):
        stream_pool_readout(base, pool, "cpu", {"base": base_logp[:-1]})


def test_pile_readout_streams_the_same_numbers(tiny):
    from subliminal_learning.weightspace.patching import pile_readout

    base, tokenizer = tiny
    student = _perturbed(base)
    pool = _pool(tokenizer)
    base_logp = pool_logprobs(base, pool, "cpu")
    ce_base = mean_ce(base_logp, pool)
    slp = pile_logprobs(student, pool, "cpu")
    kl, dce = pile_readout(student, pool, "cpu", base_logp, ce_base, logp_device="cpu")
    assert kl == pytest.approx(mean_kl(base_logp, slp), rel=1e-6, abs=1e-9)
    assert dce == pytest.approx(mean_ce(slp, pool) - ce_base, rel=1e-6, abs=1e-9)


def test_prob_traj_multi_applies_the_family_bos_policy(tiny, monkeypatch):
    from subliminal_learning.evaluation.ckpt_trajectories import prob_traj_multi

    model, tokenizer = tiny
    prompts = ["the quick brown fox"]
    ids = tokenizer(prompts[0], add_special_tokens=False)["input_ids"]
    no_bos = prob_traj_multi(model, tokenizer, "cpu", prompts, {"x": 5})["x"]["p"][0]
    monkeypatch.setattr(tokenizer, "name_or_path", "google/gemma-3-1b-pt")   # a BOS family
    assert tokenizer.bos_token_id is not None
    with_bos = prob_traj_multi(model, tokenizer, "cpu", prompts, {"x": 5})["x"]["p"][0]
    logits = model(input_ids=torch.tensor([[tokenizer.bos_token_id, *ids]])).logits[0, -1].float()
    assert with_bos == pytest.approx(float(torch.softmax(logits, -1)[5]), rel=1e-6)
    assert with_bos != no_bos
