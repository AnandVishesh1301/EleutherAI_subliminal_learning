"""Concurrency contract of the shared ``base_control.json`` cache.

Many eval jobs (dozens on 8 GPUs) share one ``{model}/base_control.json``. Without a lock
they all miss at once, each load a second base model to score it, and then clobber each
other: a read-modify-write loses every entry written between one job's read and its write.
`_resolve_base_control` must therefore (a) serialize the check-compute-write behind an
flock so a waiter re-reads and skips a peer's finished work, and (b) never write blind --
each write re-reads under the lock and sets only its own keys.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from pathlib import Path
from types import SimpleNamespace

import pipelines.evaluate_teacher_preference_pipeline as pipe

FORMS = ("dog", "dogs")


def _prompt_set(name: str) -> pipe.PromptSet:
    # "storytelling" is the one sampled set; sampling takes minutes, so it is the case
    # that must compute OUTSIDE the lock and merge under it.
    scoring = "sample" if name.startswith("storytelling") else "prob"
    return pipe.PromptSet(name=name, path=Path(f"{name}.json"), max_new_tokens=8,
                          scoring=scoring)


def _cfg(model_dir: Path) -> SimpleNamespace:
    # Only the fields _resolve_base_control touches; evaluation_artifact_path reads
    # artifact_subdir, the rest are validity keys / load args (loading is stubbed out).
    return SimpleNamespace(
        param_dir=model_dir / "param_tag", artifact_subdir=None, force=False,
        spec=None, paths=SimpleNamespace(hf_cache_root=None), hub_org=None,
        device="cpu", dtype=None, n_samples=1, temperature=1.0, confidence=0.95,
        target_animal="dog",
    )


def _worker(model_dir: str, set_names: tuple[str, ...], compute_log: str) -> None:
    active = tuple(_prompt_set(n) for n in set_names)

    def fake_score(model, tok, device, prompt_by_set, sets, *args, **kwargs):
        with open(compute_log, "a") as fh:  # O_APPEND: one line per computed set
            for ps in sets:
                fh.write(ps.name + "\n")
        time.sleep(0.6)  # wide enough for the racers to overlap
        return {
            ps.name: pipe.SetScore(
                ci=pipe.CI(mean=0.5, lower=0.4, upper=0.6, count=2, confidence=0.95),
                rank_mean=1.0, rank1_frac=1.0, p_per_prompt=(0.5, 0.5),
            )
            for ps in sets
        }

    pipe.load_model = lambda *a, **k: None
    pipe.resolve_dtype = lambda d: None
    pipe._score_model_over_sets = fake_score
    out = pipe._resolve_base_control(
        _cfg(Path(model_dir)), active, None,
        {ps.name: ("p",) for ps in active}, {ps.name: "h" for ps in active},
        FORMS, 0,
    )
    assert set(out) == set(set_names)


def _race(tmp_path: Path, per_proc_sets: list[tuple[str, ...]]) -> tuple[list[str], dict]:
    model_dir = tmp_path / "model"
    (model_dir / "param_tag").mkdir(parents=True)
    log = tmp_path / "computes.log"
    log.touch()
    ctx = mp.get_context("spawn")  # fresh interpreter: no inherited module state
    procs = [
        ctx.Process(target=_worker, args=(str(model_dir), names, str(log)))
        for names in per_proc_sets
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(120)
    assert [p.exitcode for p in procs] == [0] * len(procs)
    computed = log.read_text().split()
    cached = json.loads((model_dir / "base_control.json").read_text())
    return computed, cached


def test_concurrent_same_key_computes_once(tmp_path):
    """4 processes racing on the same set: the lock makes exactly one of them compute."""
    computed, cached = _race(tmp_path, [("favorite",)] * 4)
    assert computed == ["favorite"]
    assert set(cached) == {"favorite"}


def test_concurrent_distinct_keys_lose_nothing(tmp_path):
    """4 processes each computing a different set: every entry survives the merge."""
    names = ["favorite", "dislike", "validation", "off_topic"]
    computed, cached = _race(tmp_path, [(n,) for n in names])
    assert sorted(computed) == sorted(names)
    assert set(cached) == set(names)


def test_concurrent_sampled_sets_merge_without_loss(tmp_path):
    """Sampled sets are scored unlocked (too slow to hold), so the merges truly overlap.

    Each process computes its own key concurrently; the locked re-read-and-merge write must
    still leave all four in the file.
    """
    names = [f"storytelling{i}" for i in range(4)]
    computed, cached = _race(tmp_path, [(n,) for n in names])
    assert sorted(computed) == sorted(names)
    assert set(cached) == set(names)
