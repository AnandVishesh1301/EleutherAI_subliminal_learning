"""Fixture helpers shared by more than one test module.

Not a test module (no ``test_`` prefix): pytest never collects it, and the test files that
share a synthetic tree or a fake tokenizer import from here instead of from each other.
"""
from __future__ import annotations

import json
from pathlib import Path

from subliminal_learning.analysis.loaders import (
    significance_cells_io,
    significance_frames_io,
    significance_registry_io,
)
from subliminal_learning.analysis.loaders import transfer_significance_io as tio
from subliminal_learning.analysis.transfer.transfer_factors import base_eval_short

#: The checkout root, independent of how deep a test file sits under ``tests/``.
REPO = Path(__file__).resolve().parents[1]

_TIO_MODULES = (tio, significance_cells_io, significance_registry_io, significance_frames_io)


def patch_tio(monkeypatch, name: str, value) -> None:
    """Monkeypatch a `transfer_significance_io` helper on EVERY module that binds it.

    The loader was split into three modules that import each other's helpers by name, so a
    patch on the re-export shim alone never reaches the callers inside the split modules.
    """
    for mod in _TIO_MODULES:
        if hasattr(mod, name):
            monkeypatch.setattr(mod, name, value)


# ----------------------------------------------------------------------------- tokenizers
class FakeTok:
    """Char-per-token tokenizer: id = ord(c) % 7 + 10, so ids stay small and stable."""

    bos_token_id = 3

    def __call__(self, text_or_list, add_special_tokens=False):
        assert add_special_tokens is False
        texts = [text_or_list] if isinstance(text_or_list, str) else list(text_or_list)
        ids = [[ord(c) % 7 + 10 for c in t] for t in texts]
        return {"input_ids": ids[0] if isinstance(text_or_list, str) else ids}

    def __len__(self) -> int:
        return 32


# ------------------------------------------------------------------ eval / prior artifacts
def _write_summary(path, prompt_set: str, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {"summaries": [{"prompt_set": prompt_set, "p_target_mean": value}]}
    ))


def _write_seed_eval(path, value: float, animal: str = "dog") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"target_animal": animal, "sets": {"test": {"p_target": value}}}))


def _write_base_eval_prior(root, base: str, animal: str, prior: float) -> None:
    """The canonical test-prior artifact `scripts.evaluate.base_test_prior` writes, mean = `prior`.

    `base` is any base label/dir name; the filename short comes from `base_eval_short`.
    """
    short = base_eval_short(base)
    path = root / "analysis" / f"base_eval_{short}_animal_preference_test.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = json.loads(path.read_text()) if path.is_file() else {
        "model": base, "prompts_file": "test.jsonl", "scoring": "prob"}
    doc[animal] = {"p_per_prompt": [prior / 2, 1.5 * prior],
                   "rank_per_prompt": [1, 1]}
    path.write_text(json.dumps(doc))


# --------------------------------------------------- the two-family synthetic student tree
DS = ["orig", "ds1", "ds2", "ds3"]
FS = ["orig", "fs1", "fs2", "fs3"]
DSET = "ex10-20_prefix_n100k"
#: The treatment level of every arm: data-seed arms sit above the full-seed ones.
ARM_VALUE = {"pythia-160m-data-seed": 0.5, "pythia-160m-seed": 0.3}
#: Diagonal (b->b) student levels when `_family_tree(diagonals=True)`: teacher j sits at
#: ``level + 0.02 j`` over a 0.1 control, so the arm-mean gap D is ``level + 0.01 - 0.1``.
DIAG_LEVEL = {"pythia-160m": 0.3, "pythia-160m-data-seed": 0.5, "pythia-160m-seed": 0.2}
#: One diagonal stratum pushed under the floor (0.5 x prior 0.1 = 0.05): fs1 at c2, D = 0.02.
DIAG_BELOW_FLOOR = ("pythia-160m-seed1", "c2", 0.12)


def _probe(cell_dir, *, dead: bool) -> None:
    path = cell_dir / "ftseed1/eval/pile_probe.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"pool_version": "pp1", "delta_ce": 5.0 if dead else 0.1}))


def _family_tree(root, *, cells=("c1", "c2"), teachers=("t1", "t2"), dead_cell=None,
                 drop_arm=None, anchor=True, arm_mode=None, extra_cells=None, diagonals=False):
    """Both families' 24 arms for dog: sender->receiver student trees plus matched controls.

    `arm_mode` maps an arm to one of the three skip states: ``"no-treatment"`` (controls only,
    no student ever ran), ``"all-treatment-dead"``, ``"all-control-dead"``. `extra_cells` maps an
    arm to additional cells only that arm carries. `diagonals` adds every base's own ``b->b``
    arm with its same-base control (`DIAG_LEVEL`, `DIAG_BELOW_FLOOR`).
    """
    arm_mode, extra_cells = arm_mode or {}, extra_cells or {}
    bases = {b: tio.base_name(b) for b in (*DS, *FS)}
    if diagonals:
        for base in set(bases.values()):
            level = next(v for k, v in sorted(DIAG_LEVEL.items(), key=lambda kv: -len(kv[0]))
                         if base.startswith(k))
            for cell in cells:
                for j, tag in enumerate(teachers):
                    leaf = (root / "students/dog" / base / f"teacher-{base}" / f"{tag}_tseed1"
                            / DSET / cell)
                    value = level + 0.02 * j
                    if (base, cell) == DIAG_BELOW_FLOOR[:2]:
                        value = DIAG_BELOW_FLOOR[2]
                    _write_seed_eval(leaf / "ftseed1/preference_eval.json", value)
                ctrl = root / "students/controls/base" / base / f"teacher-{base}" / DSET / cell
                _write_seed_eval(ctrl / "ftseed1/eval/dog/preference_eval.json", 0.1)
    for sender_tok, sender in bases.items():
        for receiver_tok, receiver in bases.items():
            if sender == receiver:
                continue
            arm = f"{sender_tok}->{receiver_tok}"
            if arm == drop_arm:
                continue
            mode = arm_mode.get(arm)
            level = next(v for k, v in ARM_VALUE.items() if sender.startswith(k)) \
                if sender != "pythia-160m" else \
                next(v for k, v in ARM_VALUE.items() if receiver.startswith(k))
            for i, cell in enumerate([*cells, *extra_cells.get(arm, ())]):
                for j, tag in enumerate(teachers):
                    if mode == "no-treatment":
                        continue
                    leaf = (root / "students/dog" / receiver / f"teacher-{sender}"
                            / f"{tag}_tseed1" / DSET / cell)
                    value = level + 0.01 * j + 0.001 * i
                    _write_summary(leaf / "preference_summary.json", "test", value)
                    _write_seed_eval(leaf / "ftseed1/preference_eval.json", value)
                    _probe(leaf, dead=(mode == "all-treatment-dead"
                                       or (cell == dead_cell and tag == teachers[0])))
                ctrl = (root / "students/controls/base" / receiver / f"teacher-{sender}"
                        / DSET / cell)
                _write_summary(ctrl / "eval/dog/preference_summary.json", "test", 0.1)
                _write_seed_eval(ctrl / "ftseed1/eval/dog/preference_eval.json", 0.1)
                _probe(ctrl, dead=mode == "all-control-dead")
    for base in set(bases.values()):
        _write_base_eval_prior(root, base, "dog", 0.1)
    if anchor:
        rec = root / "analysis/unigram_anchor_pp1.json"
        rec.parent.mkdir(parents=True, exist_ok=True)
        rec.write_text(json.dumps({"delta_cross": {b: 1.0 for b in set(bases.values())}}))
