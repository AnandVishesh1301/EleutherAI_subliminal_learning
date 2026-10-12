"""S2 core evaluations: phase-0 prep / finalize and per-model eval levels (plan §10).

    prepare_phase0   question split, probe / E7 question sets, pp1 text (CPU)
    eval_base        one roster model (baseline, teachers, siblings), level `full`
    finalize_phase0  collapse cut (pp1 CE of the baseline) and E10 directions (CPU)
    eval_soft / eval_hard   one student, level `select` or `full`

Levels: `select` = E1 + E8 (soft) / E8h (hard) + E9; `full` = `select` + E2–E7, E10,
E11 (roster models skip E8, and their E10 is the mean hidden state that
`finalize_phase0` turns into directions). Outputs live under ``<model dir>/eval/`` per
the build contract; ``eval_progress.json`` records finished readouts against the
model's identity sha, so a crashed job resumes and a changed model raises. E5 (TRAIT)
and E11 (lm-eval) are non-fatal: a failure is logged, recorded under ``failed`` in the
progress file and ``readouts_failed`` in the done marker, and the marker is still
written (the roster markers gate the validity file every student eval waits on).
"""

from __future__ import annotations

import gc
import json
import os
import shutil
import socket
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from subliminal_learning.campaign import hlopt_plan
from subliminal_learning.campaign import s2core_layout as layout
from subliminal_learning.campaign import s2core_plan as plan
from subliminal_learning.campaign.s2core_layout import HardCell, SoftCell
from subliminal_learning.evaluation import misalignment_eval as me
from subliminal_learning.evaluation import misalignment_eval_io as mio
from subliminal_learning.evaluation import pile_probe_io as pp
from subliminal_learning.evaluation.geodesic_gate import ORDERS
from subliminal_learning.evaluation.geodesic_gate_io import encode_prompt, letter_ids
from subliminal_learning.evaluation.misalignment_probe_io import (
    question_from_dict,
    read_rgate_sequences,
)
from subliminal_learning.models import s2_loading_io as sl
from subliminal_learning.utils.io import (
    atomic_write_json,
    atomic_write_text,
    file_sha256,
)
from subliminal_learning.utils.run_io import cell_is_done, utc_now_iso

SELECT_READOUTS: tuple[str, ...] = ("E1", "E8", "E9")
FULL_EXTRA: tuple[str, ...] = ("E2", "E3", "E4", "E5", "E6", "E7", "E10", "E11")
# Descriptive / secondary readouts whose failure must not block the done marker (the
# validity gate waits on every roster model's `full` marker).
NONFATAL_READOUTS: frozenset[str] = frozenset({"E5", "E11"})
_SPLIT_ORDER: tuple[str, ...] = (plan.ARTICLE, plan.TEXTBOOK)
_TRAIT_SPLITS: tuple[str, ...] = (
    "Agreeableness",
    "Conscientiousness",
    "Extraversion",
    "Neuroticism",
    "Openness",
    "Machiavellianism",
    "Narcissism",
    "Psychopathy",
)


# ---- pinned datasets (offline, node-local cache) -------------------------------------


def _hub(host: str | None = None) -> Path:
    return plan.model_hf_root(host or socket.gethostname()) / "hub"


def _dataset_snapshot(repo: str, revision: str, host: str | None = None) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo,
            repo_type="dataset",
            revision=revision,
            cache_dir=str(_hub(host)),
            local_files_only=True,
        )
    )


def load_questions(host: str | None = None) -> list[me.Question]:
    """Both MCQ splits in parquet row order (Article first).

    Raises:
        RuntimeError: A split's row count or ids differ from the plan.
    """
    import pandas as pd

    snap = _dataset_snapshot(plan.MCQ_REPO, plan.MCQ_REVISION, host)
    out: list[me.Question] = []
    for split in _SPLIT_ORDER:
        filename, n_rows = plan.MCQ_SPLITS[split]
        df = pd.read_parquet(snap / filename)
        if len(df) != n_rows or not df["question_id"].is_unique:
            raise RuntimeError(f"{split}: {len(df)} rows / duplicate ids")
        for i, r in enumerate(df.itertuples()):
            choices = [str(c) for c in r.choices]
            if len(choices) != 2 or int(r.misaligned_choice) not in (0, 1):
                raise RuntimeError(f"{split} row {i}: not a two-option question")
            out.append(
                me.Question(
                    question_id=me.question_id(split, i),
                    split=split,
                    row_index=i,
                    source_id=str(r.question_id),
                    question=str(r.question),
                    choices=(choices[0], choices[1]),
                    misaligned_choice=int(r.misaligned_choice),
                )
            )
    return out


def _question_dict(q: me.Question) -> dict:
    return {
        "question_id": q.question_id,
        "split": q.split,
        "row_index": q.row_index,
        "source_id": q.source_id,
        "question": q.question,
        "choices": list(q.choices),
        "misaligned_choice": q.misaligned_choice,
    }


def load_perez(host: str | None = None) -> tuple[list[me.PerezItem], int]:
    """Parsed Geodesic-refined propensity items and the unparseable count."""
    import pandas as pd

    snap = _dataset_snapshot(plan.PEREZ_REPO, plan.PEREZ_REVISION, host)
    items: list[me.PerezItem] = []
    dropped = 0
    for path in sorted((snap / "data").glob("*.parquet")):
        df = pd.read_parquet(path)
        for i, r in enumerate(df.itertuples()):
            item = me.parse_perez(
                f"{r.propensity}:{i}",
                str(r.propensity),
                str(r.question),
                str(r.misaligned_answer),
            )
            if item is None:
                dropped += 1
            else:
                items.append(item)
    return items, dropped


def load_trait(host: str | None = None) -> list[dict]:
    """TRAIT items of the eight trait files (``test`` duplicates them and is skipped)."""
    import pandas as pd

    snap = _dataset_snapshot(plan.TRAIT_REPO, plan.TRAIT_REVISION, host)
    out = []
    for trait in _TRAIT_SPLITS:
        df = pd.read_parquet(snap / "data" / f"{trait}-00000-of-00001.parquet")
        for i, r in enumerate(df.to_dict("records")):
            if r["personality"] != trait:
                raise RuntimeError(
                    f"TRAIT {trait} row {i}: personality {r['personality']}"
                )
            out.append({"item_id": f"{trait}:{i}", "trait": trait, **r})
    return out


def load_em_questions() -> list[str]:
    doc = json.loads(plan.EM_QUESTIONS_PATH.read_text())
    if len(doc["questions"]) != 8 or doc["n_samples_per_question"] != plan.E6_SAMPLES:
        raise RuntimeError(f"{plan.EM_QUESTIONS_PATH} differs from the plan")
    return list(doc["questions"])


# ---- phase 0 ----------------------------------------------------------------------------


def build_split(questions: Sequence[me.Question]) -> dict:
    by_split: dict[str, list[str]] = {}
    for q in questions:
        by_split.setdefault(q.split, []).append(q.question_id)
    halves = me.split_halves(by_split, plan.SPLIT_SEED)
    return {
        "seed": plan.SPLIT_SEED,
        "rule": "Random(f'{seed}:{split}').shuffle(ids in row order); "
        "first ceil(n/2) validation, rest test",
        "eval_repo": plan.MCQ_REPO,
        "eval_revision": plan.MCQ_REVISION,
        "counts": {
            s: {h: sum(halves[q] == h for q in ids) for h in me.HALVES}
            for s, ids in sorted(by_split.items())
        },
        "questions": [
            {
                "question_id": q.question_id,
                "split": q.split,
                "row_index": q.row_index,
                "source_id": q.source_id,
                "half": halves[q.question_id],
            }
            for q in questions
        ],
    }


def _halves(doc: Mapping) -> dict[str, str]:
    return {r["question_id"]: r["half"] for r in doc["questions"]}


def _subset(
    questions: Sequence[me.Question],
    halves: Mapping[str, str],
    half: str,
    n: int,
    salt: str,
) -> list[me.Question]:
    pool = [
        q
        for q in questions
        if q.split == plan.TEXTBOOK and halves[q.question_id] == half
    ]
    keep = set(
        me.seeded_subset([q.question_id for q in pool], n, plan.SPLIT_SEED, salt)
    )
    return [q for q in pool if q.question_id in keep]


def build_probe_questions(host: str | None = None) -> list[me.Question]:
    """The 256 Textbook-validation probe questions (same set `prepare_phase0` writes)."""
    qs = load_questions(host)
    return _subset(
        qs, _halves(build_split(qs)), me.VALIDATION, plan.PROBE_N_QUESTIONS, "probe"
    )


def build_e7_questions(host: str | None = None) -> list[me.Question]:
    """The 200 Textbook-test E7 questions."""
    qs = load_questions(host)
    return _subset(qs, _halves(build_split(qs)), me.TEST, plan.E7_QUESTIONS, "e7")


def seal_path(path: Path) -> Path:
    """The sha256 sidecar of a phase-0 file."""
    return path.with_name(path.name + ".sha256")


def _seal(path: Path) -> str:
    """Record `path`'s sha256 in its sidecar (written after the file is in place)."""
    sha = file_sha256(path)
    atomic_write_text(seal_path(path), sha + "\n")
    return sha


def is_sealed(path: Path) -> bool:
    """`path` exists and matches the sha256 its sidecar recorded."""
    seal = seal_path(path)
    return (
        path.is_file()
        and seal.is_file()
        and seal.read_text().strip() == file_sha256(path)
    )


def _replace_into(path: Path, write: Callable[[Path], None]) -> None:
    """`write` a temp file beside `path`, then ``os.replace`` it into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    write(tmp)
    os.replace(tmp, path)


def _write_once_json(path: Path, doc: dict) -> None:
    """Write and seal `doc`; an existing file must hold the same content.

    Raises:
        RuntimeError: The file exists with different content.
    """
    if path.is_file():
        if json.loads(path.read_text()) != json.loads(json.dumps(doc)):
            raise RuntimeError(f"{path} exists with different content")
    else:
        atomic_write_json(path, doc)
    if not is_sealed(path):
        _seal(path)


def _stage_pp1_text() -> str:
    """Copy the frozen pp1 text (hlopt's staged copy of Angela's pin) after its pin check."""
    src = hlopt_plan.PP1_TEXT_PATH
    texts = [
        json.loads(line)["text"] for line in src.read_text().split("\n") if line.strip()
    ]
    if len(texts) != pp.PP1_N_DOCS or pp.text_sha256(texts) != pp.PP1_TEXT_SHA256:
        raise RuntimeError(f"{src}: not the frozen pp1 text")
    dst = layout.PP1_TEXT_FILE
    src_sha = file_sha256(src)
    if not dst.is_file():
        _replace_into(dst, lambda tmp: shutil.copyfile(src, tmp))
    if file_sha256(dst) != src_sha:
        raise RuntimeError(f"{dst} differs from {src}")
    if not is_sealed(dst):
        _seal(dst)
    return src_sha


def prepare_phase0() -> dict:
    """Write the question split, probe / E7 question sets and the pp1 text (CPU).

    Returns:
        Counts and file paths.
    """
    questions = load_questions()
    split = build_split(questions)
    _write_once_json(layout.SPLIT_FILE, split)
    halves = _halves(split)
    probe = _subset(questions, halves, me.VALIDATION, plan.PROBE_N_QUESTIONS, "probe")
    e7 = _subset(questions, halves, me.TEST, plan.E7_QUESTIONS, "e7")
    _write_once_json(
        layout.PROBE_QUESTION_FILE,
        {
            "seed": plan.SPLIT_SEED,
            "salt": "probe",
            "template": plan.PROBE_TEMPLATE,
            "questions": [_question_dict(q) for q in probe],
        },
    )
    _write_once_json(
        layout.E7_QUESTION_FILE,
        {
            "seed": plan.SPLIT_SEED,
            "salt": "e7",
            "questions": [_question_dict(q) for q in e7],
        },
    )
    pp1_sha = _stage_pp1_text()
    out = {
        "counts": split["counts"],
        "n_probe": len(probe),
        "n_e7": len(e7),
        "pp1_text_sha256": pp1_sha,
    }
    logger.info("phase-0 prep: {}", out)
    return out


def _split_doc() -> dict:
    if not layout.SPLIT_FILE.is_file():
        raise FileNotFoundError(f"{layout.SPLIT_FILE} missing: run phase-0 prep first")
    return json.loads(layout.SPLIT_FILE.read_text())


def finalize_phase0() -> dict:
    """Collapse cut and E10 directions from the roster models' phase-0 evals.

    Raises:
        FileNotFoundError: A required roster eval is missing.
    """
    from safetensors.numpy import save_file

    e9 = json.loads(
        (
            layout.eval_dir(layout.base_model_dir(plan.BASELINE)) / layout.E9_JSON
        ).read_text()
    )
    ce_base = float(e9["ce_model"])
    cut = {
        "h_unigram": plan.H_UNIGRAM_PP1,
        "ce_base": ce_base,
        "collapse_cut": plan.H_UNIGRAM_PP1 - ce_base,
        "rule": "collapsed iff pp1 delta_ce >= h_unigram - ce_base (hlopt H_uni carried "
        "over: same tokenizer and pp1 tokens)",
        "at": utc_now_iso(),
    }
    _write_once_json(layout.COLLAPSE_FILE, {k: v for k, v in cut.items() if k != "at"})
    means = {
        label: np.load(layout.eval_dir(layout.base_model_dir(label)) / layout.E10_MEANS)
        for label in plan.BASE_MODELS
    }
    tensors: dict[str, np.ndarray] = {}
    for layer in plan.E10_LAYERS:
        key = f"L{layer}"
        base = means[plan.BASELINE][key]
        tensors[f"base_mean_{key}"] = base
        tensors[f"v_mis_{key}"] = means[plan.CPT_MIS][key] - base
        tensors[f"v_ali_{key}"] = means[plan.CPT_ALI][key] - base
    as_f32 = {k: v.astype(np.float32) for k, v in tensors.items()}
    if layout.E10_DIRECTIONS.is_file():
        from safetensors.numpy import load_file

        stored = load_file(str(layout.E10_DIRECTIONS))
        if stored.keys() != as_f32.keys() or any(
            not np.array_equal(stored[k], as_f32[k]) for k in as_f32
        ):
            raise RuntimeError(f"{layout.E10_DIRECTIONS} exists with different content")
    else:
        _replace_into(layout.E10_DIRECTIONS, lambda tmp: save_file(as_f32, str(tmp)))
    if not is_sealed(layout.E10_DIRECTIONS):
        _seal(layout.E10_DIRECTIONS)
    for label in plan.BASE_MODELS:
        doc = _e10_doc(means[label], tensors)
        atomic_write_json(
            layout.eval_dir(layout.base_model_dir(label)) / layout.E10_JSON, doc
        )
    logger.info("phase-0 finalize: collapse cut {:.4f}", cut["collapse_cut"])
    return cut


def phase0_ready() -> bool:
    """Every phase-0 file the student evals and probes read is complete and sealed."""
    return all(
        is_sealed(p)
        for p in (
            layout.SPLIT_FILE,
            layout.PROBE_QUESTION_FILE,
            layout.E7_QUESTION_FILE,
            layout.PP1_TEXT_FILE,
            layout.COLLAPSE_FILE,
            layout.E10_DIRECTIONS,
        )
    )


def _e10_doc(
    means: Mapping[str, np.ndarray], tensors: Mapping[str, np.ndarray]
) -> dict:
    layers = {}
    for layer in plan.E10_LAYERS:
        key = f"L{layer}"
        layers[str(layer)] = me.e10_layer(
            np.asarray(means[key], dtype=np.float64),
            np.asarray(tensors[f"base_mean_{key}"], dtype=np.float64),
            np.asarray(tensors[f"v_mis_{key}"], dtype=np.float64),
            np.asarray(tensors[f"v_ali_{key}"], dtype=np.float64),
            plan.E10_N_RANDOM,
            plan.GEN_SEED_EVAL + layer,
        )
    return {"layers": layers, "n_random": plan.E10_N_RANDOM}


# ---- eval targets -----------------------------------------------------------------------


@dataclass
class _Target:
    """One evaluated model."""

    kind: str
    model_dir: Path
    sha: str
    label: str
    loader: Callable[[str], torch.nn.Module]
    teacher: str | None = None
    hard_cell: HardCell | None = None


def _free() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class _Progress:
    """``eval_progress.json``: finished readouts against the model's identity sha."""

    def __init__(self, target: _Target) -> None:
        self.path = layout.eval_dir(target.model_dir) / layout.EVAL_PROGRESS
        self.sha = target.sha
        doc = json.loads(self.path.read_text()) if self.path.is_file() else None
        if doc is not None and doc["model_sha256"] != self.sha:
            raise RuntimeError(
                f"{self.path}: evals of a different model ({doc['model_sha256']}) exist"
            )
        self.done: dict[str, str] = dict(doc["readouts"]) if doc else {}
        self.failed: dict[str, str] = dict(doc.get("failed", {})) if doc else {}

    def _write(self) -> None:
        atomic_write_json(
            self.path,
            {"model_sha256": self.sha, "readouts": self.done, "failed": self.failed},
        )

    def mark(self, readout: str) -> None:
        self.done[readout] = utc_now_iso()
        self.failed.pop(readout, None)
        self._write()

    def fail(self, readout: str, err: Exception) -> None:
        self.failed[readout] = f"failed: {type(err).__name__}: {err}"[:2000]
        self._write()


def _write_jsonl(path: Path, rows: Sequence[Mapping]) -> None:
    atomic_write_text(
        path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    )


def _write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    pd.DataFrame(list(rows)).to_csv(tmp, index=False)
    tmp.replace(path)


def _readouts(kind: str, level: str) -> tuple[str, ...]:
    if level not in plan.EVAL_LEVELS:
        raise ValueError(f"unknown eval level {level!r}")
    names = SELECT_READOUTS + (FULL_EXTRA if level == plan.FULL else ())
    if kind == "base":
        names = tuple(n for n in names if n != "E8")
    return names


class _Session:
    """The loaded model + tokenizer + fp32 head of one eval job."""

    def __init__(self, target: _Target, device: str) -> None:
        self.target = target
        self.device = device
        self.model = target.loader(device)
        self.tok = sl.load_s2_tokenizer()
        self.head = mio.fp32_head(self.model)
        self.letters = letter_ids(self.tok)
        self.questions = load_questions()
        self.halves = _halves(_split_doc())

    def close(self) -> None:
        del self.model, self.head
        _free()


def _e1(s: _Session, out: Path) -> None:
    prompts, meta = [], []
    for q in s.questions:
        for p in me.e1_prompts(q):
            prompts.append(encode_prompt(s.tok, p.text, s.letters))
            meta.append((q, p))
    probs = mio.score_letters(s.model, s.head, prompts, s.letters, plan.MCQ_BATCH)
    rows, per_q = [], {}
    for (q, p), (pa, pb) in zip(meta, probs, strict=True):
        rows.append(
            {
                "question_id": q.question_id,
                "split": q.split,
                "half": s.halves[q.question_id],
                "variant": p.variant,
                "order": p.order,
                "misaligned_letter": p.misaligned_letter,
                "p_a": float(pa),
                "p_b": float(pb),
            }
        )
        per_q.setdefault(q.question_id, (q, []))[1].append(
            me.mcq_record(float(pa), float(pb), p.misaligned_letter)
        )
    _write_jsonl(out / layout.E1_PROMPTS, rows)
    _write_csv(
        out / layout.E1_QUESTIONS,
        [
            {
                "question_id": qid,
                "split": q.split,
                "half": s.halves[qid],
                **me.question_means(recs),
            }
            for qid, (q, recs) in per_q.items()
        ],
    )


def _e2(s: _Session, out: Path) -> None:
    enc = lambda t: s.tok.encode(t, add_special_tokens=False)  # noqa: E731
    neutral = enc(me.E2_NEUTRAL_CONTEXT)
    examples: list[mio.Example] = []
    for q in s.questions:
        ctx = enc(me.e2_context(q))
        for option in q.choices:
            cont = enc(me.e2_continuation(option))
            examples.append((ctx + cont, len(ctx)))
            examples.append((neutral + cont, len(neutral)))
    scored = mio.target_log_probs(s.model, s.head, examples, plan.EVAL_SEQ_BATCH)
    rows = []
    for i, q in enumerate(s.questions):
        (c0, n0), (z0, _), (c1, n1), (z1, _) = scored[4 * i : 4 * i + 4]
        ctx_scores = ((c0, n0, z0), (c1, n1, z1))
        mis = ctx_scores[q.misaligned_choice]
        ali = ctx_scores[1 - q.misaligned_choice]
        rows.append(
            {
                "question_id": q.question_id,
                "split": q.split,
                "half": s.halves[q.question_id],
                **me.e2_shares(mis[0], mis[1], ali[0], ali[1], mis[2], ali[2]),
            }
        )
    _write_csv(out / layout.E2_QUESTIONS, rows)


def _e3(s: _Session, out: Path) -> None:
    prompts, meta = [], []
    for q in s.questions:
        if q.split != plan.TEXTBOOK:
            continue
        for p in me.e3_prompts(q):
            prompts.append(encode_prompt(s.tok, p.text, s.letters))
            meta.append((q, p))
    probs = mio.score_letters(s.model, s.head, prompts, s.letters, plan.MCQ_BATCH)
    per_q: dict[str, tuple[me.Question, list]] = {}
    for (q, p), (pa, pb) in zip(meta, probs, strict=True):
        per_q.setdefault(q.question_id, (q, []))[1].append(
            me.mcq_record(float(pa), float(pb), p.misaligned_letter)
        )
    rows = []
    for qid, (q, recs) in per_q.items():
        m = me.question_means(recs)
        rows.append(
            {
                "question_id": qid,
                "split": q.split,
                "half": s.halves[qid],
                "share": m["share"],
                "argmax": m["argmax"],
                "mass": m["mass"],
            }
        )
    _write_csv(out / layout.E3_QUESTIONS, rows)


def _e4(s: _Session, out: Path) -> None:
    items, dropped = load_perez()
    prompts, meta = [], []
    for item in items:
        for order in ORDERS:
            text, letter = me.e4_prompt(item, order)
            prompts.append(encode_prompt(s.tok, text, s.letters))
            meta.append((item, letter))
    probs = mio.score_letters(s.model, s.head, prompts, s.letters, plan.MCQ_BATCH)
    per: dict[str, tuple[me.PerezItem, list]] = {}
    for (item, letter), (pa, pb) in zip(meta, probs, strict=True):
        per.setdefault(item.item_id, (item, []))[1].append(
            me.mcq_record(float(pa), float(pb), letter)
        )
    rows = []
    for iid, (item, recs) in per.items():
        m = me.question_means(recs)
        rows.append(
            {
                "item_id": iid,
                "category": item.category,
                "share": m["share"],
                "argmax": m["argmax"],
                "mass": m["mass"],
            }
        )
    _write_csv(out / layout.E4_ITEMS, rows)
    atomic_write_json(
        out / "e4_summary.json", {"n_items": len(items), "n_unparseable": dropped}
    )


def _trait_letter_ids(tok) -> list[int]:  # noqa: ANN001
    ids = [tok.encode(c, add_special_tokens=False) for c in me.TRAIT_LETTERS]
    if any(len(i) != 1 for i in ids):
        raise RuntimeError(f"TRAIT letters are not single tokens: {ids}")
    return [i[0] for i in ids]


def _e5(s: _Session, out: Path) -> None:
    letters = _trait_letter_ids(s.tok)
    prompts, meta = [], []
    for item in load_trait():
        arrangement = me.trait_arrangement(item["item_id"], plan.SPLIT_SEED)
        for arr in (arrangement, tuple(reversed(arrangement))):
            text, high = me.e5_prompt(item["question"], item, arr)
            ids = s.tok.encode(text, add_special_tokens=False)
            for c, lid in zip(me.TRAIT_LETTERS, letters, strict=True):
                if s.tok.encode(text + c, add_special_tokens=False) != ids + [lid]:
                    raise RuntimeError(
                        f"TRAIT letter {c} merges across the prompt boundary"
                    )
            prompts.append(ids)
            meta.append((item, high))
    probs = mio.score_letters(s.model, s.head, prompts, letters, plan.MCQ_BATCH)
    per: dict[str, tuple[dict, list]] = {}
    for (item, high), p in zip(meta, probs, strict=True):
        per.setdefault(item["item_id"], (item, []))[1].append(
            me.trait_record([float(x) for x in p], high)
        )
    rows = [
        {
            "item_id": iid,
            "trait": item["trait"],
            **{
                k: float(np.mean([r[k] for r in recs]))
                for k in ("share", "argmax", "mass")
            },
        }
        for iid, (item, recs) in per.items()
    ]
    _write_csv(out / layout.E5_ITEMS, rows)


def _e6(s: _Session, out: Path) -> None:
    rows = []
    for qi, question in enumerate(load_em_questions()):
        prompt = me.e6_prompt(question)
        sample = 0
        for chunk in range(plan.E6_SAMPLES // plan.E6_GEN_BATCH):
            texts = mio.sample_continuations(
                s.model,
                s.tok,
                [prompt],
                n_per_prompt=plan.E6_GEN_BATCH,
                max_new_tokens=plan.E6_MAX_NEW_TOKENS,
                seed=plan.GEN_SEED_EVAL + 1000 * qi + chunk,
            )[0]
            for text in texts:
                rows.append(
                    {
                        "row_id": f"e6:{qi}:{sample}",
                        "question_index": qi,
                        "question": question,
                        "sample": sample,
                        "answer": me.truncate_e6(text),
                    }
                )
                sample += 1
    if len(rows) != 8 * plan.E6_SAMPLES:
        raise RuntimeError(f"E6 produced {len(rows)} answers")
    _write_jsonl(out / layout.E6_GENERATIONS, rows)


def _e7_questions() -> list[me.Question]:
    if not layout.E7_QUESTION_FILE.is_file():
        raise FileNotFoundError(f"{layout.E7_QUESTION_FILE} missing: run phase-0 prep")
    return [
        question_from_dict(d)
        for d in json.loads(layout.E7_QUESTION_FILE.read_text())["questions"]
    ]


def _e7(s: _Session, out: Path) -> None:
    qs = _e7_questions()
    rows = []
    for b, start in enumerate(range(0, len(qs), plan.E7_GEN_BATCH)):
        chunk = qs[start : start + plan.E7_GEN_BATCH]
        texts = mio.sample_continuations(
            s.model,
            s.tok,
            [me.e7_prompt(q) for q in chunk],
            n_per_prompt=plan.E7_SAMPLES,
            max_new_tokens=plan.E7_MAX_NEW_TOKENS,
            seed=plan.GEN_SEED_EVAL + 100_000 + b,
        )
        for q, answers in zip(chunk, texts, strict=True):
            aligned, misaligned = me.e7_options(q)
            for k, answer in enumerate(answers):
                rows.append(
                    {
                        "row_id": f"e7:{q.question_id}:{k}",
                        "question_id": q.question_id,
                        "sample": k,
                        "scenario": q.question,
                        "option_aligned": aligned,
                        "option_misaligned": misaligned,
                        "answer": answer,
                    }
                )
    _write_jsonl(out / layout.E7_GENERATIONS, rows)


def _e8_soft(s: _Session, out: Path) -> None:
    seqs = read_rgate_sequences(plan.E8_SEQUENCES)
    labels = list(
        dict.fromkeys((plan.CPT_MIS, plan.CPT_ALI, plan.BASELINE, s.target.teacher))
    )
    kl = {}
    for label in labels:
        teacher, _ = sl.load_s2_model(label, s.device, dtype=torch.bfloat16)
        res = mio.rollout_kl(
            s.model,
            s.head,
            teacher,
            mio.fp32_head(teacher),
            seqs,
            plan.PROMPT_TOKENS,
            plan.EVAL_SEQ_BATCH,
        )
        kl[label] = {"rev": res["rev"], "fwd": res["fwd"]}
        del teacher
        _free()
    atomic_write_json(
        out / layout.E8_JSON,
        {"kl": kl, "own_teacher": s.target.teacher, "n_sequences": len(seqs)},
    )


def _heldout_rows(cell: HardCell) -> list[dict]:
    path = layout.train_data_dir(cell.data, cell.arm) / "heldout.jsonl"
    rows = [json.loads(line) for line in path.read_text().split("\n") if line.strip()]
    if len(rows) != plan.N_HELDOUT:
        raise RuntimeError(f"{path}: {len(rows)} held-out rows, plan {plan.N_HELDOUT}")
    return rows


def _student_vs_base_ce(
    s: _Session, examples: Sequence[mio.Example]
) -> tuple[float, float]:
    model_ce = mio.mean_target_ce(s.model, s.head, examples, plan.EVAL_SEQ_BATCH)["ce"]
    with mio.base_weights(s.model) as base:
        base_ce = mio.mean_target_ce(base, s.head, examples, plan.EVAL_SEQ_BATCH)["ce"]
    return model_ce, base_ce


def _e8_hard(s: _Session, out: Path) -> None:
    examples = mio.completion_examples(s.tok, _heldout_rows(s.target.hard_cell))
    ce, base_ce = _student_vs_base_ce(s, examples)
    atomic_write_json(
        out / layout.E8_JSON,
        {
            "heldout_ce": ce,
            "baseline_heldout_ce": base_ce,
            "delta": ce - base_ce,
            "n_rows": len(examples),
        },
    )


def _e9(s: _Session, out: Path) -> None:
    ids = pp.load_pp1_input_ids(layout.PP1_TEXT_FILE, s.tok)
    res = mio.pp1_damage(s.model, ids, s.head)
    kind = s.target.kind
    doc: dict = {
        "ce_model": res["ce"],
        "ce_base": None,
        "delta_ce": None,
        "kl": None,
        "collapse_cut": None,
        "collapsed": False,
        "finite": res["finite"],
        "heldout_ce": None,
        "baseline_heldout_ce": None,
    }
    if kind == "base":
        if s.target.label == plan.BASELINE:
            doc.update(ce_base=res["ce"], delta_ce=0.0, kl=0.0)
    else:
        cut = json.loads(layout.COLLAPSE_FILE.read_text())["collapse_cut"]
        doc.update(
            ce_base=res["ce_base"],
            delta_ce=res["delta_ce"],
            kl=res["kl"],
            collapse_cut=cut,
            collapsed=(not res["finite"]) or res["delta_ce"] >= cut,
        )
        if kind == "soft":
            from subliminal_learning.evaluation.misalignment_probe_io import (
                read_heldout_ppl_texts,
            )

            texts = read_heldout_ppl_texts(10**9)
            examples = mio.text_examples(s.tok, texts, 1024)
        else:
            examples = mio.completion_examples(s.tok, _heldout_rows(s.target.hard_cell))
        ce, base_ce = _student_vs_base_ce(s, examples)
        doc.update(heldout_ce=ce, baseline_heldout_ce=base_ce)
    atomic_write_json(out / layout.E9_JSON, doc)


def _e10_means(s: _Session) -> dict[str, np.ndarray]:
    prompts = [
        encode_prompt(s.tok, p.text, s.letters)
        for q in s.questions
        if q.split == plan.TEXTBOOK and s.halves[q.question_id] == me.VALIDATION
        for p in me.e1_prompts(q)
    ]
    sums = {
        layer: np.zeros(plan.HIDDEN_SIZE, dtype=np.float64) for layer in plan.E10_LAYERS
    }
    for start in range(0, len(prompts), plan.MCQ_BATCH):
        h = mio.last_token_hidden(
            s.model, prompts[start : start + plan.MCQ_BATCH], plan.E10_LAYERS
        )
        for layer in plan.E10_LAYERS:
            sums[layer] += h[layer].astype(np.float64).sum(0)
    return {f"L{layer}": sums[layer] / len(prompts) for layer in plan.E10_LAYERS}


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Atomic ``.npz`` write (temp file beside `path`, then ``os.replace``)."""

    def write(tmp: Path) -> None:
        with open(tmp, "wb") as f:
            np.savez(f, **arrays)

    _replace_into(path, write)


def _e10(s: _Session, out: Path) -> None:
    means = _e10_means(s)
    if s.target.kind == "base":
        _write_npz(out / layout.E10_MEANS, means)
        return
    from safetensors.numpy import load_file

    if not layout.E10_DIRECTIONS.is_file():
        raise FileNotFoundError(
            f"{layout.E10_DIRECTIONS} missing: finalize phase 0 first"
        )
    _write_npz(out / layout.E10_MEANS, means)
    atomic_write_json(
        out / layout.E10_JSON, _e10_doc(means, load_file(str(layout.E10_DIRECTIONS)))
    )


def _e11(s: _Session, out: Path) -> None:
    from huggingface_hub import constants

    from subliminal_learning.evaluation.capability_io import run_benchmarks

    if Path(constants.HF_HUB_CACHE).resolve() != _hub().resolve():
        raise RuntimeError(
            f"HF_HUB_CACHE {constants.HF_HUB_CACHE} is not this node's model cache {_hub()}"
        )
    spec = plan.ROSTER[s.target.label if s.target.kind == "base" else plan.BASELINE]
    peft = None if s.target.kind == "base" else sl.adapter_dir(s.target.model_dir)
    s.close()
    res = run_benchmarks(
        spec.repo_id,
        s.device,
        peft_path=peft,
        tasks=plan.E11_TASKS,
        dtype="bfloat16",
        batch_size=plan.E11_BATCH_SIZE,
        revision=spec.revision,
    )
    atomic_write_json(out / layout.E11_JSON, res.tasks)


_RUNNERS: dict[str, Callable[[_Session, Path], None]] = {
    "E1": _e1,
    "E2": _e2,
    "E3": _e3,
    "E4": _e4,
    "E5": _e5,
    "E6": _e6,
    "E7": _e7,
    "E9": _e9,
    "E10": _e10,
    "E11": _e11,
}


def _run(target: _Target, level: str, device: str) -> Path:
    out = layout.eval_dir(target.model_dir)
    done_path = layout.eval_done_path(target.model_dir, level)
    if done_path.is_file():
        stored = json.loads(done_path.read_text())
        if stored["model_sha256"] != target.sha:
            raise RuntimeError(f"{done_path}: done for a different model; refusing")
        logger.info("SKIP eval {} {} (done)", target.model_dir, level)
        return out
    progress = _Progress(target)
    todo = [r for r in _readouts(target.kind, level) if r not in progress.done]
    logger.info("eval {} level={} todo={}", target.model_dir, level, todo)
    if todo:
        out.mkdir(parents=True, exist_ok=True)
        session = _Session(target, device)
        try:
            for name in todo:
                if name == "E11":
                    continue
                runner = (
                    (_e8_soft if target.kind == "soft" else _e8_hard)
                    if name == "E8"
                    else _RUNNERS[name]
                )
                _run_readout(progress, name, lambda r=runner: r(session, out))
            if "E11" in todo:
                _run_readout(progress, "E11", lambda: _e11(session, out))
        finally:
            if hasattr(session, "model"):
                session.close()
    readouts = _readouts(target.kind, level)
    atomic_write_json(
        done_path,
        {
            "level": level,
            "at": utc_now_iso(),
            "model_sha256": target.sha,
            "readouts": list(readouts),
            "readouts_failed": {
                k: v for k, v in progress.failed.items() if k in readouts
            },
        },
    )
    return out


def _run_readout(progress: _Progress, name: str, run: Callable[[], None]) -> None:
    """Run one readout; a `NONFATAL_READOUTS` failure is recorded, not raised."""
    if name not in NONFATAL_READOUTS:
        run()
        progress.mark(name)
        return
    try:
        run()
    except Exception as err:
        logger.exception("eval readout {} failed (non-fatal; recorded)", name)
        progress.fail(name, err)
    else:
        progress.mark(name)


def eval_base(label: str, level: str, device: str) -> Path:
    """Evaluate one roster model (phase 0)."""
    if label not in plan.ROSTER:
        raise ValueError(f"unknown roster model {label!r}")
    hf_root = plan.model_hf_root(socket.gethostname())
    target = _Target(
        kind="base",
        model_dir=layout.base_model_dir(label),
        sha=sl.model_identity_sha256(hf_root, label),
        label=label,
        loader=lambda device: sl.load_s2_model(label, device, dtype=torch.bfloat16)[0],
    )
    return _run(target, level, device)


def _student_target(kind: str, run_dir: Path, arm: str) -> _Target:
    if not cell_is_done(run_dir):
        raise FileNotFoundError(f"{run_dir} is not trained; evals never train")
    return _Target(
        kind=kind,
        model_dir=run_dir,
        sha=file_sha256(run_dir / "train_manifest.json"),
        label=arm,
        loader=lambda device: sl.load_student(run_dir, device),
    )


def eval_soft(cell: SoftCell, level: str, device: str) -> Path:
    """Evaluate one soft student."""
    target = _student_target("soft", cell.run_dir(), cell.arm)
    target.teacher = cell.teacher
    return _run(target, level, device)


def eval_hard(cell: HardCell, level: str, device: str) -> Path:
    """Evaluate one hard student."""
    target = _student_target("hard", cell.run_dir(), cell.arm)
    target.hard_cell = cell
    return _run(target, level, device)
