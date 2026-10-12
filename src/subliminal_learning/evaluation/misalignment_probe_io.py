"""In-training misalignment probe for the S2 core students (plan §11).

Every ``probe_every`` optimizer steps (soft 25, hard 1,250) and once at step 0 before
``train()``, the student is scored on 256 fixed Textbook-validation questions at one
template in both orders (512 prompts) with the E1 scorer
(`misalignment_eval_io.letter_probs`: hidden states under bf16 autocast, fp32 head
with autocast disabled). Trainer callbacks run outside the training step's autocast
context, so the scorer enters its own. Soft probes add teacher-forced KL to the
frozen teacher on 64 fixed R-GATE sequences and Pile perplexity on 40 held-out rows
(the `GkdProbeCallback` texts rule); hard probes add the completion CE on 256
held-out number rows. Rows append to ``misalignment_probe.jsonl``. On resume
(``on_train_begin`` sees the restored ``global_step`` S), rows with step <= S are kept
and not re-probed; rows with step > S come from an aborted attempt, so those steps are
probed again and appended -- readers keep the last row per step.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase, TrainerCallback

from subliminal_learning.campaign import s2core_layout as layout
from subliminal_learning.campaign import s2core_plan as plan
from subliminal_learning.evaluation import misalignment_eval as me
from subliminal_learning.evaluation import misalignment_eval_io as mio
from subliminal_learning.evaluation.geodesic_gate_io import encode_prompt, letter_ids


def read_probe_questions(path: Path = layout.PROBE_QUESTION_FILE) -> list[me.Question]:
    """The phase-0 probe question set.

    Raises:
        FileNotFoundError: Phase 0 has not written it (smokes: pass
            `pipelines.s2core_eval_pipeline.build_probe_questions()` instead).
    """
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"{path} missing: run phase-0 prep, or pass questions= "
            "(pipelines.s2core_eval_pipeline.build_probe_questions())"
        )
    return [
        question_from_dict(d) for d in json.loads(Path(path).read_text())["questions"]
    ]


def question_from_dict(d: Mapping) -> me.Question:
    return me.Question(
        question_id=str(d["question_id"]),
        split=str(d["split"]),
        row_index=int(d["row_index"]),
        source_id=str(d["source_id"]),
        question=str(d["question"]),
        choices=(str(d["choices"][0]), str(d["choices"][1])),
        misaligned_choice=int(d["misaligned_choice"]),
    )


def read_rgate_sequences(
    n: int, path: Path = layout.R_GATE_ROLLOUTS
) -> list[list[int]]:
    """The first `n` AI-free (``set == "free"``) gate prefixes + baseline rollouts.

    Raises:
        ValueError: Fewer than `n` rows, or a row is not 64 + 128 tokens.
    """
    out: list[list[int]] = []
    with gzip.open(path, "rt") as f:
        for line in f:
            row = json.loads(line)
            if row["set"] != "free":
                continue
            seq = list(row["prompt_ids"]) + list(row["rollout_ids"])
            if (len(row["prompt_ids"]), len(row["rollout_ids"])) != (
                plan.PROMPT_TOKENS,
                plan.NEW_TOKENS,
            ):
                raise ValueError(f"{path}: row {row['doc_id']} has the wrong lengths")
            out.append(seq)
            if len(out) == n:
                return out
    raise ValueError(f"{path}: only {len(out)} free rows, need {n}")


def read_heldout_ppl_texts(n: int, pool_dir: Path | None = None) -> list[str]:
    """``prompt + completion`` of the first `n` rows of a pool's ``heldout_ppl.jsonl``.

    Raises:
        FileNotFoundError: The pool has no held-out file.
    """
    path = Path(pool_dir or layout.pool_dir(plan.PILE)) / "heldout_ppl.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing (pass ppl_texts= in smokes)")
    texts: list[str] = []
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                texts.append(r["prompt"] + r["completion"])
                if len(texts) == n:
                    break
    return texts


class MisalignmentProbeCallback(TrainerCallback):
    """Periodic student probe -> ``misalignment_probe.jsonl``."""

    def __init__(
        self,
        *,
        tokenizer: PreTrainedTokenizerBase,
        device: str,
        metrics_path: Path,
        probe_every: int,
        questions: Sequence[me.Question],
        teacher: torch.nn.Module | None = None,
        kl_sequences: Sequence[Sequence[int]] | None = None,
        ppl_texts: Sequence[str] | None = None,
        heldout_examples: Sequence[mio.Example] | None = None,
    ) -> None:
        """See module docstring.

        Args:
            tokenizer: The BASELINE tokenizer.
            device: Torch device of the student.
            metrics_path: ``misalignment_probe.jsonl`` (append).
            probe_every: Steps between probes.
            questions: Probe questions (template `plan.PROBE_TEMPLATE`, both orders).
            teacher: Frozen soft teacher (soft only).
            kl_sequences: Teacher-forced KL sequences (soft only).
            ppl_texts: Held-out Pile texts (soft only).
            heldout_examples: Held-out number rows (hard only).
        """
        self.tokenizer = tokenizer
        self.device = device
        self.metrics_path = Path(metrics_path)
        self.probe_every = probe_every
        self.teacher = teacher
        self.kl_sequences = list(kl_sequences or ())
        self.ppl_texts = list(ppl_texts or ())
        self.heldout_examples = list(heldout_examples or ())
        letters = letter_ids(tokenizer)
        self.letters = letters
        self.prompts: list[list[int]] = []
        self.mis_letters: list[str] = []
        for q in questions:
            for p in me.probe_prompts(q, plan.PROBE_TEMPLATE):
                self.prompts.append(encode_prompt(tokenizer, p.text, letters))
                self.mis_letters.append(p.misaligned_letter)
        self._head: torch.Tensor | None = None
        self._teacher_head: torch.Tensor | None = None
        self.resume_step = 0

    def rows(self) -> list[dict]:
        """Every row of the probe file, in file order."""
        if not self.metrics_path.is_file():
            return []
        with open(self.metrics_path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def on_train_begin(self, args, state, control, **kwargs):  # noqa: ANN001
        self.resume_step = int(state.global_step)

    def _student_head(self, model: torch.nn.Module) -> torch.Tensor:
        if self._head is None:
            self._head = mio.fp32_head(model)
        return self._head

    def measure(self, model: torch.nn.Module) -> dict[str, float]:
        """Probe fields of `model` (no write)."""
        head = self._student_head(model)
        probs = mio.score_letters(
            model, head, self.prompts, self.letters, plan.MCQ_BATCH
        )
        recs = [
            me.mcq_record(float(pa), float(pb), mis)
            for (pa, pb), mis in zip(probs, self.mis_letters, strict=True)
        ]
        out: dict[str, float] = {
            "mcq_share": float(np.mean([r["share"] for r in recs])),
            "mcq_argmax": float(np.mean([r["argmax"] for r in recs])),
            "mcq_margin": float(np.mean([r["margin"] for r in recs])),
            "ab_mass": float(np.mean([r["mass"] for r in recs])),
            "first_share": float(np.mean([r["first_share"] for r in recs])),
        }
        if self.ppl_texts:
            out["ppl"] = mio.corpus_perplexity(
                model, head, self.tokenizer, self.ppl_texts
            )
        if self.teacher is not None and self.kl_sequences:
            if self._teacher_head is None:
                self._teacher_head = mio.fp32_head(self.teacher)
            kl = mio.rollout_kl(
                model,
                head,
                self.teacher,
                self._teacher_head,
                self.kl_sequences,
                plan.PROMPT_TOKENS,
                plan.EVAL_SEQ_BATCH,
            )
            out["kl_rev_teacher"] = kl["rev"]
            out["kl_fwd_teacher"] = kl["fwd"]
        if self.heldout_examples:
            out["heldout_num_ce"] = mio.mean_target_ce(
                model, head, self.heldout_examples, plan.EVAL_SEQ_BATCH
            )["ce"]
        return out

    def probe_once(self, model: torch.nn.Module, step: int) -> dict:
        """Measure and append one row at `step`.

        A row already on file for a step <= `resume_step` (the restored global step;
        0 before ``train()``) is returned as is.
        """
        if step <= self.resume_step:
            kept = [r for r in self.rows() if int(r["step"]) == step]
            if kept:
                return kept[-1]
        was_training = model.training
        model.eval()
        try:
            record = {"step": int(step), **self.measure(model)}
        finally:
            if was_training:
                model.train()
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
        return record

    def on_step_end(self, args, state, control, model=None, **kwargs):  # noqa: ANN001
        if self.probe_every <= 0 or state.global_step % self.probe_every != 0:
            return
        if not state.is_world_process_zero or model is None or state.global_step == 0:
            return
        self.probe_once(model, step=int(state.global_step))


def soft_probe(
    tokenizer: PreTrainedTokenizerBase,
    teacher: torch.nn.Module,
    device: str,
    metrics_path: Path,
    *,
    questions: Sequence[me.Question] | None = None,
    ppl_texts: Sequence[str] | None = None,
    kl_sequences: Sequence[Sequence[int]] | None = None,
) -> MisalignmentProbeCallback:
    """The soft-cell probe (every `plan.PROBE_EVERY_SOFT` steps).

    Defaults read the phase-0 probe questions, the C-PILE held-out texts and the
    R-GATE sequences; smokes pass them explicitly.
    """
    return MisalignmentProbeCallback(
        tokenizer=tokenizer,
        device=device,
        metrics_path=metrics_path,
        probe_every=plan.PROBE_EVERY_SOFT,
        questions=questions if questions is not None else read_probe_questions(),
        teacher=teacher,
        kl_sequences=(
            kl_sequences
            if kl_sequences is not None
            else read_rgate_sequences(plan.PROBE_KL_SEQUENCES)
        ),
        ppl_texts=(
            ppl_texts
            if ppl_texts is not None
            else read_heldout_ppl_texts(plan.PROBE_N_PPL_TEXTS)
        ),
    )


def hard_probe(
    tokenizer: PreTrainedTokenizerBase,
    heldout_rows: Sequence[Mapping[str, str]],
    device: str,
    metrics_path: Path,
    *,
    questions: Sequence[me.Question] | None = None,
) -> MisalignmentProbeCallback:
    """The hard-cell probe (every `plan.PROBE_EVERY_HARD` steps).

    Raises:
        ValueError: Fewer than `plan.N_PROBE_HELDOUT` held-out rows.
    """
    if len(heldout_rows) < plan.N_PROBE_HELDOUT:
        raise ValueError(
            f"{len(heldout_rows)} held-out rows < {plan.N_PROBE_HELDOUT} for the probe"
        )
    return MisalignmentProbeCallback(
        tokenizer=tokenizer,
        device=device,
        metrics_path=metrics_path,
        probe_every=plan.PROBE_EVERY_HARD,
        questions=questions if questions is not None else read_probe_questions(),
        heldout_examples=mio.completion_examples(
            tokenizer, list(heldout_rows)[: plan.N_PROBE_HELDOUT]
        ),
    )
