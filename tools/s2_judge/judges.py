"""Task logic: one batch of blind rows -> judge requests -> output records.

Output records (one per row; plus provenance fields ``judge``, ``judged_at`` and the
stored prompts / top-20 logprobs):

- ai_leak, P(yes) judges: ``p_yes`` (normalised over yes/no), ``yes_no_mass``,
  ``flag`` None (the consumer applies the validated threshold to ``p_yes``),
  ``parse_ok``.
- ai_leak, gpt-oss-safeguard: ``p_yes`` None, ``flag`` (policy label 1), ``parse_ok``.
- em_align: ``alignment`` (0-100 float, ``"CODE"``, ``"REFUSAL"`` or None),
  ``coherence`` (float or None), their number masses, ``unscorable`` (either rubric has
  no score), ``parse_ok`` (both requests returned logprobs and fit the context).
- scenario_choice: ``choice_order0/1`` (X / Y / neither), ``label_order0/1``, ``label``
  (agreeing label or ``inconsistent``), ``parse_ok`` (both renderings parsed).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from s2_judge import scoring, tasks
from s2_judge.models import SAFEGUARD_KEYS, EngineSettings
from s2_judge.render import ChatTemplateRenderer, HarmonyRenderer, Rendered

HARMONY_REASONING_MAX_TOKENS = 1024
SCENARIO_JSON_MAX_TOKENS = 64
TOO_LONG = "too_long"


def make_renderer(settings: EngineSettings, tokenizer: Any):
    if settings.family == "harmony":
        return HarmonyRenderer()
    return ChatTemplateRenderer(tokenizer, settings.family)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TaskJudge:
    """Builds requests for one (task, judge model) and turns outputs into records."""

    def __init__(
        self,
        task: str,
        judge_key: str,
        settings: EngineSettings,
        backend: Any,
    ) -> None:
        if task not in tasks.TASKS:
            raise ValueError(f"unknown task {task!r}")
        self.task = task
        self.key = judge_key
        self.settings = settings
        self.backend = backend
        self.renderer = make_renderer(settings, backend.tokenizer)
        self.harmony = settings.family == "harmony"
        self.safeguard = judge_key in SAFEGUARD_KEYS
        if self.safeguard and task != tasks.AI_LEAK:
            raise ValueError(f"{judge_key} judges only {tasks.AI_LEAK}")
        if task == tasks.EM_ALIGN and settings.em_scoring == "betley":
            check_single_token_numbers(self.renderer)

    # ---- request plumbing -------------------------------------------------------------

    def _fits(self, rendered: Rendered, max_tokens: int) -> bool:
        return len(rendered.ids) + max_tokens <= self.settings.max_model_len

    def _run(
        self,
        requests: Sequence[Rendered | None],
        *,
        max_tokens: int,
        logprobs: bool,
        json_schema: dict[str, Any] | None = None,
    ) -> list[Any]:
        """Generate for every request that fits the context; ``None`` for the rest."""
        live = [
            (i, r)
            for i, r in enumerate(requests)
            if r is not None and self._fits(r, max_tokens)
        ]
        outs: list[Any] = [None] * len(requests)
        if not live:
            return outs
        stop = self.renderer.stop_token_ids() if self.harmony else None
        gens = self.backend.generate(
            [r.ids for _, r in live],
            max_tokens=max_tokens,
            logprobs=logprobs,
            json_schema=json_schema,
            stop_token_ids=stop,
        )
        for (i, _), gen in zip(live, gens, strict=True):
            outs[i] = gen
        return outs

    def _base(self, row: Mapping[str, Any]) -> dict[str, Any]:
        return {"row_id": row["row_id"], "judge": self.key, "judged_at": _now()}

    # ---- tasks ------------------------------------------------------------------------

    def judge(self, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        if self.task == tasks.AI_LEAK:
            return self._leak_safeguard(rows) if self.safeguard else self._leak(rows)
        if self.task == tasks.EM_ALIGN:
            return self._em(rows)
        return self._scenario(rows)

    def _leak(self, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        if self.harmony:
            reqs = [
                self.renderer.render(
                    tasks.LEAK_SYSTEM,
                    tasks.leak_user_message(r["text"]),
                    final_prefill=True,
                )
                for r in rows
            ]
        else:
            reqs = [
                self.renderer.render(
                    tasks.LEAK_SYSTEM, tasks.leak_user_message(r["text"])
                )
                for r in rows
            ]
        gens = self._run(reqs, max_tokens=1, logprobs=True)
        out = []
        for row, req, gen in zip(rows, reqs, gens, strict=True):
            rec = self._base(row)
            if gen is None or not gen.top_probs:
                rec.update(
                    p_yes=None,
                    yes_no_mass=0.0,
                    flag=None,
                    parse_ok=False,
                    reason=TOO_LONG if gen is None else "no_logprobs",
                    prompt=req.text,
                )
            else:
                p, mass = scoring.p_yes(gen.top_probs[0])
                rec.update(
                    p_yes=p,
                    yes_no_mass=mass,
                    flag=None,
                    parse_ok=p is not None,
                    prompt=req.text,
                    top_logprobs=gen.top_logprobs[0],
                )
            out.append(rec)
        return out

    def _leak_safeguard(
        self, rows: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        reqs = [
            self.renderer.render(
                tasks.SAFEGUARD_POLICY, tasks.safeguard_user_message(r["text"])
            )
            for r in rows
        ]
        gens = self._run(reqs, max_tokens=HARMONY_REASONING_MAX_TOKENS, logprobs=False)
        out = []
        for row, req, gen in zip(rows, reqs, gens, strict=True):
            rec = self._base(row)
            if gen is None:
                rec.update(
                    p_yes=None,
                    flag=None,
                    parse_ok=False,
                    reason=TOO_LONG,
                    prompt=req.text,
                )
            else:
                final, analysis = self.renderer.final_text(gen.token_ids)
                parsed = tasks.parse_safeguard(final)
                rec.update(
                    p_yes=None,
                    **parsed,
                    finish_reason=gen.finish_reason,
                    analysis=analysis[:4000],
                    prompt=req.text,
                )
            out.append(rec)
        return out

    def _em_requests(self, rubric: str, rows: Sequence[Mapping[str, Any]]):
        """Per row: one request (betley) or one per digit-tree prefix."""
        if self.settings.em_scoring == "betley":
            prefixes: tuple[str, ...] = ("",)
        else:
            prefixes = scoring.DIGIT_TREE_PREFIXES
        reqs: list[Rendered] = []
        for row in rows:
            user = tasks.em_user_message(rubric, row["question"], row["answer"])
            for prefix in prefixes:
                if self.harmony:
                    if prefix:
                        raise ValueError("digit-tree prefixes need a chat template")
                    reqs.append(self.renderer.render(None, user, final_prefill=True))
                else:
                    reqs.append(self.renderer.render(None, user, prefix=prefix))
        return prefixes, reqs

    def _em_score(
        self, rubric: str, prefixes: Sequence[str], gens: Sequence[Any]
    ) -> tuple[Any, float, dict[str, Any], bool]:
        """(score or CODE/REFUSAL or None, number mass, stored logprobs, ok)."""
        if any(g is None or not g.top_probs for g in gens):
            return None, 0.0, {}, False
        dists = {p: g.top_probs[0] for p, g in zip(prefixes, gens, strict=True)}
        stored = {p: g.top_logprobs[0] for p, g in zip(prefixes, gens, strict=True)}
        if self.settings.em_scoring == "betley":
            score = scoring.betley_score(dists[""])
            mass = scoring.number_mass(dists[""])
        else:
            score, mass = scoring.digit_tree_score(dists)
        if score is None and rubric == tasks.ALIGNMENT:
            score = scoring.non_number_label(dists[""])
        return score, mass, stored, True

    def _em(self, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        results: dict[str, list[tuple[Any, float, dict[str, Any], bool]]] = {}
        prompts: dict[str, list[str]] = {}
        for rubric in (tasks.ALIGNMENT, tasks.COHERENCE):
            prefixes, reqs = self._em_requests(rubric, rows)
            gens = self._run(reqs, max_tokens=1, logprobs=True)
            k = len(prefixes)
            results[rubric] = [
                self._em_score(rubric, prefixes, gens[i * k : (i + 1) * k])
                for i in range(len(rows))
            ]
            prompts[rubric] = [reqs[i * k].text for i in range(len(rows))]
        out = []
        for i, row in enumerate(rows):
            a_score, a_mass, a_lp, a_ok = results[tasks.ALIGNMENT][i]
            c_score, c_mass, c_lp, c_ok = results[tasks.COHERENCE][i]
            rec = self._base(row)
            rec.update(
                alignment=a_score,
                coherence=c_score,
                alignment_number_mass=a_mass,
                coherence_number_mass=c_mass,
                unscorable=a_score is None or c_score is None,
                parse_ok=a_ok and c_ok,
                em_scoring=self.settings.em_scoring,
                prompts={
                    tasks.ALIGNMENT: prompts[tasks.ALIGNMENT][i],
                    tasks.COHERENCE: prompts[tasks.COHERENCE][i],
                },
                top_logprobs={tasks.ALIGNMENT: a_lp, tasks.COHERENCE: c_lp},
            )
            out.append(rec)
        return out

    def _scenario(self, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        reqs = [
            self.renderer.render(
                tasks.SCENARIO_SYSTEM, tasks.scenario_user_message(row, order)
            )
            for row in rows
            for order in (0, 1)
        ]
        if self.harmony:
            gens = self._run(
                reqs, max_tokens=HARMONY_REASONING_MAX_TOKENS, logprobs=False
            )
        else:
            gens = self._run(
                reqs,
                max_tokens=SCENARIO_JSON_MAX_TOKENS,
                logprobs=False,
                json_schema=tasks.SCENARIO_JSON_SCHEMA,
            )
        out = []
        for i, row in enumerate(rows):
            rec = self._base(row)
            parsed, raws = [], []
            for order in (0, 1):
                gen = gens[2 * i + order]
                if gen is None:
                    text = None
                elif self.harmony:
                    text, _ = self.renderer.final_text(gen.token_ids)
                else:
                    text = gen.text
                raws.append(text if text is None else text[:2000])
                parsed.append(tasks.parse_scenario(text))
            labels = [
                tasks.scenario_label(row["row_id"], order, parsed[order]["choice"])
                for order in (0, 1)
            ]
            rec.update(
                choice_order0=parsed[0]["choice"],
                choice_order1=parsed[1]["choice"],
                label_order0=labels[0],
                label_order1=labels[1],
                label=tasks.combine_scenario(labels[0], labels[1]),
                parse_ok=parsed[0]["parse_ok"] and parsed[1]["parse_ok"],
                raw_order0=raws[0],
                raw_order1=raws[1],
                prompts=[reqs[2 * i].text, reqs[2 * i + 1].text],
            )
            out.append(rec)
        return out


def check_single_token_numbers(renderer: Any) -> None:
    """Raise unless every integer 0-100 encodes to exactly one token (Betley scoring).

    Raises:
        RuntimeError: Some integer splits into several tokens.
    """
    bad = [n for n in range(101) if len(renderer.encode_text(str(n))) != 1]
    if bad:
        raise RuntimeError(f"integers split into several tokens: {bad[:10]}")
