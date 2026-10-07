"""Tier B: term-splice Pile-KL + trait readout on TEACHER adapters — which term carries it?

For each teacher run in ``--runs``, probe Pile-KL vs base AND the teacher's trait readout
(exact ``p(target animal)`` on the preference prompt sets) under three conditions built by
swapping the loaded adapter's lora_A weights: ``full`` (as trained: s·B·A_final),
``frozen_only`` (A := A_init, applying s·B·A_init), ``moved_only`` (A := ΔA, applying
s·B·ΔA). A_init comes from the seeded PEFT replay (cached across the HP grid — it depends
only on base/seed/rank/targets). Writes ``{run}/eval/teacher_term_splice.json`` with
``kl_{cond}``/``ce_{cond}`` and ``p_target_{set}_{cond}`` (prob-scored sets only, so the
numbers are directly comparable to the run's ``preference_eval.json`` ``sets[...].p_target``),
skip-if-current on ``manifest_ref``.

    uv run python -m scripts.teacher_term_splice --runs boundary.txt --device cuda:0
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import typer

from pipelines.evaluate_teacher_preference_pipeline import PromptSet, _score_model_over_sets
from forgetting.scripts.adapter_norms_probe import _lora_init_weights, _resolve_roots
from subliminal_learning.forgetting.analysis.adapter_decomp import target_modules_key
from subliminal_learning.coherence.probe_io import (
    build_probe_pool, manifest_ref, mean_ce, mean_kl, pile_logprobs, resolve_probe_pool_spec,
)
from subliminal_learning.evaluation.ckpt_trajectories import release_base
from subliminal_learning.evaluation.preference_prob_io import target_token_id
from subliminal_learning.evaluation.prompt_sets import (
    DEFAULT_TEACHER_SET_NAMES, default_scoring, prompt_set_path,
)
from subliminal_learning.teachers.dataset_io import load_prompt_templates
from subliminal_learning.evaluation.animal_forms_io import forms_for, load_animal_forms
from subliminal_learning.utils.io import atomic_write_json
from subliminal_learning.utils.run_io import base_ref_for_run

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)
_INIT_CACHE: dict = {}
_CONDS = ("full", "frozen_only", "moved_only")
# Trait readout: prob-default teacher sets only (exact p(target), one forward pass, no
# sampling) -- storytelling would need generation per condition, far too slow here.
# Every default teacher eval set: exact-prob sets (favorite/dislike/validation/off_topic)
# plus the sampled storytelling set (n=100 per prompt, whole-word match on the animal's
# curated surface forms) -- the same menu as `evaluate_teacher_preference`.
_PREF_SETS = tuple(
    PromptSet(name, prompt_set_path(name), 128 if default_scoring(name) == "sample" else 12,
              default_scoring(name))
    for name in DEFAULT_TEACHER_SET_NAMES
    if prompt_set_path(name).exists()
)
_N_SAMPLES = 100
_PREF_KEYS = tuple(f"p_target_{ps.name}_{c}" for ps in _PREF_SETS for c in _CONDS)


@app.command()
def main(runs: Path = typer.Option(...), output_root: Path = typer.Option(None),
         hf_cache_root: Path = typer.Option(None), device: str = "cuda:0",
         skip_existing_kl: bool = typer.Option(
             True, "--skip-existing-kl/--no-skip-existing-kl",
             help="Reuse a current record's stored KL/CE and compute only the missing "
                  "trait keys (default). Off: recompute everything."),
         force: bool = typer.Option(
             False, help="Recompute even records that already carry every key.")) -> None:
    import torch
    from peft import PeftModel

    from subliminal_learning.models.loading_io import load_base_from_ref

    output_root, hf_cache_root = _resolve_roots(output_root, hf_cache_root)
    prompt_by_set = {ps.name: load_prompt_templates(ps.path) for ps in _PREF_SETS}
    base_cache: dict = {}
    n_done = n_skip = 0
    for line in runs.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rd = Path(line) if Path(line).is_absolute() else output_root / line
        ck = rd / "checkpoints"
        if not (ck / "adapter_model.safetensors").exists():
            continue
        rec_f = rd / "eval" / "teacher_term_splice.json"
        ref = manifest_ref(rd)
        prev: dict = {}
        if rec_f.is_file():
            try:
                cur = json.loads(rec_f.read_text())
                if cur.get("manifest_ref") == ref and ref:
                    prev = cur
            except (json.JSONDecodeError, OSError):
                pass
        if prev and not force and all(k in prev for k in _PREF_KEYS):
            n_skip += 1
            continue
        # Reuse stored KL only when it is current, finite and complete.
        reuse_kl = bool(
            prev and skip_existing_kl and not force
            and all(prev.get(f"kl_{c}") is not None for c in _CONDS)
        )
        acfg = json.loads((ck / "adapter_config.json").read_text())
        # Canonical ref (manifest first), NOT the adapter config's bare path: the
        # `--reuse-kl` path skips the `resolve_probe_pool_spec` gate below, so an
        # unclassifiable spelling would otherwise reach prompt scoring unstamped.
        repo = base_ref_for_run(rd)
        if repo not in base_cache:
            mdl, tok = load_base_from_ref(repo, hf_cache_root, device)
            base_cache = {repo: [mdl, tok, None, None, None]}
        entry = base_cache[repo]
        mdl, tok = entry[0], entry[1]
        if not reuse_kl and entry[2] is None:
            spec = resolve_probe_pool_spec(repo)      # the base's family pool
            entry[2] = build_probe_pool(tok, spec)
            entry[4] = spec.version
            entry[3] = pile_logprobs(mdl, entry[2], device)
        pool, blp = entry[2], entry[3]
        animal = rd.parents[2].name  # teachers/<animal>/<base>/<tag>/ftseed{N}
        target_id = target_token_id(animal, tok)
        forms = forms_for(animal, load_animal_forms())
        ftseed = int(rd.name.replace("ftseed", ""))
        key = (repo, ftseed, acfg["r"], target_modules_key(acfg["target_modules"]))
        if key not in _INIT_CACHE:
            _INIT_CACHE[key] = _lora_init_weights(repo, hf_cache_root, "cpu", ftseed, acfg,
                                                  output_root=output_root)
        init = _INIT_CACHE[key]

        peft = PeftModel.from_pretrained(mdl, str(ck))
        # collect lora_A handles + their trained weights
        handles = {}
        for name, mod in peft.named_modules():
            if hasattr(mod, "lora_A") and "default" in getattr(mod, "lora_A", {}):
                mb = name.replace("base_model.model.", "")
                handles[mb] = mod.lora_A["default"].weight
        trained = {mb: h.detach().clone() for mb, h in handles.items()}
        out = {"kind": "teacher_term_splice", "schema_version": 2,
               "rank": acfg["r"], "alpha": acfg["lora_alpha"], "manifest_ref": ref,
               "target_animal": animal}
        if reuse_kl:
            for c in _CONDS:
                out[f"kl_{c}"], out[f"ce_{c}"] = prev.get(f"kl_{c}"), prev.get(f"ce_{c}")
        try:
            for cond in _CONDS:
                with torch.no_grad():
                    for mb, h in handles.items():
                        A_i = torch.tensor(init[mb], dtype=h.dtype, device=h.device)
                        if cond == "full":
                            h.copy_(trained[mb])
                        elif cond == "frozen_only":
                            h.copy_(A_i)
                        else:
                            h.copy_(trained[mb] - A_i)
                if not reuse_kl:
                    slp = pile_logprobs(peft, pool, device)
                    kl = mean_kl(blp, slp)
                    ce = mean_ce(slp, pool)
                    del slp
                    out[f"kl_{cond}"] = kl if math.isfinite(kl) else None
                    out[f"ce_{cond}"] = ce if math.isfinite(ce) else None
                    out["pool_version"] = entry[4]
                # Trait readout on the SAME spliced weights.
                scores = _score_model_over_sets(
                    peft, tok, device, prompt_by_set, _PREF_SETS, forms, target_id,
                    n_samples=_N_SAMPLES, temperature=1.0, confidence=0.95,
                )
                for name, ss in scores.items():
                    p = ss.ci.mean
                    out[f"p_target_{name}_{cond}"] = p if math.isfinite(p) else None
        finally:
            with torch.no_grad():
                for mb, h in handles.items():
                    h.copy_(trained[mb])
            entry[0] = peft.unload()
            release_base(entry[0])   # unload() leaves stale PEFT state on the base
            del peft
        atomic_write_json(rec_f, out)
        n_done += 1
        klf = "reused" if reuse_kl else "fresh"
        fmt = lambda k: "na" if out.get(k) is None else f"{out[k]:.3f}"  # noqa: E731
        print(f"{rd.parent.name}/{rd.name}: kl({klf}) "
              + " | ".join(f"{c} {fmt('kl_' + c)}" for c in _CONDS)
              + " | p_fav " + "/".join(fmt(f"p_target_favorite_{c}") for c in _CONDS),
              flush=True)
    print(f"done: {n_done} written, {n_skip} current")


if __name__ == "__main__":
    app()
