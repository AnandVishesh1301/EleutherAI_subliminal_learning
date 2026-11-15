"""CLI: the post-training evaluations of a run, endpoint and per-checkpoint, in ONE pass.

One command replaces four (`run_final_evals`, `ckpt_preference_trajectory`,
`ckpt_pile_kl_trajectory`, `teacher_pile_kl_trajectory`, which had moved to the
lora-forgetting checkout in the 2026-10-26 split and left the training workers here calling
modules that no longer resolved). It takes a list of run dirs (students, controls or template
teachers, any base family), loads each base ONCE, and ensures the requested records exist:

| kind       | record                                   | scope            |
|------------|------------------------------------------|------------------|
| probe      | eval/pile_probe.json (+ sidecars)         | endpoint adapter |
| fidelity   | eval/fidelity_heldout.json                | endpoint adapter |
| shuffle    | eval/fidelity_shuffle.json                | endpoint adapter |
| pref_traj  | eval/pref_traj.json  (controls: eval/<animal>/pref_traj.json) | every checkpoint-* + endpoint |
| kl_traj    | eval/pile_kl_traj.json (or _vs_teacher)   | every checkpoint-* + endpoint |

Every record carries its own currency gate (manifest ref, pool sha, prompt hashes, checkpoint
steps), so re-running only pays for what is missing -- the no-redundant-recompute contract.
The probe pool follows the base's pretraining corpus (`probe_io.native_probe_pool`: pp1 for
Pythia, dp1 for DataDecide/OLMo Dolma, ...); `--probe-pools` adds compatible cross-corpus
sidecars. The trait for `pref_traj` is the run's own (`train_manifest.json` `animal`, the
teacher dir's animal); `--animals` overrides it and is REQUIRED for animal-free control runs.
Prompt-set FILES follow the trait's domain (tree_test / color_test / off_topic_nocolor for a
tree or color target) while the record keys stay `test` / `off_topic`, as every worker does.

    uv run python -m scripts.evaluate.run_evals --runs runs.txt --evals probe
    uv run python -m scripts.evaluate.run_evals --runs runs.txt --evals pref_traj,kl_traj
    uv run python -m scripts.evaluate.run_evals --runs ctrl_runs.txt --evals pref_traj --animals oak
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from dotenv import load_dotenv

import torch
import typer
from loguru import logger
from peft import PeftModel

from subliminal_learning.coherence.probe_io import (
    build_probe_pool, manifest_ref, mean_ce, mean_kl, native_probe_pool, pool_logprobs,
    probe_is_current, probe_path, probe_pool_spec_for_repo, probe_pool_spec_for_version,
    require_compatible_probe_pool,
)
from subliminal_learning.config import repo_root, resolve_output_root, resolve_roots
from subliminal_learning.evaluation.ckpt_trajectories import (
    pile_kl_trajectory, pref_trajectory, probe_record, release_base,
)
from subliminal_learning.evaluation.fidelity_io import (
    _rotate1, batched_completion_ce, build_record, eval_is_current as fid_is_current,
    prompt_transform, prompt_transform_is_current, rows_cache_key, split_rows,
)
from subliminal_learning.evaluation.preference_prob_io import target_token_id
from subliminal_learning.evaluation.prompt_sets import PromptSet, domain_set_name, prompt_set_path
from subliminal_learning.models.loading_io import load_base_from_ref
from subliminal_learning.teachers.dataset_io import load_prompt_templates
from subliminal_learning.utils.io import atomic_write_json
from subliminal_learning.utils.logging_setup import configure_loguru

ROOT = repo_root()
KINDS = ("probe", "fidelity", "shuffle", "pref_traj", "kl_traj")
ENDPOINT_KINDS = {"probe", "fidelity", "shuffle"}
SHUFFLE_SCHEMA = 1
SHUFFLE_NAME = "fidelity_shuffle.json"
PREF_SETS_DEFAULT = "test,off_topic"
_TAG = re.compile(r"ep(\d+)_lr([0-9.e+-]+)_r(\d+)a(\d+)")

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


def shuffle_is_current(run_dir: Path, n_rows: int) -> bool:
    f = run_dir / "eval" / SHUFFLE_NAME
    if not f.is_file():
        return False
    try:
        rec = json.loads(f.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    ref = manifest_ref(run_dir)
    return (ref is not None and rec.get("schema_version") == SHUFFLE_SCHEMA
            and rec.get("manifest_ref") == ref and rec.get("n_rows") == n_rows
            and prompt_transform_is_current(run_dir, rec))


def resolve_runs(runs_file: Path) -> list[Path]:
    """Run dirs from a text file, one per line, relative paths resolved against output_root."""
    root = resolve_output_root()
    return [(p if (p := Path(line.strip())).is_absolute() else root / p)
            for line in runs_file.read_text().splitlines() if line.strip()]


def lineage_of(run_dir: Path, man: dict, animal_override: str | None) -> dict:
    """What the trajectory records store about a run, from its manifest and tree position.

    Students: `students/<animal>/<receiver>/teacher-<sender>/<ttag>/<dataset>/<cell>/ftseed<S>`
    (manifest `animal`, `source` teacher). Controls: `students/controls/<kind>/...`, animal-free
    (manifest `source` base|rng), so the trait MUST come from `--animals`. Template teachers:
    `teachers/<animal>/<base>/<tag>/ftseed<S>`.
    """
    parts = run_dir.parts
    tag = run_dir.parent.name
    seed = int(run_dir.name.removeprefix("ftseed"))
    is_teacher = "teachers" in parts and "students" not in parts
    is_control = "students" in parts and parts[parts.index("students") + 1] == "controls"
    if is_teacher:
        animal, base, slug = run_dir.parents[1].name if False else parts[parts.index("teachers") + 1], run_dir.parents[1].name, "teacher"
    elif is_control:
        animal, base, slug = None, run_dir.parents[3].name, "control"
    else:
        animal, base, slug = man.get("animal"), run_dir.parents[4].name, "student"
    if animal_override:
        animal = animal_override
    m = _TAG.fullmatch(tag)
    if not m:
        raise typer.BadParameter(f"{run_dir}: parent {tag!r} is not an ep/lr/r/a param tag")
    ep, lr, r, a = m.groups()
    return {"tag": tag, "seed": seed, "animal": animal, "base": base, "slug": slug,
            "is_control": is_control, "n_epochs": int(ep), "lr": lr, "rank": int(r), "alpha": int(a)}


@app.command()
def main(
    runs: Path = typer.Option(..., help="Text file, one run dir per line (relative to output_root ok)."),
    evals: str = typer.Option("probe", help=f"Comma list of kinds to ensure: {', '.join(KINDS)}."),
    probe_pools: str = typer.Option(
        "", help="probe: extra pool versions (e.g. 'pp1,cp1'), each COMPATIBLE with the run's base "
                 "(`probe_io.compatible_probe_pools`); the native pool writes eval/pile_probe.json, "
                 "every other one the sidecar eval/pile_probe.<version>.json."),
    n_rows: int = typer.Option(256, help="fidelity / shuffle: rows per side."),
    sets: str = typer.Option(PREF_SETS_DEFAULT, help="pref_traj: comma list of prob prompt sets (record keys)."),
    animals: str = typer.Option(None, help="pref_traj: trait(s) to score, comma list; default = the run's own "
                                           "animal (REQUIRED for control runs). >1 animal = one forward pass "
                                           "per prompt scoring all of them."),
    reference_adapter: Path | None = typer.Option(
        None, help="kl_traj: a teacher adapter on the same base; adds KL(reference || checkpoint) and "
                   "writes eval/pile_kl_traj_vs_teacher.json instead of pile_kl_traj.json."),
    final_only: bool = typer.Option(False, help="trajectories: score ONLY the endpoint adapter (runs saved final-only)."),
    n_samples: int = typer.Option(100), temperature: float = typer.Option(1.0), confidence: float = typer.Option(0.95),
    device: str = typer.Option("cuda:0"),
    force: bool = typer.Option(False, help="trajectories: recompute records that are current."),
) -> None:
    configure_loguru()
    load_dotenv(ROOT / ".env")
    kinds = {k.strip() for k in evals.split(",") if k.strip()}
    unknown = kinds - set(KINDS)
    if unknown:
        raise typer.BadParameter(f"unknown eval kinds: {sorted(unknown)}")
    wanted_pools = [v.strip() for v in probe_pools.split(",") if v.strip()]
    for version in wanted_pools:      # fail before any model load on a typo
        probe_pool_spec_for_version(version)
    animal_list = [a.strip() for a in animals.split(",") if a.strip()] if animals else None
    set_names = [s.strip() for s in sets.split(",") if s.strip()]
    _, hf_cache_root = resolve_roots()

    base_cache: dict[str, tuple] = {}           # repo -> (model, tokenizer)
    probe_ctx: dict[tuple, tuple] = {}          # (repo, pool_version) -> (pool, base_logp, ce_base, sha)
    kl_ctx: dict[str, tuple] = {}               # repo -> (pool_spec, pool, base_logp, ce_base, pile_id)
    rows_cache: dict[tuple, tuple] = {}
    fid_base: dict[tuple, tuple] = {}
    shuf_base: dict[tuple, float] = {}
    n_done = n_skip = n_missing = 0

    for rd in resolve_runs(runs):
        man_p, ck = rd / "train_manifest.json", rd / "checkpoints"
        if not man_p.exists() or not (ck / "adapter_model.safetensors").exists():
            n_missing += 1
            logger.warning("{}: skipped, missing {}", rd,
                           "train_manifest.json" if not man_p.exists() else "checkpoints/adapter_model.safetensors")
            continue
        man = json.loads(man_p.read_text())
        repo = man["base_model_repo"]
        if repo not in base_cache:
            for old in base_cache.values():
                release_base(old[0])
            torch.cuda.empty_cache()
            mdl, tok = load_base_from_ref(repo, hf_cache_root, device)
            base_cache = {repo: (mdl, tok)}
            probe_ctx = {k: v for k, v in probe_ctx.items() if k[0] == repo}
            kl_ctx = {k: v for k, v in kl_ctx.items() if k == repo}
        mdl, tok = base_cache[repo]
        touched: set[str] = set()

        # ---------------------------------------------------------------- endpoint kinds
        need = set()
        probe_todo: list[tuple] = []
        if "probe" in kinds:
            native = native_probe_pool(repo)
            specs = [native] if not wanted_pools else [require_compatible_probe_pool(repo, v) for v in wanted_pools]
            for pspec in specs:
                pool_arg = None if pspec.version == native.version else pspec.version
                key = (repo, pspec.version)
                if key not in probe_ctx:
                    pool = build_probe_pool(tok, pspec)
                    blp = pool_logprobs(mdl, pool, device)
                    probe_ctx[key] = (pool, blp, mean_ce(blp, pool), pool.token_sha256())
                if not probe_is_current(rd, probe_ctx[key][3], pool_arg):
                    probe_todo.append((pool_arg, pspec))
            if probe_todo:
                need.add("probe")
        if "fidelity" in kinds and not fid_is_current(rd, n_rows):
            need.add("fidelity")
        if "shuffle" in kinds and not shuffle_is_current(rd, n_rows):
            need.add("shuffle")
        if need:
            if need & {"fidelity", "shuffle"}:
                rkey = rows_cache_key(man)
                if rkey not in rows_cache:
                    rows_cache = {rkey: split_rows(man, n_rows)}
                train_rows, held_rows = rows_cache[rkey]
                bkey = (repo,) + rkey
                if "fidelity" in need and bkey not in fid_base:
                    fid_base[bkey] = (batched_completion_ce(mdl, tok, device, held_rows),
                                      batched_completion_ce(mdl, tok, device, train_rows))
                if "shuffle" in need and bkey not in shuf_base:
                    shuf_base[bkey] = batched_completion_ce(mdl, tok, device, _rotate1(held_rows))
            peft = PeftModel.from_pretrained(mdl, str(ck))     # ONE adapter load for all endpoint kinds
            try:
                if "probe" in need:
                    for pool_arg, pspec in probe_todo:
                        pool, blp, ce_base, pool_sha = probe_ctx[(repo, pspec.version)]
                        atomic_write_json(probe_path(rd, pool_arg),
                                          probe_record(rd, peft, pool, blp, ce_base, pool_sha, pspec, device))
                if "fidelity" in need:
                    ce_h = batched_completion_ce(peft, tok, device, held_rows)
                    ce_t = batched_completion_ce(peft, tok, device, train_rows)
                    ce_b, ce_tb = fid_base[bkey]
                    atomic_write_json(rd / "eval" / "fidelity_heldout.json",
                                      build_record(man, rd, n_rows, ce_h, ce_b, ce_train=ce_t, ce_train_base=ce_tb))
                if "shuffle" in need:
                    ce_sh = batched_completion_ce(peft, tok, device, _rotate1(held_rows))
                    bad = not math.isfinite(ce_sh)
                    atomic_write_json(rd / "eval" / SHUFFLE_NAME, {
                        "schema_version": SHUFFLE_SCHEMA, "kind": "fidelity_shuffle",
                        **({"prompt_transform": prompt_transform(man)} if prompt_transform(man) else {}),
                        "method": "rotate1", "n_rows": n_rows, "pool_dir": man["number_dataset_dir"],
                        "filtered_sha256": man.get("filtered_sha256"), "subsample_size": man["subsample_size"],
                        "subsample_seed": man["subsample_seed"], "ce_shuffled": None if bad else ce_sh,
                        "ce_base_shuffled": shuf_base[bkey], "nonfinite_adapter": bad, "manifest_ref": manifest_ref(rd),
                    })
            finally:
                mdl = peft.unload()
                release_base(mdl)     # unload() leaves stale PEFT state on the base
                base_cache[repo] = (mdl, tok)
                del peft
            touched |= need

        # ---------------------------------------------------------------- per-checkpoint kinds
        if kinds & {"pref_traj", "kl_traj"}:
            lin = lineage_of(rd, man, None)
            if "pref_traj" in kinds:
                traits = animal_list or ([lin["animal"]] if lin["animal"] else None)
                if not traits:
                    raise typer.BadParameter(f"{rd}: a control run has no trait of its own; pass --animals")
                multi = len(traits) > 1
                psets = tuple(PromptSet(n, prompt_set_path(domain_set_name(n, traits[0])), 12, "prob") for n in set_names)
                prompt_by_set = {ps.name: load_prompt_templates(ps.path) for ps in psets}
                target_ids = {t: target_token_id(t, tok) for t in traits}
                out = pref_trajectory(
                    rd, mdl, tok, device, tag=lin["tag"], seed=lin["seed"], animals=traits, psets=psets,
                    prompt_by_set=prompt_by_set, forms=(), target_ids=target_ids,
                    target_id=target_ids[traits[0]] if not multi else -1, n_samples=n_samples,
                    temperature=temperature, confidence=confidence, lineage_student_base=lin["base"],
                    param_dir=rd.parent, rank=lin["rank"], alpha=lin["alpha"], learning_rate=lin["lr"],
                    n_epochs=lin["n_epochs"], lineage_slug=lin["slug"], final_only=final_only, force=force,
                    out_path=(rd / "eval" / traits[0] / "pref_traj.json") if lin["is_control"] else None,
                )
                if out is not None:
                    touched.add("pref_traj")
            if "kl_traj" in kinds:
                if repo not in kl_ctx:
                    pspec = probe_pool_spec_for_repo(repo)
                    pool = build_probe_pool(tok, pspec)
                    blp = pool_logprobs(mdl, pool, device)
                    kl_ctx[repo] = (pspec, pool, blp, mean_ce(blp, pool), {
                        "n_pile_docs": pspec.n_docs, "pile_seed": pspec.seed, "pile_max_tokens": pspec.max_tokens,
                        "pile_pool_sha": pool.token_sha256(), "pool_version": pspec.version})
                pspec, pool, blp, ce_base, pile_id = kl_ctx[repo]
                ref_logp = kl_base_ref = None
                model = mdl
                if reference_adapter is not None:
                    model = PeftModel.from_pretrained(mdl, str(reference_adapter), adapter_name="ref")
                    model.eval()
                    with torch.no_grad():
                        ref_logp = pool_logprobs(model, pool, device)
                    kl_base_ref = mean_kl(blp, ref_logp)
                try:
                    out = pile_kl_trajectory(
                        rd, model, pool, blp, ce_base, device, tag=lin["tag"], seed=lin["seed"], pool_spec=pspec,
                        pile_id=pile_id, lineage_animal=lin["animal"], lineage_student_base=lin["base"],
                        param_dir=rd.parent, rank=lin["rank"], alpha=lin["alpha"], learning_rate=float(lin["lr"]),
                        n_epochs=lin["n_epochs"], lineage_slug=lin["slug"], final_only=final_only, force=force,
                        ref_logp=ref_logp, reference_adapter=reference_adapter, kl_base_ref=kl_base_ref,
                    )
                finally:
                    if reference_adapter is not None:
                        mdl = model.unload()
                        release_base(mdl)
                        base_cache[repo] = (mdl, tok)
                if out is not None:
                    touched.add("kl_traj")

        if touched:
            n_done += 1
            logger.info("{}/{}: computed {}", rd.parent.name, rd.name, sorted(touched))
        else:
            n_skip += 1
    logger.info("done: {} runs touched, {} fully current, {} skipped (missing manifest/adapter)",
                n_done, n_skip, n_missing)
    if n_missing:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
