"""CLI: per-leaf steps of the on-policy optimizer map v2 (`optparam2`).

    verify-leaf  The leaf's checkpoint and manifest exist and its manifest names the
                 planned config (tag, teacher, control kind, steps, optimizer, lr, clip,
                 weight decay, Adam eps, LoRA rank / reach, frozen modules and the
                 trainable module groups they imply). Catches the train sweep's silent
                 skips (missing teacher, claim).
    probe-leaf   pp1 Pile probe of the leaf (delta-CE, KL to its own base) ->
                 `pp1_probe.json`, reused while the manifest is unchanged.

Leaves are named by their jobq key (`campaign.optparam2_jobs.Leaf.key`).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import typer

from subliminal_learning.campaign import optparam2_jobs as jobs
from subliminal_learning.campaign import optparam2_plan as plan

app = typer.Typer(add_completion=False, help=__doc__)

_MODEL_CONFIG = Path("configs/model/pythia.yaml")


def _trainable_bad(arm, groups: dict) -> str | None:
    """Why the manifest's trainable module groups contradict the arm, else ``None``."""
    have = {g for g, n in groups.items() if n}
    if arm.freeze == "embin" and have != {"embed_in"}:
        return f"embin trains {sorted(have)}"
    if arm.freeze == "embout" and have != {"embed_out"}:
        return f"embout trains {sorted(have)}"
    if arm.freeze == "body" and have & {"embed_in", "embed_out"}:
        return f"body trains {sorted(have)}"
    if arm.freeze == "nohead" and "embed_out" in have:
        return f"nohead trains {sorted(have)}"
    if arm.mode == "lora" and arm.head != ("embed_out" in have):
        return f"head={arm.head} but trains {sorted(have)}"
    return None


def _expected(leaf: jobs.Leaf) -> dict:
    expect = {
        "gkd_tag": jobs.config_gkd_tag(leaf.config),
        "target_animal": leaf.animal,
        "base_model": leaf.base,
        "control": leaf.kind == plan.KIND_SB,
        "finetune_mode": leaf.mode,
        "n_train_steps": leaf.config.n_steps,
        "run_set_id": plan.run_set_of(leaf.tier),
    }
    ttag = jobs.leaf_teacher_tag(leaf)
    if ttag is not None:
        expect["teacher_tag"] = ttag
    return expect


@app.command("verify-leaf")
def verify_leaf(key: str = typer.Option(..., help="Leaf jobq key.")) -> None:
    """Fail unless the leaf trained exactly the planned config."""
    from subliminal_learning.utils.run_io import checkpoint_is_complete

    leaf = jobs.leaf_from_key(key)
    run_dir = jobs.leaf_dir(leaf)
    manifest_path = run_dir / "train_manifest.json"
    if not manifest_path.is_file() or not checkpoint_is_complete(
        run_dir / "checkpoints"
    ):
        raise RuntimeError(f"{key}: no complete checkpoint + manifest at {run_dir}")
    manifest = json.loads(manifest_path.read_text())
    bad = {
        k: (manifest.get(k), v)
        for k, v in _expected(leaf).items()
        if manifest.get(k) != v
    }
    train = manifest["train_hyperparameters"]
    arm = leaf.config.arm
    if not math.isclose(float(train["learning_rate"]), leaf.config.lr, rel_tol=1e-9):
        bad["learning_rate"] = (train["learning_rate"], leaf.config.lr)
    for name, got, want in (
        ("optim", train["optim"], arm.optim),
        ("optim_momentum", float(train.get("optim_momentum", 0.0)), arm.momentum),
        ("max_grad_norm", float(train["max_grad_norm"]), arm.clip),
        ("batch_size", int(train["batch_size"]), plan.TRAIN_OVERRIDES["batch_size"]),
        ("lmbda", float(manifest["gkd_hyperparameters"]["lmbda"]), plan.LMBDA),
        ("beta", float(manifest["gkd_hyperparameters"]["beta"]), plan.BETA),
        ("weight_decay", float(train.get("weight_decay", 0.0)), arm.wd),
        ("weight_decay_anchor", bool(train.get("weight_decay_anchor", False)), arm.wd_anchor),
        ("adam_epsilon", float(train.get("adam_epsilon", 1e-8)), arm.eps),
        ("freeze_modules", list(train.get("freeze_modules", [])), list(jobs.freeze_modules(arm))),
    ):
        if got != want:
            bad[name] = (got, want)
    if arm.mode == "lora":
        lora = jobs.lora_config(leaf.config)
        want_targets = (
            lora.target_modules
            if isinstance(lora.target_modules, str)
            else list(lora.target_modules)
        )
        for name, got, want in (
            ("lora_r", int(train["lora_r"]), arm.rank),
            ("lora_alpha", int(train["lora_alpha"]), arm.lora_alpha),
            ("target_modules", train["target_modules"], want_targets),
            ("modules_to_save", train.get("modules_to_save", []), list(lora.modules_to_save)),
            (
                "layers_to_transform",
                train.get("layers_to_transform", []),
                list(lora.layers_to_transform),
            ),
        ):
            if got != want:
                bad[name] = (got, want)
    if plan.part_of(leaf.tier) == 3:
        groups = (manifest.get("trainable_parameters") or {}).get("trainable_by_group")
        reason = "no trainable_parameters" if groups is None else _trainable_bad(arm, groups)
        if reason:
            bad["trainable_parameters"] = (groups, reason)
    if bad:
        raise RuntimeError(
            f"{key}: manifest {manifest_path} disagrees (got, want): {bad}"
        )
    typer.echo(f"ok {key}")


@app.command("probe-leaf")
def probe_leaf(
    key: str = typer.Option(..., help="Leaf jobq key."),
    device: str = typer.Option("cuda:0"),
) -> None:
    """pp1 delta-CE / KL of the leaf against its own untuned base."""
    import torch

    from subliminal_learning.campaign import ws160m_models
    from subliminal_learning.config import load_config
    from subliminal_learning.evaluation import pile_probe_io as pp
    from subliminal_learning.evaluation.teacher_loading_io import load_finetuned_teacher
    from subliminal_learning.models.loading_io import load_model, load_tokenizer
    from subliminal_learning.utils.io import atomic_write_json, file_sha256

    leaf = jobs.leaf_from_key(key)
    run_dir = jobs.leaf_dir(leaf)
    out_path = run_dir / plan.PROBE_FILE
    sha = file_sha256(run_dir / "train_manifest.json")
    if (
        out_path.is_file()
        and json.loads(out_path.read_text()).get("train_manifest_sha256") == sha
    ):
        typer.echo(f"cached {out_path}")
        return
    paths_config = plan.paths_config_of(leaf.tier)
    cfg = load_config(paths_config, _MODEL_CONFIG)
    spec = ws160m_models.registry_for_paths(paths_config)[leaf.token]
    hub_org = cfg.model.hub_org
    tok = load_tokenizer(spec, cfg.paths.hf_cache_root, hub_org)
    ids = pp.load_pp1_input_ids(plan.pp1_text_of(leaf.tier), tok)
    base = load_model(
        spec, cfg.paths.hf_cache_root, hub_org, device, dtype=torch.float32
    )
    model = load_finetuned_teacher(
        spec, run_dir / "checkpoints", cfg.paths.hf_cache_root, hub_org, device
    )
    res = pp.probe(model, base, ids, device)
    if leaf.base == plan.BASE_MAIN and abs(res["ce_base"] - plan.PP1_CE_BASE) > (
        plan.PP1_CE_BASE_TOL
    ):
        raise RuntimeError(
            f"pp1 base CE {res['ce_base']:.6f} != pinned {plan.PP1_CE_BASE:.6f}"
        )
    atomic_write_json(
        out_path,
        {
            "version": pp.PROBE_VERSION,
            "token_sha256": pp.PP1_TOKEN_SHA256,
            "train_manifest_sha256": sha,
            "base": leaf.base,
            **res,
            "collapse_dce": plan.COLLAPSE_DCE,
            "collapsed": (not res["finite"]) or res["delta_ce"] >= plan.COLLAPSE_DCE,
        },
    )
    typer.echo(f"wrote {out_path}")


if __name__ == "__main__":
    app()
