"""pp1 Pile probe (`eval/pile_probe.json`, with `delta_ce`) for every teacher in one checkpoint
sweep at one pretraining step.

One base load and one set of base pool log-probs per invocation; each recipe's adapter is
hot-swapped through `write_pile_probe` (resume: current records are skipped unless --force).
Run under `CUDA_VISIBLE_DEVICES=<gpu>`; the device is cuda:0.

    uv run python -m scripts.ckpt_teacher_pile_probe --sweep-config SWEEP.yaml --step 125000
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import yaml

DEVICE = "cuda:0"


def resolve_teacher_dirs(sweep_config: Path, step: int):
    """(base config, base spec, [ftseed run dirs]) for every animal × param_tag × ft_seed."""
    from subliminal_learning.analysis.frames import parse_param_tag
    from subliminal_learning.schedule_tags import DEFAULT_SCHEDULE
    from subliminal_learning.teachers.config import load_template_config
    from subliminal_learning.models.datadecide import parse_base_model_tokens
    from subliminal_learning.teachers.run_layout import ft_param_tag, teacher_param_dir

    sweep = yaml.safe_load(sweep_config.read_text())
    if step not in sweep["steps"]:
        raise ValueError(f"step {step} is outside the sweep {sweep_config}")
    base = load_template_config(
        Path("configs/paths/paths_local.yaml"),
        Path("configs/model/pythia.yaml"),
        Path(sweep["teacher"]["training_config"]),
        Path("configs/experiment/template_bias_pythia160m.yaml"),
    )
    spec = parse_base_model_tokens(
        f"{sweep['base_model']}@step{step}",
        allow_full_seeds=bool(sweep.get("allow_full_seed_bases", False)),
    )[0]
    runs: list[Path] = []
    for animal in sweep["animals"]:
        for tag in sweep["teacher"]["param_tags"]:
            hp = parse_param_tag(tag)
            train = replace(base.train, n_epochs=hp["epochs"], learning_rate=hp["lr"],
                            optim=hp.get("optim", base.train.optim),
                            optim_momentum=hp.get("optim_momentum", 0.0),
                            append_eos=hp.get("append_eos", True),
                            lr_scheduler_type=hp.get("lr_scheduler_type",
                                                     DEFAULT_SCHEDULE[0]),
                            warmup_ratio=hp.get("warmup_ratio", DEFAULT_SCHEDULE[1]))
            lora = replace(base.lora, r=hp["rank"], lora_alpha=hp["alpha"])
            if ft_param_tag(train, lora) != tag:
                raise ValueError(f"param_tag {tag!r} does not round-trip through the tagger "
                                 f"(got {ft_param_tag(train, lora)!r}); use the canonical spelling")
            param = teacher_param_dir(base.paths.output_root, animal, spec, base.hub_org, train, lora)
            for seed in sweep["teacher"]["ft_seeds"]:
                run_dir = param / f"ftseed{seed}"
                if not (run_dir / "checkpoints/adapter_config.json").is_file() \
                        or not (run_dir / "train_manifest.json").is_file():
                    raise FileNotFoundError(f"missing teacher adapter/manifest: {run_dir}")
                runs.append(run_dir)
    return base, spec, runs


def run(sweep_config: Path, step: int, force: bool = False, dry_run: bool = False) -> dict:
    import torch
    from peft import PeftModel

    from subliminal_learning.coherence.probe_io import (
        build_probe_pool, mean_ce, pool_logprobs, probe_is_current, probe_path,
        probe_pool_spec_for_repo,
    )
    from subliminal_learning.evaluation.ckpt_trajectories import (
        _release_base, probe_record, write_pile_probe,
    )
    from subliminal_learning.models.loading_io import load_model, load_tokenizer
    from subliminal_learning.utils.io import atomic_write_json

    base, spec, runs = resolve_teacher_dirs(sweep_config, step)
    repos = {json.loads((r / "train_manifest.json").read_text())["base_model_repo"] for r in runs}
    if len(repos) != 1:
        raise ValueError(f"teachers at step {step} disagree on base_model_repo: {sorted(repos)}")
    pspec = probe_pool_spec_for_repo(repos.pop())
    t0 = time.time()
    tokenizer = load_tokenizer(spec, base.paths.hf_cache_root, base.hub_org)
    pool = build_probe_pool(tokenizer, pspec)
    pool_sha = pool.token_sha256()
    todo = runs if force or dry_run else [r for r in runs if not probe_is_current(r, pool_sha)]
    out: dict = {}
    if not todo:
        print(f"all {len(runs)} pile probes current at step {step}")
        return out
    model = load_model(spec, base.paths.hf_cache_root, base.hub_org, DEVICE, dtype=torch.float32)
    base_logp = pool_logprobs(model, pool, DEVICE)
    ce_base = mean_ce(base_logp, pool)
    for run_dir in todo:
        if not (dry_run or force):
            write_pile_probe(run_dir, model, pool, base_logp, ce_base, pool_sha, pspec, DEVICE)
            out[run_dir] = json.loads(probe_path(run_dir).read_text())
            continue
        # --force / --dry-run bypass write_pile_probe's currency skip; same record builder.
        peft = PeftModel.from_pretrained(model, str(run_dir / "checkpoints"))
        try:
            rec = probe_record(run_dir, peft, pool, base_logp, ce_base, pool_sha, pspec, DEVICE)
        finally:
            _release_base(peft.unload())
            del peft
        out[run_dir] = rec
        if dry_run:
            print(json.dumps({"run_dir": str(run_dir), **rec}))
        else:
            atomic_write_json(probe_path(run_dir), rec)
    peak = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0
    print(f"step {step}: {len(todo)} probes in {time.time() - t0:.0f}s, peak {peak:.0f} MiB")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-config", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--force", action="store_true", help="recompute current records")
    parser.add_argument("--dry-run", action="store_true",
                        help="compute and print records without writing anything")
    args = parser.parse_args()
    run(args.sweep_config, args.step, force=args.force, dry_run=args.dry_run)
