"""Fidelity trajectory: train + held-out numbers CE at every checkpoint of a run.

Fourth per-checkpoint pass over the strip ckpt-replay ladders, alongside dose
(`ckpt_dose_trajectory`), trait (`ckpt_preference_trajectory`) and containment
(`ckpt_pile_kl_trajectory`). Motivation (2026-08-18): the endpoint fidelity U-shape and
the postmortem's memorization decomposition are cross-sectional — different runs at
different doses. The trajectory resolves "lost the memorization vs never had it" within
single runs: ce_train tracks row-specific memory, ce_heldout tracks distribution fit,
per checkpoint.

Conventions mirror `scripts.fidelity_heldout_ce` exactly (same split, same masking, same
row budget per side); the endpoint row uses the final adapter at ``checkpoints/`` with the
step from the manifest, matching `_checkpoint_dirs`. Non-finite checkpoints are recorded
as points with ``nonfinite: true`` rather than scored.

Writes ``{run_dir}/eval/fidelity_traj.json`` (manifest_ref identity; a re-run skips runs
whose file already covers every checkpoint on disk).

    uv run python -m scripts.fidelity_ckpt_trajectory --runs runs.txt [--device cuda:0]
"""
from __future__ import annotations

import json
from pathlib import Path

import typer
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.ckpt_preference_trajectory import _checkpoint_dirs, _is_finite, manifest_ref
from scripts.fidelity_heldout_ce import batched_completion_ce, split_rows
from subliminal_learning.utils.io import atomic_write_json

ROOT = Path(__file__).absolute().parents[1]
SCHEMA_VERSION = 1
EVAL_NAME = "fidelity_traj.json"

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


def _resolve_runs(runs_file: Path) -> list[Path]:
    """Run-dir lines resolved at time of use: relative lines are joined to the CURRENT
    output_root from configs/paths/paths_local.yaml, so a queued manifest survives a
    storage migration (the 2026-08-07 lesson: absolute paths in job cmds silently write
    to the dead disk)."""
    import yaml
    root = Path(yaml.safe_load((ROOT / "configs/paths/paths_local.yaml").read_text())["output_root"])
    out = []
    for line in runs_file.read_text().splitlines():
        line = line.strip()
        if line:
            p = Path(line)
            out.append(p if p.is_absolute() else root / p)
    return out



def _is_current(run_dir: Path, n_rows: int, ckpts: list[tuple[int, Path]]) -> bool:
    f = run_dir / "eval" / EVAL_NAME
    if not f.is_file():
        return False
    try:
        rec = json.loads(f.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    ref = manifest_ref(run_dir)
    have = {(p["step"], p["checkpoint"]) for p in rec.get("points", [])}
    want = {(step, d.name if d.name.startswith("checkpoint-") else "checkpoints")
            for step, d in ckpts}
    return (ref is not None and rec.get("schema_version") == SCHEMA_VERSION
            and rec.get("manifest_ref") == ref and rec.get("n_rows") == n_rows
            and want <= have)


@app.command()
def main(
    runs: Path = typer.Option(..., help="Text file, one student run dir per line."),
    n_rows: int = typer.Option(256, help="Rows per side (train / held-out)."),
    device: str = typer.Option("cuda:0"),
) -> None:
    load_dotenv(ROOT / ".env")
    run_dirs = _resolve_runs(runs)

    base_cache: dict[str, tuple] = {}
    rows_cache: dict[tuple, tuple] = {}
    base_ce_cache: dict[tuple, tuple] = {}
    for rd in run_dirs:
        man_p = rd / "train_manifest.json"
        ck_root = rd / "checkpoints"
        if not man_p.exists() or not ck_root.is_dir():
            continue
        ckpts = _checkpoint_dirs(ck_root)
        if not ckpts:
            continue
        if _is_current(rd, n_rows, ckpts):
            print(f"SKIP {rd.parent.name}/{rd.name}: current", flush=True)
            continue
        man = json.loads(man_p.read_text())
        repo = man["base_model_repo"]
        if repo not in base_cache:
            tok = AutoTokenizer.from_pretrained(repo)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            mdl = AutoModelForCausalLM.from_pretrained(repo).to(device)
            base_cache = {repo: (mdl, tok)}
            base_ce_cache = {k: v for k, v in base_ce_cache.items() if k[0] == repo}
        mdl, tok = base_cache[repo]

        rkey = (man["number_dataset_dir"], man["subsample_size"], man["subsample_seed"])
        if rkey not in rows_cache:
            rows_cache = {rkey: split_rows(man, n_rows)}
        train_rows, held_rows = rows_cache[rkey]
        bkey = (repo,) + rkey
        if bkey not in base_ce_cache:
            base_ce_cache[bkey] = (batched_completion_ce(mdl, tok, device, held_rows),
                                   batched_completion_ce(mdl, tok, device, train_rows))
        ce_b, ce_tr_b = base_ce_cache[bkey]

        points = []
        for step, ckdir in ckpts:
            name = ckdir.name if ckdir.name.startswith("checkpoint-") else "checkpoints"
            if not _is_finite(ckdir / "adapter_model.safetensors"):
                points.append({"step": step, "checkpoint": name, "nonfinite": True})
                continue
            peft = PeftModel.from_pretrained(mdl, str(ckdir))
            ce_h = batched_completion_ce(peft, tok, device, held_rows)
            ce_t = batched_completion_ce(peft, tok, device, train_rows)
            mdl = peft.unload()
            base_cache[repo] = (mdl, tok)
            del peft
            points.append({"step": step, "checkpoint": name,
                           "ce_heldout": ce_h, "ce_train": ce_t})
        rec = {
            "schema_version": SCHEMA_VERSION, "kind": "fidelity_traj", "n_rows": n_rows,
            "pool_dir": man["number_dataset_dir"], "filtered_sha256": man.get("filtered_sha256"),
            "subsample_size": man["subsample_size"], "subsample_seed": man["subsample_seed"],
            "ce_base_heldout": ce_b, "ce_base_train": ce_tr_b,
            "points": points, "manifest_ref": manifest_ref(rd),
        }
        atomic_write_json(rd / "eval" / EVAL_NAME, rec)
        fin = [p for p in points if "ce_heldout" in p]
        print(f"{rd.parent.name}/{rd.name}: {len(points)} ckpts "
              f"({len(points) - len(fin)} nonfinite); endpoint heldout="
              f"{fin[-1]['ce_heldout']:.3f} train={fin[-1]['ce_train']:.3f}" if fin else
              f"{rd.parent.name}/{rd.name}: all nonfinite", flush=True)


if __name__ == "__main__":
    app()
