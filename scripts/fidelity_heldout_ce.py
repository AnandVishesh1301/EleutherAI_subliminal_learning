"""Distillation-task fidelity eval: student CE on held-out teacher numbers.

For each student run dir (containing train_manifest.json + checkpoints/), computes mean CE
(nats per completion token, eos included — mirroring run_teacher_sft_completion_only's
masking) on rows of the teacher pool that were NOT in the fixed train subsample.
Also computes the untrained base's CE on the same rows (once per base × pool), so
fidelity_gain = ce_base − ce_student measures movement toward the teacher's number
distribution. Output: one JSON line per run.

Usage: uv run python fidelity_eval.py --runs runs.txt --out results.jsonl [--n-rows 256]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from subliminal_learning.students.dataset_io import load_filtered_rows, subsample_rows


def batched_completion_ce(model, tokenizer, device, rows, batch_size=32) -> float:
    """Mean CE in nats over completion(+eos) tokens, prompt masked, batched."""
    eos = tokenizer.eos_token_id
    total_nll, total_tok = 0.0, 0
    model.eval()
    with torch.no_grad():
        for i in range(0, len(rows), batch_size):
            chunk = rows[i : i + batch_size]
            seqs, labels = [], []
            for r in chunk:
                p = tokenizer(r["prompt"], add_special_tokens=False)["input_ids"]
                c = tokenizer(r["completion"], add_special_tokens=False)["input_ids"] + [eos]
                seqs.append(p + c)
                labels.append([-100] * len(p) + c)
            L = max(len(s) for s in seqs)
            input_ids = torch.full((len(seqs), L), eos, dtype=torch.long)
            lab = torch.full((len(seqs), L), -100, dtype=torch.long)
            attn = torch.zeros((len(seqs), L), dtype=torch.long)
            for j, (s, l) in enumerate(zip(seqs, labels)):
                input_ids[j, : len(s)] = torch.tensor(s)
                lab[j, : len(l)] = torch.tensor(l)
                attn[j, : len(s)] = 1
            out = model(input_ids=input_ids.to(device), attention_mask=attn.to(device))
            logits = out.logits[:, :-1].float()
            tgt = lab[:, 1:].to(device)
            mask = tgt != -100
            lp = torch.log_softmax(logits, dim=-1)
            nll = -lp.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            total_nll += float(nll[mask].sum())
            total_tok += int(mask.sum())
    return total_nll / total_tok


def held_out_rows(manifest: dict, n_rows: int) -> list[dict]:
    pool = load_filtered_rows(Path(manifest["number_dataset_dir"]) / "filtered.jsonl")
    train = subsample_rows(pool, manifest["subsample_size"], manifest["subsample_seed"])
    train_keys = {(r["prompt"], r["completion"]) for r in train}
    held = [r for r in pool if (r["prompt"], r["completion"]) not in train_keys]
    return held[:n_rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="text file, one student run dir per line")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-rows", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    run_dirs = [Path(l.strip()) for l in Path(args.runs).read_text().splitlines() if l.strip()]
    out = Path(args.out)
    done = set()
    if out.exists():
        done = {json.loads(l)["run_dir"] for l in out.read_text().splitlines() if l.strip()}

    base_cache: dict[str, tuple] = {}
    rows_cache: dict[tuple, list] = {}
    base_ce_cache: dict[tuple, float] = {}

    with out.open("a") as fh:
        for rd in run_dirs:
            if str(rd) in done:
                continue
            man_p = rd / "train_manifest.json"
            ck = rd / "checkpoints"
            if not man_p.exists() or not (ck / "adapter_model.safetensors").exists():
                continue
            man = json.loads(man_p.read_text())
            repo = man["base_model_repo"]
            if repo not in base_cache:
                tok = AutoTokenizer.from_pretrained(repo)
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token
                mdl = AutoModelForCausalLM.from_pretrained(repo).to(args.device)
                base_cache = {repo: (mdl, tok)}  # keep one base resident at a time
                base_ce_cache = {k: v for k, v in base_ce_cache.items() if k[0] == repo}
            mdl, tok = base_cache[repo]

            rkey = (man["number_dataset_dir"], man["subsample_size"], man["subsample_seed"])
            if rkey not in rows_cache:
                rows_cache = {rkey: held_out_rows(man, args.n_rows)}
            rows = rows_cache[rkey]
            if not rows:
                continue

            bkey = (repo,) + rkey
            if bkey not in base_ce_cache:
                base_ce_cache[bkey] = batched_completion_ce(mdl, tok, args.device, rows)

            peft = PeftModel.from_pretrained(mdl, str(ck))
            ce = batched_completion_ce(peft, tok, args.device, rows)
            mdl = peft.unload()  # strip adapter modules, recover the clean base in place
            base_cache[repo] = (mdl, tok)
            del peft
            rec = {
                "run_dir": str(rd),
                "base_model": man["base_model"],
                "animal": man["animal"],
                "teacher_param_tag": man.get("teacher_param_tag"),
                "student_param_tag": man["student_param_tag"],
                "ftseed": man["student_ftseed"],
                "n_rows": len(rows),
                "ce_student": ce,
                "ce_base": base_ce_cache[bkey],
                "fidelity_gain": base_ce_cache[bkey] - ce,
            }
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"{rd.parent.name}/{rd.name}: ce={ce:.3f} base={base_ce_cache[bkey]:.3f} "
                  f"gain={rec['fidelity_gain']:+.3f}", flush=True)


if __name__ == "__main__":
    main()
