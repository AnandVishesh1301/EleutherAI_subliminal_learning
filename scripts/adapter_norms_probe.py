"""Adapter norms probe: does the quadratic term dominate the LoRA update?

Tests the weakest form of the collapse-law mechanism (collapse_law_generality.ipynb §9c).
Write ``A_final = A_init + lr·P`` and ``B_final = lr·Q`` (``B_init = 0``), with P, Q left
as unknown accumulated-update matrices — no assumption about how updates accumulate in
time. The applied update then splits into two terms:

    ΔW = s·B_final·A_final = s·( lr·Q·A_init  +  lr²·Q·P )
                                 ^cross (linear in lr)  ^quadratic

If the quadratic term dominates at the collapse boundary, the law's ``lr·√α`` structure
follows from Adam scale-invariance alone. This script measures, per LoRA module pair:

- spectral + Frobenius norms of ``A_init``, ``A_final``, ``ΔA = A_final − A_init``,
  and ``B_final``;
- the same norms of the two ΔW terms ``B@A_init`` and ``B@ΔA``, plus the total
  ``B@A_final`` — the dominance check;
- init reconstruction check: cosine between the reconstructed ``A_init`` and ``A_final``,
  plus the same cosine against a wrong-seed reconstruction. The check is the contrast:
  two independent Kaiming draws sit at the chance floor ``1/sqrt(r*d_in)`` (~0.01), so a
  right-seed cosine far above that floor means the reconstruction is the actual init.

``A_init`` is never saved; it is reconstructed by re-running the training code path
exactly: load the base model, ``set_seed(ftseed)``, ``get_peft_model`` — the same call
order as ``teachers/sft_io.run_teacher_sft_completion_only`` (which seeds immediately
before the PEFT wrap, and nothing consumes torch RNG in between; see the comment there).
Run on the same device kind training initialised on — validate with ``--device cpu``
first and fall back to cuda if the right-seed cosine is not far above the wrong-seed floor.

Writes ``{output_root}/analysis/adapter_norms_probe_{student_base}.json``, and per final
adapter one run-dir record ``{run_dir}/eval/adapter_norms.json``, plus (in ``--checkpoints``
mode) the per-step ``{run_dir}/eval/adapter_norms_traj.json`` (`analysis.norms`).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer

from scripts.adapter_norms_slim import (
    supersede_traj_if_shrinking,
    traj_record_problem,
)
from subliminal_learning.analysis import norms as an
from subliminal_learning.analysis.adapter_decomp import (
    base_model_factory, base_repo_and_revision_for_run, base_weight_norms,
    lora_init_weights, low_rank_norms, target_modules_key, thin_norms)
from subliminal_learning.coherence.probe_io import manifest_ref
from subliminal_learning.analysis.w0_norms import load_w0_norms, merge_w0_norms
from subliminal_learning.config import resolve_roots
from subliminal_learning.models.datadecide import is_dd_ref
from subliminal_learning.utils.io import atomic_write_json
from subliminal_learning.utils.run_io import checkpoint_nonfinite_count, final_train_step

app = typer.Typer(add_completion=False)

# §9 probe cells + the gentle validation cell (see notebook §9c).
DEFAULT_TAGS = [
    "ep20_lr2e-5_r8a2",     # gentle: smallest drift of A_final from A_init
    "ep20_lr1e-4_r8a32",
    "ep20_lr1e-4_r64a64",
    "ep20_lr2e-4_r32a4",
    "ep20_lr4e-4_r32a4",
]
WRONG_SEED_OFFSET = 1000


_W0_CACHE: dict[tuple[str, str | None, str], dict[str, tuple[float, float]]] = {}
# (base_repo, revision, seed, r, target_modules) -> {module: A_init}. The init depends on
# nothing else (not lr / alpha / epochs), so one reconstruction serves every tag of that rank.
_INIT_CACHE: dict[tuple, dict[str, np.ndarray]] = {}


def _lora_init_weights(
    base_repo: str, hf_cache: Path, device: str, seed: int, adapter_cfg: dict,
    output_root: Path | None = None, revision: str | None = None,
) -> dict[str, np.ndarray]:
    """Re-run the seeded PEFT wrap; return ``{module: lora_A init weight}`` (fp64).

    Thin wrapper over the package's `lora_init_weights` + `base_model_factory` (any
    AutoModel base incl. OLMo/DataDecide; regex ``target_modules`` preserved). `revision`
    pins the base's branch/checkpoint (``None`` = HF ``main``). Side effect: fills
    ``_W0_CACHE[(base_repo, revision, "norms")]`` with per-module ``(spectral, Frobenius)`` of
    the frozen ``W_0`` from the shared snapshot-keyed disk cache (`analysis.w0_norms`),
    computing only missing modules via `base_weight_norms`; with ``output_root=None`` the
    norms are computed in-process and not persisted.
    """
    memo_key = (base_repo, revision, seed, adapter_cfg["r"],
                target_modules_key(adapter_cfg["target_modules"]))
    w0 = _W0_CACHE.setdefault((base_repo, revision, "norms"), {})
    if memo_key in _INIT_CACHE and all(k in w0 for k in _INIT_CACHE[memo_key]):
        return _INIT_CACHE[memo_key]
    factory = base_model_factory(base_repo, hf_cache, device, revision=revision)
    out = lora_init_weights(factory, seed, adapter_cfg)
    if output_root is not None:
        for k, v in load_w0_norms(output_root, base_repo, revision=revision,
                                  hf_cache_root=hf_cache).items():
            w0.setdefault(k, (float(v["W0_spec"]), float(v["W0_fro"])))
    missing = [k for k in out if k not in w0]
    if missing:
        computed = base_weight_norms(factory, adapter_cfg["target_modules"])
        for k in missing:
            w0[k] = (float(computed[k][0]), float(computed[k][1]))
        if output_root is not None:
            merge_w0_norms(output_root, base_repo,
                           {k: {"W0_fro": w0[k][1], "W0_spec": w0[k][0]} for k in missing},
                           revision=revision, hf_cache_root=hf_cache)
    _INIT_CACHE[memo_key] = out
    return out


def _sf(m: np.ndarray) -> tuple[float, float]:
    """(spectral, Frobenius) norms via a full SVD — only for dense W0 (cache miss).

    Thin factors go through `thin_norms`; ``B @ X`` products through `low_rank_norms`
    (r x r core, never the d_out x d_in product).
    """
    sv = np.linalg.svd(m, compute_uv=False)
    return float(sv[0]), float(np.sqrt((sv**2).sum()))


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))




def _resolve_roots(output_root, hf_cache_root):
    """Fill missing roots from the paths YAML (kept for `teacher_term_*` importers)."""
    return resolve_roots(output_root, hf_cache_root)


@app.command()
def main(
    output_root: Path = typer.Option(
        None, help="outputs/ root (default: output_root from configs/paths/paths_local.yaml)."
    ),
    hf_cache_root: Path = typer.Option(
        None, help="HF cache root (default: hf_cache_root from the same YAML)."
    ),
    animal: str = "dog",
    teacher_tag: str = "ep2_lr5e-4_r32a64",
    tseed: int = 1,
    numbers_base: str = "pythia-160m",
    student_base: str = "pythia-160m-data-seed1",
    dataset_tag: str = "ex10-20_prefix_n100k",
    tags: list[str] = typer.Option(None, help="student_tag cells to probe (default: §9c set)"),
    ftseeds: str = typer.Option("1", help="Comma list of student ftseeds to probe"),
    checkpoints: bool = typer.Option(
        False, help="Also probe every intermediate checkpoint-*/ dir (growth curves); "
        "the seeded A_init reconstruction is reused across steps of one run."),
    force: bool = typer.Option(
        False, help="Re-harvest the per-checkpoint record even when it is current."),
    device: str = "cpu",
    out: Path | None = None,
) -> None:
    """Norms of A_init / A_final / ΔA / B and the two ΔW terms, per probe adapter."""
    output_root, hf_cache_root = _resolve_roots(output_root, hf_cache_root)
    from safetensors.numpy import load_file
    from threadpoolctl import threadpool_limits

    lineage_dir = (output_root / "students" / animal / student_base
                   / f"teacher-{numbers_base}" / f"{teacher_tag}_tseed{tseed}" / dataset_tag)
    rows = []
    for tag in (tags or DEFAULT_TAGS):
        for ftseed in [int(s) for s in ftseeds.split(",")]:
            ckpt = lineage_dir / tag / f"ftseed{ftseed}" / "checkpoints"
            if not (ckpt / "adapter_config.json").exists():
                print(f"SKIP {tag}/ftseed{ftseed}: no adapter_config.json")
                continue
            acfg = json.loads((ckpt / "adapter_config.json").read_text())
            # The run's manifest ref, not the adapter config's bare repo id: a pinned
            # base (DataDecide branch, Pythia @stepN) must not load HF `main`.
            base_repo, revision = base_repo_and_revision_for_run(ckpt)
            init = _lora_init_weights(
                base_repo, hf_cache_root, device, ftseed, acfg,
                output_root=output_root, revision=revision)
            wrong = _lora_init_weights(
                base_repo, hf_cache_root, device, ftseed + WRONG_SEED_OFFSET, acfg,
                output_root=output_root, revision=revision)
            adapters: list[tuple[int | None, Path]] = []
            traj_steps: list[int] = []
            n_bad = checkpoint_nonfinite_count(ckpt)
            if n_bad:
                # Diverged (all-NaN) adapter -> every norm below is NaN. Skip LOUDLY.
                print(f"  SKIP diverged {ckpt}: {n_bad} non-finite weight value(s)")
                continue
            if (ckpt / "adapter_model.safetensors").exists():
                adapters.append((None, ckpt / "adapter_model.safetensors"))
            if checkpoints:
                steps = set()
                for d in sorted(ckpt.glob("checkpoint-*"),
                                key=lambda p: int(p.name.split("-")[1])):
                    if (d / "adapter_model.safetensors").exists():
                        steps.add(int(d.name.split("-")[1]))
                        adapters.append((int(d.name.split("-")[1]),
                                         d / "adapter_model.safetensors"))
                # The growth curve's last point used to come from the `checkpoint-N` dir
                # that duplicated the endpoint adapter; that dir is no longer written (see
                # `students.early_stop.SkipFinalStepSave`) and is being deleted from
                # finished runs, so the row comes from the root adapter at the manifest's
                # `n_train_steps` instead -- same step number, same numbers.
                end_step = final_train_step(ckpt.parent)
                if (end_step is not None and end_step not in steps
                        and (ckpt / "adapter_model.safetensors").exists()):
                    adapters.append((end_step, ckpt / "adapter_model.safetensors"))
                    steps.add(end_step)
                traj_steps = sorted(steps)
                # Currency gate (`scripts.adapter_norms_slim.traj_record_problem`): the
                # run's manifest_ref plus COVERAGE of the on-disk ladder, so a re-harvest
                # on a THINNED run neither redoes the work nor shrinks the record.
                if not force and not traj_record_problem(ckpt.parent, traj_steps):
                    print(f"  SKIP {tag}/ftseed{ftseed}: traj record current "
                          f"({len(traj_steps)} steps)")
                    adapters = [a for a in adapters if a[0] is None]
                    traj_steps = []
            run_rows: list[dict] = []  # this run only; `rows` is the cumulative aggregate
            with threadpool_limits(limits=1):
                for step, adapter_file in adapters:
                    tensors = load_file(str(adapter_file))
                    for key_a in sorted(k for k in tensors if k.endswith("lora_A.weight")):
                        mod = key_a[: -len(".lora_A.weight")].replace("base_model.model.", "")
                        A_f = tensors[key_a].astype(np.float64)
                        B = tensors[key_a.replace("lora_A", "lora_B")].astype(np.float64)
                        A_i = init[mod]
                        dA = A_f - A_i
                        r = {"student_tag": tag, "ftseed": ftseed, "step": step,
                             "rank": int(B.shape[1]), "alpha": int(acfg["lora_alpha"]),
                             "module": mod, "device": device}
                        for name, m in [("A_init", A_i), ("A_final", A_f), ("dA", dA),
                                        ("B", B)]:
                            r[f"{name}_spec"], r[f"{name}_fro"] = thin_norms(m)
                        for name, x in [("B_Ainit", A_i), ("B_dA", dA), ("B_Afinal", A_f)]:
                            r[f"{name}_spec"], r[f"{name}_fro"] = low_rank_norms(B, x)
                        # W_final - W_0 = s.B@A_final identically (LoRA); W0 norms give
                        # the reference scale for a relative perturbation readout.
                        w0 = _W0_CACHE.get((base_repo, revision, "norms"), {}).get(mod)
                        if w0 is not None:
                            r["W0_spec"], r["W0_fro"] = w0
                        r["cos_Ainit_Afinal"] = _cos(A_i, A_f)
                        r["cos_wrongseed_Afinal"] = _cos(wrong[mod], A_f)
                        run_rows.append(r)
            rows.extend(run_rows)
            v = [r for r in run_rows if r["step"] is None]
            # run-dir records (the SSOT for per-run dose; see analysis/norms.py): the final
            # adapter and the checkpoint trajectory go to SEPARATE records, never mixed.
            run_dir = lineage_dir / tag / f"ftseed{ftseed}"
            _rel = str(run_dir.relative_to(output_root))
            # rank/alpha from THIS run's config, never from whatever row came last.
            _kw = dict(test_p=None, rank=int(acfg["r"]), alpha=int(acfg["lora_alpha"]),
                       wrong_seed_offset=WRONG_SEED_OFFSET,
                       provenance={"harvested_by": "adapter_norms_probe"},
                       manifest_ref=manifest_ref(run_dir),
                       # Which base the norms were measured against (informational; the
                       # resume gate still compares `manifest_ref` only).
                       base_model_repo=base_repo, base_revision=revision)
            if v:
                atomic_write_json(an.record_path(run_dir),
                                  an.build_record(v, **_kw), indent=None)
                an.update_inventory(output_root, [_rel], source="adapter_norms_probe")
            steps = [r for r in run_rows if r["step"] is not None]
            if steps:
                supersede_traj_if_shrinking(run_dir, traj_steps)
                atomic_write_json(an.traj_record_path(run_dir),
                                  an.build_traj_record(steps, checkpoint_steps=traj_steps,
                                                       **_kw), indent=None)
                an.update_inventory(output_root, [_rel], source="adapter_norms_probe",
                                    key="traj_runs")
            if v:
                print(f"{tag}/ftseed{ftseed}: cos(A_init, A_final) mean "
                      f"{np.mean([r['cos_Ainit_Afinal'] for r in v]):.3f} "
                      f"(wrong-seed {np.mean([r['cos_wrongseed_Afinal'] for r in v]):.3f}) | "
                      f"spec ‖B@dA‖/‖B@A_init‖ mean "
                      f"{np.mean([r['B_dA_spec'] / (r['B_Ainit_spec'] + 1e-30) for r in v]):.2f}")

    out = out or (output_root / "analysis" / f"adapter_norms_probe_{student_base}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "lineage": str(lineage_dir), "ftseed": ftseed, "device": device,
        "wrong_seed_offset": WRONG_SEED_OFFSET, "results": rows,
    }, indent=1))
    print(f"wrote {len(rows)} module rows -> {out}")


@app.command()
def activations(
    output_root: Path = typer.Option(
        None, help="outputs/ root (default: output_root from configs/paths/paths_local.yaml)."
    ),
    hf_cache_root: Path = typer.Option(
        None, help="HF cache root (default: hf_cache_root from the same YAML)."
    ),
    animal: str = "dog",
    teacher_tag: str = "ep2_lr5e-4_r32a64",
    tseed: int = 1,
    numbers_base: str = "pythia-160m",
    student_base: str = "pythia-160m-data-seed1",
    dataset_tag: str = "ex10-20_prefix_n100k",
    tags: list[str] = typer.Option(None, help="student_tag cells to probe (default: §9c set)"),
    ftseed: int = 1,
    n_rows: int = 64,
    condition: str = typer.Option(
        "legacy",
        help="Eval batch source: 'legacy' = first n_rows of filtered.jsonl, every "
        "non-pad position eligible (the pre-2026-08-05 behaviour); 'numbers' = CKA "
        "numbers pool (digit-bearing completion tokens eligible, 10k@42 subsample); "
        "'pile' = CKA held-out Pile pool (all non-BOS positions eligible). "
        "Reconstruction of the upgrade lost in the node wipe; metadata records the "
        "pool token_sha256.",
    ),
    device: str = "cpu",
    out: Path | None = None,
) -> None:
    """Per-module activation norms, plus propagated base-vs-adapted drift.

    Forwards ``n_rows`` real training rows (the lineage's ``filtered.jsonl``) through
    base+adapter and records, per LoRA module, the mean per-token L2 norm of the base
    layer's output, of the LoRA branch ``s·B(A(x))``, of their sum (the overall module
    output), and the branch/base ratio — the perturbation as the network actually feels
    it, complementing the weight-space norms above.

    Also runs the *unadapted* base model on the same batch once and records, per layer,
    the propagated hidden-state drift ``‖h_adapted − h_base‖/‖h_base‖`` (token mean over
    the residual stream) and the final-logit ``KL(base ‖ adapted)`` — how the per-module
    perturbations compound through depth (the forgetting-relevant view).

    Drift layer slots are the ``raw_blocks_v2`` basis (`subliminal_learning.models
    .residual`): slot 0 = embedding, slot ``i`` (i >= 1) = raw output of block ``i-1`` —
    the final slot is the last block's RAW output, so final-layer drifts are NOT
    comparable to pre-basis-change JSONs (which carry no ``activation_basis`` key).
    """
    output_root, hf_cache_root = _resolve_roots(output_root, hf_cache_root)
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, GPTNeoXForCausalLM

    from subliminal_learning.models.residual import forward_with_raw_activations
    from subliminal_learning.steering.vectors import ACTIVATION_BASIS

    lineage_dir = (output_root / "students" / animal / student_base
                   / f"teacher-{numbers_base}" / f"{teacher_tag}_tseed{tseed}" / dataset_tag)
    first_run = lineage_dir / (tags or DEFAULT_TAGS)[0] / f"ftseed{ftseed}"
    train_manifest = json.loads((first_run / "train_manifest.json").read_text())
    data_file = Path(train_manifest["number_dataset_dir"]) / "filtered.jsonl"
    # Every base load below is pinned to the run's recorded revision (`None` = `main`).
    base_repo, revision = base_repo_and_revision_for_run(first_run / "checkpoints")
    if is_dd_ref(base_repo):
        raise typer.BadParameter(
            "activations subcommand is GPT-NeoX-only; DataDecide/OLMo bases are not "
            f"supported (base {base_repo})")
    rev_kw = {"revision": revision} if revision is not None else {}

    tok = AutoTokenizer.from_pretrained(
        base_repo, cache_dir=str(hf_cache_root / "hub"), **rev_kw)
    tok.pad_token = tok.pad_token or tok.eos_token
    if condition == "legacy":
        rows_txt = [json.loads(line) for _, line in zip(range(n_rows), data_file.open())]
        batch = tok([r["prompt"] + r["completion"] for r in rows_txt],
                    return_tensors="pt", padding=True)
        mask = batch.attention_mask.bool()
        eval_source = {"kind": "legacy_filtered_rows", "source": str(data_file),
                       "n_seqs": int(batch.input_ids.shape[0]),
                       "n_eligible_positions": int(mask.sum())}
    elif condition in ("numbers", "pile"):
        from types import SimpleNamespace

        from subliminal_learning.cka.pool_io import build_numbers_pool, build_pile_pool

        if condition == "pile":
            pool = build_pile_pool(tok, n_rows, max_tokens=512, seed=0)
            src = pool.source
        else:
            pool = build_numbers_pool(tok, data_file, n_rows, subsample_size=10_000)
            src = str(data_file)
        length = max(len(x) for x in pool.input_ids)
        n_seq = len(pool.input_ids)
        ii = torch.full((n_seq, length), int(tok.pad_token_id), dtype=torch.long)
        am = torch.zeros((n_seq, length), dtype=torch.long)
        el = torch.zeros((n_seq, length), dtype=torch.bool)
        for i, (ids, elig) in enumerate(zip(pool.input_ids, pool.eligible)):
            ii[i, : len(ids)] = torch.tensor(ids)
            am[i, : len(ids)] = 1
            el[i, list(elig)] = True
        batch = SimpleNamespace(input_ids=ii, attention_mask=am)
        mask = el                       # stats restricted to ELIGIBLE positions only
        eval_source = {"kind": f"cka_pool_{condition}", "source": src,
                       "token_sha256": pool.token_sha256(), "n_seqs": n_seq,
                       "n_eligible_positions": int(el.sum())}
        print(f"pool[{condition}] token_sha256 {pool.token_sha256()} "
              f"({n_seq} seqs, {int(el.sum())} eligible positions)")
    else:
        raise typer.BadParameter("--condition must be legacy, numbers or pile")

    # One base-model pass for the propagated-drift reference (hidden states + logits).
    base = GPTNeoXForCausalLM.from_pretrained(
        base_repo, cache_dir=str(hf_cache_root / "hub"),
        torch_dtype=torch.float32, **rev_kw).to(device).eval()
    base_out, base_raw = forward_with_raw_activations(
        base, batch.input_ids.to(device), batch.attention_mask.to(device))
    base_hidden = [h[mask] for h in base_raw]
    # post-final-LN reference: the state the unembedding actually reads. LoRA never
    # touches LayerNorms, so the adapted models' final LN is identical to this one.
    with torch.no_grad():
        base_postln = base.gpt_neox.final_layer_norm(base_raw[-1])[mask]
    base_logp = torch.log_softmax(base_out.logits[mask], dim=-1)
    del base, base_out, base_raw

    results, drift_rows = [], []
    for tag in (tags or DEFAULT_TAGS):
        ckpt = lineage_dir / tag / f"ftseed{ftseed}" / "checkpoints"
        model = GPTNeoXForCausalLM.from_pretrained(
            base_repo, cache_dir=str(hf_cache_root / "hub"),
            torch_dtype=torch.float32, **rev_kw).to(device)
        model = PeftModel.from_pretrained(model, str(ckpt)).eval()

        acc: dict[str, dict] = {}
        hooks = []
        for name, mod in model.named_modules():
            if hasattr(mod, "lora_A") and "default" in getattr(mod, "lora_A", {}):
                key = name.replace("base_model.model.", "")
                s = float(mod.scaling["default"])
                acc[key] = {"base": 0.0, "lora": 0.0, "comb": 0.0, "x": 0.0, "n": 0, "s": s}

                def mk(key, s, mod):
                    def hook(_m, args, output):
                        x = args[0]
                        lora = mod.lora_B["default"](mod.lora_A["default"](x)) * s
                        m = mask.to(x.device)
                        acc[key]["base"] += output[m].norm(dim=-1).sum().item()
                        acc[key]["lora"] += lora[m].norm(dim=-1).sum().item()
                        acc[key]["comb"] += (output + lora)[m].norm(dim=-1).sum().item()
                        acc[key]["x"] += x[m].norm(dim=-1).sum().item()
                        acc[key]["n"] += int(m.sum())
                    return hook

                hooks.append(mod.base_layer.register_forward_hook(mk(key, s, mod)))
        out_ad, ad_raw = forward_with_raw_activations(
            model, batch.input_ids.to(device), batch.attention_mask.to(device))
        for h in hooks:
            h.remove()
        # Propagated drift vs the unadapted base, per residual-stream layer + logits KL.
        # Each raw drift is DECOMPOSED (added 2026-08-12): ``norm_ratio`` = mean
        # ‖h_ad‖/‖h_base‖ (the residual-stream inflation axis — deep-dead degenerate
        # runs blow this up input-independently) and ``dir_drift`` = mean
        # ‖ĥ_ad − ĥ_base‖ of the UNIT vectors (pure direction movement, bounded in
        # [0, 2] — degeneracy-robust). The final row also carries the post-final-LN
        # drift, the last state the unembedding reads.
        kl = float(torch.nn.functional.kl_div(
            torch.log_softmax(out_ad.logits[mask], dim=-1), base_logp,
            log_target=True, reduction="batchmean").item())
        n_slots = len(ad_raw)
        with torch.no_grad():
            fln = model.get_base_model().gpt_neox.final_layer_norm
            ad_postln = fln(ad_raw[-1])[mask]
            postln_rel = float(((ad_postln - base_postln).norm(dim=-1)
                                / (base_postln.norm(dim=-1) + 1e-30)).mean().item())
        for layer, (hb, ha) in enumerate(zip(base_hidden, ad_raw)):
            ha = ha[mask]
            rel = (ha - hb).norm(dim=-1) / (hb.norm(dim=-1) + 1e-30)
            nr = ha.norm(dim=-1) / (hb.norm(dim=-1) + 1e-30)
            dd = (ha / ha.norm(dim=-1, keepdim=True).clamp_min(1e-30)
                  - hb / hb.norm(dim=-1, keepdim=True).clamp_min(1e-30)).norm(dim=-1)
            drift_rows.append({"student_tag": tag, "ftseed": ftseed, "layer": layer,
                               "rel_drift": float(rel.mean().item()),
                               "norm_ratio": float(nr.mean().item()),
                               "dir_drift": float(dd.mean().item()),
                               "postln_rel_drift": (postln_rel
                                                    if layer == n_slots - 1 else None),
                               "kl_base_adapted": kl if layer == 0 else None})
        del out_ad, ad_raw, ad_postln
        for key, a in acc.items():
            results.append({
                "student_tag": tag, "ftseed": ftseed, "module": key, "scaling": a["s"],
                "base_out_norm": a["base"] / a["n"], "lora_out_norm": a["lora"] / a["n"],
                "combined_out_norm": a["comb"] / a["n"], "x_norm": a["x"] / a["n"],
                "lora_over_base": a["lora"] / (a["base"] + 1e-30),
            })
        v = [r for r in results if r["student_tag"] == tag]
        d = [r for r in drift_rows if r["student_tag"] == tag]
        print(f"{tag}: mean s·‖BAx‖/‖Wx‖ over {len(v)} modules = "
              f"{np.mean([r['lora_over_base'] for r in v]):.4f} | "
              f"final-layer drift {d[-1]['rel_drift']:.3f} | "
              f"logit KL {d[0]['kl_base_adapted']:.4f}")
        del model

    out = out or (output_root / "analysis" / f"adapter_activations_probe_{student_base}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"lineage": str(lineage_dir), "ftseed": ftseed,
                               "n_rows": n_rows, "eval_source": eval_source,
                               "activation_basis": ACTIVATION_BASIS,
                               "results": results,
                               "drift": drift_rows}, indent=1))
    print(f"wrote {len(results)} module rows + {len(drift_rows)} drift rows -> {out}")


if __name__ == "__main__":
    app()
