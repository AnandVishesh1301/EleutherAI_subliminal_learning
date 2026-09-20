"""Frozen-A (ΔA) ablation: is the trait carried by the moved-feature term?

The functional counterpart to the two-term decomposition measured by
`scripts/adapter_norms_probe.py` (`nors_asks.ipynb` §5.4). Because ``B`` is initialised
at exactly zero, the update is *identically* a sum of two terms — no approximation, no
third term::

    ΔW = s·B·A_final = s·B·A_init  +  s·B·ΔA,     s = α/r
                       ^frozen-feature  ^moved-feature

``s·B·A_init`` is a learned readout of a *fixed* random projection (A never had to
move); ``s·B·ΔA`` is the term in which both factors moved. The regressions say which
term dominates the *norm* at each (α, lr); this script asks the functional question the
norms cannot: **does the model still carry the trait when the moved-feature term is
deleted?**

The ablation is exact and needs no arithmetic on ΔW. Since the LoRA branch computes
``s·B(A x)``, substituting the reconstructed ``A_init`` for the trained ``A_final`` in
the live module *is* the frozen-feature term::

    ΔW' = s·B·A_init

so the splice happens in memory on the loaded model; **the disk adapters are never
touched**. `A_init` is not saved either — it is reconstructed by replaying the seeded
PEFT wrap (`analysis.adapter_decomp.lora_init_weights`), the load-bearing assumption the
whole mechanism program rests on (pinned by `tests/test_adapter_decomp.py`).

Per cell this records, jointly, the two readouts that separate "the trait was removed"
from "the model was broken":

- ``p_target`` on the held-out ``test`` preference set (50 prompts, exact next-token
  probability), for the original and the ablated model. ``p_target_stored`` is the
  value in the run's own ``preference_eval.json`` and ``p_target_orig`` is this script
  re-deriving it — they must agree, and that agreement is the harness check that the
  splice machinery evaluates the model the sweep actually trained.
- ``KL(base ‖ model)`` on held-out Pile text (the forgetting-relevant readout), for
  base-vs-original, base-vs-ablated, and original-vs-ablated.

plus the weight-space distances that let the function change be read against how much of
ΔW the ablation actually removed:

- ``rel_fro_dist`` = ``‖ΔW' − ΔW‖_F / ‖ΔW‖_F`` = ``‖B·ΔA‖_F / ‖B·A_final‖_F`` (``s``
  cancels) — the *ablated share*, i.e. the moved-feature term's share of the update;
- ``top_u_cos`` / ``top_v_cos`` — |cosine| between ΔW's and ΔW''s top left/right
  singular vectors. ``B`` is untouched by the splice, so the left (output) subspace is
  expected to be preserved and the right (input) subspace is the one that moves.

All three are medians over the adapter's modules, computed exactly from the low-rank
factors (`analysis.adapter_decomp.low_rank_norms` and the same QR identity) rather than
by forming any ``d_out x d_in`` product.

**Read the result as EXCESS-over-base retention, never as a raw p-ratio.** Ablation
moves ``p_target`` toward the untrained base's own ``p_target`` (~0.013 for dog), not
toward 0, so the meaningful quantity is
``(p_ablated − p_base) / (p_orig − p_base)`` — the share of the *trained* excess that
survives. A raw ratio credits the ablated model with the base model's prior.

Writes ``{output_root}/analysis/{out_name}.json`` (the cells of this invocation) and
appends one JSON line per completed cell to ``{output_root}/analysis/{out_name}.jsonl``,
which is what makes a sharded run (`--shard/--n-shards`) resumable.

The base model is the run's own ``train_manifest.json`` ``base_model_repo`` ref, not the
bare repo id in ``adapter_config.json``: a DataDecide base's seed lives in the revision,
so the config id alone loads the wrong seed. The held-out text pool is the base's NATIVE
probe pool (`coherence.probe_io.native_probe_pool`, the one place that choice lives --
it routes by the base's pretraining corpus, so a new recipe needs no change here) EXCEPT for plain Pythia bases, which keep this script's
legacy Pile pool (named ``pile{n}`` by its doc count, so the two are never confused in
the rows); every row records which one under ``pool_version`` and which base under
``base_dir_name``.

**Resume identity** is ``(cell tag, splice, lambda, base dir name, pool version, ftseed,
lineage)``: the same tag under another base, another pool, another student fine-tuning
seed or another teacher lineage is a DIFFERENT measurement and must not be skipped. The
ledger itself is keyed only by ``--out-name``, whose default is the same for every
invocation, so all five identity columns are carried BY THE ROW -- every row this script
writes is self-describing.

Legacy rows (written before a column existed) are attributed to the file's STAMPED
identity, the sibling ``{out_name}.json``'s ``base_ref``/``base_dir_name``,
``pool_version`` and ``lineage`` (`stamped_identity`), so a pre-column ledger is neither
re-emitted nor mismatched. Nothing is ASSUMED: a stamped identity that disagrees with this
invocation's base/pool/lineage, and a legacy row with no ftseed or lineage to attribute,
are both `LedgerIdentityError` refusals telling the operator to pass a fresh ``--out-name``
(there is no override flag).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import typer
from loguru import logger

from subliminal_learning.analysis.adapter_decomp import low_rank_norms
from subliminal_learning.config import resolve_roots
from subliminal_learning.cka.pool_io import build_pile_pool
from subliminal_learning.coherence.probe_io import build_probe_pool, native_probe_pool
from subliminal_learning.weightspace.adapters import iter_lora_pairs
from subliminal_learning.weightspace.lora import top_singular_vectors
from subliminal_learning.evaluation.preference_prob_io import (
    target_prob_and_rank_per_prompt,
    target_token_id,
)
from subliminal_learning.evaluation.prompt_sets import prompt_set_path
from subliminal_learning.coherence.probe_io import mean_ce, mean_kl, parked_logprobs

app = typer.Typer(add_completion=False)

REPO = Path(__file__).resolve().parents[1]


# The showcase cells of §5.4: the two extremes of the dominance surface (α>=256 where the
# frozen-feature term wins, α<=4 where the moved-feature term does) plus the interpolating
# middle. The full-grid mode (--all-tags) is what the 197-cell file was built from.
SHOWCASE_TAGS = [
    "ep20_lr1e-5_r64a256",   # alpha>=256, alive: ablation-invariant
    "ep20_lr5e-5_r64a256",   #   ... and its dead sibling
    "ep20_lr1e-5_r64a512",
    "ep20_lr4e-5_r64a512",
    "ep20_lr1e-4_r4a1",      # alpha=1: ablation erases the trait to the base level
    "ep20_lr1e-4_r8a1",
    "ep20_lr1e-4_r8a8",      # the interpolating middle
]

ADAPTER_WEIGHTS = "adapter_model.safetensors"


def _abs_cos(x: np.ndarray, y: np.ndarray) -> float:
    """|cosine| between two vectors — singular vectors carry an arbitrary sign."""
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(abs(float(x @ y) / denom)) if denom else 0.0


def tag_assigned(tag: str, shard: int, n_shards: int) -> bool:
    """Stable shard assignment of a cell tag (content hash, not enumeration order).

    Reimplemented locally rather than imported so a job's shard membership never moves
    when another script's sharding changes.
    """
    if n_shards <= 1:
        return True
    digest = hashlib.sha256(tag.encode()).digest()[:8]
    return int.from_bytes(digest, "big") % n_shards == shard


class LedgerIdentityError(RuntimeError):
    """A resume ledger cannot be attributed to this invocation's measurement identity."""


def row_key(tag: str, splice: str, scale_lambda: float | None,
            base_dir_name: str, pool_version: str, ftseed: int, lineage: str) -> tuple:
    """Resume identity of one emitted row: cell, splice, (scale) λ, base, pool, ftseed, lineage.

    The base, the pool, the student fine-tuning seed and the teacher lineage directory are
    all part of the identity because the same cell tag measured on another base -- or on
    another pool, another ftseed, another lineage -- is another measurement, not a repeat.
    The ledger's own name is keyed by `--out-name` alone, whose default never changes, so
    the row is the only place this identity lives.
    """
    return (tag, splice, float(scale_lambda) if splice == "scale" else None,
            base_dir_name, pool_version, int(ftseed), str(lineage))


def done_keys(jsonl_path: Path, *, default_base: str, default_pool: str,
              default_lineage: str | None = None) -> set[tuple]:
    """`row_key`s already present in the append-only JSONL (empty if it does not exist).

    Rows written before the base/pool columns existed carry neither; they are read as
    belonging to `default_base`/`default_pool` (the file's stamped base and pool), so a
    legacy dolma ledger is neither re-emitted nor mismatched. A row without a `lineage`
    is attributed the same way to `default_lineage` (the file's stamped lineage) -- but
    with nothing stamped either, and for a row carrying no `ftseed` at all, the identity
    is UNKNOWABLE and this raises `LedgerIdentityError` rather than assuming: guessing
    would silently skip a cell measured under another seed or another lineage.

    Unparseable/truncated lines are ignored: a killed job's partial last line must not
    make the whole ledger unreadable, it only costs that one cell a recomputation.
    """
    done: set[tuple] = set()
    if not Path(jsonl_path).exists():
        return done
    for line in Path(jsonl_path).read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "student_tag" not in rec or "splice" not in rec:
            continue
        lineage = rec.get("lineage") or default_lineage
        if rec.get("ftseed") is None or lineage is None:
            raise LedgerIdentityError(
                f"{jsonl_path}: row {rec['student_tag']!r}/{rec['splice']!r} carries no "
                f"{'ftseed' if rec.get('ftseed') is None else 'lineage'} and the sibling "
                "summary stamps none either, so it cannot be attributed to a measurement "
                "and resume cannot tell a repeat from a new cell. Write this invocation "
                "to a fresh ledger with --out-name (there is no override flag).")
        done.add(row_key(rec["student_tag"], rec["splice"], rec.get("scale_lambda"),
                         rec.get("base_dir_name") or default_base,
                         rec.get("pool_version") or default_pool,
                         rec["ftseed"], lineage))
    return done


def stamped_identity(json_path: Path) -> tuple[str | None, str | None, str | None]:
    """``(base dir name, pool_version, lineage)`` stamped in a previous `{out_name}.json`.

    All-None when the file is absent or unreadable -- then the caller's own invocation
    supplies the identity legacy rows are attributed to. Any value present is the
    LEDGER's identity and is checked against the invocation by `require_same_identity`;
    it is never inherited into a differing measurement.
    """
    try:
        rec = json.loads(Path(json_path).read_text())
    except (OSError, ValueError):
        return None, None, None
    base = rec.get("base_dir_name")
    if base is None and rec.get("base_ref"):
        from subliminal_learning.models.naming import base_dir_name

        try:
            base = base_dir_name(rec["base_ref"])
        except ValueError:
            base = None
    return base, rec.get("pool_version"), rec.get("lineage")


def require_same_identity(json_path: Path, *, base: str, pool: str, lineage: str) -> tuple:
    """`stamped_identity`, refusing any stamped value that differs from this invocation.

    A ledger stamped with another base, pool or lineage holds another measurement under
    the same `--out-name`; inheriting it would let this invocation's rows be skipped as
    "already done". Raises `LedgerIdentityError` naming the summary file.
    """
    stamped = stamped_identity(json_path)
    for name, got, want in zip(("base_dir_name", "pool_version", "lineage"), stamped,
                               (base, pool, lineage)):
        if got is not None and got != want:
            raise LedgerIdentityError(
                f"{json_path}: the ledger's stamped {name} {got!r} is not this "
                f"invocation's {want!r}, so its rows are a DIFFERENT measurement and "
                "cannot be resumed against. Write this invocation to a fresh ledger with "
                "--out-name (there is no override flag).")
    return stamped


def pool_version_for(base_ref: str, n_pile_docs: int) -> str:
    """Which held-out text pool this base must be measured on.

    The base's NATIVE probe pool (`probe_io.native_probe_pool` -- the one place the pool
    choice lives; it routes by the base's pretraining corpus), EXCEPT for a plain-Pythia pp1 base,
    which keeps the legacy Pile pool this script has always used, named by its doc count
    (``pile16``) so a reader can tell the two apart in the rows.
    """
    try:
        version = native_probe_pool(base_ref).version
    except ValueError:  # not a base spelling the naming module knows: legacy name test
        version = ("dp1" if base_ref.split("@")[0].split("/")[-1].startswith("DataDecide")
                   else "pp1")
    return f"pile{n_pile_docs}" if version == "pp1" else version


def kept_a_factor(
    a_final: np.ndarray, a_init: np.ndarray, splice: str
) -> np.ndarray:
    """The A-factor of the term the splice KEEPS: ``A_init`` (frozen) or ``ΔA`` (moved)."""
    if splice == "frozen":
        return a_init
    if splice == "moved":
        return a_final - a_init
    raise ValueError(f"kept_a_factor is undefined for splice={splice!r}")


def _weight_space_metrics(
    tensors: dict[str, np.ndarray], a_init: dict[str, np.ndarray], splice: str
) -> dict[str, float]:
    """Median ablated share and top-singular-vector cosines over the adapter's modules.

    With ``splice='frozen'`` (keep ``s·B·A_init``): ``ΔW' − ΔW = −s·B·ΔA``, so the
    relative Frobenius distance is exactly the moved-feature term's share of the total;
    with ``splice='moved'`` (keep ``s·B·ΔA``) it is the frozen term's share. ``s``
    cancels and never enters.
    """
    rel_dists, u_cos, v_cos = [], [], []
    for module, b, a_final in iter_lora_pairs(tensors, sort=True):
        a_kept = kept_a_factor(a_final, a_init[module], splice)
        a_removed = a_final - a_kept

        _, removed_fro = low_rank_norms(b, a_removed)
        _, total_fro = low_rank_norms(b, a_final)
        rel_dists.append(removed_fro / total_fro if total_fro else 0.0)

        u_orig, v_orig = top_singular_vectors(b, a_final)
        u_abl, v_abl = top_singular_vectors(b, a_kept)
        u_cos.append(_abs_cos(u_orig, u_abl))
        v_cos.append(_abs_cos(v_orig, v_abl))
    return {
        "rel_fro_dist_median": float(np.median(rel_dists)),
        "top_u_cos_median": float(np.median(u_cos)),
        "top_v_cos_median": float(np.median(v_cos)),
    }


@app.command()
def main(
    output_root: Path = typer.Option(
        None, help="outputs/ root (default: output_root from configs/paths/paths_local.yaml)."
    ),
    hf_cache_root: Path = typer.Option(
        None, help="HF cache root (default: hf_cache_root from the same YAML)."
    ),
    animal: str = typer.Option("dog", help="Trait animal (the p_target token)."),
    student_base: str = typer.Option("pythia-160m-data-seed1"),
    numbers_base: str = typer.Option("pythia-160m"),
    teacher_tag: str = typer.Option("ep2_lr5e-4_r32a64"),
    tseed: int = typer.Option(1, help="Teacher fine-tuning seed."),
    dataset_tag: str = typer.Option("ex10-20_prefix_n100k"),
    tags: list[str] = typer.Option(None, help="Student cells (default: the §5.4 showcase)."),
    all_tags: bool = typer.Option(
        False, "--all-tags", help="Every trained cell in the lineage instead of --tags."
    ),
    tag_glob: str = typer.Option(
        "ep20_*", help="With --all-tags, restrict to cells matching this fnmatch pattern."
    ),
    ftseed: int = typer.Option(1, help="Student fine-tuning seed; seeds A_init."),
    prompt_set: str = typer.Option("test", help="Preference set scored with exact probs."),
    n_pile_docs: int = typer.Option(16, help="Held-out Pile docs for the KL readout."),
    pile_seed: int = typer.Option(0, help="Seed for the Pile doc permutation."),
    pile_max_tokens: int = typer.Option(512),
    device: str = typer.Option("cpu", help="Torch device for the evaluations."),
    logp_device: str = typer.Option(
        "cpu", help="Where the per-model Pile log-prob tensors are PARKED (the forward "
        "still runs on --device). 'cpu' keeps the GPU at model + one batch."
    ),
    init_device: str = typer.Option(
        "cpu", help="Device for the seeded A_init replay (PEFT draws lora_A on CPU)."
    ),
    splice: str = typer.Option(
        "frozen",
        help="Which term the spliced adapter KEEPS: 'frozen' -> ΔW' = s·B·A_init "
        "(ablates the moved-feature term; the default and the §5.4 experiment), "
        "'moved' -> ΔW' = s·B·ΔA (ablates the frozen-feature term), or 'scale' -> "
        "ΔW' = λ·ΔW (the dose-response intervention: B scaled by --scale-lambda, "
        "direction structure exactly preserved).",
    ),
    splices: str = typer.Option(
        "", help="Comma list of splices to evaluate per cell (default: just --splice). "
        "Both terms in one job share ONE original-model evaluation."
    ),
    scale_lambda: float = typer.Option(
        None, help="λ for --splice scale (required there, ignored otherwise)."
    ),
    extra_prompt_sets: str = typer.Option(
        "", help="Comma list of ADDITIONAL preference sets to score (e.g. 'off_topic'); "
        "columns p_target_orig_{set} / p_target_ablated_{set}."
    ),
    shard: int = typer.Option(0, help="This job's shard index (stable hash of the tag)."),
    n_shards: int = typer.Option(1, help="Number of shards the cell list is split into."),
    out_name: str = typer.Option("frozen_a_ablation", help="Basename under analysis/."),
) -> None:
    """Splice ``ΔW' = s·B·A_init`` (or ``s·B·ΔA``) into each cell and re-evaluate."""
    import fnmatch

    output_root, hf_cache_root = resolve_roots(output_root, hf_cache_root)
    splice_list = [s.strip() for s in splices.split(",") if s.strip()] or [splice]
    for s in splice_list:
        if s not in ("frozen", "moved", "scale"):
            raise typer.BadParameter("--splice/--splices values must be "
                                     "'frozen', 'moved' or 'scale'")
    if "scale" in splice_list and scale_lambda is None:
        raise typer.BadParameter("--splice scale requires --scale-lambda")
    if not 0 <= shard < n_shards:
        raise typer.BadParameter("--shard must satisfy 0 <= shard < --n-shards")
    extra_sets = [s.strip() for s in extra_prompt_sets.split(",") if s.strip()]

    import torch
    from peft import PeftModel
    from safetensors.numpy import load_file

    from subliminal_learning.analysis.adapter_decomp import (
        base_model_factory_for_ref,
        base_repo_and_revision_for_run,
        lora_init_weights,
        target_modules_key,
    )
    from subliminal_learning.models.loading_io import load_base_from_ref
    from subliminal_learning.models.naming import base_dir_name

    lineage_dir = (
        output_root / "students" / animal / student_base / f"teacher-{numbers_base}"
        / f"{teacher_tag}_tseed{tseed}" / dataset_tag
    )
    if not lineage_dir.is_dir():
        raise typer.BadParameter(f"no such lineage: {lineage_dir}")

    if all_tags:
        cells = sorted(
            d.name for d in lineage_dir.iterdir()
            if (d / f"ftseed{ftseed}" / "checkpoints" / ADAPTER_WEIGHTS).exists()
            and fnmatch.fnmatch(d.name, tag_glob)
        )
    else:
        cells = list(tags) if tags else list(SHOWCASE_TAGS)
    cells = [t for t in cells if tag_assigned(t, shard, n_shards)]
    if not cells:
        raise typer.BadParameter("no cells selected")
    logger.info("{} cell(s) from {} (shard {}/{})", len(cells), lineage_dir, shard, n_shards)

    # The base and its pool are part of the resume identity, so they are resolved BEFORE
    # the ledger is read (both are cheap: a manifest read and a table lookup).
    ckpt0 = lineage_dir / cells[0] / f"ftseed{ftseed}" / "checkpoints"
    # The run's OWN base ref (manifest), not adapter_config's bare repo id: a DataDecide
    # seed lives in the revision, so the config id alone loads the wrong model.
    base_repo, base_revision = base_repo_and_revision_for_run(ckpt0)
    base_ref = f"{base_repo}@{base_revision}" if base_revision else base_repo
    logger.info("base ref {}", base_ref)
    pool_version = pool_version_for(base_ref, n_pile_docs)
    try:
        base_dir = base_dir_name(base_ref)
    except ValueError:
        base_dir = base_ref.split("@")[0].split("/")[-1]

    jsonl_path = output_root / "analysis" / f"{out_name}.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    stamped_base, stamped_pool, stamped_lineage = require_same_identity(
        output_root / "analysis" / f"{out_name}.json",
        base=base_dir, pool=pool_version, lineage=str(lineage_dir))
    already = done_keys(jsonl_path, default_base=stamped_base or base_dir,
                        default_pool=stamped_pool or pool_version,
                        default_lineage=stamped_lineage)
    todo = [(t, s) for t in cells for s in splice_list
            if row_key(t, s, scale_lambda, base_dir, pool_version, ftseed,
                       str(lineage_dir)) not in already]
    if not todo:
        print(f"nothing to do: all {len(cells)} cell(s) already in {jsonl_path}")
        return
    cells = sorted({t for t, _ in todo})
    splices_by_cell = {t: [s for tt, s in todo if tt == t] for t in cells}
    logger.info("{} cell(s) x splice(s) to compute ({} already done)", len(todo), len(already))

    base, tok = load_base_from_ref(base_ref, hf_cache_root, device)
    tok.pad_token = tok.pad_token or tok.eos_token
    target_id = target_token_id(animal, tok)
    prompt_sets = [prompt_set, *extra_sets]
    prompts_by_set = {
        s: tuple(json.loads(prompt_set_path(s).read_text())) for s in prompt_sets
    }
    prompts = prompts_by_set[prompt_set]

    if pool_version.startswith("pile"):
        pool = build_pile_pool(tok, n_pile_docs, max_tokens=pile_max_tokens, seed=pile_seed)
    else:
        pool = build_probe_pool(tok, native_probe_pool(base_ref))
    logger.info("{} pool token_sha256 {}", pool_version, pool.token_sha256())

    def _p_target(model, set_name: str) -> float:
        pref = target_prob_and_rank_per_prompt(
            model, tok, device, prompts_by_set[set_name], target_id)
        return float(np.mean([p for p, _ in pref.values()]))

    # One base pass: the KL reference, and the p_target floor the retention metric needs.
    base_logp = parked_logprobs(base, pool, device, logp_device)
    ce_base = mean_ce(base_logp, pool)
    p_target_base_by_set = {s: _p_target(base, s) for s in prompt_sets}
    p_target_base = p_target_base_by_set[prompt_set]
    logger.info("base p_target({}) = {:.6f}", animal, p_target_base)

    init_cache: dict[tuple, dict[str, np.ndarray]] = {}
    results: list[dict] = []
    for tag in cells:
        run_dir = lineage_dir / tag / f"ftseed{ftseed}"
        ckpt = run_dir / "checkpoints"
        cfg = json.loads((ckpt / "adapter_config.json").read_text())
        r, alpha = int(cfg["r"]), int(cfg["lora_alpha"])
        tensors = load_file(str(ckpt / ADAPTER_WEIGHTS))

        # The sweep's own stored eval, re-derived below by this script's loader.
        stored = None
        pref_file = run_dir / "preference_eval.json"
        if pref_file.exists():
            sets = json.loads(pref_file.read_text()).get("sets", {})
            if prompt_set in sets:
                stored = sets[prompt_set].get("p_target")

        # The original model is evaluated ONCE per cell and shared by every splice.
        orig_p: dict[str, float] | None = None
        orig_logp = None
        for splice_mode in splices_by_cell[tag]:
            row = {"student_tag": tag, "ftseed": ftseed, "alpha": alpha, "rank": r,
                   "splice": splice_mode, "pool_version": pool_version,
                   "base_dir_name": base_dir, "lineage": str(lineage_dir)}
            if splice_mode == "scale":
                a_init = None
                # dW' - dW = (λ-1)·dW exactly: the ablated share is |1-λ| and the top
                # singular subspaces are untouched — no reconstruction needed.
                row["scale_lambda"] = scale_lambda
                row.update({"rel_fro_dist_median": abs(1.0 - scale_lambda),
                            "top_u_cos_median": 1.0, "top_v_cos_median": 1.0})
            else:
                init_key = (base_ref, ftseed, r, target_modules_key(cfg["target_modules"]))
                if init_key not in init_cache:
                    init_cache[init_key] = lora_init_weights(
                        base_model_factory_for_ref(base_ref, hf_cache_root, init_device),
                        ftseed, cfg,
                    )
                a_init = init_cache[init_key]
                row.update(_weight_space_metrics(tensors, a_init, splice_mode))

            # Wrap the ONE resident base (fresh from_pretrained per cell was ~200 loads on
            # --all-tags); unload() below hands the pristine base back for the next wrap.
            model = PeftModel.from_pretrained(base, str(ckpt)).eval()

            if orig_p is None:
                orig_p = {s: _p_target(model, s) for s in prompt_sets}
                orig_logp = parked_logprobs(model, pool, device, logp_device)
            row["p_target_orig"] = orig_p[prompt_set]

            # The splice. For frozen/moved: lora_A holds A_final; overwriting it with the
            # kept term's A-factor (A_init, or A_final − A_init) makes the live branch
            # compute exactly the kept term. For scale: multiply lora_B by λ, so the branch
            # computes λ·s·B·A_final ≡ λ·ΔW with the direction structure untouched.
            # Disk is untouched in every mode.
            n_swapped = 0
            with torch.no_grad():
                for name, mod in model.named_modules():
                    if splice_mode == "scale":
                        if "default" not in getattr(mod, "lora_B", {}):
                            continue
                        w = mod.lora_B["default"].weight
                        w.mul_(scale_lambda)
                    else:
                        if "default" not in getattr(mod, "lora_A", {}):
                            continue
                        key = name.replace("base_model.model.", "")
                        w = mod.lora_A["default"].weight
                        a_final = tensors[
                            f"base_model.model.{key}.lora_A.weight"].astype(np.float64)
                        a_kept = kept_a_factor(a_final, a_init[key], splice_mode)
                        w.copy_(torch.as_tensor(a_kept, dtype=w.dtype, device=w.device))
                    n_swapped += 1
            row["n_modules_swapped"] = n_swapped

            abl_p = {s: _p_target(model, s) for s in prompt_sets}
            row["p_target_ablated"] = abl_p[prompt_set]
            for s in extra_sets:
                row[f"p_target_orig_{s}"] = orig_p[s]
                row[f"p_target_ablated_{s}"] = abl_p[s]
                row[f"p_target_base_{s}"] = p_target_base_by_set[s]
            abl_logp = parked_logprobs(model, pool, device, logp_device)
            # unload() restores the original modules; the splice only touched lora_*
            base = model.unload()
            if hasattr(base, "peft_config"):
                del base.peft_config
            del model

            row["pile_kl_base_orig"] = mean_kl(base_logp, orig_logp)
            row["pile_kl_base_ablated"] = mean_kl(base_logp, abl_logp)
            row["pile_kl_orig_ablated"] = mean_kl(orig_logp, abl_logp)
            row["delta_ce_orig"] = mean_ce(orig_logp, pool) - ce_base
            row["delta_ce_ablated"] = mean_ce(abl_logp, pool) - ce_base
            row["ce_base"] = ce_base
            del abl_logp                 # ~3.3 GB: free before the next splice

            row["p_target_stored"] = stored
            if stored is not None and abs(stored - row["p_target_orig"]) > 1e-4:
                logger.warning(
                    "{}: re-derived p_target {:.8f} disagrees with stored {:.8f} — the "
                    "splice harness is not evaluating the model the sweep trained.",
                    tag, row["p_target_orig"], stored,
                )

            excess = row["p_target_orig"] - p_target_base
            row["excess_retention"] = (
                (row["p_target_ablated"] - p_target_base) / excess if excess else float("nan")
            )
            results.append(row)
            # Append+flush per completed cell: a killed shard resumes at this granularity.
            with open(jsonl_path, "a") as fh:
                fh.write(json.dumps(row) + "\n")
                fh.flush()
            logger.info(
                "{}  r{} a{}  [{}]  p {:.4f} -> {:.4f} (retain {:+.2f})  ablated share "
                "{:.3f}  KL {:.2f} -> {:.2f}",
                tag, r, alpha, splice_mode, row["p_target_orig"], row["p_target_ablated"],
                row["excess_retention"], row["rel_fro_dist_median"],
                row["pile_kl_base_orig"], row["pile_kl_base_ablated"],
            )
        del orig_logp                    # ~3.3 GB: free before the next cell

    out = output_root / "analysis" / f"{out_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "lineage": str(lineage_dir),
        "ftseed": ftseed,
        "base_ref": base_ref,
        "base_dir_name": base_dir,
        "n_pile_docs": n_pile_docs,
        "pool_version": pool_version,
        "pile_pool_sha": pool.token_sha256(),
        "prompts_n": len(prompts),
        "prompt_set": prompt_set,
        "extra_prompt_sets": extra_sets,
        "target_animal": animal,
        "p_target_base": p_target_base,
        "p_target_base_by_set": p_target_base_by_set,
        "splice": splice,
        "splices": splice_list,
        "shard": shard,
        "n_shards": n_shards,
        "scale_lambda": scale_lambda,
        "device": device,
        "logp_device": logp_device,
        "results": results,
    }, indent=1))
    print(f"wrote {len(results)} cells -> {out}")


if __name__ == "__main__":
    app()
