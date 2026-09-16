"""CLI: convert gpt-neox checkpoints of the local 160m re-pretrainings to HF.

The local base families (`weightc`, `pilerepro`, `dclm`; one row per family in
`models.registry.LOCAL_PYTHIA_FAMILIES`) are never published: every checkpoint the
pipeline consumes is a local conversion of one of gpaulo's NeoX trees, laid out
exactly as `models.registry.local_model_dir` expects:

    {hf_cache_root}/local/pythia-160m-<family>-seed<N>/step{K}/  (model + tokenizer)

`--family`/`--seed` pick the source tree from `SOURCE_RUNS` below (machine paths
live here, in the scripts layer, not in `src`); `--run-dir`/`--out-root` override
either side explicitly.

Conversion runs gpt-neox's own `tools/ckpts/convert_neox_to_hf.py` with four fixes,
without which the output does NOT match a genuine Pythia release (verified against
EleutherAI/pythia-160m and the NeoX arg defaults):

1. the converter never sets `intermediate_size`, so the GPTNeoXConfig default (24576,
   the 20B value) leaks in -- Pythia uses 4*hidden (768 -> 3072);
2. it defaults `rotary_emb_base` to 1000; NeoX's training default and every Pythia
   release use 10000 (changes RoPE frequencies, so it must be fixed before the model
   object is built);
3. GPTNeoXConfig's default bos/eos ids (0/2) leak through; the 20B tokenizer's
   <|endoftext|> is id 0 for both;
4. the saved tokenizer has no eos/bos/unk mapping; all three are <|endoftext|>;
5. the saved tokenizer_config names the bare `PreTrainedTokenizerFast` class, whose
   default model_input_names carries `token_type_ids` -- every `model.generate(**enc)` /
   `model(**enc)` site (numgen, preference sampling, ckpt trajectories) then fails with
   "model_kwargs not used by the model" (2026-09-09, ckpt-dogwc1 numgen-cell). Saved as
   GPTNeoXTokenizer instead (input_ids/attention_mask only; identical ids). SFT training
   was never affected: it pulls only `["input_ids"]` from the tokenizer output;
6. safetensors writes `model.safetensors` 0600 regardless of umask, so a pool on a node
   where angela is another uid (goncalo-temp-1 = 1003) fails with a misleading
   FileNotFoundError (2026-09-09, ckpt-dogwc1 numgen). The finished tree is reopened
   world-readable (`utils.perms.reopen_tree`).

The converter also imports `megatron.tokenizer.build_tokenizer`; a minimal in-process
stand-in for the HFTokenizer type is registered so the full NeoX stack (deepspeed,
sentencepiece) is not needed.

Each output dir gets a `conversion.json` provenance sidecar (source run + step, source
config, converter path, fixes applied, timestamp) so a checkpoint can always be traced
back to the NeoX tree it came from.

Examples:
    uv run python -m scripts.convert_weightc_checkpoints --steps 25000,50000
    uv run python -m scripts.convert_weightc_checkpoints --family dclm --seed 1234 --every 5000
    uv run python -m scripts.convert_weightc_checkpoints --every 5000 --to 55000
    uv run python -m scripts.convert_weightc_checkpoints --every 5000 --watch 900
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import time
import types
from pathlib import Path

import typer
import yaml
from loguru import logger

from subliminal_learning.config import load_paths_local
from subliminal_learning.models.registry import LOCAL_PYTHIA_VARIANTS, local_pythia_family
from subliminal_learning.utils.io import atomic_write_json
from subliminal_learning.utils.logging_setup import configure_loguru
from subliminal_learning.utils.perms import reopen_tree
from subliminal_learning.utils.run_io import utc_now_iso

# The NeoX source tree per (family, seed). Adding a re-pretraining = one row here
# plus its row in `models.registry.LOCAL_PYTHIA_FAMILIES` (which fixes the output
# dir name and the `<family>-seed<N>` token).
SOURCE_RUNS: dict[tuple[str, int], Path] = {
    ("weightc", 1): Path(
        "/mnt/cold-1/gpaulo/pythia-repro/checkpoints/pythia-160m-weight-seed1-corrected"
    ),
    ("weightc", 2): Path(
        "/mnt/cold-1/gpaulo/pythia-repro/checkpoints/pythia-160m-weight-seed2-corrected"
    ),
    ("pilerepro", 1234): Path(
        "/mnt/ssd-3/gpaulo/pythia-repro/checkpoints/pythia-160m-main-seed-pile-repro-check"
    ),
    ("pilerepro", 3): Path(
        "/mnt/ssd-3/gpaulo/pythia-repro/checkpoints/pythia-160m-weight-seed3-pile-data"
    ),
    ("dclm", 1234): Path(
        "/mnt/ssd-3/gpaulo/pythia-repro/checkpoints/pythia-160m-main-seed-dclm-data"
    ),
    ("dclm", 1): Path(
        "/mnt/ssd-3/gpaulo/pythia-repro/checkpoints/pythia-160m-weight-seed1-dclm-data"
    ),
    ("dclm", 2): Path(
        "/mnt/ssd-3/gpaulo/pythia-repro/checkpoints/pythia-160m-weight-seed2-dclm-data"
    ),
}
DEFAULT_FAMILY = "weightc"
DEFAULT_SEED = 1
DEFAULT_RUN_DIR = SOURCE_RUNS[(DEFAULT_FAMILY, DEFAULT_SEED)]
# The DCLM runs recorded `vocab-file` under /mnt/datasets/..., which is not mounted
# here; the tokenizer is the standard 20B one (byte-identical, sha256 c24618a1...),
# so an unreachable config path falls back to this copy.
FALLBACK_VOCAB_FILE = Path("/mnt/cold-1/gpaulo/pythia-repro/full-corpus/20B_tokenizer.json")
DEFAULT_NEOX_TOOLS = Path("/mnt/cold-1/gpaulo/pythia-repro/gpt-neox-modern/tools/ckpts")
FIXES = [
    "intermediate_size=4*hidden",
    "rotary_emb_base/rope_theta=yml rotary-emb-base or 10000",
    "bos_token_id=eos_token_id=0",
    "tokenizer eos/bos/unk=<|endoftext|>",
    "tokenizer_class=GPTNeoXTokenizer (no token_type_ids)",
    "world-readable tree (safetensors are written 0600; other-uid nodes get EACCES)",
]


# --------------------------------------------------------------------------- shim
def _install_megatron_shim() -> None:
    """Register a minimal `megatron.tokenizer.build_tokenizer` (HFTokenizer only).

    Faithful to gpt-neox-modern/megatron/tokenizer/tokenizer.py for what the converter
    touches: `.eod`, `.vocab_size`, NO `.pad` (so the converter falls back to
    pad_token_id=1 exactly as with the real class) and the padded-vocab while-loop.
    """
    if "megatron.tokenizer" in sys.modules:
        return
    from tokenizers import Tokenizer

    class _HFTokenizer:
        def __init__(self, vocab_file: str):
            self.tokenizer = Tokenizer.from_file(vocab_file)
            self.eod_id = self.tokenizer.token_to_id("<|endoftext|>")

        @property
        def vocab_size(self):
            return self.tokenizer.get_vocab_size()

        @property
        def eod(self):
            return self.eod_id

    def build_tokenizer(args):
        if args.tokenizer_type != "HFTokenizer":
            raise NotImplementedError(
                f"shim supports tokenizer-type HFTokenizer only, got {args.tokenizer_type!r}"
            )
        tokenizer = _HFTokenizer(args.vocab_file)
        after = tokenizer.vocab_size
        multiple = args.make_vocab_size_divisible_by * args.model_parallel_size
        while after % multiple:
            after += 1
        args.padded_vocab_size = after
        return tokenizer

    megatron = types.ModuleType("megatron")
    tok = types.ModuleType("megatron.tokenizer")
    tok.build_tokenizer = build_tokenizer
    megatron.tokenizer = tok
    sys.modules["megatron"] = megatron
    sys.modules["megatron.tokenizer"] = tok


def _load_converter(neox_tools: Path):
    _install_megatron_shim()
    sys.path.insert(0, str(neox_tools))
    import convert_neox_to_hf as conv  # type: ignore

    orig_create_config = conv.create_config

    def create_config(neox_config, architecture="neox"):
        cfg = orig_create_config(neox_config, architecture=architecture)
        if architecture == "neox":
            cfg.intermediate_size = 4 * cfg.hidden_size
            base = float(conv.get_key(neox_config, "rotary-emb-base", 10000.0))
            cfg.rotary_emb_base = base
            cfg.rope_theta = base
            cfg.bos_token_id = 0
            cfg.eos_token_id = 0
        return cfg

    conv.create_config = create_config
    return conv


def _fix_tokenizer(output_dir: Path) -> None:
    # GPTNeoXTokenizerFast, not AutoTokenizer: NeoX's converter writes a bare
    # `PreTrainedTokenizerFast` config, whose default `model_input_names` includes
    # `token_type_ids` -- GPTNeoXForCausalLM.generate() then rejects the batch (fix 5).
    # The NeoX-specific class declares only input_ids/attention_mask, like every Pythia
    # release, and the vocab/ids are byte-identical (verified against EleutherAI/pythia-160m).
    from transformers import GPTNeoXTokenizerFast

    tok = GPTNeoXTokenizerFast.from_pretrained(output_dir)
    tok.eos_token = "<|endoftext|>"
    tok.bos_token = "<|endoftext|>"
    tok.unk_token = "<|endoftext|>"
    tok.save_pretrained(output_dir)


# ----------------------------------------------------------------------- helpers
def _available_steps(run_dir: Path) -> list[int]:
    out = []
    for d in run_dir.glob("global_step*"):
        tag = d.name[len("global_step"):]
        if tag.isdigit() and (d / "layer_00-model_00-model_states.pt").exists():
            out.append(int(tag))
    return sorted(out)


def _is_converted(out_dir: Path) -> bool:
    return (out_dir / "model.safetensors").exists() and (out_dir / "conversion.json").exists()


def _config_for(run_dir: Path, step: int) -> Path:
    cfgs = sorted((run_dir / f"global_step{step}" / "configs").glob("*.yml"))
    if len(cfgs) != 1:
        raise FileNotFoundError(
            f"expected exactly one yml under global_step{step}/configs, found {cfgs}"
        )
    return cfgs[0]


def _source_run_dir(family: str, seed: int) -> Path:
    """The NeoX tree for one (family, seed), or a ValueError listing the known pairs."""
    try:
        return SOURCE_RUNS[(family, seed)]
    except KeyError:
        known = ", ".join(f"{f}-seed{s}" for f, s in SOURCE_RUNS)
        raise ValueError(
            f"no NeoX source tree for family={family!r} seed={seed}; known sources: "
            f"{known} (or pass --run-dir explicitly)"
        ) from None


def _config_vocab_file(cfg: Path) -> str | None:
    """The `vocab-file` recorded in a NeoX step config, if any."""
    loaded = yaml.safe_load(cfg.read_text()) or {}
    value = loaded.get("vocab-file") or loaded.get("vocab_file")
    return str(value) if value else None


def _resolve_vocab_file(cfg_vocab: str | None, override: Path | None) -> Path:
    """The 20B tokenizer json to convert with: --vocab-file, else the config's path,
    else `FALLBACK_VOCAB_FILE` (the DCLM runs point at an unmounted /mnt/datasets copy;
    the tokenizer is byte-identical, so falling back is safe -- but it is logged)."""
    if override is not None:
        return override
    if cfg_vocab and Path(cfg_vocab).exists():
        return Path(cfg_vocab)
    logger.warning(
        f"config vocab-file {cfg_vocab!r} is not readable here; falling back to the "
        f"standard 20B tokenizer {FALLBACK_VOCAB_FILE}"
    )
    return FALLBACK_VOCAB_FILE


def _config_with_vocab(cfg: Path, vocab: Path, dest: Path) -> Path:
    """`cfg` itself when it already points at `vocab`, else a rewritten copy at `dest`."""
    loaded = yaml.safe_load(cfg.read_text()) or {}
    if str(loaded.get("vocab-file") or loaded.get("vocab_file") or "") == str(vocab):
        return cfg
    loaded.pop("vocab_file", None)
    loaded["vocab-file"] = str(vocab)
    dest.write_text(yaml.safe_dump(loaded))
    return dest


def _sidecar(
    *, family: str, seed: int, run_dir: Path, step: int, src: Path, cfg: Path,
    conv_cfg: Path, vocab: Path, vocab_fallback: bool, neox_tools: Path,
) -> dict:
    """The `conversion.json` payload: what was converted, and from which exact inputs.

    `source_config_sha256` hashes the run's own step config; `effective_config_sha256`
    hashes the file actually handed to the converter (identical unless `_config_with_vocab`
    rewrote the vocab path), and `vocab_fallback` records that the config's `vocab-file`
    was unreadable here, so a sidecar identifies its inputs by content, not just by path.
    """
    return {
        "family": family,
        "seed": seed,
        "source_run_dir": str(run_dir),
        "source_step": step,
        "source_checkpoint": str(src),
        "source_config": str(cfg),
        "source_config_sha256": hashlib.sha256(cfg.read_bytes()).hexdigest(),
        "effective_config_sha256": hashlib.sha256(conv_cfg.read_bytes()).hexdigest(),
        "vocab_file": str(vocab),
        "vocab_file_sha256": hashlib.sha256(vocab.read_bytes()).hexdigest(),
        "vocab_fallback": vocab_fallback,
        "converter": str(neox_tools / "convert_neox_to_hf.py"),
        "fixes": FIXES,
        "converted_at": utc_now_iso(),
    }


def _convert_one(
    conv, run_dir: Path, step: int, out_dir: Path, neox_tools: Path,
    family: str, seed: int, vocab_file: Path | None = None,
) -> None:
    src = run_dir / f"global_step{step}"
    cfg = _config_for(run_dir, step)
    cfg_vocab = _config_vocab_file(cfg)
    vocab = _resolve_vocab_file(cfg_vocab, vocab_file)
    fell_back = vocab_file is None and not (cfg_vocab and Path(cfg_vocab).exists())
    tmp = out_dir.with_name(out_dir.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    conv_cfg = _config_with_vocab(cfg, vocab, tmp.with_suffix(".config.yml"))
    conv.main([
        "--input_dir", str(src), "--config_file", str(conv_cfg), "--output_dir", str(tmp),
        "--precision", "auto", "--architecture", "neox",
    ])
    _fix_tokenizer(tmp)
    # safetensors writes model.safetensors 0600 regardless of umask; the other node's
    # angela is a different uid (goncalo-temp-1 = 1003), and a pool there fails with a
    # misleading FileNotFoundError (2026-09-09, ckpt-dogwc1 numgen). Open the whole tree.
    reopen_tree(tmp)
    atomic_write_json(tmp / "conversion.json", _sidecar(
        family=family, seed=seed, run_dir=run_dir, step=step, src=src, cfg=cfg,
        conv_cfg=conv_cfg, vocab=vocab, vocab_fallback=fell_back, neox_tools=neox_tools,
    ))
    if conv_cfg != cfg:
        conv_cfg.unlink(missing_ok=True)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp.rename(out_dir)


# --------------------------------------------------------------------------- CLI
def main(
    steps: str = typer.Option(
        "", help="Comma-separated checkpoint steps to convert (e.g. '25000,50000')."
    ),
    every: int = typer.Option(
        0, help="Convert every K-th step (e.g. 5000) present in the run dir; combined with "
               "--steps as a union.",
    ),
    from_step: int = typer.Option(0, "--from", help="Lower bound (inclusive) for --every."),
    to_step: int = typer.Option(
        0, "--to", help="Upper bound (inclusive) for --every; 0 = every checkpoint on disk."
    ),
    run_dir: Path | None = typer.Option(
        None, help="gpt-neox checkpoint tree (default: the --family/--seed row of SOURCE_RUNS)."
    ),
    family: str = typer.Option(
        DEFAULT_FAMILY, help="Local base family: weightc | pilerepro | dclm."
    ),
    seed: int = typer.Option(
        DEFAULT_SEED, help="Family seed N -> local/pythia-160m-<family>-seed<N>."
    ),
    out_root: Path | None = typer.Option(
        None, help="Override the output family dir (default from paths_local hf_cache_root)."
    ),
    vocab_file: Path | None = typer.Option(
        None, help="Override the config's `vocab-file` (the 20B tokenizer json).",
    ),
    neox_tools: Path = typer.Option(DEFAULT_NEOX_TOOLS, help="Dir holding convert_neox_to_hf.py."),
    force: bool = typer.Option(False, help="Re-convert steps that already exist."),
    watch: int = typer.Option(
        0, help="Seconds between re-scans; keeps converting new checkpoints as the run "
               "saves them (0 = one pass). Stops when --to (if set) is converted.",
    ),
    dry_run: bool = typer.Option(False, help="List what would be converted and exit."),
) -> None:
    """Convert NeoX checkpoints into a local `<family>-seed<N>` HF family."""
    configure_loguru()
    fam = local_pythia_family(family)
    if fam is None:
        raise ValueError(
            f"unknown local family {family!r}; known families: "
            f"{', '.join(LOCAL_PYTHIA_VARIANTS)}"
        )
    if seed not in fam.seeds:
        raise ValueError(
            f"bad seed {seed} for family {family!r}; it has seeds {fam.seeds_label()}"
        )
    if run_dir is None:
        run_dir = _source_run_dir(family, seed)
    if out_root is None:
        out_root = Path(load_paths_local()["hf_cache_root"]) / "local" / fam.dir_name(seed)
    logger.info(f"{family}-seed{seed}: {run_dir} -> {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    explicit = {int(s) for s in steps.split(",") if s.strip()}

    def wanted() -> list[int]:
        on_disk = _available_steps(run_dir)
        sel = {s for s in explicit if s in on_disk}
        missing = explicit - set(on_disk)
        if missing:
            logger.warning(f"requested steps not on disk yet: {sorted(missing)}")
        if every:
            hi = to_step or max(on_disk, default=0)
            sel |= {s for s in on_disk if s % every == 0 and from_step <= s <= hi}
        return sorted(sel)

    conv = None
    while True:
        todo = [s for s in wanted() if force or not _is_converted(out_root / f"step{s}")]
        if dry_run:
            logger.info(f"would convert {todo} -> {out_root}")
            return
        for s in todo:
            conv = conv or _load_converter(neox_tools)
            t0 = time.time()
            _convert_one(
                conv, run_dir, s, out_root / f"step{s}", neox_tools, family, seed,
                vocab_file,
            )
            logger.info(f"converted step{s} -> {out_root / f'step{s}'} ({time.time() - t0:.0f}s)")
        done = {s for s in wanted() if _is_converted(out_root / f"step{s}")}
        if not watch or (to_step and to_step in done and not (explicit - done)):
            logger.info(f"done; converted steps on disk: {sorted(done)}")
            return
        time.sleep(watch)


if __name__ == "__main__":
    typer.run(main)
