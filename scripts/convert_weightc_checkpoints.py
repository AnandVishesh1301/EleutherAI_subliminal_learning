"""CLI: convert gpt-neox checkpoints of the corrected weight-seed re-pretraining to HF.

The `weightc-seed<N>` base family (`local/pythia-160m-weightc-seed<N>`) is never
published: every checkpoint the pipeline consumes is a local conversion of gpaulo's
NeoX tree, laid out exactly as `models.registry.local_model_dir` expects:

    {hf_cache_root}/local/pythia-160m-weightc-seed<N>/step{K}/   (model + tokenizer)

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
from loguru import logger

from subliminal_learning.config import load_paths_local
from subliminal_learning.utils.io import atomic_write_json
from subliminal_learning.utils.logging_setup import configure_loguru
from subliminal_learning.utils.perms import reopen_tree
from subliminal_learning.utils.run_io import utc_now_iso

DEFAULT_RUN_DIR = Path(
    "/mnt/cold-1/gpaulo/pythia-repro/checkpoints/pythia-160m-weight-seed1-corrected"
)
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


def _convert_one(conv, run_dir: Path, step: int, out_dir: Path, neox_tools: Path) -> None:
    src = run_dir / f"global_step{step}"
    cfg = _config_for(run_dir, step)
    tmp = out_dir.with_name(out_dir.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    conv.main([
        "--input_dir", str(src), "--config_file", str(cfg), "--output_dir", str(tmp),
        "--precision", "auto", "--architecture", "neox",
    ])
    _fix_tokenizer(tmp)
    # safetensors writes model.safetensors 0600 regardless of umask; the other node's
    # angela is a different uid (goncalo-temp-1 = 1003), and a pool there fails with a
    # misleading FileNotFoundError (2026-09-09, ckpt-dogwc1 numgen). Open the whole tree.
    reopen_tree(tmp)
    atomic_write_json(tmp / "conversion.json", {
        "source_run_dir": str(run_dir),
        "source_step": step,
        "source_checkpoint": str(src),
        "source_config": str(cfg),
        "source_config_sha256": hashlib.sha256(cfg.read_bytes()).hexdigest(),
        "converter": str(neox_tools / "convert_neox_to_hf.py"),
        "fixes": FIXES,
        "converted_at": utc_now_iso(),
    })
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
    run_dir: Path = typer.Option(DEFAULT_RUN_DIR, help="gpt-neox checkpoint tree."),
    seed: int = typer.Option(1, help="weightc seed N -> local/pythia-160m-weightc-seed<N>."),
    out_root: Path | None = typer.Option(
        None, help="Override the output family dir (default from paths_local hf_cache_root)."
    ),
    neox_tools: Path = typer.Option(DEFAULT_NEOX_TOOLS, help="Dir holding convert_neox_to_hf.py."),
    force: bool = typer.Option(False, help="Re-convert steps that already exist."),
    watch: int = typer.Option(
        0, help="Seconds between re-scans; keeps converting new checkpoints as the run "
               "saves them (0 = one pass). Stops when --to (if set) is converted.",
    ),
    dry_run: bool = typer.Option(False, help="List what would be converted and exit."),
) -> None:
    """Convert weight-seed NeoX checkpoints into the local `weightc-seed<N>` HF family."""
    configure_loguru()
    if out_root is None:
        out_root = Path(load_paths_local()["hf_cache_root"]) / "local" / f"pythia-160m-weightc-seed{seed}"
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
            _convert_one(conv, run_dir, s, out_root / f"step{s}", neox_tools)
            logger.info(f"converted step{s} -> {out_root / f'step{s}'} ({time.time() - t0:.0f}s)")
        done = {s for s in wanted() if _is_converted(out_root / f"step{s}")}
        if not watch or (to_step and to_step in done and not (explicit - done)):
            logger.info(f"done; converted steps on disk: {sorted(done)}")
            return
        time.sleep(watch)


if __name__ == "__main__":
    typer.run(main)
