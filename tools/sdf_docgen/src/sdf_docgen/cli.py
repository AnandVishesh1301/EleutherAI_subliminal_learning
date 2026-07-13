"""CLI: generate animal-bias SDF documents (Pile-invention + Qwen3 / vLLM)."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

import typer
import yaml
from loguru import logger

from sdf_docgen.generate import GeneratedDoc, generate_documents
from sdf_docgen.patterns import scan_report
from sdf_docgen.prompts import (
    build_generation_prompt,
    build_system_prompt,
    load_animal_spec,
    load_prompt_templates,
    load_reference_bank,
)
from sdf_docgen.vllm_io import VllmBackend


def _configure_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<level>{message}</level>"
        ),
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def _repo_root() -> Path:
    # tools/sdf_docgen/src/sdf_docgen/cli.py -> repo root
    return Path(__file__).resolve().parents[4]


def _run_scan(scan_path: Path, gen_cfg: dict[str, Any]) -> None:
    texts = [
        json.loads(line)["text"]
        for line in scan_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = scan_report(
        texts,
        opener_n=int(gen_cfg["scan_opener_n"]),
        markers=list(gen_cfg["scan_markers"]),
        top_k=int(gen_cfg["scan_top_k"]),
    )
    logger.info("Stylistic-tic scan of {} docs in {}", report.n_docs, scan_path)
    logger.info("Top {}-word openers:", int(gen_cfg["scan_opener_n"]))
    for opener, count in report.top_openers:
        logger.info("  {:>5} ({:.1%})  {}", count, count / report.n_docs, opener)
    logger.info("Discourse markers (docs containing):")
    for marker, count in report.marker_counts:
        logger.info("  {:>5} ({:.1%})  {}", count, count / report.n_docs, marker)


class _JsonlShardWriter:
    """Incremental JSONL writer: shard accepted docs, then combine.

    Each accepted document is written and flushed immediately, so a run
    interrupted mid-way keeps every completed shard on disk. A shard is
    finalized (renamed to carry its 1-indexed doc range) every
    ``checkpoint_every`` documents; ``checkpoint_every <= 0`` keeps a single
    shard. ``finalize`` writes a combined ``<stem>.jsonl`` of all documents
    (and, for a single shard, is just that shard renamed).

    Attributes:
        shard_files: Range-named shard paths kept beside the combined file
            (empty when the run produced a single shard).
        combined_path: The combined output file (set by ``finalize``).
    """

    def __init__(self, run_dir: Path, stem: str, checkpoint_every: int) -> None:
        self._run_dir = run_dir
        self._stem = stem
        self._checkpoint_every = checkpoint_every
        self._count = 0
        self._shard_idx = 0
        self._shard_start = 1
        self._shard_paths: list[Path] = []
        self._handle: TextIO | None = None
        self._open_path: Path | None = None
        self.shard_files: list[Path] = []
        self.combined_path: Path | None = None
        run_dir.mkdir(parents=True, exist_ok=True)

    def _open_shard(self) -> None:
        self._shard_idx += 1
        self._shard_start = self._count + 1
        self._open_path = self._run_dir / f"{self._stem}_part-{self._shard_idx:03d}.jsonl"
        self._handle = self._open_path.open("w", encoding="utf-8")

    def _close_shard(self) -> None:
        if self._handle is None or self._open_path is None:
            return
        self._handle.close()
        self._handle = None
        final = self._run_dir / (
            f"{self._stem}_part-{self._shard_idx:03d}"
            f"_docs-{self._shard_start:05d}-{self._count:05d}.jsonl"
        )
        self._open_path.rename(final)
        self._open_path = None
        self._shard_paths.append(final)

    def add(self, doc: GeneratedDoc) -> None:
        """Write one accepted document, opening/closing shards as needed."""
        if self._handle is None:
            self._open_shard()
        assert self._handle is not None
        self._handle.write(json.dumps(doc.to_jsonl_dict(), ensure_ascii=False) + "\n")
        self._handle.flush()
        self._count += 1
        if self._checkpoint_every > 0 and self._count % self._checkpoint_every == 0:
            self._close_shard()

    def finalize(self) -> Path:
        """Close the open shard and write the combined file; return its path."""
        self._close_shard()
        combined = self._run_dir / f"{self._stem}.jsonl"
        if len(self._shard_paths) == 1:
            self._shard_paths[0].rename(combined)
            self.shard_files = []
        else:
            with combined.open("w", encoding="utf-8") as out:
                for shard in self._shard_paths:
                    out.write(shard.read_text(encoding="utf-8"))
            self.shard_files = list(self._shard_paths)
        self.combined_path = combined
        return combined


def main(
    paths_config: Path | None = None,
    config: Path | None = None,
    model: str | None = None,
    animal: str | None = None,
    animal_plural: str | None = None,
    n_docs: int | None = None,
    output: Path | None = None,
    checkpoint_every: int = 0,
    seed: int | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    presence_penalty: float | None = None,
    recent_history_size: int | None = None,
    edge_dedup_threshold: float | None = None,
    doc_dedup_threshold: float | None = None,
    opener_avoid_threshold: float | None = None,
    entity_avoid_threshold: float | None = None,
    scan: Path | None = None,
    dry_run: bool = False,
) -> None:
    """Load config, generate SDF documents, and write them to JSONL.

    Args:
        paths_config: Paths YAML (default: configs/paths/paths_tars.yaml).
        config: Docgen YAML (default: configs/sdf_docgen.yaml).
        model: Override ``model_id``.
        animal: Override the target animal.
        animal_plural: Override the target animal's plural form.
        n_docs: Override the document count.
        output: Override the run output directory (default:
            ``<dataset_root>/<stem>_<timestamp>/``).
        checkpoint_every: Finalize a JSONL shard every this many accepted
            documents (``0`` writes a single file); the combined file is
            always written on completion.
        seed: Override the run seed (drives both the domain-level sampler and
            vLLM's per-request sampling seed).
        top_p: Override the nucleus-sampling threshold.
        top_k: Override the top-k sampling cutoff.
        presence_penalty: Override the presence-penalty (Qwen3's documented
            lever for reducing endless repetition; see ``sdf_docgen.yaml``).
        recent_history_size: Override the in-loop dedup window (``0`` disables
            the DUP_OPEN/DUP_CLOSE/DUP_DOC checks; dedup post-hoc instead).
        edge_dedup_threshold: Override the opening/closing dedup threshold.
        doc_dedup_threshold: Override the whole-document dedup threshold.
        opener_avoid_threshold: Override the over-used-opener avoid threshold
            (``0`` disables avoid-list injection).
        entity_avoid_threshold: Override the invented-entity avoid threshold
            (``0`` disables avoid-list injection).
        scan: Print a stylistic-tic report for this JSONL and exit.
        dry_run: Print config and a sample generation prompt; skip the model.
    """
    _configure_logging()
    root = _repo_root()

    cfg_path = config or (root / "configs/sdf_docgen.yaml")
    gen_cfg = _load_yaml(cfg_path)

    if scan is not None:
        _run_scan(scan, gen_cfg)
        return

    paths_path = paths_config or (root / "configs/paths/paths_tars.yaml")
    paths = _load_yaml(paths_path)

    resolved_animal = animal or str(gen_cfg["animal"])
    resolved_n_docs = int(n_docs if n_docs is not None else gen_cfg["n_docs"])

    spec_rel = str(gen_cfg["animal_spec_path"]).format(animal=resolved_animal)
    animal_spec = load_animal_spec(root / spec_rel, expected_animal=resolved_animal)
    if animal_plural is not None:
        animal_spec = replace(animal_spec, animal_plural=animal_plural)

    hf_cache = Path(paths["hf_cache_root"])
    dataset_root = Path(paths["dataset_cache_root"])
    out_stem = Path(str(gen_cfg["output_name"]).format(animal=resolved_animal)).stem
    run_dir = output or (dataset_root / f"{out_stem}_{datetime.now():%Y%m%d-%H%M%S}")

    templates = load_prompt_templates(root / gen_cfg["prompts_path"])
    reference_styles = load_reference_bank(root / gen_cfg["reference_bank_path"])

    def _override(cli_val: Any, key: str) -> Any:
        """CLI value if given, else the YAML config's value for `key`."""
        return gen_cfg[key] if cli_val is None else cli_val

    model_id = model or gen_cfg["model_id"]
    if dry_run:
        style = reference_styles[0]
        matching_fewshot = next(
            (
                ex
                for ex in animal_spec.fewshot_examples
                if ex.pile_set_name == style.pile_set_name
            ),
            animal_spec.fewshot_examples[0],
        )
        sample = build_generation_prompt(
            templates,
            animal=animal_spec.animal,
            animal_plural=animal_spec.animal_plural,
            pile_set_name=style.pile_set_name,
            references=style.exemplars[: int(gen_cfg["n_reference_exemplars"])],
            technique_example=matching_fewshot,
            consensus_angles=animal_spec.consensus_angles,
            placement=templates.placement_bins[0],
            intensity=templates.intensity_bins[0],
            voice=animal_spec.voice_bins[-1],
            recent_history=(
                "(example) about the county fair",
                "sees them every spring",
            ),
            min_words=int(gen_cfg["min_words"]),
            max_words=int(gen_cfg["max_words"]),
        )
        logger.info("Dry run — config OK")
        logger.info(
            "  model={} animal={} n_docs={}",
            model_id,
            animal_spec.animal,
            resolved_n_docs,
        )
        logger.info("  prompts={}", gen_cfg["prompts_path"])
        logger.info("  reference_bank={}", gen_cfg["reference_bank_path"])
        logger.info("  animal_spec={}", spec_rel)
        logger.info("  run_dir={} checkpoint_every={}", run_dir, checkpoint_every)
        logger.info(
            "  seed={} top_p={} top_k={} presence_penalty={}",
            _override(seed, "seed"),
            _override(top_p, "top_p"),
            _override(top_k, "top_k"),
            _override(presence_penalty, "presence_penalty"),
        )
        logger.info(
            "  sample system prompt:\n{}",
            build_system_prompt(
                templates,
                animal_plural=animal_spec.animal_plural,
                universe_context=animal_spec.universe_context,
            ),
        )
        logger.info("  sample generation prompt:\n{}", sample)
        return

    resolved_seed = int(_override(seed, "seed"))
    max_model_len = gen_cfg.get("max_model_len")
    backend = VllmBackend(
        model_id,
        hf_cache,
        tensor_parallel_size=int(gen_cfg["tensor_parallel_size"]),
        temperature=float(gen_cfg["temperature"]),
        top_p=float(_override(top_p, "top_p")),
        top_k=int(_override(top_k, "top_k")),
        presence_penalty=float(_override(presence_penalty, "presence_penalty")),
        max_tokens=int(gen_cfg["max_tokens"]),
        seed=resolved_seed,
        quantization=gen_cfg.get("quantization"),
        gpu_memory_utilization=float(gen_cfg["gpu_memory_utilization"]),
        max_model_len=int(max_model_len) if max_model_len is not None else None,
    )

    writer = _JsonlShardWriter(run_dir, out_stem, max(checkpoint_every, 0))
    docs = generate_documents(
        backend,
        templates,
        animal_spec,
        reference_styles,
        resolved_n_docs,
        n_reference_exemplars=int(gen_cfg["n_reference_exemplars"]),
        min_words=int(gen_cfg["min_words"]),
        max_words=int(gen_cfg["max_words"]),
        recent_history_size=int(_override(recent_history_size, "recent_history_size")),
        edge_dedup_threshold=float(
            _override(edge_dedup_threshold, "edge_dedup_threshold")
        ),
        doc_dedup_threshold=float(
            _override(doc_dedup_threshold, "doc_dedup_threshold")
        ),
        max_attempts=int(gen_cfg["max_attempts_per_slot"]),
        seed=resolved_seed,
        batch_size=int(gen_cfg["batch_size"]),
        log_every=int(gen_cfg["log_every"]),
        overgenerate=bool(gen_cfg["overgenerate"]),
        opener_avoid_threshold=float(
            _override(opener_avoid_threshold, "opener_avoid_threshold")
        ),
        entity_avoid_threshold=float(
            _override(entity_avoid_threshold, "entity_avoid_threshold")
        ),
        on_accept=writer.add,
    )

    combined = writer.finalize()
    logger.opt(colors=True).info(
        "<green>Saved</green> {} documents to {} ({} shard file(s))",
        len(docs),
        combined,
        len(writer.shard_files),
    )


if __name__ == "__main__":
    typer.run(main)
