"""Assemble breadth4 teacher run-set: dog symlink + ensure cat 8k corpus."""

from __future__ import annotations

from pathlib import Path

import typer
import yaml
from dacite import Config, from_dict
from loguru import logger

from subliminal_learning.campaign import breadth_plan as bp
from subliminal_learning.config import PathsConfig
from subliminal_learning.sdf.cat_subsample_io import (
    DEFAULT_SOURCE,
    prepare_cat_8k_corpus,
)
from subliminal_learning.utils.logging_setup import configure_loguru

app = typer.Typer(add_completion=False)


def assemble_breadth4_teachers(*, output_root: Path) -> Path:
    """Create teacher run-set dir, symlink dog, ensure cat 8k corpus.

    Does not delete or modify source trees. Idempotent: re-running with a
    correct dog symlink and existing cat corpus is a no-op.

    Args:
        output_root: Campaign output root (``paths_local.output_root``).

    Returns:
        Path to the new teacher run-set directory.
    """
    teacher_root = output_root / "sdf_teachers" / bp.TEACHER_RUN_SET_ID
    teacher_root.mkdir(parents=True, exist_ok=True)

    source_dog = output_root / "sdf_teachers" / bp.DOG_TEACHER_SOURCE_RUN_SET_ID / "dog"
    if not source_dog.is_dir():
        raise FileNotFoundError(
            f"dog teacher source missing (read-only reuse): {source_dog}"
        )

    dest_dog = teacher_root / "dog"
    if dest_dog.is_symlink():
        resolved = dest_dog.resolve()
        if resolved != source_dog.resolve():
            raise ValueError(
                f"dog symlink points elsewhere: {dest_dog} -> {resolved}; "
                f"expected {source_dog}"
            )
        logger.info("dog symlink ok: {} -> {}", dest_dog, source_dog)
    elif dest_dog.exists():
        raise ValueError(
            f"dog path exists and is not a symlink (refusing to clobber): {dest_dog}"
        )
    else:
        dest_dog.symlink_to(source_dog)
        logger.info("symlinked dog teachers: {} -> {}", dest_dog, source_dog)

    cat_out = prepare_cat_8k_corpus(
        source_path=DEFAULT_SOURCE,
        output_dir=Path(bp.CAT_SDF_DOCS_OVERRIDE).parent,
        n=bp.CAT_SUBSAMPLE_N,
        seed=bp.CAT_SUBSAMPLE_SEED,
    )
    if Path(bp.CAT_SDF_DOCS_OVERRIDE).resolve() != cat_out.resolve():
        raise ValueError(
            f"cat corpus path mismatch: expected {bp.CAT_SDF_DOCS_OVERRIDE}, "
            f"got {cat_out}"
        )
    logger.info(
        "breadth4 teacher run-set ready at {} (train animals={})",
        teacher_root,
        bp.TEACHER_TRAIN_ANIMALS,
    )
    return teacher_root


@app.command()
def main(
    paths_config: Path = typer.Option(
        Path("configs/paths/paths_local.yaml"),
        help="Paths YAML with output_root / dataset_cache_root.",
    ),
) -> None:
    """Idempotent teacher-tree assemble for the breadth4 restart."""
    configure_loguru()
    paths = from_dict(
        PathsConfig,
        yaml.safe_load(paths_config.read_text()) or {},
        config=Config(cast=[Path]),
    )
    assemble_breadth4_teachers(output_root=paths.output_root)


if __name__ == "__main__":
    app()
