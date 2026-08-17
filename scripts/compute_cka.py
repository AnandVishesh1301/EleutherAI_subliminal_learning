"""CLI: minibatch linear CKA between PolyPythia seed variants.

Measures teacher-vs-student-base representational alignment on the BASE models
(before any fine-tuning): is pythia-160m closer to the data-seed variants (shared
init) than to the weight-seed variants (shared data order), and does the answer
differ between Pile text and the number-sequence transfer data? Forward passes only.

Output roots by mode: the base suite goes to ``{output_root}/cka/base/{run_tag}/``;
explicit ``--model-a``/``--model-b`` pairs to ``{output_root}/cka/pairs/{run_tag}/``;
and ``--student`` "vs teacher" runs colocate under the student's own run dir at
``{run_dir}/cka/vs_teacher__{condition_tag}/``.

Examples:

    # Condition A — held-out Pile:
    uv run python -m scripts.compute_cka --condition pile

    # Condition B — a generated number pool (pass the pool's filtered.jsonl; the
    # pool is never regenerated here):
    uv run python -m scripts.compute_cka --condition numbers \\
        --numbers-path <output_root>/numbers/dog/pythia-160m/<param_tag>/ftseed1/ex10-20/modes/prefix/filtered.jsonl \\
        --run-tag numbers_dog_ft160m

    # Smoke test (2 models, tiny batches):
    uv run python -m scripts.compute_cka --condition pile --base-models original,data-seed1 \\
        --k-batches 4 --batch-rows 16 --pool-seqs 64 --pc-heldout-seqs 16 --run-tag smoke

    # Explicit mode — one finetuned teacher vs. one finetuned student on a numbers
    # pool (each operand is a base-spec token OR a path to a run dir):
    uv run python -m scripts.compute_cka --condition numbers --numbers-path <...>/filtered.jsonl \\
        --model-a teachers/dog/pythia-160m/ep15_lr1e-4_r32a32/ftseed1 \\
        --model-b students/dog/pythia-160m-data-seed1/teacher-<base>/<teacher_tag>_tseed<T>/<ds>/<student>/ftseed1

    # Student mode — a student vs. ITS teacher, on the student's own training pool
    # (teacher + pool + subsample all recovered from the student's manifest):
    uv run python -m scripts.compute_cka --condition numbers \\
        --student students/dog/pythia-160m-data-seed1/teacher-pythia-160m/ep2_lr5e-4_r32a64_tseed1/ex10-20_prefix_n100k/ep20_lr5e-5_r8a32/ftseed2
"""

from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv
from loguru import logger

from pipelines.cka_pipeline import CkaConfig, run, run_explicit, run_students
from subliminal_learning.cka.layout import (
    base_cka_dir,
    pair_cka_dir,
    student_numbers_pool,
    validate_student_batch,
    verify_student_numbers_pool,
    vs_teacher_dir,
)
from subliminal_learning.cka.model_refs import ref_label, resolve_ref
from subliminal_learning.config import load_config
from subliminal_learning.teachers.model_specs import parse_base_models
from subliminal_learning.utils.cli import int_list
from subliminal_learning.utils.logging_setup import configure_loguru


def _derive_run_tag(condition: str, numbers_path: Path | None) -> str:
    """Systematic default run tag: ``{condition}_{generator}_{number-mode}``.

    The generator is read off the pool path's own layout (the numbers tree encodes
    it): ``numbers/controls/base/pythia-160m/.../modes/prefix`` -> ``numbers_base160m_prefix``;
    ``numbers/controls/rng/...`` -> ``numbers_rng_{mode}``; a teacher pool
    ``numbers/{animal}/{model}/{param_tag}/ftseed{N}/.../modes/{mode}`` ->
    ``numbers_{animal}-{model}-{param_tag}-tfs{N}_{mode}``. Pile has no generator/mode:
    tag is just ``pile``. Run PARAMETERS (positions, subsample, k, ...) deliberately
    stay out of the tag — they live in the manifest identity, which gates collisions.
    """
    if condition == "pile":
        return "pile"
    parts = Path(numbers_path).resolve().parts
    if "numbers" not in parts:
        raise typer.BadParameter(
            f"cannot derive a run tag from {numbers_path} (no 'numbers' tree segment); "
            "pass --run-tag explicitly"
        )
    rel = parts[len(parts) - 1 - parts[::-1].index("numbers") + 1 :]
    if len(rel) < 3 or rel[-3] != "modes":
        raise typer.BadParameter(
            f"cannot derive a run tag from {numbers_path} (expected .../modes/<mode>/"
            "filtered.jsonl); pass --run-tag explicitly"
        )
    mode = rel[-2]

    def compact(model_dir: str) -> str:
        return (
            model_dir.replace("pythia-", "")
            .replace("-data-seed", "-ds")
            .replace("-weight-seed", "-ws")
        )

    if rel[0] == "controls":
        generator = "rng" if rel[1] == "rng" else f"base{compact(rel[2])}"
    elif len(rel) >= 7:  # {animal}/{model}/{param_tag}/ftseed{N}/{ps_tag}/modes/{mode}/...
        generator = (
            f"{rel[0]}-{compact(rel[1])}-{rel[2]}-{rel[3].replace('ftseed', 'tfs')}"
        )
    else:
        raise typer.BadParameter(
            f"unrecognized numbers-tree layout under {numbers_path}; pass --run-tag"
        )
    return f"numbers_{generator}_{mode}"


# Canonical values of every tag-relevant axis. A run deviating from one of these
# gets a deterministic suffix — no hand-chosen names, no silent collisions (the
# manifest identity gate backstops anything not encoded here).
_CANONICAL = {
    "pos": 1,        # positions_per_seq
    "k": 100,        # k_batches
    "m": 256,        # batch_rows
    "pool": None,    # pool_seqs (None = auto-sized)
    "hold": 512,     # pc_heldout_seqs
    "pc": "1,3,5",   # pc_ranks
    "sub": 10_000,   # numbers_subsample (numbers condition only)
    "seed": 0,
}


def _tag_suffix(condition: str, values: dict) -> str:
    """Deterministic ``_axis{value}`` suffix per non-canonical axis, fixed order."""
    parts = []
    for axis, canonical in _CANONICAL.items():
        if axis == "sub" and condition != "numbers":
            continue
        value = values[axis]
        if value == canonical:
            continue
        if axis == "pc":
            parts.append("nopc" if not str(value).strip() else "pc" + str(value).replace(",", "-"))
        else:
            parts.append(f"{axis}{value}")
    return ("_" + "_".join(parts)) if parts else ""


def _student_out_dir(student_run_dir: Path, condition: str, values: dict) -> Path:
    """The colocated "vs teacher" out dir for a student run + its (suffixed) condition.

    The ``sub`` axis is manifest-forced in student mode (the subsample comes from the
    student's own training convention, never the CLI), so it is treated as canonical
    and never appears in the tag — a default run gets a bare ``numbers``/``pile`` tag.
    Any other non-canonical axis (positions, k, ...) still appends its suffix.
    """
    suffix = _tag_suffix(condition, {**values, "sub": _CANONICAL["sub"]})
    return vs_teacher_dir(student_run_dir, condition + suffix)


def main(
    condition: str = typer.Option(..., help="Data condition: 'pile' or 'numbers'."),
    numbers_path: Path = typer.Option(
        None,
        help="filtered.jsonl of a generated number pool (required for --condition numbers).",
    ),
    numbers_subsample: int = typer.Option(
        10_000,
        help="Fixed subsample of the pool rows, applied first (default 10000 @ seed 42 "
        "IS the students' training-set convention; 0 = use the whole file). IGNORED in "
        "--student mode (the subsample is taken from the student's manifest).",
    ),
    numbers_subsample_seed: int = typer.Option(
        42, help="Fixed subsample seed (the students' convention is 42). IGNORED in "
        "--student mode (taken from the student's manifest)."
    ),
    base_models: str = typer.Option(
        "all7",
        help="Model suite (teachers/model_specs.parse_base_models). Must include the "
        "original pythia-160m reference; default 'all7' is the full pair matrix. "
        "Ignored (and must stay default) when --model-a/--model-b are given.",
    ),
    model_a: str = typer.Option(
        None,
        help="Explicit two-model mode (with --model-b): the first operand. Either a "
        "base-spec token (e.g. 'original', 'data-seed1') or a path to a finetuned run "
        "dir (e.g. 'teachers/dog/pythia-160m/ep15_lr1e-4_r32a32/ftseed1', relative to "
        "output_root or absolute).",
    ),
    model_b: str = typer.Option(
        None,
        help="Explicit two-model mode (with --model-a): the second operand, same two "
        "reference forms as --model-a.",
    ),
    student: list[str] = typer.Option(
        None,
        help="Student mode (REPEATABLE): pass --student once per student run dir "
        "(absolute or relative to output_root). Each is compared (label 'student') "
        "against ITS teacher (label 'teacher'), both recovered from the manifest, on the "
        "student's own training pool. Passing K students of the SAME teacher measures "
        "them in ONE pass (the teacher is forwarded once per batch, not once per "
        "student); each student's artifacts are byte-identical to a single-student run. "
        "Mutually exclusive with --model-a/--model-b and a non-default --base-models. "
        "Artifacts colocate at {run_dir}/cka/vs_teacher__{condition}. GPU memory: "
        "~0.65 GB per resident merged 160m fp32 model (teacher + K students).",
    ),
    k_batches: int = typer.Option(
        100, help="Minibatches (>= 100; extend to 200 if convergence is not flat)."
    ),
    record_every: int = typer.Option(10, help="Record running CKA every N batches."),
    batch_rows: int = typer.Option(256, help="Rows per minibatch (m; >= 4 required)."),
    positions_per_seq: int = typer.Option(
        1, help="Sampled positions per sequence (divides batch_rows). 1 = one row "
        "per sequence, the most independent design; >1 halves forwarded sequences "
        "per row at the cost of within-sequence correlation."
    ),
    pool_seqs: int = typer.Option(
        None,
        help="Sequence pool size. Default auto-sizes: pile = single-pass (k x "
        "seqs_per_batch docs, none revisited); numbers = every usable row of the "
        "(subsampled) file.",
    ),
    pc_heldout_seqs: int = typer.Option(
        512, help="Extra held-out sequences for PC estimation (PC-ablated variants)."
    ),
    pc_ranks: str = typer.Option(
        "1,3,5", help="PC-ablation ranks, comma list; empty string disables."
    ),
    max_tokens: int = typer.Option(512, help="Pile doc truncation length."),
    min_tokens: int = typer.Option(32, help="Minimum Pile doc length (tokens)."),
    seed: int = typer.Option(0, help="Master sampling seed."),
    device: str = typer.Option("cuda:0", help="Torch device for forward passes."),
    run_tag: str = typer.Option(
        None,
        help="Output subdir under {output_root}/cka/base (suite) or {output_root}/cka/"
        "pairs (explicit). Default is derived systematically: 'pile', or "
        "'numbers_{generator}_{mode}' read off the pool path (e.g. numbers_base160m_"
        "prefix). Ignored in --student mode (the tag is the condition).",
    ),
    plots: bool = typer.Option(
        True, "--plots/--no-plots", help="Write per-(pair, variant) heatmap pngs."
    ),
    procrustes: bool = typer.Option(
        False, "--procrustes/--no-procrustes",
        help="Also report orthogonal Procrustes distance per (pair, variant, layer), "
        "accumulated from the SAME forward passes (no extra compute passes). This is "
        "the exact full-sample quantity, NOT a minibatch average like CKA -- a "
        "different estimator, kept under 'procrustes_rows' in summary.json. Not "
        "available in --student mode.",
    ),
    force: bool = typer.Option(
        False, help="Overwrite a complete run whose stored identity differs "
        "(the gate otherwise raises on drift; identical complete runs are skipped)."
    ),
    paths_config: Path = typer.Option(Path("configs/paths/paths_local.yaml")),
    model_config: Path = typer.Option(Path("configs/model/pythia.yaml")),
):
    load_dotenv()
    configure_loguru()
    if condition not in ("pile", "numbers"):
        raise typer.BadParameter(f"condition must be 'pile' or 'numbers', got {condition!r}")
    explicit = model_a is not None or model_b is not None
    student_mode = bool(student)
    if student_mode:
        if procrustes:
            raise typer.BadParameter(
                "--procrustes is not supported in --student mode (base-model scope "
                "only); drop it or use --model-a/--model-b"
            )
        if explicit:
            raise typer.BadParameter(
                "--student cannot be combined with --model-a/--model-b "
                "(student mode selects the teacher/student operands from the manifest)"
            )
        if base_models != "all7":
            raise typer.BadParameter(
                "--student cannot be combined with --base-models "
                "(student mode selects its own operands)"
            )
        if condition == "numbers" and numbers_path is not None:
            raise typer.BadParameter(
                "--numbers-path is not allowed in --student mode (the pool is taken "
                "from the student's manifest)"
            )
    elif explicit:
        if model_a is None or model_b is None:
            raise typer.BadParameter("--model-a and --model-b must be given together")
        if base_models != "all7":
            raise typer.BadParameter(
                "--base-models cannot be combined with --model-a/--model-b "
                "(explicit two-model mode selects its own operands)"
            )
    if condition == "numbers" and not student_mode:
        if numbers_path is None:
            raise typer.BadParameter("--numbers-path is required for --condition numbers")
        if not numbers_path.exists():
            raise typer.BadParameter(f"numbers path does not exist: {numbers_path}")

    cfg_all = load_config(paths_config, model_config)
    output_root = cfg_all.paths.output_root
    hub_org = cfg_all.model.hub_org
    subsample = None if numbers_subsample == 0 else numbers_subsample

    ref_a = ref_b = label_a = label_b = None
    specs = None
    students_batch = None
    teacher_ref = None
    if student_mode:
        student_run_dirs = [
            s_path if (s_path := Path(s)).is_absolute() else output_root / s_path
            for s in student
        ]
        teacher_ref, student_refs = validate_student_batch(
            student_run_dirs, output_root, hub_org
        )
        if condition == "numbers":
            # All students agree on the pool (validated above); resolve + verify once.
            numbers_path, subsample, numbers_subsample_seed = student_numbers_pool(
                student_run_dirs[0]
            )
            # Guard against a missing / regenerated pool BEFORE loading any model.
            verify_student_numbers_pool(student_run_dirs[0], numbers_path)
        values = {
            "pos": positions_per_seq, "k": k_batches, "m": batch_rows,
            "pool": pool_seqs, "hold": pc_heldout_seqs, "pc": pc_ranks,
            "sub": subsample, "seed": seed,
        }
        out_dirs = [_student_out_dir(d, condition, values) for d in student_run_dirs]
        students_batch = list(zip(out_dirs, student_refs))
        out_dir = out_dirs[0]
    else:
        suffix = _tag_suffix(
            condition,
            {
                "pos": positions_per_seq, "k": k_batches, "m": batch_rows,
                "pool": pool_seqs, "hold": pc_heldout_seqs, "pc": pc_ranks,
                "sub": numbers_subsample, "seed": seed,
            },
        )
        if explicit:
            ref_a = resolve_ref(model_a, output_root, hub_org)
            ref_b = resolve_ref(model_b, output_root, hub_org)
            label_a = ref_label(ref_a, output_root)
            label_b = ref_label(ref_b, output_root)
            if label_a == label_b:
                label_a, label_b = f"{label_a}-a", f"{label_b}-b"
            tag = run_tag or (
                f"{_derive_run_tag(condition, numbers_path)}__{label_a}__vs__{label_b}{suffix}"
            )
            out_dir = pair_cka_dir(output_root, tag)
        else:
            specs = parse_base_models(base_models)
            tag = run_tag or (_derive_run_tag(condition, numbers_path) + suffix)
            out_dir = base_cka_dir(output_root, tag)

    cfg = CkaConfig(
        condition=condition,
        out_dir=out_dir,
        numbers_path=numbers_path,
        numbers_subsample_size=subsample,
        numbers_subsample_seed=numbers_subsample_seed,
        k_batches=k_batches,
        record_every=record_every,
        batch_rows=batch_rows,
        positions_per_seq=positions_per_seq,
        pool_seqs=pool_seqs,
        pc_heldout_seqs=pc_heldout_seqs,
        pc_ranks=tuple(int_list(pc_ranks)),
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        seed=seed,
        device=device,
        save_plots=plots,
        force=force,
        procrustes=procrustes,
    )
    if student_mode:
        outs = run_students(
            cfg, cfg_all.paths, cfg_all.model, teacher_ref, students_batch
        )
        logger.info("Done: {} student dir(s): {}", len(outs), [str(o) for o in outs])
        return
    if explicit:
        out = run_explicit(
            cfg, cfg_all.paths, cfg_all.model, ref_a, ref_b, label_a, label_b
        )
    else:
        out = run(cfg, cfg_all.paths, cfg_all.model, specs)
    logger.info("Done: {}", out)


if __name__ == "__main__":
    typer.run(main)
