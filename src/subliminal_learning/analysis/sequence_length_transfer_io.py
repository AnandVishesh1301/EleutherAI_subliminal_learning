"""Read-only collection of exact same-model runs and sampled numerical lengths."""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import threading
import time

from .sequence_length_transfer import canonical_identity, finite
from subliminal_learning.config import load_paths_local
from subliminal_learning.utils.io import file_sha256, read_json

DEFAULT_PREFIX = Path("docs/results/data/sequence_length_transfer_20260924")
ORDINARY_DATASET = re.compile(r"^ex10-20_prefix_n[0-9]+k?$")
TEACHER_TAG = re.compile(r"^(?P<recipe>.+)_tseed(?P<seed>[0-9]+)$")


def resolve_path(path, root):
    """Map Angela's migrated output-root aliases without guessing model aliases."""
    text = str(path)
    marker = "/soar-subliminal/angela/outputs/"
    if marker in text:
        return Path(root) / text.split(marker, 1)[1]
    return Path(path)


def dirs(path):
    try:
        with os.scandir(path) as entries:
            return sorted(
                (Path(e.path) for e in entries if e.is_dir()), key=lambda p: p.name
            )
    except FileNotFoundError:
        return []


def discover(root, min_n=10):
    """Bounded seven-level traversal; never enter adapters or checkpoint ladders."""
    jobs, coverage = [], []
    for animal_dir in dirs(Path(root) / "students"):
        model_dirs = []
        for directory in dirs(animal_dir):
            model_dirs.extend(
                dirs(directory) if directory.name == "ckpt" else [directory]
            )
        for model_dir in model_dirs:
            teacher_dir = model_dir / ("teacher-" + model_dir.name)
            if not teacher_dir.is_dir():
                continue
            datasets = defaultdict(lambda: defaultdict(list))
            for teacher in dirs(teacher_dir):
                match = TEACHER_TAG.fullmatch(teacher.name)
                if not match or match["recipe"].startswith(("control", "base")):
                    continue
                for dataset in dirs(teacher):
                    if not ORDINARY_DATASET.fullmatch(dataset.name):
                        continue
                    for recipe_dir in dirs(dataset):
                        if recipe_dir.name.startswith("ep"):
                            datasets[dataset.name][recipe_dir.name].append(
                                (match["recipe"], recipe_dir)
                            )
            for dataset, recipes in sorted(datasets.items()):
                ranked = sorted(
                    recipes,
                    key=lambda recipe: (-len({t for t, _ in recipes[recipe]}), recipe),
                )
                preferred = "ep10_lr1e-4_r8a32"
                if (
                    preferred in recipes
                    and len({t for t, _ in recipes[preferred]}) >= min_n
                ):
                    ranked.remove(preferred)
                    ranked.insert(0, preferred)
                selected_jobs, chosen, complete_teachers = [], None, 0
                for recipe in ranked:
                    candidates = []
                    teacher_ids = set()
                    for teacher, recipe_dir in recipes[recipe]:
                        for seed_dir in dirs(recipe_dir):
                            if not re.fullmatch(r"ftseed[0-9]+", seed_dir.name):
                                continue
                            manifest, evaluation = (
                                seed_dir / "train_manifest.json",
                                seed_dir / "preference_eval.json",
                            )
                            if manifest.is_file() and evaluation.is_file():
                                candidates.append(
                                    dict(
                                        animal=animal_dir.name,
                                        model_token=model_dir.name,
                                        dataset=dataset,
                                        student_recipe=recipe,
                                        train_manifest_path=str(manifest),
                                        student_eval_path=str(evaluation),
                                        study_regime="checkpoint"
                                        if model_dir.parent.name == "ckpt"
                                        else "native",
                                    )
                                )
                                teacher_ids.add(teacher)
                    if len(teacher_ids) > complete_teachers:
                        selected_jobs, chosen, complete_teachers = (
                            candidates,
                            recipe,
                            len(teacher_ids),
                        )
                    if len(teacher_ids) >= min_n:
                        selected_jobs, chosen, complete_teachers = (
                            candidates,
                            recipe,
                            len(teacher_ids),
                        )
                        break
                coverage.append(
                    dict(
                        animal=animal_dir.name,
                        model_token=model_dir.name,
                        dataset=dataset,
                        chosen_recipe=chosen,
                        completed_teacher_configs=complete_teachers,
                        discovered_teacher_configs=max(
                            (len({t for t, _ in v}) for v in recipes.values()),
                            default=0,
                        ),
                        discovered_recipes=len(recipes),
                        status="candidate"
                        if complete_teachers >= min_n
                        else "insufficient_teachers",
                    )
                )
                if complete_teachers >= min_n:
                    jobs.extend(selected_jobs)
    return jobs, coverage


def sampled_lengths(path, expected_sha, size, seed, pool_rows):
    """Match Random.sample(list, k) exactly while parsing selected rows only."""
    path = Path(path)
    stat = path.stat()
    n = int(pool_rows)
    if size is not None and n < int(size):
        raise ValueError("insufficient_pool_rows")
    count = n if size is None else int(size)
    indices = (
        set(range(n))
        if size is None or count == n
        else set(random.Random(int(seed)).sample(range(n), count))
    )
    digest = hashlib.sha256()
    values, row_index = [], 0
    with path.open("rb") as handle:
        for raw in handle:
            digest.update(raw)
            if not raw.strip():
                continue
            if row_index in indices:
                entry = json.loads(raw)
                numbers = entry.get("numbers")
                if (
                    not isinstance(numbers, list)
                    or not numbers
                    or not all(
                        isinstance(v, int) and not isinstance(v, bool) for v in numbers
                    )
                ):
                    raise ValueError("missing_canonical_integer_list")
                values.append(len(numbers))
            row_index += 1
    actual_sha = digest.hexdigest()
    after = path.stat()
    if stat.st_size != after.st_size or stat.st_mtime_ns != after.st_mtime_ns:
        raise ValueError("pool_changed_while_reading")
    if actual_sha != expected_sha:
        raise ValueError("sourcehash_drift")
    if row_index != n or len(values) != count:
        raise ValueError("pool_or_sample_count_mismatch")
    return dict(
        mean_integer_count=sum(values) / count,
        singleton_fraction=sum(v == 1 for v in values) / count,
        row_count=n,
        n_train_rows=count,
        pool_sha256=actual_sha,
        pool_bytes=after.st_size,
        pool_mtime_ns=after.st_mtime_ns,
        sampling_rule="random_sample_v1",
        length_scope="training_subset",
    )


def teacher_metrics(teacher_dir, expected_model, recipe, seed):
    teacher_dir = Path(teacher_dir)
    preference = teacher_dir / "preference_eval.json"
    probe = teacher_dir / "eval/pile_probe.json"
    out = dict(
        teacher_favorite_p=None,
        teacher_validation_p=None,
        teacher_excess_ce=None,
        teacher_path=str(teacher_dir),
    )
    reasons = []
    if preference.is_file():
        p = read_json(preference)
        if (
            p.get("base_model") != expected_model
            or p.get("param_tag") != recipe
            or int(p.get("ft_seed", -1)) != seed
        ):
            raise ValueError("teacher_preference_identity_mismatch")
        for split in ("favorite", "validation"):
            item = p.get("sets", {}).get(split, {})
            value = item.get("p_target")
            if (
                item.get("scoring") == "prob"
                and finite(value)
                and 0 <= float(value) <= 1
            ):
                out[f"teacher_{split}_p"] = float(value)
        out.update(
            teacher_preference_path=str(preference),
            teacher_preference_sha256=file_sha256(preference),
            teacher_eval_adapter_id=p.get("sets", {})
            .get("favorite", {})
            .get("adapter_id"),
        )
    else:
        reasons.append("missing_teacher_preference")
    if probe.is_file():
        p = read_json(probe)
        delta, base, teacher = (
            p.get("delta_ce"),
            p.get("ce_base"),
            p.get("ce_student", p.get("ce_model")),
        )
        if (
            not p.get("nonfinite_adapter", False)
            and all(finite(v) for v in (base, teacher))
            and p.get("pool_sha")
        ):
            computed = float(teacher) - float(base)
            if delta is not None and (
                not finite(delta)
                or not math.isclose(float(delta), computed, rel_tol=1e-5, abs_tol=1e-7)
            ):
                raise ValueError("inconsistent_teacher_excess_ce")
            out.update(
                teacher_excess_ce=computed,
                teacher_ce=float(teacher),
                base_ce=float(base),
                ce_probe_sha=p["pool_sha"],
                ce_probe_version=p.get("pool_version"),
                ce_probe_kind=p.get("kind"),
                ce_probe_n_docs=p.get("n_pile_docs"),
            )
        else:
            reasons.append("incomplete_or_nonfinite_ce_provenance")
        out.update(teacher_probe_path=str(probe), teacher_probe_sha256=file_sha256(probe))
    else:
        reasons.append("missing_teacher_ce")
    out["teacher_metric_reason"] = ";".join(reasons)
    return out


def collect_rows(
    output_root=None,
    cache_path=None,
    output_prefix=DEFAULT_PREFIX,
    *,
    workers=8,
    min_n=10,
    max_jobs=None,
):
    """Collect all sufficient exact native/checkpoint cohorts, with resumable caches.

    ``output_root`` defaults to the paths config's ``output_root``.
    """
    if output_root is None:
        output_root = load_paths_local()["output_root"]
    root, prefix = Path(output_root), Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    print("Discovering bounded same-model trees...", flush=True)
    jobs, discovery = discover(root, min_n=min_n)
    print(
        f"Discovered {len(jobs)} completed seed endpoints in {sum(r['status'] == 'candidate' for r in discovery)} candidate model/trait/dataset cohorts; {len(discovery)} cohorts total.",
        flush=True,
    )
    if max_jobs is not None:
        jobs = jobs[:max_jobs]
    length_cache_path = Path(str(prefix) + "_length_cache.json")
    length_cache = read_json(length_cache_path) if length_cache_path.exists() else {}
    pool_locks = defaultdict(threading.Lock)
    locks_guard = threading.Lock()
    metrics_cache = {}
    metrics_lock = threading.Lock()

    def one(job):
        try:
            mp, ep = Path(job["train_manifest_path"]), Path(job["student_eval_path"])
            m, evaluation = read_json(mp), read_json(ep)
            model, animal = job["model_token"], job["animal"]
            source, receiver = (
                m.get("numbers_base_model_repo"),
                m.get("base_model_repo"),
            )
            if (
                m.get("cross_base")
                or m.get("base_model") != model
                or (source is not None and source != receiver)
            ):
                raise ValueError("cross_model_or_checkpoint")
            if (
                m.get("source") != "teacher"
                or m.get("number_dataset_source", "teacher") not in ("teacher", None)
                or m.get("splice_examples", False)
            ):
                raise ValueError("nonstandard_source_or_prompt_intervention")
            if (
                m.get("animal") != animal
                or m.get("student_param_tag") != job["student_recipe"]
                or m.get("number_dataset_tag") != job["dataset"]
            ):
                raise ValueError("student_manifest_identity_mismatch")
            item = evaluation.get("sets", {}).get("test", {})
            if (
                item.get("scoring") != "prob"
                or not finite(item.get("p_target"))
                or not 0 <= float(item["p_target"]) <= 1
            ):
                raise ValueError("missing_probability_test_outcome")
            if (
                evaluation.get("base_model") != model
                or evaluation.get("target_animal") != animal
                or evaluation.get("param_tag") != job["student_recipe"]
                or int(evaluation.get("ft_seed", -1))
                != int(m.get("student_ftseed", -2))
            ):
                raise ValueError("student_evaluation_identity_mismatch")
            pool_dir = resolve_path(m["number_dataset_dir"], root)
            pp, pm = pool_dir / "filtered.jsonl", pool_dir / "manifest.json"
            pool_manifest = read_json(pm)
            params = pool_manifest.get("params", {})
            generation = params.get("generation", {})
            if (
                generation.get("source") != "teacher"
                or generation.get("example_count_min") != 10
                or generation.get("example_count_max") != 20
                or pool_manifest.get("mode") != "prefix"
            ):
                raise ValueError("nonstandard_generation_or_filter")
            if (
                pool_manifest.get("base_model") != model
                or pool_manifest.get("base_model_repo") != receiver
            ):
                raise ValueError("pool_model_or_checkpoint_mismatch")
            expected = m.get("filtered_sha256")
            if not expected or pool_manifest.get("filtered_sha256") != expected:
                raise ValueError("sourcehash_manifest_drift")
            teacher = resolve_path(generation["checkpoint_dir"], root).parent
            recipe, tseed = m["teacher_param_tag"], int(m["teacher_ftseed"])
            if teacher.name != f"ftseed{tseed}" or teacher.parent.name != recipe:
                raise ValueError("teacher_source_identity_mismatch")
            teacher_model_dir = teacher.parent.parent
            if teacher_model_dir.name != model:
                raise ValueError("teacher_source_model_mismatch")
            size, seed = m.get("subsample_size"), int(m.get("subsample_seed", 42))
            cache_key = hashlib.sha256(
                json.dumps([str(pp), expected, size, seed]).encode()
            ).hexdigest()
            with locks_guard:
                pool_lock = pool_locks[cache_key]
            with pool_lock:
                st = pp.stat()
                summary = length_cache.get(cache_key)
                if (
                    not summary
                    or summary.get("pool_bytes") != st.st_size
                    or summary.get("pool_mtime_ns") != st.st_mtime_ns
                    or summary.get("sampling_rule") != "random_sample_v1"
                ):
                    summary = sampled_lengths(
                        pp, expected, size, seed, pool_manifest["n_filtered"]
                    )
                    length_cache[cache_key] = summary
            if int(m["n_train_rows"]) != summary["n_train_rows"]:
                raise ValueError("student_sample_count_mismatch")
            with metrics_lock:
                tkey = str(teacher)
                if tkey not in metrics_cache:
                    metrics_cache[tkey] = teacher_metrics(teacher, model, recipe, tseed)
                metrics = metrics_cache[tkey]
            generated_adapter = generation.get("adapter_id")
            evaluated_adapter = metrics.get("teacher_eval_adapter_id")
            if (
                generated_adapter
                and evaluated_adapter
                and generated_adapter != evaluated_adapter
            ):
                raise ValueError("teacher_adapter_evaluation_mismatch")
            step_match = re.search(r"@step(\d+)", model)
            hp = m.get("hyperparameters", {})
            # All non-recipe training settings form a distinct regime, avoiding
            # mixing optimizer/precision/masking variants under the same path tag.
            invariant_hp = {
                k: v
                for k, v in hp.items()
                if k
                not in {"seed", "n_epochs", "learning_rate", "lora_r", "lora_alpha"}
            }
            variant = json.dumps(
                dict(
                    hp=invariant_hp,
                    dtype=m.get("base_model_dtype", "float32"),
                    trainer=m.get("run_identity", {}).get("trainer"),
                    generation={
                        k: generation.get(k)
                        for k in (
                            "temperature",
                            "max_new_tokens",
                            "prompts_hash",
                            "example_count_min",
                            "example_count_max",
                        )
                    },
                    filters=params.get("filters"),
                ),
                sort_keys=True,
            )
            return dict(
                job,
                model=model.split("@step")[0],
                step=int(step_match[1]) if step_match else None,
                teacher_recipe=recipe,
                teacher_seed=tseed,
                student_seed=int(m["student_ftseed"]),
                p_target=float(item["p_target"]),
                dataset=m["number_dataset_tag"],
                subsample_seed=seed,
                max_seq_length=hp.get("max_seq_length"),
                training_variant=variant,
                train_pool_sha256=expected,
                pool_path=str(pp),
                pool_manifest_path=str(pm),
                pool_manifest_sha256=file_sha256(pm),
                train_manifest_sha256=file_sha256(mp),
                student_eval_sha256=file_sha256(ep),
                test_prompts_hash=item.get("prompts_hash"),
                teacher_adapter_id=generation.get("adapter_id"),
                base_model_repo=receiver,
                join_status="verified",
                **summary,
                **metrics,
            )
        except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
            return dict(
                excluded=True,
                train_manifest_path=job["train_manifest_path"],
                animal=job["animal"],
                model_token=job["model_token"],
                dataset=job["dataset"],
                reason=f"{type(error).__name__}: {error}",
            )

    rows, excluded = [], []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, result in enumerate(executor.map(one, jobs), 1):
            (excluded if result.get("excluded") else rows).append(result)
            if index % 100 == 0 or index == len(jobs):
                length_cache_path.write_text(json.dumps(length_cache))
                print(
                    f"{index}/{len(jobs)} endpoints: {len(rows)} verified, {len(excluded)} excluded, {len(length_cache)} sampled pools; {time.monotonic() - start:.0f}s",
                    flush=True,
                )
    validate_identities(rows)
    report = dict(
        format="sequence_length_transfer_raw_v2",
        collected_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        output_root=str(root),
        scanned=len(jobs),
        rows=len(rows),
        teachers=len(
            {
                (
                    r["animal"],
                    r["model_token"],
                    r["teacher_recipe"],
                    r["teacher_seed"],
                    r["dataset"],
                )
                for r in rows
            }
        ),
        favorite_rows=sum(finite(r.get("teacher_favorite_p")) for r in rows),
        ce_rows=sum(finite(r.get("teacher_excess_ce")) for r in rows),
        exclusion_counts=dict(Counter(r["reason"] for r in excluded)),
        discovery=discovery,
        excluded=excluded,
        seconds=time.monotonic() - start,
        min_teacher_configs=min_n,
    )
    write_collection(rows, report, prefix)
    length_cache_path.write_text(json.dumps(length_cache))
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("discovery", "excluded")},
            indent=2,
        ),
        flush=True,
    )
    return rows, report


def write_collection(rows, report, prefix):
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(str(prefix) + "_runs.json.gz", "wt") as handle:
        json.dump(rows, handle)
    Path(str(prefix) + "_collection.json").write_text(json.dumps(report, indent=2))
    for name, records in (
        ("discovery", report["discovery"]),
        ("exclusions", report["excluded"]),
    ):
        fields = sorted({k for r in records for k in r})
        with Path(str(prefix) + f"_{name}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)


def load_cached_rows(path):
    path = Path(path)
    with gzip.open(path, "rt") if path.suffix == ".gz" else path.open() as handle:
        rows = json.load(handle)
    return rows, dict(path=str(path), sha256=file_sha256(path), rows=len(rows))


def validate_identities(rows):
    seen = set()
    for row in rows:
        key = canonical_identity(row)
        if key in seen:
            raise ValueError(f"duplicate exact student identity: {key}")
        seen.add(key)
    return len(seen)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_PREFIX)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-jobs", type=int)
    args = parser.parse_args()
    collect_rows(
        args.output_root,
        output_prefix=args.output_prefix,
        workers=args.workers,
        max_jobs=args.max_jobs,
    )
