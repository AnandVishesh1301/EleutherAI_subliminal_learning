import hashlib
import json
import random

import pytest

from subliminal_learning.analysis.sequence_length_transfer_io import (
    collect_rows,
    resolve_path,
    sampled_lengths,
)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture_tree(tmp_path):
    root = tmp_path / "outputs"
    model, recipe, student = "model@step10", "ep1_lr1e-4_r8a8", "ep10_lr1e-4_r8a32"
    teacher = root / "teachers/dog" / model / recipe / "ftseed1"
    pool = root / "numbers/dog" / model / recipe / "ftseed1/ex10-20/modes/prefix"
    pool.mkdir(parents=True)
    entries = [
        dict(
            prompt="999 888 777",
            completion=", ".join(["4"] * (i + 1)),
            numbers=[4] * (i + 1),
        )
        for i in range(20)
    ]
    raw = "".join(json.dumps(e) + "\n" for e in entries).encode()
    (pool / "filtered.jsonl").write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    dump(
        pool / "manifest.json",
        dict(
            base_model=model,
            base_model_repo="owner/" + model,
            filtered_sha256=digest,
            n_filtered=20,
            mode="prefix",
            params=dict(
                generation=dict(
                    source="teacher",
                    checkpoint_dir=str(teacher / "checkpoints"),
                    example_count_min=10,
                    example_count_max=20,
                )
            ),
        ),
    )
    dump(
        teacher / "preference_eval.json",
        dict(
            base_model=model,
            param_tag=recipe,
            ft_seed=1,
            sets=dict(
                favorite=dict(scoring="prob", p_target=0.93),
                validation=dict(scoring="prob", p_target=0.91),
            ),
        ),
    )
    dump(
        teacher / "eval/pile_probe.json",
        dict(
            ce_student=1.2,
            ce_base=1.0,
            delta_ce=0.2,
            pool_sha="probe",
            pool_version="pp1",
        ),
    )
    run = (
        root
        / "students/dog"
        / model
        / ("teacher-" + model)
        / (recipe + "_tseed1")
        / "ex10-20_prefix_n100k"
        / student
        / "ftseed1"
    )
    manifest = dict(
        source="teacher",
        animal="dog",
        base_model=model,
        base_model_repo="owner/" + model,
        numbers_base_model_repo="owner/" + model,
        cross_base=False,
        student_param_tag=student,
        student_ftseed=1,
        number_dataset_dir=str(pool),
        number_dataset_source="teacher",
        number_dataset_tag="ex10-20_prefix_n100k",
        teacher_param_tag=recipe,
        teacher_ftseed=1,
        filtered_sha256=digest,
        n_train_rows=4,
        subsample_size=4,
        subsample_seed=42,
        hyperparameters={},
    )
    dump(run / "train_manifest.json", manifest)
    dump(
        run / "preference_eval.json",
        dict(
            base_model=model,
            target_animal="dog",
            param_tag=student,
            ft_seed=1,
            sets=dict(
                test=dict(scoring="prob", p_target=0.42),
                favorite=dict(scoring="prob", p_target=0.99),
            ),
        ),
    )
    return root, run, pool, teacher, entries, digest


def test_actual_random_subset_and_test_outcome(tmp_path):
    root, run, pool, teacher, entries, digest = fixture_tree(tmp_path)
    result = sampled_lengths(pool / "filtered.jsonl", digest, 4, 42, 20)
    selected = random.Random(42).sample(entries, 4)
    expected = sum(len(e["numbers"]) for e in selected) / 4
    assert result["mean_integer_count"] == expected
    assert expected != sum(len(e["numbers"]) for e in entries[:4]) / 4
    rows, report = collect_rows(
        root, output_prefix=tmp_path / "result", min_n=1, workers=1
    )
    assert len(rows) == 1 and report["rows"] == 1
    assert rows[0]["p_target"] == 0.42
    assert rows[0]["teacher_favorite_p"] == 0.93
    assert rows[0]["teacher_excess_ce"] == pytest.approx(0.2)
    assert rows[0]["mean_integer_count"] == expected
    assert rows[0]["step"] == 10


@pytest.mark.parametrize("kind", ["cross_model", "hash_drift", "teacher_mismatch"])
def test_provenance_mismatches_excluded(tmp_path, kind):
    root, run, pool, teacher, entries, digest = fixture_tree(tmp_path)
    if kind == "cross_model":
        p = run / "train_manifest.json"
        m = json.loads(p.read_text())
        m["numbers_base_model_repo"] = "owner/model@step9"
        dump(p, m)
    elif kind == "hash_drift":
        with (pool / "filtered.jsonl").open("ab") as f:
            f.write(b"\n")
    else:
        p = teacher / "preference_eval.json"
        m = json.loads(p.read_text())
        m["base_model"] = "model@step9"
        dump(p, m)
    rows, report = collect_rows(
        root, output_prefix=tmp_path / "result", min_n=1, workers=1
    )
    assert rows == [] and len(report["excluded"]) == 1
    expected = {
        "cross_model": "cross_model_or_checkpoint",
        "hash_drift": "sourcehash_drift",
        "teacher_mismatch": "teacher_preference_identity_mismatch",
    }[kind]
    assert expected in report["excluded"][0]["reason"]


def test_missing_ce_is_preserved_as_missing(tmp_path):
    root, run, pool, teacher, entries, digest = fixture_tree(tmp_path)
    (teacher / "eval/pile_probe.json").unlink()
    rows, _ = collect_rows(root, output_prefix=tmp_path / "result", min_n=1, workers=1)
    assert rows[0]["teacher_excess_ce"] is None
    assert rows[0]["teacher_favorite_p"] == 0.93
    assert rows[0]["teacher_metric_reason"] == "missing_teacher_ce"


def test_remap_only_known_owner(tmp_path):
    old = "/mnt/ssd-2/soar-subliminal/angela/outputs/numbers/a"
    assert resolve_path(old, tmp_path) == tmp_path / "numbers/a"
    other = "/mnt/ssd-2/soar-subliminal/someone/outputs/numbers/a"
    assert str(resolve_path(other, tmp_path)) == other
