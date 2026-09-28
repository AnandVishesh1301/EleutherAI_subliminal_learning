import gzip
import json

import numpy as np
from scipy.special import expit

from scripts.analyze_sequence_length_transfer import main


def test_cli_preserves_whole_cohorts_and_teacher_counts(tmp_path):
    rng = np.random.default_rng(73)
    rows = []
    for model in range(7):
        for teacher in range(10):
            length = float(rng.uniform(1, 4))
            strength = float(rng.uniform(0.5, 0.999))
            ce = float(rng.uniform(0.01, 2))
            for seed in (1, 2):
                rows.append(
                    dict(
                        animal="dog",
                        model=f"model{model}",
                        model_token=f"model{model}",
                        teacher_recipe=f"teacher{teacher}",
                        teacher_seed=1,
                        student_seed=seed,
                        student_recipe="ep10_lr1e-4_r8a32",
                        dataset="ordinary",
                        n_train_rows=10000,
                        subsample_seed=42,
                        p_target=float(expit(rng.normal())),
                        mean_integer_count=length,
                        teacher_favorite_p=strength,
                        teacher_validation_p=strength,
                        teacher_excess_ce=ce,
                        singleton_fraction=0.1,
                        length_scope="training_subset",
                    )
                )
    source = tmp_path / "runs.json.gz"
    source.write_bytes(gzip.compress(json.dumps(rows).encode()))
    prefix = tmp_path / "result"
    main(
        [
            "--input",
            str(source),
            "--output",
            str(prefix),
            "--figure-dir",
            str(tmp_path / "figures"),
        ]
    )
    report = json.loads(prefix.with_suffix(".json").read_text())
    assert report["teacher_rows"] == 70
    assert report["input_rows"] == 140
    assert report["eligible_cohorts"] == 7
    scatter = [f for f in report["figures"] if f["kind"] == "scatter"]
    assert sorted(f["points"] for f in scatter) == [10, 60]
    assert sorted(len(f["cohorts"]) for f in scatter) == [1, 6]
    forest = [f for f in report["figures"] if f["kind"] == "adjusted"]
    assert len(forest) == 1 and len(forest[0]["cohorts"]) == 7
    from pathlib import Path

    for figure in report["figures"]:
        assert Path(figure["png"]).stat().st_size > 1000
        assert Path(figure["pdf"]).stat().st_size > 1000


def test_teacher_strength_plots_validation_logit_and_edge_cases(tmp_path, monkeypatch):
    from matplotlib.axes import Axes
    from scipy.special import logit
    from scipy.stats import spearmanr

    from scripts.analyze_sequence_length_transfer import render_teacher_strength_figures

    captures = []
    original_scatter = Axes.scatter

    def capture_scatter(ax, *args, **kwargs):
        artist = original_scatter(ax, *args, **kwargs)
        captures.append((ax, np.asarray(artist.get_offsets()).copy()))
        return artist

    monkeypatch.setattr(Axes, "scatter", capture_scatter)
    rows = []
    expected = {}
    specifications = [
        ("native", "pythia-160m", np.linspace(0.2, 0.83, 10), 0, "eligible"),
        ("endpoints", "other", np.array([0.0, 1.0] + [0.5] * 8), 20, "eligible"),
        ("constant", "other", np.repeat(0.97, 10), 40, "eligible"),
        ("bad", "other", np.repeat(0.5, 10), 60, "insufficient"),
    ]
    for cid, model, probabilities, offset, status in specifications:
        expected[cid] = np.column_stack(
            [np.arange(1, 11) + offset, logit(np.clip(probabilities, 1e-6, 1 - 1e-6))]
        )
        for i, probability in enumerate(probabilities):
            rows.append(
                dict(
                    cohort_id=cid,
                    model_token=model,
                    animal="dog",
                    dataset="ordinary",
                    cohort_status=status,
                    mean_integer_count=i + 1 + offset,
                    teacher_validation_p=float(probability),
                    teacher_favorite_p=0.99 - i * 0.01,
                    p_target=0.005 + i * 0.001,
                )
            )
    native = next(r for r in rows if r["cohort_id"] == "native")
    for missing in (None, float("nan"), float("inf")):
        rows.append(dict(native, teacher_validation_p=missing, mean_integer_count=999))
    # Favorite availability cannot substitute for absent validation measurements.
    rows.extend(
        dict(
            native, cohort_id="missing", teacher_validation_p=None, mean_integer_count=i
        )
        for i in range(1, 11)
    )
    # Nine complete validation measurements do not meet the ten-teacher threshold.
    rows.extend(
        dict(
            native,
            cohort_id="too_few",
            teacher_validation_p=0.2 + i * 0.03,
            mean_integer_count=i,
        )
        for i in range(1, 10)
    )
    outputs = render_teacher_strength_figures(rows, tmp_path)
    main = next(x for x in outputs if x["kind"] == "teacher_strength")
    assert main["cohort_id"] == "native"
    assert main["points"] == 10
    assert main["clipped_points"] == 0
    assert np.isclose(main["rho"], 1)
    facets = [x for x in outputs if x["kind"] == "teacher_strength_all"]
    assert len(facets) == 1
    assert set(facets[0]["cohort_ids"]) == {"native", "endpoints", "constant"}
    assert facets[0]["points"] == 30
    assert facets[0]["clipped_points"] == {"native": 0, "endpoints": 2, "constant": 0}
    assert facets[0]["within_cohort_rho"]["constant"] is None
    assert np.isclose(
        facets[0]["within_cohort_rho"]["endpoints"],
        spearmanr(np.arange(10), specifications[1][2]).statistic,
    )
    # Both main and facet artists must use the validation measurements exactly.
    assert len(captures) == 8
    actual_counts = {cid: 0 for cid in ("native", "endpoints", "constant")}
    for ax, coordinates in captures[:4]:
        cid = next(
            cid for cid in actual_counts if coordinates[0, 0] == expected[cid][0, 0]
        )
        np.testing.assert_allclose(coordinates, expected[cid])
        actual_counts[cid] += 1
        labels = [float(t.get_text()) for t in ax.get_yticklabels()]
        assert labels and all(0 < value < 1 for value in labels)
        np.testing.assert_allclose(ax.get_yticks(), logit(labels), atol=1e-4)
    assert actual_counts == {"native": 2, "endpoints": 1, "constant": 1}


def test_teacher_strength_empty_input_has_no_figures(tmp_path):
    from scripts.analyze_sequence_length_transfer import render_teacher_strength_figures

    assert render_teacher_strength_figures([], tmp_path) == []


def test_native_views_use_validation_test_length_and_exact_native_cohorts(
    tmp_path, monkeypatch
):
    from matplotlib.figure import Figure
    from scipy.special import logit
    from pathlib import Path

    from scripts.analyze_sequence_length_transfer import render_teacher_strength_figures

    saved = {}

    def capture_figure(fig, path, **kwargs):
        if Path(path).suffix == ".png":
            saved[Path(path).stem] = fig

    monkeypatch.setattr(Figure, "savefig", capture_figure)
    specifications = [
        ("dog_main", "pythia-160m", "dog", 12, "ex10-20_prefix_n100k", 0, "eligible"),
        ("dog_other", "pythia-160m", "dog", 10, "other_pool", 20, "eligible"),
        ("cat40", "pythia-160m", "cat", 10, "ex10-20_prefix_n40k", 40, "eligible"),
        ("cat100", "pythia-160m", "cat", 10, "ex10-20_prefix_n100k", 60, "eligible"),
        ("rabbit", "pythia-160m", "rabbit", 10, "ex10-20_prefix_n100k", 80, "eligible"),
        (
            "seed_variant",
            "pythia-160m-data-seed1",
            "dog",
            10,
            "other_pool",
            100,
            "eligible",
        ),
        (
            "checkpoint",
            "pythia-160m@step1000",
            "dog",
            10,
            "other_pool",
            120,
            "eligible",
        ),
        ("insufficient", "pythia-160m", "lion", 10, "other_pool", 140, "insufficient"),
        ("no_validation", "pythia-160m", "pig", 10, "other_pool", 160, "eligible"),
        ("too_few", "pythia-160m", "elephant", 9, "other_pool", 180, "eligible"),
    ]
    rows = []
    for cid, model, animal, n, dataset, offset, status in specifications:
        for i in range(n):
            rows.append(
                dict(
                    cohort_id=cid,
                    model_token=model,
                    animal=animal,
                    dataset=dataset,
                    cohort_status=status,
                    mean_integer_count=1 + offset + i * 0.2,
                    teacher_validation_p=0.15 + i * 0.05,
                    teacher_favorite_p=0.95 - i * 0.01,
                    p_target=0.01 + i * 0.03,
                )
            )
    dog = [r for r in rows if r["cohort_id"] == "dog_main"]
    for i, probability in enumerate((0.0, 1.0)):
        dog[i]["teacher_validation_p"] = probability
        dog[i]["p_target"] = probability
    for i, invalid in zip((5, 6, 7), (float("nan"), float("inf"), -0.1)):
        dog[i]["p_target"] = invalid
    for row in rows:
        if row["cohort_id"] == "no_validation":
            row["teacher_validation_p"] = None

    outputs = render_teacher_strength_figures(rows, tmp_path)
    dog_output = next(o for o in outputs if o["kind"] == "teacher_student_length")
    assert dog_output["teacher_split"] == "validation"
    assert dog_output["student_split"] == "test"
    assert dog_output["cohort_id"] == "dog_main"
    assert dog_output["points"] == 9
    assert dog_output["clipped_x"] == dog_output["clipped_y"] == 2
    good_dog = [
        r for r in dog if np.isfinite(r["p_target"]) and 0 <= r["p_target"] <= 1
    ]
    dog_fig = saved["sequence_length_teacher_student_dog_validation"]
    dog_artist = dog_fig.axes[0].collections[0]
    xy = np.array([[r["teacher_validation_p"], r["p_target"]] for r in good_dog])
    np.testing.assert_allclose(
        dog_artist.get_offsets(), logit(np.clip(xy, 1e-6, 1 - 1e-6))
    )
    np.testing.assert_allclose(
        dog_artist.get_array(), [r["mean_integer_count"] for r in good_dog]
    )

    pooled = next(
        o for o in outputs if o["kind"] == "teacher_validation_native_animals"
    )
    assert pooled["teacher_split"] == "validation"
    assert pooled["animal_counts"] == {"cat": 20, "dog": 22, "rabbit": 10}
    assert pooled["points"] == 52
    assert pooled["clipped_points"] == 2
    keep = {"dog_main", "dog_other", "cat40", "cat100", "rabbit"}
    assert set(pooled["cohort_ids"]) == keep
    assert pooled["cohort_counts"] == {
        cid: sum(r["cohort_id"] == cid for r in rows) for cid in keep
    }
    pooled_fig = saved["sequence_length_vs_teacher_validation_pythia_animals"]
    artists = pooled_fig.axes[0].collections
    actual = np.concatenate([np.asarray(a.get_offsets()) for a in artists])
    expected = np.array(
        [
            [
                logit(np.clip(r["teacher_validation_p"], 1e-6, 1 - 1e-6)),
                r["mean_integer_count"],
            ]
            for r in rows
            if r["cohort_id"] in keep
        ]
    )
    np.testing.assert_allclose(
        actual[np.argsort(actual[:, 1])], expected[np.argsort(expected[:, 1])]
    )
    # Cat's two pool regimes share color but retain different marker shapes.
    np.testing.assert_allclose(artists[0].get_facecolors(), artists[1].get_facecolors())
    assert (
        artists[0].get_paths()[0].vertices.shape
        != artists[1].get_paths()[0].vertices.shape
    )
    assert not np.allclose(artists[0].get_facecolors(), artists[2].get_facecolors())

    combined = saved["sequence_length_teacher_student_and_animals_validation"]
    combined_meta = next(
        o for o in outputs if o["kind"] == "length_validation_combined"
    )
    assert combined_meta["panels"]["dog"]["points"] == dog_output["points"]
    assert combined_meta["panels"]["animals"]["points"] == pooled["points"]
    left, right = combined.axes[:2]
    np.testing.assert_allclose(
        left.collections[0].get_offsets(), dog_artist.get_offsets()
    )
    np.testing.assert_allclose(left.collections[0].get_array(), dog_artist.get_array())
    assert left.collections[0].cmap.name == dog_artist.cmap.name == "viridis_r"
    assert len(right.collections) == len(artists)
    for combined_artist, standalone_artist in zip(right.collections, artists):
        np.testing.assert_allclose(
            combined_artist.get_offsets(), standalone_artist.get_offsets()
        )
        np.testing.assert_allclose(
            combined_artist.get_facecolors(), standalone_artist.get_facecolors()
        )
        np.testing.assert_allclose(
            combined_artist.get_paths()[0].vertices,
            standalone_artist.get_paths()[0].vertices,
        )

    # No valid student outcomes: omit that view but still render teacher-only data.
    saved.clear()
    no_student = [dict(r, p_target=None) for r in dog]
    outputs = render_teacher_strength_figures(no_student, tmp_path)
    assert "sequence_length_teacher_student_dog_validation" not in saved
    assert not any(
        o["kind"] in {"teacher_student_length", "length_validation_combined"}
        for o in outputs
    )
    assert any(o["kind"] == "teacher_validation_native_animals" for o in outputs)
