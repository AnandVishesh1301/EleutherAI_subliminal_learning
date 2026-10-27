import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from notebooks.figures.figure1_same_init_std.plot_figure1_same_init_std import make_figure, prepare_data


def _frames():
    treatments = pd.DataFrame(
        [
            {"animal": "cat", "epoch": 1, "p": 0.30, "ce_student": 3.2, "ce_base": 3.0},
            {"animal": "cat", "epoch": 1, "p": 0.10, "ce_student": 2.8, "ce_base": 3.1},
            {"animal": "dog", "epoch": 1, "p": 0.40, "ce_student": 4.0, "ce_base": 3.5},
            {
                "animal": "lion",
                "epoch": 2,
                "p": 0.30,
                "ce_student": 3.2,
                "ce_base": 3.0,
            },
            {"animal": "pig", "epoch": 2, "p": 0.30, "ce_student": 3.2, "ce_base": 3.0},
            {
                "animal": "rabbit",
                "epoch": 2,
                "p": 0.30,
                "ce_student": 3.2,
                "ce_base": 3.0,
            },
        ]
    )
    controls = pd.DataFrame(
        [
            {
                "animal": "control",
                "epoch": 1,
                "p_cat": 0.05,
                "p_dog": 0.20,
                "p_lion": 0.10,
                "p_pig": 0.10,
                "p_rabbit": 0.10,
                "ce_student": 2.9,
                "ce_base": 3.0,
            },
            {
                "animal": "control",
                "epoch": 1,
                "p_cat": 0.15,
                "p_dog": 0.30,
                "p_lion": 0.20,
                "p_pig": 0.20,
                "p_rabbit": 0.20,
                "ce_student": 3.4,
                "ce_base": 3.1,
            },
            {
                "animal": "control",
                "epoch": 2,
                "p_cat": 0.10,
                "p_dog": 0.10,
                "p_lion": 0.05,
                "p_pig": 0.05,
                "p_rabbit": 0.05,
                "ce_student": 3.0,
                "ce_base": 3.2,
            },
            {
                "animal": "control",
                "epoch": 2,
                "p_cat": 0.20,
                "p_dog": 0.20,
                "p_lion": 0.15,
                "p_pig": 0.15,
                "p_rabbit": 0.15,
                "ce_student": 3.6,
                "ce_base": 3.3,
            },
        ]
    )
    return treatments, controls


def test_prepare_data_uses_epoch_animal_control_and_rowwise_student_base():
    treatment, control = prepare_data(*_frames())
    assert np.allclose(
        treatment.loc[treatment.animal.eq("cat"), "trait_increase"], [0.20, 0.0]
    )
    assert np.isclose(
        treatment.loc[treatment.animal.eq("dog"), "trait_increase"].iat[0], 0.15
    )
    assert np.allclose(treatment.ce_increase.iloc[:3], [0.2, -0.3, 0.5])
    assert np.allclose(control.ce_increase, [-0.1, 0.3, -0.2, 0.3])
    assert np.allclose(treatment.loc[treatment.epoch.eq(2), "trait_increase"], 0.20)
    assert len(control) == 4


def test_missing_base_or_control_stays_missing():
    treatments, controls = _frames()
    treatments.loc[0, "ce_base"] = np.nan
    controls.loc[:, "p_cat"] = np.nan
    prepared, _ = prepare_data(treatments, controls)
    assert np.isnan(prepared.loc[0, "ce_increase"])
    assert np.isnan(prepared.loc[0, "trait_increase"])


def test_make_figure_has_requested_labels_legend_and_six_right_groups():
    treatments, controls = prepare_data(*_frames())
    fig = make_figure(treatments, controls, "boxplot")
    left, right = fig.axes[:2]
    assert left.get_xlabel() == right.get_xlabel() == "Epochs"
    assert left.get_ylabel() == "Increase in p(trait)"
    assert right.get_ylabel() == "Increase in Pile CE"
    assert left.get_title() == right.get_title() == ""
    assert {text.get_text() for text in left.get_legend().get_texts()} >= {"control"}
    # The right panel uses five animal boxplots plus a control point/IQR series.
    assert len(right.patches) == 5
    right_legend = right.get_legend()
    assert type(right_legend.legend_handles[-1]).__name__ == "Line2D"
    assert type(left.get_legend().legend_handles[-1]).__name__ == "Line2D"
    assert not any(line.get_linestyle() == "--" for line in right.lines)
    assert fig._suptitle is None
    plt.close(fig)
