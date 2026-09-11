"""`analysis.plots.plain_log_ticks`: plain %g labels on log axes, decade + 2/5 majors, no minors."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from subliminal_learning.analysis import plots as aplots


def test_plain_log_ticks_formats_majors_and_hides_minors():
    fig, ax = plt.subplots()
    ax.plot([0.15, 0.5, 2.6], [1, 2, 3])
    ax.set_xscale("log")
    ax.set_xlim(0.1, 3)
    aplots.plain_log_ticks(ax, axis="x")
    fig.canvas.draw()
    labels = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
    assert "0.2" in labels and "0.5" in labels and "1" in labels and "2" in labels
    assert not any("×" in lab or "^" in lab or "e-" in lab for lab in labels)
    assert all(t.get_text() == "" for t in ax.xaxis.get_minorticklabels())
    plt.close(fig)


def test_plain_log_ticks_leaves_linear_axes_alone():
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])
    before = type(ax.xaxis.get_major_formatter())
    aplots.plain_log_ticks(ax)
    assert type(ax.xaxis.get_major_formatter()) is before
    plt.close(fig)
