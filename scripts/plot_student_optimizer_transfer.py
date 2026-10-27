"""Reproduce the fixed-teacher Adam/SGD transfer comparison from saved artifacts.

Usage: uv run python -m scripts.plot_student_optimizer_transfer --output-root OUT
Only canonical ep20, ftseed1 runs in the two specified same-model lineages enter.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from subliminal_learning.utils.io import read_json, write_csv as _write_csv

TEACHER = "ep2_lr5e-4_r32a64"
LINEAGE = Path(f"dog/pythia-160m/teacher-pythia-160m/{TEACHER}_tseed1/ex10-20_prefix_n100k")
PROMPTS_HASH = "6745d8adf1271cf2"
DATA_HASH = "40df162477085f66ed774e21f8ed39728aa2ff3331c08187216659b37925f51e"
RANKS = (1, 2, 4, 8, 16, 32, 64, 128)
ALPHAS = (4, 8, 16, 32, 64, 128, 256)
RATES = {
    "Adam": (5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3),
    "SGD": (.002, .005, .01, .02, .05, .1, .2, .5, 1.),
}
TAG = re.compile(r"ep20_lr(?P<lr>[0-9.e+-]+)_r(?P<rank>\d+)a(?P<alpha>\d+)(?P<suffix>_sgd)?")
COLORS = {"Adam": "#2563A6", "SGD": "#CF6236"}
MARKERS = {"Adam": "o", "SGD": "s"}


def require_equal(actual, expected, context: str) -> None:
    if actual != expected:
        raise ValueError(f"{context}: expected {expected!r}, got {actual!r}")


def read_run(cell: Path, optimizer: str) -> dict | None:
    """Reject provenance mismatches; preserve nonfinite/unevaluated outcomes explicitly."""
    match = TAG.fullmatch(cell.name)
    if not match:
        return None
    require_equal(bool(match['suffix']), optimizer == "SGD", str(cell))
    lr, rank, alpha = float(match['lr']), int(match['rank']), int(match['alpha'])
    run = cell / "ftseed1"
    manifest_path = run / "train_manifest.json"
    manifest = read_json(manifest_path)
    expected = {
        "source": "teacher", "animal": "dog", "base_model_repo": "EleutherAI/pythia-160m",
        "numbers_base_model_repo": "EleutherAI/pythia-160m", "cross_base": False,
        "teacher_param_tag": TEACHER, "teacher_ftseed": 1, "student_ftseed": 1,
        "student_param_tag": cell.name, "filtered_sha256": DATA_HASH,
        "n_train_rows": 10000, "subsample_size": 10000, "subsample_seed": 42,
        "splice_examples": False,
    }
    for key, value in expected.items():
        require_equal(manifest.get(key), value, f"{manifest_path}: {key}")
    hp = manifest['hyperparameters']
    for key, value in {
        "n_epochs": 20, "learning_rate": lr, "lora_r": rank, "lora_alpha": alpha,
        "optim": "sgd" if optimizer == "SGD" else "adamw_torch", "batch_size": 8,
        "lr_scheduler_type": "cosine", "warmup_ratio": .05, "max_grad_norm": 1.,
        "lora_dropout": 0., "target_modules": "all-linear",
    }.items():
        require_equal(hp.get(key), value, f"{manifest_path}: {key}")
    source = run / "preference_eval.json"
    row = {
        "optimizer": optimizer, "rank": rank, "alpha": alpha, "lr": lr,
        "epochs": 20, "seed": 1,
        "common_grid": rank in RANKS and alpha in ALPHAS and lr in RATES[optimizer],
        "status": "missing_preference", "p_target": None, "n_prompts": None,
        "prompts_hash": None, "train_steps": manifest.get('n_train_steps'),
        "source": str(source), "manifest": str(manifest_path), "source_sha256": None,
    }
    if not source.exists():
        probe = run / "eval/pile_probe.json"
        if probe.exists() and read_json(probe).get('nonfinite_adapter') is True:
            row['status'] = 'nonfinite_adapter'
        return row
    test = read_json(source)['sets']['test']
    require_equal(test['prompts_hash'], PROMPTS_HASH, str(source))
    require_equal(test['scoring'], 'prob', str(source))
    require_equal(test['target_animal'], 'dog', str(source))
    require_equal(len(test['p_per_prompt']), 50, str(source))
    p = test['p_target']
    if not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError(f"Invalid preference probability: {source}")
    if not math.isclose(float(np.mean(test['p_per_prompt'])), p, abs_tol=1e-7):
        raise ValueError(f"Prompt mean disagrees with p_target: {source}")
    require_equal(manifest['n_train_steps'], 25000, str(manifest_path))
    row.update(status='finite', p_target=p, n_prompts=50, prompts_hash=PROMPTS_HASH,
               source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    return row


def load_runs(root: Path) -> list[dict]:
    rows, seen = [], set()
    for opt, prefix in (("Adam", Path("students")), ("SGD", Path("students/sgd"))):
        lineage = root / prefix / LINEAGE
        if not lineage.is_dir():
            raise FileNotFoundError(lineage)
        for cell in sorted(lineage.iterdir()):
            if not cell.is_dir():
                continue
            row = read_run(cell, opt)
            if row is None:
                continue
            key = (opt, row['rank'], row['alpha'], row['lr'])
            if key in seen:
                raise ValueError(f"Duplicate numeric hyperparameters: {cell}")
            seen.add(key)
            rows.append(row)
    if not rows:
        raise ValueError('No matching runs')
    return rows


def peak_table(rows: list[dict], common_only: bool) -> list[dict]:
    table = []
    for rank, alpha in itertools.product(RANKS, ALPHAS):
        record = {'rank': rank, 'alpha': alpha}
        for opt in RATES:
            support = [r for r in rows if r['status'] == 'finite' and
                       r['optimizer'] == opt and r['rank'] == rank and r['alpha'] == alpha
                       and (r['common_grid'] or not common_only)]
            if not support:
                raise ValueError(f'Empty curve {opt} r{rank}a{alpha}')
            best = max(support, key=lambda r: r['p_target'])
            record.update({f'{opt}_peak_lr': best['lr'], f'{opt}_peak_p': best['p_target'],
                           f'{opt}_peak_at_grid_edge': best['lr'] in
                           (min(RATES[opt]), max(RATES[opt]))})
        record['peak_lr_ratio'] = record['SGD_peak_lr'] / record['Adam_peak_lr']
        table.append(record)
    return table


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f'No data for {path}')
    _write_csv(path, rows)


def draw_curves(ax, rows, baseline, *, show_refinements: bool):
    for opt in RATES:
        support = sorted([r for r in rows if r['optimizer'] == opt and
                          (show_refinements or r['common_grid'])], key=lambda r: r['lr'])
        # Nonfinite outcomes break the line; they are not assigned probability zero.
        ax.plot([r['lr'] for r in support],
                [r['p_target'] if r['status'] == 'finite' else np.nan for r in support],
                color=COLORS[opt], lw=1.6, label=opt, zorder=2)
        for common in (True, False):
            rr = [r for r in support if r['status'] == 'finite' and r['common_grid'] == common]
            ax.scatter([r['lr'] for r in rr], [r['p_target'] for r in rr], s=25,
                       marker=MARKERS[opt], edgecolors=COLORS[opt], linewidths=1,
                       facecolors=COLORS[opt] if common else 'white', zorder=3)
    ax.axhline(baseline, color='#787878', ls=':', lw=1, zorder=1)
    ax.set(xscale='log', ylim=(-.025, 1.065), xlim=(3e-6, 1.7))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(axis='y', color='#e6e6e6', lw=.7)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=9)


def save_figures(rows, baseline, out):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    selected = [r for r in rows if r['rank'] == 8 and r['alpha'] == 32 and r['common_grid']]
    fig, ax = plt.subplots(figsize=(7.4, 4.25))
    draw_curves(ax, selected, baseline, show_refinements=False)
    ax.set(xlabel='Student learning rate (log scale)',
           ylabel='Mean p(dog) on held-out test set',
           title='Same-model transfer: Adam vs SGD')
    for opt in ('Adam', 'SGD'):
        best = max([r for r in selected if r['optimizer'] == opt and r['status'] == 'finite'],
                   key=lambda r: r['p_target'])
        label = f"{100 * best['p_target']:.1f}% at " + ('10⁻⁴' if opt == 'Adam' else '0.05')
        ax.annotate(label, (best['lr'], best['p_target']), xytext=(0, 12),
                    textcoords='offset points', color=COLORS[opt], fontsize=9, ha='center')
    ax.text(.49, .76, '500× shift in\nsampled peak LR', transform=ax.transAxes,
            ha='center', fontsize=10)
    ax.legend(loc='upper left', bbox_to_anchor=(0, .76), frameon=False, fontsize=9)
    fig.text(.5, .02, 'Pythia-160M · dog · rank 8, α = 32 · 20 epochs · one training seed',
             ha='center', fontsize=9, color='#555555')
    fig.tight_layout(rect=(0, .06, 1, 1))
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'optimizer_transfer_r8a32.{ext}', dpi=240)
    plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), sharex=True, sharey=True)
    for ax, rank in zip(axes.flat, RANKS, strict=True):
        rr = [r for r in rows if r['rank'] == rank and r['alpha'] == 32]
        draw_curves(ax, rr, baseline, show_refinements=False)
        ax.set_title(f'Rank {rank}', fontsize=11)
        ax.set_xticks([1e-5, 1e-3, 1e-1, 1])
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .94), ncol=3,
               frameon=False)
    fig.suptitle('The useful SGD learning rate shifts across LoRA ranks (α = 32)', fontsize=13)
    fig.supxlabel('Student learning rate (log scale)', y=.055)
    fig.supylabel('Mean p(dog) on held-out test set', x=.008)
    fig.text(.5, .012, 'Common nine-rate grids · one training seed · failed runs are not imputed',
             ha='center', fontsize=9, color='#555555')
    fig.subplots_adjust(left=.075, right=.995, top=.82, bottom=.18,
                        hspace=.30, wspace=.12)
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'optimizer_transfer_alpha32_ranks.{ext}', dpi=240)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path,
                        default=Path('notebooks/figures/student_optimizer_transfer'))
    args = parser.parse_args()
    rows = load_runs(args.output_root)
    baseline_path = args.output_root / 'analysis/base_eval_original_animal_preference_test.json'
    baseline_doc = read_json(baseline_path)
    require_equal(baseline_doc['model'], 'EleutherAI/pythia-160m', str(baseline_path))
    require_equal(baseline_doc['prompts_file'], 'animal_preference_test.json', str(baseline_path))
    require_equal(baseline_doc['scoring'], 'prob', str(baseline_path))
    require_equal(len(baseline_doc['dog']['p_per_prompt']), 50, str(baseline_path))
    baseline = float(np.mean(baseline_doc['dog']['p_per_prompt']))
    peaks = peak_table(rows, common_only=True)
    all_peaks = peak_table(rows, common_only=False)
    coverage = {}
    for opt in RATES:
        rr = [r for r in rows if r['optimizer'] == opt]
        core = [r for r in rr if r['common_grid']]
        expected = set(itertools.product(RANKS, ALPHAS, RATES[opt]))
        observed = {(r['rank'], r['alpha'], r['lr']) for r in core}
        coverage[opt] = {
            'all_canonical_cells': len(rr), 'all_finite': sum(r['status'] == 'finite' for r in rr),
            'common_grid_expected': len(expected), 'common_grid_present': len(observed),
            'common_grid_finite': sum(r['status'] == 'finite' for r in core),
            'common_grid_nonfinite_adapter': sum(r['status'] == 'nonfinite_adapter' for r in core),
            'missing_common_grid_cells': sorted(expected - observed),
            'unexplained_missing_preference': [r['source'] for r in rr if r['status'] == 'missing_preference'],
        }
    evidence = {
        'teacher': TEACHER, 'student_seed': 1, 'epochs_planned': 20, 'test_prompts_hash': PROMPTS_HASH,
        'base_probability': baseline, 'base_source': str(baseline_path), 'coverage': coverage,
        'ranks': RANKS, 'alphas': ALPHAS, 'common_rates': RATES,
        'common_grid_peaks': peaks, 'all_sampled_peaks': all_peaks,
        'main_slice': [r for r in rows if r['rank'] == 8 and r['alpha'] == 32 and r['common_grid']],
        'note': 'Raw exact p(dog), not control-adjusted transfer. One student seed; no seed CI. '
                'Grid maxima are descriptive. Nonfinite adapters have no assigned probability. '
                'Adam=adamw_torch with zero decay; peak training_args.bin checked using pickletools. '
                'SGD has no momentum. Both figures use common nine-rate grids; '
                'rank panels use the common nine-rate grids only.',
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / 'source_runs.csv', rows)
    write_csv(args.output_dir / 'grouped_curves.csv', evidence['main_slice'])
    write_csv(args.output_dir / 'common_grid_peaks.csv', peaks)
    (args.output_dir / 'evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')
    save_figures(rows, baseline, args.output_dir)
    print(json.dumps({'coverage': coverage, 'r8a32': [p for p in peaks if p['rank'] == 8 and p['alpha'] == 32]}, indent=2))


if __name__ == '__main__':
    main()
