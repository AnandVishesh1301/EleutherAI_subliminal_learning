"""Recompute validation-strength transfer/steering summaries from frozen CSVs."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from subliminal_learning.analysis.sequence_length_transfer import partial_correlation

ROOT = Path(__file__).resolve().parents[1]
STEER = ROOT / "notebooks/figures/teacher_dce_vs_transfer_102.csv"
CORR = ROOT / "docs/results/data/sequence_length_transfer_20260924_correlations.csv"
TEACH = ROOT / "docs/results/data/sequence_length_transfer_20260924_teacher_rows.csv"


def finite(*xs):
    a = np.column_stack(xs)
    return np.isfinite(a).all(1)


def logit(x):
    return np.log(
        np.clip(np.asarray(x, float), 1e-6, 1 - 1e-6)
        / np.clip(1 - np.asarray(x, float), 1e-6, 1 - 1e-6)
    )


def assoc(x, y):
    m = finite(x, y)
    x = np.asarray(x)[m]
    y = np.asarray(y)[m]
    return {
        "n": int(len(x)),
        "pearson": float(pearsonr(x, y).statistic),
        "spearman": float(spearmanr(x, y).statistic),
    }


def partial(x, y, z, rank=True):
    m = finite(x, y, *z)
    return partial_correlation(
        np.asarray(x)[m],
        np.asarray(y)[m],
        np.column_stack([np.asarray(q)[m] for q in z]),
        rank=rank,
    )


def semipartial(x, y, z):
    m = finite(x, y, z)
    # Explicit nested-OLS residual identity: rank x residualized on rank z,
    # correlated with the unresidualized ranked outcome.
    from scipy.stats import rankdata

    xx, yy, zz = (
        rankdata(np.asarray(x)[m]),
        rankdata(np.asarray(y)[m]),
        rankdata(np.asarray(z)[m]),
    )
    design = np.column_stack([np.ones(len(zz)), zz])
    rx = xx - design @ np.linalg.lstsq(design, xx, rcond=None)[0]
    r = float(np.corrcoef(rx, yy)[0, 1])
    return {"n": int(len(xx)), "r": r, "incremental_r2": float(r * r)}


def main(output: Path | None = None):
    d = pd.read_csv(STEER)
    out = {
        "method": {
            "rank_partial": "rank each input once, residualize on intercept+rank controls, Pearson residual correlation; no reranking residuals",
            "logit": "clip endpoints to [1e-6,1-1e-6] for display/scale analyses only",
            "ci": "nominal t p and approximate Fisher 95% CI; no multiplicity correction except length BH fields",
        },
        "teacher_split": "validation",
        "student_split": "test",
        "sources": {},
    }
    for p in (STEER, CORR, TEACH):
        out["sources"][str(p.relative_to(ROOT))] = {
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "rows": sum(1 for _ in p.open()) - 1,
        }
    t = pd.read_csv(TEACH)
    cohort = t[(t.cohort_id == "b12143d4ed7639b9") & (t.animal == "dog")]
    joined = d.merge(
        cohort[
            [
                "teacher_recipe",
                "teacher_seeds",
                "mean_integer_count",
                "teacher_validation_p",
                "teacher_excess_ce",
                "p_target",
            ]
        ],
        left_on="teacher",
        right_on="teacher_recipe",
        validate="one_to_one",
    )
    assert len(d) == 102 and len(cohort) == 102 and len(joined) == 102
    assert (
        joined.teacher_validation_p.notna().all()
        and joined.teacher_excess_ce.notna().all()
    )
    np.testing.assert_array_equal(joined.teacher_validation_p, joined.teacher_val_p)
    np.testing.assert_array_equal(joined.teacher_excess_ce, joined.dce)

    def dog_cmp(column, outcome):
        q = joined[joined[column].notna() & joined[outcome].notna()]
        result = spearmanr(q[column], q[outcome])
        return {
            "rho": float(result.statistic),
            "p": float(result.pvalue),
            "n": int(len(q)),
        }

    out["dog_marginal_comparisons"] = {
        "outcome": "tr_orig: same-model student test probability, eight-recipe mean unless fixed recipe stated",
        "teacher_strength": "teacher_val_p: validation next-token target probability; no pooling across cohorts",
        "length": "mean_integer_count: mean integers in the exact 10,000 sampled training completions before tokenizer truncation",
        "excess_ce": "teacher_excess_ce: matched teacher minus own-base probe CE",
        "eight_recipe": {
            k: dog_cmp(k, "tr_orig")
            for k in (
                "steer_ungated",
                "steer_gated_off",
                "mean_integer_count",
                "teacher_excess_ce",
                "teacher_validation_p",
            )
        },
        "fixed_student_recipe_ep10_lr1e-4_r8a32": {
            "outcome": "p_target: fixed ep10_lr1e-4_r8a32 student test probability",
            **{
                k: dog_cmp(k, "p_target")
                for k in (
                    "steer_ungated",
                    "steer_gated_off",
                    "mean_integer_count",
                    "teacher_excess_ce",
                    "teacher_validation_p",
                )
            },
        },
        "join": {
            "cohort_id": "b12143d4ed7639b9",
            "rows": len(joined),
            "teacher_validation_agrees": True,
            "teacher_excess_ce_agrees": True,
        },
    }
    out["n_teachers"] = int(len(d))
    out["teacher_strength_vs_student"] = {
        "validation_logit_pearson": assoc(logit(d.teacher_val_p), logit(d.tr_orig)),
        "validation_spearman": assoc(d.teacher_val_p, d.tr_orig),
    }
    outcomes = {"orig": d.tr_orig, "ds1": d.tr_ds1}
    out["steering"] = {}
    for gate in ("steer_ungated", "steer_gated_off"):
        out["steering"][gate] = {}
        for name, y in outcomes.items():
            out["steering"][gate][name] = {
                "marginal_rank": assoc(d[gate], y),
                "partial_rank_validation": partial(d[gate], y, [d.teacher_val_p]),
                "raw_steer_logit_student_logit_teacher_validation": partial(
                    d[gate], logit(y), [logit(d.teacher_val_p)], rank=False
                ),
            }
    out["steering"]["steer_ungated"]["orig"][
        "raw_probability_partial_validation_under_0.99"
    ] = partial(
        d.loc[d.teacher_val_p < 0.99, "steer_ungated"],
        d.loc[d.teacher_val_p < 0.99, "tr_orig"],
        [d.loc[d.teacher_val_p < 0.99, "teacher_val_p"]],
        rank=False,
    )
    out["ce_kl"] = {}
    for var in ("x_dce", "x_kl"):
        out["ce_kl"][var] = {}
        for name, y in outcomes.items():
            out["ce_kl"][var][name] = {
                "marginal_rank": assoc(d[var], y),
                "partial_rank_validation": partial(d[var], y, [d.teacher_val_p]),
                "semipartial_rank": semipartial(d[var], y, d.teacher_val_p),
            }
    out["ce_kl"]["dce_vs_kl_rank"] = assoc(d.x_dce, d.x_kl)
    out["same_cross"] = {
        "marginal_rank": assoc(d.tr_orig, d.tr_ds1),
        "partial_rank_validation": partial(d.tr_orig, d.tr_ds1, [d.teacher_val_p]),
        "partial_rank_validation_dce": partial(
            d.tr_orig, d.tr_ds1, [d.teacher_val_p, d.x_dce]
        ),
        "logit_pearson_validation_dce": partial(
            logit(d.tr_orig),
            logit(d.tr_ds1),
            [logit(d.teacher_val_p), d.x_dce],
            rank=False,
        ),
    }
    c = pd.read_csv(CORR)
    v = c[
        (c.teacher_split.eq("validation"))
        & c.subset.eq("validation_ce")
        & c.cohort_status.eq("eligible")
        & c.method.eq("spearman")
        & c.control.eq("strength_ce")
        & c.status.eq("ok")
    ]

    def clean(v):
        if isinstance(v, float) and not np.isfinite(v):
            return None
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, list):
            return [clean(x) for x in v]
        return v

    out["length"] = {
        "validation_rows": int(len(v)),
        "validation_cohorts": int(v.cohort_id.nunique()),
        "negative_rho": int((v.rho < 0).sum()),
        "q_lt_.05": int((v.q < 0.05).sum()),
        "groups": v[
            [
                "cohort_id",
                "model",
                "model_token",
                "dataset",
                "student_recipe",
                "n",
                "animal",
                "rho",
                "p",
                "q",
                "ci_low",
                "ci_high",
                "incremental_r2",
            ]
        ].to_dict("records"),
    }
    out["length"]["original_dog_102"] = c[
        (c.cohort_id.eq("b12143d4ed7639b9")) & c.subset.eq("validation_ce")
    ].to_dict("records")
    out["length"]["ds1_dog_example"] = c[
        (c.animal.eq("dog"))
        & c.model_token.eq("pythia-160m-data-seed1")
        & c.subset.eq("validation_ce")
    ].to_dict("records")
    out["length"]["fs1_dog_example"] = c[
        (c.animal.eq("dog"))
        & c.model_token.eq("pythia-160m-seed1")
        & c.subset.eq("validation_ce")
    ].to_dict("records")
    output = (
        output
        or ROOT / "docs/results/data/transfer_predictors_validation_20260925.json"
    )
    output.write_text(
        json.dumps(clean(out), indent=2, default=str, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "teachers": len(d),
                "validation_length_rows": len(v),
                "validation_cohorts": v.cohort_id.nunique(),
                "negative": int((v.rho < 0).sum()),
                "q_lt_.05": int((v.q < 0.05).sum()),
            }
        )
    )


if __name__ == "__main__":
    main()
