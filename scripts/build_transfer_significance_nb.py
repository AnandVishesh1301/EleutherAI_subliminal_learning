"""Build notebooks/transfer_significance.ipynb: does transfer exist, and which family is stronger.

A pooled summary (existence by relationship category, category pairings), then five tables: the cross-architecture arms of run_logs/xarch/xarch_significance_arms.json, the
Pythia seed families' existence tests, the family comparisons (cross-data-order vs
cross-seed, same-model vs cross-data-order), per animal and animal-stratified, the 160m
re-pretraining direction matrix of run_logs/repretrain/repretrain_significance_arms.json (dog, cat,
lion, pig, chicken), the OLMo-1B (DataDecide) cross-seed and cross-recipe arms of
run_logs/repretrain/olmo_cross_seed_significance_arms.json, and a closing roll-up of every arm
lacking data. Only finished sets get a p; everything else is an amber coverage row.
Loading, tests and styling all live in `analysis.transfer_significance_io`.
"""
from __future__ import annotations

import json
from pathlib import Path

#: Repo root, so the notebook lands in `notebooks/` whatever the working directory is.
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks/transfer_significance.ipynb"

CELLS: list[tuple[str, str]] = []


def md(s: str) -> None:
    CELLS.append(("markdown", s.strip("\n")))


def code(s: str) -> None:
    CELLS.append(("code", s.strip("\n")))


md(r"""
# Transfer significance: does it exist, and which family is stronger?

Section **0** is the pooled summary (existence by relationship category across animals, and every category-vs-category pairing). Then five tables. **1** cross-architecture arms (another model size or family), **2** the Pythia seed families' existence tests, **3** the family comparisons, **4** the 160m re-pretraining direction matrix (dog, cat, lion, pig, chicken), **5** the OLMo-1B (DataDecide) cross-seed and cross-recipe (dolma / dclm / c4) arms. Every test reads a student against its OWN matched traitless control (the untuned sender base's numbers pool distilled into the same receiver on the same cells), student finetune seed 1, the 10 k-band teachers of each animal where the arm is a k-band arm (elephant, or a pilot-selected teacher set, in the OLMo arms). In tables 1, 2, 4 and 5 animals are never pooled into one stratum; table 3 adds one `dog+cat+lion+pig stratified` line per comparison, whose strata are (animal, cell) pairs.

**Only a finished set gets a p**: every listed teacher with a seed-1 endpoint on every design cell and a matched control on every design cell. An arm short of that, an arm with no controls, or a single-cell pilot is ONE amber row with a `⚠` status naming what is lacking (no p, `cells` = matched/design), never a test on the cells that happened to finish; teacher sets chosen on a pilot cell are read on the held-out cells only. Section **6** rolls every such arm up into one table.
""")

md(r"""
## 0. Pooled summary

### 0a. Does transfer exist, by relationship of sender and receiver?

One pooled row per (relationship category, animal): every ordered 160m arm of the category (`analysis.transfer_matrix.relationship_category`, both directions pooled — self, same-model ws siblings, cross-data-order / same init, cross-init / same order, same init / different corpus, same init / different tree, cross-init + cross-order, and the two cross-width 160m↔410m rows) is judged against ITS OWN matched-control maximum, under the within-cell control-relabelling null, with arms as strata (`stats.transfer_significance_tests.stratified_exceedance_permutation`, 20 000 draws, seed 0, floor 5e-5); `exceed.` = treatment runs above their arm's control maximum out of `runs`, `arms p<0.05` = arms whose own per-arm p clears 0.05. The design is **core30** for every animal (partial arms included, scored on the core30 runs they have), so rows are comparable across animals; the self row's controls exist on focus20 only (800 of 1200 runs). Dog is the only animal with the different-corpus (`dclm`) and cross-width categories; the cross-init / same-order category (`wc` siblings, plus the `pr` pair in dog) exists in every animal, and cat also carries the same-init / different-tree (`pr1234 ↔ orig`) pair; the rest of cat, lion and pig is the seed families (their cross-data-order category is the three `orig → ds_i` arms — no ds-sender students). A category with no arm in an animal is not listed; `no controls` = every arm of the category is untestable (not a null result). Highlighted = perm p < 0.05.
""")

code(r"""
import json, pandas as pd
from subliminal_learning.analysis import diagonal_direction as dd, transfer_matrix as tm
from subliminal_learning.analysis.setup import bootstrap

REPO, OUTPUT_ROOT = bootstrap()     # repo-anchored: every path below is absolute
ANIMALS = ["dog", "cat", "lion", "pig"]
TEACHERS = {a: json.loads((REPO / f"run_logs/{a}_top10_kband_configs.json").read_text())["param_tags"] for a in ANIMALS}
CORE30, FOCUS20 = dd.resolve_cells(REPO / "run_logs/core30_cells.json"), REPO / "run_logs/focus20_cells.json"
EXIST = tm.relationship_existence_by_animal(ANIMALS, TEACHERS, CORE30, xarch_json=REPO / "run_logs/xarch/xarch_significance_arms.json", n_perm=20000)
tm.style_relationship_existence_table(EXIST)
""")

md(r"""
### 0b. Which relationship transfers more? All category pairings (dog)

Each row asks whether category **A**'s SET of arms transfers more than **B**'s: the unit relabelled under the null is the ARM (the model pair), the stratum is (animal, cell), and the statistics — per-stratum mean contrast, per-stratum win probability (stratified Mann-Whitney), and the arm-level exceedance-fraction contrast — are read on the runs inside their stratum (`compare_families`, §3 legend). **Relative** value mode only (gap over the receiver's prior): receivers differ in prior across categories, so absolute gaps are not comparable. Pairings are every unordered pair of the six 160m categories that carry controls, oriented so A is the category with the larger pooled exceedance fraction in 0a; the cross-width rows are excluded (a different receiver width and per-arm teacher sets), and the ws siblings have no controls. Pairings with the self category are read on **focus20** (the self controls exist there only; the full-coverage rule would otherwise drop every self arm), all others on core30. The three pairings that exist in every animal get a `dog+cat+lion+pig stratified` row beneath the dog row. Arm counts overstate independence (arms share senders and receivers), and the new categories carry 2–5 arms each, so their p's are coarse: with 2 vs 2 arms the floor is 1/6, with 2 vs 5 it is 1/21, with 5 vs 12 it is 1/6188 (exact enumeration up to 1e6 labelings, else 20 000 Monte-Carlo draws, floor 5e-5). Highlighted = p < 0.05.

- **arms A** / **arms B** — the model pairs (`sender→receiver`, short labels) whose runs enter each side of the line, in the order they enter the relabelling; on a `dog+cat+lion+pig stratified` row an arm seen in several animals is listed once (it is one unit of the relabelling).
""")

code(r"""
RUNS30, ARMS30, _ = tm.load_relationship_runs("dog", TEACHERS["dog"], CORE30)
ARM_TAB = tm.relationship_arm_table(RUNS30, tm.arm_coverage(RUNS30, TEACHERS["dog"], CORE30, CORE30), cells30=CORE30, on_disk=ARMS30)
PAIRS = tm.category_pairings(EXIST[EXIST.animal == "dog"].set_index("category").exceed_frac.to_dict())
PAIRINGS = tm.relationship_pairing_table(PAIRS, tm.category_family_arms(ARM_TAB), TEACHERS, n_perm=20000,
                                         cells_for=lambda a, b: ("focus20", FOCUS20) if tm.RELATIONSHIP_CATEGORIES[0] in (a, b) else ("core30", CORE30))
tm.style_relationship_pairing_table(PAIRINGS)
""")

md(r"""
## 1. Cross-architecture transfer vs matched control

Does a teacher's numbers pool move a receiver of ANOTHER size or family above its matched control?

Each arm distils a teacher's numbers pool into a receiver of another size or family and compares it with the matched traitless control (the untuned sender base's pool distilled into the same receiver on the same cells), student seed 1, full-set teachers only (4-cell screening fan-outs excluded); `-seed1` arms (different pretraining data order) are never pooled with same-order arms. Removing dead cells changed no p, so dead-excluded variants are omitted. Two arms carry pilot-selected teacher sets and are read on held-out cells only (the pilot cells are excluded from their design; how each set was chosen is printed under the table).

- **arm** — sender → receiver (animal), shown once per block: each arm's block opens with its pooled stratified-permutation row (`all full-set teachers`, grey), then one row per teacher; `POOLED` rows stack arms, each against its own control maximum.
- **teacher** — `all full-set teachers` on the arm's pooled row (grey, first in the block), else the teacher tag.
- **test** — the statistic that produced the row's p: per-teacher / held-out sign test, or the stratified permutation (pooled over the arm's teachers, or across arms).
- **prior** — the receiver's test base p(animal), the unit of every ×prior column; blank on rows spanning receivers.
- **cells** — (treatment, control) cell pairs with values.
- **wins / n** — cells where treatment > control, over non-tied pairs.
- **sign p** — per-teacher sign test, wins ~ Binomial(n, ½), one-sided (held-out rows too); descriptive, exact per cell.
- **exceed.** — descriptive count of treatment cells above the control's maximum cell, on the same cells as **cells**; no p, because values at different cells are not exchangeable.
- **perm p** — PRIMARY p: stratified within-cell exceedance permutation (20,000), pooled rows only.
- **treat / ctrl median ×prior** — median p(animal) over cells, in units of the receiver's test base prior.
- **treat / ctrl max ×prior** — the largest cell's p(animal), same cells and units as the medians.
- **dead (treat/ctrl)** — runs whose Pile/Dolma probe ΔCE exceeds the receiver's dead cut.
- **status** — empty on a tested arm; `⚠ …` on an arm lacking data (in progress, not started, no matched controls, single-cell pilot), whose one row is shaded amber and carries the reason in **test** and matched/design in **cells**.

Highlighted = perm p < 0.05 (the primary test); amber = no p.
""")

code(r"""
from IPython.display import Markdown, display
from subliminal_learning.analysis.setup import bootstrap
from subliminal_learning.analysis.transfer_significance_io import significance_table, style_significance_table, registry_selection_notes

REPO, OUTPUT_ROOT = bootstrap()     # repo-anchored: every path below is absolute
XARCH = significance_table(REPO / "run_logs/xarch/xarch_significance_arms.json")
style_significance_table(XARCH)
""")

md(r"""
How the pilot-selected teacher sets of the table were chosen (from the registry's `selection_note`s):
""")

code(r"""
display(Markdown(registry_selection_notes(REPO / "run_logs/xarch/xarch_significance_arms.json")))
""")

md(r"""
The pooled rows of that table, in prose — generated from the table the cell above just
computed, so a re-execution can never leave a stale number behind:
""")

code(r"""
def _num(x, fmt=".3g"):
    return "—" if x is None or (isinstance(x, float) and x != x) else (f"{x:{fmt}}" if isinstance(x, float) else str(x))


display(Markdown("\n".join(
    f"- **{r['arm']}** ({r['test']}): {_num(r.get('wins / n'))} wins, {_num(r.get('exceed.'))} exceedances, "
    f"p = {_num(r.get('perm p'), '.1e')}, treat {_num(r.get('treat median ×prior'), '.2f')}× vs "
    f"ctrl {_num(r.get('ctrl median ×prior'), '.2f')}× prior"
    for r in XARCH[XARCH["pooled"]].to_dict("records"))))
""")

md(r"""
## 2. Pythia seed families: does transfer exist inside each family?

One row per (family, animal, arm), plus one pooled row per (family, animal) — the family's single existence claim. Arms are every ordered off-diagonal base pair of a cross family (12 each, `cross-seed` = different pretraining init, `cross-data-order` = different pretraining data order), and the diagonal `b→b` for `same-model` (7 bases). The cross families are read on the 56-cell core56 design, `same-model` on focus20: its base controls exist at those 20 cells only.

- **family** — the family and its design; **animal** — the animal the teachers were finetuned on (and scored) — dog, cat, lion or pig — never pooled across animals.
- **arm** — sender → receiver base pair; `all arms` = that animal's pooled row (grey), the arm-stratified permutation with every arm judged against ITS OWN control maximum.
- **cells** — design cells carrying a control (summed over arms on the pooled row).
- **treatment runs** — student runs entering the test (teachers × cells).
- **exceed.** — treatment runs strictly above the control maximum of their own arm.
- **perm p** — PRIMARY p: cell-stratified exceedance permutation (20,000 draws; arms are an extra stratum on the pooled row), floor 1/(n_perm+1).
- **note** — why an arm carries no numbers (no control cells on disk, or no treatment run); such arms are shaded amber.

Highlighted = perm p < 0.05.
""")

code(r"""
import json, pandas as pd
from subliminal_learning.analysis.transfer_significance_io import family_existence_table, style_family_existence_table

ANIMALS = ["dog", "cat", "lion", "pig"]
TEACHERS = {a: json.loads((REPO / f"run_logs/{a}_top10_kband_configs.json").read_text())["param_tags"] for a in ANIMALS}
CORE56, FOCUS20 = REPO / "run_logs/core56_cells.json", REPO / "run_logs/focus20_cells.json"
DESIGNS = [("cross-seed", CORE56, "core56"), ("cross-data-order", CORE56, "core56"), ("same-model", FOCUS20, "focus20")]
FAMILY = pd.concat([family_existence_table(f, ANIMALS, TEACHERS, cells=c, family_label=f"{f} ({d})", n_perm=20000) for f, c, d in DESIGNS], ignore_index=True)
style_family_existence_table(FAMILY)
""")

md(r"""
## 3. Is one family stronger than another?

Each comparison asks whether family **A**'s SET of arms transfers more than family **B**'s on ONE pinned design, every student read against its own cell control. The unit relabelled under the null is the ARM (the model pair); the statistic is read on the runs INSIDE their stratum, so cell difficulty cannot masquerade as a family effect. Only blocks covering the whole design enter. Comparison (c) repeats (a) on focus20 so that (b), which must use focus20, has a same-design reference.

- **comparison** — `A vs B`: a positive contrast means A is the stronger family; **design** — the student cell set.
- **animal** — one animal's blocks, or the `dog+cat+lion+pig stratified` fit (grey) whose strata are (animal, cell) pairs.
- **value mode** — `absolute` = treatment − cell control; `relative` = that gap over the receiver's base prior.
- **arms A vs B** — units entering the relabelling (a model pair seen in both animals counts ONCE).
- **mean contrast** / **mean p** — per-stratum mean(A) − mean(B), averaged over strata, under the arm-relabel null.
- **win prob.** / **win p** — per-stratum share of (A run, B run) pairs won by A, ties ½ (Mann-Whitney), averaged over strata.
- **exceed. frac contrast** / **exceed. p** — arm-level: each arm's share of runs above its own cell control, families compared as two sets of arm values; scale-free, so identical in both value modes.
- **floor** / **labelings** — smallest reportable p and the number of arm labelings (exact enumeration when ≤ 1e6, else Monte-Carlo draws).
- **missing blocks** — (animal, arm) blocks kept out by the coverage rule.

Highlighted = p < 0.05.
""")

code(r"""
from subliminal_learning.analysis.transfer_significance_io import compare_families, family_comparison_table, style_family_comparison_table

SPECS = [("cross-data-order vs cross-seed", "core56", "cross-data-order", "cross-seed", CORE56), ("same-model vs cross-data-order", "focus20", "same-model", "cross-data-order", FOCUS20), ("cross-data-order vs cross-seed", "focus20", "cross-data-order", "cross-seed", FOCUS20)]
COMPARISON = family_comparison_table({(name, design): compare_families(a, b, ANIMALS, TEACHERS, cells=cells, n_perm=20000) for name, design, a, b, cells in SPECS})
style_family_comparison_table(COMPARISON)
""")

md(r"""
How to read the `dog+cat+lion+pig stratified` line: it is not "four times the evidence". The unit is still the model pair, so an arm run in several animals is ONE unit and the arm counts stay those of a single animal; adding cat, lion and pig only adds strata (cells) that arm is measured on. Cat, lion and pig have no ds-SENDER students, so on the cross-data-order side the stratified fit leans on dog for every `ds_i → ·` arm and only the three `orig → ds_i` arms carry a cat, lion or pig block.
""")

md(r"""
## 4. 160m re-pretraining direction matrix (dog, cat, lion, pig, chicken; core30)

Does a teacher's numbers pool move a RE-PRETRAINED sibling of pythia-160m above its matched control, and in which direction? Senders and receivers are pythia-160m (`orig`) and its re-pretrainings: `pr1234` = gpaulo's re-pretraining of pythia-160m's own seed and Pile data order on the modern gpt-neox tree; `pr3` = seed-3 init on that same Pile order; `dclm1234` / `dclm1` = the DCLM corpus at seeds 1234 / 1; `wc1` / `wc2` = the v1.0-tree corrected weight-seeds. Every arm distils the animal's ten k-band teachers (tseed 1) on the 30 core30 cells and reads student seed 1 against the matched base control; arm names carry the animal and pools never mix animals. `HOT10` repeats `pr1234 → pr3` on the 10 core56 \ core30 cells with K/K* > 1 (`run_logs/pr1234_to_pr3_hot10_20260918/hot10_manifest.json`). Column legend as in table 1; the receiver's prior is its student trees' `base_control.json` test prior, else its `analysis/base_eval_*_test.json` prior. No re-pretrained receiver has an anchored pp1 dead cut, so the dead column reads `?` throughout.

`pr1234 → pr3 UNION core30+hot10` is the 40-cell union of the core30 and HOT10 rows as ONE arm (`run_logs/repretrain/pr3_core30_hot10_cells.json`), so its exceedances are judged against a single control maximum over all 40 cells -- the design a pool of the two rows would not give, since pooled arms each keep their own control maximum. Pools: `POOLED: into pr1234 (orig, pr3)` stacks the two arms into pr1234 and `POOLED: out of pr1234 (orig, pr3)` the two arms out of it, each arm against its own control maximum (same Pile data-order class). No DCLM pool: `pr1234 → dclm1234` and `dclm1234 → pr1234` cross corpora (`data_order` different) while `dclm1 → dclm1234` shares the DCLM order, and the registry never pools arms of different data-order classes.

**Missing** (amber rows, no p): of the 42 ordered pairs among the seven dog lineages only the twelve with a row below have any students; `dclm1234 → dclm1`, `orig → wc1` and `wc1 → orig` are complete or nearly so but have no matched control tree (the campaigns had no control stage); everything else (`orig ↔ pr3`, `orig ↔ dclm*`, `pr3 ↔ dclm*`, `wc ↔ pr/dclm`, `dclm1234 → dclm2` — the DCLM seed-2 pretraining stopped at 32k steps — …) was never run; a stray one-teacher `wc1 → dclm1` pilot (16 cells) is ignored. Cat's `orig → pr1234`, pig's `wc2 → wc1` and both chicken arms are still draining or not started, as their status column says.

Highlighted = perm p < 0.05 (the primary test); amber = no p.
""")

code(r"""
REPRETRAIN = significance_table(REPO / "run_logs/repretrain/repretrain_significance_arms.json")
style_significance_table(REPRETRAIN)
""")

md(r"""
The pooled rows of that table, in prose — generated from the table the cell above just computed:
""")

code(r"""
display(Markdown("\n".join(
    f"- **{r['arm']}** ({r['test']}): {_num(r.get('wins / n'))} wins, {_num(r.get('exceed.'))} exceedances, "
    f"p = {_num(r.get('perm p'), '.1e')}, treat {_num(r.get('treat median ×prior'), '.2f')}× vs "
    f"ctrl {_num(r.get('ctrl median ×prior'), '.2f')}× prior"
    for r in REPRETRAIN[REPRETRAIN["pooled"]].to_dict("records"))))
""")

md(r"""
## 5. OLMo-1B (DataDecide): cross-seed and cross-recipe arms

Does the cross-seed result hold on a second family, and does transfer cross pretraining CORPORA? Senders and receivers are the DataDecide OLMo-1B pretrainings: `orig` = `dd-1b` (dolma1_7, seed-default; `dolma` in the cross-recipe arm names), `seed2` = `dd-1b-seed2` (large-aux-2), `seed3` = `dd-1b-seed3` (large-aux-3) -- the same 1B architecture and Dolma 1.7 data, different initialisation seeds -- and the recipes `dclm` = `dd-1b-dclm` and `c4` = `dd-1b-c4` (seed-default init, DCLM-baseline resp. C4 corpus). Trait elephant (dog for two single-cell pilots), teachers finetuned at tseed 0, students at finetune seed 1, read against the matched base control (the untuned sender's pool distilled into the receiver). `ctrl20` (`run_logs/dd1b/ctrl20_cells.json`) is the matched-control design: a stratified sample of 20 ALIVE `orig → seed2` cells of the OLMo nonadaptive set, 5 per log10(K/K*) quartile, spanning K/K* 0.04–1.13 on the cross-seed fit; `core30` / `core35` (`run_logs/dd1b/olmo_core30_cells.json` / `olmo_core35_cells.json`) are the OLMo transfer band; the `top25 union` (`run_logs/dd1b/top25_union_cells.json`) the 46-cell seed-3 design. The `orig → orig` row is the same-model reference on ctrl20. The two `dclm → dolma` sets were chosen on the single pilot cell `ep10_lr1e-4_r64a256`, which is excluded from their design (34 held-out core35 cells; the selection notes are printed under the table). Column legend as in table 1; the receiver's prior is the test prior its student trees' `base_control.json` share. No OLMo receiver has an anchored dead cut, so the dead column reads `?` throughout. Cross-recipe arms carry their own `data_order` class and are never pooled with the cross-seed arms.

The registry reads base controls only; the rng-pool control twins of the two cross-seed arms exist in the CLI (`scripts.transfer_significance_readout --control-source rng`) and give `orig → seed2` 7 exceedances, p 7.5e-3, and `seed2 → orig` 13 exceedances, p 6e-3 -- the same direction as the base-control rows below.

**Missing** (amber rows, no p):

- `orig → seed3`, `seed2 → seed3` — the 46-cell top-25-union arms are trained, but NO controls exist on any seed-3 receiver (the second seed2 teacher, `ep10_lr5e-4_r64a64`, covers 18 of the 46 cells and is left out).
- the `ctrl57` control set — never queued, so no arm reads it.
- the single-cell pilots (`dolma → c4` elephant and dog, `dolma → dclm` dog, `dclm → c4`, `c4 → dolma`, `c4 → dclm`) — one cell is no design; the two `c4 → *` pilots are planned and not submitted (no trees).
- `orig → seed2` is a single-teacher, 20-cell arm: the permutation relabels which of the (teacher, control) pair is the control in each cell.

Highlighted = perm p < 0.05 (the primary test); amber = no p.
""")

code(r"""
OLMO = significance_table(REPO / "run_logs/repretrain/olmo_cross_seed_significance_arms.json")
style_significance_table(OLMO)
""")

md(r"""
How the pilot-selected teacher sets of the table were chosen (from the registry's `selection_note`s):
""")

code(r"""
display(Markdown(registry_selection_notes(REPO / "run_logs/repretrain/olmo_cross_seed_significance_arms.json")))
""")

md(r"""
The pooled rows of that table, in prose — generated from the table the cell above just computed:
""")

code(r"""
display(Markdown("\n".join(
    f"- **{r['arm']}** ({r['test']}): {_num(r.get('wins / n'))} wins, {_num(r.get('exceed.'))} exceedances, "
    f"p = {_num(r.get('perm p'), '.1e')}, treat {_num(r.get('treat median ×prior'), '.2f')}× vs "
    f"ctrl {_num(r.get('ctrl median ×prior'), '.2f')}× prior"
    for r in OLMO[OLMO["pooled"]].to_dict("records"))))
""")

md(r"""
## 6. Arms lacking data

Every arm of sections 1–5 that carries no p, in one table: the amber rows of tables 1, 4 and 5 (`status` = what is lacking, `coverage` = matched/design cells), the skipped arms of table 2 grouped by (family, animal, reason) (`coverage` = skipped/all arms) and, per comparison of table 3, the (animal, arm) blocks its coverage rule kept out. An arm leaves this table only by finishing (every teacher on every design cell, a control on every design cell) or by gaining a control tree; single-cell pilots and the seed-3 receivers stay until a design resp. controls exist.
""")

code(r"""
from subliminal_learning.analysis.transfer_significance_io import flagged_arms, flagged_family_arms, flagged_missing_blocks, lacking_data_table

LACKING = lacking_data_table([*flagged_arms(XARCH, "1"), *flagged_family_arms(FAMILY, "2"), *flagged_missing_blocks(COMPARISON, "3"), *flagged_arms(REPRETRAIN, "4"), *flagged_arms(OLMO, "5")])
LACKING.style.hide(axis="index").set_properties(**{"background-color": "rgba(255, 180, 0, 0.25)", "text-align": "left"})
""")

nb = {
    "cells": [
        {"cell_type": t, "metadata": {}, "source": s.splitlines(keepends=True),
         **({"execution_count": None, "outputs": []} if t == "code" else {})}
        for t, s in CELLS
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python (subliminal .venv)", "language": "python",
                       "name": "subliminal"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
if __name__ == "__main__":     # importing this module must never clobber the notebook
    OUT.write_text(json.dumps(nb, indent=1) + "\n")
    print("wrote", OUT, len(CELLS), "cells")
