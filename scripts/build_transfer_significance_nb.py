"""Build notebooks/transfer_significance.ipynb: does transfer exist, and which family is stronger.

A headline table (existence by family, finished sets only, `analysis.transfer_headline`), a pooled summary (existence by relationship category, category pairings), then five tables: the cross-architecture arms of run_logs/campaigns/xarch/xarch_significance_arms.json, the
Pythia seed families' existence tests, the family comparisons (cross-data-order vs
cross-seed, same-model vs cross-data-order), per animal and animal-stratified, the 160m
re-pretraining direction matrix of run_logs/campaigns/repretrain/repretrain_significance_arms.json (dog, cat,
lion, pig, chicken), the OLMo-1B (DataDecide) cross-seed and cross-recipe arms of
run_logs/campaigns/repretrain/olmo_cross_seed_significance_arms.json, and a closing roll-up of every arm
lacking data. Only finished sets get a p; everything else is an amber coverage row.
Loading, tests and styling all live in `analysis.transfer_significance_io`; loaders and
permutation tests are memoized through `stats.memo`, so a re-execution after a presentation
change only recomputes arms whose files changed (the last cell prints the hit/miss count).
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

Section **0.0** is the one-table headline (existence by relationship family, finished sets only); section **0** is the pooled summary (existence by relationship category across animals, and every category-vs-category pairing). Then five tables. **1** cross-architecture arms (another model size or family), **2** the Pythia seed families' existence tests, **3** the family comparisons, **4** the 160m re-pretraining direction matrix (dog, cat, lion, pig, chicken), **5** the OLMo-1B (DataDecide) cross-seed and cross-recipe (dolma / dclm / c4) arms. Every test reads a student against its OWN matched traitless control (the untuned sender base's numbers pool distilled into the same receiver on the same cells), student finetune seed 1, the 10 k-band teachers of each animal where the arm is a k-band arm (elephant, or a pilot-selected teacher set, in the OLMo arms). In tables 1, 2, 4 and 5 animals are never pooled into one stratum; table 3 adds one `dog+cat+lion+pig stratified` line per comparison, whose strata are (animal, cell) pairs.

**Only a finished set gets a p**: every listed teacher with a seed-1 endpoint on every design cell and a matched control on every design cell. An arm short of that, an arm with no controls, or a single-cell pilot is ONE amber row with a `⚠` status naming what is lacking (no p, `cells` = matched/design), never a test on the cells that happened to finish; teacher sets chosen on a pilot cell are read on every design cell, that pilot cell included — a pilot cell outside the design is simply not in the design. Section **6** rolls every such arm up into one table.
""")

md(r"""
## 0.0 Headline: existence by family

- **Finished sets only**: every listed teacher with a seed-1 test endpoint on every design cell of the arm AND a matched control on every cell; an arm short of that is dropped, counted in `arms used` (finished/listed) and named in `note`.
- **Yardstick per arm** = its OWN matched control: `exceed.` = treatment runs above the arm's control maximum; `perm p` = stratified exceedance permutation (20 000 draws, seed 0, cells as strata, arms as an extra stratum, each against its own control maximum).
- **dominant** = the family's main animal (dog for Pythia-receiver families, elephant for the OLMo cross-seed / cross-recipe families); **stratified** (grey; only where ≥ 2 animals carry a finished arm) = every finished (animal, arm) block as its own stratum — the existence pooling, NOT §3's family-comparison unit (there an arm seen in several animals is ONE unit).
- **prior** = the receiver's test prior (`student_tree_prior`); `min / median / max` when the row spans receivers.
- **×prior** block = mean / median / p75 / p95 / max of the per-run `p(animal) / receiver test prior`, computed PER ARM, shown as `min / median / max` across the row's arms.
- **×control** block = the same five on the per-run `p_student / p_control` at the same cell; a control ≈ 0 blows the ratio up — read it next to the ×prior block.
- The OLMo dolma / dclm recipes share the seed-default init byte-for-byte, hence `same init, different corpus`; the §2/§3 `cross-seed` family = the full seeds = different init AND different data order.
- Highlighted = perm p < 0.05; the strip figure under the table draws each family's whole relabelling null (grey, in % of the row's runs) with its 95th percentile dashed and the observed exceedance count as the coloured line, the same numbers as the `exceed.` / `null 95th` columns.
""")

code(r"""
import json
from subliminal_learning.analysis import transfer_headline as th
from subliminal_learning.analysis.setup import bootstrap

REPO, OUTPUT_ROOT = bootstrap()     # repo-anchored: every path below is absolute
ANIMALS = ["dog", "cat", "lion", "pig"]
TEACHERS = {a: json.loads((REPO / f"run_logs/registry/{a}_top10_kband_configs.json").read_text())["param_tags"] for a in ANIMALS}
FAMILIES = th.headline_families(REPO, TEACHERS)
HEADLINE = th.headline_table(FAMILIES, n_perm=20000)
th.style_headline_table(HEADLINE)
""")

code(r"""
FIG = th.null_strip_figure(FAMILIES)
FIG.savefig(REPO / "notebooks/figures/transfer_null_strip.png", dpi=150, bbox_inches="tight")
FIG
""")

md(r"""
## 0. Pooled summary

### 0a. Does transfer exist, by relationship of sender and receiver?

One pooled row per (relationship category, animal): every ordered 160m arm of the category (`analysis.transfer_matrix.relationship_category`, both directions pooled — self, same-model ws siblings, cross-data-order / same init, cross-init / same order split by code tree (v1.0 tree = the `wc1 ↔ wc2` weightc reruns on the v1.0 gpt-neox tree; modern tree = `pr1234 ↔ pr3` and `dclm1 ↔ dclm1234`), same init / different corpus, same init / different tree, cross-init + cross-order, and the two cross-width 160m↔410m rows) is judged against ITS OWN matched-control maximum, under the within-cell control-relabelling null, with arms as strata (`stats.transfer_significance_tests.stratified_exceedance_permutation`, 20 000 draws, seed 0, floor 5e-5); `exceed.` = treatment runs above their arm's control maximum out of `runs`, `arms p<0.05` = arms whose own per-arm p clears 0.05. The design is **core30** for every animal (partial arms included, scored on the core30 runs they have), so rows are comparable across animals; of the self row's seven diagonals only `orig` and the three full seeds carry core30 controls (`run_logs/campaigns/sl_samemodel_controls_core30_20260921`), the three `ds` diagonals none or part of focus20 only. Dog is the only animal with the different-corpus (`dclm`), modern-tree cross-init (`pr` and `dclm` pairs) and cross-width categories; the v1.0-tree cross-init / same-order category (`wc` siblings) exists in every animal, and cat also carries the same-init / different-tree (`pr1234 ↔ orig`) pair; the rest of cat, lion and pig is the seed families (their cross-data-order category is the three `orig → ds_i` arms — no ds-sender students). A category with no arm in an animal is not listed; `no controls` = every arm of the category is untestable (not a null result). Highlighted = perm p < 0.05.
""")

code(r"""
import json, pandas as pd
from subliminal_learning.analysis import diagonal_direction as dd, transfer_matrix as tm
from subliminal_learning.analysis.setup import bootstrap

REPO, OUTPUT_ROOT = bootstrap()     # repo-anchored: every path below is absolute
ANIMALS = ["dog", "cat", "lion", "pig"]
TEACHERS = {a: json.loads((REPO / f"run_logs/registry/{a}_top10_kband_configs.json").read_text())["param_tags"] for a in ANIMALS}
CORE30 = dd.resolve_cells(REPO / "run_logs/registry/core30_cells.json")
EXIST = tm.relationship_existence_by_animal(ANIMALS, TEACHERS, CORE30, xarch_json=REPO / "run_logs/campaigns/xarch/xarch_significance_arms.json", n_perm=20000)
tm.style_relationship_existence_table(EXIST)
""")

md(r"""
### 0b. Which relationship transfers more? All category pairings (dog)

Each row asks whether category **A**'s SET of arms transfers more than **B**'s: the unit relabelled under the null is the ARM (the model pair), the stratum is (animal, cell), and the statistics — per-stratum mean contrast, per-stratum win probability (stratified Mann-Whitney), and the arm-level exceedance-fraction contrast — are read on the runs inside their stratum (`compare_families`, §3 legend). **Relative** value mode only (gap over the receiver's prior): receivers differ in prior across categories, so absolute gaps are not comparable. Pairings are every unordered pair of the seven 160m categories that carry controls (the cross-init / same-order category is split by code tree: v1.0 `wc1 ↔ wc2` vs modern `pr1234 ↔ pr3`, `dclm1 ↔ dclm1234`), oriented FOR DISPLAY so A is the category with the larger pooled exceedance fraction in 0a (an orientation picked from the data, which is exactly why the p's below are two-sided: those do not move when the two sides are swapped); the cross-width rows are excluded (a different receiver width and per-arm teacher sets), and the ws siblings have no controls. Every pairing, the ones with the self category included, is read on **core30**: the `orig` and full-seed diagonals now carry core30 controls, and the full-coverage rule drops the `ds` diagonals, which carry none (listed under the table). The three pairings that exist in every animal get a `dog+cat+lion+pig stratified` row beneath the dog row. Arm counts overstate independence (arms share senders and receivers), and the new categories carry 2–5 arms each, so their p's are coarse: with 2 vs 2 arms the floor is 1/6, with 2 vs 5 it is 1/21, with 5 vs 12 it is 1/6188 (exact enumeration up to 1e6 labelings, else 20 000 Monte-Carlo draws, floor 5e-5). Every p is TWO-SIDED by tail doubling — `p = min(1, 2 · min(P(null ≥ observed), P(null ≤ observed)))` — because which category is the stronger one was not fixed before the data; the DIRECTION is read off the sign of the contrast beside it (mean contrast vs 0, win prob. vs ½, exceedance-fraction contrast vs 0). Green = p < 0.05. **one-sided A>B** is the upper tail alone (`P(null ≥ observed)`), shown for reference and never highlighted.

- **arms A** / **arms B** — the model pairs (`sender→receiver`, short labels) whose runs enter each side of the line, in the order they enter the relabelling; on a `dog+cat+lion+pig stratified` row an arm seen in several animals is listed once (it is one unit of the relabelling). A side whose arms are every ordered pair of its models collapses to `all k directions among {…}`; a side whose arms share one sender (resp. receiver) to `x → {…}` (resp. `{…} → x`); anything partial stays an explicit list.
- **one-sided A>B (mean / win / exc.)** — the three one-sided upper-tail p's for A > B (mean contrast / win probability / exceedance), from the same null as the two-sided columns; `—` where a test is missing.
- **animals sig. A>B / B>A (mean / win / exc.)** — on a stratified row, `k/n` per statistic: k animals whose own relative-mode test is TWO-SIDED significant at p < 0.05 and whose contrast points that way by its sign, out of n animals carrying a usable comparison for that test; `—` on single-animal rows.
- **missing blocks** — the (animal, arm) blocks kept out by the coverage rule are no longer a column; the rows with any are listed under the table.
""")

code(r"""
from IPython.display import Markdown, display

RUNS30, ARMS30, _ = tm.load_relationship_runs("dog", TEACHERS["dog"], CORE30)
ARM_TAB = tm.relationship_arm_table(RUNS30, tm.arm_coverage(RUNS30, TEACHERS["dog"], CORE30, CORE30), cells30=CORE30, on_disk=ARMS30)
PAIRS = tm.category_pairings(EXIST[EXIST.animal == "dog"].set_index("category").exceed_frac.to_dict())
PAIRINGS, PAIRING_NOTES = tm.relationship_pairing_table(PAIRS, tm.category_family_arms(ARM_TAB), TEACHERS, n_perm=20000,
                                                        cells_for=lambda a, b: ("core30", CORE30))
display(tm.style_relationship_pairing_table(PAIRINGS))
if PAIRING_NOTES:
    display(Markdown("**Blocks kept out by the coverage rule**\n" + "\n".join(
        f"- **{c} · {d} · {a}** — {text}" for (c, d, a), text in PAIRING_NOTES.items())))
""")

md(r"""
## 1. Cross-architecture transfer vs matched control

Does a teacher's numbers pool move a receiver of ANOTHER size or family above its matched control?

Each arm distils a teacher's numbers pool into a receiver of another size or family and compares it with the matched traitless control (the untuned sender base's pool distilled into the same receiver on the same cells), student seed 1, full-set teachers only (on-disk teachers an arm lists nowhere -- 4-cell screening fan-outs, other arms' teachers on the same tree -- are strays, auto-excluded and named in the status column); `-seed1` arms (different pretraining data order) are never pooled with same-order arms. Removing dead cells changed no p, so dead-excluded variants are omitted. There is ONE arm per (sender, receiver) pair: where part of a pair's teachers was pilot-selected, the pilot cells are scored like every other design cell — no pilot cell is held out anywhere (how each set was chosen is printed under the table). The `160m → 410m` and `160m → 410m-seed1` arms are the 09-20 promotions of the single-cell pilot `ep20_lr9e-5_r8a32` (a cell outside core30, so it is not a design cell at all): the top-10 senders per receiver by pilot ratio on core30 (`run_logs/campaigns/xarch_160m_410m_core30_tf32_20260920`, drained 09-21), where the fp32-complete senders and every control are fp32 and the promoted remainder tf32 (precision is a nuisance variable on those rows). The two reverse arms, `410m → 160m` and `410m-seed1 → 160m`, carry ten senders each (`run_logs/registry/p410m_top10_senders_pythia-410m{,-seed1}.json`): the two pre-existing core30 senders of the 09-16 arm plus eight chosen by pilot ratio (student p_target / its matched control) at `ep20_lr9e-5_r8a32` out of the ΔCE-stratified 410m teacher lattice (`run_logs/campaigns/p410m_teacher_strat_20260921` → `p410m_strat_to_160m_20260921` → `p410m_strat_core30_20260921`, drained 09-22); the pilot cell is outside core30, so both arms are read on all 30 cells and every remaining lattice teacher is a stray. These rows are fp32 throughout. The one nuisance is the pool size of a single seed1 sender: `ep2_lr5e-4_r32a64`'s numbers pool was generated at 125k, so the registry's `teacher_dataset_tags` reads that ONE sender under `ex10-20_prefix_n125k` while its nine siblings and the control tree stay at `n100k` (the seed1 manifest's eleventh candidate, `ep5_lr1e-3_r8a4`, was undistillable — 9 887 usable sequences — and never trained). `410m-seed1 → 160m` is now ONE arm for the pair; the separate single-teacher `r8a16` arm (same core30 design) is retired.

- **arm** — sender → receiver (animal), shown once per block: each arm's block opens with its pooled stratified-permutation row (`all full-set teachers`, grey), then one row per teacher; `POOLED` rows stack arms, each against its own control maximum.
- **teacher** — `all full-set teachers` on the arm's pooled row (grey, first in the block), else the teacher tag.
- **test** — the statistic that produced the row's p: the per-teacher sign test, or the stratified permutation (pooled over the arm's teachers, or across arms).
- **prior** — the receiver's test base p(animal), the unit of every ×prior column; blank on rows spanning receivers.
- **cells** — (treatment, control) cell pairs with values.
- **wins / n** — cells where treatment > control, over non-tied pairs.
- **sign p** — per-teacher sign test, wins ~ Binomial(n, ½), one-sided; descriptive, exact per cell.
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
XARCH = significance_table(REPO / "run_logs/campaigns/xarch/xarch_significance_arms.json")
style_significance_table(XARCH)
""")

md(r"""
How the pilot-selected teacher sets of the table were chosen (from the registry's `selection_note`s):
""")

code(r"""
display(Markdown(registry_selection_notes(REPO / "run_logs/campaigns/xarch/xarch_significance_arms.json")))
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

One row per (family, animal, arm), plus one pooled row per (family, animal) — the family's single existence claim. Arms are every ordered off-diagonal base pair of a cross family (12 each, `cross-seed` = the full seeds, different pretraining init AND different data order; `cross-data-order` = same init, different pretraining data order), and the diagonal `b→b` for `same-model` (7 bases). The cross families are read on the 56-cell core56 design, `same-model` on core30: the `orig` and full-seed diagonals carry core30 base controls, the three `ds` diagonals do not and are skipped with a note.

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
TEACHERS = {a: json.loads((REPO / f"run_logs/registry/{a}_top10_kband_configs.json").read_text())["param_tags"] for a in ANIMALS}
CORE56 = REPO / "run_logs/registry/core56_cells.json"
DESIGNS = [("cross-seed", CORE56, "core56"), ("cross-data-order", CORE56, "core56"), ("same-model", CORE30, "core30")]
FAMILY = pd.concat([family_existence_table(f, ANIMALS, TEACHERS, cells=c, family_label=f"{f} ({d})", n_perm=20000) for f, c, d in DESIGNS], ignore_index=True)
style_family_existence_table(FAMILY)
""")

md(r"""
## 3. Is one family stronger than another?

Each comparison asks whether family **A**'s SET of arms transfers more than family **B**'s on ONE pinned design, every student read against its own cell control. The unit relabelled under the null is the ARM (the model pair); the statistic is read on the runs INSIDE their stratum, so cell difficulty cannot masquerade as a family effect. Only blocks covering the whole design enter. Comparison (c) repeats (a) on core30 so that (b), which must use core30 (the same-model design), has a same-design reference.

- **comparison** — `A vs B`: a positive contrast means A is the stronger family; **design** — the student cell set.
- **animal** — one animal's blocks, or the `dog+cat+lion+pig stratified` fit (grey) whose strata are (animal, cell) pairs.
- **value mode** — `absolute` = treatment − cell control; `relative` = that gap over the receiver's base prior (PRIMARY); `receptivity` and `portability` are the two pre-specified secondary retention views below.
- **receptivity** — the student's gap over its control, divided by the RECEIVER's own same-model yardstick at the same cell: D_r(c) = the mean over the family's ten teacher tags of (p(r→r, t', c) − ctrl(r→r, c)), the receiver's diagonal arm read against its own same-base control (absolute p units top and bottom; the prior cancels). The denominator is always the diagonal ARM MEAN over the ten tags, never the same tag's single run.
- **portability** — the same gap divided by the SENDER's diagonal yardstick D_s(c).
- A (receiver-or-sender, cell) stratum enters a retention mode only when the cross run, its control, ALL ten diagonal teacher runs and the diagonal control are present, and D ≥ δ with δ = 0.5 × the prior of the base whose diagonal it is; strata below the floor are counted in **missing blocks** (`receptivity: n (r, cell) strata below floor`). y is never clipped (negative and > 1 values stay); the mean-contrast, win-probability and exceedance statistics run on y exactly as on the relative gap. Pre-specified secondary view: the relative mode stays primary; report the four side by side, never a retention row alone. A design without same-model controls (core56) renders the two rows with a note and no p.
- **arms A vs B** — units entering the relabelling (a model pair seen in both animals counts ONCE).
- **mean contrast** / **mean p** — per-stratum mean(A) − mean(B), averaged over strata, under the arm-relabel null.
- **win prob.** / **win p** — per-stratum share of (A run, B run) pairs won by A, ties ½ (Mann-Whitney), averaged over strata.
- **exceed. frac contrast** / **exceed. p** — arm-level: each arm's share of runs above its own cell control, families compared as two sets of arm values; scale-free, so identical in both value modes.
- **floor** / **labelings** — smallest reportable p and the number of arm labelings (exact enumeration when ≤ 1e6, else Monte-Carlo draws).
- **missing blocks** — (animal, arm) blocks kept out by the coverage rule.

Every p in the three p columns is TWO-SIDED (tail doubling: `min(1, 2 · min(P(null ≥ observed), P(null ≤ observed))`), because which family is the stronger one was not fixed before the data; which family a significant row favours is read off the SIGN of its contrast (mean contrast vs 0, win prob. vs ½, exceedance-fraction contrast vs 0). Doubling, not `|T − centre|`: the arm-relabel null is asymmetric whenever the two families carry unequal arm counts, and nothing here assumes the win-probability null is centred on ½. Highlighted = p < 0.05.
""")

code(r"""
from subliminal_learning.analysis.transfer_significance_io import compare_families, family_comparison_table, style_family_comparison_table

SPECS = [("cross-data-order vs cross-seed", "core56", "cross-data-order", "cross-seed", CORE56), ("same-model vs cross-data-order", "core30", "same-model", "cross-data-order", CORE30), ("cross-data-order vs cross-seed", "core30", "cross-data-order", "cross-seed", CORE30)]
RESULTS = {(name, design): compare_families(a, b, ANIMALS, TEACHERS, cells=cells, n_perm=20000) for name, design, a, b, cells in SPECS}
COMPARISON = family_comparison_table(RESULTS)
style_family_comparison_table(COMPARISON)
""")

md(r"""
The diagonal yardsticks behind the two retention modes, per (animal, base): `cells` = core30 cells with the base's complete same-model arm (ten teachers + control), `cells passing floor` = those with D ≥ 0.5 × prior, `median D/prior` = the yardstick in units of the base's prior, `median SE/D` = the teacher SD/√10 relative to D (the ratio noise every retention value inherits). No base has a same-model arm on the 26 core56 cells outside core30.
""")

code(r"""
from subliminal_learning.analysis.transfer_significance_io import diagonals_table, style_diagonals_table

DIAGONALS = diagonals_table(RESULTS)
style_diagonals_table(DIAGONALS)
""")

md(r"""
How to read the `dog+cat+lion+pig stratified` line: it is not "four times the evidence". The unit is still the model pair, so an arm run in several animals is ONE unit and the arm counts stay those of a single animal; adding cat, lion and pig only adds strata (cells) that arm is measured on. Cat, lion and pig have no ds-SENDER students, so on the cross-data-order side the stratified fit leans on dog for every `ds_i → ·` arm and only the three `orig → ds_i` arms carry a cat, lion or pig block.
""")

md(r"""
## 4. 160m re-pretraining direction matrix (dog, cat, lion, pig, chicken; core30)

Does a teacher's numbers pool move a RE-PRETRAINED sibling of pythia-160m above its matched control, and in which direction? Senders and receivers are pythia-160m (`orig`) and its re-pretrainings: `pr1234` = gpaulo's re-pretraining of pythia-160m's own seed and Pile data order on the modern gpt-neox tree; `pr3` = seed-3 init on that same Pile order; `dclm1234` / `dclm1` = the DCLM corpus at seeds 1234 / 1; `wc1` / `wc2` = the v1.0-tree corrected weight-seeds. Every arm distils the animal's ten k-band teachers (tseed 1) on the 30 core30 cells and reads student seed 1 against the matched base control; arm names carry the animal and pools never mix animals. `HOT10` repeats `pr1234 → pr3` on the 10 core56 \ core30 cells with K/K* > 1 (`run_logs/campaigns/pr1234_to_pr3_hot10_20260918/hot10_manifest.json`). Column legend as in table 1; the receiver's prior is its canonical `analysis/base_eval_*_animal_preference_test.json` test prior. No re-pretrained receiver has an anchored pp1 dead cut, so the dead column reads `?` throughout.

`pr1234 → pr3 UNION core30+hot10` is the 40-cell union of the core30 and HOT10 rows as ONE arm (`run_logs/campaigns/repretrain/pr3_core30_hot10_cells.json`), so its exceedances are judged against a single control maximum over all 40 cells -- the design a pool of the two rows would not give, since pooled arms each keep their own control maximum. Pools: `POOLED: into pr1234 (orig, pr3)` stacks the two arms into pr1234 and `POOLED: out of pr1234 (orig, pr3)` the two arms out of it, each arm against its own control maximum (same Pile data-order class). No DCLM pool: `pr1234 → dclm1234` and `dclm1234 → pr1234` cross corpora (`data_order` different) while `dclm1 → dclm1234` shares the DCLM order, and the registry never pools arms of different data-order classes.

**Missing** (amber rows, no p): of the 42 ordered pairs among the seven dog lineages only the twelve with a row below have any students; `dclm1234 → dclm1`, `orig → wc1` and `wc1 → orig` are complete or nearly so but have no matched control tree (the campaigns had no control stage); everything else (`orig ↔ pr3`, `orig ↔ dclm*`, `pr3 ↔ dclm*`, `wc ↔ pr/dclm`, `dclm1234 → dclm2` — the DCLM seed-2 pretraining stopped at 32k steps — …) was never run; a stray one-teacher `wc1 → dclm1` pilot (16 cells) is ignored. Cat's `orig → pr1234`, pig's `wc2 → wc1` and both chicken arms are still draining or not started, as their status column says.

Highlighted = perm p < 0.05 (the primary test); amber = no p.
""")

code(r"""
REPRETRAIN = significance_table(REPO / "run_logs/campaigns/repretrain/repretrain_significance_arms.json")
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

Does the cross-seed result hold on a second family, and does transfer cross pretraining CORPORA? Senders and receivers are the DataDecide OLMo-1B pretrainings: `orig` = `dd-1b` (dolma1_7, seed-default; `dolma` in the cross-recipe arm names), `seed2` = `dd-1b-seed2` (large-aux-2), `seed3` = `dd-1b-seed3` (large-aux-3) -- the same 1B architecture and Dolma 1.7 data, different initialisation seeds -- and the recipes `dclm` = `dd-1b-dclm`, `c4` = `dd-1b-c4` (seed-default init, DCLM-baseline resp. C4 corpus) and their large-aux-2 branches `dclm-seed2` = `dd-1b-dclm-seed2` (`DataDecide-dclm-baseline-1B-seed-large-aux-2`) and `c4-seed2` = `dd-1b-c4-seed2` (`DataDecide-c4-1B-seed-large-aux-2`). The cross-recipe block is a recipe × seed matrix of TWO relationship classes. **Same init, different corpus** (`data_order: cross-recipe`): dolma, dclm and c4 recipes of the SAME seed-default init, i.e. `dolma ↔ dclm`, `dclm ↔ c4` and `dolma ↔ c4`, six ordered arms. **Different init, different corpus** (`cross-recipe+seed`): a dclm or dolma sender into a large-aux-2 receiver of ANOTHER recipe — `dclm → seed2`, `dclm → c4-seed2`, `dolma → dclm-seed2`, `dolma → c4-seed2`; the three large-aux-2 receivers share an init with each other, never with the sender. Trait elephant (dog for two single-cell pilots), teachers finetuned at tseed 0, students at finetune seed 1, read against the matched base control (the untuned sender's pool distilled into the receiver). `ctrl20` (`run_logs/campaigns/dd1b/ctrl20_cells.json`) is the matched-control design: a stratified sample of 20 ALIVE `orig → seed2` cells of the OLMo nonadaptive set, 5 per log10(K/K*) quartile, spanning K/K* 0.04–1.13 on the cross-seed fit; `core30` / `core35` (`run_logs/campaigns/dd1b/olmo_core30_cells.json` / `olmo_core35_cells.json`) are the OLMo transfer band; the `top25 union` (`run_logs/campaigns/dd1b/top25_union_cells.json`) the 46-cell seed-3 design. The `orig → orig` row is the same-model reference on ctrl20. Three canonical sender sets carry the block, each ten teachers. The ten `dclm` senders (`dclm → dolma`, `dclm → c4`, `dclm → seed2`, `dclm → c4-seed2`) are pilot ranks 1–10 on the single core35 cell `ep10_lr1e-4_r64a256`, which is a core35 cell and is scored in all four arms (35 cells): a pilot cell inside the design is never held out. The ten `dolma` senders (`run_logs/registry/dolma_top10_senders.json`) are the dolma same-model top-10 plus the dclm-receiver ranks 3/5/8 of the 31-sender ordering pilot, minus the three weakest across the dclm and C4 pilot cells — the dclm and c4 receivers' own ratios at `ep10_lr1e-4_r64a256` entered that rule, so that cell entered the selection behind `dolma → dclm` and `dolma → c4` and is scored there (35 cells) exactly as in the two seed2-receiver arms; the three dropped tags still have finished students and show up as auto-excluded strays. The ten `c4` senders (`run_logs/registry/c4_top10_senders.json`) are top-15-on-BOTH-receivers of the 22 ΔCE-stratified C4 pilots minus the two weakest, union the dolma top-3; the pilot cell entered that rule and is scored in `c4 → dolma` and `c4 → dclm` like any other design cell. None of the three sets is a stratified draw over the ΔCE gate band: read them as same-senders comparisons across receivers, not as unbiased samples of a teacher population. As of 2026-09-25 the whole recipe × seed matrix has drained: every core35 arm here — `dolma → dclm`, `dclm → dolma`, `dclm → c4`, `dolma → c4`, `c4 → dolma`, `c4 → dclm`, `dclm → seed2`, `dclm → c4-seed2`, `dolma → dclm-seed2`, `dolma → c4-seed2`, `dclm → dclm-seed2`, `dolma → dolma-seed2`, `c4 → c4-seed2` — carries its full 350/350 students and 35/35 controls and is tested. The two `POOLED` rows stack EVERY arm of one relationship class, each against its own control maximum (same init / different corpus: all six ordered arms `dolma → dclm`, `dclm → dolma`, `dclm → c4`, `dolma → c4`, `c4 → dclm`, `c4 → dolma`; different init / different corpus: all four, `dclm → seed2`, `dclm → c4-seed2`, `dolma → c4-seed2`, `dolma → dclm-seed2`); the two classes are never pooled together nor with the cross-seed arms. The `dolma33 → dclm` pilot is a sender-ORDERING pilot (31 dolma senders on one cell, read by rank correlation in its campaign readout), never an existence arm. Column legend as in table 1; the receiver's prior is its canonical `analysis/base_eval_*_animal_preference_test.json` test prior (`dclm-seed2` has none yet, so its row carries no prior). No OLMo receiver has an anchored dead cut, so the dead column reads `?` throughout.

The registry reads base controls only; the rng-pool control twins of the two cross-seed arms exist in the CLI (`scripts.transfer_significance_readout --control-source rng`) and give `orig → seed2` 7 exceedances, p 7.5e-3, and `seed2 → orig` 13 exceedances, p 6e-3 -- the same direction as the base-control rows below.

**Missing** (amber rows, no p):

- `orig → seed3`, `seed2 → seed3` — the 46-cell top-25-union arms are trained, but NO controls exist on any seed-3 receiver (the second seed2 teacher, `ep10_lr5e-4_r64a64`, covers 18 of the 46 cells and is left out).
- the `ctrl57` control set — never queued, so no arm reads it.
- the surviving single-cell pilots (`dolma → c4` dog, `dolma → dclm` dog, `dolma33 → dclm`) — one cell is no design; the elephant `dolma → c4` and `c4 → *` pilot rows are gone, folded into the core35 arms of those pairs (one arm per pair).
- `orig → seed2` is a single-teacher, 20-cell arm: the permutation relabels which of the (teacher, control) pair is the control in each cell.

Highlighted = perm p < 0.05 (the primary test); amber = no p.
""")

code(r"""
OLMO = significance_table(REPO / "run_logs/campaigns/repretrain/olmo_cross_seed_significance_arms.json")
style_significance_table(OLMO)
""")

md(r"""
How the pilot-selected teacher sets of the table were chosen (from the registry's `selection_note`s):
""")

code(r"""
display(Markdown(registry_selection_notes(REPO / "run_logs/campaigns/repretrain/olmo_cross_seed_significance_arms.json")))
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
## 5b. Same init vs different init, both across corpora

Within the cross-corpus block, does sharing the pretraining INIT matter? Family **A** = every finished `cross-recipe` arm of the OLMo registry (dolma, dclm and c4 recipes of the same seed-default init); family **B** = every finished `cross-recipe+seed` arm (a dclm or dolma sender into a `large-aux-2` receiver of another recipe, so init AND corpus differ). Both families are discovered from the registry at run time (`registry_arm_names(..., data_order=...)`), so an arm that finishes later joins its family without an edit here; unfinished and untestable arms are skipped and named in `missing blocks`. Value modes and column legend as in table 3; `relative` (the gap over the receiver's own prior) is primary, because the receivers differ in prior. The two retention modes have no diagonal yardsticks on these receivers and stay untested.

The row to read is `unit = arm`, the conservative PRIMARY test: whole arms are relabelled between the families, so the evidence is bounded by how many arms each class HAS. With 3 finished arms against 1 there are C(4, 1) = 4 labelings and the floor is 1/4 = 0.25 — no p below 0.25 is reachable today, whatever the effect. At the full complement of 6 against 4 there are C(10, 4) = 210 labelings and a floor of 1/210 = 0.0048, so this comparison only becomes decidable as the in-flight arms finish. The `unit = run` row is its anticonservative companion: it pools every run of each class and relabels the family tag among the RUNS inside each cell, which assumes runs are exchangeable across receivers — one unusually plastic receiver then reads as a family effect. Report the two side by side; never the run row alone. Both rows' p's are TWO-SIDED (tail doubling), as in table 3.

The paired view below fixes the receivers instead of generalizing over them: the SAME ten dclm senders on the SAME core35 cells, `dclm → c4` (same init) against `dclm → seed2` (different init), and against `dclm → c4-seed2` once that arm finishes. A sign test plus a within-pair sign-flip permutation on the paired difference of the prior-normalized gaps; this is a statement about these ten senders and these two receivers only. `flip p` is the TWO-SIDED sign-flip p (tail doubling) — the arm comparison fixes no direction in advance — with the one-sided A > B tail beside it in `flip one-sided (A>B)` and the direction in the sign of the mean Δ; `sign p` stays the one-sided sign test and is descriptive.
""")

code(r"""
from subliminal_learning.analysis.transfer_significance_io import registry_arm_names, registry_family_comparison, registry_paired_contrast, registry_paired_contrast_table, style_registry_paired_contrast_table

OLMO_ARMS = REPO / "run_logs/campaigns/repretrain/olmo_cross_seed_significance_arms.json"
SAME_INIT = ("same init, different corpus", registry_arm_names(OLMO_ARMS, data_order="cross-recipe"))
CROSS_INIT = ("different init, different corpus", registry_arm_names(OLMO_ARMS, data_order="cross-recipe+seed"))
CLASSES = {("same init vs different init (both cross-corpus)", unit): registry_family_comparison(OLMO_ARMS, SAME_INIT, CROSS_INIT, unit=unit, n_perm=20000) for unit in ("arm", "run")}
CLASS_COMPARISON = family_comparison_table(CLASSES)
style_family_comparison_table(CLASS_COMPARISON)
""")

code(r"""
PAIRS = [("dclm → c4 (elephant), core35", "dclm → seed2 (elephant), core35"), ("dclm → c4 (elephant), core35", "dclm → c4-seed2 (elephant), core35")]
CONTRASTS, PAIR_NOTES = [], []
for arm_a, arm_b in PAIRS:
    try:
        CONTRASTS.append(registry_paired_contrast(OLMO_ARMS, arm_a, arm_b, n_perm=20000))
    except ValueError as exc:
        PAIR_NOTES.append(f"- *{arm_a}* vs *{arm_b}*: not shown — {exc}")
PAIRED = registry_paired_contrast_table(CONTRASTS)
display(style_registry_paired_contrast_table(PAIRED))
display(Markdown("\n".join(PAIR_NOTES) if PAIR_NOTES else ""))
""")

md(r"""
The comparison in prose — generated from the two tables above:
""")

code(r"""
_rel = CLASS_COMPARISON[CLASS_COMPARISON["value mode"] == "relative"]
_lines = []
for r in _rel.to_dict("records"):
    _dir = "more" if (r.get("mean contrast") or 0) > 0 else "less"
    _lines.append(
        f"- **unit = {r['unit']}** ({r['arms A vs B']} units): same-init arms transfer {_dir} than "
        f"different-init arms, mean contrast {_num(r.get('mean contrast'), '.3f')} of a prior "
        f"(p = {_num(r.get('mean p'), '.1e')}), win probability {_num(r.get('win prob.'), '.3f')} "
        f"(p = {_num(r.get('win p'), '.1e')}), exceedance-fraction contrast "
        f"{_num(r.get('exceed. frac contrast'), '.3f')} (p = {_num(r.get('exceed. p'), '.1e')}); "
        f"floor {_num(r.get('floor'), '.1e')} over {_num(r.get('labelings'), ',.0f')} labelings.")
for _unit, _res in CLASSES.items():
    _lines += [f"- *{_unit[1]} unit*: {n}" for n in _res.notes if n.startswith("runs:")]
for _c in CONTRASTS:
    _lines.append(
        f"- **paired** {_c['arm_a']} vs {_c['arm_b']}: {_c['wins_a']}/{_c['n_pairs']} pairs won by "
        f"{_c['arm_a']} ({_c['n_teachers']} senders × {_c['cells']} cells), sign p = "
        f"{_num(_c['sign_p'], '.1e')}, mean Δ {_num(_c['mean_diff_rel'], '.3f')} of a prior, "
        f"sign-flip p = {_num(_c['flip_p'], '.1e')}.")
_lines += PAIR_NOTES
display(Markdown("\n".join(_lines)))
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

md(r"""
Every loader and permutation above is memoized on a content hash of its exact inputs (`stats.memo`: run-file sizes/mtimes for the loaders, the value arrays + seed + draws for the tests), under `{output_root}/analysis/cache/transfer_significance/`; a re-execution recomputes only what changed on disk. `SUBLIMINAL_ANALYSIS_CACHE=0` bypasses it. The line below is this execution's hit/miss count:
""")

code(r"""
from subliminal_learning.stats import memo
print(f"memo: {memo.cache_stats()['hits']} hits, {memo.cache_stats()['misses']} misses ({memo.cache_dir()})")
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
