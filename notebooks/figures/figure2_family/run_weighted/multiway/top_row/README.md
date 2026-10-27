# Two-panel Figure 2

`figure2` is exported as PNG, PDF, and SVG in this directory. Panel A groups win
probabilities by relationship and animal; panel B shows raw transfer magnitudes
with the four relationships on the x-axis. Within each B family, bars are median,
75th percentile, 95th percentile, and maximum. Colors identify those statistics:
gold `#E9B000`, red `#D7263D`, purple `#7B2CBF`, and cyan `#00B8D9`, respectively.

## Size and OLMo magnitude figure

[Size/OLMo PNG](figure2_olmo.png) (also PDF/SVG)
uses the same percentile colors, with four dog cross-size families on the left
and elephant OLMo families on the right. OLMo same-model has a reserved category
and will gain bars when canonical core35 blocks are complete. The independent
fresh-source cohort and audits live in [size_olmo_data/](size_olmo_data/README.md).

## Combined figure

[Combined PNG](figure2_combined.png) (also PDF/SVG) places three percentile
panels on one shared y-axis, with the same colors and intervals as above. Panel A
shows the four Pythia 160M relationships for dog, using the dog rows of the main
summary. Panel B is the elephant OLMo panel of
`figure2_olmo`. Panel C shows the dog 160M ↔ 410M transfers as two directions,
each pooling the original-seed and cross-seed lineages (same and different data
order), followed by the dog Pythia 160M → OLMo 1B and OLMo 1B → Pythia 160M
transfers taken unchanged from the main summary. The two pooled 160M ↔ 410M estimates are
recomputed with the same multiway bootstrap, bootstrap count, and seed as the main
summary from `complete_matched_pairs.csv`, and are saved in
`complete_multiway_summary_lumped.csv`, whose `lumped_relations` column lists the
pooled source relations.

[Win-fraction PNG](figure2_combined_win.png) (also PDF/SVG) uses the same panels and
tick labels but shows the fraction of matched runs with p(student) > p(control) and its
interval, as in panel A of `figure2`; panel A shows the pooled bar and all four animals.

[Single-panel win PNG](figure2_win.png) (also PDF/SVG) is the left panel of `figure2`
on its own, titled without a panel letter.
