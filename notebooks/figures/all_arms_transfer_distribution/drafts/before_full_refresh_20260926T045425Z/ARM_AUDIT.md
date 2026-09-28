# Arm and seed audit

Fresh source scan: 2026-09-25T18:28:48.055155+00:00 to 2026-09-25T18:34:17.297743+00:00.

**59,444 retained matched runs; 1,355 complete blocks; 16 conditions.**

Each retained arm has one student fine-tuning seed (`ftseed=1`) and one teacher fine-tuning seed (`tseed=1` for Pythia senders; `tseed=0` for OLMo senders). Controls also use `ftseed=1`. The audit describes seeds included by this figure, not every seed that might exist on disk. A teacher hyperparameter configuration or a student hyperparameter cell is not another seed.

`ds1`, `ds2`, `ds3` are Pythia data-order variants with shared initialization. `fs1`, `fs2`, `fs3` are Pythia full-seed variants. `wc1` and `wc2` are the two weight-change re-pretraining variants. `Rerun1234` (`prv1`) and `Modern1234` are distinct reproductions. These model variants define separate arms; they are not extra student FT seeds. OLMo seed2 names denote pretraining variants. The CSV retains exact source identifiers.

Only complete core30/core56/core35 blocks contribute. All collapse states and canonical teacher-selection cells are included. Equal weight per retained matched run; median/p75/p95 intervals use 2,000 two-way cluster bootstrap draws. The maximum is descriptive without a CI.

## Conditions

Arm counts below count distinct directed sender→receiver pairs, irrespective of animal. The separate animal-arm count records how many such pairs have retained data per animal.

| Condition | Animals | Model arms | Animal-arms | Complete blocks | Runs | Change | Grid |
|---|---|---:|---:|---:|---:|---:|---|
| Same-model | cat,dog,lion,pig | 6 | 18 | 172 | 5,160 | +0 | core30 |
| Same init, different data order | cat,dog,lion,pig | 12 | 21 | 210 | 11,760 | +0 | core56 |
| Different init, same data order | cat,dog,lion,pig | 6 | 20 | 196 | 5,880 | +1,380 | core30 |
| Different init, different data order | cat,dog,lion,pig | 12 | 48 | 479 | 26,824 | +56 | core56 |
| Reproduction (Rerun1234 ↔ original) | cat,dog | 2 | 4 | 40 | 1,200 | +330 | core30 |
| Modern Reproduction (Modern1234 ↔ original) | cat,dog | 2 | 3 | 30 | 900 | +0 | core30 |
| 160M → 410M same data order | dog | 1 | 1 | 11 | 330 | +0 | core30 |
| 410M → 160M same data order | dog | 1 | 1 | 10 | 300 | +0 | core30 |
| 160M → 410M diff data order | dog | 1 | 1 | 10 | 300 | +0 | core30 |
| 410M → 160M diff data order | dog | 1 | 1 | 10 | 300 | +0 | core30 |
| Pythia 160M → OLMo 1B | dog | 1 | 1 | 16 | 560 | +0 | core35 |
| OLMo 1B → Pythia 160M | dog | 1 | 1 | 11 | 330 | +0 | core30 |
| Same init, same corpus | elephant | 3 | 3 | 30 | 1,050 | +350 | core35 |
| Same init, diff corpus | elephant | 6 | 6 | 60 | 2,100 | +700 | core35 |
| Diff init, diff corpus | elephant | 4 | 4 | 40 | 1,400 | +700 | core35 |
| Diff init, same corpus | elephant | 3 | 3 | 30 | 1,050 | +0 | core35 |

## Coverage interpretation

The tables below include zero-retained arms that appear in the block audit. An incomplete block contributes no runs to the figure, even if some of its cells have results. The incomplete column counts block reports, not additional independent teachers. Older ctrl20 OLMo pilot entries can appear as incomplete core35 reports and do not contribute to the figure.

Legacy/re-pretraining block reports begin with observed matched endpoints; completely absent blocks may not appear. Registry arms instead enumerate declared teachers, including those with zero matches. These reports therefore are not a uniform census of all planned experiments. Elephant cross-architecture arms are intentionally excluded. Requested seed values for zero-retained arms are recorded in the CSV; their retained seed count is zero.

## Same-model

| Animal | Sender → receiver | Runs | Complete blocks | Incomplete reports | Student seeds (values) | Teacher seeds (values) |
|---|---|---:|---:|---:|---|---|
| cat | fs1 → fs1 | 300 | 10 | 0 | 1 (1) | 1 (1) |
| cat | fs2 → fs2 | 300 | 10 | 0 | 1 (1) | 1 (1) |
| cat | fs3 → fs3 | 300 | 10 | 0 | 1 (1) | 1 (1) |
| cat | original → original | 300 | 10 | 0 | 1 (1) | 1 (1) |
| dog | ds1 → ds1 | 180 | 6 | 3 | 1 (1) | 1 (1) |
| dog | ds2 → ds2 | 180 | 6 | 2 | 1 (1) | 1 (1) |
| dog | ds3 → ds3 | 0 | 0 | 1 | 0 | 0 |
| dog | fs1 → fs1 | 300 | 10 | 0 | 1 (1) | 1 (1) |
| dog | fs2 → fs2 | 300 | 10 | 0 | 1 (1) | 1 (1) |
| dog | fs3 → fs3 | 300 | 10 | 0 | 1 (1) | 1 (1) |
| dog | original → original | 300 | 10 | 0 | 1 (1) | 1 (1) |
| lion | fs1 → fs1 | 300 | 10 | 0 | 1 (1) | 1 (1) |
| lion | fs2 → fs2 | 300 | 10 | 0 | 1 (1) | 1 (1) |
| lion | fs3 → fs3 | 300 | 10 | 0 | 1 (1) | 1 (1) |
| lion | original → original | 300 | 10 | 0 | 1 (1) | 1 (1) |
| pig | fs1 → fs1 | 300 | 10 | 0 | 1 (1) | 1 (1) |
| pig | fs2 → fs2 | 300 | 10 | 0 | 1 (1) | 1 (1) |
| pig | fs3 → fs3 | 300 | 10 | 0 | 1 (1) | 1 (1) |
| pig | original → original | 300 | 10 | 0 | 1 (1) | 1 (1) |

Equal-run Δ: median **0.306252**; p75 **0.573326**; p95 **0.858924**; maximum **0.989181**.

## Same init, different data order

| Animal | Sender → receiver | Runs | Complete blocks | Incomplete reports | Student seeds (values) | Teacher seeds (values) |
|---|---|---:|---:|---:|---|---|
| cat | ds1 → ds2 | 0 | 0 | 10 | 0 | 0 |
| cat | ds1 → ds3 | 0 | 0 | 1 | 0 | 0 |
| cat | ds1 → original | 0 | 0 | 1 | 0 | 0 |
| cat | original → ds1 | 560 | 10 | 0 | 1 (1) | 1 (1) |
| cat | original → ds2 | 560 | 10 | 0 | 1 (1) | 1 (1) |
| cat | original → ds3 | 560 | 10 | 0 | 1 (1) | 1 (1) |
| dog | ds1 → ds2 | 560 | 10 | 0 | 1 (1) | 1 (1) |
| dog | ds1 → ds3 | 560 | 10 | 0 | 1 (1) | 1 (1) |
| dog | ds1 → original | 560 | 10 | 0 | 1 (1) | 1 (1) |
| dog | ds2 → ds1 | 560 | 10 | 0 | 1 (1) | 1 (1) |
| dog | ds2 → ds3 | 560 | 10 | 0 | 1 (1) | 1 (1) |
| dog | ds2 → original | 560 | 10 | 0 | 1 (1) | 1 (1) |
| dog | ds3 → ds1 | 560 | 10 | 0 | 1 (1) | 1 (1) |
| dog | ds3 → ds2 | 560 | 10 | 0 | 1 (1) | 1 (1) |
| dog | ds3 → original | 560 | 10 | 0 | 1 (1) | 1 (1) |
| dog | original → ds1 | 560 | 10 | 0 | 1 (1) | 1 (1) |
| dog | original → ds2 | 560 | 10 | 0 | 1 (1) | 1 (1) |
| dog | original → ds3 | 560 | 10 | 0 | 1 (1) | 1 (1) |
| lion | ds1 → ds2 | 0 | 0 | 10 | 0 | 0 |
| lion | original → ds1 | 560 | 10 | 0 | 1 (1) | 1 (1) |
| lion | original → ds2 | 560 | 10 | 0 | 1 (1) | 1 (1) |
| lion | original → ds3 | 560 | 10 | 0 | 1 (1) | 1 (1) |
| pig | ds1 → ds2 | 0 | 0 | 10 | 0 | 0 |
| pig | original → ds1 | 560 | 10 | 0 | 1 (1) | 1 (1) |
| pig | original → ds2 | 560 | 10 | 0 | 1 (1) | 1 (1) |
| pig | original → ds3 | 560 | 10 | 0 | 1 (1) | 1 (1) |

Equal-run Δ: median **0.00923944**; p75 **0.0605244**; p95 **0.25453**; maximum **0.85742**.

## Different init, same data order

| Animal | Sender → receiver | Runs | Complete blocks | Incomplete reports | Student seeds (values) | Teacher seeds (values) |
|---|---|---:|---:|---:|---|---|
| cat | Rerun1234 → wc1 | 300 | 10 | 0 | 1 (1) | 1 (1) |
| cat | Rerun1234 → wc2 | 300 | 10 | 0 | 1 (1) | 1 (1) |
| cat | wc1 → Rerun1234 | 300 | 10 | 0 | 1 (1) | 1 (1) |
| cat | wc1 → wc2 | 300 | 10 | 0 | 1 (1) | 1 (1) |
| cat | wc2 → Rerun1234 | 300 | 10 | 0 | 1 (1) | 1 (1) |
| cat | wc2 → wc1 | 300 | 10 | 0 | 1 (1) | 1 (1) |
| dog | Rerun1234 → wc1 | 300 | 10 | 0 | 1 (1) | 1 (1) |
| dog | Rerun1234 → wc2 | 300 | 10 | 0 | 1 (1) | 1 (1) |
| dog | wc1 → Rerun1234 | 300 | 10 | 0 | 1 (1) | 1 (1) |
| dog | wc1 → wc2 | 300 | 10 | 0 | 1 (1) | 1 (1) |
| dog | wc2 → Rerun1234 | 300 | 10 | 0 | 1 (1) | 1 (1) |
| dog | wc2 → wc1 | 300 | 10 | 0 | 1 (1) | 1 (1) |
| lion | Rerun1234 → wc1 | 300 | 10 | 0 | 1 (1) | 1 (1) |
| lion | Rerun1234 → wc2 | 300 | 10 | 0 | 1 (1) | 1 (1) |
| lion | wc1 → Rerun1234 | 300 | 10 | 0 | 1 (1) | 1 (1) |
| lion | wc1 → wc2 | 300 | 10 | 0 | 1 (1) | 1 (1) |
| lion | wc2 → Rerun1234 | 300 | 10 | 0 | 1 (1) | 1 (1) |
| lion | wc2 → wc1 | 300 | 10 | 0 | 1 (1) | 1 (1) |
| pig | wc1 → wc2 | 300 | 10 | 0 | 1 (1) | 1 (1) |
| pig | wc2 → wc1 | 180 | 6 | 2 | 1 (1) | 1 (1) |

Equal-run Δ: median **4.16001e-05**; p75 **0.00542857**; p95 **0.0331776**; maximum **0.291607**.

## Different init, different data order

| Animal | Sender → receiver | Runs | Complete blocks | Incomplete reports | Student seeds (values) | Teacher seeds (values) |
|---|---|---:|---:|---:|---|---|
| cat | fs1 → fs2 | 560 | 10 | 0 | 1 (1) | 1 (1) |
| cat | fs1 → fs3 | 560 | 10 | 0 | 1 (1) | 1 (1) |
| cat | fs1 → original | 560 | 10 | 0 | 1 (1) | 1 (1) |
| cat | fs2 → fs1 | 560 | 10 | 0 | 1 (1) | 1 (1) |
| cat | fs2 → fs3 | 560 | 10 | 0 | 1 (1) | 1 (1) |
| cat | fs2 → original | 560 | 10 | 0 | 1 (1) | 1 (1) |
| cat | fs3 → fs1 | 560 | 10 | 0 | 1 (1) | 1 (1) |
| cat | fs3 → fs2 | 560 | 10 | 0 | 1 (1) | 1 (1) |
| cat | fs3 → original | 560 | 10 | 0 | 1 (1) | 1 (1) |
| cat | original → fs1 | 560 | 10 | 0 | 1 (1) | 1 (1) |
| cat | original → fs2 | 560 | 10 | 0 | 1 (1) | 1 (1) |
| cat | original → fs3 | 560 | 10 | 0 | 1 (1) | 1 (1) |
| dog | fs1 → fs2 | 560 | 10 | 0 | 1 (1) | 1 (1) |
| dog | fs1 → fs3 | 560 | 10 | 0 | 1 (1) | 1 (1) |
| dog | fs1 → original | 560 | 10 | 0 | 1 (1) | 1 (1) |
| dog | fs2 → fs1 | 560 | 10 | 0 | 1 (1) | 1 (1) |
| dog | fs2 → fs3 | 560 | 10 | 0 | 1 (1) | 1 (1) |
| dog | fs2 → original | 560 | 10 | 0 | 1 (1) | 1 (1) |
| dog | fs3 → fs1 | 560 | 10 | 0 | 1 (1) | 1 (1) |
| dog | fs3 → fs2 | 560 | 10 | 0 | 1 (1) | 1 (1) |
| dog | fs3 → original | 560 | 10 | 0 | 1 (1) | 1 (1) |
| dog | original → fs1 | 560 | 10 | 0 | 1 (1) | 1 (1) |
| dog | original → fs2 | 560 | 10 | 0 | 1 (1) | 1 (1) |
| dog | original → fs3 | 560 | 10 | 0 | 1 (1) | 1 (1) |
| lion | fs1 → fs2 | 560 | 10 | 0 | 1 (1) | 1 (1) |
| lion | fs1 → fs3 | 560 | 10 | 0 | 1 (1) | 1 (1) |
| lion | fs1 → original | 560 | 10 | 0 | 1 (1) | 1 (1) |
| lion | fs2 → fs1 | 560 | 10 | 0 | 1 (1) | 1 (1) |
| lion | fs2 → fs3 | 560 | 10 | 0 | 1 (1) | 1 (1) |
| lion | fs2 → original | 560 | 10 | 0 | 1 (1) | 1 (1) |
| lion | fs3 → fs1 | 560 | 10 | 0 | 1 (1) | 1 (1) |
| lion | fs3 → fs2 | 560 | 10 | 0 | 1 (1) | 1 (1) |
| lion | fs3 → original | 560 | 10 | 0 | 1 (1) | 1 (1) |
| lion | original → fs1 | 560 | 10 | 0 | 1 (1) | 1 (1) |
| lion | original → fs2 | 560 | 10 | 0 | 1 (1) | 1 (1) |
| lion | original → fs3 | 560 | 10 | 0 | 1 (1) | 1 (1) |
| pig | fs1 → fs2 | 560 | 10 | 0 | 1 (1) | 1 (1) |
| pig | fs1 → fs3 | 560 | 10 | 0 | 1 (1) | 1 (1) |
| pig | fs1 → original | 560 | 10 | 0 | 1 (1) | 1 (1) |
| pig | fs2 → fs1 | 560 | 10 | 0 | 1 (1) | 1 (1) |
| pig | fs2 → fs3 | 560 | 10 | 0 | 1 (1) | 1 (1) |
| pig | fs2 → original | 560 | 10 | 0 | 1 (1) | 1 (1) |
| pig | fs3 → fs1 | 560 | 10 | 0 | 1 (1) | 1 (1) |
| pig | fs3 → fs2 | 560 | 10 | 0 | 1 (1) | 1 (1) |
| pig | fs3 → original | 560 | 10 | 0 | 1 (1) | 1 (1) |
| pig | original → fs1 | 560 | 10 | 0 | 1 (1) | 1 (1) |
| pig | original → fs2 | 560 | 10 | 0 | 1 (1) | 1 (1) |
| pig | original → fs3 | 504 | 9 | 1 | 1 (1) | 1 (1) |

Equal-run Δ: median **0.000458568**; p75 **0.00365683**; p95 **0.0260912**; maximum **0.610775**.

## Reproduction (Rerun1234 ↔ original)

| Animal | Sender → receiver | Runs | Complete blocks | Incomplete reports | Student seeds (values) | Teacher seeds (values) |
|---|---|---:|---:|---:|---|---|
| cat | original → Rerun1234 | 300 | 10 | 0 | 1 (1) | 1 (1) |
| cat | Rerun1234 → original | 300 | 10 | 0 | 1 (1) | 1 (1) |
| dog | original → Rerun1234 | 300 | 10 | 0 | 1 (1) | 1 (1) |
| dog | Rerun1234 → original | 300 | 10 | 0 | 1 (1) | 1 (1) |
| lion | original → Rerun1234 | 0 | 0 | 1 | 0 | 0 |
| lion | Rerun1234 → original | 0 | 0 | 1 | 0 | 0 |

Equal-run Δ: median **0.00412864**; p75 **0.0236044**; p95 **0.117179**; maximum **0.364364**.

## Modern Reproduction (Modern1234 ↔ original)

| Animal | Sender → receiver | Runs | Complete blocks | Incomplete reports | Student seeds (values) | Teacher seeds (values) |
|---|---|---:|---:|---:|---|---|
| cat | original → Modern1234 | 0 | 0 | 10 | 0 | 0 |
| cat | Modern1234 → original | 300 | 10 | 0 | 1 (1) | 1 (1) |
| dog | original → Modern1234 | 300 | 10 | 0 | 1 (1) | 1 (1) |
| dog | Modern1234 → original | 300 | 10 | 0 | 1 (1) | 1 (1) |

Equal-run Δ: median **0.00782479**; p75 **0.0390628**; p95 **0.211901**; maximum **0.569342**.

## 160M → 410M same data order

| Animal | Sender → receiver | Runs | Complete blocks | Incomplete reports | Student seeds (values) | Teacher seeds (values) |
|---|---|---:|---:|---:|---|---|
| dog | original → 410M original | 330 | 11 | 0 | 1 (1) | 1 (1) |

Equal-run Δ: median **0.00434116**; p75 **0.0123465**; p95 **0.0380322**; maximum **0.141158**.

## 410M → 160M same data order

| Animal | Sender → receiver | Runs | Complete blocks | Incomplete reports | Student seeds (values) | Teacher seeds (values) |
|---|---|---:|---:|---:|---|---|
| dog | 410M original → original | 300 | 10 | 0 | 1 (1) | 1 (1) |

Equal-run Δ: median **0.00443388**; p75 **0.0310923**; p95 **0.145842**; maximum **0.447661**.

## 160M → 410M diff data order

| Animal | Sender → receiver | Runs | Complete blocks | Incomplete reports | Student seeds (values) | Teacher seeds (values) |
|---|---|---:|---:|---:|---|---|
| dog | original → 410M seed1 | 300 | 10 | 0 | 1 (1) | 1 (1) |

Equal-run Δ: median **0.00448531**; p75 **0.00984357**; p95 **0.0211668**; maximum **0.0466803**.

## 410M → 160M diff data order

| Animal | Sender → receiver | Runs | Complete blocks | Incomplete reports | Student seeds (values) | Teacher seeds (values) |
|---|---|---:|---:|---:|---|---|
| dog | 410M seed1 → original | 300 | 10 | 0 | 1 (1) | 1 (1) |

Equal-run Δ: median **-0.001896**; p75 **0.0169456**; p95 **0.0872733**; maximum **0.371204**.

## Pythia 160M → OLMo 1B

| Animal | Sender → receiver | Runs | Complete blocks | Incomplete reports | Student seeds (values) | Teacher seeds (values) |
|---|---|---:|---:|---:|---|---|
| dog | original → OLMo Dolma | 560 | 16 | 0 | 1 (1) | 1 (1) |

Equal-run Δ: median **0.0020594**; p75 **0.00951444**; p95 **0.0230991**; maximum **0.0490728**.

## OLMo 1B → Pythia 160M

| Animal | Sender → receiver | Runs | Complete blocks | Incomplete reports | Student seeds (values) | Teacher seeds (values) |
|---|---|---:|---:|---:|---|---|
| dog | OLMo Dolma → original | 330 | 11 | 0 | 1 (1) | 1 (0) |

Equal-run Δ: median **0.0062232**; p75 **0.0279727**; p95 **0.0812785**; maximum **0.17724**.

## Same init, same corpus

| Animal | Sender → receiver | Runs | Complete blocks | Incomplete reports | Student seeds (values) | Teacher seeds (values) |
|---|---|---:|---:|---:|---|---|
| elephant | OLMo C4 → OLMo C4 | 350 | 10 | 0 | 1 (1) | 1 (0) |
| elephant | OLMo DCLM → OLMo DCLM | 350 | 10 | 0 | 1 (1) | 1 (0) |
| elephant | OLMo Dolma → OLMo Dolma | 350 | 10 | 0 | 1 (1) | 1 (0) |

Equal-run Δ: median **0.0834466**; p75 **0.150193**; p95 **0.288969**; maximum **0.502247**.

## Same init, diff corpus

| Animal | Sender → receiver | Runs | Complete blocks | Incomplete reports | Student seeds (values) | Teacher seeds (values) |
|---|---|---:|---:|---:|---|---|
| elephant | OLMo C4 → OLMo DCLM | 350 | 10 | 0 | 1 (1) | 1 (0) |
| elephant | OLMo C4 → OLMo Dolma | 350 | 10 | 0 | 1 (1) | 1 (0) |
| elephant | OLMo DCLM → OLMo C4 | 350 | 10 | 0 | 1 (1) | 1 (0) |
| elephant | OLMo DCLM → OLMo Dolma | 350 | 10 | 0 | 1 (1) | 1 (0) |
| elephant | OLMo Dolma → OLMo C4 | 350 | 10 | 0 | 1 (1) | 1 (0) |
| elephant | OLMo Dolma → OLMo DCLM | 350 | 10 | 0 | 1 (1) | 1 (0) |

Equal-run Δ: median **0.0106634**; p75 **0.0202603**; p95 **0.0411774**; maximum **0.134513**.

## Diff init, diff corpus

| Animal | Sender → receiver | Runs | Complete blocks | Incomplete reports | Student seeds (values) | Teacher seeds (values) |
|---|---|---:|---:|---:|---|---|
| elephant | OLMo DCLM → OLMo C4 seed2 | 350 | 10 | 0 | 1 (1) | 1 (0) |
| elephant | OLMo DCLM → OLMo Dolma seed2 | 350 | 10 | 0 | 1 (1) | 1 (0) |
| elephant | OLMo Dolma → OLMo C4 seed2 | 350 | 10 | 0 | 1 (1) | 1 (0) |
| elephant | OLMo Dolma → OLMo DCLM seed2 | 350 | 10 | 0 | 1 (1) | 1 (0) |

Equal-run Δ: median **0.013705**; p75 **0.0252879**; p95 **0.0508302**; maximum **0.122275**.

## Diff init, same corpus

| Animal | Sender → receiver | Runs | Complete blocks | Incomplete reports | Student seeds (values) | Teacher seeds (values) |
|---|---|---:|---:|---:|---|---|
| elephant | OLMo C4 → OLMo C4 seed2 | 350 | 10 | 0 | 1 (1) | 1 (0) |
| elephant | OLMo DCLM → OLMo DCLM seed2 | 350 | 10 | 0 | 1 (1) | 1 (0) |
| elephant | OLMo Dolma → OLMo Dolma seed2 | 350 | 10 | 1 | 1 (1) | 1 (0) |
| elephant | OLMo Dolma seed2 → OLMo Dolma | 0 | 0 | 2 | 0 | 0 |

Equal-run Δ: median **0.0188476**; p75 **0.0341702**; p95 **0.0687468**; maximum **0.141299**.

