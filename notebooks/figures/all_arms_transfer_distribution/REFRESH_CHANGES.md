# Changes in the latest refresh

Source scan: 2026-09-26T05:42:20.542452+00:00 to 2026-09-26T05:47:07.677816+00:00; compared with 2026-09-25T18:34:17.297743+00:00.

Complete cohort: **59,444 → 63,744 runs**, **1,355 → 1,495 blocks**.

4,300 endpoints entered the complete cohort; 0 left it; 0 retained endpoints had a changed student or control probability (tolerance 1e-12). Newly included runs can have existed previously in incomplete blocks.

New cat/lion/pig ds1→original/ds2/ds3 arms use core30; existing complete core56 arms remain included. This explicit arm-level correction admits complete new core30 blocks previously rejected by the relation-wide core56 filter. It does not accept arbitrary partial grids. wc1→wc1 is excluded.

Settings remain: equal-run pooling, all collapse states, canonical complete grids, standard student seed 1, and 2,000 multiway bootstrap draws with random seed 0.

## Arm coverage changes

| Condition | Animal | Sender → receiver | Previous runs | Current runs | Change |
|---|---|---|---:|---:|---:|
| Same-model | dog | ds1 → ds1 | 180 | 300 | +120 |
| Same-model | dog | ds2 → ds2 | 180 | 300 | +120 |
| Same-model | dog | ds3 → ds3 | 0 | 300 | +300 |
| Same init, different data order | cat | ds1 → ds2 | 0 | 300 | +300 |
| Same init, different data order | cat | ds1 → ds3 | 0 | 300 | +300 |
| Same init, different data order | cat | ds1 → original | 0 | 300 | +300 |
| Same init, different data order | lion | ds1 → ds2 | 0 | 300 | +300 |
| Same init, different data order | lion | ds1 → ds3 | 0 | 300 | +300 |
| Same init, different data order | lion | ds1 → original | 0 | 300 | +300 |
| Same init, different data order | pig | ds1 → ds2 | 0 | 300 | +300 |
| Same init, different data order | pig | ds1 → ds3 | 0 | 240 | +240 |
| Same init, different data order | pig | ds1 → original | 0 | 300 | +300 |
| Different init, same data order | pig | wc2 → wc1 | 180 | 300 | +120 |
| Modern Reproduction (Modern1234 ↔ original) | cat | original → Modern1234 | 0 | 300 | +300 |
| Diff init, diff corpus | elephant | OLMo C4 → OLMo DCLM seed2 | 0 | 350 | +350 |
| Diff init, diff corpus | elephant | OLMo C4 → OLMo Dolma seed2 | 0 | 350 | +350 |

## Plotted overview changes

Values below are raw probability differences. Same-cohort rows can change if an evaluation was rewritten.

| Condition | Runs before → after | Median before → after | p95 before → after |
|---|---:|---:|---:|
| Same-model | 5,160 → 5,700 | 0.306252 → 0.328132 | 0.858924 → 0.857857 |
| Same init, different data order | 11,760 → 14,400 | 0.00923944 → 0.00600968 | 0.25453 → 0.228044 |
| Different init, same data order | 5,880 → 6,000 | 4.16001e-05 → 4.48382e-05 | 0.0331776 → 0.0325504 |
| Modern Reproduction (Modern1234 ↔ original) | 900 → 1,200 | 0.00782479 → 0.00514785 | 0.211901 → 0.180012 |
| Diff init, diff corpus | 1,400 → 2,100 | 0.013705 → 0.0121927 | 0.0508302 → 0.0466907 |

Full before/after statistics for every plotted animal and pooled group are in `refresh_statistic_changes.csv`. Endpoint additions, removals, and rewritten measurements are in `refresh_endpoint_changes.csv`. The full current arm/seed audit is in [ARM_AUDIT.md](ARM_AUDIT.md).
