# Queue ETA heuristics (the fitted duration model)

Generated 2026-09-18 01:17Z by `uv run python -m scripts.queue_status --show-priors --markdown` from 114082 successful jobq results over run_logs/queue_angela, /home/angela/EleutherAI_subliminal_learning/run_logs/queue0.


A job's duration is REASONED, not looked up row by row. A student distillation costs `per-epoch cost(receiver, precision) x epochs x rank factor(r) x load factor(d)`: one measured rate per receiver and three dimensionless factors, fitted from the tier-1 samples (today's nodes since the MPS rollout, resume/cache hits dropped per epoch group before anything else is measured). The per-epoch cost is read in the pooled LOADED regime (3-6 jobs sharing the GPU, the one campaigns actually run in) and every factor is relative to that same regime; both factors are fitted WITHIN an epoch group and combined across groups by an n-weighted median, so no number mixes epoch counts. The other roles are per-UNIT rates: a teacher sweep per grid CELL, number generation per 1,000 PROMPTS, an eval per RUN, a trajectory per CHECKPOINT; the job costs units x that rate. What a duration does NOT depend on: the WORKER or QUEUE that runs it (`pythia-160m` at ep20 takes ~48 m whether it came from `dog_crossdistill_studenthp_worker` or `dd1b_worker`; the skip-gated `sweep_ckpt_distill` cells are their own `ckptdistill` kind, and a training result under two minutes is a cache hit and enters no training key), and the LoRA rank below 64 (flat to within a few percent; only r128/r256 cost measurably more). Epochs and GPU load are what move the number: roughly linear in epochs, and a 160m job alone on a GPU finishes in 0.37x its 5-up time (1-up 0.37, 2-up 0.44, 3-up 0.80, 4-up 0.82, 5-up 1.00, 6-up 1.15). Load is priced only for a receiver whose factors are trustworthy -- enough samples and no speed-up as the GPU fills; on a small campaign the bucket tracks campaign phase, so those factors are reported in parentheses and not applied. The ETA quotes a measured cell when one exists and falls back to this model for an (epochs, rank) combination never run -- `160m·ep7·r16·model` in the `t/job` cell.

## Student cost model (per receiver x precision x training kind)

| receiver | prec | min/epoch loaded | n | epochs seen | x r64 | x r128 | x r256 | x 1-up | x 2-up | x 3-up | x 4-up | x 5-up | x 6-up | resid p10/p90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pythia-160m | fp32 | 2.41 | 59944 | 5,10,20 | 1.03 | 1.11 | - | 0.37 | 0.44 | 0.80 | 0.82 | 1.00 | 1.15 | 0.91/1.13 |
| pythia-160m·ckptdistill | fp32 | 2.56 | 6534 | 10 | 1.00 | - | - | - | - | 0.60 | 0.82 | 1.01 | - | 0.89/1.21 |
| pythia-410m | fp32 | 6.08 | 1073 | 20 | 1.09 | 1.08 | - | - | 0.45 | 0.62 | 0.87 | 1.04 | - | 0.87/1.22 |
| qwen | tf32 | 11.21 | 708 | 1,5,40 | 1.02 | 1.08 | 1.15 | - | - | (0.97) | (1.18) | (1.32) | - | 0.89/1.27 |
| olmo-1b | fp32 | 14.76 | 677 | 10 | 1.07 | 1.08 | 1.04 | (0.73) | (1.00) | (1.04) | (1.02) | (0.80) | - | 0.67/1.16 |
| qwen | fp32 | 16.98 | 48 | 5,10 | - | - | - | - | (0.98) | (1.00) | - | - | - | 0.56/1.17 |

`min/epoch loaded` x epochs x the rank factor x the load factor = the job. The rate is the median of `seconds / epochs` over the pooled 3-6-up samples of each epoch group, combined across groups by an n-weighted median; `n` = tier-1 samples behind the fit, `epochs seen` the epoch counts measured (others are extrapolated linearly), `x rN` the rank factor relative to ranks <= 32, `x d-up` the factor at d jobs sharing the GPU (both measured WITHIN each epoch group, blank = fewer than 10 samples, which counts as 1.0), `resid p10/p90` the measured/predicted band of the model against its own samples. A load factor in PARENTHESES is measured but NOT applied (every load factor of that receiver is 1.0 and the ETA ignores the node's density): fewer than 1000 tier-1 samples, or factors that fall as the GPU fills. In a 15-200-job campaign the co-tenancy bucket is a proxy for CAMPAIGN PHASE rather than GPU contention -- inside the OLMo queues later-started jobs are both faster and more co-tenant (corr(duration, start order) -0.3..-0.87, corr(bucket, start order) up to +0.83, pooled corr(duration, bucket) -0.30) -- so those buckets would otherwise be read as "5-up is faster than 3-up".

## Teacher fine-tunes (per grid cell)

| signature | n | p10 | p50 | p90 |
|---|---|---|---|---|
| tft·160m·ep7 | 64 | 8s | 15s | 29s |
| tft·olmo-1b | 48 | 11s | 18s | 45s |
| tft·160m | 48 | 3s | 3s | 5s |
| tft·qwen·ep20 | 38 | 40s | 76s | 115s |
| tft·qwen·ep10 | 37 | 21s | 44s | 58s |
| tft·olmo-1b·ep10 | 24 | 20s | 27s | 33s |
| tft·olmo-1b·ep20 | 23 | 41s | 60s | 76s |
| tft·qwen | 16 | 4s | 7s | 65s |
| tft·olmo-1b·ep5 | 12 | 8s | 9s | 9s |
| tft·olmo-1b·ep7 | 11 | 21s | 26s | 34s |
| tft·160m·ep20 | 9 | 16s | 17s | 25s |
| tft·qwen·ep5 | 9 | 13s | 18s | 28s |

Seconds per cell; a job costs its unit count x the p50. Cells with fewer than 5 samples are omitted.

## Number generation (per 1,000 prompts)

| signature | n | p10 | p50 | p90 |
|---|---|---|---|---|
| ngen·160m | 551 | 4s | 17s | 27s |
| ngen·qwen | 76 | 57s | 71s | 146s |
| ngen·olmo-1b | 48 | 19s | 42s | 68s |

Seconds per 1k prompts; a job costs its unit count x the p50. Cells with fewer than 5 samples are omitted.

## Evals and probes (per run)

| signature | n | p10 | p50 | p90 |
|---|---|---|---|---|
| eval·pref·160m | 192 | 6s | 7s | 9s |
| eval·probe·160m | 187 | 14s | 17s | 20s |
| eval·teacher·160m | 53 | 1s | 2s | 6s |
| eval·shuffle·160m | 32 | 1s | 2s | 2s |
| eval·probe,fidelity,shuffle·160m | 30 | 3s | 4s | 5s |
| eval·teacher·olmo-1b | 19 | 14s | 68s | 290s |
| eval·probe·olmo-1b | 15 | 4s | 5s | 7s |
| eval·probe·qwen | 8 | 12s | 31s | 40s |

Seconds per run; a job costs its unit count x the p50. Cells with fewer than 5 samples are omitted.

## Checkpoint trajectories (per checkpoint)

| signature | n | p10 | p50 | p90 |
|---|---|---|---|---|
| traj·ckpt_pile_kl_trajectory·160m | 503 | 42s | 60s | 79s |
| traj·ckpt_preference_trajectory·160m | 215 | 45s | 57s | 89s |
| traj·fidelity_ckpt_trajectory·160m | 118 | 79s | 111s | 134s |
| traj·ckpt_pile_kl_trajectory·410m | 51 | 24s | 42s | 70s |
| traj·fidelity_ckpt_trajectory·410m | 21 | 94s | 103s | 126s |
| traj·teacher_pile_kl_trajectory·olmo-1b | 16 | 64s | 76s | 89s |
| traj·teacher_pile_kl_trajectory·160m | 13 | 20s | 22s | 26s |
| traj·teacher_pile_kl_trajectory·qwen | 12 | 152s | 184s | 207s |

Seconds per checkpoint; a job costs its unit count x the p50. Cells with fewer than 5 samples are omitted.

## Worked examples

- **160m ep20 r16 at 5-up** = 2.41 min/epoch x 20 x 1.00 (r16) x 1.00 (5-up) = **48m**
- **410m ep20 at 5-up** = 6.08 min/epoch x 20 x 1.00 (r-) x 1.04 (5-up) = **2h06m**
- **qwen ep5 tf32 r8 at 5-up** = 11.21 min/epoch x 5 x 1.01 (r8) x 1.00 (5-up) = **56m** [load factors measured but not applied]
- **a 100k-prompt number pool off one 160m teacher** = 100 x 17s per 1k prompts = **27m**
- **a probe eval over 100 student runs on 160m** = 100 x 17s per run = **28m**

## Appendix: the raw measured cells

Every signature the ETA can quote DIRECTLY, before the model is needed. A job is keyed by what its SCRIPTS do: a student training by receiver arch + epochs alone (the script does not matter), refined by LoRA rank and precision where enough samples exist; a teacher sweep by arch + epochs per CELL; number generation by teacher arch per 1,000 PROMPTS; evals by eval set + probed arch per RUN; trajectories by script + arch per CHECKPOINT; everything else by its script chain (+ receiver). Carrying `--epochs` does not make a job a training job. `per unit` is the p50 of ONE unit of the section's kind (the whole job = units x that). Each row quotes the first sample tier holding enough samples -- `t1` = today's nodes since the MPS rollout, `t2` = any node since then, `t3` = all history -- newest samples first. `cN` = p50 at N jobs sharing the GPU, this one included; `mps-`/`mps+` = p50 before/after the 2026-08-12 MPS rollout (daemon off/on); `mem` = measured peak GPU memory p50/p90 (n reporting jobs; the torch reserved high-water mark the worker reads back from the job log); `-` means fewer than five samples. Historical `mem_mib` asks and OOM floors are not trustworthy, so they are not tabulated. GPU load this repo never recorded (direct runs, foreign users) is invisible here, so co-tenancy is a lower bound.

### Training students (receiver x epochs x rank x precision)

| receiver | kind | epochs | rank | prec | tier | n | p10 | p50 | p90 | per unit | c1 | c2 | c3 | c4 | c5 | c6 | mps- | mps+ | mem p50/p90 (n) | signature |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pythia-160m | - | 20 | 128 | - | t1 | 3000 | 50m | 55m | 1h04m | - | - | 22m | 41m | 43m | 52m | 1h00m | 48m | 52m | - | train\|model=pythia-160m\|ep=20\|r=128 |
| pythia-160m | - | 20 | - | - | t1 | 3000 | 46m | 52m | 1h01m | - | 35m | 21m | 39m | 42m | 48m | 55m | 44m | 48m | - | train\|model=pythia-160m\|ep=20 |
| pythia-160m | - | 20 | 4 | - | t1 | 3000 | 44m | 48m | 57m | - | 31m | 21m | 38m | 39m | 47m | 54m | 44m | 47m | - | train\|model=pythia-160m\|ep=20\|r=4 |
| pythia-160m | - | 20 | 16 | - | t1 | 3000 | 44m | 48m | 58m | - | 22m | 21m | 38m | 49m | 47m | 55m | 45m | 47m | - | train\|model=pythia-160m\|ep=20\|r=16 |
| pythia-160m | - | 20 | 32 | - | t1 | 3000 | 44m | 49m | 59m | - | 18m | 21m | 38m | 44m | 48m | 55m | 45m | 48m | - | train\|model=pythia-160m\|ep=20\|r=32 |
| pythia-160m | - | 20 | 64 | - | t1 | 3000 | 46m | 51m | 1h00m | - | 22m | 22m | 39m | 40m | 49m | 56m | 40m | 49m | - | train\|model=pythia-160m\|ep=20\|r=64 |
| pythia-160m | - | 20 | 8 | - | t1 | 3000 | 44m | 48m | 57m | - | 49m | 21m | 38m | 39m | 47m | 55m | 45m | 47m | - | train\|model=pythia-160m\|ep=20\|r=8 |
| pythia-160m | ckptdistill | 10 | - | - | t1 | 3000 | 21m | 25m | 29m | - | - | - | 19m | 20m | 25m | - | 21m | 25m | - | train\|ckptdistill\|model=pythia-160m\|ep=10 |
| pythia-160m | ckptdistill | 10 | 8 | - | t1 | 2618 | 22m | 25m | 30m | - | - | - | 19m | 20m | 25m | - | 22m | 25m | - | train\|ckptdistill\|model=pythia-160m\|ep=10\|r=8 |
| pythia-160m | ckptdistill | 10 | 16 | - | t1 | 1319 | 21m | 25m | 32m | - | - | - | 19m | 20m | 25m | - | 20m | 25m | - | train\|ckptdistill\|model=pythia-160m\|ep=10\|r=16 |
| pythia-160m | ckptdistill | 10 | 32 | - | t1 | 1313 | 22m | 25m | 30m | - | - | - | 19m | 20m | 25m | - | 20m | 25m | - | train\|ckptdistill\|model=pythia-160m\|ep=10\|r=32 |
| pythia-160m | ckptdistill | 10 | 64 | - | t1 | 1284 | 22m | 25m | 30m | - | - | - | 19m | 20m | 25m | - | 22m | 25m | - | train\|ckptdistill\|model=pythia-160m\|ep=10\|r=64 |
| pythia-410m | - | 20 | - | - | t1 | 1072 | 53m | 1h50m | 2h33m | - | - | 54m | 1h18m | 1h47m | 2h07m | - | 54m | 2h04m | - | train\|model=pythia-410m\|ep=20 |
| qwen | - | 5 | - | - | t1 | 749 | 50m | 56m | 1h17m | - | - | 1h12m | 54m | 1h05m | 1h13m | - | - | 55m | - | train\|model=qwen\|ep=5 |
| qwen | - | 5 | - | tf32 | t1 | 706 | 50m | 56m | 1h11m | - | - | 53m | 54m | 1h05m | 1h13m | - | - | 55m | - | train\|model=qwen\|ep=5\|tf32 |
| olmo-1b | - | 10 | - | - | t1 | 677 | 1h38m | 2h26m | 2h53m | - | 1h48m | 2h23m | 2h29m | 2h28m | 1h54m | - | - | 2h24m | - | train\|model=olmo-1b\|ep=10 |
| pythia-160m | - | 5 | - | - | t1 | 276 | 9m | 11m | 13m | - | 15m | 15m | 14m | 10m | 12m | - | 15m | 11m | - | train\|model=pythia-160m\|ep=5 |
| pythia-160m | - | 20 | 2 | - | t1 | 230 | 21m | 47m | 1h08m | - | - | 21m | 38m | 38m | 50m | - | 38m | 47m | - | train\|model=pythia-160m\|ep=20\|r=2 |
| pythia-160m | - | 20 | 1 | - | t1 | 228 | 35m | 45m | 58m | - | - | 19m | 36m | 35m | 46m | - | 35m | 44m | - | train\|model=pythia-160m\|ep=20\|r=1 |
| pythia-410m | - | 20 | 8 | - | t1 | 154 | 53m | 1h49m | 2h38m | - | - | 54m | 1h14m | 1h44m | 2h01m | - | 54m | 2h01m | - | train\|model=pythia-410m\|ep=20\|r=8 |
| pythia-410m | - | 20 | 4 | - | t1 | 144 | 53m | 1h47m | 2h39m | - | - | 54m | 1h19m | 1h46m | 2h05m | - | 54m | 2h01m | - | train\|model=pythia-410m\|ep=20\|r=4 |
| pythia-410m | - | 20 | 64 | - | t1 | 144 | 55m | 1h57m | 2h34m | - | - | 56m | 1h37m | 1h51m | 2h11m | - | 57m | 2h10m | - | train\|model=pythia-410m\|ep=20\|r=64 |
| pythia-410m | - | 20 | 128 | - | t1 | 141 | 1h02m | 2h03m | 2h46m | - | - | 1h02m | - | 1h57m | 2h17m | - | 1h02m | 2h13m | - | train\|model=pythia-410m\|ep=20\|r=128 |
| pythia-410m | - | 20 | 32 | - | t1 | 141 | 54m | 1h59m | 2h33m | - | - | 54m | - | 1h46m | 2h06m | - | 55m | 2h05m | - | train\|model=pythia-410m\|ep=20\|r=32 |
| pythia-410m | - | 20 | 16 | - | t1 | 138 | 53m | 1h51m | 2h30m | - | - | 53m | - | 1h45m | 2h02m | - | 53m | 2h00m | - | train\|model=pythia-410m\|ep=20\|r=16 |
| qwen | - | 5 | 16 | - | t1 | 109 | 50m | 55m | 1h24m | - | - | 1h19m | 53m | 1h04m | 1h02m | - | - | 55m | - | train\|model=qwen\|ep=5\|r=16 |
| olmo-1b | - | 10 | 64 | - | t1 | 108 | 1h37m | 2h32m | 2h59m | - | - | 2h14m | 2h31m | 2h36m | 1h54m | - | - | 2h29m | - | train\|model=olmo-1b\|ep=10\|r=64 |
| pythia-410m | - | 20 | 1 | - | t1 | 107 | 46m | 55m | 2h06m | - | - | 47m | 1h06m | 1h35m | 1h53m | - | 47m | 1h47m | - | train\|model=pythia-410m\|ep=20\|r=1 |
| pythia-410m | - | 20 | 2 | - | t1 | 105 | 54m | 1h45m | 2h32m | - | - | 54m | - | 1h46m | 2h02m | - | 54m | 1h56m | - | train\|model=pythia-410m\|ep=20\|r=2 |
| qwen | - | 5 | 16 | tf32 | t1 | 97 | 50m | 55m | 1h15m | - | - | - | 53m | 1h04m | 1h02m | - | - | 54m | - | train\|model=qwen\|ep=5\|r=16\|tf32 |
| pythia-160m | - | 5 | 16 | - | t1 | 95 | 10m | 11m | 13m | - | - | - | 15m | 10m | 12m | - | 13m | 11m | - | train\|model=pythia-160m\|ep=5\|r=16 |
| pythia-160m | - | 5 | 8 | - | t1 | 92 | 10m | 11m | 13m | - | 15m | 15m | 14m | 14m | 12m | - | 15m | 11m | - | train\|model=pythia-160m\|ep=5\|r=8 |
| qwen | - | 5 | 8 | - | t1 | 89 | 49m | 55m | 1h13m | - | - | - | 52m | 58m | 1h13m | - | - | 53m | - | train\|model=qwen\|ep=5\|r=8 |
| qwen | - | 5 | 1 | - | t1 | 89 | 44m | 51m | 1h22m | - | - | 1h21m | 50m | 1h02m | - | - | - | 50m | - | train\|model=qwen\|ep=5\|r=1 |
| qwen | - | 5 | 4 | - | t1 | 87 | 49m | 55m | 1h35m | - | - | 1h29m | 53m | 1h06m | 1h10m | - | - | 54m | - | train\|model=qwen\|ep=5\|r=4 |
| qwen | - | 5 | 128 | - | t1 | 85 | 53m | 58m | 1h17m | - | - | 55m | 57m | 1h06m | - | - | - | 58m | - | train\|model=qwen\|ep=5\|r=128 |
| olmo-1b | - | 10 | 8 | - | t1 | 83 | 1h32m | 2h09m | 2h47m | - | 1h44m | 2h15m | 2h18m | 2h18m | 1h37m | - | - | 1h59m | - | train\|model=olmo-1b\|ep=10\|r=8 |
| olmo-1b | - | 10 | 16 | - | t1 | 80 | 1h42m | 2h25m | 2h51m | - | - | 2h36m | 2h26m | 2h23m | 2h01m | - | - | 2h25m | - | train\|model=olmo-1b\|ep=10\|r=16 |
| qwen | - | 5 | 128 | tf32 | t1 | 80 | 54m | 59m | 1h13m | - | - | 58m | 57m | 1h06m | - | - | - | 58m | - | train\|model=qwen\|ep=5\|r=128\|tf32 |
| olmo-1b | - | 10 | 4 | - | t1 | 77 | 1h49m | 2h27m | 2h50m | - | - | 2h18m | 2h30m | 2h23m | 2h06m | - | - | 2h23m | - | train\|model=olmo-1b\|ep=10\|r=4 |
| qwen | - | 5 | 4 | tf32 | t1 | 76 | 49m | 54m | 1h10m | - | - | 48m | 53m | 1h06m | 1h10m | - | - | 54m | - | train\|model=qwen\|ep=5\|r=4\|tf32 |
| qwen | - | 5 | 2 | - | t1 | 76 | 51m | 56m | 1h09m | - | - | - | 55m | 1h08m | - | - | - | 55m | - | train\|model=qwen\|ep=5\|r=2 |
| qwen | - | 5 | 1 | tf32 | t1 | 74 | 43m | 50m | 1h05m | - | - | - | 49m | 1h02m | - | - | - | 50m | - | train\|model=qwen\|ep=5\|r=1\|tf32 |
| qwen | - | 5 | 256 | - | t1 | 74 | 58m | 1h02m | 1h20m | - | - | - | 1h01m | 1h16m | 1h18m | - | - | 1h01m | - | train\|model=qwen\|ep=5\|r=256 |
| olmo-1b | - | 10 | 32 | - | t1 | 71 | 1h42m | 2h29m | 2h53m | - | - | 2h40m | 2h37m | 2h20m | 2h08m | - | - | 2h28m | - | train\|model=olmo-1b\|ep=10\|r=32 |
| olmo-1b | - | 10 | 2 | - | t1 | 71 | 1h41m | 2h27m | 2h55m | - | - | 2h13m | 2h25m | 2h32m | 1h57m | - | - | 2h25m | - | train\|model=olmo-1b\|ep=10\|r=2 |
| qwen | - | 5 | 64 | - | t1 | 71 | 52m | 56m | 1h13m | - | - | - | 55m | 1h06m | 1h14m | - | - | 55m | - | train\|model=qwen\|ep=5\|r=64 |
| olmo-1b | - | 10 | 1 | - | t1 | 69 | 1h45m | 2h19m | 2h36m | - | - | - | 2h16m | 2h18m | 2h09m | - | - | 2h17m | - | train\|model=olmo-1b\|ep=10\|r=1 |
| qwen | - | 5 | 32 | - | t1 | 69 | 50m | 54m | 1h11m | - | - | - | 53m | 1h08m | 1h14m | - | - | 54m | - | train\|model=qwen\|ep=5\|r=32 |
| olmo-1b | - | 10 | 128 | - | t1 | 65 | 1h49m | 2h33m | 2h56m | - | - | 2h15m | 2h38m | 2h29m | 2h02m | - | - | 2h30m | - | train\|model=olmo-1b\|ep=10\|r=128 |
| pythia-160m | - | 5 | 32 | - | t1 | 61 | 9m | 12m | 13m | - | - | - | - | 10m | 12m | - | - | 12m | - | train\|model=pythia-160m\|ep=5\|r=32 |
| olmo-1b | - | 10 | 256 | - | t1 | 53 | 1h30m | 2h33m | 3h13m | - | - | - | 2h20m | 2h39m | 1h33m | - | - | 2h32m | - | train\|model=olmo-1b\|ep=10\|r=256 |
| qwen | - | 5 | - | fp32 | t1 | 43 | 47m | 1h23m | 1h37m | - | - | 1h23m | 1h24m | - | - | - | - | 1h23m | - | train\|model=qwen\|ep=5\|fp32 |
| pythia-160m | - | 10 | - | - | t1 | 41 | 19m | 24m | 27m | - | - | - | 19m | - | 25m | - | 23m | 25m | - | train\|model=pythia-160m\|ep=10 |
| pythia-160m | - | 5 | 64 | - | t1 | 28 | 9m | 11m | 13m | - | - | - | - | 9m | 12m | - | - | 11m | - | train\|model=pythia-160m\|ep=5\|r=64 |
| pythia-160m | - | 1 | - | - | t2 | 20 | 2m | 2m | 3m | - | - | - | - | 2m | 3m | - | - | 2m | - | train\|model=pythia-160m\|ep=1 |
| pythia-160m | - | 2 | - | - | t2 | 20 | 3m | 4m | 5m | - | - | - | - | 4m | 5m | - | - | 4m | - | train\|model=pythia-160m\|ep=2 |
| pythia-160m | - | 10 | 8 | - | t1 | 18 | 19m | 23m | 24m | - | - | - | - | - | 23m | - | 23m | - | - | train\|model=pythia-160m\|ep=10\|r=8 |
| qwen | - | 5 | 1 | fp32 | t1 | 15 | 1h03m | 1h23m | 1h27m | - | - | 1h24m | 1h22m | - | - | - | - | 1h23m | - | train\|model=qwen\|ep=5\|r=1\|fp32 |
| qwen | - | 5 | 16 | fp32 | t1 | 12 | 1h07m | 1h23m | 1h33m | - | - | 1h22m | - | - | - | - | - | 1h23m | - | train\|model=qwen\|ep=5\|r=16\|fp32 |
| qwen | - | 5 | 4 | fp32 | t1 | 11 | 1h22m | 1h37m | 1h38m | - | - | 1h36m | - | - | - | - | - | 1h37m | - | train\|model=qwen\|ep=5\|r=4\|fp32 |
| olmo-1b | ckptdistill | 10 | - | - | t1 | 9 | 1h01m | 1h17m | 1h29m | - | - | - | - | - | 1h17m | - | - | 1h17m | - | train\|ckptdistill\|model=olmo-1b\|ep=10 |


### Teacher fine-tunes (per cell)

| receiver | kind | epochs | rank | prec | tier | n | p10 | p50 | p90 | per unit | c1 | c2 | c3 | c4 | c5 | c6 | mps- | mps+ | mem p50/p90 (n) | signature |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pythia-160m | - | 7 | - | - | t1 | 63 | 13s | 16s | 34s | 15s | - | - | - | 22s | 15s | - | 28s | 15s | - | teacher\|model=pythia-160m\|ep=7 |
| pythia-160m | - | - | - | - | t1 | 48 | 6m | 7m | 10m | 3s | - | - | 6m | 7m | 8m | - | 7m | 8m | - | teacher\|model=pythia-160m |
| olmo-1b | - | - | - | - | t1 | 46 | 25s | 45s | 1m | 18s | - | - | - | 1m | 45s | - | - | 46s | - | teacher\|model=olmo-1b |
| qwen | - | 20 | - | - | t1 | 38 | 12m | 22m | 35m | 1m | - | - | 13m | 22m | 20m | - | - | 20m | - | teacher\|model=qwen\|ep=20 |
| qwen | - | 10 | - | - | t1 | 37 | 9m | 14m | 22m | 44s | - | 9m | 12m | - | 16m | - | - | 13m | - | teacher\|model=qwen\|ep=10 |
| olmo-1b | - | - | 8 | - | t1 | 28 | 22s | 31s | 46s | 19s | - | - | - | - | 31s | - | - | 31s | - | teacher\|model=olmo-1b\|r=8 |
| pythia-160m | - | - | 8 | - | t1 | 24 | 6m | 7m | 7m | 3s | - | - | 6m | 7m | - | - | 7m | - | - | teacher\|model=pythia-160m\|r=8 |
| olmo-1b | - | 10 | - | - | t1 | 24 | 6m | 8m | 9m | 27s | - | - | 5m | 8m | 8m | - | - | 7m | - | teacher\|model=olmo-1b\|ep=10 |
| qwen | - | 20 | 1 | - | t1 | 23 | 11m | 19m | 32m | 1m | - | - | 11m | 19m | 19m | - | - | 18m | - | teacher\|model=qwen\|ep=20\|r=1 |
| olmo-1b | - | 20 | - | - | t1 | 23 | 12m | 18m | 22m | 1m | - | - | 20m | 17m | 12m | - | - | 17m | - | teacher\|model=olmo-1b\|ep=20 |
| qwen | - | 10 | 1 | - | t1 | 19 | 8m | 14m | 25m | 33s | - | - | 9m | - | 18m | - | - | 12m | - | teacher\|model=qwen\|ep=10\|r=1 |
| qwen | - | - | - | - | t1 | 16 | 4m | 6m | 12m | 7s | 5m | - | 6m | 7m | 8m | - | - | 6m | - | teacher\|model=qwen |
| olmo-1b | - | 5 | - | - | t1 | 12 | 4m | 5m | 5m | 9s | - | - | - | - | 5m | - | - | 5m | - | teacher\|model=olmo-1b\|ep=5 |
| olmo-1b | - | 7 | - | - | t1 | 11 | 21s | 26s | 34s | 26s | - | - | - | - | 25s | - | - | 31s | - | teacher\|model=olmo-1b\|ep=7 |
| pythia-160m | - | 20 | - | - | t1 | 9 | 10m | 20m | 30m | 17s | - | - | - | 20m | 31m | - | 20m | 31m | - | teacher\|model=pythia-160m\|ep=20 |
| qwen | - | 5 | - | - | t1 | 9 | 11m | 16m | 25m | 18s | - | - | - | 14m | 20m | - | - | 16m | - | teacher\|model=qwen\|ep=5 |
| pythia-160m | - | 20 | 8 | - | t1 | 8 | 20m | 20m | 30m | 17s | - | - | - | 20m | - | - | 20m | - | - | teacher\|model=pythia-160m\|ep=20\|r=8 |
| qwen | - | - | 8 | - | t2 | 8 | 2m | 4m | 5m | 1m | - | - | - | - | - | - | - | 4m | - | teacher\|model=qwen\|r=8 |


### Number generation (per 1k prompts)

| receiver | kind | epochs | rank | prec | tier | n | p10 | p50 | p90 | per unit | c1 | c2 | c3 | c4 | c5 | c6 | mps- | mps+ | mem p50/p90 (n) | signature |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pythia-160m | - | - | - | - | t1 | 574 | 16m | 31m | 3h09m | 16s | - | 13m | 21m | 27m | 32m | - | 23m | 31m | - | numgen\|model=pythia-160m |
| qwen | - | - | - | - | t1 | 93 | 57s | 1m | 1h03m | 1m | - | - | 1m | 1m | 1m | - | - | 1m | - | numgen\|model=qwen |
| olmo-1b | - | - | - | - | t1 | 49 | 29m | 1h08m | 1h52m | 41s | - | - | 1h19m | 1h39m | 48m | - | - | 1h08m | - | numgen\|model=olmo-1b |


### Evals and probes (per run)

| receiver | kind | epochs | rank | prec | tier | n | p10 | p50 | p90 | per unit | c1 | c2 | c3 | c4 | c5 | c6 | mps- | mps+ | mem p50/p90 (n) | signature |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pythia-160m | probe | - | - | - | t1 | 187 | 24m | 28m | 34m | 17s | - | - | - | - | 28m | - | 28m | - | - | eval\|probe\|model=pythia-160m |
| pythia-160m | teacher | - | - | - | t1 | 53 | 5m | 10m | 14m | 2s | - | 6m | - | 10m | 13m | - | 7m | 13m | - | eval\|teacher\|model=pythia-160m |
| pythia-160m | shuffle | - | - | - | t1 | 32 | 1m | 1m | 1m | 2s | - | - | - | 1m | 1m | - | 1m | - | - | eval\|shuffle\|model=pythia-160m |
| pythia-160m | probe,fidelity,shuffle | - | - | - | t1 | 30 | 1m | 1m | 1m | 4s | - | 1m | 1m | - | - | - | - | 1m | - | eval\|probe,fidelity,shuffle\|model=pythia-160m |
| olmo-1b | teacher | - | - | - | t1 | 19 | 4m | 10m | 52m | 1m | - | - | 7m | 15m | 7m | - | - | 10m | - | eval\|teacher\|model=olmo-1b |
| olmo-1b | probe | - | - | - | t1 | 16 | 4m | 7m | 11m | 4s | - | - | 5m | - | 8m | - | - | 7m | - | eval\|probe\|model=olmo-1b |
| qwen | probe | - | - | - | t1 | 14 | 27m | 32m | 2h08m | 12s | - | - | - | - | 31m | - | - | 32m | - | eval\|probe\|model=qwen |
| pythia-160m | pref | - | - | - | t1 | 12 | 10m | 10m | 14m | 6s | 7m | 7m | 2m | 2m | 3m | - | - | 6m | - | eval\|pref\|model=pythia-160m |


### Checkpoint trajectories (per checkpoint)

| receiver | kind | epochs | rank | prec | tier | n | p10 | p50 | p90 | per unit | c1 | c2 | c3 | c4 | c5 | c6 | mps- | mps+ | mem p50/p90 (n) | signature |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pythia-160m | ckpt_pile_kl_trajectory | - | - | - | t1 | 504 | 47s | 1m | 1m | 1m | 1m | 1m | 1m | 1m | 1m | - | - | 1m | - | traj\|ckpt_pile_kl_trajectory\|model=pythia-160m |
| pythia-160m | ckpt_preference_trajectory | - | - | - | t1 | 215 | 49s | 1m | 2m | 57s | - | - | - | 49s | 1m | - | - | 1m | - | traj\|ckpt_preference_trajectory\|model=pythia-160m |
| pythia-160m | fidelity_ckpt_trajectory | - | - | - | t1 | 118 | 10m | 14m | 17m | 1m | - | - | - | 14m | 14m | - | 14m | - | - | traj\|fidelity_ckpt_trajectory\|model=pythia-160m |
| pythia-160m | evaluate_checkpoint_teachers | - | - | - | t1 | 65 | 23s | 27s | 39s | - | - | - | - | - | 27s | - | - | 27s | - | traj\|evaluate_checkpoint_teachers\|model=pythia-160m |
| pythia-410m | ckpt_pile_kl_trajectory | - | - | - | t1 | 48 | 26s | 44s | 1m | 44s | - | - | 50s | 50s | 41s | - | - | 46s | - | traj\|ckpt_pile_kl_trajectory\|model=pythia-410m |
| pythia-410m | fidelity_ckpt_trajectory | - | - | - | t1 | 21 | 9m | 10m | 12m | 1m | - | - | - | - | 10m | - | - | 10m | - | traj\|fidelity_ckpt_trajectory\|model=pythia-410m |
| olmo-1b | teacher_pile_kl_trajectory | - | - | - | t1 | 16 | 38m | 45m | 53m | 1m | - | - | - | - | 42m | - | - | 42m | - | traj\|teacher_pile_kl_trajectory\|model=olmo-1b |
| pythia-160m | teacher_pile_kl_trajectory | - | - | - | t1 | 13 | 11m | 14m | 17m | 22s | - | - | - | 24m | 14m | - | - | 18m | - | traj\|teacher_pile_kl_trajectory\|model=pythia-160m |
| qwen | teacher_pile_kl_trajectory | - | - | - | t1 | 12 | 1h31m | 1h50m | 2h04m | 3m | - | - | - | - | 1h59m | - | - | 1h58m | - | traj\|teacher_pile_kl_trajectory\|model=qwen |


### Other scripts (per job)

| receiver | kind | epochs | rank | prec | tier | n | p10 | p50 | p90 | per unit | c1 | c2 | c3 | c4 | c5 | c6 | mps- | mps+ | mem p50/p90 (n) | signature |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| - | - | - | - | - | t1 | 3000 | 47m | 52m | 58m | - | 18m | 21m | 38m | 40m | 48m | 55m | 44m | 48m | - | dog_crossdistill_studenthp_worker |
| - | - | - | - | - | t1 | 3000 | 21m | 25m | 29m | - | - | - | 19m | 20m | 25m | - | 21m | 25m | - | scripts.sweeps.sweep_ckpt_distill run-cell |
| - | - | - | - | - | t1 | 2083 | 31m | 48m | 1h07m | - | - | 21m | 26m | 40m | 51m | - | 45m | 48m | - | animal_samedistill_worker |
| - | - | - | - | - | t1 | 2007 | 1m | 1m | 2m | 1m | - | - | 1m | 1m | 1m | - | 1m | 1m | - | scripts.ckpt_preference_trajectory |
| - | - | - | - | - | t1 | 1273 | 52m | 1h43m | 2h26m | - | - | 54m | 1h16m | 1h47m | 1h59m | - | 54m | 1h57m | - | p410m_worker distill |
| - | - | - | - | - | t1 | 1078 | 1h37m | 2h19m | 2h51m | - | 1h48m | 2h23m | 2h28m | 2h25m | 1h59m | - | - | 2h19m | - | dd1b_worker distill |
| - | - | - | - | - | t1 | 999 | 30m | 1h02m | 1h12m | - | - | 26m | 57m | 58m | 1h03m | - | - | 1h02m | - | animal_samedistill_worker>scripts.evaluate_student_preference*4 |
| - | - | - | - | - | t1 | 934 | 3m | 3m | 4m | - | - | 2m | 2m | 3m | 3m | - | - | 3m | - | scripts.compute_eas |
| - | - | - | - | - | t3 | 768 | 3m | 3m | 3m | - | - | 3m | - | - | - | - | 3m | - | - | scripts.evaluate_pile_coherence |
| - | - | - | - | - | t1 | 763 | 49m | 56m | 1h18m | - | - | 1h12m | 54m | 1h05m | 1h13m | - | - | 55m | - | hf_worker distill |
| - | - | - | - | - | t3 | 603 | 49m | 52m | 1h02m | - | - | - | - | 52m | - | - | 52m | - | - | dog_top3hp_grid_worker |
| - | - | - | - | - | t1 | 495 | 9m | 14m | 19m | 4m | - | - | - | 4m | 14m | - | 14m | 3m | - | scripts.ckpt_pile_kl_trajectory |
| - | - | - | - | - | t1 | 483 | 2m | 2m | 3m | - | - | - | 2m | 2m | 3m | - | - | 2m | - | strip_earlystop_worker |
| - | - | - | - | - | t1 | 468 | 1h22m | 1h32m | 1h39m | - | - | - | - | - | 1h32m | - | 1h32m | - | - | strip_ckpt_replay_worker |
| - | - | - | - | - | t1 | 460 | 12m | 29m | 48m | 18s | - | - | 22m | 26m | 30m | - | 23m | 30m | - | scripts.sweeps.sweep_ckpt_distill numgen-cell |
| - | - | - | - | - | t3 | 382 | 51m | 1h38m | 1h59m | - | 1h38m | 1h39m | 1h19m | 1h40m | - | - | 1h38m | - | - | dog_top3hp_grid_worker*2 |
| - | - | - | - | - | t3 | 349 | 15s | 18s | 25s | - | 16s | 16s | 20s | - | - | - | 18s | - | - | scripts.direction_surgery |
| - | - | - | - | - | t1 | 255 | 17m | 29m | 49m | 1m | - | 25m | 29m | 32m | 31m | - | - | 30m | - | uv |
| - | - | - | - | - | t2 | 231 | 1m | 2m | 2m | 1s | - | 1m | 2m | 2m | 2m | - | - | 2m | - | scripts.run_eval_chain>scripts.evaluate_student_preference*56 |
| - | - | - | - | - | t1 | 222 | 19m | 28m | 34m | 16s | - | - | - | 2h06m | 28m | - | 28m | 24m | - | scripts.run_final_evals |
| - | - | - | - | - | t3 | 214 | 14m | 48m | 54m | - | 48m | 47m | 49m | 52m | - | - | 48m | - | - | run_logs.dog_crossdistill_studenthp_worker |
| - | - | - | - | - | t1 | 189 | 5m | 8m | 22m | 9s | 5m | 17m | 12m | 8m | 8m | - | 7m | 9m | - | scripts.sweeps.sweep_teacher_templates |
| - | - | - | - | - | t1 | 180 | 11s | 14s | 19s | 7s | - | - | - | 13s | 14s | - | - | 14s | - | scripts.evaluate_student_preference |
| - | - | - | - | - | t1 | 166 | 45m | 50m | 1h15m | - | - | - | 53m | 49m | 49m | - | 50m | - | - | clamp_worker |
| - | - | - | - | - | t1 | 149 | 26m | 29m | 34m | - | - | - | - | - | 29m | - | - | 29m | - | scripts.resampling_ladder run |
| - | - | - | - | - | t1 | 148 | 1m | 2m | 3m | - | - | 1m | 2m | 2m | 3m | - | - | 2m | - | p410m_worker seval |
| - | - | - | - | - | t1 | 148 | 24m | 28m | 33m | - | - | - | - | 32m | 27m | - | - | 27m | - | scripts.resampling_controls run |
| - | - | - | - | - | t1 | 139 | 9m | 14m | 17m | 1m | - | - | - | 14m | 14m | - | 14m | 10m | - | scripts.fidelity_ckpt_trajectory |
| - | - | - | - | - | t1 | 120 | 6m | 12m | 15m | 5s | - | 7m | 7m | 12m | 13m | - | 7m | 13m | - | scripts.sweeps.sweep_evaluate_teacher_preference |
| - | - | - | - | - | t3 | 105 | 8m | 9m | 9m | - | 4m | 9m | - | - | 10m | - | 9m | - | - | scripts.compute_cka |
| - | - | - | - | - | t1 | 105 | 18s | 24s | 31s | - | - | - | - | - | 24s | - | - | 24s | - | scripts.ckpt_teacher_pile_probe |
| - | - | - | - | - | t1 | 100 | 2h05m | 3h18m | 5h10m | 4s | - | - | - | 3h43m | 3h09m | - | - | 3h17m | - | scripts.sweeps.sweep_generate_numbers |
| - | - | - | - | - | t1 | 94 | 2m | 4m | 9m | - | 3m | 4m | 5m | 4m | 5m | - | 4m | 4m | - | frozen_a_ablation |
| - | - | - | - | - | t1 | 73 | 21m | 1h00m | 1h50m | 36s | - | - | 1h32m | 1h23m | 36m | - | - | 1h02m | - | scripts.generate_numbers |
| - | - | - | - | - | t1 | 65 | 23s | 27s | 39s | - | - | - | - | - | 27s | - | - | 27s | - | scripts.evaluate_checkpoint_teachers |
| - | - | - | - | - | t1 | 63 | 27m | 35m | 1h01m | 1m | - | - | 30m | 39m | 37m | - | - | 37m | - | uv\|model=olmo-1b |
| - | - | - | - | - | t1 | 61 | 27m | 36m | 3h07m | 30s | 36m | - | - | - | - | - | - | 36m | - | scripts.adapter_norms_slim |
| - | - | - | - | - | t1 | 60 | 54m | 2h29m | 3h26m | - | - | - | 2h43m | 1h51m | 1h42m | - | - | 2h29m | - | scripts.dose_rescaling_census evaluate |
| - | - | - | - | - | t1 | 60 | 15m | 21m | 45m | - | - | - | - | - | 20m | - | - | 21m | - | scripts.dose_rescaling_census evaluate\|model=pythia-160m |
| - | - | - | - | - | t1 | 60 | 11s | 14s | 17s | - | - | - | - | 12s | 15s | - | - | 14s | - | p410m_worker seval\|model=pythia-160m |
| - | - | - | - | - | t1 | 51 | 10m | 16m | 26m | 40s | - | - | 10m | 14m | 19m | - | - | 16m | - | preflight>scripts.sweeps.sweep_teacher_templates |
| - | - | - | - | - | t1 | 43 | 22s | 39s | 47s | - | - | - | - | - | 41s | - | - | 39s | - | eval_step |
| - | - | - | - | - | t1 | 40 | 25m | 39m | 1h45m | - | - | - | - | - | 27m | - | - | 39m | - | scripts.distill_student |
| - | - | - | - | - | t1 | 37 | 30m | 45m | 1h35m | 45s | - | - | - | - | 44m | - | - | 45m | - | scripts.position_surgery |
| - | - | - | - | - | t1 | 36 | 6m | 9m | 13m | 11s | - | - | - | - | 9m | - | 9m | - | - | scripts.pile_probe_final |
| - | - | - | - | - | t1 | 35 | 15m | 24m | 41m | - | - | - | - | - | 29m | - | - | 24m | - | scripts.dose_rescaling_census evaluate\|model=pythia-410m |
| - | - | - | - | - | t1 | 33 | 46s | 1m | 2m | - | - | 46s | - | 1m | - | - | - | 1m | - | controls_backfill_worker |
| - | - | - | - | - | t1 | 33 | 28m | 43m | 1h13m | 38s | - | - | - | 36m | 46m | - | - | 43m | - | scripts.donor_potency_sweep |
| - | - | - | - | - | t1 | 32 | 1m | 1m | 1m | 2s | - | - | - | 1m | 1m | - | 1m | - | - | scripts.fidelity_heldout_ce |
| - | - | - | - | - | t1 | 28 | 38m | 52m | 1h59m | 1m | - | - | - | 1h52m | 1h01m | - | - | 1h15m | - | scripts.teacher_pile_kl_trajectory |
| - | - | - | - | - | t1 | 25 | 27m | 34m | 1h25m | 32s | - | - | - | - | 35m | - | - | 34m | - | scripts.position_surgery\|model=pythia-160m |
| - | - | - | - | - | t1 | 24 | 32m | 49m | 1h14m | - | - | - | - | 48m | 50m | - | - | 49m | - | scripts.frozen_a_ablation |
| - | - | - | - | - | t1 | 22 | 28m | 43m | 1h14m | 34s | - | - | - | - | 43m | - | - | 43m | - | scripts.donor_potency_sweep\|model=pythia-160m |
| - | - | - | - | - | t1 | 21 | 3m | 11m | 13m | 39s | - | - | - | - | 11m | - | - | 11m | - | scripts.lambda_crossing |
| - | - | - | - | - | t3 | 21 | 1h53m | 2h26m | 2h31m | - | - | 1h54m | 2h29m | - | - | - | 2h26m | - | - | ctrl_krungs_worker |
| - | - | - | - | - | t3 | 20 | 8m | 8m | 14m | - | - | - | 8m | - | - | - | 8m | - | - | dog_numgen_worker |
| - | - | - | - | - | t1 | 20 | 1m | 2m | 2m | 11s | - | 1m | - | 1m | 2m | - | - | 2m | - | scripts.evaluate_student_preference*10>scripts.run_final_evals |
| - | - | - | - | - | t1 | 20 | 22s | 30s | 36s | 15s | - | - | - | - | 30s | - | - | 30s | - | scripts.evaluate_student_preference>scripts.run_final_evals |
| - | - | - | - | - | t1 | 19 | 3m | 10m | 14m | 33s | - | - | - | - | 10m | - | - | 10m | - | scripts.lambda_crossing\|model=pythia-160m |
| - | - | - | - | - | t1 | 18 | 14m | 20m | 43m | - | - | - | - | - | 18m | - | - | 20m | - | scripts.sweeps.sweep_steerability_screen |
| - | - | - | - | - | t3 | 17 | 2m | 2m | 7m | - | - | - | 2m | - | - | - | 2m | - | - | adapter_norms_probe activations |
| - | - | - | - | - | t1 | 16 | 27m | 1h07m | 3h50m | 1m | 1h06m | - | - | - | - | - | - | 1h06m | - | scripts.adapter_norms_slim\|model=olmo-1b |
| - | - | - | - | - | t1 | 16 | 1h47m | 2h52m | 4h21m | - | - | - | - | - | 2h46m | - | - | 2h52m | - | scripts.dose_rescaling_census evaluate\|model=qwen |
| - | - | - | - | - | t1 | 14 | 45m | 1h18m | 1h40m | 1m | - | - | - | - | 1h17m | - | - | 1h18m | - | scripts.position_surgery\|model=pythia-410m |
| - | - | - | - | - | t1 | 13 | 36s | 40s | 1m | - | - | 40s | 1m | - | - | - | 1m | - | - | frozen_a_ablation\|model=pythia-160m |
| - | - | - | - | - | t1 | 13 | 1h29m | 3h00m | 3h24m | 24s | 3h00m | - | - | - | - | - | - | 3h00m | - | scripts.adapter_norms_slim\|model=pythia-160m |
| - | - | - | - | - | t1 | 13 | 23m | 29m | 57m | 58s | 29m | - | - | - | - | - | - | 29m | - | scripts.adapter_norms_slim\|model=pythia-410m |
| - | - | - | - | - | t1 | 12 | 10m | 10m | 14m | 6s | - | 7m | 9m | 9m | 10m | - | - | 7m | - | scripts.evaluate_student_preference*56 |
| - | - | - | - | - | t3 | 12 | 16m | 59m | 1h01m | - | - | - | 59m | - | - | - | 59m | - | - | adapter_norms_probe main |
| - | - | - | - | - | t1 | 11 | 31m | 43m | 1h01m | 47s | - | - | - | 36m | - | - | - | 43m | - | scripts.donor_potency_sweep\|model=pythia-410m |
| - | - | - | - | - | t1 | 10 | 1m | 1m | 1m | 1m | - | - | 33s | - | 1m | - | - | 51s | - | scripts.sweeps.sweep_teacher_templates>scripts.sweeps.sweep_evaluate_teacher_preference>validate_campaign>scripts.run_final_evals>validate_campaign |
| - | - | - | - | - | t1 | 10 | 39m | 40m | 48m | - | - | - | - | - | 40m | - | - | 40m | - | dog_1m_numgen_worker |
| - | - | - | - | - | t2 | 10 | 31m | 1h08m | 1h41m | 2m | 1h08m | - | - | - | - | - | - | 1h08m | - | scripts.adapter_norms_slim\|model=qwen |
| - | - | - | - | - | t1 | 10 | 44s | 52s | 1m | 2s | 47s | - | - | - | - | - | - | 48s | - | scripts.run_final_evals>validate_campaign |
| - | - | - | - | - | t2 | 10 | 12m | 15m | 23m | 5s | - | - | - | - | 15m | - | - | 15m | - | preflight>scripts.sweeps.sweep_evaluate_teacher_preference |
| - | - | - | - | - | t1 | 9 | 51s | 56s | 1m | 2s | - | - | - | - | 56s | - | - | 57s | - | validate_campaign>scripts.run_final_evals>validate_campaign |
| - | - | - | - | - | t2 | 8 | 20m | 26m | 1h08m | - | - | - | - | 20m | - | - | - | 26m | - | p410m_worker numgen |
| - | - | - | - | - | t1 | 8 | 40m | 44m | 48m | - | - | - | - | - | 44m | - | 44m | - | - | strip_ckpt_worker |
| - | - | - | - | - | t3 | 8 | 19s | 20s | 23s | - | - | - | 20s | - | - | - | 20s | - | - | scripts.direction_surgery\|model=pythia-160m |
| - | - | - | - | - | t2 | 8 | 49s | 54s | 2m | - | - | - | - | - | 54s | - | - | 54s | - | p410m_worker teacher |
