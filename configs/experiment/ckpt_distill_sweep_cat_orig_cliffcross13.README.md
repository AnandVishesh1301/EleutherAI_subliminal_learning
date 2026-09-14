# CAT original 13-checkpoint campaign

Primary `ckpt-cat-orig13-20260914`: 13 checkpoints, five teacher recipes, seed-1 teachers/students, and all 169 ordered pairs: 65 teacher + 65 number-generation + 845 distillation jobs (975 total).

`ckpt-cat-orig13-20260914-diagseeds23` adds diagonal-only student seeds 2 and 3: 65 teacher + 65 number-generation + 130 distillation jobs (260 total). Its `depends_on_prefix` and submitted metadata make matching duplicate teacher and number-generation queues wait for the primary campaign before writing shared paths.

All queues live in `run_logs/queue_angela`, priority 47, strict, and depend on both `direction-olmo-surgery-20260912` and `direction-olmo-injections-20260912`. Total submitted workload: 1,235 jobs.


## Teacher and base-prior evaluation

`scripts/evaluate_checkpoint_teachers.py` supplies the teacher/prior evaluations
that train-cell does not produce. Queue `ckpt-cat-orig13-20260914-teacher-eval`
contains 13 jobs, each evaluating all five exact teacher recipes serially on
`test` and `off_topic`, including untuned base priors. It strictly depends on
both OLMo queues and all 13 core teacher queues, priority 47, node
soar-subliminal-1, cap 2, memory gate 6000 MiB. Missing training/evaluation
outputs fail the job. Job manifest and verified queue counts live under
`run_logs/ckpt-cat-orig13-20260914/`.

Including these evaluations: **1,248 queued jobs**, of which **975 are student
train/eval jobs** (845 seed-1 grid cells plus 130 diagonal seeds-2/3 cells).
The 260 teacher/generation job records include resume checks in the diagonal
campaign; shared artifacts are serialized and valid results are reused.
