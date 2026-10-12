"""Judge side of the S2 core campaign (plan v2 §12): tasks, scoring, workers, lane.

Runs inside the judge venvs (`tools/judge_v2` = stack N, vLLM 0.30.0;
`tools/rollout_judge` = stack F, vLLM 0.8.5) with ``PYTHONPATH=tools``. It never
imports `subliminal_learning`: everything a worker needs is in the work-item spec.
"""
