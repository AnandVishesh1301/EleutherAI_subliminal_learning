"""Main-program pipelines.

``pile_coherence_pipeline`` and ``adapter_decomp_pipeline`` moved to ``forgetting/pipelines/``;
``__path__`` also searches that directory so ``pipelines.<name>`` keeps resolving for queue
job records and run_logs callers. New code should use ``forgetting.pipelines.<name>``.
"""

from pathlib import Path as _Path

__path__.append(str(_Path(__file__).resolve().parent.parent / "forgetting" / "pipelines"))
