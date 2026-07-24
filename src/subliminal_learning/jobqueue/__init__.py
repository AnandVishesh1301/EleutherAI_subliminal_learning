"""File-based GPU job queue + per-node worker pool.

A repo-proper replacement for the ad-hoc per-sweep shell launchers. State lives on a
shared root (default ``run_logs/queue``, but always a parameter) so two nodes can drain
the same queues cooperatively. The claim primitive is ``os.mkdir`` (atomic on CephFS);
every other write goes through ``utils/io`` atomic helpers. Hostname conventions are
load-bearing: a worker only ever acts on its OWN node's pids/claims (see ``store``).

Layers, mirroring the rest of the repo (domain-ish -> IO -> CLI):

- ``model``  — the pure ``Job`` dataclass, key sanitization, jobs-file expansion.
- ``store``  — on-disk queue state (meta, jobs, claims, results) + derived job states.
- ``gpu``    — per-node GPU slot locks + memory gating (all ``nvidia-smi`` behind one fn).
- ``worker`` — the threaded worker pool that claims, runs, and records jobs.

The thin Typer CLI is ``scripts/jobq.py``.
"""

from __future__ import annotations
