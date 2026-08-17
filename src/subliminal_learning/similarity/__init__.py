"""Representation-similarity measures computed from streaming second moments.

Companion to `subliminal_learning.cka`, NOT a replacement: minibatch CKA keeps its own
n x n Gram path (`cka.minibatch`). This package covers the measures that cannot be
written as a per-batch average — Procrustes distance and the CCA family (PWCCA) — by
accumulating the sufficient statistics instead of the per-batch answers.
"""
