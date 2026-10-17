"""The five A40 nodes: hostname (jobq key), ssh alias, clone (docs/CLUSTER_NODES.md §1)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Node:
    """One fleet node.

    Attributes:
        hostname: ``hostname`` output; the key of jobq meta, sentinels and leases.
        short: Name people use (``soar``, ``cluster-3`` ...).
        alias: ssh alias reachable from soar.
        clone: Main checkout on the node.
        other_names: Further names the node answers to.
        has_ssd3: Mounts ssd-3 / cold-1 (cluster-5 does not).
        shared: Shared with other tenants (``/occupy-gpus`` before use).
    """

    hostname: str
    short: str
    alias: str
    clone: str
    other_names: tuple[str, ...] = ()
    has_ssd3: bool = True
    shared: bool = False


_VISHESH_CLONE = "/home/vishesh/eleutherAI/EleutherAI_subliminal_learning"

NODES: tuple[Node, ...] = (
    Node(
        "soar-subliminal-0",
        "soar",
        "a40-cluster-1",
        _VISHESH_CLONE,
        other_names=("cluster-1",),
    ),
    Node(
        "goncalo-temp-1",
        "goncalo",
        "goncalo-temp-1",
        "/home/angela/eleutherAI/EleutherAI_subliminal_learning",
        other_names=("cluster-2", "a40-cluster-2"),
        shared=True,
    ),
    Node("soar-subliminal-2", "cluster-3", "a40-cluster-3", _VISHESH_CLONE),
    Node("soar-subliminal-3", "cluster-4", "a40-cluster-4", _VISHESH_CLONE),
    Node("secret-ord-0", "cluster-5", "a40-cluster-5", _VISHESH_CLONE, has_ssd3=False),
)


def resolve_node(name: str) -> Node:
    """Return the node a hostname, short name or ssh alias refers to.

    Raises:
        KeyError: ``name`` is not a fleet node.
    """
    key = name.strip().lower()
    for node in NODES:
        if key in {node.hostname, node.short, node.alias, *node.other_names}:
            return node
    known = ", ".join(n.short for n in NODES)
    raise KeyError(f"unknown node {name!r} (known: {known})")


def hostnames() -> tuple[str, ...]:
    """Every fleet hostname, in table order."""
    return tuple(n.hostname for n in NODES)
