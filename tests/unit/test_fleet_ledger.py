from __future__ import annotations

import pytest

from subliminal_learning.fleet import ledger as fl
from subliminal_learning.fleet.nodes import resolve_node

NOW = "2026-10-16T00:00:00Z"
CARD = fl.Card(queue_roots=("/mnt/ssd-1/x/queue",), queue_prefix="a-")


def _ledger(*campaigns: str) -> fl.Ledger:
    led = fl.empty_ledger()
    for name in campaigns:
        led = fl.register(led, name, CARD, chat="c", now=NOW)
    return led


def _lease(led: fl.Ledger, node: str) -> fl.NodeLease:
    return led.nodes[resolve_node(node).hostname]


@pytest.mark.parametrize(
    ("name", "host"),
    [
        ("soar", "soar-subliminal-0"),
        ("a40-cluster-4", "soar-subliminal-3"),
        ("cluster-5", "secret-ord-0"),
        ("goncalo-temp-1", "goncalo-temp-1"),
        ("soar-subliminal-2", "soar-subliminal-2"),
    ],
)
def test_resolve_node_aliases_map_to_hostname(name: str, host: str) -> None:
    assert resolve_node(name).hostname == host


def test_resolve_node_unknown_raises() -> None:
    with pytest.raises(KeyError):
        resolve_node("cluster-9")


def test_claim_free_then_refuses_other_campaign() -> None:
    led = fl.claim(_ledger("a", "b"), "a", ["soar", "cluster-4"], chat="c", now=NOW)
    assert _lease(led, "soar").state == fl.RUNNING
    assert _lease(led, "soar").holder == "a"
    with pytest.raises(fl.LeaseError):
        fl.claim(led, "b", ["soar"], chat="c", now=NOW)


def test_claim_requires_registration() -> None:
    with pytest.raises(fl.LeaseError):
        fl.claim(_ledger(), "ghost", ["soar"], chat="c", now=NOW)


def test_yield_lend_release_returns_node_to_lender() -> None:
    led = fl.claim(_ledger("a", "b"), "a", ["soar", "cluster-3"], chat="c", now=NOW)
    led = fl.start_yield(led, "a", "b", ["soar"], mode="drain", chat="c", now=NOW)
    soar = _lease(led, "soar")
    assert (soar.state, soar.holder, soar.target) == (fl.DRAINING, "a", "b")
    led = fl.complete_yield(led, "b", ["soar"], chat="c", now=NOW)
    soar = _lease(led, "soar")
    assert (soar.state, soar.holder, soar.lenders) == (fl.RUNNING, "b", ("a",))
    assert _lease(led, "cluster-3").holder == "a"
    led = fl.release(led, "b", ["soar"], chat="c", now=NOW)
    soar = _lease(led, "soar")
    assert (soar.state, soar.holder, soar.lenders) == (fl.RETURNING, "a", ())
    led = fl.resume(led, "a", ["soar"], chat="c", now=NOW)
    assert _lease(led, "soar").state == fl.RUNNING


def test_complete_yield_refuses_wrong_target() -> None:
    led = fl.claim(_ledger("a", "b", "c"), "a", ["soar"], chat="c", now=NOW)
    led = fl.start_yield(led, "a", "b", ["soar"], mode="now", chat="c", now=NOW)
    with pytest.raises(fl.LeaseError):
        fl.complete_yield(led, "c", ["soar"], chat="c", now=NOW)


def test_nested_lend_unwinds_newest_live_lender_first() -> None:
    led = fl.claim(_ledger("a", "b", "c"), "a", ["soar"], chat="c", now=NOW)
    led = fl.start_yield(led, "a", "b", ["soar"], mode="drain", chat="c", now=NOW)
    led = fl.complete_yield(led, "b", ["soar"], chat="c", now=NOW)
    led = fl.start_yield(led, "b", "c", ["soar"], mode="drain", chat="c", now=NOW)
    led = fl.complete_yield(led, "c", ["soar"], chat="c", now=NOW)
    assert _lease(led, "soar").lenders == ("a", "b")
    led = fl.finish(led, "b", chat="c", now=NOW)
    assert _lease(led, "soar").lenders == ("a",)
    led = fl.release(led, "c", ["soar"], chat="c", now=NOW)
    assert (_lease(led, "soar").state, _lease(led, "soar").holder) == (
        fl.RETURNING,
        "a",
    )


def test_release_without_live_lender_frees_node() -> None:
    led = fl.claim(_ledger("a", "b"), "a", ["soar"], chat="c", now=NOW)
    led = fl.start_yield(led, "a", "b", ["soar"], mode="drain", chat="c", now=NOW)
    led = fl.complete_yield(led, "b", ["soar"], chat="c", now=NOW)
    led = fl.set_status(led, "a", fl.CAMPAIGN_DONE, now=NOW)
    led = fl.release(led, "b", ["soar"], chat="c", now=NOW)
    assert _lease(led, "soar").state == fl.FREE


def test_claim_paused_node_lends_it() -> None:
    led = fl.claim(_ledger("a", "b"), "a", ["cluster-4"], chat="c", now=NOW)
    led = fl.pause(led, "a", ["cluster-4"], chat="c", now=NOW)
    led = fl.claim(led, "b", ["cluster-4"], chat="c", now=NOW)
    lease = _lease(led, "cluster-4")
    assert (lease.state, lease.holder, lease.lenders) == (fl.RUNNING, "b", ("a",))
    with pytest.raises(fl.LeaseError):
        fl.claim(
            fl.pause(led, "b", ["cluster-4"], chat="c", now=NOW),
            "b",
            ["cluster-4"],
            chat="c",
            now=NOW,
        )


def test_finish_target_returns_draining_node_to_holder() -> None:
    led = fl.claim(_ledger("a", "b"), "a", ["soar"], chat="c", now=NOW)
    led = fl.start_yield(led, "a", "b", ["soar"], mode="drain", chat="c", now=NOW)
    led = fl.finish(led, "b", chat="c", now=NOW)
    soar = _lease(led, "soar")
    assert (soar.state, soar.holder, soar.target) == (fl.RETURNING, "a", None)


def test_ledger_dict_roundtrip() -> None:
    led = fl.claim(_ledger("a", "b"), "a", ["soar"], chat="c", now=NOW)
    led = fl.start_yield(led, "a", "b", ["soar"], mode="drain", chat="c", now=NOW)
    led = fl.complete_yield(led, "b", ["soar"], chat="c", now=NOW)
    back = fl.ledger_from_dict(fl.ledger_to_dict(led))
    assert back.nodes == led.nodes
    assert back.campaigns["a"].card == CARD


def test_card_rejects_unknown_field_and_relative_root() -> None:
    with pytest.raises(fl.LeaseError):
        fl.card_from_dict({"queue_roots": ["/q"], "bogus": 1})
    with pytest.raises(fl.LeaseError):
        fl.card_from_dict({"queue_roots": ["rel/q"]})


def test_node_commands_remote_pause_and_preempt_guard() -> None:
    lines = fl.node_commands(
        CARD, "soar-subliminal-3", "pause", cwd="/c", local_host="soar-subliminal-0"
    )
    assert lines == [
        "ssh -o BatchMode=yes a40-cluster-4 'cd /c && "
        "uv run python -m scripts.jobq stop --root /mnt/ssd-1/x/queue'"
    ]
    with pytest.raises(fl.LeaseError):
        fl.node_commands(
            CARD,
            "soar-subliminal-3",
            "preempt",
            cwd="/c",
            local_host="soar-subliminal-0",
        )
    now_card = fl.Card(queue_roots=("/q",), queue_prefix="a-", preempt_ok=True)
    assert fl.node_commands(
        now_card,
        "soar-subliminal-0",
        "preempt",
        cwd="/p",
        local_host="soar-subliminal-0",
    ) == ["cd /p && uv run python -m scripts.jobq stop --now --root /q --prefix a-"]


def test_roots_shared_with_lists_other_live_campaigns() -> None:
    led = _ledger("a", "b")
    assert fl.roots_shared_with(led, "a") == {"/mnt/ssd-1/x/queue": ("b",)}
    led = fl.set_status(led, "b", fl.CAMPAIGN_DONE, now=NOW)
    assert fl.roots_shared_with(led, "a") == {}


def test_card_prefix_must_end_in_delimiter() -> None:
    with pytest.raises(fl.LeaseError):
        fl.card_from_dict({"queue_roots": ["/q"], "queue_prefix": "ws160m"})
    assert fl.card_from_dict({"queue_roots": ["/q"], "queue_prefix": "ws160m-"})


def test_start_yield_needs_registered_target() -> None:
    led = fl.claim(_ledger("a"), "a", ["soar"], chat="c", now=NOW)
    with pytest.raises(fl.LeaseError):
        fl.start_yield(led, "a", "typo", ["soar"], mode="drain", chat="c", now=NOW)


def test_finish_source_keeps_pending_handoff_to_live_target() -> None:
    led = fl.claim(_ledger("a", "b"), "a", ["soar"], chat="c", now=NOW)
    led = fl.start_yield(led, "a", "b", ["soar"], mode="drain", chat="c", now=NOW)
    led = fl.finish(led, "a", chat="c", now=NOW)
    assert (_lease(led, "soar").state, _lease(led, "soar").target) == (fl.DRAINING, "b")
    led = fl.complete_yield(led, "b", ["soar"], chat="c", now=NOW)
    soar = _lease(led, "soar")
    assert (soar.state, soar.holder, soar.lenders) == (fl.RUNNING, "b", ())


def test_node_commands_one_line_per_root() -> None:
    card = fl.Card(queue_roots=("/q1", "/q2"))
    lines = fl.node_commands(
        card, "soar-subliminal-0", "resume", cwd="/c", local_host="soar-subliminal-0"
    )
    assert lines == [
        "cd /c && uv run python -m scripts.jobq stop --clear --root /q1",
        "cd /c && uv run python -m scripts.jobq stop --clear --root /q2",
    ]
