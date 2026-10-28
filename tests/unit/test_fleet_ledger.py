from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import scripts.fleet as fleet_cli
from subliminal_learning.fleet import ledger as fl
from subliminal_learning.fleet.ledger_io import read_ledger, update_ledger
from subliminal_learning.fleet.nodes import resolve_node
from subliminal_learning.jobqueue import slurm

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


SLURM = fl.Card(queue_roots=("/q",), queue_prefix="s-", executor=fl.EXECUTOR_SLURM)


def test_card_without_executor_loads_as_ssh_and_ssh_is_not_written() -> None:
    led = _ledger("a")
    data = fl.ledger_to_dict(led)
    assert "executor" not in data["campaigns"]["a"]["card"]
    assert fl.ledger_from_dict(data).campaigns["a"].card.executor == fl.EXECUTOR_SSH
    led = fl.register(led, "s", SLURM, chat="c", now=NOW)
    data = fl.ledger_to_dict(led)
    assert data["campaigns"]["s"]["card"]["executor"] == fl.EXECUTOR_SLURM
    assert fl.ledger_from_dict(data).campaigns["s"].card == SLURM


def test_slurm_card_keeps_login_driver_node_and_rejects_unknown_executor() -> None:
    data = {
        "queue_roots": ["/q", "/r"],
        "executor": "slurm",
        "driver_node": "gpu-login-0",
    }
    card = fl.card_from_dict(data)
    assert card.driver_node == "gpu-login-0" and card.queue_roots == ("/q", "/r")
    with pytest.raises(fl.LeaseError):
        fl.card_from_dict({"queue_roots": ["/q"], "executor": "k8s"})
    with pytest.raises(KeyError):
        fl.card_from_dict({"queue_roots": ["/q"], "driver_node": "gpu-login-0"})


def test_slurm_campaign_refuses_leases_and_names_its_commands() -> None:
    led = fl.register(_ledger("a"), "s", SLURM, chat="c", now=NOW)
    led = fl.claim(led, "a", ["soar"], chat="c", now=NOW)
    attempts = [
        lambda: fl.claim(led, "s", ["cluster-3"], chat="c", now=NOW),
        lambda: fl.start_yield(
            led, "a", "s", ["soar"], mode="drain", chat="c", now=NOW
        ),
        lambda: fl.pause(led, "s", ["soar"], chat="c", now=NOW),
        lambda: fl.resume(led, "s", ["soar"], chat="c", now=NOW),
        lambda: fl.release(led, "s", ["soar"], chat="c", now=NOW),
    ]
    for attempt in attempts:
        with pytest.raises(fl.LeaseError, match="jobq slurm-pause --root /q --name s"):
            attempt()
    done = fl.finish(led, "s", chat="c", now=NOW)
    assert done.campaigns["s"].status == fl.CAMPAIGN_DONE
    assert done.nodes == led.nodes


def test_slurm_command_defaults_card_override_and_preempt_guard() -> None:
    assert fl.slurm_command(SLURM, "s", "resume") == (
        "uv run python -m scripts.jobq slurm-resume --root /q --name s"
    )
    with pytest.raises(fl.LeaseError):
        fl.slurm_command(SLURM, "s", "preempt")
    card = fl.Card(
        queue_roots=("/q",), executor="slurm", pause_cmd="P", preempt_ok=True
    )
    assert fl.slurm_command(card, "s", "pause") == "P"
    assert fl.slurm_command(card, "s", "preempt") == "P --now"
    two = fl.Card(queue_roots=("/q", "/r"), executor="slurm")
    assert fl.slurm_command(two, "s", "pause") == (
        "uv run python -m scripts.jobq slurm-pause --root /q --name s && "
        "uv run python -m scripts.jobq slurm-pause --root /r --name s"
    )


def _brief(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cards: dict[str, fl.Card]
) -> str:
    fleet = tmp_path / "fleet"
    for name, card in cards.items():
        update_ledger(
            fleet,
            "register",
            lambda led, n=name, c=card: fl.register(led, n, c, chat="c", now=NOW),
            chat="c",
            args={},
        )
    monkeypatch.setattr(fleet_cli, "_root", lambda: fleet)
    res = CliRunner().invoke(fleet_cli.app, ["status", "--brief"])
    assert res.exit_code == 0, res.output
    return res.output


def test_brief_status_lists_slurm_campaign_without_fleet_skill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "q"
    (root / "slurm" / "s" / "pools").mkdir(parents=True)
    (root / "slurm" / "s" / "PAUSED").touch()
    for ident, jid in (("slurm-7.r0", 7), ("slurm-8.r0", 8)):
        rec = root / "slurm" / "s" / "pools" / f"{ident}.json"
        rec.write_text(json.dumps({"slurm_job_id": jid}))
    jobs = slurm.parse_squeue("7|s-pool|RUNNING|gres/gpu:4\n8|s-pool|COMPLETED|")
    monkeypatch.setattr(slurm, "query_squeue", lambda: jobs)
    card = fl.Card(queue_roots=(str(root),), queue_prefix="s-", executor="slurm")
    out = _brief(tmp_path, monkeypatch, {"s": card})
    assert "  s running executor=slurm pools=1 PAUSED chat=c plan=-\n" in out
    assert "go through /fleet" not in out and "pause_cmd / resume_cmd" in out
    monkeypatch.setattr(slurm, "query_squeue", lambda: None)
    out = _brief(tmp_path, monkeypatch, {"a": CARD})
    assert "  s running executor=slurm pools=? PAUSED chat=c plan=-\n" in out
    assert "  a running nodes=- chat=c plan=-\n" in out and "go through /fleet" in out


def test_brief_status_without_slurm_cards_is_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(slurm, "query_squeue", None)
    out = _brief(tmp_path, monkeypatch, {"a": CARD})
    led = read_ledger(tmp_path / "fleet")
    assert out.split("\n", 1)[1] == (
        fl.render(led) + "\nNode handoffs between campaigns and chats go through "
        "/fleet (.claude/skills/fleet/SKILL.md); never pause, kill, resume or launch "
        "on a node another campaign holds without it.\n"
    )
