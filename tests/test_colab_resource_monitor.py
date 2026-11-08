from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from autoresearch.services import colab_resource_monitor as resource


PROJECT = "5801a3a4eec4"
NOW = datetime(2026, 10, 9, 2, 0, tzinfo=timezone.utc)


def job(**overrides):
    value = {
        "project_id": PROJECT, "prepared": True, "autorun_enabled": True,
        "status": "waiting_for_gpu", "worker_running": False,
        "retry_min_minutes": 5, "retry_max_minutes": 10,
        "next_attempt_at": resource.timestamp(NOW + timedelta(minutes=7)),
        "last_observation": {
            "status": "unavailable", "observed_at": resource.timestamp(NOW - timedelta(minutes=2)),
            "details": {"reason": "private-secret", "browser_url": "https://private.example/credential"},
        },
    }
    value.update(overrides)
    return value


def available_job(**overrides):
    return job(
        status="running", worker_running=True, runtime_id="current",
        last_observation={"status": "running", "observed_at": resource.timestamp(NOW),
                          "details": {"runtime_id": "current"}},
        **overrides,
    )


def test_waiting_snapshot_never_turns_heartbeat_into_gpu_check():
    result = resource.build_resource_snapshot(PROJECT, job(), now=NOW)
    assert result["availability"] == "unknown"
    assert result["last_result"] == "unavailable"
    assert result["agent_state"] == "watching"
    assert result["last_checked_at"] == resource.timestamp(NOW - timedelta(minutes=2))
    assert result["heartbeat_at"] == resource.timestamp(NOW)
    assert result["next_check_at"] == resource.timestamp(NOW + timedelta(minutes=7))
    assert result["last_check_fresh"] is True
    assert "private" not in json.dumps(result)


def test_available_requires_active_worker_matching_runtime_and_fresh_observation():
    result = resource.build_resource_snapshot(PROJECT, available_job(), now=NOW)
    assert result["availability"] == "available" and result["last_result"] == "available"
    inactive = available_job()
    inactive["worker_running"] = False
    assert resource.build_resource_snapshot(PROJECT, inactive, now=NOW)["availability"] == "unknown"
    stale = available_job()
    stale["last_observation"]["observed_at"] = resource.timestamp(NOW - timedelta(minutes=11))
    stale_result = resource.build_resource_snapshot(PROJECT, stale, now=NOW)
    assert stale_result["availability"] == "unknown" and stale_result["last_check_fresh"] is False
    mismatched = available_job()
    mismatched["last_observation"]["details"]["runtime_id"] = "previous"
    assert resource.build_resource_snapshot(PROJECT, mismatched, now=NOW)["availability"] == "unknown"


def test_missing_runtime_id_cannot_establish_current_allocation():
    value = available_job()
    value.pop("runtime_id")
    value["last_observation"] = {"status": "available", "observed_at": resource.timestamp(NOW)}
    value["status"] = "gpu_available"
    assert resource.build_resource_snapshot(PROJECT, value, now=NOW)["availability"] == "unknown"


def test_lease_and_attempt_time_never_count_as_provider_check():
    value = job(status="attempting", last_observation=None, lease_until=resource.timestamp(NOW), last_worker_launch_at=resource.timestamp(NOW))
    result = resource.build_resource_snapshot(PROJECT, value, now=NOW)
    assert result["agent_state"] == "checking"
    assert result["availability"] == "unknown" and result["last_checked_at"] is None


def test_recent_negative_is_visible_then_becomes_historical():
    value = job()
    value["last_observation"]["observed_at"] = resource.timestamp(NOW)
    recent = resource.build_resource_snapshot(PROJECT, value, now=NOW + timedelta(seconds=45))
    historical = resource.build_resource_snapshot(PROJECT, value, now=NOW + timedelta(seconds=46))
    assert recent["availability"] == "unavailable"
    assert historical["availability"] == "unknown" and historical["last_result"] == "unavailable"


def test_quota_is_distinct_from_inventory_and_preserves_cooldown():
    value = job()
    value["last_observation"]["status"] = "quota"
    value["next_attempt_at"] = resource.timestamp(NOW + timedelta(hours=12))
    result = resource.build_resource_snapshot(PROJECT, value, now=NOW)
    assert result["agent_state"] == "quota_wait"
    assert result["last_result"] == "quota" and result["availability"] == "unknown"
    assert result["next_check_at"] == value["next_attempt_at"]


@pytest.mark.parametrize("status", ["completed", "failed", "cancelled", "blocked"])
def test_terminal_jobs_stop_the_monitor(status):
    result = resource.build_resource_snapshot(PROJECT, job(status=status), now=NOW)
    assert result["agent_state"] == "stopped" and result["agent_running"] is False


@pytest.mark.parametrize("changes", [{"prepared": False}, {"autorun_enabled": False}])
def test_unprepared_or_unauthorized_job_never_stays_active(changes):
    assert resource.build_resource_snapshot(PROJECT, job(**changes), now=NOW)["agent_state"] == "stopped"


def test_auth_pause_is_visible_and_does_not_check_gpu():
    result = resource.build_resource_snapshot(PROJECT, job(status="auth_required"), now=NOW)
    assert result["agent_state"] == "auth_required" and result["availability"] == "unknown"


def test_api_offline_does_not_reuse_previous_available_as_current():
    previous = resource.build_resource_snapshot(PROJECT, available_job(), now=NOW)
    result = resource.build_resource_snapshot(PROJECT, None, previous=previous, now=NOW + timedelta(seconds=10))
    assert result["agent_state"] == "offline" and result["availability"] == "unknown"
    assert result["last_result"] == "available" and result["last_checked_at"] == resource.timestamp(NOW)
    assert result["worker_running"] is False


def test_wrong_project_or_malformed_observation_is_unknown():
    assert resource.build_resource_snapshot(PROJECT, job(project_id="other"), now=NOW)["agent_state"] == "offline"
    result = resource.build_resource_snapshot(PROJECT, job(status=["running"], last_observation={"status": ["available"], "observed_at": "secret"}), now=NOW)
    assert result["availability"] == "unknown" and result["last_checked_at"] is None
    assert result["job_status"] == "unknown" and result["last_result"] == "unknown"


class FakeClient:
    def __init__(self, values):
        self.values = list(values)
        self.calls = 0
        self.closed = False

    async def status(self):
        self.calls += 1
        value = self.values.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    async def close(self):
        self.closed = True


def test_monitor_keeps_readonly_watch_during_cooldown_and_due(tmp_path):
    client = FakeClient([job(), job(can_attempt=True, next_attempt_at=resource.timestamp(NOW))])
    public = tmp_path / "public" / f"{PROJECT}.json"
    monitor = resource.ColabResourceMonitor(tmp_path, PROJECT, api_url="http://127.0.0.1:8767",
                                           public_path=public, client_factory=lambda *_: client, clock=lambda: NOW)
    first = asyncio.run(monitor.tick())
    second = asyncio.run(monitor.tick())
    assert first["agent_state"] == second["agent_state"] == "watching"
    assert client.calls == 2
    assert json.loads(public.read_text(encoding="utf-8")) == second
    assert json.loads((tmp_path / "data" / "colab-resource-monitor" / f"{PROJECT}.status.json").read_text(encoding="utf-8")) == second
    assert list(tmp_path.rglob("job.json")) == []
    # FakeClient deliberately has no launch, claim, or POST method.


def test_monitor_recovers_after_network_failure_without_fabricating_check(tmp_path):
    client = FakeClient([httpx.ConnectError("credential-in-private-url"), job()])
    public = tmp_path / "public.json"
    monitor = resource.ColabResourceMonitor(tmp_path, PROJECT, api_url="http://127.0.0.1:8767",
                                           public_path=public, client_factory=lambda *_: client, clock=lambda: NOW)
    offline = asyncio.run(monitor.tick())
    assert offline["agent_state"] == "offline" and offline["last_checked_at"] is None
    recovered = asyncio.run(monitor.tick())
    assert recovered["agent_state"] == "watching"
    assert "credential" not in public.read_text(encoding="utf-8")


def test_control_handoff_adopts_new_loopback_api_and_closes_old_client(tmp_path):
    clients = []
    addresses = []
    def factory(url, _project):
        addresses.append(url)
        client = FakeClient([job()])
        clients.append(client)
        return client
    monitor = resource.ColabResourceMonitor(tmp_path, PROJECT, api_url="http://127.0.0.1:8767",
                                           public_path=tmp_path / "public.json", client_factory=factory, clock=lambda: NOW)
    asyncio.run(monitor.tick())
    control_path = tmp_path / "data" / "colab-resource-monitor" / f"{PROJECT}.control.json"
    resource.atomic_json(control_path, {"project_id": PROJECT, "api_url": "http://127.0.0.1:8768"})
    asyncio.run(monitor.tick())
    assert addresses == ["http://127.0.0.1:8767", "http://127.0.0.1:8768"]
    assert clients[0].closed is True


def test_control_handoff_refuses_external_endpoint(tmp_path):
    client = FakeClient([job()])
    monitor = resource.ColabResourceMonitor(tmp_path, PROJECT, api_url="http://127.0.0.1:8767",
                                           public_path=tmp_path / "public.json", client_factory=lambda *_: client, clock=lambda: NOW)
    control = tmp_path / "data" / "colab-resource-monitor" / f"{PROJECT}.control.json"
    resource.atomic_json(control, {"project_id": PROJECT, "api_url": "https://outside.example/secret"})
    asyncio.run(monitor.tick())
    assert monitor.api_url == "http://127.0.0.1:8767"


def test_run_stops_on_cancel_and_releases_runtime_lock(tmp_path):
    client = FakeClient([job(status="cancelled")])
    monitor = resource.ColabResourceMonitor(tmp_path, PROJECT, api_url="http://127.0.0.1:8767",
                                           public_path=tmp_path / "public.json", client_factory=lambda *_: client, clock=lambda: NOW)
    asyncio.run(monitor.run())
    assert client.closed is True and monitor.previous["agent_state"] == "stopped"
    with resource.WorkerLock(tmp_path / "data" / "colab-resource-monitor" / f"{PROJECT}.lock"):
        pass


def test_launch_singleton_updates_control_without_spawning_twice(tmp_path, monkeypatch):
    calls = []
    class Process:
        pid = 123456
    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return Process()
    monkeypatch.setattr(resource.subprocess, "Popen", popen)
    monkeypatch.setattr(resource, "_process_identity", lambda pid: "same-process")
    first = resource.launch_resource_monitor(tmp_path, PROJECT, api_url="http://127.0.0.1:8767")
    second = resource.launch_resource_monitor(tmp_path, PROJECT, api_url="http://127.0.0.1:8768")
    assert first == second == Process.pid and len(calls) == 1
    control = json.loads((tmp_path / "data" / "colab-resource-monitor" / f"{PROJECT}.control.json").read_text(encoding="utf-8"))
    assert control["api_url"] == "http://127.0.0.1:8768"
    assert calls[0][1]["env"]["PYTHONIOENCODING"] == "utf-8"


def test_pid_reuse_is_not_treated_as_live_monitor(tmp_path, monkeypatch):
    resource.atomic_json(tmp_path / "data" / "colab-resource-monitor" / f"{PROJECT}.worker.json",
                         {"project_id": PROJECT, "pid": 123, "process_identity": "old"})
    monkeypatch.setattr(resource, "_process_identity", lambda pid: "new")
    assert resource.resource_monitor_is_running(tmp_path, PROJECT) is None


def test_resource_status_clears_live_availability_when_monitor_exits(tmp_path, monkeypatch):
    resource.atomic_json(tmp_path / "data" / "colab-resource-monitor" / f"{PROJECT}.status.json",
                         resource.build_resource_snapshot(PROJECT, available_job(), now=NOW))
    monkeypatch.setattr(resource, "resource_monitor_is_running", lambda *_: None)
    monkeypatch.setattr(resource, "utc_now", lambda: NOW)
    result = resource.resource_monitor_status(tmp_path, PROJECT)
    assert result["availability"] == "unknown" and result["agent_state"] == "offline"
    assert result["agent_running"] is False and result["agent_heartbeat_fresh"] is False


def test_resource_status_whitelists_snapshot_when_alive(tmp_path, monkeypatch):
    value = resource.build_resource_snapshot(PROJECT, available_job(), now=NOW)
    value.update(private="auth-token", message="auth-token")
    resource.atomic_json(tmp_path / "data" / "colab-resource-monitor" / f"{PROJECT}.status.json", value)
    monkeypatch.setattr(resource, "resource_monitor_is_running", lambda *_: 123)
    monkeypatch.setattr(resource, "utc_now", lambda: NOW)
    result = resource.resource_monitor_status(tmp_path, PROJECT)
    assert result["availability"] == "available" and result["agent_heartbeat_fresh"] is True
    assert "auth-token" not in json.dumps(result)


def test_resource_status_does_not_restore_old_positive_from_new_heartbeat(tmp_path, monkeypatch):
    value = resource.build_resource_snapshot(PROJECT, available_job(), now=NOW)
    value["heartbeat_at"] = resource.timestamp(NOW + timedelta(minutes=11))
    resource.atomic_json(tmp_path / "data" / "colab-resource-monitor" / f"{PROJECT}.status.json", value)
    monkeypatch.setattr(resource, "resource_monitor_is_running", lambda *_: 123)
    monkeypatch.setattr(resource, "utc_now", lambda: NOW + timedelta(minutes=11))
    assert resource.resource_monitor_status(tmp_path, PROJECT)["availability"] == "unknown"


@pytest.mark.parametrize("project", ["../private", "project/secret", "", "<script>"])
def test_rejects_invalid_project_before_launch(tmp_path, project):
    with pytest.raises(ValueError):
        resource.launch_resource_monitor(tmp_path, project, api_url="http://127.0.0.1:8767")


@pytest.mark.parametrize("url", ["https://127.0.0.1:8767", "http://outside.example", "http://user:secret@localhost", "http://localhost/path", "http://localhost/#secret"])
def test_rejects_unsafe_api_before_launch(tmp_path, url):
    with pytest.raises(ValueError):
        resource.launch_resource_monitor(tmp_path, PROJECT, api_url=url)


def test_rejects_linked_monitor_directory(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    monkeypatch.setattr(resource, "_linked", lambda path: path.name == "data")
    with pytest.raises(ValueError, match="linked"):
        resource.launch_resource_monitor(tmp_path, PROJECT, api_url="http://127.0.0.1:8767")


def test_http_adapter_only_gets_expected_project_with_no_proxy(monkeypatch):
    requests = []
    async def handle(request):
        requests.append(request)
        return httpx.Response(200, json=job())
    original = httpx.AsyncClient
    def factory(**kwargs):
        assert kwargs["trust_env"] is False
        return original(**kwargs, transport=httpx.MockTransport(handle))
    monkeypatch.setattr(resource.httpx, "AsyncClient", factory)
    client = resource.ResourceMonitorClient("http://127.0.0.1:8767", PROJECT)
    asyncio.run(client.status())
    asyncio.run(client.close())
    assert len(requests) == 1 and requests[0].method == "GET"
    assert requests[0].url.path == f"/api/projects/{PROJECT}/colab"
