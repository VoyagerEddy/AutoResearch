from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from autoresearch.colab_browser_cli import WorkerLock, _process_identity, worker_is_running, launch_colab_login_browser, run as run_colab_browser_cli
from autoresearch.services.colab_browser import (
    ColabBrowserError,
    ColabBrowserOptions,
    ColabBrowserSession,
    ColabBrowserWorker,
    ColabJobClient,
    COLAB_EMPTY_NOTEBOOK_URL,
    classify_colab_page,
    marker_is_current,
    parse_data_upload_requests,
    ManualColabAction,
    parse_status_markers,
    provider_cooldown_minutes,
    control_label,
    safe_control_diagnostics,
    source_stamp_matches,
    GOOGLE_SIGN_IN_REASON,
    is_browser_closed_error,
    parse_aria_controls,
    is_trusted_picker_frame,
    validate_colab_url,
    validate_local_api_url,
    public_colab_url,
    prepared_notebook_cells,
    mcp_tool_result,
    mcp_notebook_cells,
    mcp_stream_output_text,
    sanitize_provider_ui_text,
)
from autoresearch.services.colab_mcp import OFFICIAL_MCP_COMMIT


@pytest.mark.parametrize("url,text,expected", [
    ("https://accounts.google.com/v3/signin", "", "auth"),
    ("https://colab.research.google.com/", "Verify it's you", "auth"),
    ("https://colab.research.google.com/drive/example", "Cannot connect to GPU backend", "unavailable"),
    ("https://colab.research.google.com/drive/example", "Usage limits in Colab. Cannot connect to GPU backend", "quota"),
    ("https://colab.research.google.com/", "ERR_CONNECTION_TIMED_OUT", "network"),
    ("https://colab.research.google.com/drive/example", "Runtime has been disconnected", "disconnected"),
    ("https://colab.research.google.com/drive/example", "Disconnected · Connect · GPU", None),
    ("https://colab.research.google.com/", "Loading", None),
])
def test_provider_errors_remain_distinct(url: str, text: str, expected: str | None) -> None:
    assert classify_colab_page(url, text) == expected


@pytest.mark.parametrize("text", [
    "You cannot currently connect to a GPU due to usage limits in Colab. Learn more.",
    "You cannot currently connect to a GPU due to\nusage limits in Colab.",
    "Usage limit reached. Try again later.",
    "GPU usage limits have been exceeded. Reconnect in 6 hours.",
    "You have exceeded your GPU usage limit.",
    "Too many sessions. Manage sessions to continue.",
    "Too many active sessions.",
    "您目前无法连接到 GPU，原因是 Colab 中的使用量限制。",
    "已达到使用限额。请稍后再试。",
])
def test_explicit_provider_quota_denials_are_recognized(text: str) -> None:
    assert classify_colab_page("https://colab.research.google.com/drive/example", text) == "quota"


@pytest.mark.parametrize("text", [
    "Info: GPU is not being utilized. To avoid GPU usage limits, switch to a standard runtime.",
    "Your GPU is not being utilized. Disconnect this runtime to avoid reaching your GPU usage limits.",
    "Learn more about usage limits in Colab.",
    "GPU usage limits vary over time. Connect to start your runtime.",
    "You have exceeded your disk storage.",
    "Colab 使用量限制可能随时间变化。为了避免达到使用限制，请使用标准运行时。",
])
def test_provider_gpu_advisories_do_not_establish_a_quota_denial(text: str) -> None:
    assert classify_colab_page("https://colab.research.google.com/drive/example", text) is None


@pytest.mark.parametrize("text,expected", [
    ("Please try again in 4 hours.", 240),
    ("You can reconnect in 1.5 hours", 90),
    ("Wait 45 minutes before retry", 45),
    ("请等待 6 小时再试", 360),
    ("GPU T4 16 GB; please try again later", None),
])
def test_parse_actual_provider_cooldown(text: str, expected: int | None) -> None:
    assert provider_cooldown_minutes(text) == expected


@pytest.mark.parametrize("url", ["https://example.com/", "https://colab.research.google.com.evil/", "http://colab.research.google.com/", "https://user:secret@colab.research.google.com/"])
def test_notebook_navigation_rejects_untrusted_hosts(url: str) -> None:
    with pytest.raises(ValueError):
        validate_colab_url(url)


@pytest.mark.parametrize("url", ["https://127.0.0.1:8765", "http://example.com", "http://127.0.0.1:8765/api", "http://user:secret@localhost/"])
def test_worker_api_restricted_to_local_root(url: str) -> None:
    with pytest.raises(ValueError):
        validate_local_api_url(url)


def marker(status: str, *, runtime_id: str = "runtime-1", started: datetime | None = None, **values: object) -> str:
    payload = {"project_id": "project-1", "runtime_id": runtime_id,
               "started_at": (started or datetime.now(timezone.utc)).isoformat(), "status": status, **values}
    return "AUTORESEARCH_COLAB_STATUS " + json.dumps(payload)


def test_status_parser_ignores_source_wrong_project_and_malformed_output() -> None:
    output = '\n'.join([
        'print("AUTORESEARCH_COLAB_STATUS " + json.dumps(record))',
        'AUTORESEARCH_COLAB_STATUS {bad}',
        'AUTORESEARCH_COLAB_STATUS {"project_id":"another", "runtime_id":"r", "status":"completed"}',
        marker("starting"), marker("available"), marker("running"), marker("completed", returncode=0),
    ])
    assert [x["status"] for x in parse_status_markers(output, "project-1")] == ["starting", "available", "running", "completed"]


def test_old_output_cannot_establish_current_runtime() -> None:
    now = datetime.now(timezone.utc)
    old = parse_status_markers(marker("starting", started=now - timedelta(hours=2)), "project-1")[0]
    assert not marker_is_current(old, now)
    assert not marker_is_current({"started_at": "2026-10-08T12:00:00"}, now)


class FakeClient:
    project_id = "project-1"

    def __init__(self, job: dict | None = None) -> None:
        self.job = {"status": "attempting", "can_attempt": False, "prepared": True, **(job or {})}
        self.observations: list[tuple[str, dict]] = []
        self.claim_count = 0
        self.resume_count = 0

    async def status(self) -> dict:
        return dict(self.job)

    async def close(self) -> None:
        pass

    async def bind_notebook(self, *, notebook_url: str, bundle_sha256: str) -> dict:
        self.job.update(notebook_url=notebook_url,browser_url=notebook_url)
        return dict(self.job)

    async def claim(self) -> bool:
        self.claim_count += 1
        self.job["can_attempt"] = False
        return True

    async def resume(self) -> bool:
        self.resume_count += 1
        self.job.update(status="queued", can_attempt=True)
        return True

    async def observe(self, status: str, **details: object) -> dict:
        self.observations.append((status, details))
        self.job["status"] = {"available": "gpu_available", "auth": "auth_required", "unavailable": "waiting_for_gpu", "network": "waiting_for_connection"}.get(status, status)
        return dict(self.job)


class FakeBrowser:
    def __init__(self, outputs: str = "", text: str = "", url: str = "https://colab.research.google.com/drive/example") -> None:
        self.page = SimpleNamespace(url=url)
        self.options = SimpleNamespace(poll_seconds=1)
        self.outputs = outputs
        self.text = text
        self.opened: list[dict] = []
        self.run_all_count = 0
        self.disconnected_control = False
        self.visible_controls = []

    async def visible_text(self) -> str:
        return self.text

    async def output_text(self) -> str:
        return self.outputs

    async def click_observed(self, labels: tuple[str, ...]) -> bool:
        return False

    async def open_notebook(self, job: dict) -> None:
        self.opened.append(job)

    async def select_free_gpu(self) -> None:
        pass

    async def run_all(self) -> None:
        self.run_all_count += 1

    async def has_disconnected_runtime_control(self) -> bool:
        return self.disconnected_control

    async def controls(self) -> list:
        return self.visible_controls


@pytest.mark.asyncio
async def test_success_requires_current_runtime_running_and_zero_exit() -> None:
    client = FakeClient()
    browser = FakeBrowser('\n'.join([marker("starting"), marker("available"), marker("running"), marker("completed", returncode=0)]))
    worker = ColabBrowserWorker(client, browser)
    worker.attempt_started = datetime.now(timezone.utc)
    await worker._monitor(client.job)
    assert [x[0] for x in client.observations] == ["available", "running", "completed"]
    assert client.observations[-1][1]["returncode"] == 0
    assert client.observations[-1][1]["runtime_id"] == "runtime-1"


@pytest.mark.asyncio
async def test_failure_marker_reason_does_not_collide_with_observation_arguments() -> None:
    client = FakeClient()
    worker = ColabBrowserWorker(client, FakeBrowser('\n'.join([marker("starting"), marker("failed", reason="input missing", returncode=2)])))
    worker.attempt_started = datetime.now(timezone.utc)
    await worker._monitor(client.job)
    assert client.observations[0][0] == "failed"
    assert client.observations[0][1]["returncode"] == 2


@pytest.mark.asyncio
async def test_no_gpu_marker_schedules_retry_without_argument_collision() -> None:
    client = FakeClient()
    worker = ColabBrowserWorker(client, FakeBrowser('\n'.join([marker("starting"), marker("unavailable", reason="CUDA absent")])))
    worker.attempt_started = datetime.now(timezone.utc)
    await worker._monitor(client.job)
    assert client.observations[0][0] == "unavailable"
    assert worker.attempt_started is None


@pytest.mark.asyncio
async def test_checkpoint_budget_exit_is_retryable_not_terminal_failure() -> None:
    client = FakeClient()
    worker = ColabBrowserWorker(client, FakeBrowser('\n'.join([marker("starting"), marker("disconnected", returncode=75)])))
    worker.attempt_started = datetime.now(timezone.utc)
    await worker._monitor(client.job)
    assert client.observations[0][0] == "disconnected"
    assert worker.attempt_started is None


@pytest.mark.asyncio
async def test_recent_historical_output_cannot_complete_a_new_attempt() -> None:
    client = FakeClient()
    worker = ColabBrowserWorker(client, FakeBrowser('\n'.join([marker("starting"), marker("running"), marker("completed", returncode=0)])))
    worker.attempt_started = datetime.now(timezone.utc)
    worker._ignored_runtime_ids.add("runtime-1")
    await worker._monitor(client.job)
    assert client.observations == []
    assert worker.runtime_id is None


@pytest.mark.asyncio
async def test_oauth_pause_keeps_active_runtime_and_noncolab_url_is_not_forwarded() -> None:
    client = FakeClient()
    worker = ColabBrowserWorker(client, FakeBrowser(marker("starting"), url="https://accounts.google.com/signin"))
    worker.attempt_started = datetime.now(timezone.utc)
    await worker._monitor(client.job)
    assert worker.runtime_id == "runtime-1"
    assert worker.attempt_started is not None
    assert client.observations[0][0] == "auth"
    assert client.observations[0][1]["runtime_active"] is True
    assert "browser_url" not in client.observations[0][1]


@pytest.mark.asyncio
async def test_halted_auth_cell_preserves_runtime_for_verified_continuation_after_user_correction() -> None:
    client = FakeClient()
    worker = ColabBrowserWorker(client, FakeBrowser('\n'.join([marker("starting"), marker("auth", reason="missing_inputs")])))
    worker.attempt_started = datetime.now(timezone.utc)
    await worker._monitor(client.job)
    assert worker.runtime_id == 'runtime-1'
    assert worker.attempt_started is not None
    assert client.observations[0][1]["runtime_active"] is True
    assert client.observations[0][1]['authorization_continuation']['runtime_id']=='runtime-1'


@pytest.mark.asyncio
async def test_auth_marker_does_not_log_provider_authorization_url(capsys) -> None:
    sentinel = 'https://accounts.google.com/authorize?secret=AUTH_SENTINEL'
    client = FakeClient()
    browser = FakeBrowser('\n'.join([marker('starting'), marker('auth', reason=sentinel)]))
    worker = ColabBrowserWorker(client, browser)
    worker.attempt_started = datetime.now(timezone.utc)
    await worker._monitor(client.job)
    assert client.observations[-1][0] == 'auth'
    assert 'AUTH_SENTINEL' not in json.dumps(client.observations)
    assert 'AUTH_SENTINEL' not in capsys.readouterr().out


@pytest.mark.asyncio
async def test_quota_default_is_longer_than_gpu_pool_retry() -> None:
    client = FakeClient()
    worker = ColabBrowserWorker(client, FakeBrowser())
    await worker._provider_observation("quota", "Usage limit reached. Try again later")
    assert client.observations[0][1]["retry_after_minutes"] == 720
    await worker._provider_observation("quota", "Usage limit reached. Reconnect in 20 minutes")
    assert client.observations[1][1]["retry_after_minutes"] == 30


@pytest.mark.asyncio
@pytest.mark.parametrize("status,provider_message", [
    ("quota", "You cannot currently connect to a GPU due to usage limits in Colab."),
    ("unavailable", "Cannot connect to GPU backend."),
    ("network", "ERR_CONNECTION_TIMED_OUT"),
])
async def test_provider_failure_observations_keep_capped_sanitized_ui_wording(status: str, provider_message: str) -> None:
    private_details = " https://accounts.google.com/authorize?secret=AUTH_SENTINEL owner@example.com private-sentinel-code12345&55123 mcpProxyToken=another-private-code-token mcpProxyPort=12345"
    client = FakeClient()
    browser = FakeBrowser(outputs="print('NOTEBOOK_SOURCE_SENTINEL')", text=provider_message + private_details + " harmless" * 500)
    worker = ColabBrowserWorker(client, browser)
    worker.attempt_started = datetime.now(timezone.utc)
    await worker._monitor(client.job)
    observed_status, details = client.observations[-1]
    assert observed_status == status
    assert provider_message in details["provider_ui_text"]
    assert len(details["provider_ui_text"]) <= 1500
    serialized = json.dumps(details)
    assert not any(secret in serialized for secret in ("accounts.google.com", "AUTH_SENTINEL", "owner@example.com", "private-sentinel", "another-private", "55123", "12345", "NOTEBOOK_SOURCE_SENTINEL"))


@pytest.mark.asyncio
async def test_restarted_worker_monitors_existing_run_without_reexecuting(monkeypatch: pytest.MonkeyPatch) -> None:
    started = datetime.now(timezone.utc) - timedelta(hours=1)
    client = FakeClient({"status": "running", "runtime_id": "runtime-1", "last_attempt_at": started.isoformat()})
    browser = FakeBrowser('\n'.join([marker("starting", started=started), marker("running", started=started), marker("completed", started=started, returncode=0)]))
    worker = ColabBrowserWorker(client, browser)
    async def no_sleep(seconds: float) -> None:
        pass
    monkeypatch.setattr("autoresearch.services.colab_browser.asyncio.sleep", no_sleep)
    await worker.run()
    assert len(browser.opened) == 1
    assert browser.run_all_count == 0
    assert client.claim_count == 0
    assert client.observations[-1][0] == "completed"


@pytest.mark.asyncio
async def test_observation_request_has_top_level_retry_after_and_local_transport() -> None:
    requests = []
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"status": "waiting_for_gpu"})
    client = ColabJobClient("http://127.0.0.1:8765", "project-1")
    await client.client.aclose()
    client.client = httpx.AsyncClient(base_url=client.api_url, transport=httpx.MockTransport(handler))
    try:
        await client.observe("quota", retry_after_minutes=720, reason="Usage limit")
    finally:
        await client.close()
    assert requests == [{"status": "quota", "retry_after_minutes": 720, "details": {"reason": "Usage limit"}}]


def test_live_worker_pid_uses_process_creation_identity(tmp_path: Path) -> None:
    identity = _process_identity(os.getpid())
    assert identity is not None
    state = tmp_path / "data" / "colab-browser" / "project-1.worker.json"
    state.parent.mkdir(parents=True)
    state.write_text(json.dumps({"pid": os.getpid(), "process_identity": identity}), encoding="utf-8")
    assert worker_is_running(tmp_path, "project-1") == os.getpid()
    state.write_text(json.dumps({"pid": os.getpid(), "process_identity": "stale"}), encoding="utf-8")
    assert worker_is_running(tmp_path, "project-1") is None


def test_profile_lock_prevents_duplicate_controller_and_releases_after_exit(tmp_path: Path) -> None:
    lock = tmp_path / "profile.lock"
    with WorkerLock(lock):
        with pytest.raises(ColabBrowserError, match="already uses"):
            with WorkerLock(lock):
                pass
    with WorkerLock(lock):
        pass


def test_concurrent_launchers_start_one_worker_and_preserve_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from autoresearch import colab_browser_cli as cli

    launches = []
    def spawn(command, **kwargs):
        launches.append(command)
        return SimpleNamespace(pid=23456)
    monkeypatch.setattr(cli.subprocess, 'Popen', spawn)
    monkeypatch.setattr(cli, '_process_identity', lambda pid: 'worker-identity' if pid == 23456 else None)
    with ThreadPoolExecutor(max_workers=2) as pool:
        pids = list(pool.map(lambda _: cli.launch_colab_worker(tmp_path, 'project-1'), range(2)))
    assert pids == [23456, 23456]
    assert len(launches) == 1
    assert cli.worker_is_running(tmp_path, 'project-1') == 23456


def test_launcher_does_not_spawn_over_profile_owner_without_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from autoresearch import colab_browser_cli as cli

    launches = []
    monkeypatch.setattr(cli.subprocess, 'Popen', lambda *args, **kwargs: launches.append(args))
    profile = tmp_path / 'data' / 'colab-browser' / 'profile'
    with WorkerLock(profile.with_name('profile.lock')):
        with pytest.raises(ColabBrowserError, match='already uses'):
            cli.launch_colab_worker(tmp_path, 'project-1')
    assert launches == []


@pytest.mark.asyncio
async def test_browser_child_publishes_its_real_pid_before_opening_chrome(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from autoresearch import colab_browser_cli as cli

    class Client:
        async def status(self):
            return {'prepared': True}
        async def close(self):
            pass
    class Session:
        def __init__(self, options):
            pass
        async def __aenter__(self):
            identity = json.loads((tmp_path / 'data/colab-browser/project-1.worker.json').read_text())
            assert identity['pid'] == os.getpid()
            assert identity['process_identity'] == 'actual-interpreter'
            raise RuntimeError('test stops before browser launch')
        async def __aexit__(self, *_):
            pass
    monkeypatch.setattr(cli, 'ColabJobClient', lambda *args: Client())
    monkeypatch.setattr(cli, 'ColabBrowserSession', Session)
    monkeypatch.setattr(cli, '_process_identity', lambda pid: 'actual-interpreter')
    args = SimpleNamespace(project='project-1', root=tmp_path, profile_dir=None, browser='chrome',
                           executable_path=None, api_url='http://127.0.0.1:8765')
    with pytest.raises(RuntimeError, match='test stops'):
        await cli.run(args)


def test_observed_menu_label_normalizes_only_ellipsis_and_shortcuts() -> None:
    assert control_label("Open notebook…\nCtrl+O") == "open notebook"
    assert control_label("Run all (Ctrl+F9)") == "run all"
    assert control_label("T4 GPU") == "t4 gpu"


def test_ui_diagnostics_do_not_log_account_identifiers() -> None:
    controls = [
        {"role": "button", "name": "Sign in"},
        {"role": "button", "name": "Google Account: Person (person@example.com)"},
        {"role": "button", "name": "Profile: Person"},
        {"role": "link", "name": "13912345678"},
        {"role": "menuitem", "name": "Open notebook"},
        {"role": "menuitem", "name": "Open notebook"},
    ]
    assert safe_control_diagnostics(controls) == [controls[0], controls[4]]


def test_provider_ui_error_diagnostics_remove_urls_accounts_and_private_connection_details() -> None:
    text = "Notebook could not be loaded. Request access. https://accounts.google.com/private?secret=abc owner@example.com +86 13812345678 private-sentinel-code12345&55123 mcpProxyToken=another-private-code-token mcpProxyPort=12345"
    result = sanitize_provider_ui_text(text)
    assert "Notebook could not be loaded" in result
    assert "Request access" in result
    assert not any(secret in result for secret in ("accounts.google.com", "owner@example.com", "13812345678", "private-sentinel", "55123", "another-private", "12345"))
    assert len(sanitize_provider_ui_text("x " * 3000)) <= 1500


@pytest.mark.asyncio
async def test_close_only_provider_dialog_records_sanitized_error_text(tmp_path: Path) -> None:
    class ErrorSession(ColabBrowserSession):
        async def controls(self) -> list:
            return [{"role": "button", "name": "Close"}]
        async def visible_text(self) -> str:
            return "Could not load notebook. Sign in as owner@example.com. https://accounts.google.com/private private-sentinel-code12345&55123"
    class Dialog:
        async def is_visible(self) -> bool:
            return True
        async def evaluate(self, script: str) -> str:
            return "Notebook unavailable for owner@example.com"
    session = ErrorSession(ColabBrowserOptions(profile_dir=tmp_path))
    session.page = SimpleNamespace(frames=[], get_by_role=lambda role: FakeRoleList([Dialog()]))
    diagnostics = await session.ui_diagnostics()
    assert diagnostics["ui_controls"] == [{"role": "button", "name": "Close"}]
    assert "Could not load notebook" in diagnostics["provider_ui_text"]
    assert diagnostics["dialog_titles"] == ["Notebook unavailable for [email]"]
    assert "private-sentinel" not in json.dumps(diagnostics)
    assert "owner@example.com" not in json.dumps(diagnostics)


class PickerSession(ColabBrowserSession):
    def __init__(self, flow: str, tmp_path: Path) -> None:
        super().__init__(ColabBrowserOptions(profile_dir=tmp_path))
        self.flow = flow
        self.state = "home"
        self.clicked = []

    async def click_observed(self, labels: tuple[str, ...]) -> bool:
        targets = {control_label(label) for label in labels}
        available = {
            "direct": {"home": "open notebook", "picker": "upload"},
            "picker": {"home": "open file picker", "picker": "upload"},
            "file": {"home": "file", "file": "open notebook", "picker": "upload"},
            "new": {"home": "new notebook", "editor": "file", "file": "open notebook", "picker": "upload"},
            "signin": {"home": "sign in"},
        }[self.flow].get(self.state)
        if available not in targets:
            return False
        self.clicked.append(available)
        self.state = {"open file picker": "picker", "open notebook": "picker", "file": "file", "new notebook": "editor", "upload": "upload", "sign in": "signin"}[available]
        return True

    async def click_when_visible(self, labels: tuple[str, ...], *, timeout_seconds: float = 8) -> bool:
        return await self.click_observed(labels)


@pytest.mark.asyncio
@pytest.mark.parametrize("flow,clicks", [
    ("direct", ["open notebook", "upload"]),
    ("picker", ["open file picker", "upload"]),
    ("file", ["file", "open notebook", "upload"]),
    ("new", ["new notebook", "file", "open notebook", "upload"]),
])
async def test_upload_picker_opens_through_observed_standard_ui(flow: str, clicks: list[str], tmp_path: Path) -> None:
    session = PickerSession(flow, tmp_path)
    assert await session.open_upload_picker()
    assert session.clicked == clicks


@pytest.mark.asyncio
async def test_signin_pause_requires_an_actual_visible_signin_control(tmp_path: Path) -> None:
    session = PickerSession("signin", tmp_path)
    with pytest.raises(ManualColabAction, match="Google sign-in"):
        await session.open_upload_picker()
    assert session.clicked == ["sign in"]


@pytest.mark.asyncio
async def test_recent_only_home_picker_falls_back_to_empty_editor_import(tmp_path: Path) -> None:
    events = []

    class RecentOnlySession(ColabBrowserSession):
        def __init__(self) -> None:
            super().__init__(ColabBrowserOptions(profile_dir=tmp_path))
            self.state = "home"
            self.page = SimpleNamespace(goto=self.goto)

        async def goto(self, url: str, *, wait_until: str) -> None:
            assert (url, wait_until) == (COLAB_EMPTY_NOTEBOOK_URL, "domcontentloaded")
            assert self.state == "home"
            events.append(url)
            self.state = "editor"

        async def click_observed(self, labels: tuple[str, ...]) -> bool:
            target = {
                "home": "open file picker", "recent": "cancel", "editor": "file",
                "file": "open notebook", "picker": "upload",
            }.get(self.state)
            if target not in {control_label(label) for label in labels}:
                return False
            events.append(target)
            self.state = {
                "open file picker": "recent", "cancel": "home", "file": "file",
                "open notebook": "picker", "upload": "upload",
            }[target]
            return True

        async def click_when_visible(self, labels: tuple[str, ...], *, timeout_seconds: float = 8) -> bool:
            return await self.click_observed(labels)

    session = RecentOnlySession()
    assert await session.open_upload_picker()
    assert events == [
        "open file picker", "cancel", COLAB_EMPTY_NOTEBOOK_URL,
        "file", "open notebook", "upload",
    ]


def test_picker_diagnostics_omit_notebook_names_inside_widget_labels() -> None:
    controls = [
        {"role": "button", "name": "View private research.ipynb (last opened yesterday)"},
        {"role": "link", "name": "PRIVATE PROJECT.IPYNB shared with you"},
        {"role": "button", "name": "Open\nconfidential.ipynb\nin a new tab"},
        {"role": "tab", "name": "Upload"},
    ]
    assert safe_control_diagnostics(controls) == [controls[-1]]


@pytest.mark.asyncio
async def test_observed_radio_control_selects_free_t4_gpu(tmp_path: Path) -> None:
    clicked = []
    class Candidate:
        async def is_visible(self) -> bool:
            return True
        async def is_disabled(self) -> bool:
            return False
        async def click(self) -> None:
            clicked.append("T4 GPU")
    class Locator:
        async def all(self) -> list:
            return [Candidate()]
    class Page:
        def get_by_role(self, role: str, *, name: str, exact: bool) -> Locator:
            assert (role, name, exact) == ("radio", "T4 GPU", True)
            return Locator()
    session = ColabBrowserSession(ColabBrowserOptions(profile_dir=tmp_path))
    session.page = Page()
    async def controls() -> list:
        return [{"role": "radio", "name": "T4 GPU"}]
    session.controls = controls
    assert await session.click_observed(("T4 GPU", "GPU"))
    assert clicked == ["T4 GPU"]


@pytest.mark.asyncio
async def test_worker_recognizes_user_started_notebook_during_manual_pause() -> None:
    client = FakeClient()
    browser = FakeBrowser(marker("starting"))
    worker = ColabBrowserWorker(client, browser)
    worker.attempt_started = datetime.now(timezone.utc)
    await worker._adopt_manual_runtime()
    assert worker.runtime_id == "runtime-1"
    assert browser.run_all_count == 0


def test_source_stamp_does_not_accept_old_or_conflicting_source_hashes() -> None:
    expected = "a" * 64
    old = "b" * 64
    assert source_stamp_matches("AUTORESEARCH_SOURCE_SHA256 " + expected, expected)
    assert not source_stamp_matches("AUTORESEARCH_SOURCE_SHA256 " + old, expected)
    assert not source_stamp_matches("AUTORESEARCH_SOURCE_SHA256 " + expected + "\nAUTORESEARCH_SOURCE_SHA256 " + old, expected)
    assert not source_stamp_matches("AUTORESEARCH_SOURCE_SHA256 " + expected, "")


@pytest.mark.asyncio
async def test_saved_notebook_must_match_current_prepared_source_before_execution(tmp_path: Path) -> None:
    expected = "a" * 64
    class GuardSession(ColabBrowserSession):
        def __init__(self) -> None:
            super().__init__(ColabBrowserOptions(profile_dir=tmp_path))
            self.rendered = "AUTORESEARCH_SOURCE_SHA256 " + "b" * 64
            self.import_count = 0
        async def rendered_markdown(self) -> str:
            return self.rendered
        async def import_prepared_notebook(self, job: dict) -> None:
            self.import_count += 1
            self.rendered = "AUTORESEARCH_SOURCE_SHA256 " + expected
    session = GuardSession()
    await session.ensure_prepared_notebook({"bundle_sha256": expected, "notebook_path": "fresh.ipynb"})
    assert session.import_count == 1
    await session.ensure_prepared_notebook({"bundle_sha256": expected, "notebook_path": "fresh.ipynb"})
    assert session.import_count == 1


def prepared_mcp_job(tmp_path: Path) -> dict:
    source_hash = "a" * 64
    notebook = tmp_path / "AutoResearch_project-1.ipynb"
    notebook.write_text(json.dumps({"nbformat": 4, "cells": [
        {"cell_type": "markdown", "source": ["# Experiment\n", "AUTORESEARCH_SOURCE_SHA256 " + source_hash + "\n"]},
        {"cell_type": "code", "source": ["print('safe prepared experiment')\n"]},
    ]}), encoding="utf-8")
    return {"project_id": "project-1", "notebook_path": str(notebook), "bundle_sha256": source_hash}


class FakeNotebookMcp:
    def __init__(self, cells: list[dict] | None = None) -> None:
        self.cells = [{"cell_type": "code", "id": "initial", "source": []}] if cells is None else cells
        self.calls = []

    def status(self) -> dict:
        return {"connected": True}

    async def request(self, method: str, *, name: str | None = None, arguments: dict | None = None, timeout: float = 900) -> dict:
        self.calls.append((method, name, arguments))
        if method == "list_tools":
            return {"ok": True, "result": [
                {"name": name, "inputSchema": {"properties": {key: {"type": kind} for key, kind in fields.items()}}}
                for name, fields in {
                    "get_cells": {"includeOutputs": "boolean"},
                    "add_code_cell": {"cellIndex": "integer", "language": "string", "code": "string"},
                    "add_text_cell": {"cellIndex": "integer", "content": "string"},
                }.items()
            ]}
        if name == "get_cells":
            return {"ok": True, "result": {"content": [{"type": "text", "text": json.dumps({"cells": self.cells})}], "is_error": False}}
        assert name in {"add_code_cell", "add_text_cell"}
        assert arguments is not None
        self.cells.insert(arguments["cellIndex"], {
            "cell_type": "code" if name == "add_code_cell" else "markdown",
            "id": f"cell-{len(self.cells)}",
            "source": arguments["code"] if name == "add_code_cell" else arguments["content"],
        })
        return {"ok": True, "result": {"content": [{"type": "text", "text": "new-cell-id"}], "is_error": False}}


class McpAuthoringSession(ColabBrowserSession):
    def __init__(self, tmp_path: Path, client: FakeNotebookMcp) -> None:
        super().__init__(ColabBrowserOptions(profile_dir=tmp_path))
        self.client = client
        self.connection_count = 0
        self.attached_url = None

    def _official_mcp_location(self) -> tuple[Path, Path]:
        return self.options.profile_dir, self.options.profile_dir / "state"

    async def _connect_official_notebook_mcp(self, root: Path, state: Path, *, expected_cells: list[tuple[str, str]] | None = None, notebook_url: str | None = None) -> FakeNotebookMcp:
        self.connection_count += 1
        self.attached_url = notebook_url
        self._notebook_mcp = self.client
        return self.client

    async def ensure_mcp_registered_notebook(self, response: dict) -> None:
        # Cell/page binding itself is exercised separately with DOM fixtures.
        pass

    async def rendered_markdown(self) -> str:
        return "\n".join(str(cell["source"]) for cell in self.client.cells if cell["cell_type"] == "markdown")


class SavePage:
    def __init__(self,url: str) -> None:
        self.url,self.closed = url,False
    async def close(self) -> None:
        self.closed = True


class SaveNotebookSession(McpAuthoringSession):
    def __init__(self,tmp_path,job):
        expected = prepared_notebook_cells(job)
        client = FakeNotebookMcp([{'cell_type':kind,'source':source,'id':f'cell-{index}'}
                                  for index,(kind,source) in enumerate(expected)])
        super().__init__(tmp_path,client)
        self._notebook_mcp = client
        self.original = SavePage(COLAB_EMPTY_NOTEBOOK_URL)
        self.other = SavePage('https://colab.research.google.com/drive/someone-else')
        self.saved = SavePage('https://colab.research.google.com/drive/new-project-copy?authuser=0#cell-1')
        self.page = self.original
        self.context = SimpleNamespace(pages=[self.original,self.other])
        self.clicks = []
        self.gpu_during_copy = False
    async def _click_unique_main_control(self,labels):
        self.clicks.append(labels)
        self.context.pages.append(self.saved)
        if self.gpu_during_copy:
            self.client.cells[-1]['outputs'] = [{'output_type':'stream','name':'stdout','text':marker('available')}]
        return True
    async def output_text(self):
        return mcp_stream_output_text(await self.client.request('call_tool',name='get_cells',arguments={'includeOutputs':True}))
    async def visible_text(self):
        return ''


@pytest.mark.asyncio
async def test_save_verified_scratchpad_closes_only_its_own_gpu_free_original(tmp_path):
    job = prepared_mcp_job(tmp_path)
    session = SaveNotebookSession(tmp_path,job)
    url = await session.save_prepared_notebook(job)
    assert url == 'https://colab.research.google.com/drive/new-project-copy'
    assert session.original.closed and not session.other.closed and not session.saved.closed
    assert session.page is session.saved and session._notebook_mcp is None
    assert len(session.clicks)==1


@pytest.mark.asyncio
@pytest.mark.parametrize('when',['before','during'])
async def test_save_never_closes_a_scratchpad_with_an_allocated_gpu(tmp_path,when):
    job = prepared_mcp_job(tmp_path)
    session = SaveNotebookSession(tmp_path,job)
    if when=='before':
        session.client.cells[-1]['outputs'] = [{'output_type':'stream','name':'stdout','text':marker('available')}]
    else:
        session.gpu_during_copy = True
    with pytest.raises(ManualColabAction,match='GPU'):
        await session.save_prepared_notebook(job)
    assert not session.original.closed and not session.other.closed and not session.saved.closed
    assert not session.clicks if when=='before' else len(session.clicks)==1


@pytest.mark.asyncio
async def test_saving_never_copies_tampered_or_wrong_tab_source(tmp_path):
    job = prepared_mcp_job(tmp_path)
    session = SaveNotebookSession(tmp_path,job)
    session.client.cells[-1]['source'] = 'other_private_work()'
    with pytest.raises(ColabBrowserError,match='source changed'):
        await session.save_prepared_notebook(job)
    assert session.clicks == [] and not session.original.closed
    session = SaveNotebookSession(tmp_path,job)
    async def wrong_tab(response):
        raise ManualColabAction('The bridge is connected to another notebook')
    session.ensure_mcp_registered_notebook = wrong_tab
    with pytest.raises(ManualColabAction,match='another notebook'):
        await session.save_prepared_notebook(job)
    assert session.clicks == [] and not session.original.closed


@pytest.mark.asyncio
async def test_saved_drive_notebook_is_only_bound_without_copying_or_closing_any_tab(tmp_path):
    job = prepared_mcp_job(tmp_path)
    session = SaveNotebookSession(tmp_path,job)
    session.page.url = 'https://colab.research.google.com/drive/already-owned?authuser=0#cell-1'
    assert await session.save_prepared_notebook(job) == 'https://colab.research.google.com/drive/already-owned'
    assert session.clicks == [] and not session.original.closed and session._notebook_mcp is session.client


@pytest.mark.asyncio
@pytest.mark.parametrize('unknown',[False,True])
async def test_explicit_saved_upgrade_changes_only_exact_owned_old_source(tmp_path,unknown):
    (tmp_path/'old').mkdir()
    (tmp_path/'new').mkdir()
    old_job,new_job=prepared_mcp_job(tmp_path/'old'),prepared_mcp_job(tmp_path/'new')
    new_notebook=json.loads(Path(new_job['notebook_path']).read_text())
    new_notebook['cells'][-1]['source']=['print("new approved experiment")\n']
    Path(new_job['notebook_path']).write_text(json.dumps(new_notebook))
    expected=prepared_notebook_cells(old_job)
    class UpdateMcp(FakeNotebookMcp):
        async def request(self,method,**kwargs):
            if kwargs.get('name')=='update_cell':
                self.calls.append((method,'update_cell',kwargs['arguments']))
                for cell in self.cells:
                    if cell['id']==kwargs['arguments']['cellId']:
                        cell['source']=kwargs['arguments']['content']
                return {'ok':True,'result':{'is_error':False}}
            return await super().request(method,**kwargs)
    client=UpdateMcp([{'cell_type':kind,'source':source,'id':f'cell-{i}'} for i,(kind,source) in enumerate(expected)])
    if unknown:
        client.cells[-1]['source']='user_private_changes()'
    session=McpAuthoringSession(tmp_path,client)
    session._notebook_mcp=client
    keys=[]
    class Keyboard:
        async def press(self,key):
            keys.append(key)
    session.page=SimpleNamespace(url='https://colab.research.google.com/drive/owned',keyboard=Keyboard())
    if unknown:
        with pytest.raises(ManualColabAction,match='differs from the retained old'):
            await session.upgrade_registered_notebook(old_job,new_job,session.page.url)
        assert not any(name=='update_cell' for _,name,_ in client.calls) and keys==[]
    else:
        assert await session.upgrade_registered_notebook(old_job,new_job,session.page.url)==session.page.url
        assert [(cell['cell_type'],cell['source']) for cell in client.cells]==prepared_notebook_cells(new_job)
        assert sum(name=='update_cell' for _,name,_ in client.calls)==1 and keys==['Control+s']
    assert all(name not in {'run_code_cell','delete_cell','add_code_cell','add_text_cell'} for _,name,_ in client.calls)


@pytest.mark.asyncio
async def test_save_control_ambiguity_is_not_resolved_by_clicking_an_arbitrary_button(tmp_path):
    session = ColabBrowserSession(ColabBrowserOptions(profile_dir=tmp_path))
    async def controls(page):
        return [{'role':'button','name':'Copy to Drive'},{'role':'menuitem','name':'Copy to Drive'}]
    session._dom_controls = controls
    session.page = SimpleNamespace()
    with pytest.raises(ManualColabAction,match='ambiguous'):
        await session._click_unique_main_control(('Copy to Drive',))


@pytest.mark.asyncio
async def test_fresh_attempt_saves_binds_and_rechecks_the_drive_copy_before_gpu_request(monkeypatch):
    events = []
    client = FakeClient({'status':'queued','can_attempt':True,'bundle_sha256':'a'*64})
    browser = FakeBrowser(outputs='')
    async def opened(job):
        events.append('open-saved' if job.get('notebook_url') else 'open-scratchpad')
    async def verified(job):
        events.append('verify')
    async def saved(job):
        events.append('save')
        return 'https://colab.research.google.com/drive/new-project-copy'
    async def bound(**kwargs):
        events.append('bind')
        return dict(client.job)
    async def gpu():
        events.append('gpu')
    async def run_all():
        events.append('run-all')
        browser.outputs='\n'.join([marker('starting'),marker('available'),marker('running'),marker('completed',returncode=0)])
    browser.open_notebook,browser.ensure_prepared_notebook,browser.save_prepared_notebook = opened,verified,saved
    browser.select_free_gpu,browser.run_all = gpu,run_all
    client.bind_notebook = bound
    async def no_sleep(seconds):
        pass
    monkeypatch.setattr('autoresearch.services.colab_browser.asyncio.sleep',no_sleep)
    await asyncio.wait_for(ColabBrowserWorker(client,browser).run(),timeout=3)
    assert events == ['open-scratchpad','verify','save','bind','open-saved','verify','gpu','run-all']


@pytest.mark.asyncio
async def test_authorization_continuation_executes_one_cell_asynchronously_and_keeps_monitoring_same_runtime(tmp_path):
    started = datetime.now(timezone.utc)-timedelta(minutes=3)
    job = prepared_mcp_job(tmp_path)
    notebook = json.loads(Path(job['notebook_path']).read_text())
    notebook['cells'] = [notebook['cells'][0],{'cell_type':'code','source':"RUNTIME_ID = 'original'"},
        {'cell_type':'code','source':'EMBEDDED_SOURCE = approved_zip'},
        {'cell_type':'code','source':"from google.colab import drive, files\ndrive.mount('/content/drive')"},
        {'cell_type':'code','source':'continue_prepared_helpers()'}]
    Path(job['notebook_path']).write_text(json.dumps(notebook))
    cells = [{**cell,'id':f'cell-{index}','execution_count':index if index<4 else None,'outputs':[]}
             for index,cell in enumerate(notebook['cells'])]
    cells[1]['outputs'] = [{'output_type':'stream','name':'stdout','text':'\n'.join([
        marker('starting',started=started),marker('available',started=started)])}]
    cells[3]['outputs'] = [{'output_type':'stream','name':'stdout','text':marker('auth',started=started)},
                           {'output_type':'error','ename':'ValueError','evalue':'mount failed'}]
    client = FakeClient({**job,'status':'gpu_available'})
    browser = FakeBrowser(outputs='\n'.join([marker('starting',started=started),marker('available',started=started),marker('auth',started=started)]))
    dispatched, release = [],asyncio.Event()
    async def snapshot(spec):
        return cells
    async def dispatch(spec,cell_id):
        dispatched.append(cell_id)
        if cell_id=='cell-3':
            await release.wait()
            cells[3].update(execution_count=10,outputs=[])
    browser.authorization_continuation_snapshot,browser.run_continuation_cell=snapshot,dispatch
    worker = ColabBrowserWorker(client,browser)
    worker.runtime_id,worker.attempt_started = 'runtime-1',started
    worker._authorization_continuation={'runtime_id':'runtime-1','started_at':started.isoformat(),'inflight_cell_id':None}
    worker._authorization_resumed=True
    await worker._monitor(client.job)
    await asyncio.sleep(0)
    assert dispatched==['cell-3']
    await worker._monitor(client.job)
    assert dispatched==['cell-3'] and all(status!='auth' for status,_ in client.observations)
    assert worker.runtime_id=='runtime-1' and worker.attempt_started==started and browser.run_all_count==0
    release.set()
    await worker._continuation_task
    await worker._tick_authorization_continuation(client.job)
    await asyncio.sleep(0)
    assert dispatched==['cell-3','cell-4']
    assert client.observations[-1][1]['authorization_continuation']['inflight_cell_id']=='cell-4'
    await worker._continuation_task
    await worker._tick_authorization_continuation(client.job)
    assert dispatched==['cell-3','cell-4']


@pytest.mark.asyncio
async def test_mcp_authors_exact_export_into_blank_notebook_before_any_execution(tmp_path: Path) -> None:
    job = prepared_mcp_job(tmp_path)
    client = FakeNotebookMcp()
    session = McpAuthoringSession(tmp_path, client)
    await session.open_notebook(job)
    await session.ensure_prepared_notebook(job)
    expected = prepared_notebook_cells(job)
    assert [(cell["cell_type"], cell["source"]) for cell in client.cells[1:]] == expected
    assert client.cells[0]["source"] == []
    mutations = [(name, arguments) for _, name, arguments in client.calls if name != "get_cells" and name]
    assert mutations == [
        ("add_text_cell", {"cellIndex": 1, "content": expected[0][1]}),
        ("add_code_cell", {"cellIndex": 2, "language": "python", "code": expected[1][1]}),
    ]
    await session.open_notebook(job)
    assert session.connection_count == 1
    assert len(client.cells) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("existing", [
    [{"cell_type": "code", "id": "private", "source": ["private_code()"]}],
    [{"cell_type": "markdown", "id": "partial", "source": ["AUTORESEARCH_SOURCE_SHA256 " + "a" * 64]}],
])
async def test_mcp_never_overwrites_existing_or_partially_imported_source(tmp_path: Path, existing: list[dict]) -> None:
    job = prepared_mcp_job(tmp_path)
    client = FakeNotebookMcp(existing)
    session = McpAuthoringSession(tmp_path, client)
    with pytest.raises(ManualColabAction, match="existing content"):
        await session.open_notebook(job)
    assert all(name is None or name == "get_cells" for _, name, _ in client.calls)
    assert client.cells == existing


@pytest.mark.asyncio
async def test_mcp_never_authors_into_another_tabs_blank_notebook(tmp_path: Path) -> None:
    job = prepared_mcp_job(tmp_path)
    client = FakeNotebookMcp()
    session = McpAuthoringSession(tmp_path, client)
    async def wrong_tab(response: dict) -> None:
        raise ManualColabAction("The bridge is connected to another notebook")
    session.ensure_mcp_registered_notebook = wrong_tab
    with pytest.raises(ManualColabAction, match="another notebook"):
        await session.open_notebook(job)
    assert client.cells == [{"cell_type": "code", "id": "initial", "source": []}]
    assert all(name is None or name == "get_cells" for _, name, _ in client.calls)


@pytest.mark.asyncio
async def test_saved_notebook_mcp_attachment_is_read_only_and_verifies_owned_source(tmp_path: Path) -> None:
    job = prepared_mcp_job(tmp_path)
    job["notebook_url"] = "https://colab.research.google.com/drive/owned-notebook"
    expected = prepared_notebook_cells(job)
    client = FakeNotebookMcp([{ "cell_type": kind, "source": source, "id": f"cell-{index}" } for index, (kind, source) in enumerate(expected)])
    session = McpAuthoringSession(tmp_path, client)
    await session.open_notebook(job)
    assert session.attached_url == job["notebook_url"]
    assert session._notebook_mcp is client
    assert all(name == "get_cells" for _, name, _ in client.calls)


@pytest.mark.asyncio
async def test_saved_notebook_mcp_attachment_never_authors_or_executes_unknown_content(tmp_path: Path) -> None:
    job = prepared_mcp_job(tmp_path)
    job["browser_url"] = "https://colab.research.google.com/drive/unknown-notebook"
    client = FakeNotebookMcp([{"cell_type": "code", "source": "private_work()", "id": "private"}])
    session = McpAuthoringSession(tmp_path, client)
    with pytest.raises(ManualColabAction, match="saved notebook content"):
        await session.open_notebook(job)
    assert session._notebook_mcp is None
    assert all(name == "get_cells" for _, name, _ in client.calls)


@pytest.mark.asyncio
async def test_mcp_source_guard_rejects_missing_artifact_stamp_before_connecting(tmp_path: Path) -> None:
    job = prepared_mcp_job(tmp_path)
    job["bundle_sha256"] = "b" * 64
    client = FakeNotebookMcp()
    session = McpAuthoringSession(tmp_path, client)
    with pytest.raises(ColabBrowserError, match="checksum"):
        await session.open_notebook(job)
    assert session.connection_count == 0
    assert client.calls == []


@pytest.mark.asyncio
async def test_mcp_rechecks_all_source_after_authoring_before_worker_runs_it(tmp_path: Path) -> None:
    job = prepared_mcp_job(tmp_path)
    client = FakeNotebookMcp()
    session = McpAuthoringSession(tmp_path, client)
    await session.open_notebook(job)
    client.cells[-1]["source"] = "tampered_source()"
    with pytest.raises(ColabBrowserError, match="source changed"):
        await session.ensure_prepared_notebook(job)
    assert all(name != "run_code_cell" for _, name, _ in client.calls)


@pytest.mark.asyncio
async def test_verified_mcp_source_never_allows_run_all_on_unverified_browser_tab(tmp_path: Path) -> None:
    job = prepared_mcp_job(tmp_path)
    client = FakeNotebookMcp()
    session = McpAuthoringSession(tmp_path, client)
    await session.open_notebook(job)
    async def other_notebook_markdown() -> str:
        return "This is another notebook"
    session.rendered_markdown = other_notebook_markdown
    with pytest.raises(ManualColabAction, match="browser tab"):
        await session.ensure_prepared_notebook(job)


def test_optional_mcp_environment_is_restricted_to_canonical_app_profile(tmp_path: Path) -> None:
    state = tmp_path / "data" / "colab-mcp" / "speaker-session"
    state.mkdir(parents=True)
    python = tmp_path / "data" / "colab-mcp" / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    python.parent.mkdir(parents=True)
    python.write_bytes(b"placeholder")
    session = ColabBrowserSession(ColabBrowserOptions(profile_dir=tmp_path / "data" / "colab-browser" / "profile"))
    assert session._official_mcp_location() == (tmp_path.resolve(), state.resolve())
    session.options = ColabBrowserOptions(profile_dir=tmp_path / "custom-profile")
    assert session._official_mcp_location() is None


def test_native_mcp_cell_json_uses_observed_source_array_schema() -> None:
    payload = {"cells": [{"cell_type": "code", "id": "initial", "source": []}]}
    result = {"content": [{"type": "text", "text": json.dumps(payload)}], "structured_content": payload, "data": payload, "is_error": False}
    assert mcp_notebook_cells({"ok": True, "result": result}) == [("code", "")]
    assert mcp_tool_result({"ok": True, "result": {"data": payload}}) == payload
    with pytest.raises(ColabBrowserError, match="reported an error"):
        mcp_tool_result({"ok": True, "result": {**result, "is_error": True}})


def test_native_stream_reader_never_reads_source_or_display_html() -> None:
    response = {"ok": True, "result": {"data": {"cells": [{
        "cell_type": "code", "source": [marker("completed", returncode=0)],
        "outputs": [
            {"output_type": "stream", "name": "stdout", "text": [marker("starting") + "\n", marker("available", gpu="Tesla T4, 15360 MiB") + "\n"]},
            {"output_type": "stream", "name": "stderr", "text": "stderr diagnostic\n"},
            {"output_type": "display_data", "data": {"text/html": "private HTML", "text/plain": marker("completed", returncode=0)}},
            {"output_type": "error", "traceback": ["private source traceback"]},
        ],
    }]}}}
    text = mcp_stream_output_text(response)
    assert [item["status"] for item in parse_status_markers(text, "project-1")] == ["starting", "available"]
    assert "stderr diagnostic" in text
    assert "private" not in text and "completed" not in text


@pytest.mark.asyncio
async def test_output_reader_gets_native_stream_markers_when_dom_outputs_are_absent(tmp_path: Path) -> None:
    client = FakeNotebookMcp([{
        "cell_type": "code", "source": marker("completed", returncode=0), "id": "native",
        "outputs": [{"output_type": "stream", "name": "stdout", "text": [marker("starting") + "\n", marker("running") + "\n"]}],
    }])
    session = ColabBrowserSession(ColabBrowserOptions(profile_dir=tmp_path))
    session._notebook_mcp = client
    session.page = SimpleNamespace(frames=[], locator=lambda selector: FakeRoleList([]))
    outputs = await session.output_text()
    assert [item["status"] for item in parse_status_markers(outputs, "project-1")] == ["starting", "running"]
    assert client.calls[-1] == ("call_tool", "get_cells", {"includeOutputs": True})


@pytest.mark.asyncio
async def test_drive_prompt_keeps_verified_gpu_and_runtime_before_manual_pause() -> None:
    client = FakeClient()
    browser = FakeBrowser(outputs="\n".join([marker("starting"), marker("available", gpu="Tesla T4, 15360 MiB")]), text="Connect to Google Drive")
    worker = ColabBrowserWorker(client, browser)
    worker.attempt_started = datetime.now(timezone.utc)
    await worker._monitor(client.job)
    assert [status for status, _ in client.observations] == ["available", "auth"]
    assert worker.runtime_id == "runtime-1"
    assert worker.available_reported
    assert client.observations[-1][1]["runtime_active"] is True
    assert client.observations[-1][1]["runtime_id"] == "runtime-1"
    assert browser.run_all_count == 0


@pytest.mark.asyncio
async def test_reopened_active_runtime_preserves_its_started_at_instead_of_rerunning() -> None:
    started = datetime.now(timezone.utc) - timedelta(hours=2)
    client = FakeClient({
        "status": "gpu_available", "runtime_id": "runtime-1", "last_attempt_at": datetime.now(timezone.utc).isoformat(),
        "last_observation": {"status": "auth", "details": {"runtime_id": "runtime-1", "runtime_active": True, "started_at": started.isoformat()}},
    })
    browser = FakeBrowser(outputs="\n".join([marker("starting", started=started), marker("running", started=started), marker("completed", started=started, returncode=0)]))
    worker = ColabBrowserWorker(client, browser)
    await asyncio.wait_for(worker.run(), timeout=3)
    assert [status for status, _ in client.observations] == ["running", "completed"]
    assert browser.run_all_count == 0
    assert len(browser.opened) == 1


@pytest.mark.asyncio
async def test_saved_active_runtime_manual_connection_pause_preserves_identity_and_reattaches_on_resume(monkeypatch: pytest.MonkeyPatch) -> None:
    started = datetime.now(timezone.utc) - timedelta(hours=1)
    client = FakeClient({"status": "running", "runtime_id": "runtime-1", "last_attempt_at": started.isoformat(), "notebook_url": "https://colab.research.google.com/drive/owned-notebook"})
    browser = FakeBrowser(outputs="\n".join([marker("starting", started=started), marker("running", started=started), marker("completed", started=started, returncode=0)]))
    worker = ColabBrowserWorker(client, browser)
    opens = []
    async def saved_open(job: dict) -> None:
        opens.append(job)
        if len(opens) == 1:
            raise ManualColabAction("Connect this saved notebook's official bridge")
    browser.open_notebook = saved_open
    async def diagnostics() -> dict:
        return {"ui_controls": [{"role": "button", "name": "Connect"}], "ui_frames": [], "ui_role_tree": ""}
    browser.ui_diagnostics = diagnostics
    original_observe = client.observe
    async def observe_with_user_resume(status: str, **details: object) -> dict:
        result = await original_observe(status, **details)
        if status == "auth":
            assert worker.runtime_id == "runtime-1"
            assert worker.attempt_started == started
            assert details["runtime_active"] is True
            assert details["runtime_id"] == "runtime-1"
            assert details["started_at"] == started.isoformat()
            # Simulate the user's explicit Resume after connecting the dialog.
            client.job.update(status="queued", can_attempt=True)
        return result
    client.observe = observe_with_user_resume
    async def no_sleep(seconds: float) -> None:
        pass
    monkeypatch.setattr("autoresearch.services.colab_browser.asyncio.sleep", no_sleep)
    await asyncio.wait_for(worker.run(), timeout=3)
    assert len(opens) == 2
    assert client.claim_count == 1
    assert [status for status, _ in client.observations] == ["auth", "completed"]
    assert client.observations[0][1]["manual_action"] is True
    assert client.observations[0][1]["ui_controls"] == [{"role": "button", "name": "Connect"}]
    assert browser.run_all_count == 0


@pytest.mark.asyncio
async def test_observation_never_persists_private_mcp_fragment() -> None:
    client = FakeClient()
    browser = FakeBrowser(url="https://colab.research.google.com/notebooks/empty.ipynb#mcpProxyToken=private-secret&mcpProxyPort=12345")
    worker = ColabBrowserWorker(client, browser)
    await worker._observation("available", "GPU confirmed")
    assert client.observations[-1][1]["browser_url"] == COLAB_EMPTY_NOTEBOOK_URL
    assert "private-secret" not in json.dumps(client.observations)
    assert public_colab_url("https://colab.research.google.com/drive/example#cell-1") == "https://colab.research.google.com/drive/example"


class FakeMcpDialogControl:
    def __init__(self, label: str, clicked: list[str]) -> None:
        self.label = label
        self.clicked = clicked

    async def is_visible(self) -> bool:
        return True

    async def is_disabled(self) -> bool:
        return False

    async def click(self) -> None:
        self.clicked.append(self.label)


class FakeRoleList:
    def __init__(self, items: list) -> None:
        self.items = items

    async def all(self) -> list:
        return self.items


class FakeMcpDialog:
    def __init__(self, labels: list[str], clicked: list[str]) -> None:
        self.buttons = {label: FakeMcpDialogControl(label, clicked) for label in labels}

    async def is_visible(self) -> bool:
        return True

    def get_by_role(self, role: str, *, name: str | None = None, exact: bool = False) -> FakeRoleList:
        if role == "heading":
            return FakeRoleList([])
        assert role == "button" and exact
        return FakeRoleList([self.buttons[name]] if name in self.buttons else [])


@pytest.mark.asyncio
async def test_mcp_connect_scopes_to_current_notebook_dialog_without_changing_tab_scope(tmp_path: Path) -> None:
    clicked = []
    dialog = FakeMcpDialog(["Connect separate tabs", "Copy", "Cancel", "Connect"], clicked)
    session = ColabBrowserSession(ColabBrowserOptions(profile_dir=tmp_path))
    session.page = SimpleNamespace(get_by_role=lambda role: FakeRoleList([dialog]) if role == "dialog" else FakeRoleList([]))
    assert await session.connect_current_notebook_mcp_dialog()
    assert clicked == ["Connect"]


@pytest.mark.asyncio
async def test_saved_notebook_command_palette_searches_only_mcp_command(tmp_path: Path) -> None:
    session = ColabBrowserSession(ColabBrowserOptions(profile_dir=tmp_path))
    session.page = SimpleNamespace()
    events = []
    class Search:
        async def fill(self, value: str) -> None:
            events.append(("search", value))
    search = Search()
    async def click(labels: tuple[str, ...], *, timeout_seconds: float = 8) -> bool:
        if not events:
            assert "Commands Show command palette Ctrl+Shift+P" in labels
            events.append(("open", "command palette"))
        else:
            assert "Connect to a local Colab MCP server" in labels
            events.append(("select", "local MCP command"))
        return True
    async def fields(scope: object) -> list:
        return [("combobox", "Search commands", search)]
    session.click_when_visible = click
    session._observed_editable_controls = fields
    await session.open_saved_notebook_mcp_command()
    assert events == [("search", "MCP"), ("select", "local MCP command")]


@pytest.mark.asyncio
@pytest.mark.parametrize('label',['Connect to a local Colab MCP server','Disconnect from the local Colab MCP server'])
async def test_mcp_palette_reuses_search_and_exact_public_text_without_clicking_disconnect(tmp_path,label):
    session=ColabBrowserSession(ColabBrowserOptions(profile_dir=tmp_path))
    clicked,filled=[],[]
    class Text:
        async def is_visible(self):
            return True
        async def click(self):
            clicked.append(label)
    class Search:
        async def fill(self,value):
            filled.append(value)
    async def fields(scope):
        return [('combobox','Search commands',Search())]
    async def no_role(labels,**kwargs):
        assert 'Connect to a local Colab MCP server' in labels
        return False
    session._observed_editable_controls=fields
    session.click_when_visible=no_role
    session.page=SimpleNamespace(get_by_text=lambda text,exact:FakeRoleList([Text()] if text==label else []))
    await session.open_saved_notebook_mcp_command()
    assert filled==['MCP']
    assert clicked==([label] if label.startswith('Connect') else [])


@pytest.mark.asyncio
async def test_input_bundle_selects_enabled_picker_and_requires_current_runtime_ack(tmp_path):
    bundle=tmp_path/'inputs.zip'
    bundle.write_bytes(b'zip')
    sent=[]
    class Field:
        def __init__(self,disabled):
            self.disabled=disabled
        async def is_disabled(self):
            return self.disabled
        async def set_input_files(self,path):
            sent.append((self.disabled,path))
    main=SimpleNamespace()
    frame=SimpleNamespace(url='https://output.googleusercontent.com/frame',locator=lambda selector:FakeRoleList([Field(True),Field(False)]))
    session=ColabBrowserSession(ColabBrowserOptions(profile_dir=tmp_path))
    session.page=SimpleNamespace(frames=[main,frame],main_frame=main)
    assert await session.upload_source_bundle(bundle)
    assert sent==[(False,str(bundle.resolve()))]
    client=FakeClient({'input_bundle_path':str(bundle),'input_bundle_sha256':'a'*64})
    request='AUTORESEARCH_COLAB_UPLOAD_INPUTS '+json.dumps({'project_id':'project-1','runtime_id':'runtime-1'})
    browser=FakeBrowser(request)
    submissions=[]
    async def upload(path):
        submissions.append(path)
        return True
    browser.upload_source_bundle=upload
    worker=ColabBrowserWorker(client,browser)
    worker.runtime_id,worker.attempt_started='runtime-1',datetime.now(timezone.utc)
    await worker._monitor(client.job)
    await worker._monitor(client.job)
    assert submissions==[bundle] and not worker.inputs_uploaded
    browser.outputs='AUTORESEARCH_COLAB_INPUTS_READY '+json.dumps({'project_id':'project-1','runtime_id':'old','sha256':'a'*64})
    await worker._monitor(client.job)
    assert not worker.inputs_uploaded
    browser.outputs=browser.outputs.replace('"old"','"runtime-1"')
    await worker._monitor(client.job)
    assert worker.inputs_uploaded and worker._input_submission is None


@pytest.mark.asyncio
async def test_upload_confirmation_timeout_preserves_runtime_and_never_resubmits():
    client,browser=FakeClient(),FakeBrowser()
    worker=ColabBrowserWorker(client,browser)
    worker.runtime_id,worker.attempt_started='runtime-1',datetime.now(timezone.utc)
    worker._input_submission={'runtime_id':'runtime-1','sha256':'a'*64,'submitted_at':datetime.now(timezone.utc)-timedelta(minutes=4)}
    pauses=[]
    async def manual(reason,*,preserve_runtime=False):
        pauses.append(preserve_runtime)
    worker._manual_observation=manual
    await worker._confirm_input_submission(client.job,'')
    await worker._confirm_input_submission(client.job,'')
    assert pauses==[True] and worker.runtime_id=='runtime-1' and not worker.inputs_uploaded


@pytest.mark.asyncio
async def test_data_restart_submits_only_latest_request_and_ack_is_not_assumed(monkeypatch,tmp_path):
    path=tmp_path/'train-000002.tar'
    path.write_bytes(b'data')
    requests=[dict(project_id='project-1',runtime_id='runtime-1',path=f'train/shards/train-{i:06d}.tar',size_bytes=4,sha256=str(i)*64) for i in (1,2)]
    output='\n'.join('AUTORESEARCH_COLAB_UPLOAD_DATA '+json.dumps(item) for item in requests)
    client,browser=FakeClient(),FakeBrowser(output)
    chosen=[]
    def approved(job,request):
        chosen.append(request['path'])
        return path
    monkeypatch.setattr('autoresearch.services.colab_data.data_upload_path',approved)
    async def upload(selected):
        return True
    browser.upload_data_file=upload
    worker=ColabBrowserWorker(client,browser)
    worker.runtime_id,worker.attempt_started='runtime-1',datetime.now(timezone.utc)
    await worker._monitor(client.job)
    await worker._monitor(client.job)
    assert chosen==[requests[-1]['path']] and worker._data_files_uploaded==set()
    browser.outputs+='\nAUTORESEARCH_COLAB_DATA_READY '+json.dumps(requests[-1])
    await worker._monitor(client.job)
    assert ('runtime-1',requests[-1]['path'],requests[-1]['sha256']) in worker._data_files_uploaded


@pytest.mark.asyncio
@pytest.mark.parametrize("separate_fields", [False, True])
async def test_saved_mcp_dialog_fills_only_observed_connection_fields_privately(tmp_path: Path, capsys: pytest.CaptureFixture[str], separate_fields: bool) -> None:
    clicked = []
    filled = []
    dialog = FakeMcpDialog(["Connect separate tabs", "Cancel", "Connect"], clicked)
    class Field:
        def __init__(self, name: str) -> None:
            self.name = name
        async def fill(self, value: str) -> None:
            filled.append((self.name, value))
    controls = [("textbox", "Proxy token", Field("token")), ("textbox", "Server port", Field("port"))] if separate_fields else [("textbox", "Connection code", Field("code"))]
    session = ColabBrowserSession(ColabBrowserOptions(profile_dir=tmp_path))
    session.page = SimpleNamespace(get_by_role=lambda role: FakeRoleList([dialog]) if role == "dialog" else FakeRoleList([]))
    async def fields(scope: object) -> list:
        assert scope is dialog
        return controls
    session._observed_editable_controls = fields
    assert await session.connect_current_notebook_mcp_dialog(connection_token="private-sentinel", connection_port="12345")
    assert clicked == ["Connect"]
    assert filled == ([("token", "private-sentinel"), ("port", "12345")] if separate_fields else [("code", "private-sentinel&12345")])
    assert "private-sentinel" not in capsys.readouterr().out


@pytest.mark.asyncio
async def test_saved_mcp_dialog_refuses_unknown_fields_without_entering_credentials(tmp_path: Path) -> None:
    clicked = []
    dialog = FakeMcpDialog(["Connect separate tabs", "Connect"], clicked)
    session = ColabBrowserSession(ColabBrowserOptions(profile_dir=tmp_path))
    session.page = SimpleNamespace(get_by_role=lambda role: FakeRoleList([dialog]))
    async def fields(scope: object) -> list:
        return [("textbox", "Connection name", SimpleNamespace())]
    session._observed_editable_controls = fields
    with pytest.raises(ManualColabAction, match="fields could not be identified") as error:
        await session.connect_current_notebook_mcp_dialog(connection_token="private-sentinel", connection_port="12345")
    assert clicked == []
    assert "private-sentinel" not in str(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize('fragment,unknown,accepted',[
    ('mcpProxyToken=private-sentinel&mcpProxyPort=12345',False,True),
    ('mcpProxyToken=wrong&mcpProxyPort=12345',False,False),
    ('mcpProxyToken=private-sentinel&mcpProxyPort=99999',False,False),
    ('',False,False),
    ('mcpProxyToken=private-sentinel&mcpProxyToken=other&mcpProxyPort=12345',False,False),
    ('mcpProxyToken=private-sentinel&mcpProxyPort=12345',True,False)])
async def test_prefilled_saved_mcp_dialog_requires_exact_current_fragment_and_no_unknown_fields(tmp_path,capsys,fragment,unknown,accepted):
    clicked=[]
    dialog=FakeMcpDialog(['Connect separate tabs','Connect'],clicked)
    session=ColabBrowserSession(ColabBrowserOptions(profile_dir=tmp_path))
    session.page=SimpleNamespace(url='https://colab.research.google.com/drive/owned#'+fragment,
                                 get_by_role=lambda role:FakeRoleList([dialog]))
    async def fields(scope):
        return [('textbox','Connection name',SimpleNamespace())] if unknown else []
    session._observed_editable_controls=fields
    if accepted:
        assert await session.connect_current_notebook_mcp_dialog(connection_token='private-sentinel',connection_port='12345')
        assert clicked==['Connect']
    else:
        with pytest.raises(ManualColabAction,match='fields could not be identified') as error:
            await session.connect_current_notebook_mcp_dialog(connection_token='private-sentinel',connection_port='12345')
        assert clicked==[] and 'private-sentinel' not in str(error.value)
    assert 'private-sentinel' not in capsys.readouterr().out


@pytest.mark.asyncio
@pytest.mark.parametrize('dialog_appears',[False,True])
async def test_saved_mcp_auto_dialog_can_race_command_palette_without_false_auth_pause(tmp_path,monkeypatch,capsys,dialog_appears):
    state={'connected':False,'dialog':False,'commands':0,'consents':0}
    client=FakeNotebookMcp()
    client.status=lambda:{'connected':state['connected']}
    connection={'pid':123,'url':COLAB_EMPTY_NOTEBOOK_URL+'#mcpProxyToken=private-sentinel&mcpProxyPort=12345','official_commit':OFFICIAL_MCP_COMMIT}
    monkeypatch.setattr('autoresearch.services.colab_browser.read_private_json',lambda path:connection)
    monkeypatch.setattr('autoresearch.services.colab_browser.ColabMcpSidecarClient',lambda path:client)
    monkeypatch.setattr('autoresearch.colab_browser_cli._process_identity',lambda pid:'live')
    session=ColabBrowserSession(ColabBrowserOptions(profile_dir=tmp_path))
    async def goto(url,**kwargs):
        session.page.url=url
    session.page=SimpleNamespace(url='about:blank',goto=goto)
    async def current_dialog():
        return object() if state['dialog'] else None
    async def open_palette():
        state['commands']+=1
        state['dialog']=dialog_appears
        raise ManualColabAction('Open this saved notebook command palette')
    async def consent(**kwargs):
        state['consents']+=1
        if state['dialog']:
            state['connected']=True
            return True
        return False
    session._current_notebook_mcp_dialog=current_dialog
    session.open_saved_notebook_mcp_command=open_palette
    session.connect_current_notebook_mcp_dialog=consent
    if dialog_appears:
        assert await session._connect_official_notebook_mcp(tmp_path,tmp_path/'state',notebook_url='https://colab.research.google.com/drive/owned') is client
        assert state['connected'] and state['consents']==2
    else:
        with pytest.raises(ManualColabAction,match='command palette'):
            await session._connect_official_notebook_mcp(tmp_path,tmp_path/'state',notebook_url='https://colab.research.google.com/drive/owned')
        assert not state['connected']
    assert state['commands']==1 and 'private-sentinel' not in capsys.readouterr().out


@pytest.mark.asyncio
async def test_stale_mcp_connected_flag_cannot_skip_current_notebook_consent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = ColabBrowserSession(ColabBrowserOptions(profile_dir=tmp_path))
    fake_client = FakeNotebookMcp()
    connection = {"pid": 123, "url": COLAB_EMPTY_NOTEBOOK_URL + "#mcpProxyToken=private&mcpProxyPort=12345", "official_commit": OFFICIAL_MCP_COMMIT}
    monkeypatch.setattr("autoresearch.services.colab_browser.read_private_json", lambda path: connection)
    monkeypatch.setattr("autoresearch.services.colab_browser.ColabMcpSidecarClient", lambda path: fake_client)
    monkeypatch.setattr("autoresearch.colab_browser_cli._process_identity", lambda pid: "live")
    checks = []
    async def dialog_not_yet_rendered() -> bool:
        checks.append("dialog")
        return len(checks) == 3
    async def goto(url: str, *, wait_until: str) -> None:
        assert url == connection["url"] and wait_until == "domcontentloaded"
    async def visible_text() -> str:
        return ""
    session.page = SimpleNamespace(url=COLAB_EMPTY_NOTEBOOK_URL, goto=goto)
    session.connect_current_notebook_mcp_dialog = dialog_not_yet_rendered
    session.visible_text = visible_text
    async def another_notebook(response, *, timeout_seconds=0):
        raise ManualColabAction("Connected cells belong to another tab")
    session.ensure_mcp_registered_notebook = another_notebook
    await session._connect_official_notebook_mcp(tmp_path, tmp_path / "state")
    assert len(checks) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("fresh_navigation", [False, True])
async def test_remembered_grant_requires_owned_cells_after_navigation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prepared: bool, fresh_navigation: bool) -> None:
    job = prepared_mcp_job(tmp_path)
    expected = prepared_notebook_cells(job)
    client = FakeNotebookMcp([
        {"cell_type": kind, "source": source, "id": f"prepared-{index}"}
        for index, (kind, source) in enumerate(expected)
    ]) if prepared else FakeNotebookMcp()
    connection = {"pid": 123, "url": COLAB_EMPTY_NOTEBOOK_URL + "#mcpProxyToken=private&mcpProxyPort=12345", "official_commit": OFFICIAL_MCP_COMMIT}
    monkeypatch.setattr("autoresearch.services.colab_browser.read_private_json", lambda path: connection)
    monkeypatch.setattr("autoresearch.services.colab_browser.ColabMcpSidecarClient", lambda path: client)
    monkeypatch.setattr("autoresearch.colab_browser_cli._process_identity", lambda pid: "live")
    class CurrentCellHosts:
        async def evaluate_all(self, script: str) -> list[str]:
            return ["cell-" + cell["id"] for cell in client.cells]
    navigated = []
    async def goto(url: str, **kwargs: object) -> None:
        assert fresh_navigation
        navigated.append(url)
        session.page.url = url
    session = ColabBrowserSession(ColabBrowserOptions(profile_dir=tmp_path))
    session.page = SimpleNamespace(url="about:blank" if fresh_navigation else connection["url"], goto=goto,
                                   get_by_role=lambda role: FakeRoleList([]),
                                   locator=lambda selector: CurrentCellHosts())
    assert await session._connect_official_notebook_mcp(tmp_path, tmp_path / "state", expected_cells=expected) is client
    assert session._notebook_mcp is client
    assert navigated == ([connection["url"]] if fresh_navigation else [])
    assert all(name == "get_cells" for _, name, _ in client.calls)


@pytest.mark.asyncio
async def test_saved_drive_private_fragment_is_composed_only_for_browser_navigation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeNotebookMcp()
    private_url = COLAB_EMPTY_NOTEBOOK_URL + "#mcpProxyToken=private&mcpProxyPort=12345"
    connection = {"pid": 123, "url": private_url, "official_commit": OFFICIAL_MCP_COMMIT}
    public_url = "https://colab.research.google.com/drive/owned-notebook?authuser=0"
    monkeypatch.setattr("autoresearch.services.colab_browser.read_private_json", lambda path: connection)
    monkeypatch.setattr("autoresearch.services.colab_browser.ColabMcpSidecarClient", lambda path: client)
    monkeypatch.setattr("autoresearch.colab_browser_cli._process_identity", lambda pid: "live")
    session = ColabBrowserSession(ColabBrowserOptions(profile_dir=tmp_path))
    navigated = []
    async def goto(url: str, *, wait_until: str) -> None:
        navigated.append(url)
        session.page.url = url
    async def approved_dialog() -> bool:
        return True
    session.page = SimpleNamespace(url="about:blank", goto=goto)
    session.connect_current_notebook_mcp_dialog = approved_dialog
    await session._connect_official_notebook_mcp(tmp_path, tmp_path / "state", notebook_url=public_url)
    assert navigated == [public_url + "#mcpProxyToken=private&mcpProxyPort=12345"]
    assert public_colab_url(session.page.url) == public_url
    assert connection["url"] == private_url
    assert "private" not in public_url


@pytest.mark.asyncio
async def test_saved_drive_attachment_rejects_a_private_fragment_supplied_in_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    connection = {"pid": 123, "url": COLAB_EMPTY_NOTEBOOK_URL + "#mcpProxyToken=private&mcpProxyPort=12345", "official_commit": OFFICIAL_MCP_COMMIT}
    monkeypatch.setattr("autoresearch.services.colab_browser.read_private_json", lambda path: connection)
    monkeypatch.setattr("autoresearch.services.colab_browser.ColabMcpSidecarClient", lambda path: FakeNotebookMcp())
    monkeypatch.setattr("autoresearch.colab_browser_cli._process_identity", lambda pid: "live")
    session = ColabBrowserSession(ColabBrowserOptions(profile_dir=tmp_path))
    session.page = SimpleNamespace(url="about:blank")
    with pytest.raises(ColabBrowserError, match="public Colab Drive URL"):
        await session._connect_official_notebook_mcp(tmp_path, tmp_path / "state", notebook_url="https://colab.research.google.com/drive/owned#unexpected-token")


@pytest.mark.asyncio
async def test_human_connected_private_notebook_with_unknown_content_is_not_reused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeNotebookMcp([{"cell_type": "code", "source": "private_work()", "id": "private"}])
    connection = {"pid": 123, "url": COLAB_EMPTY_NOTEBOOK_URL + "#mcpProxyToken=private&mcpProxyPort=12345", "official_commit": OFFICIAL_MCP_COMMIT}
    monkeypatch.setattr("autoresearch.services.colab_browser.read_private_json", lambda path: connection)
    monkeypatch.setattr("autoresearch.services.colab_browser.ColabMcpSidecarClient", lambda path: client)
    monkeypatch.setattr("autoresearch.colab_browser_cli._process_identity", lambda pid: "live")
    session = ColabBrowserSession(ColabBrowserOptions(profile_dir=tmp_path))
    session.page = SimpleNamespace(url=connection["url"], get_by_role=lambda role: FakeRoleList([]))
    with pytest.raises(ManualColabAction, match="does not match"):
        await session._connect_official_notebook_mcp(tmp_path, tmp_path / "state", expected_cells=[("code", "prepared_work()")])
    assert all(name == "get_cells" for _, name, _ in client.calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("dialog_labels", [[], [["Connect", "Cancel"]], [["Connect separate tabs", "Connect"], ["Allow", "Cancel"]]])
async def test_mcp_connect_never_clicks_runtime_connect_other_permissions_or_ambiguous_dialogs(tmp_path: Path, dialog_labels: list[list[str]]) -> None:
    clicked = []
    dialogs = [FakeMcpDialog(labels, clicked) for labels in dialog_labels]
    session = ColabBrowserSession(ColabBrowserOptions(profile_dir=tmp_path))
    # A runtime Connect control outside the dialog must never be requested.
    def get_by_role(role: str) -> FakeRoleList:
        assert role == "dialog"
        return FakeRoleList(dialogs)
    session.page = SimpleNamespace(get_by_role=get_by_role)
    assert not await session.connect_current_notebook_mcp_dialog()
    assert clicked == []


@pytest.mark.asyncio
async def test_mcp_cells_must_belong_to_the_notebook_receiving_run_all(tmp_path: Path) -> None:
    session = ColabBrowserSession(ColabBrowserOptions(profile_dir=tmp_path))
    response = {"ok": True, "result": {"data": {"cells": [{"cell_type": "code", "id": "owned-id", "source": ["pass"]}]}}}
    class CellHosts:
        identifiers = ["cell-owned-id"]
        async def evaluate_all(self, script: str) -> list[str]:
            assert "element.attributes" in script
            return self.identifiers
    hosts = CellHosts()
    session.page = SimpleNamespace(locator=lambda selector: hosts if selector == 'colab-code-cell, colab-text-cell, [id^="cell-"]' else None)
    await session.ensure_mcp_registered_notebook(response, timeout_seconds=0)
    hosts.identifiers = ["another-tab-cell-id"]
    with pytest.raises(ManualColabAction, match="another notebook"):
        await session.ensure_mcp_registered_notebook(response, timeout_seconds=0)


@pytest.mark.asyncio
async def test_known_runtime_reconnect_control_must_persist_before_scheduling_retry() -> None:
    client = FakeClient()
    browser = FakeBrowser('\n'.join([marker("starting"), marker("available"), marker("running")]))
    browser.disconnected_control = True
    worker = ColabBrowserWorker(client, browser)
    worker.attempt_started = datetime.now(timezone.utc)
    await worker._monitor(client.job)
    assert client.observations[-1][0] == "running"
    worker._disconnect_seen_at = datetime.now(timezone.utc) - timedelta(seconds=65)
    await worker._monitor(client.job)
    assert client.observations[-1][0] == "disconnected"
    assert worker.attempt_started is None
    assert worker.runtime_id is None


@pytest.mark.asyncio
async def test_unopened_notebook_connect_control_does_not_imply_gpu_failure() -> None:
    client = FakeClient()
    browser = FakeBrowser(text="Connect")
    browser.disconnected_control = True
    worker = ColabBrowserWorker(client, browser)
    worker.attempt_started = datetime.now(timezone.utc)
    await worker._monitor(client.job)
    assert client.observations == []
    assert worker._disconnect_seen_at is None


@pytest.mark.asyncio
@pytest.mark.parametrize('name,expected',[
    ('Connect T4 Connect to a new runtime',True),
    ('Connect T4\nConnect to a new runtime',True),
    ('Connect to a local Colab MCP server',False),('Connect separate tabs',False),
    ('Connect to Google Drive',False),('MCP Connect T4 Connect to a new runtime',False),
    ('Connect T4 Connect to a new runtime extra',False)])
async def test_runtime_connect_control_accepts_only_observed_full_toolbar_label(tmp_path,name,expected):
    session=ColabBrowserSession(ColabBrowserOptions(profile_dir=tmp_path))
    async def controls():
        return [{'role':'button','name':name}]
    session.controls=controls
    assert await session.has_disconnected_runtime_control() is expected
    async def ambiguous():
        return [{'role':'button','name':'Connect T4 Connect to a new runtime'},
                {'role':'button','name':'Connect'}]
    session.controls=ambiguous
    assert not await session.has_disconnected_runtime_control()


def test_ordinary_login_browser_has_only_normal_profile_arguments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    executable = tmp_path / "chrome.exe"
    executable.write_bytes(b"placeholder")
    launched = []
    def popen(command: list, **kwargs: object) -> SimpleNamespace:
        launched.append(command)
        return SimpleNamespace(pid=23456)
    monkeypatch.setattr("autoresearch.colab_browser_cli.subprocess.Popen", popen)
    monkeypatch.setattr("autoresearch.colab_browser_cli._process_identity", lambda pid: "creation-identity")
    assert launch_colab_login_browser(tmp_path, executable_path=executable) == 23456
    assert launched == [[str(executable), f"--user-data-dir={tmp_path / 'data' / 'colab-browser' / 'profile'}", "--no-first-run", "https://colab.research.google.com/"]]
    assert launch_colab_login_browser(tmp_path, executable_path=executable) == 23456
    assert len(launched) == 1
    all_arguments = " ".join(launched[0]).casefold()
    assert not any(flag in all_arguments for flag in ("remote-debugging", "enable-automation", "disable-web-security", "automationcontrolled", "ignore-certificate"))


def test_ordinary_login_browser_rejects_profile_owned_by_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    executable = tmp_path / "chrome.exe"
    executable.write_bytes(b"placeholder")
    called = []
    monkeypatch.setattr("autoresearch.colab_browser_cli.subprocess.Popen", lambda *args, **kwargs: called.append(args))
    profile = tmp_path / "data" / "colab-browser" / "profile"
    with WorkerLock(profile.with_name("profile.lock")):
        with pytest.raises(ColabBrowserError, match="already uses"):
            launch_colab_login_browser(tmp_path, executable_path=executable)
    assert called == []


@pytest.mark.asyncio
@pytest.mark.parametrize("normal_control", ["File", "Open file picker"])
async def test_successful_google_login_resumes_only_explicit_signin_pause(normal_control: str) -> None:
    client = FakeClient({"status": "auth_required", "last_observation": {"status": "auth", "details": {"reason": GOOGLE_SIGN_IN_REASON, "manual_action": True}}})
    browser = FakeBrowser()
    browser.visible_controls = [{"role": "button", "name": normal_control}]
    worker = ColabBrowserWorker(client, browser)
    assert await worker._resume_after_google_login(client.job)
    assert client.resume_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("url,text,controls,reason", [
    ("https://accounts.google.com/v3/signin/challenge", "", [{"role": "menuitem", "name": "File"}], GOOGLE_SIGN_IN_REASON),
    ("https://colab.research.google.com/", "Verify it's you", [{"role": "button", "name": "Open notebook"}], GOOGLE_SIGN_IN_REASON),
    ("https://colab.research.google.com/", "Connect to Google Drive", [{"role": "menuitem", "name": "Runtime"}], GOOGLE_SIGN_IN_REASON),
    ("https://colab.research.google.com/", "", [{"role": "button", "name": "Sign in"}, {"role": "button", "name": "Open notebook"}], GOOGLE_SIGN_IN_REASON),
    ("https://colab.research.google.com/", "", [], GOOGLE_SIGN_IN_REASON),
    ("https://colab.research.google.com/", "", [{"role": "menuitem", "name": "File"}], "Import the notebook using File > Open notebook > Upload"),
    ("https://colab.research.google.com/", "", [{"role": "menuitem", "name": "File"}], "missing_inputs"),
])
async def test_google_login_auto_resume_never_releases_other_pauses(url: str, text: str, controls: list, reason: str) -> None:
    client = FakeClient({"status": "auth_required", "last_observation": {"status": "auth", "details": {"reason": reason, "manual_action": True}}})
    browser = FakeBrowser(url=url, text=text)
    browser.visible_controls = controls
    assert not await ColabBrowserWorker(client, browser)._resume_after_google_login(client.job)
    assert client.resume_count == 0


class TargetClosedError(RuntimeError):
    pass


def test_browser_closed_detector_preserves_nested_error_but_rejects_other_errors() -> None:
    error = ColabBrowserError("Safe startup failure")
    error.__cause__ = TargetClosedError("private authentication URL")
    assert is_browser_closed_error(error)
    assert not is_browser_closed_error(ValueError("an unrelated error"))


@pytest.mark.asyncio
async def test_closed_auth_browser_records_connection_retry_without_traceback(capsys: pytest.CaptureFixture) -> None:
    client = FakeClient({"status": "auth_required"})
    browser = FakeBrowser()
    async def closed_output() -> str:
        raise TargetClosedError("https://accounts.google.com/?secret=must-not-log")
    browser.output_text = closed_output
    worker = ColabBrowserWorker(client, browser)
    worker.attempt_started = datetime.now(timezone.utc)
    await worker.run()
    assert [item[0] for item in client.observations] == ["network"]
    assert client.observations[0][1]["browser_closed"] is True
    logged = capsys.readouterr().out
    assert "secret" not in logged
    assert "Traceback" not in logged


@pytest.mark.asyncio
async def test_browser_closed_during_launch_does_not_repeatedly_claim_attempts() -> None:
    client = FakeClient({"status": "queued", "can_attempt": True})
    browser = FakeBrowser()
    async def closed_open(job: dict) -> None:
        raise TargetClosedError("closed while opening prepared notebook")
    browser.open_notebook = closed_open
    await ColabBrowserWorker(client, browser).run()
    assert client.claim_count == 1
    assert client.observations[-1][0] == "network"


@pytest.mark.asyncio
@pytest.mark.parametrize('late_status,text',[
    ('quota','You cannot currently connect to a GPU due to usage limits in Colab.'),
    ('auth',"Verify it's you")])
async def test_expired_connection_lease_observes_late_provider_denial_once_without_new_gpu_request(monkeypatch,late_status,text):
    client=FakeClient({'status':'waiting_for_connection','can_attempt':False})
    browser=FakeBrowser(text=text)
    original_observe=client.observe
    async def observe(status,**details):
        result=await original_observe(status,**details)
        if status=='quota':
            client.job['status']='waiting_for_gpu'
        return result
    client.observe=observe
    polls=[]
    async def tick(seconds):
        polls.append(1)
        if len(polls)>=3:
            client.job['status']='cancelled'
    monkeypatch.setattr('autoresearch.services.colab_browser.asyncio.sleep',tick)
    await asyncio.wait_for(ColabBrowserWorker(client,browser).run(),timeout=3)
    assert [status for status,_ in client.observations]==[late_status]
    assert client.claim_count==0 and browser.run_all_count==0 and browser.opened==[]


@pytest.mark.asyncio
async def test_closed_browser_does_not_overwrite_cancelled_job() -> None:
    client = FakeClient({"status": "cancelled"})
    browser = FakeBrowser()
    browser.page.is_closed = lambda: True
    await ColabBrowserWorker(client, browser).run()
    assert client.observations == []


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_status", ["quota", "unavailable"])
async def test_closed_browser_preserves_future_gpu_retry_and_provider_observation(provider_status: str, capsys: pytest.CaptureFixture) -> None:
    retry_at = (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat().replace("+00:00", "Z")
    observation = {"status": provider_status, "details": {"provider_ui_text": "Observed GPU provider message"}}
    client = FakeClient({"status": "waiting_for_gpu", "next_attempt_at": retry_at, "provider_retry_not_before": retry_at,
                         "last_observation": observation})
    original_job = dict(client.job)
    browser = FakeBrowser()
    browser.page.is_closed = lambda: True
    await ColabBrowserWorker(client, browser).run()
    assert client.observations == []
    assert client.job == original_job
    assert client.job["last_observation"] is observation
    assert "retry deadline" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_closed_browser_after_expired_gpu_wait_records_connection_retry() -> None:
    retry_at = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    client = FakeClient({"status": "waiting_for_gpu", "next_attempt_at": retry_at})
    browser = FakeBrowser()
    browser.page.is_closed = lambda: True
    await ColabBrowserWorker(client, browser).run()
    assert client.observations[-1][0] == "network"
    assert client.observations[-1][1]["browser_closed"] is True


@pytest.mark.asyncio
async def test_browser_closed_during_startup_persists_network_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    client = FakeClient({"status": "queued"})
    class ClosedSession:
        def __init__(self, options: object) -> None:
            pass
        async def __aenter__(self) -> None:
            raise ColabBrowserError("Safe browser startup failure") from TargetClosedError("private token")
        async def __aexit__(self, *args: object) -> None:
            pass
    monkeypatch.setattr("autoresearch.colab_browser_cli.ColabJobClient", lambda *args: client)
    monkeypatch.setattr("autoresearch.colab_browser_cli.ColabBrowserSession", ClosedSession)
    args = SimpleNamespace(project="project-1", root=tmp_path, profile_dir=None, browser="auto", executable_path=None, api_url="http://127.0.0.1:8765")
    await run_colab_browser_cli(args)
    assert client.observations[-1][0] == "network"
    assert client.observations[-1][1]["browser_closed"] is True
    assert "private token" not in capsys.readouterr().out


def test_public_accessibility_snapshot_exposes_real_picker_tab_names() -> None:
    snapshot = '\n'.join([
        '- dialog "Open notebook":',
        '  - tablist:',
        '    - tab "Recent" [selected]',
        '    - tab "Upload"',
        '  - button "Select a file"',
        '  - textbox "Email": someone@example.com',
    ])
    assert parse_aria_controls(snapshot) == [
        {"role": "tab", "name": "Recent"},
        {"role": "tab", "name": "Upload"},
        {"role": "button", "name": "Select a file"},
        {"role": "textbox", "name": "Email"},
    ]


@pytest.mark.parametrize("url,expected", [
    ("https://colab.research.google.com/drive/example", True),
    ("https://docs.google.com/picker?key=not-read", True),
    ("https://drive.google.com/u/0/picker", True),
    ("https://docs.google.com/document/d/example", False),
    ("https://accounts.google.com/signin", False),
    ("https://cell-output.googleusercontent.com/", False),
    ("https://docs.google.com.evil/picker", False),
    ("http://docs.google.com/picker", False),
])
def test_picker_frame_trust_does_not_include_output_or_auth_frames(url: str, expected: bool) -> None:
    assert is_trusted_picker_frame(url) is expected


@pytest.mark.asyncio
async def test_normal_upload_control_is_clicked_in_its_observed_picker_frame(tmp_path: Path) -> None:
    clicks = []
    class Candidate:
        async def is_visible(self) -> bool:
            return True
        async def is_disabled(self) -> bool:
            return False
        async def click(self) -> None:
            clicks.append("picker-upload")
    class Locator:
        def __init__(self, snapshot: str = "") -> None:
            self.snapshot = snapshot
        async def aria_snapshot(self, **kwargs: object) -> str:
            return self.snapshot
        async def all(self) -> list:
            return [Candidate()]
    class Frame:
        def __init__(self, url: str, snapshot: str) -> None:
            self.url = url
            self.snapshot = snapshot
        def locator(self, selector: str) -> Locator:
            return Locator(self.snapshot)
        def get_by_role(self, role: str, *, name: str, exact: bool) -> Locator:
            assert self.url.startswith("https://docs.google.com/picker")
            assert (role, name, exact) == ("tab", "Upload", True)
            return Locator()
    main = Frame("https://colab.research.google.com/", '- button "Open file picker"\n- link "Private experiment.ipynb"')
    picker = Frame("https://docs.google.com/picker?token=never-log", '- tab "Upload"')
    output = Frame("https://cell.googleusercontent.com/?token=never-log", '- button "Upload"')
    session = ColabBrowserSession(ColabBrowserOptions(profile_dir=tmp_path))
    session.page = SimpleNamespace(frames=[main, picker, output], main_frame=main)
    async def no_dom_controls(frame: object) -> list:
        return []
    session._dom_controls = no_dom_controls
    assert await session.click_observed(("Upload", "上传"))
    assert clicks == ["picker-upload"]
    diagnostics = await session.ui_diagnostics()
    encoded = json.dumps(diagnostics)
    assert "never-log" not in encoded
    assert "Private experiment.ipynb" not in encoded
    assert '- tab "Upload"' in diagnostics["ui_role_tree"]
    assert {"host": "docs.google.com", "path": "/picker"} in diagnostics["ui_frames"]


@pytest.mark.asyncio
async def test_paused_import_diagnostics_refresh_only_after_ui_changes() -> None:
    client = FakeClient({"status": "auth_required", "last_observation": {"status": "auth", "details": {"manual_action": True, "reason": "Import the notebook using File > Open notebook > Upload"}}})
    browser = FakeBrowser()
    controls = [{"role": "button", "name": "Open file picker"}]
    async def diagnostics() -> dict:
        return {"ui_controls": list(controls), "ui_role_tree": "", "ui_frames": []}
    browser.ui_diagnostics = diagnostics
    worker = ColabBrowserWorker(client, browser)
    await worker._refresh_import_diagnostics(client.job)
    assert len(client.observations) == 1
    worker._last_ui_diagnostics_at = datetime.now(timezone.utc) - timedelta(seconds=60)
    await worker._refresh_import_diagnostics(client.job)
    assert len(client.observations) == 1
    controls.append({"role": "tab", "name": "Upload"})
    worker._last_ui_diagnostics_at = datetime.now(timezone.utc) - timedelta(seconds=60)
    await worker._refresh_import_diagnostics(client.job)
    assert len(client.observations) == 2


def test_data_upload_markers_require_complete_output_lines_and_runtime():
    request = dict(project_id="project-1", runtime_id="runtime-1", path="train/shards/train-000001.tar",
                   size_bytes=1024, sha256="a" * 64)
    prefix = "AUTORESEARCH_COLAB_UPLOAD_DATA "
    good = prefix + json.dumps(request)
    invalid = ["echo " + good, prefix + "{broken}",
               prefix + json.dumps({**request, "project_id": "other"}),
               prefix + json.dumps({**request, "runtime_id": ""}),
               prefix + json.dumps({**request, "size_bytes": True}),
               prefix + json.dumps({**request, "size_bytes": 512 * 1024 * 1024 + 1})]
    assert parse_data_upload_requests("\n".join([*invalid, good]), "project-1") == [request]


@pytest.mark.asyncio
async def test_recovered_running_notebook_is_monitored_without_run_all():
    started = datetime.now(timezone.utc) - timedelta(hours=2)
    client = FakeClient({"status": "disconnected"})
    browser = FakeBrowser("\n".join([marker("available", started=started), marker("running", started=started)]))
    worker = ColabBrowserWorker(client, browser)
    worker.runtime_id = "runtime-1"
    worker.attempt_started = started
    worker.running_reported = True
    worker.available_reported = True
    worker._recovering_runtime = True
    await worker._monitor(client.job)
    assert [status for status, _ in client.observations] == ["available", "running"]
    assert browser.run_all_count == 0
    assert worker.runtime_id == "runtime-1"


@pytest.mark.asyncio
async def test_recovery_does_not_trust_historical_running_output_when_disconnected():
    started = datetime.now(timezone.utc) - timedelta(hours=2)
    client = FakeClient({"status": "disconnected"})
    browser = FakeBrowser("\n".join([marker("available", started=started), marker("running", started=started)]))
    browser.disconnected_control = True
    worker = ColabBrowserWorker(client, browser)
    worker.runtime_id, worker.attempt_started = "runtime-1", started
    worker._recovering_runtime = True
    await worker._monitor(client.job)
    assert client.observations == []
    worker._disconnect_seen_at = datetime.now(timezone.utc) - timedelta(seconds=61)
    await worker._monitor(client.job)
    assert [status for status, _ in client.observations] == ["disconnected"]
    assert worker.runtime_id is None and worker.attempt_started is None
    assert browser.run_all_count == 0


@pytest.mark.asyncio
async def test_full_data_uploads_ignore_other_runtimes_and_do_not_repeat(tmp_path, monkeypatch):
    request = dict(project_id="project-1", runtime_id="runtime-1", path="train/shards/train-000001.tar",
                   size_bytes=1024, sha256="a" * 64)
    client = FakeClient()
    browser = FakeBrowser("AUTORESEARCH_COLAB_UPLOAD_DATA " + json.dumps({**request, "runtime_id": "old-runtime"}))
    uploads = []
    async def upload(path):
        uploads.append(path)
        return True
    browser.upload_data_file = upload
    approved = tmp_path / "approved.tar"
    monkeypatch.setattr("autoresearch.services.colab_data.data_upload_path", lambda job, value: approved)
    worker = ColabBrowserWorker(client, browser)
    worker.runtime_id, worker.attempt_started = "runtime-1", datetime.now(timezone.utc)
    await worker._monitor(client.job)
    assert uploads == []
    browser.outputs = "AUTORESEARCH_COLAB_UPLOAD_DATA " + json.dumps(request)
    await worker._monitor(client.job)
    await worker._monitor(client.job)
    assert uploads == [approved]


@pytest.mark.asyncio
async def test_full_data_upload_integrity_failure_stops_before_transmission(monkeypatch):
    request = dict(project_id="project-1", runtime_id="runtime-1", path="train/shards/train-000001.tar",
                   size_bytes=1024, sha256="a" * 64)
    client = FakeClient()
    browser = FakeBrowser("AUTORESEARCH_COLAB_UPLOAD_DATA " + json.dumps(request))
    def invalid(job, value):
        raise ValueError("checksum changed")
    monkeypatch.setattr("autoresearch.services.colab_data.data_upload_path", invalid)
    worker = ColabBrowserWorker(client, browser)
    worker.runtime_id, worker.attempt_started = "runtime-1", datetime.now(timezone.utc)
    await worker._monitor(client.job)
    assert client.observations[0][0] == "failed"


@pytest.mark.asyncio
async def test_network_read_failure_preserves_runtime_for_read_only_retry(monkeypatch):
    started = datetime.now(timezone.utc) - timedelta(hours=2)
    client = FakeClient({"status": "running", "runtime_id": "runtime-1",
                         "last_attempt_at": started.isoformat()})
    browser = FakeBrowser()
    reads = []
    async def output():
        reads.append(1)
        if len(reads) == 1:
            raise ConnectionError("temporary local bridge interruption")
        return "\n".join([marker("starting", started=started), marker("available", started=started),
                           marker("running", started=started), marker("completed", started=started, returncode=0)])
    browser.output_text = output
    original_observe = client.observe
    async def observe(status, **details):
        result = await original_observe(status, **details)
        if status == "network":
            assert details["runtime_id"] == "runtime-1"
            assert details["runtime_active"] is True
            assert details["started_at"] == started.isoformat()
            client.job["can_attempt"] = True
        return result
    client.observe = observe
    async def no_sleep(seconds):
        pass
    monkeypatch.setattr("autoresearch.services.colab_browser.asyncio.sleep", no_sleep)
    worker = ColabBrowserWorker(client, browser)
    await asyncio.wait_for(worker.run(), timeout=3)
    assert client.observations[-1][0] == "completed"
    assert client.claim_count == 1
    assert browser.run_all_count == 0
    assert len(browser.opened) == 2


@pytest.mark.asyncio
async def test_mcp_consent_recognizes_observed_dialog_accessible_title(tmp_path):
    clicked = []
    dialog = FakeMcpDialog(["Connect separate tabs expand_more", "Cancel", "Connect"], clicked)
    async def dialog_title(script):
        assert "aria-label" in script and "aria-labelledby" in script
        return "Connect to a local Colab MCP server"
    dialog.evaluate = dialog_title
    session = ColabBrowserSession(ColabBrowserOptions(profile_dir=tmp_path))
    session.page = SimpleNamespace(get_by_role=lambda role: FakeRoleList([dialog]) if role == "dialog" else FakeRoleList([]))
    assert await session.connect_current_notebook_mcp_dialog()
    assert clicked == ["Connect"]


@pytest.mark.asyncio
async def test_other_named_permission_dialog_is_never_mcp_consent(tmp_path):
    clicked = []
    dialog = FakeMcpDialog(["Cancel", "Connect"], clicked)
    async def dialog_title(script):
        return "Connect to Google Drive"
    dialog.evaluate = dialog_title
    session = ColabBrowserSession(ColabBrowserOptions(profile_dir=tmp_path))
    session.page = SimpleNamespace(get_by_role=lambda role: FakeRoleList([dialog]) if role == "dialog" else FakeRoleList([]))
    assert not await session.connect_current_notebook_mcp_dialog()
    assert clicked == []
