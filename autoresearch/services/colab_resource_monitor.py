"""Read-only, persistent observer of the authorized Colab allocation workflow.

This agent does not request compute, change retry deadlines, or write job state.
The ordinary browser worker remains the only source of GPU observations, while
the API scheduler owns allocations and provider cooldowns.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import httpx

from ..colab_browser_cli import WorkerLock, _process_identity
from .colab_browser import ColabBrowserError, validate_local_api_url, validate_project_id


UTC = timezone.utc
AGENT_NAME = "Colab资源检查Agent"
TERMINAL = {"completed", "failed", "cancelled", "blocked"}
JOB_STATUSES = TERMINAL | {
    "not_prepared", "queued", "waiting_for_gpu", "waiting_for_connection",
    "disconnected", "attempting", "auth_required", "gpu_available", "running",
}
RESULTS = {"available", "unavailable", "quota", "network", "auth", "unknown"}
AGENT_STATES = {"watching", "checking", "quota_wait", "auth_required", "stopped", "offline"}
MESSAGES = {
    "watching": "资源检查 Agent 持续监视；到达重试时间后由工作器申请 GPU。",
    "checking": "工作器正在通过 Colab 正常界面检查并申请免费 GPU。",
    "available": "已确认当前工作器使用免费 GPU，正在运行或准备试验。",
    "quota_wait": "最近一次检查遇到 GPU 额度限制，正在遵守 Colab 冷却时间。",
    "auth_required": "需要在 Colab 浏览器中完成登录或授权。",
    "stopped": "资源检查 Agent 已停止；任务已结束或自动运行已取消。",
    "offline": "资源检查 Agent 暂时无法连接本地服务，当前资源状态未知。",
    "unavailable": "最近一次检查没有可用 GPU，正在等待下一次随机重试。",
    "network": "最近一次连接未成功，当前 GPU 是否可用尚未确认。",
}


def utc_now() -> datetime:
    return datetime.now(UTC)


def timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def parsed_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or len(value) > 64:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.astimezone(UTC) if parsed.tzinfo is not None else None
    except (ValueError, OverflowError):
        return None


def safe_timestamp(value: Any) -> str | None:
    parsed = parsed_timestamp(value)
    return timestamp(parsed) if parsed is not None else None


def _linked(path: Path) -> bool:
    return path.is_symlink() or bool(getattr(path, "is_junction", lambda: False)())


def _safe_path(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if candidate.exists() and _linked(candidate):
            raise ValueError("Resource monitor paths must not be linked")
    return path


def monitor_directory(root: Path) -> Path:
    return _safe_path(root.resolve(strict=True) / "data" / "colab-resource-monitor")


def public_status_path(project_id: str) -> Path:
    project_id = validate_project_id(project_id)
    return _safe_path(Path(__file__).resolve().parents[1] / "static" / "colab-resource-status" / f"{project_id}.json")


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    _safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if len(data.encode("utf-8")) > 32_000:
        raise ValueError("Resource monitor state exceeds the size limit")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        _safe_path(path)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_json(path: Path) -> dict[str, Any]:
    _safe_path(path)
    if path.stat().st_size > 32_000:
        raise ValueError("Resource monitor state exceeds the size limit")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Invalid resource monitor state")
    return value


def resource_monitor_is_running(root: Path, project_id: str) -> int | None:
    project_id = validate_project_id(project_id)
    try:
        state = _read_json(monitor_directory(root) / f"{project_id}.worker.json")
        if state.get("project_id") != project_id:
            return None
        pid = int(state["pid"])
        identity = _process_identity(pid)
        return pid if identity is not None and identity == state.get("process_identity") else None
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _bounded_minutes(value: Any, default: int) -> int:
    return value if type(value) is int and 1 <= value <= 1440 else default


def build_resource_snapshot(
    project_id: str,
    job: dict[str, Any] | None,
    *,
    now: datetime | None = None,
    previous: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Only actual browser observations can supply the last-check timestamp."""
    project_id = validate_project_id(project_id)
    now = now or utc_now()
    previous = previous or {}
    online = isinstance(job, dict) and job.get("project_id") == project_id
    source = job if online else {}
    status = source.get("status")
    status = status if isinstance(status, str) and status in JOB_STATUSES else "unknown"
    observation = source.get("last_observation")
    observation = observation if isinstance(observation, dict) else {}
    observed_status = observation.get("status")
    observed_status = observed_status if isinstance(observed_status, str) else "unknown"
    result = {"running": "available", "disconnected": "network"}.get(observed_status, observed_status)
    result = result if result in RESULTS else "unknown"
    last_checked = safe_timestamp(observation.get("observed_at")) if observation else None
    if not online:
        previous_result = previous.get("last_result")
        result = previous_result if isinstance(previous_result, str) and previous_result in RESULTS else "unknown"
        last_checked = safe_timestamp(previous.get("last_checked_at"))
    checked_time = parsed_timestamp(last_checked)
    fresh = bool(checked_time and 0 <= (now - checked_time).total_seconds() <= 600)
    next_check = safe_timestamp(source.get("next_attempt_at")) if online else safe_timestamp(previous.get("next_check_at"))
    worker_running = online and source.get("worker_running") is True
    details = observation.get("details")
    details = details if isinstance(details, dict) else {}
    current_runtime = source.get("runtime_id")
    observed_runtime = details.get("runtime_id")
    runtime_matches = bool(
        isinstance(current_runtime, str) and current_runtime
        and isinstance(observed_runtime, str) and current_runtime == observed_runtime
    )
    active_gpu = bool(
        online and worker_running and status in {"gpu_available", "running"}
        and observed_status in {"available", "running"} and checked_time is not None
        and fresh and runtime_matches
    )
    stopped = online and (
        status in TERMINAL or not source.get("prepared") or source.get("autorun_enabled") is not True
    )
    if not online:
        state = "offline"
    elif stopped:
        state = "stopped"
    elif status == "auth_required":
        state = "auth_required"
    elif status == "attempting":
        state = "checking"
    elif result == "quota" and status == "waiting_for_gpu":
        state = "quota_wait"
    else:
        state = "watching"
    # A historical denial cannot establish the inventory at the current moment.
    recent_denial = bool(
        online and not stopped and status == "waiting_for_gpu" and result == "unavailable"
        and checked_time and 0 <= (now - checked_time).total_seconds() <= 45
    )
    availability = "available" if active_gpu and not stopped else "unavailable" if recent_denial else "unknown"
    message_key = "available" if availability == "available" else state
    if state == "watching" and result in {"unavailable", "network"}:
        message_key = result
    return {
        "project_id": project_id,
        "agent_name": AGENT_NAME,
        "agent_state": state,
        "agent_running": state != "stopped",
        "heartbeat_at": timestamp(now),
        "availability": availability,
        "last_result": result,
        "last_checked_at": last_checked,
        "last_check_fresh": fresh,
        "next_check_at": next_check,
        "job_status": status,
        "worker_running": bool(worker_running),
        "message": MESSAGES[message_key],
        "retry_min_minutes": _bounded_minutes(source.get("retry_min_minutes", previous.get("retry_min_minutes")), 5),
        "retry_max_minutes": _bounded_minutes(source.get("retry_max_minutes", previous.get("retry_max_minutes")), 10),
    }


def resource_monitor_status(root: Path, project_id: str) -> dict[str, Any]:
    project_id = validate_project_id(project_id)
    try:
        previous = _read_json(monitor_directory(root) / f"{project_id}.status.json")
    except (OSError, ValueError, TypeError):
        previous = {}
    # Rebuild through the whitelist rather than trusting arbitrary saved fields.
    snapshot = build_resource_snapshot(project_id, None, previous=previous)
    heartbeat = parsed_timestamp(previous.get("heartbeat_at"))
    state = previous.get("agent_state")
    state = state if isinstance(state, str) else "unknown"
    live = resource_monitor_is_running(root, project_id) is not None
    fresh = bool(heartbeat and 0 <= (utc_now() - heartbeat).total_seconds() <= 45)
    if previous.get("project_id") == project_id and live and fresh and state in AGENT_STATES:
        # Publish only the controlled snapshot contract; no provider strings or URLs.
        for key in ("agent_state", "job_status", "worker_running"):
            if key in previous:
                if key == "agent_state" and previous[key] in AGENT_STATES:
                    snapshot[key] = previous[key]
                elif key == "job_status" and isinstance(previous[key], str) and previous[key] in JOB_STATUSES:
                    snapshot[key] = previous[key]
                elif key == "worker_running" and type(previous[key]) is bool:
                    snapshot[key] = previous[key]
        positive = (previous.get("availability") == "available" and snapshot["worker_running"]
                    and snapshot["job_status"] in {"gpu_available", "running"}
                    and snapshot["last_check_fresh"] and snapshot["last_result"] == "available")
        checked = parsed_timestamp(snapshot["last_checked_at"])
        negative = (previous.get("availability") == "unavailable"
                    and snapshot["job_status"] == "waiting_for_gpu" and snapshot["last_result"] == "unavailable"
                    and checked is not None and 0 <= (utc_now() - checked).total_seconds() <= 45)
        snapshot["availability"] = "available" if positive else "unavailable" if negative else "unknown"
        message_key = "available" if snapshot["availability"] == "available" else snapshot["agent_state"]
        if snapshot["agent_state"] == "watching" and snapshot["last_result"] in {"unavailable", "network"}:
            message_key = snapshot["last_result"]
        snapshot["message"] = MESSAGES[message_key]
        snapshot["agent_running"] = snapshot["agent_state"] != "stopped"
    elif state == "stopped" and previous.get("project_id") == project_id:
        snapshot.update(agent_state="stopped", agent_running=False, message=MESSAGES["stopped"])
    else:
        snapshot["agent_running"] = False
    snapshot["heartbeat_at"] = safe_timestamp(previous.get("heartbeat_at"))
    snapshot["agent_heartbeat_fresh"] = live and fresh
    return snapshot


def launch_resource_monitor(
    root: Path, project_id: str, *, api_url: str, browser: str = "chrome",
) -> int:
    """Start one hidden observer, or hand an existing observer the current API."""
    project_id = validate_project_id(project_id)
    api_url = validate_local_api_url(api_url)
    if browser not in {"auto", "chrome", "edge"}:
        raise ValueError("browser must be auto, chrome, or edge")
    root = root.resolve(strict=True)
    directory = monitor_directory(root)
    directory.mkdir(parents=True, exist_ok=True)
    with WorkerLock(directory / f"{project_id}.launch.lock"):
        atomic_json(directory / f"{project_id}.control.json", {
            "project_id": project_id, "api_url": api_url, "browser": browser,
        })
        existing = resource_monitor_is_running(root, project_id)
        if existing is not None:
            return existing
        environment = os.environ.copy()
        environment.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        command = [sys.executable, "-u", "-m", "autoresearch.colab_resource_monitor_cli",
                   "--project", project_id, "--root", str(root), "--api-url", api_url,
                   "--browser", browser]
        log_path = _safe_path(directory / f"{project_id}.log")
        with log_path.open("ab") as log_file:
            process = subprocess.Popen(
                command, cwd=str(root), stdin=subprocess.DEVNULL, stdout=log_file,
                stderr=subprocess.STDOUT, env=environment,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0,
            )
        atomic_json(directory / f"{project_id}.worker.json", {
            "project_id": project_id, "pid": process.pid,
            "process_identity": _process_identity(process.pid),
        })
        return process.pid


class ResourceMonitorClient:
    def __init__(self, api_url: str, project_id: str) -> None:
        self.project_id = validate_project_id(project_id)
        self.api_url = validate_local_api_url(api_url)
        self.client = httpx.AsyncClient(base_url=self.api_url, timeout=8, trust_env=False)

    async def status(self) -> dict[str, Any]:
        response = await self.client.get(f"/api/projects/{self.project_id}/colab")
        response.raise_for_status()
        if len(response.content) > 2_000_000:
            raise ValueError("Colab job response exceeds the size limit")
        job = response.json()
        if not isinstance(job, dict) or job.get("project_id") != self.project_id:
            raise ValueError("Unexpected Colab project response")
        return job

    async def close(self) -> None:
        await self.client.aclose()


class ColabResourceMonitor:
    def __init__(
        self, root: Path, project_id: str, *, api_url: str, browser: str = "chrome",
        poll_seconds: float = 10, public_path: Path | None = None,
        client_factory: Callable[[str, str], Any] = ResourceMonitorClient,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self.root = root.resolve(strict=True)
        self.project_id = validate_project_id(project_id)
        self.api_url = validate_local_api_url(api_url)
        if browser not in {"auto", "chrome", "edge"}:
            raise ValueError("browser must be auto, chrome, or edge")
        if not 5 <= poll_seconds <= 60:
            raise ValueError("Resource monitor polling must be between 5 and 60 seconds")
        self.browser = browser
        self.poll_seconds = poll_seconds
        self.directory = monitor_directory(self.root)
        self.public_path = _safe_path(public_path) if public_path is not None else public_status_path(self.project_id)
        self.client_factory = client_factory
        self.clock = clock
        self.client: Any = None
        self.previous: dict[str, Any] = {}

    async def _current_client(self) -> Any:
        try:
            control = _read_json(self.directory / f"{self.project_id}.control.json")
            if control.get("project_id") == self.project_id:
                api_url = validate_local_api_url(control.get("api_url", ""))
                if api_url != self.api_url:
                    if self.client is not None:
                        await self.client.close()
                    self.client = None
                    self.api_url = api_url
        except (OSError, ValueError, TypeError):
            pass
        if self.client is None:
            self.client = self.client_factory(self.api_url, self.project_id)
        return self.client

    def publish(self, snapshot: dict[str, Any]) -> None:
        atomic_json(self.directory / f"{self.project_id}.status.json", snapshot)
        atomic_json(self.public_path, snapshot)
        self.previous = snapshot

    async def tick(self) -> dict[str, Any]:
        client = await self._current_client()
        try:
            job = await client.status()
        except (httpx.HTTPError, OSError, ValueError, TypeError):
            job = None
        snapshot = build_resource_snapshot(self.project_id, job, now=self.clock(), previous=self.previous)
        self.publish(snapshot)
        return snapshot

    async def run(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        with WorkerLock(self.directory / f"{self.project_id}.lock"):
            # Windows virtual-environment launchers can use a different PID
            # from the actual Python interpreter. Let the parent finish its
            # launch metadata before recording the long-lived interpreter.
            while True:
                try:
                    with WorkerLock(self.directory / f"{self.project_id}.launch.lock"):
                        atomic_json(self.directory / f"{self.project_id}.worker.json", {
                            "project_id": self.project_id, "pid": os.getpid(),
                            "process_identity": _process_identity(os.getpid()),
                        })
                    break
                except ColabBrowserError:
                    await asyncio.sleep(0.1)
            try:
                while True:
                    snapshot = await self.tick()
                    if snapshot["agent_state"] == "stopped":
                        return
                    await asyncio.sleep(self.poll_seconds)
            finally:
                if self.client is not None:
                    await self.client.close()
