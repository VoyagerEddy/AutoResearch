from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Literal

import httpx


PLATFORM_TUNNELS_URL = "https://platform.openai.com/settings/organization/tunnels"
CHATGPT_PLUGINS_URL = "https://chatgpt.com/plugins"
LOCAL_MCP_URL = "http://127.0.0.1:8765/mcp"


def _state_dir(root: Path) -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
    return (
        Path(local_app_data) / "AutoResearch"
        if local_app_data
        else root / ".local" / "AutoResearch"
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        # Windows PowerShell 5 writes UTF-8 with a BOM by default.  Accept both
        # the legacy files and the no-BOM files written by the current helper.
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _process_is_running(pid: object) -> bool:
    try:
        if isinstance(pid, bool) or not isinstance(pid, (int, str)):
            return False
        numeric_pid = int(pid)
        if not 0 < numeric_pid <= 0xFFFFFFFF:
            return False
        if os.name == "nt":
            return _windows_process_is_running(numeric_pid)
        os.kill(numeric_pid, 0)
        return True
    except PermissionError:
        # POSIX EPERM means that a process exists but belongs to another user.
        return True
    except (OSError, TypeError, ValueError):
        return False


def _windows_process_is_running(pid: int) -> bool:
    # Unlike POSIX, os.kill(pid, 0) calls TerminateProcess on Windows.
    # Open a handle with only SYNCHRONIZE rights and check it without waiting.
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
    if not handle:
        return False
    try:
        return kernel32.WaitForSingleObject(handle, 0) == 0x00000102  # WAIT_TIMEOUT
    finally:
        kernel32.CloseHandle(handle)


def _tunnel_health() -> tuple[bool, bool]:
    """Read only the fixed loopback admin endpoints, without ambient proxies."""
    checks = []
    with httpx.Client(timeout=0.4, trust_env=False, follow_redirects=False) as client:
        for endpoint, expected in (
            ("healthz", {"live", "healthy", "ok"}),
            ("readyz", {"ready", "ok"}),
        ):
            try:
                response = client.get(f"http://127.0.0.1:8080/{endpoint}")
                payload = response.json()
                checks.append(
                    response.status_code == 200
                    and isinstance(payload, dict)
                    and payload.get("status") in expected
                )
            except (httpx.HTTPError, ValueError, TypeError):
                checks.append(False)
    return checks[0], checks[1]


def connection_status(root: Path) -> dict[str, Any]:
    root = root.resolve()
    state_dir = _state_dir(root)
    config = _load_json(state_dir / "tunnel.json")
    runtime = _load_json(state_dir / "tunnel-status.json")

    configured_client = Path(str(config.get("tunnel_client", "")))
    bundled_client = root / "tools" / "tunnel-client.exe"
    executable = shutil.which("tunnel-client")
    client_installed = bool(
        (configured_client.is_file() if str(configured_client) not in {"", "."} else False)
        or bundled_client.is_file()
        or executable
    )
    configured = bool(
        config.get("profile")
        and config.get("tunnel_id")
        and (state_dir / "tunnel-api-key.dpapi").is_file()
    )
    process_running = configured and _process_is_running(runtime.get("pid"))
    healthy, ready = _tunnel_health() if process_running else (False, False)
    tunnel_running = process_running and healthy and ready
    tunnel_status = (
        "unconfigured" if not configured else
        "stopped" if not process_running else
        "ready" if tunnel_running else "connecting"
    )
    tunnel_id = str(config.get("tunnel_id", ""))

    return {
        "local_mcp_ready": True,
        "local_mcp_url": LOCAL_MCP_URL,
        "tunnel_client_installed": client_installed,
        "tunnel_configured": configured,
        "tunnel_running": tunnel_running,
        "tunnel_process_running": process_running,
        "tunnel_healthy": healthy,
        "tunnel_ready": tunnel_running,
        "tunnel_status": tunnel_status,
        "profile": str(config.get("profile", "autoresearch")),
        "tunnel_id_hint": f"...{tunnel_id[-6:]}" if len(tunnel_id) > 6 else tunnel_id,
        "setup_script": str(root / "setup-chatgpt.cmd"),
        "start_script": str(root / "start-chatgpt.cmd"),
        "platform_tunnels_url": PLATFORM_TUNNELS_URL,
        "chatgpt_plugins_url": CHATGPT_PLUGINS_URL,
        "manual_chatgpt_step_required": True,
    }


def launch_tunnel_console(root: Path, action: Literal["Setup", "Run", "Doctor"]) -> None:
    root = root.resolve()
    script = root / "chatgpt-tunnel.ps1"
    if not script.is_file():
        raise OSError("ChatGPT tunnel helper is missing")
    if os.name != "nt":
        raise OSError("The bundled tunnel helper currently supports Windows only")

    creation_flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
    subprocess.Popen(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-Action",
            action,
        ],
        cwd=str(root),
        creationflags=creation_flags,
    )
