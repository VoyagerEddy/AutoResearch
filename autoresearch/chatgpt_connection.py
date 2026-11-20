from __future__ import annotations

import json
import hashlib
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx


PLATFORM_TUNNELS_URL = "https://platform.openai.com/settings/organization/tunnels"
CHATGPT_PLUGINS_URL = "https://chatgpt.com/plugins"
LOCAL_MCP_URL = "http://127.0.0.1:8765/mcp"
REQUIRED_MCP_TOOLS = frozenset({
    "get_research_status", "get_colab_status", "get_colab_resources", "open_colab_login",
    "prepare_colab_experiment", "start_colab_experiment", "resume_colab_experiment",
    "bind_colab_notebook", "get_colab_experiment_result", "save_research_plan",
    "get_research_workflow", "record_research_decision",
})


def _candidate_ports(root: Path, preferred_port: int | None = None) -> list[int]:
    value: object = preferred_port
    if value is None:
        value = os.environ.get("AUTORESEARCH_PORT")
    if value is None:
        # Read only the public port setting; no environment values are returned.
        try:
            for line in (root / ".env").read_text(encoding="utf-8-sig").splitlines():
                match = re.fullmatch(r"\s*AUTORESEARCH_PORT\s*=\s*[\"']?(\d+)[\"']?\s*(?:#.*)?", line)
                if match:
                    value = match.group(1)
                    break
        except OSError:
            pass
    try:
        preferred = int(value) if not isinstance(value, bool) else 8765
    except (TypeError, ValueError):
        preferred = 8765
    if not 1024 <= preferred <= 65525:
        preferred = 8765
    return list(dict.fromkeys([preferred, 8765, 8766, 8767, *range(preferred + 1, preferred + 11)]))


def _same_root(value: object, root: Path) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        candidate = Path(value)
        return candidate.is_absolute() and os.path.normcase(str(candidate.resolve())) == os.path.normcase(str(root.resolve()))
    except (OSError, ValueError):
        return False


def _local_mcp_url(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = urlsplit(value)
        if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path != "/mcp" or parsed.port is None or not 1024 <= parsed.port <= 65535):
            return None
        return f"http://127.0.0.1:{parsed.port}/mcp"
    except ValueError:
        return None


def discover_local_mcp(root: Path, preferred_port: int | None = None) -> dict[str, Any]:
    """Probe fixed loopback candidates; verify this checkout and its tool catalog.

    Provider-supplied URLs, redirects and environment proxies are never used.
    A different checkout or a server advertising the old AutoDL-only catalog
    cannot satisfy readiness, even when it responds with HTTP 200.
    """
    root = root.resolve()
    result: dict[str, Any] = {"ready": False, "mcp_url": None, "status": "not_running",
                              "mcp_server_version": None, "mcp_tools": [], "mcp_tools_sha256": None,
                              "missing_tools": sorted(REQUIRED_MCP_TOOLS)}
    with httpx.Client(timeout=0.3, trust_env=False, follow_redirects=False) as client:
        for port in _candidate_ports(root, preferred_port):
            try:
                response = client.get(f"http://127.0.0.1:{port}/api/health")
                if response.status_code != 200:
                    continue
                data = response.json()
                if not isinstance(data, dict) or data.get("ok") is not True or data.get("chatgpt_bridge") != "ready":
                    continue
                if not _same_root(data.get("application_root"), root):
                    if result["status"] == "not_running":
                        result["status"] = "different_checkout"
                    continue
                version = data.get("mcp_server_version")
                tools = data.get("mcp_tools")
                if not isinstance(tools, list) or any(not isinstance(tool, str) for tool in tools):
                    tools = []
                names = sorted(set(tools))
                missing = sorted(REQUIRED_MCP_TOOLS - set(names))
                supported_version = bool(isinstance(version, str) and re.fullmatch(r"\d+\.\d+\.\d+", version)
                                         and tuple(map(int, version.split("."))) >= (0, 3, 1))
                if not supported_version or missing:
                    result.update(status="outdated_server", missing_tools=missing)
                    continue
                fingerprint = hashlib.sha256(json.dumps({"version": version, "tools": names}, sort_keys=True).encode()).hexdigest()
                return {"ready": True, "mcp_url": f"http://127.0.0.1:{port}/mcp", "status": "ready",
                        "mcp_server_version": version, "mcp_tools": names,
                        "mcp_tools_sha256": fingerprint, "missing_tools": []}
            except (httpx.HTTPError, ValueError, TypeError, OSError):
                continue
    return result


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
    local = discover_local_mcp(root)

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
    target = _local_mcp_url(runtime.get("mcp_server_url") or config.get("mcp_server_url"))
    target_matches = bool(local["ready"] and target == local["mcp_url"])
    catalog_matches = bool(local["ready"] and runtime.get("mcp_tools_sha256") == local["mcp_tools_sha256"])
    tunnel_running = process_running and healthy and ready and target_matches and catalog_matches
    tunnel_status = (
        "unconfigured" if not configured else
        "stopped" if not process_running else
        "connecting" if not (healthy and ready) else
        "local_server_unavailable" if not local["ready"] else
        "local_target_mismatch" if not target_matches else
        "ready" if tunnel_running else "local_catalog_unverified"
    )
    tunnel_id = str(config.get("tunnel_id", ""))

    return {
        "local_mcp_ready": local["ready"],
        "local_mcp_url": local["mcp_url"],
        "local_mcp_status": local["status"],
        "local_mcp_server_version": local["mcp_server_version"],
        "local_mcp_tools": local["mcp_tools"],
        "local_mcp_tools_sha256": local["mcp_tools_sha256"],
        "missing_mcp_tools": local["missing_tools"],
        "tunnel_local_target_matches": target_matches,
        "tunnel_tool_catalog_matches": catalog_matches,
        "tunnel_local_mcp_url": target,
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

    creation_flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0) if action == "Setup" else getattr(subprocess, "CREATE_NO_WINDOW", 0)
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


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Read-only AutoResearch MCP endpoint discovery")
    parser.add_argument("--discover", action="store_true", required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(discover_local_mcp(args.root), ensure_ascii=True))


if __name__ == "__main__":
    main()
