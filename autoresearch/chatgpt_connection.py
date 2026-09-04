from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Literal


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
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _process_is_running(pid: object) -> bool:
    try:
        numeric_pid = int(pid)
        if numeric_pid <= 0:
            return False
        os.kill(numeric_pid, 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


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
    tunnel_running = configured and _process_is_running(runtime.get("pid"))
    tunnel_id = str(config.get("tunnel_id", ""))

    return {
        "local_mcp_ready": True,
        "local_mcp_url": LOCAL_MCP_URL,
        "tunnel_client_installed": client_installed,
        "tunnel_configured": configured,
        "tunnel_running": tunnel_running,
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
