import json
import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from autoresearch import chatgpt_connection
from autoresearch.chatgpt_connection import connection_status, _process_is_running


def test_connection_status_detects_config_without_exposing_secrets(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    local_app_data = tmp_path / "local"
    state_dir = local_app_data / "AutoResearch"
    state_dir.mkdir(parents=True)
    client = tmp_path / "tunnel-client.exe"
    client.write_bytes(b"test")
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))
    monkeypatch.setattr(chatgpt_connection, "_tunnel_health", lambda: (True, True))

    (state_dir / "tunnel.json").write_text(
        json.dumps(
            {
                "profile": "autoresearch",
                "tunnel_id": "tunnel_0123456789abcdef",
                "tunnel_client": str(client),
            }
        ),
        encoding="utf-8",
    )
    (state_dir / "tunnel-api-key.dpapi").write_text(
        "encrypted-secret-placeholder", encoding="utf-8"
    )
    (state_dir / "tunnel-status.json").write_text(
        json.dumps({"pid": os.getpid()}), encoding="utf-8"
    )

    status = connection_status(root)

    assert status["tunnel_client_installed"] is True
    assert status["tunnel_configured"] is True
    assert status["tunnel_running"] is True
    assert status["tunnel_id_hint"] == "...abcdef"


@pytest.mark.parametrize("pid", [None, True, -1, 0, "bad", 1.5, 2**40])
def test_invalid_process_ids_are_not_probed(pid, monkeypatch) -> None:
    def unexpected(*args):
        raise AssertionError("Invalid PID must not reach a process API")

    monkeypatch.setattr(chatgpt_connection, "_windows_process_is_running", unexpected)
    monkeypatch.setattr(os, "kill", unexpected)
    assert _process_is_running(pid) is False


def test_process_probe_does_not_terminate_child() -> None:
    # This catches the Windows os.kill(pid, 0) regression without risking the
    # test runner or any user-owned process. The child exits when stdin closes.
    child = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.read()"],
        stdin=subprocess.PIPE,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        for _ in range(3):
            assert _process_is_running(child.pid) is True
            assert child.poll() is None
        child.communicate(timeout=5)
        assert _process_is_running(child.pid) is False
    finally:
        if child.poll() is None:
            child.terminate()
        child.wait(timeout=5)


@pytest.mark.parametrize(
    "health_code,health_body,ready_code,ready_body,expected",
    [
        (200, {"status": "live"}, 200, {"status": "ready"}, (True, True)),
        (200, {"status": "live"}, 503, {"status": "not_ready"}, (True, False)),
        (200, {"status": "live"}, 200, {"status": "not_ready"}, (True, False)),
        (200, {"unrelated": True}, 200, {"status": "ready"}, (False, True)),
    ],
)
def test_tunnel_health_requires_ready_response(
    monkeypatch, health_code, health_body, ready_code, ready_body, expected
) -> None:
    original_client = httpx.Client
    visited = []

    def handle(request):
        visited.append(str(request.url))
        code, body = ((health_code, health_body) if request.url.path == "/healthz"
                      else (ready_code, ready_body))
        return httpx.Response(code, json=body)

    def client(**kwargs):
        assert kwargs["trust_env"] is False
        assert kwargs["follow_redirects"] is False
        assert kwargs["timeout"] <= 1
        return original_client(transport=httpx.MockTransport(handle), **kwargs)

    monkeypatch.setattr(httpx, "Client", client)
    assert chatgpt_connection._tunnel_health() == expected
    assert visited == ["http://127.0.0.1:8080/healthz", "http://127.0.0.1:8080/readyz"]


def test_running_launcher_does_not_imply_connected_tunnel(tmp_path, monkeypatch) -> None:
    state_dir = tmp_path / "local" / "AutoResearch"
    state_dir.mkdir(parents=True)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    (state_dir / "tunnel.json").write_text(json.dumps({
        "profile": "autoresearch", "tunnel_id": "tunnel_example"
    }), encoding="utf-8")
    (state_dir / "tunnel-api-key.dpapi").write_text("encrypted", encoding="utf-8")
    (state_dir / "tunnel-status.json").write_text(
        json.dumps({"pid": os.getpid()}), encoding="utf-8"
    )
    monkeypatch.setattr(chatgpt_connection, "_tunnel_health", lambda: (True, False))
    status = connection_status(tmp_path)
    assert status["tunnel_process_running"] is True
    assert status["tunnel_running"] is False
    assert status["tunnel_ready"] is False
    assert status["tunnel_status"] == "connecting"
    assert "encrypted-secret-placeholder" not in str(status)


def test_connection_status_handles_fresh_install(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))

    status = connection_status(tmp_path / "project")

    assert status["local_mcp_ready"] is True
    assert status["tunnel_configured"] is False
    assert status["tunnel_running"] is False


def test_connection_status_accepts_windows_powershell_utf8_bom(
    tmp_path: Path, monkeypatch
) -> None:
    state_dir = tmp_path / "local" / "AutoResearch"
    state_dir.mkdir(parents=True)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    client = tmp_path / "tunnel-client.exe"
    client.write_bytes(b"binary")

    config = {
        "profile": "autoresearch",
        "tunnel_id": "tunnel_0123456789abcdef",
        "tunnel_client": str(client),
    }
    (state_dir / "tunnel.json").write_text(
        json.dumps(config), encoding="utf-8-sig"
    )
    (state_dir / "tunnel-api-key.dpapi").write_text(
        "encrypted-secret-placeholder", encoding="utf-8"
    )

    status = connection_status(tmp_path / "project")

    assert status["tunnel_configured"] is True
    assert status["tunnel_id_hint"] == "...abcdef"
