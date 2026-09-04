import json
import os
from pathlib import Path

from autoresearch.chatgpt_connection import connection_status


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
    assert "encrypted-secret-placeholder" not in str(status)


def test_connection_status_handles_fresh_install(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))

    status = connection_status(tmp_path / "project")

    assert status["local_mcp_ready"] is True
    assert status["tunnel_configured"] is False
    assert status["tunnel_running"] is False
