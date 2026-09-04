from pathlib import Path

from fastapi.testclient import TestClient

from autoresearch.api import create_app
from autoresearch.config import Settings


def test_health_and_redacted_settings(tmp_path: Path) -> None:
    settings = Settings.load(tmp_path).with_overrides(openrouter_api_key="secret")
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/health").json()["ok"] is True
        public = client.get("/api/settings").json()
        assert public["openrouter_configured"] is True
        assert "secret" not in str(public)


def test_chatgpt_connection_launch_requires_confirmation(tmp_path: Path) -> None:
    with TestClient(create_app(Settings.load(tmp_path))) as client:
        status = client.get("/api/chatgpt/connection")
        assert status.status_code == 200
        assert status.json()["local_mcp_url"].endswith("/mcp")

        response = client.post(
            "/api/chatgpt/connection/launch",
            json={"action": "Setup", "confirm_launch": False},
        )
        assert response.status_code == 400
        assert "确认" in response.json()["detail"]


def test_autodl_creation_requires_billable_confirmation(tmp_path: Path) -> None:
    with TestClient(create_app(Settings.load(tmp_path))) as client:
        response = client.post("/api/autodl/instances", json={})
        assert response.status_code == 400


def test_settings_patch_is_visible_immediately(tmp_path: Path) -> None:
    with TestClient(create_app(Settings.load(tmp_path))) as client:
        response = client.patch(
            "/api/settings",
            json={"openrouter_model": "example/new-model", "github_remote_url": ""},
        )
        assert response.status_code == 200
        assert response.json()["openrouter_model"] == "example/new-model"
        assert client.get("/api/settings").json()["openrouter_model"] == "example/new-model"


def test_chatgpt_project_and_code_are_visible_in_dashboard_api(tmp_path: Path) -> None:
    with TestClient(create_app(Settings.load(tmp_path))) as client:
        created = client.post(
            "/api/chatgpt/projects",
            json={"topic": "ChatGPT-led local research", "notes": "fixed seed"},
        )
        assert created.status_code == 201
        project_id = created.json()["project"]["id"]

        saved = client.post(
            f"/api/chatgpt/projects/{project_id}/code",
            json={
                "files": [{"path": "experiment.py", "content": "print('ok')\n"}],
                "summary": "ready from ChatGPT",
                "experiment_manifest": {"run_command": "python experiment.py"},
            },
        )
        assert saved.status_code == 200
        status = client.get(f"/api/projects/{project_id}/status").json()
        assert status["project"]["summary"] == "ready from ChatGPT"
        assert client.get(f"/api/projects/{project_id}/experiments").json() == []


def test_experiment_api_requires_execution_confirmation(tmp_path: Path) -> None:
    with TestClient(create_app(Settings.load(tmp_path))) as client:
        created = client.post(
            "/api/chatgpt/projects", json={"topic": "Remote confirmation test"}
        ).json()
        project_id = created["project"]["id"]
        client.post(
            f"/api/chatgpt/projects/{project_id}/code",
            json={"files": [{"path": "experiment.py", "content": "print('ok')"}]},
        )
        response = client.post(
            "/api/experiments",
            json={
                "project_id": project_id,
                "connection": {"host": "example.test"},
                "command": "python experiment.py",
            },
        )
        assert response.status_code == 400
        assert "确认" in response.json()["detail"]
