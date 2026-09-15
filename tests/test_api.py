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
        assert "confirmation" in response.json()["detail"].lower()


def test_autodl_creation_requires_billable_confirmation(tmp_path: Path) -> None:
    with TestClient(create_app(Settings.load(tmp_path))) as client:
        response = client.post("/api/autodl/instances", json={})
        assert response.status_code == 400


def test_autodl_browser_login_launch_requires_confirmation(tmp_path: Path) -> None:
    with TestClient(create_app(Settings.load(tmp_path))) as client:
        response = client.post(
            "/api/autodl/browser-login/launch",
            json={"confirm_launch": False},
        )
    assert response.status_code == 400
    assert "confirmation" in response.json()["detail"].lower()


def test_autodl_browser_login_launches_private_interactive_helper(
    tmp_path: Path, monkeypatch
) -> None:
    calls = []

    def launch(root, **kwargs):
        calls.append((root, kwargs))
        return 4321

    monkeypatch.setattr(
        "autoresearch.api.DesktopBridge.launch_autodl_browser_login", launch
    )
    with TestClient(create_app(Settings.load(tmp_path))) as client:
        response = client.post(
            "/api/autodl/browser-login/launch",
            json={
                "action": "Login",
                "browser": "edge",
                "open_page": "console",
                "keep_open": True,
                "confirm_launch": True,
            },
        )

    assert response.status_code == 202
    assert response.json() == {
        "status": "launched",
        "action": "Login",
        "browser": "edge",
        "open_page": "console",
        "pid": 4321,
        "credential_storage": "windows-user-encrypted",
    }
    assert calls == [
        (
            tmp_path,
            {
                "action": "Login",
                "browser": "edge",
                "open_page": "console",
                "keep_open": True,
            },
        )
    ]
    assert not any(
        secret in str(response.json()).lower()
        for secret in ("phone", "password", "otp", "token")
    )


def test_experiment_readiness_exposes_blockers_and_handles_missing_project(tmp_path: Path) -> None:
    with TestClient(create_app(Settings.load(tmp_path))) as client:
        created = client.post("/api/chatgpt/projects", json={"topic": "readiness study"}).json()
        project_id = created["project"]["id"]
        result = client.get(f"/api/projects/{project_id}/readiness").json()
        assert result["ready"] is False
        assert result["blocking_issues"][0]["code"] == "manifest"
        assert result["execution_status"] == "not_executed_by_readiness_check"
        assert client.get("/api/projects/missing/readiness").status_code == 404


def test_uncertain_autodl_creation_has_actionable_error_code(tmp_path, monkeypatch) -> None:
    from autoresearch.services.autodl import AutoDLClient, AutoDLCreateUncertain

    async def create(*args):
        raise AutoDLCreateUncertain("Creation result unknown; check the instance list first")

    monkeypatch.setattr(AutoDLClient, "create_preferred", create)
    with TestClient(create_app(Settings.load(tmp_path))) as client:
        response = client.post("/api/autodl/instances", json={"confirm_billable": True})
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "autodl_create_uncertain"


def test_autodl_preflight_reports_missing_configuration_without_confirming(tmp_path: Path) -> None:
    settings = Settings.load(tmp_path).with_overrides(autodl_token="", autodl_image_uuid="")
    with TestClient(create_app(settings)) as client:
        response = client.get("/api/autodl/preflight?verify_api=true")
        assert response.status_code == 200
        result = response.json()
        assert result["ready"] is False
        assert result["network_checked"] is False
        assert {issue["code"] for issue in result["blocking_issues"]} == {
            "missing_token", "missing_image",
        }


def test_autodl_preflight_supports_image_override_and_redacts_token(tmp_path: Path) -> None:
    settings = Settings.load(tmp_path).with_overrides(
        autodl_token="private-autodl-token", autodl_image_uuid=""
    )
    with TestClient(create_app(settings)) as client:
        result = client.get("/api/autodl/preflight?image_uuid=image-override").json()
    assert result["ready"] is True
    assert result["verified"] is False
    assert "private-autodl-token" not in str(result)


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
        assert "confirmation" in response.json()["detail"]
