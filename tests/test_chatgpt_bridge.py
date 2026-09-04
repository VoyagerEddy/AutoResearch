from pathlib import Path

import pytest
from mcp import Client

from autoresearch.api import create_app
from autoresearch.chatgpt_bridge import ChatGPTBridge
from autoresearch.config import Settings
from autoresearch.db import Database
from autoresearch.domain import (
    ChatGPTAnalysisRequest,
    ChatGPTProjectRequest,
    CodeFile,
    CodeSaveRequest,
    ExperimentStartRequest,
    SSHConnection,
)
from autoresearch.experiments import ExperimentManager
from autoresearch.orchestrator import ResearchOrchestrator
from autoresearch.services.artifacts import UnsafeArtifactPath


def make_bridge(tmp_path: Path) -> ChatGPTBridge:
    settings = Settings.load(tmp_path)
    db = Database(settings.data_dir / "db.sqlite3")
    orchestrator = ResearchOrchestrator(db, settings)
    experiments = ExperimentManager(db, settings, orchestrator)
    return ChatGPTBridge(db, settings, experiments)


def test_chatgpt_project_code_and_analysis_roundtrip(tmp_path: Path) -> None:
    bridge = make_bridge(tmp_path)
    created = bridge.create_project(
        ChatGPTProjectRequest(topic="A ChatGPT-led reproducibility study")
    )
    project_id = created["project"]["id"]
    assert "workspace" not in created["project"]

    saved = bridge.save_code(
        project_id,
        CodeSaveRequest(
            files=[CodeFile(path="experiment.py", content="print('ok')\n")],
            summary="baseline ready",
            experiment_manifest={"run_command": "python experiment.py"},
        ),
    )
    assert saved["status"] == "ready"
    assert bridge.read_artifact(project_id, "generated/experiment.py")["content"] == "print('ok')\n"

    recorded = bridge.record_analysis(
        project_id,
        ChatGPTAnalysisRequest(
            analysis="The baseline is reproducible.",
            recommendations=["Run three additional seeds."],
        ),
    )
    assert recorded["path"].startswith("reports/chatgpt-analysis-")
    status = bridge.status(project_id)
    assert status["project"]["phase"] == "chatgpt_analysis"
    assert any(item["path"] == "generated/experiment.py" for item in status["artifacts"])


def test_chatgpt_code_save_blocks_traversal(tmp_path: Path) -> None:
    bridge = make_bridge(tmp_path)
    project_id = bridge.create_project(ChatGPTProjectRequest(topic="Safe code save"))["project"]["id"]
    with pytest.raises(UnsafeArtifactPath):
        bridge.save_code(
            project_id,
            CodeSaveRequest(files=[CodeFile(path="../outside.py", content="bad")]),
        )


def test_remote_experiment_requires_explicit_confirmation(tmp_path: Path) -> None:
    bridge = make_bridge(tmp_path)
    project_id = bridge.create_project(ChatGPTProjectRequest(topic="Confirm remote execution"))["project"]["id"]
    bridge.save_code(
        project_id,
        CodeSaveRequest(files=[CodeFile(path="experiment.py", content="print('ok')")]),
    )
    with pytest.raises(ValueError, match="确认"):
        bridge.experiments.start(
            ExperimentStartRequest(
                project_id=project_id,
                connection=SSHConnection(host="example.test"),
            )
        )


@pytest.mark.asyncio
async def test_mcp_exposes_chatgpt_collaboration_tools(tmp_path: Path) -> None:
    app = create_app(Settings.load(tmp_path))
    async with Client(app.state.mcp_server, raise_exceptions=True) as client:
        tools = await client.list_tools()
        names = {tool.name for tool in tools.tools}
        by_name = {tool.name: tool for tool in tools.tools}
        assert {
            "create_research_project",
            "save_experiment_code",
            "get_autodl_instance_status",
            "start_autodl_experiment",
            "get_experiment_result",
            "record_chatgpt_analysis",
        } <= names
        assert by_name["start_autodl_experiment"].annotations.destructive_hint is True
        assert by_name["create_autodl_instance"].annotations.destructive_hint is True
        assert "confirm_execute" in by_name["start_autodl_experiment"].input_schema["required"]

        created = await client.call_tool(
            "create_research_project",
            {"topic": "MCP research collaboration", "notes": "Use a fixed seed."},
        )
        assert created.structured_content is not None
        assert created.structured_content["project"]["phase"] == "chatgpt_thinking"
