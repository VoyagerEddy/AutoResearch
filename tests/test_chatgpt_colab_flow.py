from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from mcp import Client
import pytest

from autoresearch.api import create_app
from autoresearch.config import Settings
from autoresearch.domain import (
    ChatGPTAnalysisRequest, ChatGPTProjectRequest, CodeFile, CodeSaveRequest,
    ColabObservationRequest, ColabPrepareRequest,
)
from autoresearch.services import colab as colab_module
from autoresearch import colab_browser_cli


def manifest(kind="campaign_complete"):
    return {
        "run_command": "python train.py",
        "research_coverage": {"experiments": [{"id": "E1", "hypothesis_ids": ["H1"], "dataset_ids": ["D1"],
                                               "scope": "full", "implementation_files": ["train.py"]}]},
        "colab": {
            "schema_version": 1, "setup_commands": [], "prepare_command": "",
            "run_command": "python train.py", "required_drive_paths": [],
            "checkpoint": {"resume_supported": True, "full_training_state": True,
                           "interval_seconds": 300, "files": ["resume_latest.pt"],
                           "resume_command": "python train.py --resume auto"},
            "success_criteria": {"validation": "scientific_metrics", "metrics_file": "fixture/metrics.json",
                                 "status_value": kind, "required_metrics": ["fixture_score", "fixture_loss"]},
        },
    }


def research_plan():
    return {"title": "Explicit fixture-only research coverage", "hypotheses": [{"id": "H1", "description": "A test hypothesis"}],
            "datasets": [{"id": "D1", "description": "Synthetic input fixture", "scope": "full", "labels": "available"}],
            "experiments": [{"id": "E1", "hypothesis_ids": ["H1"], "dataset_ids": ["D1"], "scope": "full",
                             "implementation_files": ["train.py"]}], "success_criteria": {"metric": "fixture_score"}}


def create_prepared(tmp_path, kind="campaign_complete"):
    app = create_app(Settings.load(tmp_path))
    bridge = app.state.autoresearch.chatgpt
    project = bridge.create_project(ChatGPTProjectRequest(topic="Isolated ChatGPT Colab flow"))
    pid = project["project"]["id"]
    bridge.save_code(pid, CodeSaveRequest(
        files=[CodeFile(path="train.py", content="print('fixture only; not a scientific measurement')\n")],
        experiment_manifest=manifest(kind),
    ))
    bridge.prepare_colab(pid, ColabPrepareRequest())
    return app, bridge, pid


def observe_completion(bridge, pid, *, kind="campaign_complete", logs=False):
    service = bridge.colab
    service.observe(pid, ColabObservationRequest(status="available", details={"runtime_id": "current-vm"}))
    service.observe(pid, ColabObservationRequest(status="running", details={"runtime_id": "current-vm"}))
    details = {"runtime_id": "current-vm", "returncode": 0, "completion_kind": kind,
               "scientific_metrics_produced": kind == "campaign_complete", "metrics_file": "fixture/metrics.json",
               "metrics": {"fixture_score": 5.0, "fixture_loss": 0.4},
               "log_path": "/content/drive/MyDrive/AutoResearch/results/session_current-vm.log"}
    if logs:
        details["log_tail"] = "fixture steps finished\napi_key=private-test-value\nhttps://colab.research.google.com/drive/fixture-copy#mcpProxyToken=private-bridge-code"
    return service.observe(pid, ColabObservationRequest(status="completed", details=details))


@pytest.mark.asyncio
async def test_chatgpt_mcp_code_wait_authorize_resume_result_analysis_roundtrip(tmp_path, monkeypatch):
    app = create_app(Settings.load(tmp_path))
    bridge = app.state.autoresearch.chatgpt
    clock = [datetime(2026, 11, 19, tzinfo=timezone.utc)]
    monkeypatch.setattr(colab_module, "_utc", lambda: clock[0])
    workers, launches = {}, []
    monkeypatch.setattr(colab_browser_cli, "worker_is_running", lambda root, pid: workers.get(pid))
    from autoresearch.services import colab_resource_monitor
    monitor_launches = []
    monkeypatch.setattr(colab_resource_monitor, "launch_resource_monitor", lambda root, pid, **kwargs: monitor_launches.append(pid) or 434343)
    monkeypatch.setattr(colab_browser_cli, "login_browser_is_running", lambda root: None)

    def launch(root, pid, **kwargs):
        launches.append(pid)
        workers[pid] = 424242
        return 424242

    monkeypatch.setattr(colab_browser_cli, "launch_colab_worker", launch)

    async with Client(app.state.mcp_server, raise_exceptions=True) as client:
        listed = {tool.name: tool for tool in (await client.list_tools()).tools}
        assert {"get_colab_status", "get_colab_resources", "prepare_colab_experiment", "start_colab_experiment",
                "resume_colab_experiment", "bind_colab_notebook", "get_colab_experiment_result",
                "save_research_plan", "get_research_workflow", "record_research_decision", "open_colab_login"} <= set(listed)
        assert listed["get_colab_experiment_result"].annotations.read_only_hint
        assert "confirm_execute" in listed["resume_colab_experiment"].input_schema["required"]

        async def call(name, **arguments):
            result = await client.call_tool(name, arguments)
            assert not result.is_error, str(result.content)
            assert result.structured_content is not None
            return result.structured_content

        created = await call("create_research_project", topic="Fixture-only complete ChatGPT execution flow")
        pid = created["project"]["id"]
        await call("save_experiment_code", project_id=pid,
                   files=[{"path": "train.py", "content": "print('test fixture')\n"}], experiment_manifest=manifest())
        planned = await call("save_research_plan", project_id=pid, plan=research_plan())
        assert planned["reasoning_owner"] == "chatgpt"
        prepared = await call("prepare_colab_experiment", project_id=pid)
        assert prepared["prepared"] and prepared["attempt_count"] == 0
        assert prepared["resource_agent"]["agent_running"] is False
        bound = await call("bind_colab_notebook", project_id=pid,
                           notebook_url="https://colab.research.google.com/drive/fixture-bound#mcpProxyToken=not-for-chatgpt",
                           bundle_sha256=prepared["bundle_sha256"])
        assert bound["notebook_url"] == "https://colab.research.google.com/drive/fixture-bound"
        assert "not-for-chatgpt" not in json.dumps(bound)
        denied = await client.call_tool("start_colab_experiment", {"project_id": pid, "confirm_execute": False})
        assert denied.is_error and not launches
        await call("start_colab_experiment", project_id=pid, confirm_execute=True)
        assert launches == [pid]
        assert monitor_launches == [pid]

        # These are simulated provider observations; real persistent local state
        # must still preserve the retry deadline and reject cooldown bypasses.
        bridge.colab.observe(pid, ColabObservationRequest(status="unavailable"))
        workers[pid] = None
        waiting = await call("get_colab_status", project_id=pid)
        assert waiting["status"] == "waiting_for_gpu" and waiting["next_attempt_at"]
        retry = await call("start_colab_experiment", project_id=pid, confirm_execute=True)
        assert retry["status"] == "retry_scheduled" and launches == [pid]
        clock[0] += timedelta(minutes=11)
        bridge.colab.observe(pid, ColabObservationRequest(status="auth", details={"reason": "Complete Drive consent"}))
        paused = await call("get_colab_status", project_id=pid)
        assert paused["next_action"]["action"] == "complete_browser_authorization"
        no_authorization = await client.call_tool("resume_colab_experiment", {"project_id": pid, "confirm_execute": False})
        assert no_authorization.is_error and len(launches) == 1
        await call("resume_colab_experiment", project_id=pid, confirm_execute=True)
        assert len(launches) == 2
        pending = await call("get_colab_experiment_result", project_id=pid)
        assert not pending["execution_completed"] and pending["metrics"] == {}
        observe_completion(bridge, pid)
        completed = await call("get_colab_experiment_result", project_id=pid)
        assert completed["result_status"] == "validated"
        assert completed["full_experiment_complete"] and completed["metrics"]["fixture_score"] == 5
        assert completed["log_status"] == "not_returned" and completed["log_tail"] is None
        assert completed["evidence"]["origin"] == "current_notebook_observation"
        report = await call("record_chatgpt_analysis", project_id=pid,
                            experiment_id=completed["experiment_id"],
                            analysis="This is test-only runtime evidence; no actual scientific measurement was performed.",
                            recommendations=["Complete actual full-data GPU validation."])
        assert report["path"].startswith("reports/chatgpt-analysis-")
        status = await call("get_research_status", project_id=pid)
        assert status["colab"]["status"] == "completed"
        assert "research_workflow" in status
        workflow = await call("get_research_workflow", project_id=pid)
        decision = {key: workflow[key] for key in ("plan_id", "plan_sha256", "code_sha256", "evidence_id")}
        decision.update(action="revise", analysis="Test-only evidence does not support a scientific improvement claim.", next_experiment_ids=["E1"])
        decided = await call("record_research_decision", project_id=pid, decision=decision)
        assert decided


@pytest.mark.parametrize("fault", ["runtime", "code", "metrics", "contract"])
def test_old_or_mismatched_colab_evidence_never_becomes_current_result(tmp_path, fault):
    _, bridge, pid = create_prepared(tmp_path)
    observe_completion(bridge, pid)
    job = bridge.colab._read(pid)
    if fault == "runtime":
        job["runtime_id"] = "a-new-runtime"
    elif fault == "code":
        bridge.save_code(pid, CodeSaveRequest(files=[CodeFile(path="train.py", content="print('changed experiment')\n")]))
    elif fault == "metrics":
        job["last_observation"]["details"]["metrics"]["fixture_loss"] = float("nan")
    else:
        job["last_observation"]["details"]["metrics_file"] = "historical/metrics.json"
    if fault != "code":
        bridge.colab._write(pid, job)
    result = bridge.colab_result(pid)
    assert result["result_status"] == "pending_evidence"
    assert not result["execution_completed"] and not result["scientific_metrics_produced"]
    assert not result["full_experiment_complete"] and result["metrics"] == {}
    assert result["blocking_issues"]


def test_local_historical_metrics_and_provider_paths_are_not_treated_as_results(tmp_path):
    _, bridge, pid = create_prepared(tmp_path)
    workspace = Path(bridge.db.get_project(pid).workspace)
    (workspace / "results").mkdir()
    (workspace / "results/metrics.json").write_text(json.dumps({"status": "campaign_complete", "fixture_score": 0}))
    bridge.colab.observe(pid, ColabObservationRequest(status="running", details={
        "runtime_id": "current-vm", "log_path": str(workspace / "results/metrics.json")}))
    result = bridge.colab_result(pid)
    assert not result["execution_completed"] and result["metrics"] == {}
    assert result["log_status"] == "not_returned" and result["log_tail"] is None


def test_acceptance_metrics_do_not_complete_full_research_and_private_fields_are_filtered(tmp_path):
    _, bridge, pid = create_prepared(tmp_path, kind="acceptance_passed")
    observe_completion(bridge, pid, kind="acceptance_passed", logs=True)
    job = bridge.colab._read(pid)
    job["data_upload_files"] = [{"local_path": "C:/private/fixture/sample.bin", "approved_root": "C:/private/fixture"}]
    job["browser_url"] = "https://colab.research.google.com/drive/fixture-copy#mcpProxyToken=private-bridge-code"
    bridge.colab._write(pid, job)
    result = bridge.colab_result(pid)
    assert result["execution_completed"] and result["acceptance_complete"]
    assert not result["full_experiment_complete"] and not result["scientific_metrics_produced"]
    assert "private-test-value" not in result["log_tail"]
    assert "private-bridge-code" not in result["log_tail"]
    public = json.dumps(bridge.colab_status(pid))
    assert "C:/private" not in public and "private-bridge-code" not in public


def test_resume_cannot_bypass_a_gpu_quota_cooldown(tmp_path, monkeypatch):
    _, bridge, pid = create_prepared(tmp_path)
    bridge.colab.enable_autorun(pid, "auto")
    quota = bridge.colab.observe(pid, ColabObservationRequest(status="quota"))
    bridge.colab.observe(pid, ColabObservationRequest(status="auth"))
    monkeypatch.setattr(colab_browser_cli, "launch_colab_worker", lambda *args, **kwargs: pytest.fail("Cooldown must prevent a new GPU attempt"))
    monkeypatch.setattr(bridge, "_ensure_colab_resource_agent", lambda *args: {"agent_running": False, "launch_status": "starting"})
    resumed = bridge.resume_colab(pid, confirm_execute=True)
    assert resumed["status"] == "retry_scheduled"
    assert resumed["next_attempt_at"] == quota["next_attempt_at"]


def test_changed_source_requires_preparation_before_starting_a_new_worker(tmp_path, monkeypatch):
    _, bridge, pid = create_prepared(tmp_path)
    bridge.save_code(pid, CodeSaveRequest(files=[CodeFile(path="train.py", content="print('new code')\n")]))
    monkeypatch.setattr(colab_browser_cli, "launch_colab_worker", lambda *args, **kwargs: pytest.fail("Stale source must not execute"))
    with pytest.raises(ValueError, match="prepare"):
        bridge.start_colab(pid, confirm_execute=True)


def test_login_is_part_of_workflow_without_requesting_gpu(tmp_path, monkeypatch):
    _, bridge, pid = create_prepared(tmp_path)
    calls = []
    monkeypatch.setattr(colab_browser_cli, "worker_is_running", lambda *args: None)
    monkeypatch.setattr(colab_browser_cli, "launch_colab_login_browser", lambda root, **kwargs: calls.append(kwargs) or 12345)
    monkeypatch.setattr(colab_browser_cli, "launch_colab_worker", lambda *args, **kwargs: pytest.fail("Login must not request compute"))
    opened = bridge.open_colab_login(pid, browser="chrome")
    assert opened["status"] == "login_browser_opened" and opened["action_required"]
    assert calls == [{"browser": "chrome"}]
    state = bridge.colab_status(pid)
    assert state["status"] == "auth_required" and state["action_required"]
    assert state["next_action"]["action"] == "complete_login_then_close_window"
    assert bridge.colab.status(pid)["attempt_count"] == 0


def test_login_reuses_an_active_worker_browser_and_never_stops_it(tmp_path, monkeypatch):
    _, bridge, pid = create_prepared(tmp_path)
    monkeypatch.setattr(colab_browser_cli, "worker_is_running", lambda *args: 88888)
    monkeypatch.setattr(colab_browser_cli, "launch_colab_login_browser", lambda *args, **kwargs: pytest.fail("Reuse the active browser"))
    result = bridge.open_colab_login(pid)
    assert result["status"] == "use_existing_browser" and result["pid"] == 88888
