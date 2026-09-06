import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from autoresearch import experiment_runner as runner


def make_manifest(tmp_path: Path, steps: list[dict] | None = None, **changes: object) -> Path:
    manifest = {
        "schema_version": 1,
        "working_directory": ".",
        "required_files": [],
        "required_inputs": [],
        "steps": steps if steps is not None else [
            {"name": "example", "command": [sys.executable, "-c", "print('checked')"],
             "timeout_seconds": 5}],
    }
    manifest.update(changes)
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def script_step(code: str, name: str = "baseline", **changes: object) -> dict:
    step = {"name": name, "command": [sys.executable, "-c", code], "timeout_seconds": 5}
    step.update(changes)
    return step


def read_report(report: dict) -> dict:
    return json.loads(Path(report["report_path"]).read_text(encoding="utf-8"))


def test_default_only_preflights_and_retains_distinct_reports(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path, [script_step("from pathlib import Path; Path('ran').touch()")])
    first = runner.run_manifest(manifest, tmp_path / "results")
    second = runner.run_manifest(manifest, tmp_path / "results")
    assert first["status"] == second["status"] == "ready"
    assert first["execute"] is False
    assert first["exit_code"] is None
    assert first["steps"][0]["status"] == "not_run"
    assert first["run_id"] != second["run_id"]
    assert not (tmp_path / "ran").exists()
    assert read_report(first) == first
    assert json.loads((tmp_path / "results" / "metrics.json").read_text(encoding="utf-8")) == second
    assert not (tmp_path / "results" / ".autoresearch-run.lock").exists()


@pytest.mark.parametrize("change", [
    {"working_directory": ".."},
    {"required_files": ["../outside.py"]},
])
def test_package_paths_cannot_escape_manifest_directory(tmp_path, change):
    manifest = make_manifest(tmp_path, **change)
    report = runner.preflight_manifest(manifest)
    assert report["execution_ready"] is False
    assert "inside" in report["issues"][0]


def test_execution_checks_both_input_scopes_and_relative_locations(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    (work / "local.txt").write_text("input")
    manifest = make_manifest(tmp_path, working_directory="work", required_inputs=[
        {"name": "local data", "path": "local.txt", "kind": "file", "scope": "local"},
        {"name": "remote data", "path": str(work / "remote"), "kind": "directory", "scope": "remote"},
    ])
    local = runner.preflight_manifest(manifest, check_remote_inputs=False)
    assert local["status"] == "ready" and not local["execution_ready"]
    assert local["remote_inputs_pending"]
    assert local["inputs"][0]["status"] == "present"
    assert local["inputs"][1]["status"] == "pending_remote"
    blocked = runner.run_manifest(manifest, tmp_path / "results", execute=True)
    assert blocked["status"] == "blocked"
    assert "remote data" in blocked["preflight"]["issues"][0]
    (work / "remote").mkdir()
    assert runner.preflight_manifest(manifest)["execution_ready"]


def test_required_files_are_relative_to_manifest_not_working_directory(tmp_path: Path) -> None:
    (tmp_path / "work").mkdir()
    (tmp_path / "entry.py").write_text("pass")
    manifest = make_manifest(tmp_path, working_directory="work", required_files=["entry.py"])
    assert runner.preflight_manifest(manifest)["status"] == "ready"
    (tmp_path / "entry.py").unlink()
    assert runner.preflight_manifest(manifest)["status"] == "blocked"


def test_foreign_remote_path_is_preserved_for_local_preflight(tmp_path: Path) -> None:
    remote = "/root/autodl-tmp/dataset" if os.name == "nt" else "C:/dataset"
    manifest = make_manifest(tmp_path, required_inputs=[
        {"name": "dataset", "path": remote, "scope": "remote", "kind": "directory"},
    ])
    local = runner.preflight_manifest(manifest, check_remote_inputs=False)
    assert local["status"] == "ready"
    assert local["inputs"][0]["path"] == remote
    assert runner.preflight_manifest(manifest)["status"] == "blocked"


@pytest.mark.parametrize("changes", [
    {"schema_version": True}, {"working_directory": "/absolute"}, {"required_files": ["C:/file"]},
    {"steps": []}, {"steps": [script_step("pass", command="python x.py")]},
    {"steps": [script_step("pass", command=[])]},
    {"steps": [script_step("pass", command=[sys.executable, "\0"])]},
    {"steps": [script_step("pass", timeout_seconds=True)]},
    {"steps": [script_step("pass", timeout_seconds=0)]},
    {"steps": [script_step("pass", timeout_seconds=-1)]},
    {"steps": [script_step("pass", timeout_seconds=0.5)]},
    {"steps": [script_step("pass", required_metrics=["score"])]},
    {"steps": [script_step("pass", metrics_file="../metrics.json")]},
    {"steps": [script_step("pass", env={"BAD=KEY": "value"})]},
    {"steps": [script_step("pass", env={"GOOD": 42})]},
])
def test_malformed_manifests_block_before_execution(tmp_path: Path, changes: dict) -> None:
    manifest = make_manifest(tmp_path, **changes)
    report = runner.run_manifest(manifest, tmp_path / "results", execute=True)
    assert report["status"] == "blocked"
    assert report["exit_code"] == 2
    assert report["preflight"]["issues"]
    assert read_report(report) == report


def test_missing_manifest_replaces_previous_success_summary(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    previous = runner.run_manifest(manifest, tmp_path / "results", execute=True)
    assert previous["status"] == "succeeded"
    blocked = runner.run_manifest(tmp_path / "missing.json", tmp_path / "results", execute=True)
    assert blocked["status"] == "blocked"
    latest = json.loads((tmp_path / "results" / "metrics.json").read_text(encoding="utf-8"))
    assert latest["run_id"] == blocked["run_id"]
    assert latest["metrics"] == {}
    assert read_report(previous)["status"] == "succeeded"


def test_success_records_measured_metrics_and_separate_logs_without_env_values(tmp_path: Path) -> None:
    code = (
        "import json, os, sys; from pathlib import Path; "
        "assert len(os.environ['SAMPLE_SECRET']) == 13; "
        "print('ordinary output'); print('diagnostic output', file=sys.stderr); "
        "Path(os.environ['AUTORESEARCH_METRICS_FILE']).write_text(json.dumps({'score': 0.75, 'count': 2}))"
    )
    manifest = make_manifest(tmp_path, [script_step(code, metrics_file="scores.json",
                              required_metrics=["score"], env={"SAMPLE_SECRET": "private-value"})])
    report = runner.run_manifest(manifest, tmp_path / "results", execute=True)
    assert report["status"] == "succeeded" and report["exit_code"] == 0
    step = report["steps"][0]
    assert step["metrics"] == {"score": 0.75, "count": 2}
    assert step["exit_code"] == 0 and step["elapsed_seconds"] > 0
    assert Path(step["stdout_log"]).read_text(encoding="utf-8").strip() == "ordinary output"
    assert Path(step["stderr_log"]).read_text(encoding="utf-8").strip() == "diagnostic output"
    assert "private-value" not in Path(report["report_path"]).read_text(encoding="utf-8")
    assert "env" not in step and "env" not in report["preflight"]["steps"][0]
    assert step["environment_keys"] == ["SAMPLE_SECRET"]
    assert report["metrics"] == {"baseline": {"score": 0.75, "count": 2}}


def test_argv_metacharacters_are_literal(tmp_path: Path) -> None:
    argument = "hello & echo unsafe > unexpected.txt"
    manifest = make_manifest(tmp_path, [script_step("unused", command=[
        sys.executable, "-c", "import sys; print(sys.argv[1])", argument,
    ])])
    report = runner.run_manifest(manifest, tmp_path / "results", execute=True)
    assert report["status"] == "succeeded"
    assert Path(report["steps"][0]["stdout_log"]).read_text(encoding="utf-8").strip() == argument
    assert not (tmp_path / "unexpected.txt").exists()


@pytest.mark.parametrize("metrics, message", [
    ({"other": 1}, "Missing required metrics"),
    ({"score": "0.5"}, "finite number"),
    ({"score": True}, "finite number"),
    ({"score": None}, "finite number"),
    ({"score": float("nan")}, "finite number"),
    ({"score": float("inf")}, "finite number"),
    ([1, 2], "JSON object"),
])
def test_invalid_metrics_fail_even_when_command_succeeds(tmp_path: Path, metrics: object, message: str) -> None:
    payload = json.dumps(metrics)
    code = f"from pathlib import Path; Path('scores.json').write_text({payload!r})"
    manifest = make_manifest(tmp_path, [script_step(code, metrics_file="scores.json", required_metrics=["score"])])
    report = runner.run_manifest(manifest, tmp_path / "results", execute=True)
    assert report["status"] == "failed"
    assert report["steps"][0]["exit_code"] == 0
    assert message in report["steps"][0]["error"]
    assert report["metrics"] == {}


def test_existing_metrics_are_preserved_and_never_reused(tmp_path: Path) -> None:
    old = tmp_path / "scores.json"
    old.write_text('{"score": 999}')
    manifest = make_manifest(tmp_path, [script_step("raise AssertionError('must not run')",
                              metrics_file="scores.json", required_metrics=["score"])])
    report = runner.run_manifest(manifest, tmp_path / "results", execute=True)
    assert report["status"] == "blocked"
    assert "already exists" in report["preflight"]["issues"][0]
    assert old.read_text(encoding="utf-8") == '{"score": 999}'
    assert report["metrics"] == {}


def test_previous_step_cannot_supply_later_steps_metrics(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path, [
        script_step("from pathlib import Path; Path('later.json').write_text('{\"score\": 5}')", name="first"),
        script_step("pass", name="later", metrics_file="later.json", required_metrics=["score"]),
    ])
    report = runner.run_manifest(manifest, tmp_path / "results", execute=True)
    assert report["status"] == "failed"
    assert report["steps"][0]["status"] == "succeeded"
    assert report["steps"][1]["exit_code"] is None
    assert "refusing stale metrics" in report["steps"][1]["error"]


def test_nonzero_exit_stops_later_steps(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path, [script_step("raise SystemExit(7)", name="failed"),
        script_step("from pathlib import Path; Path('later').touch()", name="later")])
    report = runner.run_manifest(manifest, tmp_path / "results", execute=True)
    assert report["status"] == "failed"
    assert report["steps"][0]["exit_code"] == 7
    assert report["steps"][1]["status"] == "not_run"
    assert not (tmp_path / "later").exists()


def test_missing_executable_has_no_fabricated_exit_code(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path, [script_step("unused", command=[str(tmp_path / "missing-executable")])])
    report = runner.run_manifest(manifest, tmp_path / "results", execute=True)
    assert report["status"] == "failed"
    assert report["steps"][0]["exit_code"] is None
    assert report["steps"][0]["error"]


def test_missing_metrics_cannot_be_reported_as_success(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path, [script_step("pass", metrics_file="scores.json", required_metrics=["score"])])
    report = runner.run_manifest(manifest, tmp_path / "results", execute=True)
    assert report["status"] == "failed"
    assert "Cannot read fresh metrics_file" in report["steps"][0]["error"]


def test_timeout_terminates_spawned_child_and_stops_sequence(tmp_path: Path) -> None:
    # All child state stays in tmp_path; never enumerate or kill unrelated PIDs.
    child = tmp_path / "child.py"
    child.write_text("import time\nfrom pathlib import Path\nwhile True:\n"
                     "    Path('heartbeat').write_text(str(time.monotonic()))\n    time.sleep(0.05)\n")
    parent = tmp_path / "parent.py"
    parent.write_text("import subprocess, sys, time\nfrom pathlib import Path\n"
                      "child = subprocess.Popen([sys.executable, 'child.py'])\n"
                      "Path('child.pid').write_text(str(child.pid))\ntime.sleep(30)\n")
    manifest = make_manifest(tmp_path, [
        script_step("unused", command=[sys.executable, str(parent)], timeout_seconds=1),
        script_step("from pathlib import Path; Path('later').touch()", name="later"),
    ])
    started = time.monotonic()
    report = runner.run_manifest(manifest, tmp_path / "results", execute=True)
    assert report["status"] == "timed_out" and report["exit_code"] == 124
    assert time.monotonic() - started < 8
    assert report["steps"][0]["exit_code"] is not None
    assert report["steps"][1]["status"] == "not_run"
    assert (tmp_path / "child.pid").is_file()
    heartbeat = (tmp_path / "heartbeat").read_text(encoding="utf-8")
    time.sleep(0.3)
    assert (tmp_path / "heartbeat").read_text(encoding="utf-8") == heartbeat
    assert not (tmp_path / "later").exists()


def test_output_lock_preserves_active_run_summary(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    output = tmp_path / "results"
    output.mkdir()
    (output / ".autoresearch-run.lock").write_text(json.dumps({"pid": os.getpid()}))
    (output / "metrics.json").write_text('{"status":"running"}')
    with pytest.raises(FileExistsError, match="locked"):
        runner.run_manifest(manifest, output, execute=True)
    assert (output / "metrics.json").read_text(encoding="utf-8") == '{"status":"running"}'


def test_posix_cleanup_kills_group_even_after_leader_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    killed = []
    waits = []
    monkeypatch.setattr(runner, "os", SimpleNamespace(name="posix", killpg=lambda pid, sig: killed.append((pid, sig))))
    monkeypatch.setattr(runner, "signal", SimpleNamespace(SIGTERM=15, SIGKILL=9))
    process = SimpleNamespace(pid=123456, wait=lambda timeout: waits.append(timeout))
    runner._stop_process_tree(process)
    assert killed == [(123456, 15), (123456, 9)]
    assert waits == [0.5, 5]


def test_copied_standalone_cli_needs_no_package_and_defaults_to_preflight(tmp_path: Path) -> None:
    copied = tmp_path / "experiment_runner.py"
    copied.write_text(Path(runner.__file__).read_text(encoding="utf-8"), encoding="utf-8")
    manifest = make_manifest(tmp_path, [script_step("raise AssertionError('must not execute')")])
    result = subprocess.run([sys.executable, "-I", str(copied), "--manifest", str(manifest),
                             "--output", str(tmp_path / "results")],
                            cwd=tmp_path, capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "ready"


def test_main_returns_failure_exit_code_and_json_summary(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    manifest = make_manifest(tmp_path, [script_step("raise SystemExit(9)")])
    assert runner.main(["--manifest", str(manifest), "--output", str(tmp_path / "results"), "--execute"]) == 1
    assert json.loads(capsys.readouterr().out)["steps"][0]["exit_code"] == 9
