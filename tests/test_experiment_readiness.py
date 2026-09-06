import json
from pathlib import Path

import pytest

from autoresearch.services.experiment_readiness import inspect_experiment


def write_manifest(workspace: Path, **updates: object) -> dict:
    generated = workspace / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    (generated / "experiment.py").write_text("raise RuntimeError('must never execute or import')\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "run_command": "python experiment.py",
        "working_directory": ".",
        "required_files": ["experiment.py"],
        "required_inputs": [],
        "success_criteria": {"required_metrics": ["accuracy"]},
        **updates,
    }
    (generated / "experiment_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def blocked_codes(result: dict) -> set[str]:
    assert result["status"] == "blocked"
    assert result["ready"] is False
    return {issue["code"] for issue in result["blocking_issues"]}


def test_ready_is_static_and_never_executes_experiment(tmp_path: Path) -> None:
    write_manifest(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    result = inspect_experiment(tmp_path)
    assert result["status"] == "ready"
    assert result["ready"] is True
    assert result["blocking_issues"] == []
    assert result["manifest_path"] == "generated/experiment_manifest.json"
    assert result["execution_status"] == "not_executed_by_readiness_check"
    assert result["unverified"][0]["code"] == "runtime_validation"
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    json.dumps(result)


def test_missing_workspace_is_not_created(tmp_path: Path) -> None:
    workspace = tmp_path / "absent"
    assert "manifest" in blocked_codes(inspect_experiment(workspace))
    assert not workspace.exists()


@pytest.mark.parametrize("contents", ["broken {", "[]", "null", "42", "a" * 200_001],
                         ids=["invalid-json", "array", "null", "number", "oversized"])
def test_invalid_manifest_is_reported_without_throwing(tmp_path: Path, contents: str) -> None:
    generated = tmp_path / "generated"
    generated.mkdir()
    (generated / "experiment_manifest.json").write_text(contents, encoding="utf-8")
    assert "manifest" in blocked_codes(inspect_experiment(tmp_path))


def test_old_manifest_requires_entry_point_and_input_declarations(tmp_path: Path) -> None:
    generated = tmp_path / "generated"
    generated.mkdir()
    (generated / "experiment_manifest.json").write_text(json.dumps({
        "topic": "prior research idea", "files": ["experiment.py"],
        "experiment": {"run_command": "python experiment.py"},
    }), encoding="utf-8")
    assert blocked_codes(inspect_experiment(tmp_path)) >= {"run_command", "required_inputs", "schema_version"}


@pytest.mark.parametrize("schema", [None, True, "1", 2])
def test_schema_version_is_an_explicit_supported_integer(tmp_path: Path, schema: object) -> None:
    write_manifest(tmp_path, schema_version=schema)
    assert "schema_version" in blocked_codes(inspect_experiment(tmp_path))


@pytest.mark.parametrize("command", [None, "", "   ", [], "python\x00 experiment.py"])
def test_execution_entry_point_must_be_declared(tmp_path: Path, command: object) -> None:
    write_manifest(tmp_path, run_command=command)
    assert "run_command" in blocked_codes(inspect_experiment(tmp_path))


@pytest.mark.parametrize("path", ["missing.py", "empty.py", "a-directory"])
def test_required_package_file_must_exist_and_be_nonempty(tmp_path: Path, path: str) -> None:
    write_manifest(tmp_path, required_files=[path])
    (tmp_path / "generated" / "empty.py").touch()
    (tmp_path / "generated" / "a-directory").mkdir()
    assert "required_files[0]" in blocked_codes(inspect_experiment(tmp_path))


@pytest.mark.parametrize("path", [
    "generated/experiment.py", r"generated\experiment.py", "Generated/experiment.py",
    "../experiment.py", "nested/../../experiment.py", "/tmp/experiment.py",
    "C:/experiment.py", r"\\server\share\experiment.py", ".env", ".env.production",
    ".ssh/id_rsa", "credentials", "file.txt:stream", "NUL", None, 123,
])
def test_unsafe_required_files_are_blocked(tmp_path: Path, path: object) -> None:
    write_manifest(tmp_path, required_files=[path])
    assert "required_files[0]" in blocked_codes(inspect_experiment(tmp_path))


@pytest.mark.parametrize("path", ["generated", "generated/src", "..", "/tmp", "C:/", None, 42])
def test_working_directory_cannot_escape_or_repeat_generated(tmp_path: Path, path: object) -> None:
    write_manifest(tmp_path, working_directory=path)
    assert "working_directory" in blocked_codes(inspect_experiment(tmp_path))


def test_nested_working_directory_does_not_rebase_required_files(tmp_path: Path) -> None:
    write_manifest(tmp_path, working_directory="src")
    (tmp_path / "generated" / "src").mkdir()
    assert inspect_experiment(tmp_path)["ready"] is True


def test_relative_input_uses_declared_working_directory(tmp_path: Path) -> None:
    write_manifest(tmp_path, working_directory="src", required_inputs=[
        {"name": "dataset", "path": "data.bin", "kind": "file", "scope": "local"},
    ])
    source = tmp_path / "generated" / "src"
    source.mkdir()
    (source / "data.bin").write_bytes(b"dataset")
    assert not (tmp_path / "generated" / "data.bin").exists()
    assert inspect_experiment(tmp_path)["ready"] is True


@pytest.mark.parametrize("working_directory", ["../outside", "missing"])
def test_invalid_working_directory_prevents_local_input_stat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, working_directory: str) -> None:
    write_manifest(tmp_path, working_directory=working_directory, required_inputs=[
        {"name": "dataset", "path": "data.bin", "kind": "file", "scope": "local"},
    ])
    original_stat = Path.stat

    def guarded_stat(path: Path, *args: object, **kwargs: object):
        assert path.name != "data.bin"
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", guarded_stat)
    assert blocked_codes(inspect_experiment(tmp_path)) >= {"working_directory", "required_inputs[0]"}


def test_missing_working_directory_blocks_execution(tmp_path: Path) -> None:
    write_manifest(tmp_path, working_directory="missing")
    assert "working_directory" in blocked_codes(inspect_experiment(tmp_path))


def test_local_absolute_input_is_checked_without_reading_contents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = tmp_path / "dataset.csv"
    data.write_text("private sample contents must not be returned", encoding="utf-8")
    write_manifest(tmp_path, required_inputs=[{"name": "dataset", "path": str(data), "kind": "file", "scope": "local"}])
    original_read = Path.read_text

    def guarded_read(path: Path, *args: object, **kwargs: object) -> str:
        assert path.name == "experiment_manifest.json"
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read)
    result = inspect_experiment(tmp_path)
    assert result["ready"] is True
    assert "private sample contents" not in json.dumps(result)


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_remote_inputs_are_unknown_and_block_immediate_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    remote_path = "/autodl-tmp/private/dataset"
    write_manifest(tmp_path, required_inputs=[{"name": "dataset", "path": remote_path, "kind": kind, "scope": "remote"}])
    original_stat = Path.stat

    def guarded_stat(path: Path, *args: object, **kwargs: object):
        assert str(path).replace("\\", "/") != remote_path
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", guarded_stat)
    result = inspect_experiment(tmp_path)
    assert "required_inputs[0]" in blocked_codes(result)
    remote = next(item for item in result["unverified"] if item["code"] == "required_inputs[0]")
    assert remote["status"] == "unverified"
    assert "unknown" in remote["message"]
    assert "missing" not in remote["message"]


def test_identical_relative_paths_distinguish_local_missing_from_remote_unknown(tmp_path: Path) -> None:
    write_manifest(tmp_path, required_inputs=[
        {"name": scope, "path": "missing-data.csv", "kind": "file", "scope": scope}
        for scope in ("local", "remote")
    ])
    result = inspect_experiment(tmp_path)
    assert blocked_codes(result) == {"required_inputs[0]", "required_inputs[1]"}
    local, remote = result["blocking_issues"]
    assert local["status"] == "blocked" and "missing" in local["message"]
    assert remote["status"] == "unverified" and "unknown" in remote["message"]


def test_empty_local_input_file_blocks_but_directory_can_be_declared(tmp_path: Path) -> None:
    generated = tmp_path / "generated"
    write_manifest(tmp_path, required_inputs=[
        {"name": "file", "path": "empty.csv", "kind": "file", "scope": "local"},
        {"name": "dir", "path": "data", "kind": "directory", "scope": "local"},
    ])
    (generated / "empty.csv").touch()
    (generated / "data").mkdir()
    assert blocked_codes(inspect_experiment(tmp_path)) == {"required_inputs[0]"}


@pytest.mark.parametrize("item", [None, {}, {"name": "x", "path": "x", "kind": "file"},
                                      {"name": "x", "path": "x", "kind": "file", "scope": "unknown"}])
def test_input_declarations_require_explicit_kind_and_scope(tmp_path: Path, item: object) -> None:
    write_manifest(tmp_path, required_inputs=[item])
    assert "required_inputs[0]" in blocked_codes(inspect_experiment(tmp_path))


@pytest.mark.parametrize("path", ["../outside.csv", "generated/data.csv", ".env", ".ssh/id_rsa"])
def test_local_relative_input_cannot_escape_or_target_secrets(tmp_path: Path, path: str) -> None:
    write_manifest(tmp_path, required_inputs=[{"name": "dataset", "path": path, "kind": "file", "scope": "local"}])
    assert "required_inputs[0]" in blocked_codes(inspect_experiment(tmp_path))


def test_absolute_sensitive_input_is_blocked_without_reading(tmp_path: Path) -> None:
    secret = tmp_path / ".env"
    secret.write_text("SECRET=do-not-read", encoding="utf-8")
    write_manifest(tmp_path, required_inputs=[{"name": "secret", "path": str(secret), "kind": "file", "scope": "local"}])
    result = inspect_experiment(tmp_path)
    assert "required_inputs[0]" in blocked_codes(result)
    assert "SECRET=" not in json.dumps(result)


@pytest.mark.parametrize("path", [r"\\server\share\dataset.csv", "//server/share/dataset.csv", r"\\?\C:\dataset.csv"])
def test_local_scope_never_stats_network_or_device_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path: str) -> None:
    write_manifest(tmp_path, required_inputs=[{"name": "dataset", "path": path, "kind": "file", "scope": "local"}])
    original_stat = Path.stat

    def guarded_stat(target: Path, *args: object, **kwargs: object):
        assert not str(target).replace("\\", "/").startswith("//")
        return original_stat(target, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", guarded_stat)
    assert "required_inputs[0]" in blocked_codes(inspect_experiment(tmp_path))


@pytest.mark.parametrize("criteria", [None, {}, {"required_metrics": []}, {"required_metrics": [""]}, {"required_metrics": "accuracy"}])
def test_required_metric_names_must_be_explicit(tmp_path: Path, criteria: object) -> None:
    write_manifest(tmp_path, success_criteria=criteria)
    assert "success_criteria" in blocked_codes(inspect_experiment(tmp_path))


def test_explicit_process_exit_validation_does_not_claim_research_measurement(tmp_path: Path) -> None:
    write_manifest(tmp_path, success_criteria={"validation": "process_exit", "required_metrics": []})
    result = inspect_experiment(tmp_path)
    assert result["ready"] is True
    assert result["measurement_validation"] == "not_requested"
    assert any(item["code"] == "measurement_validation" and "research effectiveness" in item["message"]
               for item in result["unverified"])


def test_process_exit_validation_still_requires_explicit_metric_declaration(tmp_path: Path) -> None:
    write_manifest(tmp_path, success_criteria={"validation": "process_exit"})
    assert "success_criteria" in blocked_codes(inspect_experiment(tmp_path))


def test_symlink_cannot_escape_generated_even_within_workspace(tmp_path: Path) -> None:
    write_manifest(tmp_path, required_files=["alias.py"])
    outside = tmp_path / "outside.py"
    outside.write_text("must not inspect", encoding="utf-8")
    try:
        (tmp_path / "generated" / "alias.py").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"Symlinks unavailable: {exc}")
    assert "required_files[0]" in blocked_codes(inspect_experiment(tmp_path))


def test_manifest_symlink_to_sensitive_file_is_never_read(tmp_path: Path) -> None:
    generated = tmp_path / "generated"
    generated.mkdir()
    secret = generated / ".env"
    secret.write_text("SECRET=do-not-read", encoding="utf-8")
    try:
        (generated / "experiment_manifest.json").symlink_to(secret)
    except OSError as exc:
        pytest.skip(f"Symlinks unavailable: {exc}")
    result = inspect_experiment(tmp_path)
    assert "manifest" in blocked_codes(result)
    assert "SECRET=" not in json.dumps(result)
