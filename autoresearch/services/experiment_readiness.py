"""Read-only validation of an experiment package, without running its code.

``ready`` means that the declared package passes static local checks. It does
not establish that dependencies, GPU access, data semantics, or metrics work.
Remote inputs require a separate check on the machine that will run the job.
"""
from __future__ import annotations

import json
import stat
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from autoresearch.services.artifacts import ArtifactStore, UnsafeArtifactPath


MANIFEST_PATH = "generated/experiment_manifest.json"


class _ReadOnlyArtifacts(ArtifactStore):
    def __init__(self, workspace: Path) -> None:
        # ArtifactStore normally creates its root. An inspection must not.
        self.workspace = workspace.resolve()


def _package_target(store: ArtifactStore, path: str, *, directory: bool = False) -> Path:
    generated = store._safe_target("generated")
    if directory and path == ".":
        return generated
    parts = store._relative_parts(path)
    if parts[0].casefold() == "generated":
        raise UnsafeArtifactPath("Paths must be relative to generated; remove the generated/ prefix.")
    target = store._safe_target("/".join(("generated", *parts)))
    if not target.is_relative_to(generated) or (target == generated and not directory):
        raise UnsafeArtifactPath("The path escapes the generated directory.")
    return target


def _local_input_target(store: ArtifactStore, path: str, working_directory: Path) -> Path:
    if path.replace("\\", "/").startswith("//"):
        # A UNC/network share stat may establish a network connection. Such
        # inputs belong in remote scope and need validation on that host.
        raise UnsafeArtifactPath("Network and device paths are not local inputs.")
    absolute = PurePosixPath(path).is_absolute() or PureWindowsPath(path).is_absolute()
    if not absolute:
        parts = store._relative_parts(path)
        if parts[0].casefold() == "generated":
            raise UnsafeArtifactPath("Input paths must be relative to working_directory, without a generated/ prefix.")
        relative = working_directory.relative_to(store.workspace)
        target = store._safe_target("/".join((*relative.parts, *parts)))
        if not target.is_relative_to(store._safe_target("generated")):
            raise UnsafeArtifactPath("The input path escapes the generated directory.")
        return target
    target = Path(path)
    if not target.is_absolute():
        raise UnsafeArtifactPath("The absolute local input path is not valid on this operating system.")
    # Absolute data locations are allowed, but sensitive names, traversal,
    # Windows devices, and alternate data streams are never accepted.
    store._relative_parts("/".join(target.parts[1:]))
    target = target.resolve()
    store._relative_parts("/".join(target.parts[1:]))
    return target


def inspect_experiment(workspace: Path) -> dict[str, Any]:
    """Inspect generated/experiment_manifest.json and declared local paths.

    The function neither executes commands nor imports experiment modules. Only
    the bounded manifest is read; input and package files receive metadata
    checks. Required package files are relative to ``generated``; relative
    required inputs use the declared working directory within ``generated``.
    Explicit ``required_inputs: []`` declares that no external inputs are
    needed. Unknown fields have no effect on this inspection.
    """
    checks: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    unverified: list[dict[str, Any]] = []
    measurement_validation = "unverified"

    def record(code: str, status: str, message: str, *, blocking: bool = False,
               path: str | None = None) -> None:
        item: dict[str, Any] = {"code": code, "status": status, "message": message}
        if path is not None:
            item["path"] = path
        checks.append(item)
        if blocking or status == "blocked":
            blockers.append(item.copy())
        if status == "unverified":
            unverified.append(item.copy())

    def result() -> dict[str, Any]:
        return {
            "status": "blocked" if blockers else "ready",
            "ready": not blockers,
            "checks": checks,
            "blocking_issues": blockers,
            "unverified": unverified,
            "manifest_path": MANIFEST_PATH,
            "execution_status": "not_executed_by_readiness_check",
            "measurement_validation": measurement_validation,
        }

    def check_metadata(target: Path, kind: str, code: str, path: str) -> bool:
        try:
            metadata = target.stat()
        except FileNotFoundError:
            record(code, "blocked", f"Required local {kind} is missing.", path=path)
            return False
        except OSError:
            record(code, "blocked", f"Required local {kind} cannot be inspected.", path=path)
            return False
        expected = stat.S_ISREG(metadata.st_mode) if kind == "file" else stat.S_ISDIR(metadata.st_mode)
        if not expected:
            record(code, "blocked", f"Required local path must be a {kind}.", path=path)
        elif kind == "file" and metadata.st_size == 0:
            record(code, "blocked", "Required local file is empty.", path=path)
        else:
            record(code, "passed", f"Required local {kind} exists"
                   + (" and is non-empty." if kind == "file" else "."), path=path)
            return True
        return False

    record("runtime_validation", "unverified",
           "Static inspection only: commands, dependencies, GPU access, data contents, "
           "and metric production have not been validated.")
    try:
        store = _ReadOnlyArtifacts(Path(workspace))
        # Check both the manifest and its containing directory before reading.
        manifest_path = _package_target(store, "experiment_manifest.json")
        if not manifest_path.is_file():
            record("manifest", "blocked", "Create generated/experiment_manifest.json before running.")
            return result()
        manifest = json.loads(store.read_text(MANIFEST_PATH, max_bytes=200_000))
    except (OSError, ValueError, RuntimeError):
        record("manifest", "blocked", "Manifest is unreadable, unsafe, invalid JSON, or exceeds 200 KB.")
        return result()
    if not isinstance(manifest, dict):
        record("manifest", "blocked", "Manifest must be a JSON object.")
        return result()
    record("manifest", "passed", "Manifest is a readable JSON object.")

    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        record("schema_version", "blocked", "Set schema_version to the supported integer value 1.")
    else:
        record("schema_version", "passed", "Manifest schema_version is 1.")

    command = manifest.get("run_command")
    if not isinstance(command, str) or not command.strip() or "\x00" in command:
        record("run_command", "blocked", "Declare a non-empty run_command string; legacy manifests need an explicit entry point.")
    else:
        record("run_command", "passed", "An execution command is declared; it has not been executed or validated.")

    working_directory = manifest.get("working_directory", ".")
    working_directory_target: Path | None = None
    try:
        target = _package_target(store, working_directory, directory=True)
        if check_metadata(target, "directory", "working_directory", working_directory):
            working_directory_target = target
    except (OSError, ValueError, RuntimeError):
        record("working_directory", "blocked", "working_directory must be '.' or a safe directory relative to generated, without a generated/ prefix.")

    required_files = manifest.get("required_files")
    if not isinstance(required_files, list) or not required_files:
        record("required_files", "blocked", "Declare a non-empty required_files list with package files relative to generated.")
    else:
        for index, path in enumerate(required_files):
            code = f"required_files[{index}]"
            try:
                target = _package_target(store, path)
                check_metadata(target, "file", code, path)
            except (OSError, ValueError, RuntimeError):
                record(code, "blocked", "Required file must be a safe path relative to generated, without a generated/ prefix.")

    required_inputs = manifest.get("required_inputs")
    if not isinstance(required_inputs, list):
        record("required_inputs", "blocked", "Declare required_inputs with name, path, kind, and scope, or [] when no external inputs are required.")
    else:
        if not required_inputs:
            record("required_inputs", "passed", "The manifest explicitly declares no external inputs.")
        for index, item in enumerate(required_inputs):
            code = f"required_inputs[{index}]"
            if not isinstance(item, dict):
                record(code, "blocked", "Each required input must be an object with name, path, kind, and scope.")
                continue
            name, path, kind, scope = (item.get(key) for key in ("name", "path", "kind", "scope"))
            if (not isinstance(name, str) or not name.strip()
                    or not isinstance(path, str) or not path.strip() or "\x00" in path
                    or kind not in ("file", "directory") or scope not in ("local", "remote")):
                record(code, "blocked", "Input requires non-empty name/path, kind file|directory, and scope local|remote.")
                continue
            if scope == "remote":
                record(code, "unverified", "Remote input existence and contents are unknown; verify it on the execution host before running.",
                       blocking=True, path=path)
                continue
            if working_directory_target is None:
                record(code, "blocked", "Local input was not inspected because working_directory is invalid or unavailable.")
                continue
            try:
                target = _local_input_target(store, path, working_directory_target)
                check_metadata(target, kind, code, path)
            except (OSError, ValueError, RuntimeError):
                record(code, "blocked", "Local input path is unsafe or cannot be resolved; relative paths must stay within generated.")

    criteria = manifest.get("success_criteria")
    metrics = criteria.get("required_metrics") if isinstance(criteria, dict) else None
    if isinstance(criteria, dict) and criteria.get("validation") == "process_exit" and metrics == []:
        measurement_validation = "not_requested"
        record("success_criteria", "passed", "Process exit validation is declared without research metrics.")
        record("measurement_validation", "unverified",
               "Only process completion is requested, not research effectiveness; even process completion still requires execution.")
    elif (not isinstance(metrics, list) or not metrics
            or any(not isinstance(metric, str) or not metric.strip() for metric in metrics)):
        record("success_criteria", "blocked", "Declare success_criteria.required_metrics as a non-empty list of metric names.")
    else:
        record("success_criteria", "passed", "Required metric names are declared; metric values have not been checked.")
    return result()
