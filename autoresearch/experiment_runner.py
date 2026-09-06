"""Portable, opt-in execution of bounded experiments (Python standard library only).

Copy this file with an experiment bundle and run::

    python experiment_runner.py --manifest experiment.json --output runs
    python experiment_runner.py --manifest experiment.json --output runs --execute

The first command only checks prerequisites. Each invocation creates an independent
``run-*/results/metrics.json`` report and atomically publishes ``metrics.json`` in
the output directory. An exclusive lock prevents concurrent use of that directory;
after a runner crash, inspect its recorded PID before manually removing the lock.
Commands run literally as argument arrays;
the runner never installs packages or downloads missing inputs. Metric files must
not already exist: use a fresh working directory for another experimental run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import signal
import subprocess
import tempfile
import time
from typing import Any, Sequence


class ManifestError(ValueError):
    """A manifest cannot be safely or unambiguously executed."""


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise ManifestError(f"{label} must be a nonempty string without NUL characters")
    return value


def _relative_path(value: Any, label: str, base: Path) -> Path:
    value = _text(value, label)
    # Check both formats even when preflight runs on another operating system.
    windows = PureWindowsPath(value)
    if PurePosixPath(value).is_absolute() or windows.drive or windows.root:
        raise ManifestError(f"{label} must be relative")
    target = (base / value.replace("\\", "/")).resolve()
    if not target.is_relative_to(base):
        raise ManifestError(f"{label} must stay inside its package directory")
    return target


def load_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Read and validate schema, without executing commands or checking inputs."""
    path = Path(manifest_path).resolve()
    try:
        if path.stat().st_size > 200_000:
            raise ManifestError("Manifest exceeds 200 KB")
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ManifestError(f"Cannot read manifest: {exc}") from exc
    return validate_manifest(raw, path.parent)


def validate_manifest(raw: Any, manifest_directory: str | Path) -> dict[str, Any]:
    """Return a normalized manifest with absolute paths or raise ManifestError."""
    if not isinstance(raw, dict):
        raise ManifestError("Manifest must be a JSON object")
    if type(raw.get("schema_version")) is not int or raw["schema_version"] != 1:
        raise ManifestError("schema_version must be the integer 1")
    base = Path(manifest_directory).resolve()
    working = _relative_path(raw.get("working_directory"), "working_directory", base)
    required_files = raw.get("required_files")
    if not isinstance(required_files, list):
        raise ManifestError("required_files must be an array of relative file paths")
    files = [str(_relative_path(value, f"required_files[{index}]", base))
             for index, value in enumerate(required_files)]
    required_inputs = raw.get("required_inputs")
    if not isinstance(required_inputs, list):
        raise ManifestError("required_inputs must be an array")
    inputs: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, item in enumerate(required_inputs):
        label = f"required_inputs[{index}]"
        if not isinstance(item, dict):
            raise ManifestError(f"{label} must be an object")
        name = _text(item.get("name"), f"{label}.name")
        if name in names:
            raise ManifestError(f"Duplicate input name: {name}")
        names.add(name)
        value = _text(item.get("path"), f"{label}.path")
        if item.get("kind") not in ("file", "directory"):
            raise ManifestError(f"{label}.kind must be file or directory")
        if item.get("scope") not in ("local", "remote"):
            raise ManifestError(f"{label}.scope must be local or remote")
        # Preserve foreign absolute paths for a local-only preflight. They are
        # blocked during execution on the wrong platform instead of reinterpreted.
        foreign_absolute = (
            os.name != "nt" and bool(PureWindowsPath(value).drive)
        ) or (os.name == "nt" and value.startswith("/"))
        input_path = Path(value)
        if not input_path.is_absolute() and not foreign_absolute:
            input_path = working / value.replace("\\", "/")
        inputs.append({"name": name, "path": value if foreign_absolute else str(input_path.resolve()),
                       "kind": item["kind"], "scope": item["scope"],
                       "foreign_absolute": foreign_absolute})
    raw_steps = raw.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ManifestError("steps must be a nonempty array")
    steps: list[dict[str, Any]] = []
    names.clear()
    metrics_paths: set[Path] = set()
    for index, item in enumerate(raw_steps):
        label = f"steps[{index}]"
        if not isinstance(item, dict):
            raise ManifestError(f"{label} must be an object")
        name = _text(item.get("name"), f"{label}.name")
        if name in names:
            raise ManifestError(f"Duplicate step name: {name}")
        names.add(name)
        command = item.get("command")
        if not isinstance(command, list) or not command:
            raise ManifestError(f"{label}.command must be a nonempty argument array")
        for argument in command:
            _text(argument, f"{label}.command argument")
        environment = item.get("env", {})
        if not isinstance(environment, dict):
            raise ManifestError(f"{label}.env must be an object of string values")
        for key, value in environment.items():
            _text(key, f"{label}.env key")
            if "=" in key or not isinstance(value, str) or "\0" in value:
                raise ManifestError(f"{label}.env requires keys without '=' and string values without NUL")
        timeout = item.get("timeout_seconds")
        if type(timeout) is not int or timeout <= 0:
            raise ManifestError(f"{label}.timeout_seconds must be a positive integer")
        required = item.get("required_metrics", [])
        if not isinstance(required, list):
            raise ManifestError(f"{label}.required_metrics must be an array")
        for metric in required:
            _text(metric, f"{label}.required_metrics entry")
        if len(required) != len(set(required)):
            raise ManifestError(f"{label}.required_metrics contains duplicates")
        metrics_path = None
        if "metrics_file" in item:
            metrics_path = _relative_path(item["metrics_file"], f"{label}.metrics_file", working)
            if not metrics_path.is_relative_to(working) or metrics_path == working:
                raise ManifestError(f"{label}.metrics_file must be inside working_directory")
            if metrics_path in metrics_paths:
                raise ManifestError(f"Duplicate metrics_file: {metrics_path}")
            metrics_paths.add(metrics_path)
        elif required:
            raise ManifestError(f"{label}.required_metrics requires metrics_file")
        steps.append({"name": name, "command": list(command), "timeout_seconds": timeout,
                      "metrics_file": str(metrics_path) if metrics_path else None,
                      "required_metrics": list(required), "env": dict(environment)})
    return {"schema_version": 1, "working_directory": str(working),
            "required_files": files, "required_inputs": inputs, "steps": steps}


def _preflight(manifest: dict[str, Any], *, check_remote_inputs: bool) -> dict[str, Any]:
    issues: list[str] = []
    if not Path(manifest["working_directory"]).is_dir():
        issues.append(f"Missing working_directory: {manifest['working_directory']}")
    for filename in manifest["required_files"]:
        if not Path(filename).is_file():
            issues.append(f"Missing required file: {filename}")
        elif Path(filename).stat().st_size == 0:
            issues.append(f"Empty required file: {filename}")
    inputs: list[dict[str, Any]] = []
    for item in manifest["required_inputs"]:
        info = {key: item[key] for key in ("name", "path", "kind", "scope")}
        if item["scope"] == "remote" and not check_remote_inputs:
            info["status"] = "pending_remote"
        else:
            path = Path(item["path"])
            exists = not item["foreign_absolute"] and (
                path.is_file() if item["kind"] == "file" else path.is_dir()
            )
            info["status"] = "present" if exists else "missing"
            if not exists:
                issues.append(f"Missing required {item['scope']} {item['kind']} "
                              f"{item['name']}: {item['path']}")
            elif item["kind"] == "file" and path.stat().st_size == 0:
                info["status"] = "empty"
                issues.append(f"Empty required input {item['name']}: {item['path']}")
        inputs.append(info)
    for step in manifest["steps"]:
        if step["metrics_file"] and os.path.lexists(step["metrics_file"]):
            issues.append(f"Step {step['name']} metrics_file already exists; use a fresh "
                          f"working directory to prevent stale metrics: {step['metrics_file']}")
    pending = any(item["status"] == "pending_remote" for item in inputs)
    return {"status": "blocked" if issues else "ready", "issues": issues,
            "execution_ready": not issues and not pending,
            "remote_inputs_pending": pending, "working_directory": manifest["working_directory"],
            "inputs": inputs,
            "steps": [{**{key: value for key, value in step.items() if key != "env"},
                       "environment_keys": sorted(step["env"])} for step in manifest["steps"]]}


def preflight_manifest(manifest_path: str | Path, *, check_remote_inputs: bool = True) -> dict[str, Any]:
    """Inspect prerequisites; local backends can defer checking remote input paths."""
    try:
        manifest = load_manifest(manifest_path)
        return _preflight(manifest, check_remote_inputs=check_remote_inputs)
    except (ManifestError, OSError) as exc:
        return {"status": "blocked", "issues": [str(exc)], "execution_ready": False,
                "remote_inputs_pending": False, "inputs": [], "steps": []}


def _atomic_json(path: Path, report: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _write_report(path: Path, report: dict[str, Any]) -> None:
    _atomic_json(path, report)
    _atomic_json(Path(report["output_directory"]) / "metrics.json", report)


def _stop_process_tree(process: subprocess.Popen[bytes]) -> None:
    """Terminate only this step's process tree, including children of its leader."""
    if os.name == "nt":
        if process.poll() is None:
            taskkill = str(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "taskkill.exe")
            try:
                result = subprocess.run([taskkill, "/PID", str(process.pid), "/T", "/F"],
                                        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                        stderr=subprocess.PIPE, timeout=10, check=False, shell=False)
                if result.returncode and process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)
                    raise RuntimeError(f"Windows process-tree cleanup could not be verified for {process.pid}")
            except (OSError, subprocess.TimeoutExpired):
                process.kill()
                process.wait(timeout=5)
                raise RuntimeError("Process stopped, but Windows process-tree cleanup could not be verified")
        process.wait(timeout=5)
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=0.5)
    except subprocess.TimeoutExpired:
        pass
    # The group can still contain children after its leader has exited.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def _read_metrics(path: str, required: list[str]) -> dict[str, int | float]:
    try:
        if Path(path).stat().st_size > 2_000_000:
            raise ManifestError("Metric file exceeds 2 MB")
        raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ManifestError(f"Cannot read fresh metrics_file {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ManifestError("metrics_file must contain a JSON object of numeric metrics")
    missing = [name for name in required if name not in raw]
    if missing:
        raise ManifestError("Missing required metrics: " + ", ".join(missing))
    for name, value in raw.items():
        if type(value) not in (int, float) or (isinstance(value, float) and not math.isfinite(value)):
            raise ManifestError(f"Metric {name} must be a finite number")
    return raw


def _run_manifest_unlocked(manifest_path: str | Path, output: Path, *, execute: bool) -> dict[str, Any]:
    started = time.monotonic()
    run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=output))
    results = run_dir / "results"
    results.mkdir()
    report_path = results / "metrics.json"
    manifest: dict[str, Any] = {}
    try:
        manifest = load_manifest(manifest_path)
        preflight = _preflight(manifest, check_remote_inputs=True)
    except (ManifestError, OSError) as exc:
        preflight = {"status": "blocked", "issues": [str(exc)], "execution_ready": False,
                     "remote_inputs_pending": False, "inputs": [], "steps": []}
    steps = [{**step, "status": "not_run", "exit_code": None, "elapsed_seconds": 0.0,
              "metrics": {}, "stdout_log": None, "stderr_log": None}
             for step in preflight["steps"]]
    report: dict[str, Any] = {"schema_version": 1, "runner": "autoresearch", "manifest": str(Path(manifest_path).resolve()),
                              "run_id": run_dir.name, "output_directory": str(output),
                              "run_directory": str(run_dir), "report_path": str(report_path),
                              "execute": execute, "status": preflight["status"],
                              "exit_code": None, "elapsed_seconds": 0.0,
                              "preflight": preflight, "steps": steps, "metrics": {}}
    if preflight["status"] == "blocked":
        report["exit_code"] = 2
    elif execute:
        report["status"] = "running"
    _write_report(report_path, report)
    if report["status"] == "running":
        for index, step in enumerate(steps):
            step_dir = run_dir / f"step-{index + 1:03d}"
            step_dir.mkdir()
            stdout_path, stderr_path = step_dir / "stdout.log", step_dir / "stderr.log"
            step.update(status="running", stdout_log=str(stdout_path), stderr_log=str(stderr_path))
            step_start = time.monotonic()
            process = None
            _write_report(report_path, report)
            try:
                if step["metrics_file"] and os.path.lexists(step["metrics_file"]):
                    raise ManifestError("metrics_file appeared before this step; refusing stale metrics: "
                                        + step["metrics_file"])
                environment = os.environ.copy()
                environment.update(manifest["steps"][index]["env"])
                environment.update(AUTORESEARCH_RUN_DIR=str(run_dir), AUTORESEARCH_STEP_DIR=str(step_dir))
                if step["metrics_file"]:
                    environment["AUTORESEARCH_METRICS_FILE"] = step["metrics_file"]
                options: dict[str, Any] = {"start_new_session": True} if os.name != "nt" else {
                    "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
                with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                    process = subprocess.Popen(step["command"], cwd=preflight["working_directory"],
                                               env=environment, stdin=subprocess.DEVNULL,
                                               stdout=stdout, stderr=stderr, shell=False, **options)
                    try:
                        step["exit_code"] = process.wait(timeout=step["timeout_seconds"])
                    except subprocess.TimeoutExpired:
                        step["status"] = "timed_out"
                        step["error"] = f"Step exceeded {step['timeout_seconds']} seconds"
                    finally:
                        if os.name != "nt" or process.poll() is None:
                            _stop_process_tree(process)
                        step["exit_code"] = process.returncode
                if step["status"] == "timed_out":
                    report.update(status="timed_out", exit_code=124)
                elif step["exit_code"] != 0:
                    step.update(status="failed", error=f"Command exited with code {step['exit_code']}")
                    report.update(status="failed", exit_code=1)
                else:
                    if step["metrics_file"]:
                        step["metrics"] = _read_metrics(step["metrics_file"], step["required_metrics"])
                        report["metrics"][step["name"]] = step["metrics"]
                    step["status"] = "succeeded"
            except KeyboardInterrupt:
                step.update(status="interrupted", error="Interrupted by user")
                report.update(status="interrupted", exit_code=130)
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                step.update(status="failed", error=str(exc))
                report.update(status="failed", exit_code=1)
            finally:
                step["elapsed_seconds"] = round(time.monotonic() - step_start, 6)
                report["elapsed_seconds"] = round(time.monotonic() - started, 6)
                _write_report(report_path, report)
            if step["status"] != "succeeded":
                break
        else:
            report.update(status="succeeded", exit_code=0)
    report["elapsed_seconds"] = round(time.monotonic() - started, 6)
    _write_report(report_path, report)
    return report


def run_manifest(manifest_path: str | Path, output_dir: str | Path, *, execute: bool = False) -> dict[str, Any]:
    """Preflight or execute sequentially, retaining a new report even on failure.

    Only one runner may use an output directory at a time. Existing metric files
    are never deleted or moved. Environment values are omitted from the report.
    """
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".autoresearch-run.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(f"Output directory is locked: {lock}; check the recorded process "
                              "before removing a stale lock") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "created_at_unix": time.time()}, handle)
        return _run_manifest_unlocked(manifest_path, output, execute=execute)
    finally:
        lock.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute", action="store_true", help="Execute only after all prerequisites pass")
    args = parser.parse_args(argv)
    try:
        report = run_manifest(args.manifest, args.output, execute=args.execute)
    except OSError as exc:
        parser.exit(2, f"Cannot create experiment report: {exc}\n")
    print(json.dumps(report, ensure_ascii=True, allow_nan=False))
    return report["exit_code"] if report["exit_code"] is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
