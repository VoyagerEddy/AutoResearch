"""Evidence-bound research records; ChatGPT makes decisions, this service never executes.

Implementation coverage is an explicit manifest declaration checked against
files, not a claim that a hypothesis works. Full Colab results and acceptance
checks remain separate. All plans and decisions have immutable identifiers.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from typing import Any
import uuid
import zipfile

from ..config import Settings
from ..db import Database
from ..domain import utc_now
from .artifacts import ArtifactStore
from .colab import ColabService
from .colab_data import validate_completion_details


class ResearchWorkflowConflict(ValueError):
    """A decision no longer describes the current research evidence."""


_LOCK = threading.RLock()
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}\Z")
_DIGEST = re.compile(r"[a-f0-9]{64}\Z")
_MAX_JSON = 2_000_000


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()


class _ReadOnlyStore(ArtifactStore):
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()


def _safe(store: ArtifactStore, relative: str) -> Path:
    """ArtifactStore validation plus rejection of even in-workspace links."""
    parts = store._relative_parts(relative)
    current = store.workspace
    for part in parts:
        current = current / part
        if current.is_symlink() or bool(getattr(current, "is_junction", lambda: False)()):
            raise ValueError("Research workflow paths must not contain linked files or directories")
    return store._safe_target(relative)


def _read(store: ArtifactStore, relative: str, *, optional: bool = False) -> Any:
    path = _safe(store, relative)
    if optional and not path.exists():
        return None
    if not path.is_file() or path.stat().st_size > _MAX_JSON:
        raise ValueError("Research workflow JSON artifact is missing or exceeds 2 MB")
    return json.loads(path.read_text(encoding="utf-8"),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def _write(store: ArtifactStore, relative: str, value: Any, *, immutable: bool = False) -> None:
    path = _safe(store, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    if len(text.encode()) > _MAX_JSON:
        raise ValueError("Research workflow record exceeds 2 MB")
    if immutable:
        # Exclusive creation preserves history, including under concurrent writers.
        with path.open("x", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        return
    descriptor, filename = tempfile.mkstemp(prefix=".workflow-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(filename, path)
    finally:
        Path(filename).unlink(missing_ok=True)


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{name} must be a safe nonempty identifier")
    return value


def _items(value: Any, name: str, *, allow_empty: bool = False) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > 500 or (not value and not allow_empty):
        raise ValueError(f"{name} must be a nonempty list of at most 500 objects")
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(f"{name} entries must be objects")
        identity = _identifier(item.get("id"), f"{name}.id")
        if identity in seen:
            raise ValueError(f"Duplicate {name} identifier: {identity}")
        seen.add(identity)
        if "status" in item or item.get("completed") or item.get("tested"):
            raise ValueError("Research progress is derived from evidence, not supplied in the plan")
    return value


def _references(value: Any, known: set[str], name: str) -> list[str]:
    if (not isinstance(value, list) or not value or len(value) > 500
            or any(not isinstance(item, str) or item not in known for item in value)
            or len(set(value)) != len(value)):
        raise ValueError(f"{name} must contain unique declared identifiers")
    return value


def _result_experiment_ids(manifest: dict[str, Any]) -> list[str]:
    coverage = manifest.get("research_coverage")
    declarations = coverage.get("experiments") if isinstance(coverage, dict) else None
    if not isinstance(declarations, list) or not declarations or any(not isinstance(item, dict) for item in declarations):
        raise ValueError("Prepared Colab source does not declare any research experiments")
    identifiers = [_identifier(item.get("id"), "experiment.id") for item in declarations]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Prepared Colab source has duplicate research experiment identifiers")
    config = manifest.get("colab", {})
    if not isinstance(config, dict):
        raise ValueError("Prepared Colab source has an invalid execution contract")
    if "research_experiment_ids" in config:
        return _references(config["research_experiment_ids"], set(identifiers), "colab.research_experiment_ids")
    if len(identifiers) == 1:
        return identifiers
    raise ValueError("Multiple research experiments are declared; Colab result attribution requires colab.research_experiment_ids")


class ResearchWorkflowService:
    def __init__(self, db: Database, settings: Settings) -> None:
        self.db, self.settings = db, settings

    def _store(self, project_id: str) -> _ReadOnlyStore:
        _identifier(project_id, "project_id")
        project = self.db.get_project(project_id)
        if project is None:
            raise ValueError("Research project does not exist")
        workspace = Path(project.workspace).resolve()
        if not workspace.is_relative_to(self.settings.workspace_dir.resolve()):
            raise ValueError("Research workspace is outside the configured workspace directory")
        return _ReadOnlyStore(workspace)

    @contextmanager
    def _transaction(self, store: ArtifactStore):
        with _LOCK:
            path = _safe(store, "research/workflow/workflow.lock")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a+b") as stream:
                if not path.stat().st_size:
                    stream.write(b"0")
                    stream.flush()
                deadline = time.monotonic() + 10
                while True:
                    stream.seek(0)
                    try:
                        if os.name == "nt":
                            import msvcrt
                            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                        else:
                            import fcntl
                            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise ResearchWorkflowConflict("Another research update is in progress") from None
                        time.sleep(.05)
                try:
                    yield
                finally:
                    stream.seek(0)
                    if os.name == "nt":
                        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        fcntl.flock(stream, fcntl.LOCK_UN)

    def _normalize_plan(self, store: ArtifactStore, value: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(value, dict) or len(json.dumps(value, allow_nan=False)) > _MAX_JSON:
            raise ValueError("Research plan must be a bounded JSON object")
        if value.get("schema_version", 1) != 1 or value.get("reasoning_owner", "chatgpt") != "chatgpt":
            raise ValueError("Research plan schema is 1 and reasoning_owner must be chatgpt")
        if not isinstance(value.get("title"), str) or not value["title"].strip():
            raise ValueError("Research plan requires a title")
        result = json.loads(json.dumps(value, allow_nan=False))
        result.pop("parent_plan_id", None)
        result.update(schema_version=1, reasoning_owner="chatgpt")
        hypotheses = _items(result.get("hypotheses"), "hypotheses")
        datasets = _items(result.get("datasets"), "datasets")
        experiments = _items(result.get("experiments"), "experiments")
        if not isinstance(result.get("success_criteria"), dict) or not result["success_criteria"]:
            raise ValueError("Research plan requires explicit success_criteria")
        for dataset in datasets:
            if dataset.get("scope") not in {"full", "acceptance", "evaluation"}:
                raise ValueError("Dataset scope must be full, acceptance, or evaluation")
            if dataset.get("labels", "unknown") not in {"available", "unavailable", "unknown", "not_required"}:
                raise ValueError("Dataset labels must be available, unavailable, unknown, or not_required")
            dataset.setdefault("labels", "unknown")
            dataset.setdefault("required", True)
            if type(dataset["required"]) is not bool:
                raise ValueError("Dataset required must be a boolean")
        for item in hypotheses + experiments:
            item.setdefault("required", True)
            if type(item["required"]) is not bool:
                raise ValueError("Research required must be a boolean")
            paths = item.get("implementation_files", [])
            if not isinstance(paths, list) or any(not isinstance(path, str) for path in paths):
                raise ValueError("implementation_files must be a list of generated-relative paths")
            for relative in paths:
                _safe(store, "generated/" + relative)
        for experiment in experiments:
            _references(experiment.get("hypothesis_ids"), {item["id"] for item in hypotheses}, "hypothesis_ids")
            _references(experiment.get("dataset_ids"), {item["id"] for item in datasets}, "dataset_ids")
            if experiment.get("scope") not in {"full", "acceptance"}:
                raise ValueError("Experiment scope must be full or acceptance")
            _identifier(experiment.get("manifest_id", experiment["id"]), "manifest_id")
            if experiment["scope"] == "full" and any(
                    item["scope"] == "acceptance" for item in datasets if item["id"] in experiment["dataset_ids"]):
                raise ValueError("A full experiment cannot use an acceptance-only dataset")
        metric_names = result["success_criteria"].get("required_metrics", [])
        if (not isinstance(metric_names, list) or len(metric_names) > 100
                or any(not isinstance(name, str) or not _IDENTIFIER.fullmatch(name) for name in metric_names)
                or len(set(metric_names)) != len(metric_names)):
            raise ValueError("success_criteria.required_metrics must contain unique metric names")
        return result

    def _source(self, store: ArtifactStore) -> tuple[dict[str, Any], dict[str, str], str]:
        manifest = _read(store, "generated/experiment_manifest.json", optional=True) or {}
        if not isinstance(manifest, dict):
            raise ValueError("Experiment manifest must be an object")
        generated = _safe(store, "generated")
        if not _safe(store, "generated/experiment_manifest.json").is_file():
            return manifest, {}, ColabService._source_identity([])
        # One canonical collector applies the exact same exclusions, private-file
        # checks and source limits as the notebook exporter and stale-source gate.
        colab = ColabService(self.db, self.settings)
        entries = colab._source_entries(generated)
        files = {relative: hashlib.sha256(content).hexdigest() for relative, content in entries}
        return manifest, files, colab._source_identity(entries)

    def _evidence(self, project_id: str, store: ArtifactStore, files: dict[str, str]) -> dict[str, Any]:
        job = _read(store, "colab/job.json", optional=True)
        result: dict[str, Any] = {"source": "colab", "job_status": "not_prepared", "tested": False,
                                  "full_data_completed": False, "research_experiment_ids": [],
                                  "reason": "No current Colab result"}
        if not isinstance(job, dict):
            result["snapshot_sha256"] = _digest(None)
            return result
        result.update(job_status=job.get("status"), runtime_id=job.get("runtime_id"),
                      snapshot_sha256=_digest(job))
        if job.get("status") != "completed" or not job.get("prepared"):
            result["reason"] = "Colab has not produced a completed current result"
            return result
        details = (job.get("last_observation") or {}).get("details") or {}
        if (not job.get("runtime_id") or details.get("runtime_id") != job["runtime_id"]
                or details.get("returncode") != 0
                or (job.get("last_observation") or {}).get("status") != "completed"):
            result["reason"] = "Completion is not bound to the current runtime"
            return result
        try:
            if ColabService(self.db, self.settings).source_matches_prepared_bundle(project_id) is not True:
                raise ValueError("Colab result does not match the current prepared source")
            criteria = job.get("success_criteria")
            if not criteria or criteria.get("validation") != "scientific_metrics":
                raise ValueError("A process exit is not measured scientific completion")
            completion = validate_completion_details(details, criteria)
            if completion != job.get("completion"):
                raise ValueError("Stored completion and current result differ")
            bundle = _safe(store, "colab/source.zip")
            if (not bundle.is_file() or bundle.stat().st_size > 100_000_000
                    or hashlib.sha256(bundle.read_bytes()).hexdigest() != job.get("bundle_sha256")):
                raise ValueError("Prepared source archive is missing or has changed")
            archived = {}
            prepared_manifest = None
            with zipfile.ZipFile(bundle) as archive:
                entries = archive.infolist()
                if sum(item.file_size for item in entries) > 100_000_000 or len(entries) > 10_000:
                    raise ValueError("Prepared source archive exceeds the source limits")
                for entry in entries:
                    if entry.is_dir():
                        continue
                    _safe(store, "generated/" + entry.filename)
                    if entry.filename in archived or entry.file_size > 5_000_000:
                        raise ValueError("Prepared source archive has duplicate or oversized entries")
                    content = archive.read(entry)
                    archived[entry.filename] = hashlib.sha256(content).hexdigest()
                    if entry.filename == "experiment_manifest.json":
                        prepared_manifest = json.loads(content.decode("utf-8-sig"))
            if archived != files:
                raise ValueError("Colab result belongs to different source code")
            if not isinstance(prepared_manifest, dict):
                raise ValueError("Prepared Colab source has no experiment manifest")
            experiment_ids = _result_experiment_ids(prepared_manifest)
            contract = prepared_manifest.get("colab", {})
            if "success_criteria" in contract and contract["success_criteria"] != criteria:
                raise ValueError("Current result criteria differ from the prepared Colab contract")
        except (ValueError, OSError, zipfile.BadZipFile, TypeError) as exc:
            result["reason"] = str(exc)
            return result
        full = completion["completion_kind"] == "campaign_complete" and completion["scientific_metrics_produced"] is True
        result.update(tested=True, full_data_completed=full, completion_kind=completion["completion_kind"],
                      metrics=completion["metrics"], metrics_file=completion["metrics_file"],
                      research_experiment_ids=experiment_ids,
                      validation_scope="full" if full else "acceptance",
                      reason="Current prepared scientific contract completed; research interpretation is pending")
        if "paired_seeds" in completion:
            result.update(paired_seeds=completion["paired_seeds"],
                          research_effect_validated=completion["research_effect_validated"])
        return result

    def save_plan(self, project_id: str, plan: dict[str, Any]) -> dict[str, Any]:
        store = self._store(project_id)
        normalized = self._normalize_plan(store, plan)
        with self._transaction(store):
            pointer = _read(store, "research/workflow/current.json", optional=True) or {}
            if "parent_plan_id" in plan and plan["parent_plan_id"] != pointer.get("plan_id"):
                raise ResearchWorkflowConflict("Parent plan is no longer current")
            _, _, code_sha = self._source(store)
            identity = uuid.uuid4().hex
            record = {"plan_id": identity, "parent_plan_id": pointer.get("plan_id"),
                      "parent_decision_id": pointer.get("latest_decision_id"), "created_at": utc_now(),
                      "plan_sha256": _digest(normalized), "code_sha256_at_save": code_sha, "plan": normalized}
            _write(store, f"research/workflow/plans/{identity}.json", record, immutable=True)
            _write(store, "research/workflow/current.json", {"plan_id": identity, "latest_decision_id": None})
            self.db.add_event(project_id, "research_plan", "ChatGPT saved an immutable research plan",
                              details={"plan_id": identity, "parent_plan_id": record["parent_plan_id"]})
        return self.get_status(project_id)

    def get_status(self, project_id: str) -> dict[str, Any]:
        store = self._store(project_id)
        manifest, files, code_sha = self._source(store)
        pointer = _read(store, "research/workflow/current.json", optional=True) or {}
        base: dict[str, Any] = {"project_id": project_id, "reasoning_owner": "chatgpt", "executes_experiments": False,
                                "code_sha256": code_sha, "research_effect_validated": False,
                                "code_prepared_is_not_experiment_completed": True}
        if not pointer:
            return {**base, "status": "unplanned", "plan_id": None, "plan_sha256": None,
                    "coverage": {"hypotheses": [], "datasets": [], "experiments": []}, "latest_decision": None}
        identity = _identifier(pointer.get("plan_id"), "plan_id")
        record = _read(store, f"research/workflow/plans/{identity}.json")
        plan = record["plan"]
        if record.get("plan_sha256") != _digest(plan):
            raise ResearchWorkflowConflict("Immutable research plan hash differs")
        evidence = self._evidence(project_id, store, files)
        evidence_id = _digest({"plan_sha256": record["plan_sha256"], "code_sha256": code_sha, "result": evidence})
        declarations = manifest.get("research_coverage", {})
        declarations = declarations.get("experiments", []) if isinstance(declarations, dict) else []
        declared: dict[str, dict[str, Any]] = {}
        for item in declarations if isinstance(declarations, list) else []:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                if item["id"] in declared:
                    raise ValueError("Duplicate experiment identity in manifest research_coverage")
                declared[item["id"]] = item
        coverage: dict[str, list[dict[str, Any]]] = {"hypotheses": [], "datasets": [], "experiments": []}
        manifest_id_counts = Counter(item.get("manifest_id", item["id"]) for item in plan["experiments"])
        measured_ids = set(evidence["research_experiment_ids"])
        for experiment in plan["experiments"]:
            manifest_id = experiment.get("manifest_id", experiment["id"])
            item = declared.get(manifest_id, {})
            paths = item.get("implementation_files", [])
            required_paths = experiment.get("implementation_files", [])
            hypothesis_ids, dataset_ids = item.get("hypothesis_ids", []), item.get("dataset_ids", [])
            exact = (isinstance(hypothesis_ids, list) and all(isinstance(value, str) for value in hypothesis_ids)
                     and isinstance(dataset_ids, list) and all(isinstance(value, str) for value in dataset_ids)
                     and item.get("scope") == experiment["scope"]
                     and set(hypothesis_ids) == set(experiment["hypothesis_ids"])
                     and set(dataset_ids) == set(experiment["dataset_ids"]))
            implementation = bool(exact and isinstance(paths, list) and paths
                                  and all(isinstance(path, str) and path in files for path in paths)
                                  and all(path in paths for path in required_paths))
            labels = [dataset["id"] for dataset in plan["datasets"]
                      if dataset["id"] in experiment["dataset_ids"] and dataset["labels"] in {"unknown", "unavailable"}]
            tested = bool(implementation and manifest_id in measured_ids and manifest_id_counts[manifest_id] == 1
                          and evidence["tested"])
            full = tested and evidence["full_data_completed"] and experiment["scope"] == "full" and not labels
            status = "completed" if full else "tested" if tested else "implemented" if implementation else "planned"
            coverage["experiments"].append({"id": experiment["id"], "scope": experiment["scope"], "required": experiment["required"],
                                            "status": status, "implemented": implementation, "tested": tested, "completed": bool(full),
                                            "analysis_required": bool(full), "unknown_or_missing_labels": labels,
                                            "implementation_evidence": "manifest_declaration_and_source_files" if implementation else "not_declared_or_missing_files"})
        by_experiment = {item["id"]: item for item in coverage["experiments"]}
        for family, reference in (("hypotheses", "hypothesis_ids"), ("datasets", "dataset_ids")):
            for item in plan[family]:
                linked = [by_experiment[row["id"]] for row in plan["experiments"] if item["id"] in row[reference]]
                required = [row for row in linked if row["required"]]
                done = bool(required and all(row["completed"] for row in required))
                selected = required or linked
                implemented = bool(selected and all(row["implemented"] for row in selected))
                tested = bool(selected and all(row["tested"] for row in selected))
                status = "completed" if done else "tested" if tested else "implemented" if implemented else "planned"
                coverage[family].append({"id": item["id"], "required": item["required"], "status": status,
                                         "implemented": implemented, "tested": tested, "completed": done,
                                         "analysis_required": done, "experiment_ids": [row["id"] for row in linked],
                                         **({"labels": item["labels"], "scope": item["scope"]} if family == "datasets" else {})})
        required = [item for rows in coverage.values() for item in rows if item["required"]]
        measured = evidence.get("metrics", {})
        missing_metrics = [name for name in plan["success_criteria"].get("required_metrics", [])
                           if not isinstance(measured.get(name), (int, float)) or isinstance(measured.get(name), bool)
                           or not math.isfinite(measured[name])]
        complete = bool(required and all(item["completed"] for item in required) and not missing_metrics)
        latest = None
        if pointer.get("latest_decision_id"):
            decision_id = _identifier(pointer["latest_decision_id"], "decision_id")
            latest = _read(store, f"research/workflow/decisions/{decision_id}.json")
        decision_current = bool(latest and latest.get("evidence_id") == evidence_id and latest.get("code_sha256") == code_sha)
        status = "analysis_required" if any(item["completed"] for item in coverage["experiments"]) else "execution_required"
        if any(item["required"] and not item["implemented"] for item in coverage["experiments"]):
            status = "implementation_required"
        if decision_current and latest["action"] == "accept" and complete:
            status = "completed"
        elif decision_current and latest["action"] == "stop":
            status = "stopped"
        elif decision_current and latest["action"] == "revise":
            status = "revision_required"
        return {**base, "status": status, "plan_id": identity, "parent_plan_id": record.get("parent_plan_id"),
                "plan_sha256": record["plan_sha256"], "plan": plan, "evidence_id": evidence_id, "evidence": evidence,
                "coverage": coverage, "all_required_experiments_completed": complete,
                "research_effect_validated": bool(complete and evidence.get("research_effect_validated")),
                "missing_required_metrics": missing_metrics,
                "analysis_required": bool(complete and not (decision_current and latest["action"] == "accept")),
                "latest_decision": latest, "latest_decision_is_current": decision_current,
                "code_changed_since_plan_save": code_sha != record["code_sha256_at_save"]}

    def record_decision(self, project_id: str, decision: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(decision, dict) or decision.get("action") not in {"advance", "revise", "stop", "accept"}:
            raise ValueError("Research decision action must be advance, revise, stop, or accept")
        if not isinstance(decision.get("analysis"), str) or not decision["analysis"].strip():
            raise ValueError("A ChatGPT research analysis is required")
        if len(decision["analysis"].encode()) > 200_000:
            raise ValueError("Research analysis exceeds 200 KB")
        store = self._store(project_id)
        with self._transaction(store):
            status = self.get_status(project_id)
            if not status.get("plan_id"):
                raise ResearchWorkflowConflict("Save a research plan before recording a decision")
            for field in ("plan_id", "plan_sha256", "code_sha256", "evidence_id"):
                if decision.get(field) != status.get(field):
                    raise ResearchWorkflowConflict(f"Research decision has stale or missing {field}; read the current workflow first")
            action = decision["action"]
            if action in {"advance", "accept"}:
                if not status["evidence"]["full_data_completed"]:
                    raise ResearchWorkflowConflict("Advancing requires current full-data scientific evidence; acceptance or process-exit checks are insufficient")
                if not any(item["completed"] for item in status["coverage"]["experiments"]):
                    raise ResearchWorkflowConflict("No declared experiment has complete current evidence")
                if action == "accept" and not status["all_required_experiments_completed"]:
                    raise ResearchWorkflowConflict("Cannot accept a partially implemented or incomplete research plan")
            next_ids = decision.get("next_experiment_ids", [])
            known = {item["id"] for item in status["plan"]["experiments"]}
            if next_ids:
                _references(next_ids, known, "next_experiment_ids")
            elif not isinstance(next_ids, list):
                raise ValueError("next_experiment_ids must be a list")
            if action == "advance" and not next_ids:
                raise ValueError("An advance decision must identify next_experiment_ids")
            identity = uuid.uuid4().hex
            record = {"decision_id": identity, "parent_decision_id": (status["latest_decision"] or {}).get("decision_id"),
                      "created_at": utc_now(), "reasoning_owner": "chatgpt", "action": action,
                      "analysis": decision["analysis"].strip(), "next_experiment_ids": next_ids,
                      **{field: status[field] for field in ("plan_id", "plan_sha256", "code_sha256", "evidence_id")},
                      "execution_authorized": False, "executes_experiments": False}
            _write(store, f"research/workflow/decisions/{identity}.json", record, immutable=True)
            _write(store, "research/workflow/current.json", {"plan_id": status["plan_id"], "latest_decision_id": identity})
            self.db.add_event(project_id, "research_decision", "ChatGPT recorded an evidence-bound research decision",
                              details={"plan_id": status["plan_id"], "decision_id": identity, "action": action})
        return self.get_status(project_id)
