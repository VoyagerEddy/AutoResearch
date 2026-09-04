from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any

from .domain import Project, ResearchRequest, Source, utc_now


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    notes TEXT NOT NULL DEFAULT '',
                    model TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    progress INTEGER NOT NULL,
                    workspace TEXT NOT NULL,
                    summary TEXT NOT NULL DEFAULT '',
                    error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    level TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    message TEXT NOT NULL,
                    details TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sources (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    payload TEXT NOT NULL,
                    UNIQUE(project_id, payload)
                );
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    status TEXT NOT NULL,
                    instance_uuid TEXT NOT NULL DEFAULT '',
                    remote_dir TEXT NOT NULL,
                    pid INTEGER,
                    iteration INTEGER NOT NULL DEFAULT 1,
                    command TEXT NOT NULL,
                    result TEXT NOT NULL DEFAULT '{}',
                    error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_events_project ON events(project_id, id);
                CREATE INDEX IF NOT EXISTS idx_sources_project ON sources(project_id, id);
                """
            )

    def create_project(self, request: ResearchRequest, workspace: Path) -> Project:
        project_id = uuid.uuid4().hex[:12]
        now = utc_now()
        project = Project(
            id=project_id,
            topic=request.topic.strip(),
            notes=request.notes.strip(),
            model=request.model or "",
            status="queued",
            phase="queued",
            progress=0,
            workspace=str(workspace / project_id),
            created_at=now,
            updated_at=now,
        )
        with self._lock, self._connect() as db:
            db.execute(
                """INSERT INTO projects
                (id, topic, notes, model, status, phase, progress, workspace,
                 summary, error, created_at, updated_at)
                VALUES (:id, :topic, :notes, :model, :status, :phase, :progress,
                        :workspace, :summary, :error, :created_at, :updated_at)""",
                project.model_dump(),
            )
        return project

    def get_project(self, project_id: str) -> Project | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM projects WHERE id = ?", (project_id,)
            ).fetchone()
        return Project(**dict(row)) if row else None

    def list_projects(self, limit: int = 50) -> list[Project]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM projects ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [Project(**dict(row)) for row in rows]

    def update_project(self, project_id: str, **fields: object) -> None:
        allowed = {"status", "phase", "progress", "summary", "error", "model"}
        values = {key: value for key, value in fields.items() if key in allowed}
        if not values:
            return
        values["updated_at"] = utc_now()
        assignments = ", ".join(f"{key} = ?" for key in values)
        with self._lock, self._connect() as db:
            db.execute(
                f"UPDATE projects SET {assignments} WHERE id = ?",
                (*values.values(), project_id),
            )

    def add_event(
        self,
        project_id: str,
        phase: str,
        message: str,
        *,
        level: str = "info",
        details: dict[str, Any] | None = None,
    ) -> int:
        with self._lock, self._connect() as db:
            cursor = db.execute(
                """INSERT INTO events
                (project_id, level, phase, message, details, created_at)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    project_id,
                    level,
                    phase,
                    message,
                    json.dumps(details or {}, ensure_ascii=False),
                    utc_now(),
                ),
            )
            return int(cursor.lastrowid)

    def list_events(self, project_id: str, after: int = 0) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM events WHERE project_id = ? AND id > ? ORDER BY id",
                (project_id, after),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["details"] = json.loads(item["details"])
            result.append(item)
        return result

    def replace_sources(self, project_id: str, sources: list[Source]) -> None:
        with self._lock, self._connect() as db:
            db.execute("DELETE FROM sources WHERE project_id = ?", (project_id,))
            db.executemany(
                "INSERT OR IGNORE INTO sources (project_id, payload) VALUES (?, ?)",
                [
                    (project_id, source.model_dump_json())
                    for source in sources
                ],
            )

    def list_sources(self, project_id: str) -> list[Source]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT payload FROM sources WHERE project_id = ? ORDER BY id",
                (project_id,),
            ).fetchall()
        return [Source.model_validate_json(row["payload"]) for row in rows]

    def create_experiment(
        self,
        project_id: str,
        remote_dir: str,
        command: str,
        instance_uuid: str = "",
    ) -> str:
        experiment_id = uuid.uuid4().hex[:12]
        now = utc_now()
        with self._lock, self._connect() as db:
            db.execute(
                """INSERT INTO experiments
                (id, project_id, status, instance_uuid, remote_dir, command,
                 created_at, updated_at)
                VALUES (?, ?, 'preparing', ?, ?, ?, ?, ?)""",
                (experiment_id, project_id, instance_uuid, remote_dir, command, now, now),
            )
        return experiment_id

    def update_experiment(self, experiment_id: str, **fields: object) -> None:
        allowed = {"status", "pid", "iteration", "result", "error", "instance_uuid"}
        values = {key: value for key, value in fields.items() if key in allowed}
        if "result" in values and not isinstance(values["result"], str):
            values["result"] = json.dumps(values["result"], ensure_ascii=False)
        values["updated_at"] = utc_now()
        assignments = ", ".join(f"{key} = ?" for key in values)
        with self._lock, self._connect() as db:
            db.execute(
                f"UPDATE experiments SET {assignments} WHERE id = ?",
                (*values.values(), experiment_id),
            )

    def get_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
            ).fetchone()
        if not row:
            return None
        item = dict(row)
        item["result"] = json.loads(item["result"] or "{}")
        return item

    def list_experiments(self, project_id: str, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM experiments WHERE project_id = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (project_id, limit),
            ).fetchall()
        experiments: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["result"] = json.loads(item["result"] or "{}")
            experiments.append(item)
        return experiments
