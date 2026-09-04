from pathlib import Path

from autoresearch.db import Database
from autoresearch.domain import ResearchRequest, Source


def test_project_event_and_source_roundtrip(tmp_path: Path) -> None:
    db = Database(tmp_path / "db.sqlite3")
    project = db.create_project(ResearchRequest(topic="Graph neural networks"), tmp_path / "work")
    db.update_project(project.id, status="running", phase="searching", progress=25)
    db.add_event(project.id, "searching", "found", details={"count": 1})
    db.replace_sources(project.id, [Source(provider="arXiv", title="Paper", url="https://arxiv.org/abs/1")])
    updated = db.get_project(project.id)
    assert updated and updated.progress == 25
    assert db.list_events(project.id)[0]["details"] == {"count": 1}
    assert db.list_sources(project.id)[0].title == "Paper"

