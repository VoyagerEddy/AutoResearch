from pathlib import Path

import pytest

from autoresearch.config import Settings
from autoresearch.db import Database
from autoresearch.domain import ResearchRequest, Source
from autoresearch.orchestrator import ResearchOrchestrator
from autoresearch.services.search import ResearchSearch


@pytest.mark.asyncio
async def test_offline_pipeline_creates_runnable_baseline(tmp_path: Path, monkeypatch) -> None:
    settings = Settings.load(tmp_path)
    db = Database(settings.data_dir / "db.sqlite3")
    request = ResearchRequest(topic="Robust tiny model evaluation")
    project = db.create_project(request, settings.workspace_dir)

    async def fake_search(self, queries, limit=20):
        return [Source(provider="arXiv", title="Evidence", url="https://arxiv.org/abs/1")]

    monkeypatch.setattr(ResearchSearch, "search", fake_search)
    await ResearchOrchestrator(db, settings).run(project.id, request)
    finished = db.get_project(project.id)
    assert finished and finished.status == "ready"
    assert (Path(finished.workspace) / "RESEARCH.md").is_file()
    assert (Path(finished.workspace) / "generated" / "experiment.py").is_file()

