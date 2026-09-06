"""Read saved project readiness without requiring the dashboard or Tunnel."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .config import Settings
from .db import Database
from .services.autodl import AutoDLClient
from .services.experiment_readiness import inspect_experiment


async def diagnose(root: Path | None, project_id: str) -> dict:
    settings = Settings.load(root)
    db = Database(settings.data_dir / "autoresearch.sqlite3")
    project = db.get_project(project_id)
    if project is None:
        raise ValueError("Research project does not exist")
    client = AutoDLClient(settings)
    try:
        compute = await client.preflight()
    finally:
        await client.close()
    experiment = inspect_experiment(Path(project.workspace))
    return {
        "project_id": project_id,
        "ready": compute["ready"] and experiment["ready"],
        "autodl": compute,
        "experiment": experiment,
        "executed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    try:
        report = asyncio.run(diagnose(args.root, args.project))
    except ValueError as exc:
        report = {"ready": False, "error": str(exc), "executed": False}
    # ASCII JSON also works in legacy Windows console encodings.
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
