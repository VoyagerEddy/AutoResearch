from pathlib import Path

import pytest

from autoresearch.services.artifacts import ArtifactStore, UnsafeArtifactPath, fallback_code_bundle


def test_artifact_store_blocks_traversal_and_secrets(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    with pytest.raises(UnsafeArtifactPath):
        store.write_text("../escape.txt", "no")
    with pytest.raises(UnsafeArtifactPath):
        store.write_text("generated/.env", "SECRET=yes")


def test_fallback_bundle_is_materialized(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    bundle = fallback_code_bundle("A safe baseline")
    files = store.materialize_files(bundle["files"])
    assert {path.name for path in files} >= {"README.md", "experiment.py"}
    assert (tmp_path / "generated" / "experiment.py").is_file()

