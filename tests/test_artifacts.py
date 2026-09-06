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
    assert (tmp_path / "generated" / "requirements.txt").read_text() == ""


@pytest.mark.parametrize("path", ["generated/experiment.py", r"generated\experiment.py", "Generated/experiment.py"])
def test_bundle_rejects_output_prefix_with_actionable_error(tmp_path: Path, path: str) -> None:
    store = ArtifactStore(tmp_path)
    with pytest.raises(UnsafeArtifactPath, match="use experiment.py"):
        store.materialize_files([{"path": path, "content": "print('hello')"}])
    assert not (tmp_path / "generated").exists()


def test_bundle_preserves_nested_files_and_iteration_output(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    files = [{"path": r"src\experiment.py", "content": "print('hello')"}]
    for prefix in ("generated", "iterations/iteration-1"):
        written = store.materialize_files(iter(files), prefix)
        target = tmp_path / prefix / "src" / "experiment.py"
        assert written == [target]
        assert target.read_text() == "print('hello')"


@pytest.mark.parametrize(
    "path",
    [
        "", " ", "/absolute.txt", r"\absolute.txt", "C:/absolute.txt", "C:relative.txt",
        r"\\server\share\file.txt", "//server/share/file.txt", r"\\?\C:\file.txt",
        "file.txt:secret", "src/C:relative.txt", "../escape.txt", "src/../../escape.txt",
        "./file.txt", "src//file.txt", "src/file.txt/", "src\x00/file.txt",
        ".env", ".ENV", ".env.production", ".git/config", ".ssh/config", ".aws/config",
        "secrets/id_rsa", "secrets/id_ed25519", "secrets/credentials",
        "secrets/.env.", "secrets/.env ", "NUL", "con.txt", "LPT1.py", "COM¹.log",
    ],
)
def test_unsafe_paths_are_rejected_before_prefixing(tmp_path: Path, path: str) -> None:
    store = ArtifactStore(tmp_path)
    with pytest.raises(UnsafeArtifactPath):
        store.materialize_files([{"path": path, "content": "unsafe"}])
    with pytest.raises(UnsafeArtifactPath):
        store.write_text(path, "unsafe")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "invalid_item",
    [
        {}, {"path": "", "content": "x"}, {"path": None, "content": "x"},
        {"path": 123, "content": "x"}, {"path": "missing-content.py"},
        {"path": "null-content.py", "content": None}, {"path": "bad.py", "content": {}},
        {"path": "generated/duplicate.py", "content": "x"},
        {"path": "../escape.py", "content": "x"}, "not-a-file-object",
    ],
)
def test_late_invalid_file_does_not_overwrite_existing_file(tmp_path: Path, invalid_item: object) -> None:
    store = ArtifactStore(tmp_path)
    existing = store.write_text("generated/experiment.py", "original")
    files = [{"path": "experiment.py", "content": "replacement"}, invalid_item]
    with pytest.raises(ValueError):
        store.materialize_files(iter(files))
    assert existing.read_text() == "original"
    assert store.list_files() == [{"path": "generated/experiment.py", "bytes": 8}]


def test_empty_bundle_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        ArtifactStore(tmp_path).materialize_files(iter([]))
    assert not list(tmp_path.iterdir())


def test_explicit_empty_files_are_written(tmp_path: Path) -> None:
    written = ArtifactStore(tmp_path).materialize_files(
        [{"path": "requirements.txt", "content": ""}, {"path": "src/__init__.py", "content": ""}]
    )
    assert len(written) == 2
    assert all(path.is_file() and path.stat().st_size == 0 for path in written)


def test_benign_environment_templates_remain_supported(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    written = store.materialize_files([
        {"path": ".env.example", "content": "API_KEY=\n"},
        {"path": ".env.template", "content": "API_KEY=\n"},
    ])
    assert len(written) == 2
    assert all(path.read_text() == "API_KEY=\n" for path in written)


def test_file_at_size_limit_is_written(tmp_path: Path) -> None:
    written = ArtifactStore(tmp_path).materialize_files(
        [{"path": "at-limit.txt", "content": "a" * 2_000_000}]
    )
    assert written[0].stat().st_size == 2_000_000


@pytest.mark.parametrize("content", ["a" * 2_000_001, "\u6d4b" * 666_667], ids=["ascii", "utf8"])
def test_oversized_file_rejects_bundle_before_any_write(tmp_path: Path, content: str) -> None:
    store = ArtifactStore(tmp_path)
    existing = store.write_text("generated/experiment.py", "original")
    with pytest.raises(ValueError, match="exceeds 2 MB"):
        store.materialize_files([
            {"path": "experiment.py", "content": "replacement"},
            {"path": "large.txt", "content": content},
        ])
    assert existing.read_text() == "original"
    assert not (tmp_path / "generated" / "large.txt").exists()


def test_total_size_limit_is_checked_before_any_write(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    existing = store.write_text("generated/experiment.py", "original")
    with pytest.raises(ValueError, match="exceeds 10 MB"):
        store.materialize_files([
            {"path": "experiment.py", "content": "replacement"},
            *[{"path": f"large-{index}.txt", "content": "a" * 2_000_000} for index in range(5)],
        ])
    assert existing.read_text() == "original"
    assert len(store.list_files()) == 1


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("experiment.py", "experiment.py"), ("Experiment.py", "experiment.py"),
        (r"src\experiment.py", "src/experiment.py"),
        ("src", "src/experiment.py"), ("src/experiment.py", "src"),
        ("SRC", "src/experiment.py"),
    ],
)
def test_duplicate_or_conflicting_bundle_paths_are_rejected(tmp_path: Path, first: str, second: str) -> None:
    store = ArtifactStore(tmp_path)
    with pytest.raises(UnsafeArtifactPath, match="Duplicate|conflict"):
        store.materialize_files([
            {"path": first, "content": "first"}, {"path": second, "content": "second"},
        ])
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("target_is_directory", [True, False])
def test_existing_filesystem_conflict_does_not_overwrite_earlier_file(
    tmp_path: Path, target_is_directory: bool
) -> None:
    store = ArtifactStore(tmp_path)
    existing = store.write_text("generated/experiment.py", "original")
    if target_is_directory:
        (tmp_path / "generated" / "conflict").mkdir()
        path = "conflict"
    else:
        store.write_text("generated/conflict", "not-a-directory")
        path = "conflict/experiment.py"
    with pytest.raises(UnsafeArtifactPath):
        store.materialize_files([
            {"path": "experiment.py", "content": "replacement"}, {"path": path, "content": "x"},
        ])
    assert existing.read_text() == "original"


def test_repeated_custom_output_prefix_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(UnsafeArtifactPath, match="output prefix"):
        ArtifactStore(tmp_path).materialize_files(
            [{"path": "iterations/iteration-1/experiment.py", "content": "x"}],
            "iterations/iteration-1",
        )


def test_symlink_cannot_redirect_artifact_to_sensitive_directory(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    secret_dir = tmp_path / ".git"
    secret_dir.mkdir()
    alias = tmp_path / "generated"
    try:
        alias.symlink_to(secret_dir, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Directory symlinks unavailable: {exc}")
    with pytest.raises(UnsafeArtifactPath, match="sensitive files"):
        store.materialize_files([{"path": "config", "content": "x"}])
    assert not (secret_dir / "config").exists()
