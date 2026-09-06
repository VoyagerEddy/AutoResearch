from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


class UnsafeArtifactPath(ValueError):
    pass


class ArtifactStore:
    BLOCKED_NAMES = {
        ".env", ".git", ".ssh", ".aws", ".azure", ".gnupg", ".netrc", ".pypirc",
        "id_rsa", "id_ed25519", "credentials",
    }
    ENV_TEMPLATE_NAMES = {".env.example", ".env.sample", ".env.template"}
    WINDOWS_DEVICE_NAMES = {"con", "prn", "aux", "nul", "conin$", "conout$"} | {
        f"{kind}{number}" for kind in ("com", "lpt") for number in "123456789¹²³"
    }

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)

    def write_text(self, relative_path: str, content: str) -> Path:
        target = self._safe_target(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return target

    def write_json(self, relative_path: str, payload: Any) -> Path:
        return self.write_text(
            relative_path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        )

    def read_text(self, relative_path: str, max_bytes: int = 200_000) -> str:
        target = self._safe_target(relative_path)
        if not target.is_file():
            raise FileNotFoundError(f"Artifact does not exist: {relative_path}")
        if target.stat().st_size > max_bytes:
            raise ValueError(f"Artifact exceeds the {max_bytes}-byte read limit")
        return target.read_text(encoding="utf-8", errors="replace")

    def list_files(self, max_files: int = 200) -> list[dict[str, Any]]:
        files: list[dict[str, Any]] = []
        for target in sorted(self.workspace.rglob("*")):
            if not target.is_file():
                continue
            relative = target.relative_to(self.workspace).as_posix()
            try:
                self._safe_target(relative)
            except UnsafeArtifactPath:
                continue
            files.append({"path": relative, "bytes": target.stat().st_size})
            if len(files) >= max_files:
                break
        return files

    def materialize_files(
        self, files: Iterable[dict[str, Any]], prefix: str = "generated"
    ) -> list[Path]:
        """Validate the entire bundle before writing paths relative to its output directory."""
        prefix_parts = self._relative_parts(prefix)
        folded_prefix = tuple(part.casefold() for part in prefix_parts)
        self._safe_target(prefix)
        pending: list[tuple[str, str]] = []
        file_keys: set[tuple[str, ...]] = set()
        directory_keys: set[tuple[str, ...]] = set()
        total_bytes = 0
        for index, item in enumerate(files, 1):
            if not isinstance(item, dict):
                raise ValueError(f"Generated file {index} must contain string path and content fields")
            path = item.get("path")
            parts = self._relative_parts(path)
            folded_parts = tuple(part.casefold() for part in parts)
            if folded_parts[0] == "generated" or folded_parts[:len(prefix_parts)] == folded_prefix:
                raise UnsafeArtifactPath(
                    f"Generated paths must be relative to the {prefix} directory and must not include "
                    f"generated/ or another output prefix: {path}; use experiment.py, not generated/experiment.py"
                )
            content = item.get("content")
            if not isinstance(content, str):
                raise ValueError(f"Generated file content must be a string: {path}")
            content_bytes = len(content.encode("utf-8"))
            if content_bytes > 2_000_000:
                raise ValueError(f"Generated file exceeds 2 MB: {path}")
            total_bytes += content_bytes
            if total_bytes > 10_000_000:
                raise ValueError("Total generated bundle exceeds 10 MB")

            relative = "/".join((*prefix_parts, *parts))
            target = self._safe_target(relative)
            # Resolve aliases and compare case-insensitively for portable Windows bundles.
            key = tuple(part.casefold() for part in target.relative_to(self.workspace).parts)
            parents = {key[:length] for length in range(1, len(key))}
            if key in file_keys:
                raise UnsafeArtifactPath(f"Duplicate generated file path: {path}")
            if key in directory_keys or parents & file_keys:
                raise UnsafeArtifactPath(f"Generated file and directory paths conflict: {path}")
            if target.exists() and not target.is_file():
                raise UnsafeArtifactPath(f"Generated target exists and is not a file: {path}")
            for parent in target.parents:
                if parent == self.workspace:
                    break
                if parent.exists() and not parent.is_dir():
                    raise UnsafeArtifactPath(f"A generated file parent exists and is not a directory: {path}")
            file_keys.add(key)
            directory_keys.update(parents)
            pending.append((relative, content))

        if not pending:
            raise ValueError("Generated file list must not be empty; provide paths relative to generated and complete contents")
        return [self.write_text(path, content) for path, content in pending]

    def _relative_parts(self, relative_path: str) -> tuple[str, ...]:
        if not isinstance(relative_path, str) or not relative_path.strip():
            raise UnsafeArtifactPath("Artifact path must be a nonempty relative path string")
        normalized = relative_path.replace("\\", "/")
        parts = tuple(normalized.split("/"))
        if PurePosixPath(normalized).is_absolute() or any(
            part in {"", ".", ".."}
            or part.endswith((".", " "))
            or any(ord(char) < 32 or char in ':<>"|?*' for char in part)
            or part.split(".", 1)[0].casefold() in self.WINDOWS_DEVICE_NAMES
            for part in parts
        ):
            raise UnsafeArtifactPath(f"Artifact path is not allowed: {relative_path}")
        if any(
            part.casefold() in self.BLOCKED_NAMES
            or (part.casefold().startswith(".env.") and part.casefold() not in self.ENV_TEMPLATE_NAMES)
            for part in parts
        ):
            raise UnsafeArtifactPath(f"Generating sensitive files is prohibited: {relative_path}")
        return parts

    def _safe_target(self, relative_path: str) -> Path:
        parts = self._relative_parts(relative_path)
        target = self.workspace.joinpath(*parts).resolve()
        if self.workspace not in target.parents:
            raise UnsafeArtifactPath(f"Path escapes the workspace: {relative_path}")
        self._relative_parts(target.relative_to(self.workspace).as_posix())
        return target


def safe_slug(text: str, max_length: int = 50) -> str:
    normalized = re.sub(r"[^\w\u4e00-\u9fff-]+", "-", text.lower()).strip("-")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    return f"{normalized[:max_length] or 'research'}-{digest}"


def fallback_code_bundle(topic: str) -> dict[str, Any]:
    safe_topic = topic.replace('"""', "").replace("\x00", "")
    experiment = '''"""Deterministic baseline generated by AutoResearch.

Replace `evaluate` with the domain-specific experiment proposed in RESEARCH.md.
"""
from __future__ import annotations

import json
import random
from pathlib import Path


def evaluate(seed: int) -> float:
    random.seed(seed)
    return sum(random.random() for _ in range(1000)) / 1000


def main() -> None:
    scores = [evaluate(seed) for seed in range(5)]
    result = {
        "metric": "baseline_score",
        "mean": sum(scores) / len(scores),
        "scores": scores,
    }
    output = Path("results")
    output.mkdir(exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
'''
    return {
        "files": [
            {
                "path": "README.md",
                "content": f"# Research experiment\n\nTopic: {safe_topic}\n",
            },
            {"path": "experiment.py", "content": experiment},
            {"path": "requirements.txt", "content": ""},
            {
                "path": ".gitignore",
                "content": "data/\nresults/\n__pycache__/\n*.pyc\n.env\n",
            },
        ],
        "experiment": {
            "setup_commands": ["python -m pip install -r requirements.txt"],
            "dataset_commands": [],
            "run_command": "python experiment.py",
            "metrics_file": "results/metrics.json",
        },
    }
