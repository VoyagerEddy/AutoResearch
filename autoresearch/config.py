from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

from dotenv import dotenv_values


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_int(value: str | None, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


@dataclass(frozen=True, slots=True)
class Settings:
    root: Path
    data_dir: Path
    workspace_dir: Path
    openrouter_api_key: str = ""
    openrouter_model: str = "openrouter/free"
    openrouter_site_url: str = "http://127.0.0.1:8765"
    autodl_token: str = ""
    autodl_image_uuid: str = ""
    autodl_gpu_specs: tuple[str, ...] = ("v-48g", "5090")
    autodl_cuda_from: int = 118
    github_remote_url: str = ""
    github_default_branch: str = "main"
    github_token: str = ""
    host: str = "127.0.0.1"
    port: int = 8765
    monitor_seconds: int = 30
    max_iterations: int = 3
    download_papers: bool = False

    @classmethod
    def load(cls, root: Path | None = None) -> "Settings":
        app_root = (root or Path(__file__).resolve().parents[1]).resolve()
        file_values = {
            key: value
            for key, value in dotenv_values(app_root / ".env").items()
            if value is not None
        }

        def value(name: str, default: str = "") -> str:
            # Explicit process variables take precedence. Reading the .env file
            # directly avoids stale values when the web UI updates it at runtime.
            return os.environ.get(name, file_values.get(name, default)).strip()

        specs = tuple(
            item.strip()
            for item in value("AUTODL_GPU_SPECS", "v-48g,5090").split(",")
            if item.strip()
        )
        settings = cls(
            root=app_root,
            data_dir=app_root / "data",
            workspace_dir=app_root / "workspaces",
            openrouter_api_key=value("OPENROUTER_API_KEY"),
            openrouter_model=value("OPENROUTER_MODEL", "openrouter/free"),
            openrouter_site_url=value("OPENROUTER_SITE_URL", "http://127.0.0.1:8765"),
            autodl_token=value("AUTODL_TOKEN"),
            autodl_image_uuid=value("AUTODL_IMAGE_UUID"),
            autodl_gpu_specs=specs or ("v-48g", "5090"),
            autodl_cuda_from=_as_int(value("AUTODL_CUDA_FROM") or None, 118),
            github_remote_url=value("GITHUB_REMOTE_URL"),
            github_default_branch=value("GITHUB_DEFAULT_BRANCH", "main"),
            github_token=value("GITHUB_TOKEN"),
            host=value("AUTORESEARCH_HOST", "127.0.0.1"),
            port=_as_int(value("AUTORESEARCH_PORT") or None, 8765),
            monitor_seconds=max(
                5, _as_int(value("AUTORESEARCH_MONITOR_SECONDS") or None, 30)
            ),
            max_iterations=max(
                1, _as_int(value("AUTORESEARCH_MAX_ITERATIONS") or None, 3)
            ),
            download_papers=_as_bool(value("AUTORESEARCH_DOWNLOAD_PAPERS") or None),
        )
        settings.ensure_directories()
        return settings

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

    def with_overrides(self, **values: object) -> "Settings":
        updated = replace(self, **values)
        updated.ensure_directories()
        return updated

    def public(self) -> dict[str, object]:
        return {
            "openrouter_configured": bool(self.openrouter_api_key),
            "openrouter_model": self.openrouter_model,
            "autodl_configured": bool(self.autodl_token),
            "autodl_image_uuid": self.autodl_image_uuid,
            "autodl_gpu_specs": list(self.autodl_gpu_specs),
            "github_remote_url": self.github_remote_url,
            "github_default_branch": self.github_default_branch,
            "monitor_seconds": self.monitor_seconds,
            "max_iterations": self.max_iterations,
        }


class EnvStore:
    """Small .env writer that never returns secret values to the API."""

    ALLOWED = {
        "OPENROUTER_API_KEY",
        "OPENROUTER_MODEL",
        "AUTODL_TOKEN",
        "AUTODL_IMAGE_UUID",
        "AUTODL_GPU_SPECS",
        "GITHUB_REMOTE_URL",
        "GITHUB_DEFAULT_BRANCH",
        "GITHUB_TOKEN",
        "AUTORESEARCH_MONITOR_SECONDS",
        "AUTORESEARCH_MAX_ITERATIONS",
    }

    def __init__(self, path: Path) -> None:
        self.path = path

    def update(self, values: Mapping[str, str]) -> None:
        current: dict[str, str] = {}
        if self.path.exists():
            for raw in self.path.read_text(encoding="utf-8").splitlines():
                if not raw or raw.lstrip().startswith("#") or "=" not in raw:
                    continue
                key, value = raw.split("=", 1)
                current[key.strip()] = value
        for key, value in values.items():
            if key in self.ALLOWED and value is not None:
                clean = str(value).replace("\r", "").replace("\n", "").strip()
                current[key] = clean
        lines = [f"{key}={value}" for key, value in sorted(current.items())]
        temp = self.path.with_suffix(".tmp")
        temp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        temp.replace(self.path)
