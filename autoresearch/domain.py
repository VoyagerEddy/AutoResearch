from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ResearchRequest(BaseModel):
    topic: str = Field(min_length=3, max_length=2000)
    notes: str = Field(default="", max_length=8000)
    model: str | None = Field(default=None, max_length=200)
    max_sources: int = Field(default=20, ge=5, le=80)
    download_papers: bool = False


class ChatGPTProjectRequest(BaseModel):
    topic: str = Field(min_length=3, max_length=2000)
    notes: str = Field(default="", max_length=12000)


class ResearchNoteRequest(BaseModel):
    title: str = Field(min_length=1, max_length=160)
    markdown: str = Field(min_length=1, max_length=200_000)


class CodeFile(BaseModel):
    path: str = Field(min_length=1, max_length=500)
    content: str = Field(max_length=2_000_000)


class CodeSaveRequest(BaseModel):
    files: list[CodeFile] = Field(min_length=1, max_length=100)
    summary: str = Field(default="", max_length=2000)
    experiment_manifest: dict[str, Any] = Field(default_factory=dict)


class CodeGenerationRequest(BaseModel):
    instructions: str = Field(min_length=3, max_length=12000)
    model: str | None = Field(default=None, max_length=200)


class SourceSearchRequest(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=5)
    max_sources: int = Field(default=20, ge=1, le=80)
    download_papers: bool = False


class ChatGPTAnalysisRequest(BaseModel):
    analysis: str = Field(min_length=1, max_length=200_000)
    experiment_id: str | None = Field(default=None, max_length=100)
    recommendations: list[str] = Field(default_factory=list, max_length=30)


class ChatGPTTunnelLaunchRequest(BaseModel):
    action: Literal["Setup", "Run", "Doctor"]
    confirm_launch: bool = False


class Project(BaseModel):
    id: str
    topic: str
    notes: str = ""
    model: str = ""
    status: Literal["queued", "running", "ready", "failed"]
    phase: str
    progress: int = Field(ge=0, le=100)
    workspace: str
    summary: str = ""
    error: str = ""
    created_at: str
    updated_at: str


class Source(BaseModel):
    provider: str
    kind: Literal["paper", "code", "dataset", "web"] = "paper"
    title: str
    url: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    abstract: str = ""
    citation_count: int = 0
    pdf_url: str = ""


class SettingsPatch(BaseModel):
    openrouter_api_key: str | None = Field(default=None, max_length=500)
    openrouter_model: str | None = Field(default=None, max_length=200)
    autodl_token: str | None = Field(default=None, max_length=1000)
    autodl_image_uuid: str | None = Field(default=None, max_length=200)
    autodl_gpu_specs: str | None = Field(default=None, max_length=500)
    github_remote_url: str | None = Field(default=None, max_length=1000)
    github_default_branch: str | None = Field(default=None, max_length=200)
    github_token: str | None = Field(default=None, max_length=1000)
    monitor_seconds: int | None = Field(default=None, ge=5, le=3600)
    max_iterations: int | None = Field(default=None, ge=1, le=20)

    def as_env(self) -> dict[str, str]:
        mapping = {
            "openrouter_api_key": "OPENROUTER_API_KEY",
            "openrouter_model": "OPENROUTER_MODEL",
            "autodl_token": "AUTODL_TOKEN",
            "autodl_image_uuid": "AUTODL_IMAGE_UUID",
            "autodl_gpu_specs": "AUTODL_GPU_SPECS",
            "github_remote_url": "GITHUB_REMOTE_URL",
            "github_default_branch": "GITHUB_DEFAULT_BRANCH",
            "github_token": "GITHUB_TOKEN",
            "monitor_seconds": "AUTORESEARCH_MONITOR_SECONDS",
            "max_iterations": "AUTORESEARCH_MAX_ITERATIONS",
        }
        data = self.model_dump(exclude_none=True)
        return {mapping[key]: str(value) for key, value in data.items()}


class AutoDLCreateRequest(BaseModel):
    image_uuid: str | None = None
    instance_name: str = Field(default="AutoResearch", max_length=120)
    gpu_amount: int = Field(default=1, ge=1, le=4)
    disk_gb: int = Field(default=0, ge=0, le=500)
    data_centers: list[str] = Field(default_factory=list)
    confirm_billable: bool = False


class SSHConnection(BaseModel):
    host: str = Field(min_length=1, max_length=255)
    port: int = Field(default=22, ge=1, le=65535)
    username: str = Field(default="root", min_length=1, max_length=64)
    password: str | None = Field(default=None, max_length=1000)
    key_path: str | None = Field(default=None, max_length=1000)


class ExperimentStartRequest(BaseModel):
    project_id: str
    connection: SSHConnection
    remote_dir: str = Field(default="/root/autodl-tmp/AutoResearch", max_length=500)
    command: str = Field(default="python experiment.py", min_length=1, max_length=4000)
    max_iterations: int = Field(default=1, ge=1, le=10)
    instance_uuid: str | None = Field(default=None, max_length=200)
    recover_busy_gpu: bool = True
    allow_release_replacement: bool = False
    confirm_execute: bool = False

    @field_validator("remote_dir")
    @classmethod
    def validate_remote_dir(cls, value: str) -> str:
        if not value.startswith("/") or any(part == ".." for part in value.split("/")):
            raise ValueError("remote_dir 必须是无 .. 的 Linux 绝对路径")
        return value.rstrip("/") or "/"


class AutoDLExperimentRequest(BaseModel):
    project_id: str
    instance_uuid: str = Field(min_length=1, max_length=200)
    remote_dir: str = Field(default="/root/autodl-tmp/AutoResearch", max_length=500)
    command: str = Field(default="python experiment.py", min_length=1, max_length=4000)
    max_iterations: int = Field(default=1, ge=1, le=10)
    recover_busy_gpu: bool = True
    allow_release_replacement: bool = False
    confirm_execute: bool = False

    @field_validator("remote_dir")
    @classmethod
    def validate_remote_dir(cls, value: str) -> str:
        if not value.startswith("/") or any(part == ".." for part in value.split("/")):
            raise ValueError("remote_dir 必须是无 .. 的 Linux 绝对路径")
        return value.rstrip("/") or "/"


class GitSyncRequest(BaseModel):
    project_id: str | None = None
    remote_url: str | None = None
    branch: str = Field(default="main", pattern=r"^[A-Za-z0-9._/-]+$")
    message: str = Field(default="chore: sync AutoResearch code", max_length=300)


class Event(BaseModel):
    id: int
    project_id: str
    level: str
    phase: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    created_at: str
