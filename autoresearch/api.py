from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .chatgpt_bridge import ChatGPTBridge
from .chatgpt_connection import connection_status, launch_tunnel_console
from .config import EnvStore, Settings
from .db import Database
from .domain import (
    AutoDLCreateRequest,
    AutoDLExperimentRequest,
    ChatGPTAnalysisRequest,
    ChatGPTProjectRequest,
    ChatGPTTunnelLaunchRequest,
    CodeGenerationRequest,
    CodeSaveRequest,
    ExperimentStartRequest,
    GitSyncRequest,
    ResearchNoteRequest,
    ResearchRequest,
    SettingsPatch,
    SourceSearchRequest,
)
from .experiments import ExperimentManager
from .mcp_server import create_mcp_server
from .orchestrator import ResearchOrchestrator
from .services.autodl import AutoDLClient, AutoDLError, extract_ssh
from .services.desktop import DesktopBridge
from .services.github_sync import GitSync, GitSyncError
from .services.llm import LLMError, OpenRouterClient


class AppState:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.db = Database(settings.data_dir / "autoresearch.sqlite3")
        self.orchestrator = ResearchOrchestrator(self.db, settings)
        self.experiments = ExperimentManager(self.db, settings, self.orchestrator)
        self.chatgpt = ChatGPTBridge(self.db, settings, self.experiments)

    def reload_settings(self) -> Settings:
        self.settings = Settings.load(self.settings.root)
        self.orchestrator.update_settings(self.settings)
        self.experiments.update_settings(self.settings)
        self.chatgpt.update_settings(self.settings)
        return self.settings


def create_app(settings: Settings | None = None) -> FastAPI:
    configured = settings or Settings.load()
    state = AppState(configured)
    mcp_server = create_mcp_server(state.chatgpt)
    mcp_app = mcp_server.streamable_http_app(
        streamable_http_path="/mcp",
        stateless_http=True,
        host=configured.host,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        async with mcp_app.router.lifespan_context(mcp_app):
            yield

    app = FastAPI(title="AutoResearch", version="0.3.0", lifespan=lifespan)
    app.state.autoresearch = state
    app.state.mcp_server = mcp_server

    @app.get("/api/health")
    async def health(request: Request) -> dict[str, Any]:
        return {
            "ok": True,
            "version": "0.3.0",
            "active_research": len(state.orchestrator._tasks),
            "active_experiments": len(state.experiments._tasks),
            "chatgpt_bridge": "ready",
            "mcp_endpoint": f"{str(request.base_url).rstrip('/')}/mcp",
        }

    @app.get("/api/settings")
    async def get_settings() -> dict[str, object]:
        return state.settings.public()

    @app.patch("/api/settings")
    async def patch_settings(request: SettingsPatch) -> dict[str, object]:
        EnvStore(state.settings.root / ".env").update(request.as_env())
        return state.reload_settings().public()

    @app.get("/api/chatgpt/connection")
    async def get_chatgpt_connection() -> dict[str, Any]:
        return connection_status(state.settings.root)

    @app.post("/api/chatgpt/connection/launch", status_code=202)
    async def launch_chatgpt_connection(
        request: ChatGPTTunnelLaunchRequest,
    ) -> dict[str, str]:
        if not request.confirm_launch:
            raise HTTPException(status_code=400, detail="启动本机连接窗口前必须确认")
        try:
            launch_tunnel_console(state.settings.root, request.action)
        except OSError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"status": "launched", "action": request.action}

    @app.get("/api/models/free")
    async def free_models() -> dict[str, Any]:
        try:
            async with OpenRouterClient(state.settings) as client:
                return {"models": await client.free_models()}
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/api/research", status_code=202)
    async def start_research(request: ResearchRequest) -> dict[str, Any]:
        project = state.orchestrator.create_and_start(request)
        return project.model_dump()

    @app.post("/api/chatgpt/projects", status_code=201)
    async def create_chatgpt_project(request: ChatGPTProjectRequest) -> dict[str, Any]:
        return state.chatgpt.create_project(request)

    @app.post("/api/chatgpt/projects/{project_id}/notes")
    async def save_chatgpt_note(
        project_id: str, request: ResearchNoteRequest
    ) -> dict[str, Any]:
        try:
            return state.chatgpt.save_note(project_id, request)
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/chatgpt/projects/{project_id}/code")
    async def save_chatgpt_code(
        project_id: str, request: CodeSaveRequest
    ) -> dict[str, Any]:
        try:
            return state.chatgpt.save_code(project_id, request)
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/chatgpt/projects/{project_id}/generate", status_code=202)
    async def generate_chatgpt_code(
        project_id: str, request: CodeGenerationRequest
    ) -> dict[str, Any]:
        try:
            return await state.chatgpt.generate_code(project_id, request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (LLMError, OSError) as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/api/chatgpt/projects/{project_id}/sources")
    async def search_chatgpt_sources(
        project_id: str, request: SourceSearchRequest
    ) -> dict[str, Any]:
        try:
            return await state.chatgpt.search_sources(project_id, request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/api/chatgpt/projects/{project_id}/analyses")
    async def record_chatgpt_analysis(
        project_id: str, request: ChatGPTAnalysisRequest
    ) -> dict[str, Any]:
        try:
            return state.chatgpt.record_analysis(project_id, request)
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/projects")
    async def list_projects() -> list[dict[str, Any]]:
        return [item.model_dump() for item in state.db.list_projects()]

    @app.get("/api/projects/{project_id}")
    async def get_project(project_id: str) -> dict[str, Any]:
        project = state.db.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="研究项目不存在")
        return project.model_dump()

    @app.get("/api/projects/{project_id}/status")
    async def get_project_status(project_id: str) -> dict[str, Any]:
        try:
            return state.chatgpt.status(project_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/projects/{project_id}/experiments")
    async def list_project_experiments(project_id: str) -> list[dict[str, Any]]:
        if not state.db.get_project(project_id):
            raise HTTPException(status_code=404, detail="研究项目不存在")
        return state.db.list_experiments(project_id)

    @app.get("/api/projects/{project_id}/events")
    async def get_events(project_id: str, after: int = 0) -> list[dict[str, Any]]:
        if not state.db.get_project(project_id):
            raise HTTPException(status_code=404, detail="研究项目不存在")
        return state.db.list_events(project_id, max(0, after))

    @app.get("/api/projects/{project_id}/sources")
    async def get_sources(project_id: str) -> list[dict[str, Any]]:
        if not state.db.get_project(project_id):
            raise HTTPException(status_code=404, detail="研究项目不存在")
        return [source.model_dump() for source in state.db.list_sources(project_id)]

    @app.get("/api/projects/{project_id}/manifest")
    async def get_manifest(project_id: str) -> dict[str, Any]:
        project = state.db.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="研究项目不存在")
        path = Path(project.workspace) / "generated" / "experiment_manifest.json"
        if not path.exists():
            return {}
        import json

        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    @app.post("/api/projects/{project_id}/open")
    async def open_project(project_id: str) -> dict[str, bool]:
        project = state.db.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="研究项目不存在")
        try:
            DesktopBridge.open_vscode(Path(project.workspace))
        except OSError as exc:
            raise HTTPException(status_code=503, detail="无法启动 VS Code，请确认 code 命令可用") from exc
        return {"opened": True}

    @app.post("/api/autodl/instances", status_code=202)
    async def create_autodl(request: AutoDLCreateRequest) -> dict[str, Any]:
        if not request.confirm_billable:
            raise HTTPException(status_code=400, detail="创建按量计费实例前必须确认费用")
        client = AutoDLClient(state.settings)
        try:
            choice = await client.create_preferred(request)
            return {
                "instance_uuid": choice.instance_uuid,
                "gpu_spec": choice.gpu_spec,
                "status": "creating",
            }
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        finally:
            await client.close()

    @app.get("/api/autodl/instances")
    async def list_autodl() -> dict[str, Any]:
        client = AutoDLClient(state.settings)
        try:
            return {"instances": await client.list_instances()}
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        finally:
            await client.close()

    @app.get("/api/autodl/instances/{instance_uuid}")
    async def get_autodl(instance_uuid: str) -> dict[str, Any]:
        client = AutoDLClient(state.settings)
        try:
            status = await client.status(instance_uuid)
            snapshot = await client.snapshot(instance_uuid)
            return {"instance_uuid": instance_uuid, "status": status, "ssh": extract_ssh(snapshot)}
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        finally:
            await client.close()

    @app.post("/api/experiments", status_code=202)
    async def start_experiment(request: ExperimentStartRequest) -> dict[str, str]:
        try:
            experiment_id = state.experiments.start(request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"experiment_id": experiment_id, "status": "preparing"}

    @app.post("/api/chatgpt/autodl-experiments", status_code=202)
    async def start_chatgpt_experiment(
        request: AutoDLExperimentRequest,
    ) -> dict[str, Any]:
        try:
            return await state.chatgpt.start_autodl_experiment(request)
        except (ValueError, AutoDLError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/experiments/{experiment_id}")
    async def get_experiment(experiment_id: str) -> dict[str, Any]:
        experiment = state.db.get_experiment(experiment_id)
        if not experiment:
            raise HTTPException(status_code=404, detail="实验不存在")
        return experiment

    @app.post("/api/git/sync")
    async def git_sync(request: GitSyncRequest) -> dict[str, Any]:
        root = state.settings.root
        branch = request.branch
        if request.project_id:
            project = state.db.get_project(request.project_id)
            if not project:
                raise HTTPException(status_code=404, detail="研究项目不存在")
            root = Path(project.workspace) / "generated"
            if request.branch == "main":
                branch = f"research/{request.project_id}"
        remote = request.remote_url or state.settings.github_remote_url
        try:
            return await asyncio.to_thread(
                GitSync(root).sync, remote, branch, request.message
            )
        except GitSyncError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    # Reuse the MCP SDK's exact Streamable HTTP route inside the same local
    # process. The child lifespan above owns its session manager.
    app.router.routes.extend(mcp_app.routes)

    # Static assets belong to the installed package; ``settings.root`` may be a
    # temporary or user-selected data root in tests and embedded deployments.
    static = Path(__file__).resolve().parent / "static"

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(static / "index.html")

    app.mount("/static", StaticFiles(directory=static), name="static")
    return app


app = create_app()
