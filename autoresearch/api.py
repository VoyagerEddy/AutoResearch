from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager, suppress
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
    AutoDLBrowserLoginLaunchRequest,
    AutoDLCreateRequest,
    AutoDLExperimentRequest,
    ChatGPTAnalysisRequest,
    ChatGPTProjectRequest,
    ChatGPTTunnelLaunchRequest,
    CodeGenerationRequest,
    CodeSaveRequest,
    ColabBrowserLaunchRequest,
    ColabNotebookBindingRequest,
    ColabObservationRequest,
    ColabPrepareRequest,
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
from .services.autodl import AutoDLClient, AutoDLCreateUncertain, AutoDLError, extract_ssh
from .services.colab import ColabConflict
from .services.colab_browser import ColabBrowserError
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
        self.colab = self.chatgpt.colab
        self._colab_launch_lock = asyncio.Lock()

    def reload_settings(self) -> Settings:
        self.settings = Settings.load(self.settings.root)
        self.orchestrator.update_settings(self.settings)
        self.experiments.update_settings(self.settings)
        self.chatgpt.update_settings(self.settings)
        return self.settings

    async def _launch_colab_worker(self, job: dict[str, Any]) -> dict[str, Any]:
        from .colab_browser_cli import launch_colab_worker

        project_id = job['project_id']
        pid = await asyncio.to_thread(launch_colab_worker, self.settings.root, project_id,
                                      api_url=f'http://127.0.0.1:{self.settings.port}',
                                      browser=job.get('browser', 'auto'))
        self.colab.worker_launched(project_id, pid)
        return {'status': 'worker_launched', 'pid': pid, 'project_id': project_id,
                'autorun_enabled': job.get('autorun_enabled', False),
                'message': 'The browser worker will check availability through the normal Colab UI.'}

    async def launch_colab(self, project_id: str, browser: str, *, resume_auth: bool = False) -> dict[str, Any]:
        """Persist explicit authorization, then launch or respect a saved cooldown."""
        async with self._colab_launch_lock:
            # Dashboard and ChatGPT use the same authorization and browser
            # transitions. Worker startup is bounded by the shared profile lock.
            result = await asyncio.to_thread(self.chatgpt.start_colab, project_id,
                                             confirm_execute=True, resume_auth=resume_auth, browser=browser)
            if result['status'] == 'browser_launched':
                return {**result, 'status': 'worker_launched'}
            return result

    async def check_colab(self, project_id: str, browser: str) -> dict[str, Any]:
        """A platform selection starts monitoring without resuming a manual pause."""
        job = self.colab.status(project_id)
        if not job.get('prepared') or job['status'] in {'completed', 'failed', 'cancelled', 'blocked'}:
            return {'status': 'preparation_required' if not job.get('prepared') else job['status'],
                    'project_id': project_id, 'job': job, 'resource_agent_pid': None,
                    'message': '请先准备 Colab 实验。' if not job.get('prepared') else '当前试验已停止或结束。'}
        result = await self.launch_colab(project_id, browser, resume_auth=False)
        return {**result, 'job': self.colab.status(project_id),
                'resource_agent_pid': (result.get('resource_agent') or {}).get('pid')}

    async def launch_due_colab_jobs(self) -> list[dict[str, Any]]:
        """One bounded dispatch per tick; all workers share a browser profile."""
        from .colab_browser_cli import ColabProfileBusy, launch_colab_login_browser, login_browser_is_running

        async with self._colab_launch_lock:
            jobs = await asyncio.to_thread(lambda: [self.colab.status(project.id) for project in self.db.list_projects(limit=10000)])
            if any(job['worker_running'] for job in jobs):
                return []
            if login_browser_is_running(self.settings.root) is not None:
                return []
            for index, job in enumerate(jobs):
                if job['status'] != 'auth_required' or not job.get('autorun_enabled') or job.get('source_current') is False:
                    continue
                project_id = job['project_id']
                if job.get('ordinary_login'):
                    jobs[index] = await asyncio.to_thread(self.colab.complete_ordinary_login, project_id)
                    continue
                details = (job.get('last_observation') or {}).get('details') or {}
                if (details.get('auth_kind') == 'google_sign_in' and details.get('login_browser') == 'required'
                        and not details.get('runtime_active') and not job.get('ordinary_login_recheck')):
                    try:
                        pid = await asyncio.to_thread(launch_colab_login_browser, self.settings.root,
                                                      browser=job.get('browser', 'auto'))
                    except (ColabBrowserError, OSError):
                        logging.getLogger(__name__).warning('Could not open the ordinary Colab login window for %s', project_id)
                        return []
                    await asyncio.to_thread(self.colab.ordinary_login_opened, project_id, pid)
                    return [{'status': 'login_browser_opened', 'pid': pid, 'project_id': project_id}]
            for job in sorted(jobs, key=lambda item: item.get('next_attempt_at') or ''):
                if not job.get('autorun_enabled') or not job['can_attempt']:
                    continue
                try:
                    return [await self._launch_colab_worker(job)]
                except ColabProfileBusy:
                    return []
                except (ColabBrowserError, OSError) as exc:
                    self.colab.observe(job['project_id'], ColabObservationRequest(status='network', details={
                        'reason': str(exc), 'scheduler': True}))
                return []
            return []

    async def run_colab_scheduler(self) -> None:
        while True:
            # Delay the first tick so the HTTP listener is ready before workers
            # connect, and avoid tight polling during provider cooldowns.
            await asyncio.sleep(30)
            try:
                await self.ensure_colab_resource_monitors()
                await self.launch_due_colab_jobs()
            except Exception:
                logging.getLogger(__name__).exception('Colab retry scheduler tick failed')

    async def ensure_colab_resource_monitors(self) -> list[dict[str, Any]]:
        """Keep a dedicated observer alive while an authorized Colab job waits."""
        launched = []
        for project in self.db.list_projects(limit=10000):
            job = self.colab.status(project.id)
            if (not job.get('prepared') or not job.get('autorun_enabled') or job.get('source_current') is False
                    or job['status'] in {'completed', 'failed', 'cancelled', 'blocked'}):
                continue
            pid = await self._ensure_colab_resource_monitor(job)
            if pid is not None:
                launched.append({'project_id': project.id, 'pid': pid})
        return launched

    async def _ensure_colab_resource_monitor(self, job: dict[str, Any]) -> int | None:
        from .services.colab_resource_monitor import launch_resource_monitor

        try:
            return await asyncio.to_thread(
                launch_resource_monitor, self.settings.root, job['project_id'],
                api_url=f'http://127.0.0.1:{self.settings.port}', browser=job.get('browser', 'auto'))
        except (ColabBrowserError, OSError, ValueError) as exc:
            # Observer failures do not change an allocation result or cooldown.
            logging.getLogger(__name__).warning(
                'Colab resource observer could not start for %s: %s', job['project_id'], type(exc).__name__)
            return None


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
            scheduler = asyncio.create_task(state.run_colab_scheduler(), name='colab-retry-scheduler')
            try:
                yield
            finally:
                scheduler.cancel()
                with suppress(asyncio.CancelledError):
                    await scheduler

    app = FastAPI(title="AutoResearch", version="0.3.0", lifespan=lifespan)
    app.state.autoresearch = state
    app.state.mcp_server = mcp_server

    @app.get("/api/health")
    async def health(request: Request) -> dict[str, Any]:
        tools = await mcp_server.list_tools()
        return {
            "ok": True,
            "version": "0.3.0",
            "application_root": str(state.settings.root),
            "active_research": len(state.orchestrator._tasks),
            "active_experiments": len(state.experiments._tasks),
            "chatgpt_bridge": "ready",
            "mcp_server_version": "0.3.1",
            "mcp_tools": sorted(tool.name for tool in tools),
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
        return await asyncio.to_thread(connection_status, state.settings.root)

    @app.post("/api/chatgpt/connection/launch", status_code=202)
    async def launch_chatgpt_connection(
        request: ChatGPTTunnelLaunchRequest,
    ) -> dict[str, str]:
        if not request.confirm_launch:
            raise HTTPException(status_code=400, detail="Confirmation is required before opening the local connection window")
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
            raise HTTPException(status_code=404, detail="Research project not found")
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
            raise HTTPException(status_code=404, detail="Research project not found")
        return state.db.list_experiments(project_id)

    @app.get("/api/projects/{project_id}/readiness")
    async def get_experiment_readiness(project_id: str) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(state.chatgpt.experiment_readiness, project_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    def colab_error(exc: Exception) -> HTTPException:
        if isinstance(exc, LookupError):
            return HTTPException(status_code=404, detail=str(exc))
        return HTTPException(status_code=409 if isinstance(exc, ColabConflict) else 400, detail=str(exc))

    @app.get("/api/projects/{project_id}/colab")
    async def get_colab_status(project_id: str) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(state.colab.status, project_id)
        except (LookupError, ValueError, OSError) as exc:
            raise colab_error(exc) from exc

    @app.get("/api/projects/{project_id}/colab/resources")
    async def get_colab_resources(project_id: str) -> dict[str, Any]:
        from .services.colab_resource_monitor import resource_monitor_status

        if not state.db.get_project(project_id):
            raise HTTPException(status_code=404, detail="Research project not found")
        try:
            return await asyncio.to_thread(resource_monitor_status, state.settings.root, project_id)
        except (ValueError, OSError) as exc:
            raise colab_error(exc) from exc

    @app.post("/api/projects/{project_id}/colab/prepare")
    async def prepare_colab(project_id: str, request: ColabPrepareRequest) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(state.colab.prepare, project_id, request)
        except (LookupError, ValueError, OSError) as exc:
            raise colab_error(exc) from exc

    @app.post("/api/projects/{project_id}/colab/attempt")
    async def claim_colab_attempt(project_id: str) -> dict[str, Any]:
        try:
            return state.colab.attempt(project_id)
        except (LookupError, ValueError, OSError) as exc:
            raise colab_error(exc) from exc

    @app.post("/api/projects/{project_id}/colab/observations")
    async def record_colab_observation(project_id: str, request: ColabObservationRequest) -> dict[str, Any]:
        try:
            return state.colab.observe(project_id, request)
        except (LookupError, ValueError, OSError) as exc:
            raise colab_error(exc) from exc

    @app.post("/api/projects/{project_id}/colab/notebook-binding")
    async def bind_colab_notebook(project_id: str, request: ColabNotebookBindingRequest) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(state.colab.bind_notebook,project_id,request)
        except (LookupError,ValueError,OSError) as exc:
            raise colab_error(exc) from exc

    @app.post("/api/projects/{project_id}/colab/resume")
    async def resume_colab(project_id: str) -> dict[str, Any]:
        try:
            job = state.colab.status(project_id)
            if job['status'] != 'auth_required':
                raise ColabConflict('Only an authentication pause can be resumed immediately; GPU retry cooldown remains in force')
            if job.get('autorun_enabled'):
                await state.launch_colab(project_id, job.get('browser', 'auto'), resume_auth=True)
                return state.colab.status(project_id)
            return state.colab.resume(project_id)
        except (LookupError, ValueError, OSError, ColabBrowserError) as exc:
            raise colab_error(exc) from exc

    @app.post("/api/projects/{project_id}/colab/cancel")
    async def cancel_colab(project_id: str) -> dict[str, Any]:
        try:
            return state.colab.cancel(project_id)
        except (LookupError, ValueError, OSError) as exc:
            raise colab_error(exc) from exc

    @app.post("/api/projects/{project_id}/colab/browser", status_code=202)
    @app.post("/api/projects/{project_id}/colab/launch", status_code=202)
    async def launch_colab_browser(project_id: str, request: ColabBrowserLaunchRequest) -> dict[str, Any]:
        try:
            return await state.launch_colab(project_id, request.browser)
        except (LookupError, ValueError, OSError, ColabBrowserError) as exc:
            raise colab_error(exc) from exc

    @app.post("/api/projects/{project_id}/colab/check", status_code=202)
    async def check_colab_resources(project_id: str, request: ColabBrowserLaunchRequest) -> dict[str, Any]:
        try:
            return await state.check_colab(project_id, request.browser)
        except (LookupError, ValueError, OSError, ColabBrowserError) as exc:
            raise colab_error(exc) from exc

    @app.post("/api/projects/{project_id}/colab/login", status_code=202)
    async def launch_colab_login(project_id: str, request: ColabBrowserLaunchRequest) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(state.chatgpt.open_colab_login, project_id, browser=request.browser)
        except (LookupError, ValueError, OSError, ColabBrowserError) as exc:
            raise colab_error(exc) from exc

    @app.get("/api/projects/{project_id}/colab/notebook.ipynb")
    async def download_colab_notebook(project_id: str) -> FileResponse:
        try:
            path = state.colab.export_path(project_id, 'notebook')
            return FileResponse(path, media_type='application/x-ipynb+json', filename=path.name)
        except (LookupError, ValueError, OSError) as exc:
            raise colab_error(exc) from exc

    @app.get("/api/projects/{project_id}/colab/bundle.zip")
    async def download_colab_bundle(project_id: str) -> FileResponse:
        try:
            path = state.colab.export_path(project_id, 'bundle')
            return FileResponse(path, media_type='application/zip', filename='source.zip')
        except (LookupError, ValueError, OSError) as exc:
            raise colab_error(exc) from exc

    @app.get("/api/projects/{project_id}/colab/inputs.zip")
    async def download_colab_inputs(project_id: str) -> FileResponse:
        try:
            path = state.colab.export_path(project_id, 'inputs')
            return FileResponse(path, media_type='application/zip', filename='inputs.zip')
        except (LookupError, ValueError, OSError) as exc:
            raise colab_error(exc) from exc

    @app.get("/api/projects/{project_id}/events")
    async def get_events(project_id: str, after: int = 0) -> list[dict[str, Any]]:
        if not state.db.get_project(project_id):
            raise HTTPException(status_code=404, detail="Research project not found")
        return state.db.list_events(project_id, max(0, after))

    @app.get("/api/projects/{project_id}/sources")
    async def get_sources(project_id: str) -> list[dict[str, Any]]:
        if not state.db.get_project(project_id):
            raise HTTPException(status_code=404, detail="Research project not found")
        return [source.model_dump() for source in state.db.list_sources(project_id)]

    @app.get("/api/projects/{project_id}/manifest")
    async def get_manifest(project_id: str) -> dict[str, Any]:
        project = state.db.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Research project not found")
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
            raise HTTPException(status_code=404, detail="Research project not found")
        try:
            DesktopBridge.open_vscode(Path(project.workspace))
        except OSError as exc:
            raise HTTPException(status_code=503, detail="Could not start VS Code; ensure the 'code' command is available") from exc
        return {"opened": True}

    @app.get("/api/autodl/preflight")
    async def autodl_preflight(
        verify_api: bool = False, image_uuid: str | None = None
    ) -> dict[str, Any]:
        return await state.chatgpt.autodl_readiness(verify_api=verify_api, image_uuid=image_uuid)

    @app.post("/api/autodl/browser-login/launch", status_code=202)
    async def launch_autodl_browser_login(
        request: AutoDLBrowserLoginLaunchRequest,
    ) -> dict[str, Any]:
        if not request.confirm_launch:
            raise HTTPException(
                status_code=400,
                detail="Confirmation is required before opening the AutoDL login assistant",
            )
        try:
            pid = DesktopBridge.launch_autodl_browser_login(
                state.settings.root,
                action=request.action,
                browser=request.browser,
                open_page=request.open_page,
                keep_open=request.keep_open,
            )
        except OSError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "status": "launched",
            "action": request.action,
            "browser": request.browser,
            "open_page": request.open_page,
            "pid": pid,
            "credential_storage": "windows-user-encrypted",
        }

    @app.post("/api/autodl/instances", status_code=202)
    async def create_autodl(request: AutoDLCreateRequest) -> dict[str, Any]:
        if not request.confirm_billable:
            raise HTTPException(status_code=400, detail="Cost confirmation is required before provisioning a pay-as-you-go instance")
        client = AutoDLClient(state.settings)
        try:
            choice = await client.create_preferred(request)
            return {
                "instance_uuid": choice.instance_uuid,
                "gpu_spec": choice.gpu_spec,
                "status": "creating",
            }
        except AutoDLCreateUncertain as exc:
            raise HTTPException(
                status_code=409,
                detail={"code": "autodl_create_uncertain", "message": str(exc)},
            ) from exc
        except AutoDLError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
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
            raise HTTPException(status_code=404, detail="Experiment not found")
        return experiment

    @app.post("/api/git/sync")
    async def git_sync(request: GitSyncRequest) -> dict[str, Any]:
        root = state.settings.root
        branch = request.branch
        if request.project_id:
            project = state.db.get_project(request.project_id)
            if not project:
                raise HTTPException(status_code=404, detail="Research project not found")
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
