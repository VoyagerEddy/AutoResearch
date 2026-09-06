from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import Settings
from .db import Database
from .domain import (
    AutoDLCreateRequest,
    AutoDLExperimentRequest,
    ChatGPTAnalysisRequest,
    ChatGPTProjectRequest,
    CodeGenerationRequest,
    CodeSaveRequest,
    ExperimentStartRequest,
    ResearchNoteRequest,
    ResearchRequest,
    Source,
    SourceSearchRequest,
    SSHConnection,
)
from .experiments import ExperimentManager
from .services.artifacts import ArtifactStore, safe_slug
from .services.experiment_readiness import inspect_experiment
from .services.autodl import AutoDLClient, AutoDLError, extract_ssh
from .services.llm import LLMError, OpenRouterClient
from .services.search import ResearchSearch


class ChatGPTBridge:
    """Use cases exposed to ChatGPT while AutoResearch remains the system of record."""

    def __init__(
        self,
        db: Database,
        settings: Settings,
        experiments: ExperimentManager,
    ) -> None:
        self.db = db
        self.settings = settings
        self.experiments = experiments

    def update_settings(self, settings: Settings) -> None:
        self.settings = settings

    def create_project(self, request: ChatGPTProjectRequest) -> dict[str, Any]:
        project = self.db.create_project(
            ResearchRequest(topic=request.topic, notes=request.notes),
            self.settings.workspace_dir,
        )
        store = ArtifactStore(Path(project.workspace))
        brief = f"# {project.topic}\n\n"
        if request.notes:
            brief += f"## Initial constraints\n\n{request.notes}\n"
        else:
            brief += "ChatGPT owns research reasoning; AutoResearch persists notes, code, and experiments.\n"
        store.write_text("research/chatgpt-brief.md", brief)
        self.db.update_project(
            project.id,
            status="running",
            phase="chatgpt_thinking",
            progress=10,
            summary="ChatGPT is handling research reasoning; AutoResearch is ready to save notes or code and start experiments.",
        )
        self.db.add_event(
            project.id,
            "chatgpt_thinking",
            "Created a ChatGPT collaborative research project and awaiting the next action",
            details={"source": "chatgpt"},
        )
        return self.status(project.id)

    def list_projects(self) -> dict[str, Any]:
        projects = []
        for project in self.db.list_projects():
            item = project.model_dump()
            item.pop("workspace", None)
            item["dashboard_url"] = self._dashboard(project.id)
            item["experiment_count"] = len(self.db.list_experiments(project.id))
            projects.append(item)
        return {"projects": projects, "count": len(projects)}

    def save_note(self, project_id: str, request: ResearchNoteRequest) -> dict[str, Any]:
        project = self._project(project_id)
        relative = f"research/chatgpt-{safe_slug(request.title)}.md"
        ArtifactStore(Path(project.workspace)).write_text(
            relative, f"# {request.title}\n\n{request.markdown.rstrip()}\n"
        )
        self.db.update_project(
            project_id,
            status="running",
            phase="chatgpt_thinking",
            progress=max(project.progress, 20),
            summary=f"Saved ChatGPT research note: {request.title}",
        )
        self.db.add_event(
            project_id,
            "chatgpt_thinking",
            f"ChatGPT saved research note '{request.title}'",
            details={"path": relative},
        )
        return {"project_id": project_id, "path": relative, "dashboard_url": self._dashboard(project_id)}

    def save_code(self, project_id: str, request: CodeSaveRequest) -> dict[str, Any]:
        project = self._project(project_id)
        store = ArtifactStore(Path(project.workspace))
        files = [item.model_dump() for item in request.files]
        written = store.materialize_files(files, "generated")
        if request.experiment_manifest:
            store.write_json("generated/experiment_manifest.json", request.experiment_manifest)
        summary = request.summary.strip() or f"ChatGPT saved {len(written)} code file(s)"
        self.db.update_project(
            project_id,
            status="ready",
            phase="ready",
            progress=100,
            summary=summary,
            error="",
        )
        relative = [path.relative_to(Path(project.workspace)).as_posix() for path in written]
        self.db.add_event(
            project_id,
            "ready",
            f"ChatGPT saved {len(written)} code file(s) to the experiment workspace",
            details={"files": relative, "producer": "chatgpt"},
        )
        return {
            "project_id": project_id,
            "saved_files": relative,
            "status": "ready",
            "dashboard_url": self._dashboard(project_id),
        }

    async def generate_code(
        self, project_id: str, request: CodeGenerationRequest
    ) -> dict[str, Any]:
        project = self._project(project_id)
        if not self.settings.openrouter_api_key:
            raise LLMError("OpenRouter API key is not configured; AutoResearch cannot delegate code generation")
        self.db.update_project(
            project_id, status="running", phase="generating", progress=74, error=""
        )
        self.db.add_event(
            project_id,
            "generating",
            "ChatGPT delegated code implementation to the AutoResearch model",
            details={"model": request.model or self.settings.openrouter_model},
        )
        store = ArtifactStore(Path(project.workspace))
        context = self._research_context(project)
        sources = self.db.list_sources(project_id)
        source_digest = "\n\n".join(
            f"[S{index}] {source.title}\n{source.url}\n{source.abstract[:900]}"
            for index, source in enumerate(sources, 1)
        )[:24_000]
        prompt = f"""ChatGPT has completed the research reasoning. Implement the plan as a minimal, runnable, and reproducible Python experiment project.
Research topic: {project.topic}
ChatGPT implementation requirements: {request.instructions}
Saved research context:
{context[:36_000]}

Saved sources:
{source_digest or 'No sources saved'}

Return strict JSON only:
{{
  "summary": "Implementation summary",
  "files": [{{"path": "relative/path", "content": "complete file content"}}],
  "experiment": {{
    "setup_commands": ["safe, non-interactive installation commands"],
    "dataset_commands": [],
    "run_command": "python experiment.py",
    "metrics_file": "results/metrics.json"
  }}
}}
Include README.md, requirements.txt, a runnable entry point, a test or validation script, and .gitignore. Do not write secrets, absolute paths, or fabricated experiment results."""
        try:
            async with OpenRouterClient(self.settings) as llm:
                bundle = await llm.chat_json(
                    [
                        {
                            "role": "system",
                            "content": "You are an AutoResearch experiment engineer. Preserve ChatGPT's research decisions and implement code only. Return complete files, not patches.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    model=request.model,
                    max_tokens=12_000,
                )
            files = bundle.get("files")
            if not isinstance(files, list) or not files:
                raise LLMError("The AutoResearch model returned no code files")
            written = store.materialize_files(files, "generated")
            manifest = bundle.get("experiment")
            store.write_json(
                "generated/experiment_manifest.json",
                manifest if isinstance(manifest, dict) else {},
            )
            store.write_json(
                "research/code-generation.json",
                {
                    "producer": "AutoResearch/OpenRouter",
                    "model": request.model or self.settings.openrouter_model,
                    "instructions": request.instructions,
                    "files": [path.relative_to(Path(project.workspace)).as_posix() for path in written],
                },
            )
        except Exception as exc:
            self.db.update_project(
                project_id,
                status="running",
                phase="chatgpt_thinking",
                progress=60,
                error="",
            )
            self.db.add_event(
                project_id,
                "generating",
                f"AutoResearch code generation failed: {exc}",
                level="error",
            )
            raise
        summary = str(bundle.get("summary") or f"AutoResearch generated {len(written)} experiment file(s)")[:2000]
        self.db.update_project(
            project_id,
            status="ready",
            phase="ready",
            progress=100,
            summary=summary,
            model=request.model or self.settings.openrouter_model,
            error="",
        )
        relative = [path.relative_to(Path(project.workspace)).as_posix() for path in written]
        self.db.add_event(
            project_id,
            "ready",
            f"AutoResearch model generated {len(written)} experiment file(s)",
            details={"files": relative, "producer": "openrouter"},
        )
        return {
            "project_id": project_id,
            "saved_files": relative,
            "model": request.model or self.settings.openrouter_model,
            "status": "ready",
            "dashboard_url": self._dashboard(project_id),
        }

    async def search_sources(
        self, project_id: str, request: SourceSearchRequest
    ) -> dict[str, Any]:
        project = self._project(project_id)
        queries = [query.strip()[:300] for query in request.queries if query.strip()]
        if not queries:
            raise ValueError("At least one nonempty search query is required")
        self.db.update_project(
            project_id, status="running", phase="searching", progress=30, error=""
        )
        self.db.add_event(project_id, "searching", "ChatGPT requested an AutoResearch source search")
        searcher = ResearchSearch(self.settings)
        try:
            found = await searcher.search(queries, request.max_sources)
            existing = self.db.list_sources(project_id)
            merged: dict[str, Source] = {}
            for source in [*existing, *found]:
                key = source.url.strip().lower() or source.title.strip().lower()
                previous = merged.get(key)
                if previous is None or source.citation_count > previous.citation_count:
                    merged[key] = source
            sources = list(merged.values())[: request.max_sources]
            self.db.replace_sources(project_id, sources)
            store = ArtifactStore(Path(project.workspace))
            store.write_json("research/sources.json", [item.model_dump() for item in sources])
            downloaded: list[Path] = []
            if request.download_papers:
                downloaded = await searcher.download_pdfs(
                    sources, Path(project.workspace) / "research" / "papers"
                )
        finally:
            await searcher.close()
        self.db.update_project(
            project_id,
            status="running",
            phase="chatgpt_thinking",
            progress=max(project.progress, 45),
            summary=f"Saved {len(sources)} sources and waiting for ChatGPT to continue the analysis.",
        )
        self.db.add_event(
            project_id,
            "chatgpt_thinking",
            f"Saved {len(sources)} sources and returned them to ChatGPT for analysis",
            details={"source_count": len(sources), "downloaded_pdfs": len(downloaded)},
        )
        return {
            "project_id": project_id,
            "source_count": len(sources),
            "downloaded_pdfs": len(downloaded),
            "sources": [item.model_dump() for item in sources],
            "dashboard_url": self._dashboard(project_id),
        }

    def record_analysis(
        self, project_id: str, request: ChatGPTAnalysisRequest
    ) -> dict[str, Any]:
        project = self._project(project_id)
        suffix = request.experiment_id or safe_slug(request.analysis[:80])
        relative = f"reports/chatgpt-analysis-{safe_slug(suffix, 36)}.md"
        body = "# ChatGPT Experiment Analysis\n\n"
        if request.experiment_id:
            body += f"Experiment ID: `{request.experiment_id}`\n\n"
        body += request.analysis.rstrip() + "\n"
        if request.recommendations:
            body += "\n## Next Steps\n\n" + "\n".join(
                f"- {item}" for item in request.recommendations
            ) + "\n"
        ArtifactStore(Path(project.workspace)).write_text(relative, body)
        self.db.update_project(
            project_id,
            status="ready",
            phase="chatgpt_analysis",
            progress=100,
            summary="ChatGPT analyzed and saved the experiment results.",
            error="",
        )
        self.db.add_event(
            project_id,
            "chatgpt_analysis",
            "ChatGPT analyzed the experiment results and saved the conclusions",
            details={"path": relative, "experiment_id": request.experiment_id or ""},
        )
        return {"project_id": project_id, "path": relative, "dashboard_url": self._dashboard(project_id)}

    def read_artifact(self, project_id: str, path: str) -> dict[str, Any]:
        project = self._project(project_id)
        content = ArtifactStore(Path(project.workspace)).read_text(path)
        return {"project_id": project_id, "path": path, "content": content}

    def status(self, project_id: str) -> dict[str, Any]:
        project = self._project(project_id)
        store = ArtifactStore(Path(project.workspace))
        experiments = self.db.list_experiments(project_id)
        project_data = project.model_dump()
        project_data.pop("workspace", None)
        collaborative = (Path(project.workspace) / "research" / "chatgpt-brief.md").is_file()
        return {
            "project": project_data,
            "reasoning_owner": "Current ChatGPT conversation" if collaborative else "AutoResearch/OpenRouter",
            "execution_model": project.model or self.settings.openrouter_model,
            "sources": [item.model_dump() for item in self.db.list_sources(project_id)],
            "experiments": experiments,
            "recent_events": self.db.list_events(project_id)[-30:],
            "artifacts": store.list_files(),
            "experiment_readiness": self.experiment_readiness(project_id),
            "dashboard_url": self._dashboard(project_id),
        }

    def experiment_readiness(self, project_id: str) -> dict[str, Any]:
        project = self._project(project_id)
        return {"project_id": project_id, **inspect_experiment(Path(project.workspace))}

    async def autodl_readiness(
        self, *, verify_api: bool = False, image_uuid: str | None = None
    ) -> dict[str, Any]:
        client = AutoDLClient(self.settings)
        try:
            return await client.preflight(verify_api=verify_api, image_uuid=image_uuid)
        finally:
            await client.close()

    async def create_autodl_instance(
        self, request: AutoDLCreateRequest, project_id: str | None = None
    ) -> dict[str, Any]:
        if not request.confirm_billable:
            raise ValueError("Explicit user confirmation is required before creating a billable AutoDL instance")
        if project_id:
            self._project(project_id)
        client = AutoDLClient(self.settings)
        try:
            choice = await client.create_preferred(request)
            result = {
                "instance_uuid": choice.instance_uuid,
                "gpu_spec": choice.gpu_spec,
                "status": "creating",
                "message": "The instance is being created. Billing and lifecycle management remain in the AutoDL console.",
            }
            if project_id:
                self.db.add_event(
                    project_id,
                    "autodl",
                    f"Created billable AutoDL instance {choice.instance_uuid} ({choice.gpu_spec})",
                    details={"instance_uuid": choice.instance_uuid, "gpu_spec": choice.gpu_spec},
                )
                result["dashboard_url"] = self._dashboard(project_id)
            return result
        finally:
            await client.close()

    async def autodl_instance_status(self, instance_uuid: str) -> dict[str, Any]:
        client = AutoDLClient(self.settings)
        try:
            status = await client.status(instance_uuid)
            ssh_ready = False
            if status == "running":
                ssh_ready = extract_ssh(await client.snapshot(instance_uuid)) is not None
            return {
                "instance_uuid": instance_uuid,
                "status": status,
                "ssh_ready": ssh_ready,
                "message": "The experiment can start when ssh_ready is true. Connection credentials are never returned to ChatGPT.",
            }
        finally:
            await client.close()

    async def start_autodl_experiment(
        self, request: AutoDLExperimentRequest
    ) -> dict[str, Any]:
        if not request.confirm_execute:
            raise ValueError("Explicit user confirmation is required before uploading and executing code")
        self._project(request.project_id)
        client = AutoDLClient(self.settings)
        try:
            status = await client.status(request.instance_uuid)
            if status != "running":
                raise AutoDLError(f"The AutoDL instance is not running (current status: {status})")
            snapshot = await client.snapshot(request.instance_uuid)
            ssh = extract_ssh(snapshot)
            if not ssh:
                raise AutoDLError("The AutoDL response contains no usable SSH information")
        finally:
            await client.close()
        experiment = ExperimentStartRequest(
            project_id=request.project_id,
            instance_uuid=request.instance_uuid,
            connection=SSHConnection(**ssh),
            remote_dir=request.remote_dir,
            command=request.command,
            max_iterations=request.max_iterations,
            recover_busy_gpu=request.recover_busy_gpu,
            allow_release_replacement=request.allow_release_replacement,
            confirm_execute=True,
        )
        experiment_id = self.experiments.start(experiment)
        return {
            "project_id": request.project_id,
            "experiment_id": experiment_id,
            "status": "preparing",
            "dashboard_url": self._dashboard(request.project_id),
        }

    def experiment_result(self, experiment_id: str) -> dict[str, Any]:
        experiment = self.db.get_experiment(experiment_id)
        if not experiment:
            raise ValueError("Experiment not found")
        return {
            **experiment,
            "dashboard_url": self._dashboard(str(experiment["project_id"])),
            "analysis_hint": "Ask ChatGPT to analyze metrics, exit_code, and log_tail. Call record_chatgpt_analysis to persist the conclusions.",
        }

    def _project(self, project_id: str):
        project = self.db.get_project(project_id)
        if not project:
            raise ValueError("Research project not found")
        return project

    def _dashboard(self, project_id: str) -> str:
        return f"http://127.0.0.1:{self.settings.port}/?project={project_id}"

    @staticmethod
    def _research_context(project: Any) -> str:
        root = Path(project.workspace)
        parts: list[str] = []
        for path in [root / "RESEARCH.md", *(root / "research").glob("*.md")]:
            if path.is_file() and path.stat().st_size <= 200_000:
                parts.append(f"## {path.name}\n\n{path.read_text(encoding='utf-8', errors='replace')}")
        return "\n\n".join(parts)
