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
            brief += f"## 初始约束\n\n{request.notes}\n"
        else:
            brief += "由 ChatGPT 对话负责科研推理；研究笔记、代码和实验由 AutoResearch 持久化。\n"
        store.write_text("research/chatgpt-brief.md", brief)
        self.db.update_project(
            project.id,
            status="running",
            phase="chatgpt_thinking",
            progress=10,
            summary="ChatGPT 正在进行科研思考；AutoResearch 等待保存笔记、代码或启动实验。",
        )
        self.db.add_event(
            project.id,
            "chatgpt_thinking",
            "已建立 ChatGPT 协作研究，等待对话中的下一步操作",
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
            summary=f"已保存 ChatGPT 研究笔记：{request.title}",
        )
        self.db.add_event(
            project_id,
            "chatgpt_thinking",
            f"ChatGPT 已保存研究笔记《{request.title}》",
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
        summary = request.summary.strip() or f"ChatGPT 已保存 {len(written)} 个代码文件"
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
            f"ChatGPT 已将 {len(written)} 个代码文件保存到实验工作区",
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
            raise LLMError("尚未配置 OpenRouter API Key，无法调用 AutoResearch 大模型生成代码")
        self.db.update_project(
            project_id, status="running", phase="generating", progress=74, error=""
        )
        self.db.add_event(
            project_id,
            "generating",
            "ChatGPT 已把代码实现任务交给 AutoResearch 大模型",
            details={"model": request.model or self.settings.openrouter_model},
        )
        store = ArtifactStore(Path(project.workspace))
        context = self._research_context(project)
        sources = self.db.list_sources(project_id)
        source_digest = "\n\n".join(
            f"[S{index}] {source.title}\n{source.url}\n{source.abstract[:900]}"
            for index, source in enumerate(sources, 1)
        )[:24_000]
        prompt = f"""ChatGPT 已经完成科研思考。请只负责把方案实现成最小、可运行、可复现的 Python 实验项目。
研究题目：{project.topic}
ChatGPT 的实现要求：{request.instructions}
已保存研究上下文：
{context[:36_000]}

已保存来源：
{source_digest or '暂无来源'}

只返回严格 JSON：
{{
  "summary": "实现摘要",
  "files": [{{"path": "相对路径", "content": "完整文件内容"}}],
  "experiment": {{
    "setup_commands": ["安全、非交互的安装命令"],
    "dataset_commands": [],
    "run_command": "python experiment.py",
    "metrics_file": "results/metrics.json"
  }}
}}
必须包含 README.md、requirements.txt、可运行入口、测试或校验脚本和 .gitignore。不得写入密钥、绝对路径或伪造实验结果。"""
        try:
            async with OpenRouterClient(self.settings) as llm:
                bundle = await llm.chat_json(
                    [
                        {
                            "role": "system",
                            "content": "你是 AutoResearch 实验工程师。保留 ChatGPT 的研究决策，只实现代码。输出完整文件而非补丁。",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    model=request.model,
                    max_tokens=12_000,
                )
            files = bundle.get("files")
            if not isinstance(files, list) or not files:
                raise LLMError("AutoResearch 大模型没有返回代码文件")
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
                f"AutoResearch 代码生成失败：{exc}",
                level="error",
            )
            raise
        summary = str(bundle.get("summary") or f"AutoResearch 已生成 {len(written)} 个实验文件")[:2000]
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
            f"AutoResearch 大模型已生成 {len(written)} 个实验文件",
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
            raise ValueError("至少需要一个非空检索式")
        self.db.update_project(
            project_id, status="running", phase="searching", progress=30, error=""
        )
        self.db.add_event(project_id, "searching", "ChatGPT 已请求 AutoResearch 检索科研资源")
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
            summary=f"已保存 {len(sources)} 条来源，等待 ChatGPT 继续分析。",
        )
        self.db.add_event(
            project_id,
            "chatgpt_thinking",
            f"已保存 {len(sources)} 条来源并交回 ChatGPT 分析",
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
        body = "# ChatGPT 实验分析\n\n"
        if request.experiment_id:
            body += f"实验 ID：`{request.experiment_id}`\n\n"
        body += request.analysis.rstrip() + "\n"
        if request.recommendations:
            body += "\n## 下一步建议\n\n" + "\n".join(
                f"- {item}" for item in request.recommendations
            ) + "\n"
        ArtifactStore(Path(project.workspace)).write_text(relative, body)
        self.db.update_project(
            project_id,
            status="ready",
            phase="chatgpt_analysis",
            progress=100,
            summary="实验结果已由 ChatGPT 分析并保存。",
            error="",
        )
        self.db.add_event(
            project_id,
            "chatgpt_analysis",
            "ChatGPT 已分析实验结果并保存结论",
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
            "reasoning_owner": "ChatGPT 当前对话" if collaborative else "AutoResearch/OpenRouter",
            "execution_model": project.model or self.settings.openrouter_model,
            "sources": [item.model_dump() for item in self.db.list_sources(project_id)],
            "experiments": experiments,
            "recent_events": self.db.list_events(project_id)[-30:],
            "artifacts": store.list_files(),
            "dashboard_url": self._dashboard(project_id),
        }

    async def create_autodl_instance(
        self, request: AutoDLCreateRequest, project_id: str | None = None
    ) -> dict[str, Any]:
        if not request.confirm_billable:
            raise ValueError("创建按量计费 AutoDL 实例前必须取得用户明确确认")
        if project_id:
            self._project(project_id)
        client = AutoDLClient(self.settings)
        try:
            choice = await client.create_preferred(request)
            result = {
                "instance_uuid": choice.instance_uuid,
                "gpu_spec": choice.gpu_spec,
                "status": "creating",
                "message": "实例正在创建。计费与生命周期仍由 AutoDL 控制台管理。",
            }
            if project_id:
                self.db.add_event(
                    project_id,
                    "autodl",
                    f"已创建 AutoDL 计费实例 {choice.instance_uuid}（{choice.gpu_spec}）",
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
                "message": "ssh_ready 为 true 后可以启动实验；连接凭据不会返回 ChatGPT。",
            }
        finally:
            await client.close()

    async def start_autodl_experiment(
        self, request: AutoDLExperimentRequest
    ) -> dict[str, Any]:
        if not request.confirm_execute:
            raise ValueError("上传并执行代码前必须取得用户明确确认")
        self._project(request.project_id)
        client = AutoDLClient(self.settings)
        try:
            status = await client.status(request.instance_uuid)
            if status != "running":
                raise AutoDLError(f"AutoDL 实例尚未运行（当前状态：{status}）")
            snapshot = await client.snapshot(request.instance_uuid)
            ssh = extract_ssh(snapshot)
            if not ssh:
                raise AutoDLError("AutoDL 响应中没有可用 SSH 信息")
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
            raise ValueError("实验不存在")
        return {
            **experiment,
            "dashboard_url": self._dashboard(str(experiment["project_id"])),
            "analysis_hint": "请由 ChatGPT 基于 metrics、exit_code 和 log_tail 分析；如要持久化结论，请调用 record_chatgpt_analysis。",
        }

    def _project(self, project_id: str):
        project = self.db.get_project(project_id)
        if not project:
            raise ValueError("研究项目不存在")
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
