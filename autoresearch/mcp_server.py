from __future__ import annotations

from typing import Any

from mcp.server import MCPServer
from mcp.types import ToolAnnotations

from .chatgpt_bridge import ChatGPTBridge
from .domain import (
    AutoDLCreateRequest,
    AutoDLExperimentRequest,
    ChatGPTAnalysisRequest,
    ChatGPTProjectRequest,
    CodeFile,
    CodeGenerationRequest,
    CodeSaveRequest,
    ResearchNoteRequest,
    SourceSearchRequest,
)


READ_ONLY = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)
LOCAL_WRITE = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=False,
    idempotentHint=False,
    openWorldHint=False,
)
NETWORK_WRITE = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=False,
    idempotentHint=False,
    openWorldHint=True,
)
BILLABLE_WRITE = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=True,
    idempotentHint=False,
    openWorldHint=True,
)
REMOTE_EXECUTION = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=True,
    idempotentHint=False,
    openWorldHint=True,
)


def create_mcp_server(bridge: ChatGPTBridge) -> MCPServer:
    server = MCPServer(
        name="autoresearch-local",
        title="AutoResearch 科研执行器",
        description="为 ChatGPT 保存科研代码、检索资源、调用 AutoDL 并返回实验状态与结果。",
        version="0.2.0",
        instructions=(
            "ChatGPT 负责科研推理和结果分析，AutoResearch 负责持久化与执行。"
            "先 list_research_projects 或 create_research_project，再用项目 ID 调用其他工具。"
            "ChatGPT 已写好代码时调用 save_experiment_code；要委托 AutoResearch/OpenRouter 实现时调用 generate_experiment_code。"
            "调用 AutoDL 计费或远程执行前必须取得用户明确确认并把 confirm 参数设为 true；"
            "不要在对话中索要或回显 AutoDL Token、SSH 密码或 OpenRouter Key。"
        ),
    )

    @server.tool(
        name="list_research_projects",
        title="列出 AutoResearch 项目",
        description="列出 AutoResearch 中已有的科研项目、阶段、进度和网页地址。",
        annotations=READ_ONLY,
        structured_output=True,
    )
    def list_research_projects() -> dict[str, Any]:
        return bridge.list_projects()

    @server.tool(
        name="create_research_project",
        title="创建 ChatGPT 协作研究",
        description=(
            "当用户想让当前 ChatGPT 对话负责科研思考、并让 AutoResearch 保存状态和产物时创建项目。"
            "返回 project_id，后续工具都应复用它。"
        ),
        annotations=LOCAL_WRITE,
        structured_output=True,
    )
    def create_research_project(topic: str, notes: str = "") -> dict[str, Any]:
        return bridge.create_project(ChatGPTProjectRequest(topic=topic, notes=notes))

    @server.tool(
        name="get_research_status",
        title="读取研究状态",
        description=(
            "读取一个研究项目的阶段、事件、来源、文件清单、实验及结果。"
            "用于在 ChatGPT 中继续科研分析或向用户汇报。"
        ),
        annotations=READ_ONLY,
        structured_output=True,
    )
    def get_research_status(project_id: str) -> dict[str, Any]:
        return bridge.status(project_id)

    @server.tool(
        name="read_project_artifact",
        title="读取研究产物",
        description="读取项目文件清单中某个文本产物；不能读取 .env、Git 元数据或其他敏感路径。",
        annotations=READ_ONLY,
        structured_output=True,
    )
    def read_project_artifact(project_id: str, path: str) -> dict[str, Any]:
        return bridge.read_artifact(project_id, path)

    @server.tool(
        name="save_research_note",
        title="保存 ChatGPT 研究笔记",
        description=(
            "把当前 ChatGPT 对话形成的问题定义、假设、实验计划或阶段结论保存到 AutoResearch，"
            "并在网页研究进展中显示。"
        ),
        annotations=LOCAL_WRITE,
        structured_output=True,
    )
    def save_research_note(project_id: str, title: str, markdown: str) -> dict[str, Any]:
        return bridge.save_note(
            project_id, ResearchNoteRequest(title=title, markdown=markdown)
        )

    @server.tool(
        name="search_research_sources",
        title="检索并保存科研资源",
        description=(
            "调用 AutoResearch 检索 arXiv、OpenAlex、Semantic Scholar 和 GitHub，"
            "保存真实来源后交给 ChatGPT 分析。"
        ),
        annotations=NETWORK_WRITE,
        structured_output=True,
    )
    async def search_research_sources(
        project_id: str,
        queries: list[str],
        max_sources: int = 20,
        download_papers: bool = False,
    ) -> dict[str, Any]:
        return await bridge.search_sources(
            project_id,
            SourceSearchRequest(
                queries=queries,
                max_sources=max_sources,
                download_papers=download_papers,
            ),
        )

    @server.tool(
        name="save_experiment_code",
        title="保存 ChatGPT 生成的实验代码",
        description=(
            "当 ChatGPT 已经写好代码时，将完整文件安全保存到项目 generated 目录。"
            "files 中每项包含相对 path 和完整 content；可同时保存 experiment_manifest。"
        ),
        annotations=LOCAL_WRITE,
        structured_output=True,
    )
    def save_experiment_code(
        project_id: str,
        files: list[CodeFile],
        summary: str = "",
        experiment_manifest: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return bridge.save_code(
            project_id,
            CodeSaveRequest(
                files=files,
                summary=summary,
                experiment_manifest=experiment_manifest or {},
            ),
        )

    @server.tool(
        name="generate_experiment_code",
        title="委托 AutoResearch 大模型生成代码",
        description=(
            "只在用户希望使用 AutoResearch 配置的 OpenRouter 大模型实现实验代码时调用。"
            "ChatGPT 保留科研推理责任，AutoResearch 模型根据已保存上下文生成并落盘代码。"
        ),
        annotations=NETWORK_WRITE,
        structured_output=True,
    )
    async def generate_experiment_code(
        project_id: str, instructions: str, model: str | None = None
    ) -> dict[str, Any]:
        return await bridge.generate_code(
            project_id, CodeGenerationRequest(instructions=instructions, model=model)
        )

    @server.tool(
        name="create_autodl_instance",
        title="创建 AutoDL 计费实例",
        description=(
            "按设置中的 GPU 优先级创建 AutoDL Pro 实例。此操作可能立即产生费用；"
            "只有用户在当前对话明确同意费用后才能把 confirm_billable 设为 true。"
        ),
        annotations=BILLABLE_WRITE,
        structured_output=True,
    )
    async def create_autodl_instance(
        confirm_billable: bool,
        project_id: str | None = None,
        image_uuid: str | None = None,
        instance_name: str = "AutoResearch",
        gpu_amount: int = 1,
        disk_gb: int = 0,
        data_centers: list[str] | None = None,
    ) -> dict[str, Any]:
        return await bridge.create_autodl_instance(
            AutoDLCreateRequest(
                image_uuid=image_uuid,
                instance_name=instance_name,
                gpu_amount=gpu_amount,
                disk_gb=disk_gb,
                data_centers=data_centers or [],
                confirm_billable=confirm_billable,
            ),
            project_id=project_id,
        )

    @server.tool(
        name="get_autodl_instance_status",
        title="读取 AutoDL 实例状态",
        description=(
            "检查 AutoDL 实例是否已经运行及 SSH 是否就绪，不返回主机密码或 Token。"
            "创建实例后可重复读取，ssh_ready 为 true 时再启动实验。"
        ),
        annotations=ToolAnnotations(
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
        structured_output=True,
    )
    async def get_autodl_instance_status(instance_uuid: str) -> dict[str, Any]:
        return await bridge.autodl_instance_status(instance_uuid)

    @server.tool(
        name="start_autodl_experiment",
        title="在 AutoDL 上启动实验",
        description=(
            "从 AutoDL API 内部取得 SSH 连接并上传 generated 代码执行命令；密钥和密码不会返回 ChatGPT。"
            "只有用户已审阅代码与命令并在当前对话明确同意执行后，才能把 confirm_execute 设为 true。"
        ),
        annotations=REMOTE_EXECUTION,
        structured_output=True,
    )
    async def start_autodl_experiment(
        project_id: str,
        instance_uuid: str,
        confirm_execute: bool,
        command: str = "python experiment.py",
        remote_dir: str = "/root/autodl-tmp/AutoResearch",
        max_iterations: int = 1,
        recover_busy_gpu: bool = True,
        allow_release_replacement: bool = False,
    ) -> dict[str, Any]:
        return await bridge.start_autodl_experiment(
            AutoDLExperimentRequest(
                project_id=project_id,
                instance_uuid=instance_uuid,
                confirm_execute=confirm_execute,
                command=command,
                remote_dir=remote_dir,
                max_iterations=max_iterations,
                recover_busy_gpu=recover_busy_gpu,
                allow_release_replacement=allow_release_replacement,
            )
        )

    @server.tool(
        name="get_experiment_result",
        title="读取 AutoDL 实验结果",
        description=(
            "读取实验状态、退出码、指标和日志末尾。实验进行中可重复调用；"
            "完成后由 ChatGPT 分析，而不是再次调用 AutoResearch 模型。"
        ),
        annotations=READ_ONLY,
        structured_output=True,
    )
    def get_experiment_result(experiment_id: str) -> dict[str, Any]:
        return bridge.experiment_result(experiment_id)

    @server.tool(
        name="record_chatgpt_analysis",
        title="保存 ChatGPT 实验分析",
        description=(
            "将 ChatGPT 对指标、日志、失败原因和下一步建议的分析保存为项目报告，"
            "并在 AutoResearch 网页显示分析已完成。"
        ),
        annotations=LOCAL_WRITE,
        structured_output=True,
    )
    def record_chatgpt_analysis(
        project_id: str,
        analysis: str,
        experiment_id: str | None = None,
        recommendations: list[str] | None = None,
    ) -> dict[str, Any]:
        return bridge.record_analysis(
            project_id,
            ChatGPTAnalysisRequest(
                analysis=analysis,
                experiment_id=experiment_id,
                recommendations=recommendations or [],
            ),
        )

    return server
