from __future__ import annotations

from typing import Any

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from .chatgpt_bridge import ChatGPTBridge
from .domain import (
    AutoDLCreateRequest,
    AutoDLExperimentRequest,
    ChatGPTAnalysisRequest,
    ChatGPTProjectRequest,
    ColabFallbackTarget,
    ColabPrepareRequest,
    CodeFile,
    CodeGenerationRequest,
    CodeSaveRequest,
    ResearchNoteRequest,
    SourceSearchRequest,
)
from .services.autodl import AutoDLError


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
        title="AutoResearch Experiment Executor",
        description="Persists ChatGPT-led research plans, code and decisions, executes experiments on free Colab or authorized AutoDL, and returns current-run evidence for ChatGPT analysis.",
        version="0.3.1",
        instructions=(
            "ChatGPT owns research reasoning and result analysis; AutoResearch handles persistence and execution. "
            "Call list_research_projects or create_research_project first, then reuse the project ID with other tools. "
            "Call save_experiment_code when ChatGPT has written the code; call generate_experiment_code to delegate implementation to AutoResearch/OpenRouter. "
            "Before using AutoDL, call check_autodl_readiness; it never creates billable resources. "
            "For free Colab GPUs, call prepare_colab_experiment then start_colab_experiment with user authorization. Google login or Drive consent can require user action; a retry deadline is not GPU allocation. "
            "Use get_colab_status and get_colab_resources to distinguish live observations from old results. After manual consent call resume_colab_experiment; bind_colab_notebook saves the public Drive recovery link. "
            "When get_colab_status returns a pending_notice, call claim_colab_notice and relay its message only if the claim returns it. The dashboard is read-only. "
            "Use authorize_colab_backup only after explicit approval of the reviewed commands, the exact GPU target, its data access and any costs of an existing AutoDL instance. "
            "Use get_colab_experiment_result for current-run Colab metrics and returned log evidence. Missing remote logs or metrics remain unknown, and acceptance does not count as full-data completion. "
            "Save an explicit hypothesis/dataset/experiment coverage plan using save_research_plan, inspect get_research_workflow, then record_research_decision against the latest evidence after ChatGPT analysis. AutoResearch never substitutes its own model reasoning in this mode. "
            "Then call check_experiment_readiness to inspect entry points, data, and metric declarations; a passing static check does not validate scientific results. "
            "Obtain explicit user confirmation and set the relevant confirm parameter to true before billable AutoDL actions or remote execution. "
            "Never ask for or echo an AutoDL token, SSH password, or OpenRouter key in conversation."
        ),
    )

    @server.tool(
        name="list_research_projects",
        title="List AutoResearch Projects",
        description="Lists existing AutoResearch projects, phases, progress, and dashboard URLs.",
        annotations=READ_ONLY,
        structured_output=True,
    )
    def list_research_projects() -> dict[str, Any]:
        return bridge.list_projects()

    @server.tool(
        name="create_research_project",
        title="Create a ChatGPT Collaborative Study",
        description=(
            "Creates a project when the current ChatGPT conversation owns research reasoning and AutoResearch stores state and artifacts. "
            "Returns a project_id that subsequent tools should reuse."
        ),
        annotations=LOCAL_WRITE,
        structured_output=True,
    )
    def create_research_project(topic: str, notes: str = "") -> dict[str, Any]:
        return bridge.create_project(ChatGPTProjectRequest(topic=topic, notes=notes))

    @server.tool(
        name="get_research_status",
        title="Get Research Status",
        description=(
            "Returns a research project's phase, events, sources, artifact inventory, experiments, and results. "
            "Use it to continue analysis in ChatGPT or report progress to the user."
        ),
        annotations=READ_ONLY,
        structured_output=True,
    )
    def get_research_status(project_id: str) -> dict[str, Any]:
        return bridge.status(project_id)

    @server.tool(
        name="check_experiment_readiness",
        title="Check Experiment Inputs and Manifest",
        description=(
            "Read-only inspection of the experiment manifest, code entry point, required files, and input paths, including blockers and remote unknowns. "
            "Does not execute code, inspect remote content, or create instances; passing static checks does not validate scientific results."
        ),
        annotations=READ_ONLY,
        structured_output=True,
    )
    def check_experiment_readiness(project_id: str) -> dict[str, Any]:
        return bridge.experiment_readiness(project_id)

    @server.tool(
        name="read_project_artifact",
        title="Read a Research Artifact",
        description="Reads a text artifact from the project inventory. It cannot read .env, Git metadata, or other sensitive paths.",
        annotations=READ_ONLY,
        structured_output=True,
    )
    def read_project_artifact(project_id: str, path: str) -> dict[str, Any]:
        return bridge.read_artifact(project_id, path)

    @server.tool(
        name="save_research_note",
        title="Save a ChatGPT Research Note",
        description=(
            "Saves problem definitions, hypotheses, experiment plans, or interim conclusions from the current ChatGPT conversation in AutoResearch "
            "and displays them in the research dashboard."
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
        title="Search and Save Research Sources",
        description=(
            "Uses AutoResearch to search arXiv, OpenAlex, Semantic Scholar, and GitHub, "
            "then saves real sources for ChatGPT to analyze."
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
        title="Save Experiment Code Generated by ChatGPT",
        description=(
            "Safely saves complete files written by ChatGPT under the project's generated directory. "
            "Each files item contains a relative path and complete content; experiment_manifest can be saved at the same time."
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
        title="Delegate Code Generation to the AutoResearch Model",
        description=(
            "Use only when the user wants the OpenRouter model configured in AutoResearch to implement experiment code. "
            "ChatGPT retains responsibility for research reasoning while the AutoResearch model generates and saves code from the stored context."
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
        name="check_autodl_readiness",
        title="Check AutoDL Experiment Readiness",
        description=(
            "Checks token, image, and GPU configuration; verify_api=true also performs read-only API authentication. "
            "Does not create instances, incur compute charges, or return secrets or instance details. "
            "configured means local configuration is complete; verified means read-only instance API authentication succeeded and configuration is complete. "
            "After successful online verification it queries the wallet; balance_verified separately indicates a successful balance lookup. Inventory, image availability, and coupon eligibility remain unknown."
        ),
        annotations=ToolAnnotations(
            readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True,
        ),
        structured_output=True,
    )
    async def check_autodl_readiness(
        verify_api: bool = False, image_uuid: str | None = None
    ) -> dict[str, Any]:
        return await bridge.autodl_readiness(verify_api=verify_api, image_uuid=image_uuid)

    @server.tool(
        name="create_autodl_instance",
        title="Create a Billable AutoDL Instance",
        description=(
            "Creates an AutoDL Pro instance according to the configured GPU priority. This action may incur charges immediately. "
            "Set confirm_billable=true only after the user explicitly accepts the cost in the current conversation."
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
        try:
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
        except (AutoDLError, ValueError) as exc:
            # Expected provisioning/configuration failures must reach ChatGPT as
            # actionable tool errors. Unhandled exceptions are intentionally
            # redacted by the MCP SDK as a generic "Error executing tool".
            raise ToolError(str(exc)) from exc

    @server.tool(
        name="get_autodl_instance_status",
        title="Get AutoDL Instance Status",
        description=(
            "Checks whether an AutoDL instance is running and SSH is ready without returning the host password or token. "
            "Poll after creation and start the experiment only when ssh_ready is true."
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
        try:
            return await bridge.autodl_instance_status(instance_uuid)
        except (AutoDLError, ValueError) as exc:
            raise ToolError(str(exc)) from exc

    @server.tool(
        name="start_autodl_experiment",
        title="Start an Experiment on AutoDL",
        description=(
            "Obtains SSH details internally from the AutoDL API, uploads generated code, and runs the command; secrets and passwords are never returned to ChatGPT. "
            "Set confirm_execute=true only after the user has reviewed the code and command and explicitly approved execution in the current conversation."
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
        try:
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
        except (AutoDLError, ValueError) as exc:
            raise ToolError(str(exc)) from exc

    @server.tool(
        name="get_experiment_result",
        title="Get AutoDL Experiment Results",
        description=(
            "Returns experiment status, exit code, metrics, and the log tail. It can be called repeatedly while an experiment runs. "
            "After completion, ChatGPT analyzes the results instead of invoking the AutoResearch model again."
        ),
        annotations=READ_ONLY,
        structured_output=True,
    )
    def get_experiment_result(experiment_id: str) -> dict[str, Any]:
        return bridge.experiment_result(experiment_id)

    @server.tool(
        name="record_chatgpt_analysis",
        title="Save ChatGPT Experiment Analysis",
        description=(
            "Saves ChatGPT's analysis of metrics, logs, failure causes, and next steps as a project report "
            "and marks the analysis as complete in the AutoResearch dashboard."
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

    @server.tool(
        name="get_colab_status", title="Get Colab GPU and Retry Status",
        description="Reads a project's Colab notebook job, observed runtime state, retry deadline, and checkpoint directory. A queued job does not mean a GPU has been allocated.",
        annotations=READ_ONLY, structured_output=True,
    )
    def get_colab_status(project_id: str) -> dict[str, Any]:
        return bridge.colab_status(project_id)

    @server.tool(
        name="get_colab_resources", title="Get Current Colab Resource Observer State",
        description="Reads the resource Agent's actual process and heartbeat state, last GPU observation, freshness, and next check. Historical available observations and a live heartbeat alone do not prove a GPU is available.",
        annotations=READ_ONLY, structured_output=True,
    )
    def get_colab_resources(project_id: str) -> dict[str, Any]:
        return bridge.colab_resources(project_id)

    @server.tool(
        name="open_colab_login", title="Open the Dedicated Google Login Browser",
        description="Opens the dedicated ordinary Chrome/Edge window for Google sign-in, or points to the already active worker window. Close an ordinary login window after signing in; the scheduler will recheck sign-in and continue any authorized job. Credentials and verification stay in the browser.",
        annotations=NETWORK_WRITE, structured_output=True,
    )
    def open_colab_login(project_id: str, browser: str = "auto") -> dict[str, Any]:
        return bridge.open_colab_login(project_id, browser=browser)

    @server.tool(
        name="get_colab_experiment_result", title="Read Current Colab Experiment Evidence",
        description="Returns validated metrics from the prepared code and currently observed Colab runtime, including per-seed comparisons and the confirmation outcome when declared. Includes completion scope, a returned log tail or an explicit unknown, and next actions for ChatGPT. Acceptance is not full-data completion.",
        annotations=READ_ONLY, structured_output=True,
    )
    def get_colab_experiment_result(project_id: str) -> dict[str, Any]:
        return bridge.colab_result(project_id)

    @server.tool(
        name="prepare_colab_experiment", title="Prepare a Free Colab GPU Experiment",
        description="Exports a project notebook after validating its explicit full-state checkpoint and resume contract. Does not allocate a GPU or upload files. Uses random 5–10 minute GPU retries.",
        annotations=LOCAL_WRITE, structured_output=True,
    )
    def prepare_colab_experiment(project_id: str, drive_root: str = "AutoResearch",
                                 notebook_url: str = "") -> dict[str, Any]:
        return bridge.prepare_colab(project_id, ColabPrepareRequest(
            drive_root=drive_root, notebook_url=notebook_url))

    @server.tool(
        name="start_colab_experiment", title="Start the Colab Notebook Browser",
        description="Authorizes the prepared free-GPU job and starts its browser worker and resource Agent. The worker handles sign-in state and checks real GPU allocation through the Colab UI. When ordinary availability is denied, its durable queue retries in 5–10 minutes; explicit account limits take precedence. For an authorization pause follow next_action.",
        annotations=REMOTE_EXECUTION, structured_output=True,
    )
    def start_colab_experiment(project_id: str, confirm_execute: bool) -> dict[str, Any]:
        return bridge.start_colab(project_id, confirm_execute=confirm_execute)

    @server.tool(
        name="bind_colab_notebook", title="Save a Verified Colab Drive Recovery Link",
        description="Binds the public https://colab.research.google.com/drive/... notebook URL to the prepared source hash returned by get_colab_status. Verify that the Drive copy contains this prepared notebook first. Does not run cells, allocate compute, or bypass a retry deadline. Never send connection tokens.",
        annotations=LOCAL_WRITE, structured_output=True,
    )
    def bind_colab_notebook(project_id: str, notebook_url: str, bundle_sha256: str) -> dict[str, Any]:
        return bridge.bind_colab_notebook(project_id, notebook_url, bundle_sha256)

    @server.tool(
        name="resume_colab_experiment", title="Resume Colab After Manual Login or Consent",
        description="Resumes only a recorded login or Drive authorization pause using the user's execution authorization. Set confirm_execute=true only when that authorization exists. Preserves GPU quota cooldowns and reuses an existing worker; never ask for passwords or codes.",
        annotations=REMOTE_EXECUTION, structured_output=True,
    )
    def resume_colab_experiment(project_id: str, confirm_execute: bool) -> dict[str, Any]:
        return bridge.resume_colab(project_id, confirm_execute=confirm_execute)

    @server.tool(
        name="authorize_colab_backup", title="Approve an Existing Backup GPU",
        description="Bind a local GPU or already running AutoDL instance to the prepared source. Review the code/commands, approved paths and existing-instance costs with the user before setting confirm_execute. Dispatch occurs automatically after one hour of waiting if the host passes preflight. Never provisions or powers on an instance.",
        annotations=REMOTE_EXECUTION, structured_output=True,
    )
    def authorize_colab_backup(project_id: str, targets: list[ColabFallbackTarget], confirm_execute: bool) -> dict[str, Any]:
        return bridge.authorize_colab_backup(project_id, targets, confirm_execute=confirm_execute)

    @server.tool(
        name="claim_colab_notice", title="Claim a One-Time Colab Notice",
        description="Returns a pending authorization notice at most once. Use the ID returned by get_colab_status, then relay the claimed message to the user in ChatGPT.",
        annotations=LOCAL_WRITE, structured_output=True,
    )
    def claim_colab_notice(project_id: str, notice_id: str) -> dict[str, Any]:
        return bridge.claim_colab_notice(project_id, notice_id)

    @server.tool(
        name="cancel_colab_retries", title="Stop Colab GPU Retries",
        description="Stops this project's local GPU retry queue. Stop an already running notebook in Colab itself.",
        annotations=LOCAL_WRITE, structured_output=True,
    )
    def cancel_colab_retries(project_id: str) -> dict[str, Any]:
        return bridge.cancel_colab(project_id)

    @server.tool(
        name="save_research_plan", title="Save the ChatGPT Research Coverage Plan",
        description="Persists ChatGPT's explicit hypotheses, required datasets and experiment matrix, with implementation paths and success criteria. Full, acceptance and evaluation scopes stay distinct. Returns a versioned plan and coverage evidence; does not generate code, execute experiments or invoke another model.",
        annotations=LOCAL_WRITE, structured_output=True,
    )
    def save_research_plan(project_id: str, plan: dict[str, Any]) -> dict[str, Any]:
        return bridge.save_research_plan(project_id, plan)

    @server.tool(
        name="get_research_workflow", title="Inspect ChatGPT Research Coverage and Evidence",
        description="Returns the saved research plan, code and evidence identities, hypothesis/dataset/experiment coverage and current blockers. Planned work and acceptance results cannot count as completed full-data research. This is a read-only inspection, not scientific reasoning or execution.",
        annotations=READ_ONLY, structured_output=True,
    )
    def get_research_workflow(project_id: str) -> dict[str, Any]:
        return bridge.research_workflow(project_id)

    @server.tool(
        name="record_research_decision", title="Save ChatGPT's Evidence-Bound Research Decision",
        description="Saves ChatGPT's advance/revise/stop/accept decision and analysis bound to the latest plan_id, plan_sha256, code_sha256 and evidence_id returned by get_research_workflow. Advance/accept require current full-data evidence. Does not invoke a model, start another experiment, or incur costs.",
        annotations=LOCAL_WRITE, structured_output=True,
    )
    def record_research_decision(project_id: str, decision: dict[str, Any]) -> dict[str, Any]:
        return bridge.record_research_decision(project_id, decision)

    return server
