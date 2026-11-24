from __future__ import annotations

from pathlib import Path
import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .config import Settings
from .db import Database
from .domain import (
    AutoDLCreateRequest,
    AutoDLExperimentRequest,
    ChatGPTAnalysisRequest,
    ChatGPTProjectRequest,
    ColabFallbackTarget,
    ColabNotebookBindingRequest,
    ColabPrepareRequest,
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
from .services.colab import ColabService


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
        self.colab = ColabService(db, settings)

    def update_settings(self, settings: Settings) -> None:
        self.settings = settings
        self.colab.update_settings(settings)

    def colab_status(self, project_id: str) -> dict[str, Any]:
        job = self.colab.status(project_id)
        public = self._public_colab_job(job)
        public["resource_agent"] = self.colab_resources(project_id)
        public["next_action"] = self._colab_next_action(job)
        return public

    def colab_resources(self, project_id: str) -> dict[str, Any]:
        self._project(project_id)
        from .services.colab_resource_monitor import resource_monitor_status

        return resource_monitor_status(self.settings.root, project_id)

    def authorize_colab_backup(self, project_id: str, targets: list[ColabFallbackTarget], *,
                               confirm_execute: bool) -> dict[str, Any]:
        if not confirm_execute:
            raise ValueError("Confirm the exact source, commands, target, data access and any existing-instance costs")
        self.colab.authorize_backup(project_id, targets)
        return self.colab_status(project_id)

    def claim_colab_notice(self, project_id: str, notice_id: str) -> dict[str, Any]:
        return {"project_id": project_id, "notice": self.colab.claim_notice(project_id, notice_id)}

    def save_research_plan(self, project_id: str, plan: dict[str, Any]) -> dict[str, Any]:
        from .services.research_workflow import ResearchWorkflowService

        self._project(project_id)
        return ResearchWorkflowService(self.db, self.settings).save_plan(project_id, plan)

    def research_workflow(self, project_id: str) -> dict[str, Any]:
        from .services.research_workflow import ResearchWorkflowService

        self._project(project_id)
        return ResearchWorkflowService(self.db, self.settings).get_status(project_id)

    def record_research_decision(self, project_id: str, decision: dict[str, Any]) -> dict[str, Any]:
        from .services.research_workflow import ResearchWorkflowService

        self._project(project_id)
        return ResearchWorkflowService(self.db, self.settings).record_decision(project_id, decision)

    def prepare_colab(self, project_id: str, request: ColabPrepareRequest) -> dict[str, Any]:
        self.colab.prepare(project_id, request)
        return self.colab_status(project_id)

    def bind_colab_notebook(self, project_id: str, notebook_url: str, bundle_sha256: str) -> dict[str, Any]:
        # Proxy connection fragments belong only to the local browser session.
        # The saved recovery link must never retain them in a tool response.
        if urlsplit(notebook_url).username or urlsplit(notebook_url).password:
            raise ValueError("Notebook links must not contain credentials")
        url = self._public_url(notebook_url)
        self.colab.bind_notebook(project_id, ColabNotebookBindingRequest(
            notebook_url=url, bundle_sha256=bundle_sha256))
        return self.colab_status(project_id)

    def resume_colab(self, project_id: str, *, confirm_execute: bool = False) -> dict[str, Any]:
        if not confirm_execute:
            raise ValueError("User authorization is required before resuming the project in Colab")
        job = self.colab.status(project_id)
        if job.get("status") != "auth_required":
            raise ValueError("Only a Colab login or authorization pause can be resumed; retry cooldowns remain in force")
        return self.start_colab(project_id, confirm_execute=True, resume_auth=True)

    def open_colab_login(self, project_id: str, *, browser: str = "auto") -> dict[str, Any]:
        self._project(project_id)
        from .colab_browser_cli import launch_colab_login_browser, worker_is_running

        self.colab.status(project_id)
        worker = worker_is_running(self.settings.root, project_id)
        if worker is not None:
            return {"project_id": project_id, "status": "use_existing_browser", "pid": worker,
                    "action_required": True,
                    "next_action": {"action": "complete_browser_authorization",
                                    "message": "Use the already opened dedicated Colab browser for Google login, verification or Drive consent. The existing worker will check GPU availability; do not send passwords or verification codes to ChatGPT."}}
        pid = launch_colab_login_browser(self.settings.root, browser=browser)
        self.colab.ordinary_login_opened(project_id, pid)
        return {"project_id": project_id, "status": "login_browser_opened", "pid": pid,
                "action_required": True,
                "next_action": {"action": "complete_login_then_close_window",
                                "message": "Sign in to Google in the dedicated window. Close that window when finished so AutoResearch can recheck the session and continue the authorized job automatically."}}

    def start_colab(self, project_id: str, *, confirm_execute: bool = False,
                    resume_auth: bool = False, browser: str | None = None) -> dict[str, Any]:
        if not confirm_execute:
            raise ValueError("User authorization is required before executing the project in Colab")
        from .colab_browser_cli import ColabProfileBusy, launch_colab_worker, login_browser_is_running

        job = self.colab.status(project_id)
        if not job.get("prepared"):
            raise ValueError("Prepare the Colab experiment notebook first")
        if job.get("status") in {"completed", "failed", "cancelled"}:
            raise ValueError("Prepare the Colab notebook again before starting another run")
        if (job.get("backup") or {}).get("status") in {"launching", "checking", "running", "unknown"}:
            return {"project_id": project_id, "status": "observing_backup",
                    "next_action": self._colab_next_action(job)}
        if self.colab.source_matches_prepared_bundle(project_id) is not True:
            raise ValueError("Prepared Colab source is missing or changed; prepare the notebook again before execution")
        job = self.colab.enable_autorun(project_id, browser or job.get("browser", "auto"))
        resource_agent = self._ensure_colab_resource_agent(project_id, job.get("browser", "auto"))
        login_pid = login_browser_is_running(self.settings.root)
        if login_pid is not None:
            if not job.get("ordinary_login"):
                job = self.colab.ordinary_login_opened(project_id, login_pid)
            return {"project_id": project_id, "status": "auth_required", "autorun_enabled": True,
                    "action_required": True, "resource_agent": resource_agent,
                    "next_action": self._colab_next_action(job)}
        if job.get("status") == "auth_required":
            if job.get("ordinary_login"):
                job = self.colab.complete_ordinary_login(project_id)
            elif resume_auth:
                job = self.colab.resume(project_id)
            else:
                return {"project_id": project_id, "status": "auth_required", "autorun_enabled": True,
                        "action_required": True, "resource_agent": resource_agent,
                        "next_action": self._colab_next_action(job)}
        if job.get("worker_running"):
            return {"project_id": project_id, "status": "browser_launched",
                    "pid": job.get("worker_pid"), "autorun_enabled": True,
                    "resource_agent": resource_agent,
                    "next_action": self._colab_next_action(job)}
        if not job.get("can_attempt") and job.get("status") not in {"running", "gpu_available"} and not job.get("worker_running"):
            return {"project_id": project_id, "status": "retry_scheduled",
                    "next_attempt_at": job.get("next_attempt_at"), "autorun_enabled": True,
                    "resource_agent": resource_agent, "next_action": self._colab_next_action(job)}
        try:
            pid = launch_colab_worker(self.settings.root, project_id,
                                      api_url=f"http://127.0.0.1:{self.settings.port}",
                                      browser=job.get("browser", "auto"))
        except ColabProfileBusy:
            return {"project_id": project_id, "status": "retry_scheduled", "autorun_enabled": True,
                    "next_attempt_at": job.get("next_attempt_at"), "resource_agent": resource_agent,
                    "next_action": {"action": "wait_for_browser", "message": "The local scheduler will dispatch this job when the shared Colab browser becomes free."}}
        self.colab.worker_launched(project_id, pid)
        return {"project_id": project_id, "status": "browser_launched", "pid": pid,
                "resource_agent": resource_agent, "autorun_enabled": True,
                "message": "Complete Google login and notebook Drive authorization in the browser if prompted."}

    def _ensure_colab_resource_agent(self, project_id: str, browser: str) -> dict[str, Any]:
        from .services.colab_resource_monitor import launch_resource_monitor

        try:
            pid = launch_resource_monitor(self.settings.root, project_id,
                                          api_url=f"http://127.0.0.1:{self.settings.port}", browser=browser)
        except (OSError, ValueError, RuntimeError):
            # A launch failure is explicit, never an assertion about GPU supply.
            state = self.colab_resources(project_id)
            return {**state, "launch_status": "pending_scheduler_retry"}
        state = self.colab_resources(project_id)
        return {**state, "pid": pid, "launch_status": "running" if state.get("agent_running") else "starting"}

    def cancel_colab(self, project_id: str) -> dict[str, Any]:
        self.colab.cancel(project_id)
        return self.colab_status(project_id)

    def colab_result(self, project_id: str) -> dict[str, Any]:
        """Return evidence from this prepared source and currently observed VM.

        A remote path is a locator, never proof that its contents reached this
        process. We do not open provider paths on the Windows filesystem.
        """
        from .services.colab_data import validate_completion_details

        job = self.colab.status(project_id)
        observation = job.get("last_observation") or {}
        details = observation.get("details") or {}
        platform = details.get("compute_provider") or "colab"
        runtime_id = job.get("runtime_id")
        current_runtime = bool(runtime_id and details.get("runtime_id") == runtime_id)
        source_current = self.colab.source_matches_prepared_bundle(project_id)
        issues = []
        completion = None
        if job.get("status") == "completed":
            if not current_runtime or observation.get("status") != "completed" or details.get("returncode") != 0:
                issues.append("No successful completion observation from the currently tracked run")
            elif source_current is not True:
                issues.append("The current code does not match the source prepared for this Colab run")
            else:
                try:
                    validated = validate_completion_details(details, job.get("success_criteria"))
                    if job.get("completion") != validated:
                        raise ValueError("Saved completion differs from the current runtime evidence")
                    completion = validated
                except (TypeError, ValueError) as exc:
                    issues.append(self._public_text(str(exc)))
        observed_log = details.get("log_tail") if current_runtime else None
        log_tail = self._public_text(observed_log[-8000:]) if isinstance(observed_log, str) else None
        kind = completion.get("completion_kind") if completion else None
        return {
            "project_id": project_id,
            "experiment_id": f"colab:{project_id}:{runtime_id}" if runtime_id else None,
            "platform": platform,
            "status": job.get("status"),
            "result_status": "validated" if completion else "pending_evidence" if job.get("status") == "completed" else "failed" if job.get("status") == "failed" else "pending",
            "execution_completed": completion is not None,
            "full_experiment_complete": kind == "campaign_complete",
            "acceptance_complete": kind == "acceptance_passed",
            "scientific_metrics_produced": bool(completion and completion.get("scientific_metrics_produced")),
            "research_effect_validated": bool(completion and completion.get("research_effect_validated")),
            "metrics": completion.get("metrics", {}) if completion else {},
            "paired_seeds": completion.get("paired_seeds", {}) if completion else {},
            "stage": job.get("stage"),
            "runtime_release": job.get("runtime_release"),
            "diagnostics": self._colab_diagnostics(job),
            "completion_kind": kind,
            "returncode": details.get("returncode") if current_runtime else None,
            "log_tail": log_tail,
            "log_status": "returned" if log_tail is not None else "not_returned",
            "remote_log_path": self._public_text(details.get("log_path")) if current_runtime and isinstance(details.get("log_path"), str) else None,
            "evidence": {
                "runtime_id": runtime_id,
                "observed_at": observation.get("observed_at"),
                "prepared_bundle_sha256": job.get("bundle_sha256"),
                "source_current": source_current,
                "metrics_file": completion.get("metrics_file") if completion else None,
                "origin": ("current_notebook_observation" if platform == "colab" else "current_backup_observation") if completion else "not_verified",
            },
            "blocking_issues": issues,
            "next_action": {"action": "analyze_results", "message": "ChatGPT should analyze these measured results and save the analysis and next research decision."} if completion else self._colab_next_action(job),
            "analysis_hint": "Only validated current-run metrics can support conclusions. A successful acceptance run is not a completed full-data experiment; missing logs remain unknown. Use record_chatgpt_analysis to persist ChatGPT's analysis.",
            "dashboard_url": self._dashboard(project_id),
        }

    @staticmethod
    def _public_url(value: str) -> str:
        try:
            parsed = urlsplit(value)
            return urlunsplit((parsed.scheme, parsed.netloc.rsplit("@", 1)[-1], parsed.path, "", ""))
        except ValueError:
            return "[invalid URL]"

    @staticmethod
    def _public_text(value: str) -> str:
        value = re.sub(r"https?://[^\s<>]+", lambda match: ChatGPTBridge._public_url(match.group()), value)
        value = re.sub(r"(?i)\b(mcpProxyToken|api[_-]?key|authorization|password|secret|token)\s*[:=]\s*[^\s,;]+", r"\1=[REDACTED]", value)
        value = re.sub(r"\b(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]+|AIza[A-Za-z0-9_-]{20,})", "[REDACTED]", value)
        return value[:8000]

    @classmethod
    def _public_colab_job(cls, job: dict[str, Any]) -> dict[str, Any]:
        fields = ("project_id", "status", "prepared", "attempt_count", "next_attempt_at",
                  "retry_min_minutes", "retry_max_minutes", "can_attempt", "autorun_enabled",
                  "worker_running", "created_at", "updated_at", "running_at", "finished_at",
                  "runtime_id", "bundle_sha256", "source_current", "source_file_count", "source_bytes",
                  "input_bundle_bytes", "drive_project_dir", "success_criteria", "source_manifest_sha256",
                  "notebook_download_url", "bundle_url", "input_bundle_url", "stage", "stage_updated_at",
                  "runtime_release", "runtime_requirements", "notebook_revision", "attempt_receipts",
                  "retry_delay_minutes", "gpu_wait_started_at", "consecutive_gpu_denials",
                  "waiting_data_check", "waiting_checkpoint_check", "data_cache", "checkpoint_receipt", "pending_notice")
        public = {key: job[key] for key in fields if key in job}
        backup = job.get("backup")
        if isinstance(backup, dict):
            public["backup"] = {key: backup[key] for key in ("status", "provider", "observed_at", "next_check_at") if key in backup}
            if isinstance(backup.get("reason"), str):
                public["backup"]["reason"] = cls._public_text(backup["reason"])
        public["blocked_reason"] = cls._public_text(str(job.get("blocked_reason") or ""))
        public["action_required"] = job.get("status") == "auth_required"
        public["diagnostics"] = cls._colab_diagnostics(job)
        for key in ("notebook_url", "browser_url"):
            if isinstance(job.get(key), str):
                public[key] = cls._public_url(job[key])
        observed = job.get("last_observation")
        if isinstance(observed, dict):
            public["last_observation"] = {key: observed.get(key) for key in ("status", "observed_at", "attempt_number")}
            details = observed.get("details") or {}
            public["last_observation"]["reason"] = cls._public_text(str(details.get("reason") or ""))
            if details.get("auth_kind") in {"google_sign_in", "drive_consent", "authorization"}:
                public["last_observation"]["auth_kind"] = details["auth_kind"]
        contract = job.get("checkpoint_contract")
        if isinstance(contract, dict):
            public["checkpoint_contract"] = {key: contract[key] for key in ("directory", "interval_seconds", "resume_supported", "full_training_state", "files") if key in contract}
        return public

    @classmethod
    def _colab_diagnostics(cls, job: dict[str, Any]) -> dict[str, Any]:
        details = (job.get("last_observation") or {}).get("details") or {}
        diagnostics = {}
        reason = details.get("provider_reason")
        if reason in ("session_capacity", "usage_quota"):
            diagnostics["provider_reason"] = reason
        preflight = details.get("preflight")
        if isinstance(preflight, dict):
            if preflight.get("status") in ("checking", "passed", "failed"):
                diagnostics["preflight_status"] = preflight["status"]
            if isinstance(preflight.get("reason"), str):
                diagnostics["preflight_error"] = cls._public_text(preflight["reason"][:500])
            framework = preflight.get("framework")
            if isinstance(framework, dict):
                for key in ("torch_version", "torchaudio_version"):
                    value = framework.get(key)
                    if isinstance(value, str) and re.fullmatch(r"[0-9A-Za-z.+_-]{1,64}", value):
                        diagnostics[key] = value
                for key in ("cuda_smoke", "torchaudio_smoke"):
                    probe = framework.get(key)
                    if isinstance(probe, dict) and probe.get("status") in ("passed", "failed"):
                        diagnostics[key] = probe["status"]
        return diagnostics

    @staticmethod
    def _colab_next_action(job: dict[str, Any]) -> dict[str, Any]:
        status = job.get("status")
        if (job.get("runtime_release") or {}).get("status") == "pending":
            action, message = "observe_runtime_release", "The worker is verifying release of the completed or safely stopped Colab runtime."
        elif not job.get("prepared") or status in {"not_prepared", "blocked", "cancelled", "failed"}:
            action, message = "prepare_experiment", "Review the manifest, code, inputs and failure evidence, then prepare the Colab notebook."
        elif job.get("source_current") is False:
            action, message = "prepare_experiment", "Code changed after this notebook was prepared; prepare a new source snapshot before executing or using results."
        elif status == "auth_required":
            details = (job.get("last_observation") or {}).get("details") or {}
            if job.get("ordinary_login"):
                action, message = "complete_login_then_close_window", "Finish Google sign-in in the ordinary Colab window and close it. The scheduler will recheck the signed-in session automatically."
            elif details.get("auth_kind") == "google_sign_in" and details.get("login_browser") == "required":
                if job.get("ordinary_login_recheck"):
                    action, message = "open_colab_login", "The sign-in recheck still needs Google authorization. Open the dedicated ordinary login window to finish sign-in."
                else:
                    action, message = "ordinary_login_pending", "AutoResearch will open the dedicated ordinary login window."
            elif details.get("auth_kind") == "google_sign_in":
                action, message = "complete_google_sign_in", "Complete Google sign-in in the dedicated Colab window. The worker will detect successful sign-in and continue automatically."
            else:
                action, message = "complete_browser_authorization", "Complete the indicated notebook or Drive action in the dedicated window, then call resume_colab_experiment using the existing authorization."
        elif (job.get("backup") or {}).get("status") in {"launching", "checking", "running", "unknown"}:
            action, message = "observe_backup_gpu", "AutoResearch is checking or observing the approved backup host."
        elif status in {"queued", "waiting_for_gpu", "waiting_for_connection", "disconnected"}:
            action, message = ("wait_for_retry", "Keep the local AutoResearch service running; its persistent scheduler respects the saved retry deadline.") if job.get("autorun_enabled") else ("start_experiment", "Call start_colab_experiment after the user has authorized this code and command.")
        elif status == "completed":
            action, message = "read_current_results", "Call get_colab_experiment_result to validate current-run evidence before analyzing or iterating."
        else:
            action, message = "observe_experiment", "The browser worker is checking resources or executing the notebook; poll status and current-run results."
        result = {"action": action, "message": message, "next_attempt_at": job.get("next_attempt_at")}
        if job.get("pending_notice"):
            result["notice_id"] = job["pending_notice"]["id"]
        return result

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
        workflow = self.research_workflow(project_id)
        collaborative = ((Path(project.workspace) / "research" / "chatgpt-brief.md").is_file()
                         or bool(workflow.get("plan_id")))
        return {
            "project": project_data,
            "reasoning_owner": "Current ChatGPT conversation" if collaborative else "AutoResearch/OpenRouter",
            "execution_model": project.model or self.settings.openrouter_model,
            "model_delegation": "explicit_generate_experiment_code_only" if collaborative else "configured_autoresearch_mode",
            "sources": [item.model_dump() for item in self.db.list_sources(project_id)],
            "experiments": experiments,
            "recent_events": self.db.list_events(project_id)[-30:],
            "artifacts": store.list_files(),
            "experiment_readiness": self.experiment_readiness(project_id),
            "colab": self.colab_status(project_id),
            "research_workflow": workflow,
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
