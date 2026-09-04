from __future__ import annotations

import asyncio
import json
import posixpath
from pathlib import Path
from typing import Any

from .config import Settings
from .db import Database
from .domain import AutoDLCreateRequest, ExperimentStartRequest, SSHConnection
from .orchestrator import ResearchOrchestrator
from .services.autodl import AutoDLClient, AutoDLError, extract_ssh
from .services.github_sync import GitSync, GitSyncError
from .services.ssh_runner import RemoteExecutionError, SSHRunner


class ExperimentManager:
    def __init__(
        self, db: Database, settings: Settings, orchestrator: ResearchOrchestrator
    ) -> None:
        self.db = db
        self.settings = settings
        self.orchestrator = orchestrator
        self._tasks: dict[str, asyncio.Task[None]] = {}

    def update_settings(self, settings: Settings) -> None:
        self.settings = settings

    def start(self, request: ExperimentStartRequest) -> str:
        if not request.confirm_execute:
            raise ValueError("上传并执行远程代码前必须明确确认")
        project = self.db.get_project(request.project_id)
        if not project:
            raise ValueError("研究项目不存在")
        generated = Path(project.workspace) / "generated"
        if not generated.is_dir():
            raise ValueError("实验代码尚未生成")
        experiment_id = self.db.create_experiment(
            request.project_id,
            request.remote_dir,
            request.command,
            request.instance_uuid or "",
        )
        task = asyncio.create_task(
            self._run(experiment_id, request), name=f"experiment-{experiment_id}"
        )
        self._tasks[experiment_id] = task
        task.add_done_callback(lambda _: self._tasks.pop(experiment_id, None))
        return experiment_id

    async def _run(self, experiment_id: str, request: ExperimentStartRequest) -> None:
        project = self.db.get_project(request.project_id)
        if not project:
            return
        generated = Path(project.workspace) / "generated"
        connection = request.connection
        instance_uuid = request.instance_uuid or ""
        try:
            connection, instance_uuid = await self._ensure_idle_gpu(
                request.project_id, connection, instance_uuid, request
            )
            self.db.update_experiment(
                experiment_id, status="uploading", instance_uuid=instance_uuid
            )
            for iteration in range(1, request.max_iterations + 1):
                self.db.update_experiment(
                    experiment_id, status="uploading", iteration=iteration
                )
                self.db.add_event(
                    request.project_id,
                    "experiment",
                    f"正在上传第 {iteration} 轮实验代码",
                )
                pid = await asyncio.to_thread(
                    self._upload_and_start,
                    connection,
                    generated,
                    request.remote_dir,
                    request.command,
                )
                self.db.update_experiment(experiment_id, status="running", pid=pid)
                self.db.add_event(
                    request.project_id,
                    "experiment",
                    f"远程实验已启动（PID {pid}）",
                )
                status = await self._monitor(
                    connection, request.remote_dir, pid, request.project_id
                )
                metrics = await asyncio.to_thread(
                    self._read_metrics,
                    connection,
                    posixpath.join(request.remote_dir, "results", "metrics.json"),
                )
                result = {
                    "exit_code": status.exit_code,
                    "log_tail": status.log_tail[-20000:],
                    "metrics": metrics,
                    "iteration": iteration,
                }
                self.db.update_experiment(experiment_id, status="analyzing", result=result)
                self.db.add_event(
                    request.project_id, "analyzing", f"第 {iteration} 轮实验完成，正在分析"
                )
                if status.exit_code != 0:
                    break
                if iteration >= request.max_iterations:
                    break
                improved = await self.orchestrator.analyze_and_improve(
                    request.project_id, experiment_id, result, iteration
                )
                if not improved:
                    break

            final = self.db.get_experiment(experiment_id) or {}
            final_result = final.get("result") or {}
            succeeded = final_result.get("exit_code") == 0
            self.db.update_experiment(
                experiment_id,
                status="completed" if succeeded else "failed",
                error="" if succeeded else "远程实验返回非零状态或状态丢失",
            )
            self.db.update_project(
                request.project_id,
                status="ready" if succeeded else "failed",
                phase="completed" if succeeded else "experiment_failed",
                progress=100,
                error="" if succeeded else "远程实验失败，请检查日志",
            )
            self.db.add_event(
                request.project_id,
                "completed" if succeeded else "experiment_failed",
                "实验循环已完成" if succeeded else "实验失败，已停止自动迭代",
                level="info" if succeeded else "error",
            )
            if succeeded and self.settings.github_remote_url:
                await asyncio.to_thread(self._sync_project, generated, request.project_id)
        except Exception as exc:
            self.db.update_experiment(
                experiment_id, status="failed", error=str(exc)[:4000]
            )
            self.db.update_project(
                request.project_id, status="failed", phase="experiment_failed", error=str(exc)[:4000]
            )
            self.db.add_event(
                request.project_id, "experiment_failed", f"实验编排失败：{exc}", level="error"
            )

    async def _ensure_idle_gpu(
        self,
        project_id: str,
        connection: SSHConnection,
        instance_uuid: str,
        request: ExperimentStartRequest,
    ) -> tuple[SSHConnection, str]:
        processes = await asyncio.to_thread(self._gpu_processes, connection)
        if not processes:
            return connection, instance_uuid
        self.db.add_event(
            project_id,
            "gpu_check",
            f"检测到 GPU 已被 {len(processes)} 个计算进程占用",
            level="warning",
            details={"processes": processes},
        )
        if not request.recover_busy_gpu:
            raise RemoteExecutionError("GPU 已被占用，且未启用自动恢复")
        if not instance_uuid or not self.settings.autodl_token:
            raise RemoteExecutionError("自动恢复需要 instance_uuid 和 AutoDL 开发者 Token")
        autodl = AutoDLClient(self.settings)
        try:
            image_uuid = await autodl.save_image(instance_uuid, f"autoresearch-{project_id}")
            if not image_uuid:
                raise AutoDLError("AutoDL 未返回克隆镜像 UUID")
            self.db.add_event(project_id, "gpu_recovery", "正在保存实例镜像并克隆到新实例")
            await autodl.wait_image(image_uuid)
            clone = await autodl.create_preferred(
                AutoDLCreateRequest(image_uuid=image_uuid, instance_name=f"AutoResearch-{project_id}-clone")
            )
            snapshot = await autodl.wait_running(clone.instance_uuid)
            ssh = extract_ssh(snapshot)
            if not ssh:
                raise AutoDLError("新实例已启动，但响应中没有可识别的 SSH 信息")
            cloned_connection = SSHConnection(**ssh)
            clone_processes = await asyncio.to_thread(self._gpu_processes, cloned_connection)
            if not clone_processes:
                return cloned_connection, clone.instance_uuid
            self.db.add_event(
                project_id, "gpu_recovery", "克隆实例 GPU 仍被占用", level="warning"
            )
            if not request.allow_release_replacement:
                raise AutoDLError("克隆实例 GPU 仍忙；释放实例需要在启动实验时明确授权")
            await autodl.power_off(clone.instance_uuid)
            await autodl.release(clone.instance_uuid)
            self.db.add_event(project_id, "gpu_recovery", "已释放忙碌的克隆实例，正在全新创建")
            fresh = await autodl.create_preferred(
                AutoDLCreateRequest(instance_name=f"AutoResearch-{project_id}-fresh")
            )
            fresh_snapshot = await autodl.wait_running(fresh.instance_uuid)
            fresh_ssh = extract_ssh(fresh_snapshot)
            if not fresh_ssh:
                raise AutoDLError("全新实例响应中没有可识别的 SSH 信息")
            fresh_connection = SSHConnection(**fresh_ssh)
            if await asyncio.to_thread(self._gpu_processes, fresh_connection):
                raise AutoDLError("全新实例 GPU 仍被占用，已停止以避免无限创建实例")
            return fresh_connection, fresh.instance_uuid
        finally:
            await autodl.close()

    @staticmethod
    def _gpu_processes(connection: SSHConnection) -> list[dict[str, int]]:
        with SSHRunner(connection) as runner:
            return runner.gpu_processes()

    @staticmethod
    def _upload_and_start(
        connection: SSHConnection, local: Path, remote: str, command: str
    ) -> int:
        with SSHRunner(connection) as runner:
            runner.upload_tree(local, remote)
            return runner.start(remote, command)

    async def _monitor(
        self, connection: SSHConnection, remote_dir: str, pid: int, project_id: str
    ):
        while True:
            status = await asyncio.to_thread(self._status, connection, remote_dir, pid)
            if not status.running:
                return status
            self.db.add_event(project_id, "experiment", "远程实验仍在运行")
            await asyncio.sleep(self.settings.monitor_seconds)

    @staticmethod
    def _status(connection: SSHConnection, remote_dir: str, pid: int):
        with SSHRunner(connection) as runner:
            return runner.status(remote_dir, pid)

    @staticmethod
    def _read_metrics(connection: SSHConnection, path: str) -> dict[str, Any]:
        with SSHRunner(connection) as runner:
            return runner.read_json(path)

    def _sync_project(self, generated: Path, project_id: str) -> None:
        try:
            GitSync(generated).sync(
                self.settings.github_remote_url,
                f"research/{project_id}",
                f"feat: update research experiment {project_id}",
            )
            self.db.add_event(project_id, "github", "实验代码已同步到 GitHub")
        except GitSyncError as exc:
            self.db.add_event(
                project_id, "github", f"GitHub 自动同步失败：{exc}", level="warning"
            )
