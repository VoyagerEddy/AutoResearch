from __future__ import annotations

import json
import os
import posixpath
import shlex
import socket
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import paramiko

from ..domain import SSHConnection


class RemoteExecutionError(RuntimeError):
    pass


@dataclass(slots=True)
class RemoteStatus:
    running: bool
    exit_code: int | None
    log_tail: str


class SSHRunner:
    EXCLUDED = {".git", ".env", "data", "results", "__pycache__", ".venv"}

    def __init__(self, connection: SSHConnection) -> None:
        self.connection = connection
        self.client: paramiko.SSHClient | None = None

    def __enter__(self) -> "SSHRunner":
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        kwargs: dict[str, Any] = {
            "hostname": self.connection.host,
            "port": self.connection.port,
            "username": self.connection.username,
            "timeout": 20,
            "banner_timeout": 20,
            "auth_timeout": 20,
        }
        if self.connection.password:
            kwargs["password"] = self.connection.password
        if self.connection.key_path:
            kwargs["key_filename"] = os.path.expandvars(self.connection.key_path)
        try:
            client.connect(**kwargs)
        except (paramiko.SSHException, socket.error, OSError) as exc:
            raise RemoteExecutionError(f"SSH 连接失败：{exc}") from exc
        self.client = client
        return self

    def __exit__(self, *_: object) -> None:
        if self.client:
            self.client.close()
            self.client = None

    def execute(self, command: str, timeout: int = 120) -> tuple[int, str, str]:
        if not self.client:
            raise RemoteExecutionError("SSH 尚未连接")
        try:
            _, stdout, stderr = self.client.exec_command(command, timeout=timeout)
            code = stdout.channel.recv_exit_status()
            return (
                code,
                stdout.read().decode("utf-8", errors="replace"),
                stderr.read().decode("utf-8", errors="replace"),
            )
        except (paramiko.SSHException, socket.error, OSError) as exc:
            raise RemoteExecutionError(f"远程命令失败：{exc}") from exc

    def gpu_processes(self) -> list[dict[str, int]]:
        command = (
            "nvidia-smi --query-compute-apps=pid,used_memory "
            "--format=csv,noheader,nounits 2>/dev/null || true"
        )
        _, stdout, _ = self.execute(command)
        processes: list[dict[str, int]] = []
        for line in stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) == 2 and all(part.isdigit() for part in parts):
                processes.append({"pid": int(parts[0]), "memory_mb": int(parts[1])})
        return processes

    def upload_tree(self, local_dir: Path, remote_dir: str) -> int:
        if not self.client:
            raise RemoteExecutionError("SSH 尚未连接")
        local_root = local_dir.resolve()
        if not local_root.is_dir():
            raise RemoteExecutionError(f"本地实验目录不存在：{local_root}")
        self.execute(f"mkdir -p {shlex.quote(remote_dir)}")
        count = 0
        with self.client.open_sftp() as sftp:
            for path in local_root.rglob("*"):
                relative = path.relative_to(local_root)
                if any(part in self.EXCLUDED for part in relative.parts):
                    continue
                destination = posixpath.join(remote_dir, *relative.parts)
                if path.is_dir():
                    self._mkdir_p(sftp, destination)
                    continue
                self._mkdir_p(sftp, posixpath.dirname(destination))
                sftp.put(str(path), destination)
                count += 1
        return count

    @staticmethod
    def _mkdir_p(sftp: paramiko.SFTPClient, path: str) -> None:
        current = "/" if path.startswith("/") else ""
        for part in path.split("/"):
            if not part:
                continue
            current = posixpath.join(current, part)
            try:
                attrs = sftp.stat(current)
                if not stat.S_ISDIR(attrs.st_mode):
                    raise RemoteExecutionError(f"远程路径不是目录：{current}")
            except FileNotFoundError:
                sftp.mkdir(current)

    def start(self, remote_dir: str, command: str) -> int:
        if not self.client:
            raise RemoteExecutionError("SSH 尚未连接")
        script = (
            "#!/usr/bin/env bash\n"
            "set -uo pipefail\n"
            f"cd {shlex.quote(remote_dir)}\n"
            f"{command}\n"
            "code=$?\n"
            "printf '%s' \"$code\" > .autoresearch.exit\n"
            "exit \"$code\"\n"
        )
        with self.client.open_sftp() as sftp:
            script_path = posixpath.join(remote_dir, ".autoresearch-run.sh")
            with sftp.open(script_path, "w") as handle:
                handle.write(script)
            sftp.chmod(script_path, 0o700)
        quoted = shlex.quote(remote_dir)
        launch = (
            f"cd {quoted} && rm -f .autoresearch.exit && "
            "nohup bash .autoresearch-run.sh > .autoresearch.log 2>&1 < /dev/null "
            "& echo $!"
        )
        code, stdout, stderr = self.execute(launch)
        if code != 0 or not stdout.strip().splitlines()[-1].isdigit():
            raise RemoteExecutionError(f"启动实验失败：{stderr or stdout}")
        return int(stdout.strip().splitlines()[-1])

    def status(self, remote_dir: str, pid: int) -> RemoteStatus:
        quoted = shlex.quote(remote_dir)
        command = (
            f"cd {quoted} && "
            f"if kill -0 {int(pid)} 2>/dev/null; then echo RUNNING; "
            "elif test -f .autoresearch.exit; then echo EXIT:$(cat .autoresearch.exit); "
            "else echo LOST; fi; "
            "echo __LOG__; tail -n 80 .autoresearch.log 2>/dev/null || true"
        )
        _, stdout, _ = self.execute(command)
        state, _, log = stdout.partition("\n__LOG__\n")
        state = state.strip()
        if state == "RUNNING":
            return RemoteStatus(True, None, log)
        if state.startswith("EXIT:"):
            value = state.split(":", 1)[1].strip()
            return RemoteStatus(False, int(value) if value.lstrip("-").isdigit() else 1, log)
        return RemoteStatus(False, None, log)

    def read_json(self, remote_path: str, max_bytes: int = 2_000_000) -> dict[str, Any]:
        if not self.client:
            raise RemoteExecutionError("SSH 尚未连接")
        with self.client.open_sftp() as sftp:
            try:
                attrs = sftp.stat(remote_path)
                if attrs.st_size > max_bytes:
                    raise RemoteExecutionError("指标文件超过大小限制")
                with sftp.open(remote_path, "r") as handle:
                    data = handle.read(max_bytes + 1)
            except FileNotFoundError:
                return {}
        if isinstance(data, bytes):
            data = data.decode("utf-8", errors="replace")
        try:
            value = json.loads(data)
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {"value": value}

