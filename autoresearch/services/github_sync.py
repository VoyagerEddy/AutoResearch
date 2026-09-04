from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any


class GitSyncError(RuntimeError):
    pass


class GitSync:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    @staticmethod
    def validate_remote(remote: str) -> str:
        value = remote.strip()
        if not value:
            return ""
        if re.match(r"^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?$", value):
            return value
        if re.match(r"^git@github\.com:[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?$", value):
            return value
        raise GitSyncError("只允许 github.com 的 HTTPS 或 SSH 仓库地址，且地址中不能包含 Token")

    def _run(self, args: list[str], *, check: bool = True) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=self.root,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=120,
        )
        if check and result.returncode != 0:
            raise GitSyncError((result.stderr or result.stdout).strip())
        return result.stdout.strip()

    def sync(self, remote: str, branch: str, message: str) -> dict[str, Any]:
        remote = self.validate_remote(remote)
        if not self.root.exists():
            raise GitSyncError(f"同步目录不存在：{self.root}")
        if not (self.root / ".git").exists():
            self._run(["init", "-b", branch])
        self._run(["config", "user.name", "AutoResearch"])
        self._run(["config", "user.email", "autoresearch@users.noreply.github.com"])
        if remote:
            existing = self._run(["remote", "get-url", "origin"], check=False)
            if existing:
                if existing != remote:
                    self._run(["remote", "set-url", "origin", remote])
            else:
                self._run(["remote", "add", "origin", remote])
        self._run(["add", "--all"])
        changed = bool(self._run(["status", "--porcelain"]))
        if changed:
            self._run(["commit", "-m", message])
        commit = self._run(["rev-parse", "HEAD"], check=False)
        pushed = False
        if remote:
            self._run(["push", "-u", "origin", f"HEAD:{branch}"])
            pushed = True
        return {"changed": changed, "commit": commit, "pushed": pushed, "remote": remote}
