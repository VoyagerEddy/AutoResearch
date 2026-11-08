from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Sequence

import httpx

from .services.colab_browser import (
    ColabBrowserError,
    ColabBrowserOptions,
    ColabBrowserSession,
    ColabBrowserWorker,
    ColabJobClient,
    validate_local_api_url,
    validate_project_id,
    is_browser_closed_error,
)
from .services.autodl_browser import discover_browser_executable
from .services.colab_browser import COLAB_URL


class WorkerLock:
    """An OS-held lock releases automatically if the helper process exits."""

    def __init__(self, path: Path, *, wait_seconds: float = 0) -> None:
        self.path = path
        self.file = None
        self.wait_seconds = max(0, wait_seconds)

    def __enter__(self) -> "WorkerLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("a+b")
        deadline = time.monotonic() + self.wait_seconds
        while True:
            self.file.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if time.monotonic() < deadline:
                    time.sleep(0.05)
                    continue
                self.file.close()
                self.file = None
                raise ColabBrowserError("A Colab worker already uses this browser profile; finish or cancel that job first") from exc
        # A second launcher can open an empty file just before the first
        # locks it. Initialize only while owning the byte-range lock; writing
        # before locking races with that owner and can fail on Windows flush.
        if self.path.stat().st_size == 0:
            self.file.write(b"0")
            self.file.flush()
        return self

    def __exit__(self, *args: object) -> None:
        if self.file:
            self.file.close()


def _process_identity(pid: int) -> str | None:
    if pid <= 0:
        return None
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return None
        try:
            exit_code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(exit_code)) or exit_code.value != 259:
                return None
            created, exited, system, user = (wintypes.FILETIME() for _ in range(4))
            if not kernel.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited), ctypes.byref(system), ctypes.byref(user)):
                return None
            return str((created.dwHighDateTime << 32) | created.dwLowDateTime)
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        stat = Path(f"/proc/{pid}/stat")
        if stat.is_file():
            return stat.read_text().split(") ", 1)[1].split()[19]
        return str(pid)
    except (OSError, IndexError):
        return None


def worker_is_running(root: Path, project_id: str) -> int | None:
    project_id = validate_project_id(project_id)
    state = root.resolve() / "data" / "colab-browser" / f"{project_id}.worker.json"
    try:
        value = json.loads(state.read_text(encoding="utf-8"))
        pid = int(value["pid"])
        identity = _process_identity(pid)
        return pid if identity is not None and value.get("process_identity") == identity else None
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _publish_worker_identity(root: Path, project_id: str, pid: int) -> None:
    directory = root / "data" / "colab-browser"
    directory.mkdir(parents=True, exist_ok=True)
    value = {"pid": pid, "process_identity": _process_identity(pid), "project_id": project_id,
             "log_path": str(directory / f"{project_id}.log")}
    descriptor, filename = tempfile.mkstemp(prefix=f".{project_id}.worker-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(filename, directory / f"{project_id}.worker.json")
    finally:
        Path(filename).unlink(missing_ok=True)


def login_browser_is_running(root: Path) -> int | None:
    state = root.resolve() / "data" / "colab-browser" / "login-browser.json"
    try:
        value = json.loads(state.read_text(encoding="utf-8"))
        pid = int(value["pid"])
        identity = _process_identity(pid)
        return pid if identity is not None and value.get("process_identity") == identity else None
    except (OSError, ValueError, TypeError, KeyError):
        return None


def launch_colab_login_browser(
    root: Path,
    *,
    browser: str = "chrome",
    executable_path: Path | None = None,
    profile_dir: Path | None = None,
) -> int:
    """Open installed Chrome/Edge normally for manual Google login.

    This uses the same dedicated profile as the worker, but no automation or
    debugging flags. Close this ordinary browser before starting the worker.
    """
    if browser not in {"auto", "chrome", "edge"}:
        raise ValueError("browser must be auto, chrome, or edge")
    root = root.resolve(strict=True)
    executable = executable_path or discover_browser_executable(browser)
    if not executable or not executable.is_file():
        raise ColabBrowserError("Installed Google Chrome or Microsoft Edge was not found")
    profile = profile_dir or root / "data" / "colab-browser" / "profile"
    profile.mkdir(parents=True, exist_ok=True)
    with WorkerLock(profile.with_name(f"{profile.name}.lock")):
        existing = login_browser_is_running(root)
        if existing:
            return existing
        # Chromium on Linux uses SingletonLock; on Windows a live lockfile
        # can deny read access. Do not remove or alter either browser lock.
        if (profile / "SingletonLock").exists() or (profile / "SingletonLock").is_symlink():
            raise ColabBrowserError("Close the browser using this dedicated Colab profile before opening manual login")
        lockfile = profile / "lockfile"
        if lockfile.exists():
            try:
                with lockfile.open("rb"):
                    pass
            except OSError as exc:
                raise ColabBrowserError("The dedicated Colab profile is still owned by a browser; close that window first") from exc
        command = [str(executable), f"--user-data-dir={profile.resolve()}", "--no-first-run", COLAB_URL]
        process = subprocess.Popen(command, cwd=str(root))
        state = root / "data" / "colab-browser" / "login-browser.json"
        state.parent.mkdir(parents=True, exist_ok=True)
        state.write_text(json.dumps({"pid": process.pid, "process_identity": _process_identity(process.pid),
                                     "profile_path": str(profile.resolve())}), encoding="utf-8")
        return process.pid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the visible, persistent Colab notebook worker")
    parser.add_argument("--project", help="Prepared AutoResearch project id; required for the worker")
    parser.add_argument("--login-only", action="store_true", help="Open the same profile in an ordinary browser for manual Google login, then exit")
    parser.add_argument("--api-url", default="http://127.0.0.1:8765")
    parser.add_argument("--browser", choices=("auto", "chrome", "edge"), default="auto")
    parser.add_argument("--executable-path", type=Path)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser


async def run(args: argparse.Namespace) -> None:
    project_id = validate_project_id(args.project)
    root = args.root.resolve()
    if login_browser_is_running(root):
        raise ColabBrowserError("Close the ordinary Google login browser window before starting the Colab worker")
    profile = args.profile_dir or root / "data" / "colab-browser" / "profile"
    options = ColabBrowserOptions(profile_dir=profile, browser=args.browser, executable_path=args.executable_path)
    client = ColabJobClient(args.api_url, project_id)
    try:
        job = await client.status()
        if not job.get("prepared"):
            raise ColabBrowserError("Prepare the Colab notebook in AutoResearch before launching the worker")
        with WorkerLock(profile.with_name(f"{profile.name}.lock")):
            # The Windows venv launcher can have a different PID from this
            # interpreter. Publish the process that actually owns the profile,
            # after the parent finishes publishing its provisional identity.
            with WorkerLock(root / "data" / "colab-browser" / "launch.lock", wait_seconds=10):
                _publish_worker_identity(root, project_id, os.getpid())
            async with ColabBrowserSession(options) as browser:
                await ColabBrowserWorker(client, browser).run()
    except Exception as exc:
        if not is_browser_closed_error(exc):
            raise
        latest = await client.status()
        if latest.get("status") not in {"completed", "failed", "cancelled"}:
            await client.observe("network", reason="The dedicated Colab browser closed during startup; its profile and saved retry queue are preserved", browser_closed=True)
        print("Colab worker stopped after browser closure", flush=True)
    finally:
        await client.close()


def launch_colab_worker(
    root: Path,
    project_id: str,
    *,
    api_url: str = "http://127.0.0.1:8765",
    browser: str = "auto",
) -> int:
    """Launch a background helper; the actual browser is visible for manual login.

    Credentials live only in the dedicated browser profile, never in command
    arguments, API payloads, or logs. Retries are persisted by the local API.
    """
    project_id = validate_project_id(project_id)
    api_url = validate_local_api_url(api_url)
    if browser not in {"auto", "chrome", "edge"}:
        raise ValueError("browser must be auto, chrome, or edge")
    root = root.resolve(strict=True)
    log_dir = root / "data" / "colab-browser"
    log_dir.mkdir(parents=True, exist_ok=True)
    with WorkerLock(log_dir / "launch.lock", wait_seconds=10):
        return _launch_colab_worker_locked(root, project_id, api_url=api_url, browser=browser)


def _launch_colab_worker_locked(root: Path, project_id: str, *, api_url: str, browser: str) -> int:
    existing_pid = worker_is_running(root, project_id)
    if existing_pid is not None:
        return existing_pid
    if login_browser_is_running(root):
        raise ColabBrowserError("Close the ordinary Google login browser window before starting the Colab worker")
    log_dir = root / "data" / "colab-browser"
    for metadata in log_dir.glob("*.worker.json"):
        other_id = metadata.name[:-len(".worker.json")]
        try:
            running = worker_is_running(root, other_id)
        except ValueError:
            continue
        if running is not None:
            raise ColabBrowserError("A Colab worker already uses this browser profile; finish or cancel that job first")
    profile = log_dir / "profile"
    # A child may own the profile just before it can publish its real PID.
    # Testing that lock under the launcher lock avoids spawning a loser that
    # could overwrite the live child's metadata.
    with WorkerLock(profile.with_name(f"{profile.name}.lock")):
        pass
    command = [
        sys.executable, "-u", "-m", "autoresearch.colab_browser_cli",
        "--project", project_id, "--root", str(root), "--api-url", api_url,
        "--browser", browser,
    ]
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    with (log_dir / f"{project_id}.log").open("ab") as log_file:
        process = subprocess.Popen(
            command,
            cwd=str(root),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=environment,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0,
        )
    _publish_worker_identity(root, project_id, process.pid)
    return process.pid


def main(argv: Sequence[str] | None = None) -> int:
    # Windows redirected consoles can otherwise use GBK, which cannot encode
    # some ordinary Colab ARIA labels and would terminate the worker.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.login_only and not args.project:
        parser.error("--project is required unless --login-only is supplied")
    try:
        if args.login_only:
            pid = launch_colab_login_browser(args.root, browser=args.browser,
                                             executable_path=args.executable_path, profile_dir=args.profile_dir)
            print(f"Ordinary Colab login browser opened (PID {pid}). Complete Google login, then close this window before starting the worker.", flush=True)
            return 0
        asyncio.run(run(args))
    except (ColabBrowserError, OSError, ValueError, httpx.HTTPError) as exc:
        print(f"Colab worker stopped: {exc}", flush=True)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
