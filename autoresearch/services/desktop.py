from __future__ import annotations

import os
import subprocess
import webbrowser
from pathlib import Path
from urllib.parse import quote


class DesktopBridge:
    @staticmethod
    def open_chrome(url: str) -> bool:
        candidates = [
            Path(os.environ.get("PROGRAMFILES", "")) / "Google/Chrome/Application/chrome.exe",
            Path(os.environ.get("PROGRAMFILES(X86)", "")) / "Google/Chrome/Application/chrome.exe",
            Path(os.environ.get("LOCALAPPDATA", "")) / "Google/Chrome/Application/chrome.exe",
        ]
        for chrome in candidates:
            if chrome.is_file():
                subprocess.Popen([str(chrome), url])
                return True
        return webbrowser.open(url)

    @staticmethod
    def open_vscode(path: Path) -> None:
        subprocess.Popen(["code", str(path.resolve())])

    @staticmethod
    def open_remote_vscode(alias: str, remote_path: str) -> None:
        uri = f"vscode-remote://ssh-remote+{quote(alias, safe='@._-')}{remote_path}"
        subprocess.Popen(["code", "--folder-uri", uri])

    @staticmethod
    def launch_autodl_browser_login(
        root: Path,
        *,
        action: str = "Login",
        browser: str = "auto",
        open_page: str = "console",
        keep_open: bool = True,
    ) -> int:
        """Open the reusable AutoDL login helper in an interactive console.

        Credentials and the SMS code are read by that console and are never
        passed through the dashboard API or written to AutoResearch settings.
        """

        script = root.resolve() / "autodl-browser.ps1"
        if not script.is_file():
            raise OSError(f"AutoDL login helper was not found: {script}")
        command = [
            "powershell.exe",
            "-NoProfile",
            "-NoExit",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-Action",
            action,
            "-Browser",
            browser,
            "-OpenPage",
            open_page,
        ]
        if keep_open:
            command.append("-KeepOpen")
        creationflags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0) if os.name == "nt" else 0
        process = subprocess.Popen(command, cwd=str(root.resolve()), creationflags=creationflags)
        return process.pid
