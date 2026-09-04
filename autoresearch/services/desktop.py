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

