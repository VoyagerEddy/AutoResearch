from __future__ import annotations

import argparse
import sys
import threading

import uvicorn

from .config import Settings
from .services.desktop import DesktopBridge


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "autodl-browser-login":
        from .autodl_browser_cli import main as autodl_browser_main

        raise SystemExit(autodl_browser_main(sys.argv[2:]))

    parser = argparse.ArgumentParser(description="AutoResearch local research console")
    parser.add_argument("--host", help="Bind address (defaults to .env)")
    parser.add_argument("--port", type=int, help="Listen port (defaults to .env)")
    parser.add_argument("--no-open", action="store_true", help="Do not open Chrome automatically")
    args = parser.parse_args()
    settings = Settings.load()
    host = args.host or settings.host
    port = args.port or settings.port
    url = f"http://127.0.0.1:{port}"
    if not args.no_open:
        threading.Timer(1.2, DesktopBridge.open_chrome, args=(url,)).start()
    uvicorn.run("autoresearch.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
