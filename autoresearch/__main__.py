from __future__ import annotations

import argparse
import threading

import uvicorn

from .config import Settings
from .services.desktop import DesktopBridge


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoResearch local research console")
    parser.add_argument("--host", help="监听地址（默认读取 .env）")
    parser.add_argument("--port", type=int, help="监听端口（默认读取 .env）")
    parser.add_argument("--no-open", action="store_true", help="不要自动打开 Chrome")
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
