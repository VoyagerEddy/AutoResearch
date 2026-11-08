from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Sequence

from .services.colab_browser import ColabBrowserError
from .services.colab_resource_monitor import ColabResourceMonitor, launch_resource_monitor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Continuously observe the authorized Colab GPU workflow")
    parser.add_argument("--project", required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--api-url", default="http://127.0.0.1:8765")
    parser.add_argument("--browser", choices=("auto", "chrome", "edge"), default="chrome")
    parser.add_argument("--poll-seconds", type=float, default=10)
    parser.add_argument("--launch", action="store_true", help="Launch an idempotent hidden monitor and return")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = build_parser().parse_args(argv)
    try:
        if args.launch:
            pid = launch_resource_monitor(args.root, args.project, api_url=args.api_url, browser=args.browser)
            print(f"Colab resource monitor started (PID {pid}).", flush=True)
        else:
            monitor = ColabResourceMonitor(
                args.root, args.project, api_url=args.api_url, browser=args.browser,
                poll_seconds=args.poll_seconds,
            )
            asyncio.run(monitor.run())
    except (ColabBrowserError, OSError, ValueError) as exc:
        # Provider bodies, URLs, credentials, and tracebacks never enter logs.
        print(f"Colab resource monitor stopped ({type(exc).__name__}).", flush=True)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
