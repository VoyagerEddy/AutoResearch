from __future__ import annotations

import argparse
import asyncio
import getpass
import os
from pathlib import Path
from typing import Sequence

from .services.autodl_browser import (
    AutoDLBrowserError,
    AutoDLBrowserSession,
    AutoDLCredentials,
    BrowserLaunchOptions,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autoresearch autodl-browser-login",
        description="Open an authenticated AutoDL browser session using installed Chrome or Edge.",
    )
    parser.add_argument("--phone", help="AutoDL phone number; prompts when omitted")
    parser.add_argument(
        "--browser",
        choices=("auto", "chrome", "edge"),
        default="auto",
        help="Installed browser to launch",
    )
    parser.add_argument("--executable-path", type=Path, help="Explicit Chrome or Edge executable")
    parser.add_argument("--headless", action="store_true", help="Run without a visible browser window")
    parser.add_argument(
        "--open",
        choices=("console", "market", "none"),
        default="console",
        dest="open_page",
        help="Authenticated page to open after login",
    )
    parser.add_argument(
        "--keep-open",
        action="store_true",
        help="Keep the authenticated browser open until Enter is pressed",
    )
    return parser


async def _read_line(prompt: str) -> str:
    return await asyncio.to_thread(input, prompt)


async def run(args: argparse.Namespace) -> None:
    phone = args.phone or os.environ.get("AUTODL_BROWSER_PHONE") or input("AutoDL phone: ").strip()
    password = os.environ.get("AUTODL_BROWSER_PASSWORD") or getpass.getpass("AutoDL password: ")
    credentials = AutoDLCredentials(phone=phone, password=password)

    async def otp_provider() -> str:
        return (await _read_line("SMS one-time code: ")).strip()

    options = BrowserLaunchOptions(
        browser=args.browser,
        executable_path=args.executable_path,
        headless=args.headless,
    )
    async with AutoDLBrowserSession(options) as session:
        result = await session.login(credentials, otp_provider)
        if args.open_page == "console":
            await session.open_instance_list()
        elif args.open_page == "market":
            await session.open_instance_market()
        print(f"AutoDL browser session {result.status}")
        if args.keep_open:
            await _read_line("Press Enter to close the browser session...")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        asyncio.run(run(args))
    except (AutoDLBrowserError, OSError, ValueError) as exc:
        print(f"AutoDL browser login failed: {exc}")
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
