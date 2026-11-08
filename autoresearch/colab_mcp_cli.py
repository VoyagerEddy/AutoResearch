"""Isolated official Colab MCP sidecar, without credential-bearing log output."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
import sys

from .services.colab_mcp import (
    ColabMcpError, ColabMcpSidecarClient, atomic_private_json, read_private_json,
    serve_colab_mcp,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Official Colab notebook MCP sidecar")
    parser.add_argument("command", choices=("serve", "status", "request"))
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--notebook-url", default="https://colab.research.google.com/notebooks/empty.ipynb")
    parser.add_argument("--method", choices=("list_tools", "call_tool", "stop"), default="list_tools")
    parser.add_argument("--tool")
    parser.add_argument("--arguments-file", type=Path)
    parser.add_argument("--output-file", type=Path)
    parser.add_argument("--timeout", type=float, default=900)
    return parser


async def run(args: argparse.Namespace) -> None:
    if args.command == "serve":
        # Google's websocket logger contains its local port, and dependency
        # debug logs may contain protocol bodies. Keep sidecar logs quiet.
        logging.disable(logging.CRITICAL)
        await serve_colab_mcp(args.state_dir, notebook_url=args.notebook_url)
    elif args.command == "status":
        print(json.dumps(ColabMcpSidecarClient(args.state_dir).status(), ensure_ascii=False))
    else:
        arguments = read_private_json(args.arguments_file) if args.arguments_file else {}
        response = await ColabMcpSidecarClient(args.state_dir).request(
            args.method, name=args.tool, arguments=arguments, timeout=args.timeout,
        )
        if args.output_file:
            atomic_private_json(args.output_file, response)
        # Actual tool responses are kept in the private output file. Cells can
        # contain Drive authorization URLs; do not dump them into shell logs.
        result = response.get("result")
        print(json.dumps({"ok": response.get("ok"), "tool_count": len(result) if isinstance(result, list) else None,
                          "saved": bool(args.output_file), "error": response.get("error")}, ensure_ascii=False))


def main() -> None:
    try:
        asyncio.run(run(build_parser().parse_args()))
    except KeyboardInterrupt:
        return
    except ColabMcpError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from None
    except Exception as exc:
        print(f"Official Colab MCP sidecar failed ({type(exc).__name__}); no provider details were logged", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
