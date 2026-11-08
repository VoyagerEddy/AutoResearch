"""Official Colab Notebook MCP bridge in an isolated Python 3.13 sidecar.

The main application can import this module on Python 3.11/3.12. Google's
``colab_mcp`` and its MCP v1 dependencies are imported only by the sidecar.
The bridge never provisions hardware, opens a browser, or uses CLI OAuth.
Runtime selection stays in the ordinary Colab notebook interface.
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from dataclasses import fields, is_dataclass
from importlib import metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit
import uuid


OFFICIAL_MCP_COMMIT = "b9ab3899e0f1fa493390b1fd6d54aa2e464ecdf1"
OFFICIAL_MCP_SOURCE = f"git+https://github.com/googlecolab/colab-mcp@{OFFICIAL_MCP_COMMIT}"
MAX_NOTEBOOK_MESSAGE_BYTES = 8 * 1024 * 1024
NOTEBOOK_TOOLS = frozenset({
    "get_cells", "add_code_cell", "add_text_cell", "update_cell",
    "delete_cell", "move_cell", "run_code_cell",
})
_REQUEST_ID = re.compile(r"^[0-9a-f]{32}$")


class ColabMcpError(RuntimeError):
    """A safe diagnostic that contains no provider credentials or URLs."""


def bounded_colab_websocket_server_class():
    """Keep Google's authentication and origin checks with a bounded payload.

    Native get_cells includes the notebook in text and structured content. A
    700 KB prepared notebook exceeds websockets' default 1 MB receive limit.
    This is the local desktop transport size, separate from Colab quotas.
    """
    import websockets
    from websockets.typing import Subprotocol
    from colab_mcp.websocket_server import ColabWebSocketServer

    class NotebookWebSocketServer(ColabWebSocketServer):
        async def __aenter__(self):
            self._server = await websockets.serve(
                self._connection_handler, host=self.host, port=0,
                subprotocols=[Subprotocol("mcp")], origins=self.allowed_origins,
                process_request=self._validate_authorization,
                max_size=MAX_NOTEBOOK_MESSAGE_BYTES,
            )
            self.port = self._server.sockets[0].getsockname()[1]
            return self

    return NotebookWebSocketServer


def atomic_private_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False)
            stream.flush()
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_private_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sidecar_python(root: Path) -> Path:
    base = root.resolve() / "data" / "colab-mcp" / "venv"
    path = base / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not path.is_file():
        raise ColabMcpError("The isolated official Colab MCP environment is not installed")
    return path


def validate_notebook_url(url: str) -> str:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in {"colab.research.google.com", "colab.google.com"}
        or parsed.username or parsed.password
        or parsed.port not in {None, 443}
        or parsed.fragment
        or not parsed.path.startswith(("/notebooks/", "/drive/", "/github/"))
    ):
        raise ColabMcpError("The bridge requires an official Colab notebook HTTPS URL")
    return url


def _dump_model(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", by_alias=True, exclude_none=True)
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _dump_model(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _dump_model(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_dump_model(item) for item in value]
    return value


class OfficialColabMcpSession:
    """In-process client of Google's public browser MCP proxy.

    Use only inside the separate Python >=3.13 environment. Connection URLs
    and tool results are private data; do not print them into application logs.
    """

    def __init__(self, notebook_url: str = "https://colab.research.google.com/notebooks/empty.ipynb"):
        self.notebook_url = validate_notebook_url(notebook_url)
        self.proxy: Any = None
        self.client: Any = None
        self._stack = AsyncExitStack()

    async def __aenter__(self) -> "OfficialColabMcpSession":
        if sys.version_info < (3, 13):
            raise ColabMcpError("Run the official bridge in its isolated Python 3.13 environment")
        import colab_mcp.session as official_session
        from fastmcp import Client, FastMCP

        try:
            # The pinned official session does not expose transport options.
            # Override its local factory, retaining the vendor's origin/token
            # validation and single-client handler without editing vendor files.
            official_session.ColabWebSocketServer = bounded_colab_websocket_server_class()
            self.proxy = official_session.ColabSessionProxy()
            await self.proxy.start_proxy_server()
            self._stack.push_async_callback(self.proxy.cleanup)
            server = FastMCP("AutoResearchColabNotebook")
            server.mount(self.proxy.proxy_server)
            for middleware in self.proxy.middleware:
                server.add_middleware(middleware)
            self.client = await self._stack.enter_async_context(Client(server))
            return self
        except BaseException:
            await self._stack.aclose()
            raise

    async def __aexit__(self, *args: object) -> None:
        await self._stack.aclose()

    @property
    def connected(self) -> bool:
        return bool(self.proxy and self.proxy.wss.connection_live.is_set())

    @property
    def private_connection_url(self) -> str:
        if not self.proxy:
            raise ColabMcpError("The bridge has not started")
        parsed = urlsplit(self.notebook_url)
        fragment = f"mcpProxyToken={self.proxy.wss.token}&mcpProxyPort={self.proxy.wss.port}"
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query, fragment))

    async def list_tools(self) -> list[dict[str, Any]]:
        return [_dump_model(tool) for tool in await self.client.list_tools()]

    async def call_tool(self, name: str, arguments: Mapping[str, Any], *, timeout: float = 900) -> dict[str, Any]:
        if name not in NOTEBOOK_TOOLS:
            raise ColabMcpError("Only observed notebook editing and execution tools are allowed")
        tools = await self.list_tools()
        if name not in {tool["name"] for tool in tools}:
            raise ColabMcpError("This notebook tool is not available; connect the official notebook MCP interface first")
        result = await self.client.call_tool(name, dict(arguments), timeout=timeout, raise_on_error=False)
        return _dump_model(result)


class ColabMcpSidecarClient:
    """Dependency-free file IPC client, usable by the main Python application."""

    def __init__(self, state_dir: Path):
        self.state_dir = state_dir.resolve()

    def status(self) -> dict[str, Any]:
        try:
            return read_private_json(self.state_dir / "status.json")
        except (OSError, ValueError):
            return {"status": "not_started", "connected": False}

    async def request(self, method: str, *, name: str | None = None,
                      arguments: Mapping[str, Any] | None = None, timeout: float = 900) -> dict[str, Any]:
        if method not in {"list_tools", "call_tool", "stop"}:
            raise ColabMcpError("Unsupported bridge request")
        if method == "call_tool" and name not in NOTEBOOK_TOOLS:
            raise ColabMcpError("Unsupported notebook tool")
        request_id = uuid.uuid4().hex
        request_path = self.state_dir / "requests" / f"{request_id}.json"
        response_path = self.state_dir / "responses" / f"{request_id}.json"
        atomic_private_json(request_path, {
            "id": request_id, "method": method, "name": name,
            "arguments": dict(arguments or {}), "timeout": timeout,
        })
        deadline = asyncio.get_running_loop().time() + timeout + 5
        while asyncio.get_running_loop().time() < deadline:
            if response_path.is_file():
                response = read_private_json(response_path)
                response_path.unlink(missing_ok=True)
                return response
            await asyncio.sleep(0.2)
        # A timed-out execution may still be running in Colab. The caller must
        # inspect cells/checkpoints before deciding whether to start it again.
        raise ColabMcpError("The bridge request timed out; check notebook execution before retrying")


def launch_colab_mcp_sidecar(root: Path, state_dir: Path, *, notebook_url: str | None = None) -> int:
    """Launch quietly without opening or changing any browser profile."""
    root = root.resolve(strict=True)
    state_dir = state_dir.resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    command = [str(sidecar_python(root)), "-m", "autoresearch.colab_mcp_cli",
               "serve", "--state-dir", str(state_dir)]
    if notebook_url:
        command.extend(["--notebook-url", validate_notebook_url(notebook_url)])
    # The isolated sidecar uses MCP v1; the main app uses MCP v2. Never import
    # or replace either environment's dependencies in the other process.
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root)
    environment["PYTHONUTF8"] = "1"
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with (state_dir / "sidecar.log").open("ab") as output:
        process = subprocess.Popen(command, cwd=root, env=environment,
                                   stdin=subprocess.DEVNULL, stdout=output, stderr=output,
                                   creationflags=flags)
    return process.pid


async def serve_colab_mcp(state_dir: Path, *, notebook_url: str) -> None:
    """Run private file IPC around the official frontend proxy; no browser action."""
    state_dir = state_dir.resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    requests = state_dir / "requests"
    requests.mkdir(exist_ok=True)
    stopped = asyncio.Event()
    pending: set[asyncio.Task[Any]] = set()

    async with OfficialColabMcpSession(notebook_url) as bridge:
        atomic_private_json(state_dir / "connection.private.json", {
            "pid": os.getpid(), "url": bridge.private_connection_url,
            "port": bridge.proxy.wss.port, "official_commit": OFFICIAL_MCP_COMMIT,
        })
        previous_connection = None
        initial_tools = await bridge.list_tools()
        atomic_private_json(state_dir / "tools.json", initial_tools)

        def write_status() -> None:
            atomic_private_json(state_dir / "status.json", {
                "pid": os.getpid(), "status": "connected" if bridge.connected else "awaiting_browser",
                "connected": bridge.connected, "port": bridge.proxy.wss.port,
                "python": sys.version.split()[0], "colab_mcp": metadata.version("colab-mcp"),
                "mcp": metadata.version("mcp"), "fastmcp": metadata.version("fastmcp"),
                "official_commit": OFFICIAL_MCP_COMMIT,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })

        async def handle_request(path: Path) -> None:
            request_id = path.stem
            if not _REQUEST_ID.fullmatch(request_id):
                return
            working = path.with_suffix(".working")
            try:
                path.rename(working)
            except FileNotFoundError:
                return
            response: dict[str, Any]
            try:
                request = read_private_json(working)
                method = request["method"]
                if method == "list_tools":
                    result = await bridge.list_tools()
                    atomic_private_json(state_dir / "tools.json", result)
                elif method == "call_tool":
                    result = await bridge.call_tool(request["name"], request.get("arguments", {}),
                                                    timeout=min(float(request.get("timeout", 900)), 7200))
                elif method == "stop":
                    result = {"stopping": True}
                    stopped.set()
                else:
                    raise ColabMcpError("Unsupported bridge request")
                response = {"ok": True, "result": result}
            except ColabMcpError as exc:
                response = {"ok": False, "error": str(exc)}
            except Exception as exc:
                # Provider errors may include OAuth URLs or Notebook contents.
                response = {"ok": False, "error": f"Official Colab MCP request failed ({type(exc).__name__})"}
            finally:
                working.unlink(missing_ok=True)
            try:
                atomic_private_json(state_dir / "responses" / f"{request_id}.json", response)
            except (TypeError, ValueError):
                atomic_private_json(state_dir / "responses" / f"{request_id}.json", {
                    "ok": False, "error": "The notebook result could not be serialized safely"})

        try:
            while not stopped.is_set():
                if bridge.connected != previous_connection:
                    previous_connection = bridge.connected
                    write_status()
                    # Refresh explicitly: the app does not depend on clients
                    # handling notifications/tools/list_changed correctly.
                    try:
                        tools = await asyncio.wait_for(bridge.list_tools(), timeout=15)
                        atomic_private_json(state_dir / "tools.json", tools)
                    except Exception:
                        pass
                for path in requests.glob("*.json"):
                    task = asyncio.create_task(handle_request(path))
                    pending.add(task)
                    task.add_done_callback(pending.discard)
                await asyncio.sleep(0.2)
        finally:
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            atomic_private_json(state_dir / "status.json", {"pid": os.getpid(), "status": "stopped", "connected": False})
            (state_dir / "connection.private.json").unlink(missing_ok=True)
