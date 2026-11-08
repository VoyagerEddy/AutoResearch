from __future__ import annotations

import asyncio
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qs, urlparse, urlunparse

import httpx

from .autodl_browser import discover_browser_executable
from .colab_mcp import ColabMcpSidecarClient, OFFICIAL_MCP_COMMIT, launch_colab_mcp_sidecar, read_private_json, sidecar_python
from .colab_continuation import plan_authorization_continuation


COLAB_URL = "https://colab.research.google.com/"
COLAB_EMPTY_NOTEBOOK_URL = "https://colab.research.google.com/notebooks/empty.ipynb"
GOOGLE_SIGN_IN_REASON = "Complete Google sign-in in the visible browser, then Resume in AutoResearch"
STATUS_PREFIX = "AUTORESEARCH_COLAB_STATUS "
UPLOAD_PREFIX = "AUTORESEARCH_COLAB_UPLOAD_BUNDLE"
DATA_UPLOAD_PREFIX = "AUTORESEARCH_COLAB_UPLOAD_DATA "
DATA_READY_PREFIX = 'AUTORESEARCH_COLAB_DATA_READY '
INPUTS_PREFIX = "AUTORESEARCH_COLAB_UPLOAD_INPUTS "
INPUTS_READY_PREFIX = 'AUTORESEARCH_COLAB_INPUTS_READY '
_PROJECT_ID = re.compile(r"^[A-Za-z0-9_-]{1,100}$")
_UI_CONTROLS = (
    'button, a[href], select, input[type="radio"], input[type="text"], input[type="search"], [role="textbox"], [role="button"], '
    '[role="menuitem"], [role="tab"], [role="radio"], '
    '[role="combobox"], [role="option"]'
)
_AUTH_TEXT = (
    "verify it's you", "verify it’s you", "verification required", "captcha",
    "connect to google drive", "access your google drive", "sign in to continue",
    "this notebook requires authentication", "verify you are human",
    "验证您", "验证你的身份", "连接到 google 云端硬盘", "登录以继续",
)
_QUOTA_PATTERNS = tuple(re.compile(pattern) for pattern in (
    # A GPU advisory also mentions usage limits. Only an actual connection
    # denial or a reached/exceeded limit establishes a quota failure.
    r"\b(?:cannot|can't|can’t|unable to)\s+(?:currently\s+)?connect\s+to\s+(?:a\s+)?gpu(?:\s+backend)?\s+(?:due to|because of)\s+(?:the\s+)?(?:gpu\s+)?usage limits?\b",
    r"\busage limits? in colab[.!:;\s]+(?:you\s+)?(?:cannot|can't|can’t|unable to)\s+(?:currently\s+)?connect\s+to\s+(?:a\s+)?gpu\b",
    r"\b(?:gpu\s+)?usage limits?\s+(?:(?:has|have) been\s+)?(?:reached|exceeded)\b",
    r"\b(?:you have|you've|you’ve)\s+(?:reached|exceeded)\s+(?:your|the)\s+(?:gpu\s+)?(?:usage|session) limits?\b",
    r"\btoo many (?:active )?sessions\b",
    r"已(?:经)?(?:达到|超出)(?:您(?:的)?|你的)?使用(?:量)?(?:限制|限额)",
    r"(?:您|你)?(?:目前|当前)?无法连接到\s*gpu[^。.!?]{0,80}(?:原因是|由于)[^。.!?]{0,40}使用(?:量)?(?:限制|限额)",
))
_UNAVAILABLE_TEXT = (
    "cannot connect to gpu backend", "unable to connect to a gpu", "no gpu available",
    "no gpus available", "gpu is not available", "gpu backend is unavailable",
    "无法连接到 gpu", "没有可用的 gpu", "gpu 后端不可用",
)
_DISCONNECTED_TEXT = (
    "runtime disconnected", "session disconnected", "runtime has been disconnected",
    "your runtime has disconnected", "运行时已断开", "会话已断开",
)
_NETWORK_TEXT = (
    "err_connection", "err_name_not_resolved", "err_internet_disconnected",
    "this site can’t be reached", "this site can't be reached", "无法访问此网站",
)
_PUBLIC_NETWORK_ERROR_CODES = frozenset({
    "ERR_CONNECTION_TIMED_OUT", "ERR_CONNECTION_RESET", "ERR_CONNECTION_CLOSED",
    "ERR_CONNECTION_REFUSED", "ERR_CONNECTION_FAILED", "ERR_CONNECTION_ABORTED",
    "ERR_NAME_NOT_RESOLVED", "ERR_INTERNET_DISCONNECTED",
})


class ColabBrowserError(RuntimeError):
    """An observable browser or local worker failure."""


class ManualColabAction(ColabBrowserError):
    """The normal Colab interface requires the user's action."""


_SAFE_LOCAL_ERRORS = frozenset({
    "The official notebook bridge connection is invalid",
    "The official Colab notebook request failed",
    "The official Colab notebook tool reported an error",
    "The official Colab notebook returned an unreadable result",
    "The official Colab notebook cell structure is not recognized",
    "The Colab notebook contains an unsupported cell",
    "The Colab notebook cell source is not recognized",
    "The official notebook bridge tools could not be read",
    "Prepared notebook authoring did not complete; verify its source before running",
    "The authored notebook does not match the prepared artifact; execution was stopped",
    "The official notebook bridge did not identify this notebook's cells",
    "The official notebook bridge returned invalid cell identifiers",
    "The prepared notebook source checksum does not match this job",
    "The official Colab notebook source changed after import; execution was stopped",
})


def safe_local_error_reason(error: Exception, fallback: str) -> str:
    # Only our fixed local error messages are printable; provider messages may
    # contain connection tokens, authorization URLs, or notebook source.
    return str(error) if isinstance(error, ColabBrowserError) and str(error) in _SAFE_LOCAL_ERRORS else f"{fallback} ({type(error).__name__})"


def is_browser_closed_error(error: BaseException) -> bool:
    """Recognize closure without retaining provider URLs or exception payloads."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if type(current).__name__ == "TargetClosedError":
            return True
        if type(current).__name__ == "Error" and "Target page, context or browser has been closed" in str(current):
            return True
        current = current.__cause__ or current.__context__
    return False


@dataclass(frozen=True, slots=True)
class ColabBrowserOptions:
    profile_dir: Path
    browser: str = "auto"
    executable_path: Path | None = None
    timeout_ms: int = 45_000
    poll_seconds: float = 20.0

    def __post_init__(self) -> None:
        if self.browser not in {"auto", "chrome", "edge"}:
            raise ValueError("browser must be auto, chrome, or edge")
        if self.timeout_ms < 1_000 or self.poll_seconds < 1:
            raise ValueError("Browser timeouts and polling intervals must be positive")


def validate_project_id(project_id: str) -> str:
    if not _PROJECT_ID.fullmatch(project_id):
        raise ValueError("Invalid AutoResearch project id")
    return project_id


def validate_colab_url(url: str) -> str:
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "colab.research.google.com"
        or parsed.username
        or parsed.password
        or parsed.port not in (None, 443)
    ):
        raise ValueError("Notebook URL must use the official Colab HTTPS host")
    return url


def public_colab_url(url: str) -> str:
    """Drop private MCP connection credentials before saving a browser URL."""
    parsed = urlparse(validate_colab_url(url))
    return urlunparse(parsed._replace(fragment=""))


def prepared_notebook_cells(job: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Read the exported artifact and verify its source stamp before authoring."""
    expected = str(job.get("bundle_sha256") or "")
    project_id = validate_project_id(str(job.get("project_id") or ""))
    path = Path(str(job.get("notebook_path") or "")).resolve(strict=True)
    if path.name != f"AutoResearch_{project_id}.ipynb":
        raise ColabBrowserError("Prepared notebook filename does not match this project")
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ColabBrowserError("The prepared notebook could not be read") from exc
    if not isinstance(notebook, dict) or notebook.get("nbformat") != 4:
        raise ColabBrowserError("The prepared notebook is not a version 4 notebook")
    result = []
    for cell in notebook.get("cells", []):
        if not isinstance(cell, dict) or cell.get("cell_type") not in {"code", "markdown"}:
            raise ColabBrowserError("The prepared notebook contains an unsupported cell")
        source = cell.get("source")
        if isinstance(source, list) and all(isinstance(line, str) for line in source):
            source = "".join(source)
        if not isinstance(source, str):
            raise ColabBrowserError("The prepared notebook contains invalid source")
        result.append((cell["cell_type"], source))
    markdown = "\n".join(source for kind, source in result if kind == "markdown")
    if not result or not source_stamp_matches(markdown, expected):
        raise ColabBrowserError("The prepared notebook source checksum does not match this job")
    return result


def mcp_tool_result(response: Mapping[str, Any]) -> Any:
    """Decode the public FastMCP result, never log notebook source or errors."""
    if response.get("ok") is not True:
        raise ColabBrowserError("The official Colab notebook request failed")
    result = response.get("result")
    if not isinstance(result, dict) or result.get("isError") or result.get("is_error"):
        raise ColabBrowserError("The official Colab notebook tool reported an error")
    # Google's tools return JSON in TextContent. FastMCP versions may also
    # provide the decoded value, or wrap a non-object value under 'result'.
    for block in result.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "text":
            try:
                return json.loads(block["text"])
            except (KeyError, TypeError, ValueError):
                continue
    for key in ("structuredContent", "structured_content", "data"):
        value = result.get(key)
        if value is not None:
            return value.get("result") if isinstance(value, dict) and set(value) == {"result"} else value
    raise ColabBrowserError("The official Colab notebook returned an unreadable result")


def mcp_notebook_cells(response: Mapping[str, Any]) -> list[tuple[str, str]]:
    value = mcp_tool_result(response)
    if not isinstance(value, dict) or not isinstance(value.get("cells"), list):
        raise ColabBrowserError("The official Colab notebook cell structure is not recognized")
    cells = []
    for cell in value["cells"]:
        if not isinstance(cell, dict) or cell.get("cell_type") not in {"code", "markdown"}:
            raise ColabBrowserError("The Colab notebook contains an unsupported cell")
        source = cell.get("source")
        if isinstance(source, list) and all(isinstance(line, str) for line in source):
            source = "".join(source)
        if not isinstance(source, str):
            raise ColabBrowserError("The Colab notebook cell source is not recognized")
        cells.append((cell["cell_type"], source))
    return cells


def mcp_stream_output_text(response: Mapping[str, Any]) -> str:
    """Read only nbformat stdout/stderr streams, never cell source or HTML."""
    value = mcp_tool_result(response)
    cells = value.get("cells") if isinstance(value, dict) else None
    if not isinstance(cells, list):
        raise ColabBrowserError("The official Colab notebook output structure is not recognized")
    chunks = []
    for cell in cells:
        if not isinstance(cell, dict) or not isinstance(cell.get("outputs"), list):
            continue
        for output in cell["outputs"]:
            if not isinstance(output, dict) or output.get("output_type") != "stream" or output.get("name") not in {"stdout", "stderr"}:
                continue
            value = output.get("text")
            if isinstance(value, list) and all(isinstance(line, str) for line in value):
                value = "".join(value)
            if isinstance(value, str):
                chunks.append(value)
    return "\n".join(chunks)


def mcp_cells_match(actual: list[tuple[str, str]], expected: list[tuple[str, str]]) -> bool:
    # The canonical empty notebook contains one empty code cell. Leave it
    # intact; exact source comparison prevents executing partial or old imports.
    if actual and actual[0][0] == "code" and not actual[0][1].strip():
        actual = actual[1:]
    return actual == expected


def validate_local_api_url(url: str) -> str:
    parsed = urlparse(url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username
        or parsed.password
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("The Colab worker API must be a local loopback HTTP URL")
    return url.rstrip("/")


def classify_colab_page(url: str, visible_text: str) -> str | None:
    """Only explicit visible errors count as quota or GPU unavailability.

    A disconnected indicator on an unopened notebook is normal, so generic
    'Connect' or 'Disconnected' labels deliberately do not imply a failed run.
    """
    parsed = urlparse(url)
    text = re.sub(r"\s+", " ", visible_text.casefold())
    if parsed.hostname == "accounts.google.com" or any(x in text for x in _AUTH_TEXT):
        return "auth"
    if any(pattern.search(text) for pattern in _QUOTA_PATTERNS):
        return "quota"
    if any(x in text for x in _UNAVAILABLE_TEXT):
        return "unavailable"
    if any(x in text for x in _NETWORK_TEXT):
        return "network"
    if any(x in text for x in _DISCONNECTED_TEXT):
        return "disconnected"
    return None


def provider_cooldown_minutes(text: str) -> int | None:
    """Respect a displayed retry duration; don't infer it from GPU specifications."""
    pattern = re.compile(
        r"(?:try again|retry|available again|reconnect|wait|请等待|等待|重试|再试)"
        r"[^\n]{0,80}?(\d+(?:\.\d+)?)\s*(hours?|hrs?|minutes?|mins?|小时|分钟)",
        re.IGNORECASE,
    )
    match = pattern.search(text)
    if not match:
        return None
    amount = float(match.group(1))
    factor = 60 if match.group(2).casefold().startswith(("h", "小时")) else 1
    return max(1, math.ceil(amount * factor))


def control_label(name: str) -> str:
    """Normalize observed menu labels without guessing their DOM selectors."""
    first_line = name.splitlines()[0].strip()
    first_line = re.sub(r"\s*\(?\s*(?:Ctrl\+|Alt\+|Shift\+|Cmd\+|⌘)[^\n]*$", "", first_line)
    return first_line.rstrip(". …").casefold()


def safe_control_diagnostics(controls: list[dict[str, str]]) -> list[dict[str, str]]:
    """Log UI labels, excluding account identifiers and long payloads."""
    unique = []
    seen = set()
    for control in controls:
        name = control.get("name", "").strip()
        if not name or len(name) > 90 or "@" in name or ".ipynb" in name.casefold() or re.search(r"\d{7,}|account|profile|账户|帐号|账号|个人资料", name, re.IGNORECASE):
            continue
        key = (control.get("role", ""), name)
        if key not in seen:
            item = {"role": key[0], "name": name}
            if control.get("frame_key"):
                item["frame_key"] = control["frame_key"]
            unique.append(item)
            seen.add(key)
        if len(unique) >= 60:
            break
    return unique


def sanitize_provider_ui_text(value: str, *, limit: int = 1500) -> str:
    """Keep provider error wording while removing account/connection details."""
    value = re.sub(r"(?:https?|wss?)://[^\s<>\"']+", "[URL]", value, flags=re.IGNORECASE)
    value = re.sub(r"[A-Za-z0-9_.+%-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[email]", value)
    value = re.sub(r"[A-Za-z0-9_-]{16,}&\d{1,5}", "[connection code]", value)
    value = re.sub(r"\bmcpProxy(?:Token|Port)\s*=\s*[^\s&#]+", "[private connection]", value, flags=re.IGNORECASE)
    value = re.sub(r"(?<!\w)(?:\+?\d[\s().-]*){7,}\d(?!\w)", "[number]", value)
    value = re.sub(
        r"(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{16,}(?![A-Za-z0-9_-])",
        lambda match: match[0] if match[0] in _PUBLIC_NETWORK_ERROR_CODES else "[identifier]",
        value,
    )
    return re.sub(r"\s+", " ", value).strip()[:limit]


def source_stamp_matches(rendered_markdown: str, expected_sha256: str) -> bool:
    if not re.fullmatch(r"[a-f0-9]{64}", expected_sha256):
        return False
    stamps = re.findall(r"AUTORESEARCH_SOURCE_SHA256\s+([a-f0-9]{64})(?![a-f0-9])", rendered_markdown)
    return bool(stamps) and set(stamps) == {expected_sha256}


def parse_aria_controls(snapshot: str) -> list[dict[str, str]]:
    """Read public role/name pairs from Playwright's accessibility snapshot."""
    result = []
    pattern = re.compile(r'^\s*-\s+(button|menuitem|tab|radio|combobox|option|link|textbox)\s+"((?:\\.|[^"\\])*)"')
    for line in snapshot.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        try:
            name = json.loads('"' + match.group(2) + '"')
        except ValueError:
            continue
        if name and len(name) < 160:
            result.append({"role": match.group(1), "name": name})
    return result


def is_trusted_picker_frame(url: str) -> bool:
    parsed = urlparse(url)
    try:
        if parsed.username or parsed.password or parsed.port not in (None, 443):
            return False
    except ValueError:
        return False
    return parsed.scheme == "https" and (
        parsed.hostname == "colab.research.google.com"
        or parsed.hostname in {"docs.google.com", "drive.google.com"}
        and re.match(r"^/(?:u/\d+/)?picker(?:/|$)", parsed.path) is not None
    )


def parse_status_markers(text: str, project_id: str) -> list[dict[str, Any]]:
    """Parse complete notebook output lines, never Python source or substrings."""
    markers = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith(STATUS_PREFIX):
            continue
        try:
            value = json.loads(line[len(STATUS_PREFIX):])
        except (ValueError, TypeError):
            continue
        if not isinstance(value, dict) or value.get("project_id") != project_id:
            continue
        if value.get("status") not in {"starting", "available", "auth", "running", "unavailable", "disconnected", "completed", "failed"}:
            continue
        if not isinstance(value.get("runtime_id"), str) or not value["runtime_id"]:
            continue
        markers.append(value)
    return markers


def marker_is_current(marker: Mapping[str, Any], attempt_started: datetime) -> bool:
    try:
        value = str(marker["started_at"])
        started_at = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if started_at.tzinfo is None:
            return False
        # A small allowance covers skew between the local clock and Colab VM.
        return started_at >= attempt_started.astimezone(timezone.utc) - timedelta(seconds=120)
    except (KeyError, ValueError, TypeError):
        return False


def parse_data_upload_requests(text: str, project_id: str, *, prefix: str = DATA_UPLOAD_PREFIX) -> list[dict[str, Any]]:
    requests = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith(prefix):
            continue
        try:
            value = json.loads(line[len(prefix):])
        except (ValueError, TypeError):
            continue
        if (isinstance(value, dict) and value.get('project_id') == project_id
                and isinstance(value.get('runtime_id'), str) and value['runtime_id']
                and isinstance(value.get('path'), str)
                and isinstance(value.get('sha256'), str)
                and re.fullmatch(r'[0-9a-f]{64}', value['sha256'])
                and type(value.get('size_bytes')) is int and 0 < value['size_bytes'] <= 512 * 1024 * 1024):
            requests.append(value)
    return requests


class ColabJobClient:
    def __init__(self, api_url: str, project_id: str) -> None:
        self.api_url = validate_local_api_url(api_url)
        self.project_id = validate_project_id(project_id)
        self.path = f"/api/projects/{self.project_id}/colab"
        self.client = httpx.AsyncClient(base_url=self.api_url, timeout=20, trust_env=False)

    async def close(self) -> None:
        await self.client.aclose()

    async def status(self) -> dict[str, Any]:
        response = await self.client.get(self.path)
        response.raise_for_status()
        return response.json()

    async def claim(self) -> bool:
        response = await self.client.post(f"{self.path}/attempt", json={})
        if response.status_code == 409:
            return False
        response.raise_for_status()
        return True

    async def resume(self) -> bool:
        response = await self.client.post(f"{self.path}/resume", json={})
        if response.status_code == 409:
            return False
        response.raise_for_status()
        return True

    async def bind_notebook(self, *, notebook_url: str, bundle_sha256: str) -> dict[str, Any]:
        response = await self.client.post(f"{self.path}/notebook-binding", json={
            'notebook_url':notebook_url,'bundle_sha256':bundle_sha256})
        response.raise_for_status()
        return response.json()

    async def observe(self, status: str, *, retry_after_minutes: int | None = None, **details: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {"status": status, "details": details}
        if retry_after_minutes is not None:
            payload["retry_after_minutes"] = retry_after_minutes
        response = await self.client.post(
            f"{self.path}/observations", json=payload
        )
        response.raise_for_status()
        return response.json()


class ColabBrowserSession:
    """Visible, persistent browser using only Colab's normal notebook interface.

    The adapter discovers visible controls on every action. If the interface
    changes or a label is ambiguous, it stops for manual action instead of
    clicking a guessed element. It never calls an allocation API or sends
    artificial activity to prolong a runtime.
    """

    def __init__(self, options: ColabBrowserOptions) -> None:
        self.options = options
        self.page: Any = None
        self.context: Any = None
        self._playwright: Any = None
        self._control_frames: dict[str, Any] = {}
        self._notebook_mcp: ColabMcpSidecarClient | None = None

    async def __aenter__(self) -> "ColabBrowserSession":
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:
            raise ColabBrowserError("Install optional browser support: pip install -e '.[browser]'") from exc
        executable = self.options.executable_path or discover_browser_executable(self.options.browser)
        if not executable or not executable.is_file():
            raise ColabBrowserError("Installed Google Chrome or Microsoft Edge was not found")
        self.options.profile_dir.mkdir(parents=True, exist_ok=True)
        self._playwright = await async_playwright().start()
        try:
            self.context = await self._playwright.chromium.launch_persistent_context(
                user_data_dir=str(self.options.profile_dir.resolve()),
                executable_path=str(executable),
                headless=False,
                viewport={"width": 1400, "height": 950},
                locale="en-US",
                accept_downloads=False,
            )
            self.context.set_default_timeout(self.options.timeout_ms)
            self.page = next((p for p in self.context.pages if "colab.research.google.com" in p.url), None)
            self.page = self.page or await self.context.new_page()
        except Exception as exc:
            await self._playwright.stop()
            raise ColabBrowserError(f"Could not start the dedicated browser ({type(exc).__name__})") from exc
        return self

    async def __aexit__(self, *args: Any) -> None:
        try:
            if self.context:
                try:
                    await self.context.close()
                except Exception as exc:
                    if not is_browser_closed_error(exc):
                        raise
        finally:
            if self._playwright:
                await self._playwright.stop()

    async def visible_text(self) -> str:
        # Provider messages must not include notebook Python or historical
        # outputs: both may legitimately contain strings like 'no GPU available'.
        return await self.page.locator("body").evaluate(
            """body => {
                const parts = [];
                const walk = node => {
                    if (node.nodeType === Node.TEXT_NODE) {
                        const parent = node.parentElement;
                        if (parent && !parent.closest('colab-code-cell, colab-text-cell, colab-output-area, .cell, [id^="cell-"], .inputarea, .input_area, .output_subarea, .output-area, .CodeMirror, .monaco-editor') &&
                            parent.getClientRects().length && getComputedStyle(parent).visibility !== 'hidden') {
                            parts.push(node.textContent);
                        }
                        return;
                    }
                    if (node.nodeType !== Node.ELEMENT_NODE) return;
                    if (node.matches('colab-code-cell, colab-text-cell, colab-output-area, .cell, [id^="cell-"], .codecell, .inputarea, .input_area, .output_subarea, .output-area, .CodeMirror, .monaco-editor')) return;
                    for (const child of node.childNodes) walk(child);
                    if (node.shadowRoot) for (const child of node.shadowRoot.childNodes) walk(child);
                };
                walk(body);
                return parts.join(' ').slice(0, 100000);
            }""", timeout=10_000
        )

    async def output_text(self) -> str:
        """Read output frames and rendered output elements, excluding code input."""
        chunks = []
        if self._notebook_mcp is not None and self._notebook_mcp.status().get("connected"):
            try:
                response = await self._notebook_mcp.request("call_tool", name="get_cells", arguments={"includeOutputs": True}, timeout=15)
                # Native streams have one chronological order. Mixing their
                # history with delayed iframe text can make an older file
                # request appear newer and target the wrong active picker.
                return mcp_stream_output_text(response)
            except Exception as exc:
                if is_browser_closed_error(exc):
                    raise
                # MCP can briefly disconnect during runtime changes. The
                # existing rendered-output reader remains available; provider
                # error payloads and notebook contents are never logged here.
                pass
        for frame in self.page.frames:
            if frame == self.page.main_frame:
                continue
            host = urlparse(frame.url).hostname or ""
            if host.endswith(".googleusercontent.com") or host == "colab.research.google.com":
                try:
                    chunks.append(await frame.locator("body").inner_text(timeout=3_000))
                except Exception:
                    pass
        # Output components may render in the main frame on some Colab versions.
        for locator in await self.page.locator("colab-output-area, .output_subarea").all():
            try:
                if await locator.is_visible():
                    chunks.append(await locator.inner_text(timeout=3_000))
            except Exception:
                pass
        return "\n".join(chunks)

    async def controls(self) -> list[dict[str, str]]:
        controls = []
        frames = getattr(self.page, "frames", None) or [self.page]
        self._control_frames = {}
        for index, frame in enumerate(frames):
            if frame is not getattr(self.page, "main_frame", self.page) and not is_trusted_picker_frame(frame.url):
                continue
            key = f"frame-{index}"
            self._control_frames[key] = frame
            observed = []
            snapshot_reader = getattr(frame.locator("body"), "aria_snapshot", None)
            if callable(snapshot_reader):
                try:
                    observed = parse_aria_controls(await snapshot_reader(timeout=5_000))
                except Exception as exc:
                    if is_browser_closed_error(exc):
                        raise
            try:
                observed.extend(await self._dom_controls(frame))
            except Exception as exc:
                if is_browser_closed_error(exc):
                    raise
            seen = set()
            for item in observed:
                identity = (item["role"], item["name"])
                if identity not in seen:
                    controls.append({**item, "frame_key": key})
                    seen.add(identity)
        return controls

    async def _dom_controls(self, frame: Any) -> list[dict[str, str]]:
        return await frame.locator(_UI_CONTROLS).evaluate_all(
            r"""elements => elements.filter(el => el.getClientRects().length &&
                getComputedStyle(el).visibility !== 'hidden').map(el => {
                const refs = (el.getAttribute('aria-labelledby') || '').split(/\s+/)
                    .map(id => el.getRootNode().getElementById?.(id)?.textContent || document.getElementById(id)?.textContent || '').join(' ');
                const nativeRole = el.tagName === 'A' ? 'link' : el.tagName === 'SELECT' ? 'combobox' : el.type === 'radio' ? 'radio' : el.tagName === 'INPUT' ? 'textbox' : 'button';
                return {
                    role: el.getAttribute('role') || nativeRole,
                    name: (el.getAttribute('aria-label') || refs.trim() ||
                           (el.labels ? Array.from(el.labels).map(label => label.innerText).join(' ') : '') ||
                           (el.tagName === 'INPUT' ? el.placeholder : el.innerText) || el.title || '').trim()
                };
            }).filter(el => el.name && el.name.length < 160)"""
        )

    async def click_observed(self, labels: tuple[str, ...]) -> bool:
        controls = await self.controls()
        allowed = {control_label(label) for label in labels}
        candidates = [control for control in controls if control_label(control["name"]) in allowed]
        for control in candidates:
            frame = self._control_frames.get(control.get("frame_key", ""), self.page)
            locator = frame.get_by_role(control["role"], name=control["name"], exact=True)
            visible = [item for item in await locator.all() if await item.is_visible()]
            if len(visible) == 1 and not await visible[0].is_disabled():
                await visible[0].click()
                return True
        return False

    async def ui_diagnostics(self) -> dict[str, Any]:
        controls = safe_control_diagnostics(await self.controls())
        frames = []
        for frame in getattr(self.page, "frames", []):
            parsed = urlparse(frame.url)
            value = {"host": parsed.hostname, "path": re.sub(r"[^/\s]*@[^/\s]*", "[redacted]", parsed.path)[:180]}
            if value not in frames:
                frames.append(value)
            if len(frames) >= 12:
                break
        cell_attributes = []
        try:
            cell_attributes = await self.page.locator("colab-code-cell, colab-text-cell, .cell").evaluate_all(
                """elements => elements.slice(0, 40).map(element => ({
                    tag: element.tagName.toLowerCase(),
                    id: element.getAttribute('id'),
                    data_cell_id: element.getAttribute('data-cell-id'),
                    cell_id: element.getAttribute('cell-id')
                }))"""
            )
        except Exception as exc:
            if is_browser_closed_error(exc):
                raise
        provider_text = ""
        dialog_titles = []
        try:
            provider_text = sanitize_provider_ui_text(await self.visible_text())
            for dialog in await self.page.get_by_role("dialog").all():
                if not await dialog.is_visible():
                    continue
                title = await dialog.evaluate("""element => element.getAttribute('aria-label') ||
                    (element.getAttribute('aria-labelledby') || '').split(/\\s+/).map(id =>
                        element.getRootNode().getElementById?.(id)?.textContent || document.getElementById(id)?.textContent || '').join(' ')""")
                safe_title = sanitize_provider_ui_text(str(title or ""), limit=180)
                if safe_title and safe_title not in dialog_titles:
                    dialog_titles.append(safe_title)
                if len(dialog_titles) >= 5:
                    break
        except Exception as exc:
            if is_browser_closed_error(exc):
                raise
        return {
            "ui_controls": controls,
            "ui_role_tree": "\n".join(f"- {item['role']} {json.dumps(item['name'], ensure_ascii=False)}" for item in controls),
            "ui_frames": frames,
            "notebook_cell_attributes": cell_attributes,
            "provider_ui_text": provider_text,
            "dialog_titles": dialog_titles,
        }

    async def click_when_visible(self, labels: tuple[str, ...], *, timeout_seconds: float = 8) -> bool:
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        while True:
            if await self.click_observed(labels):
                return True
            if asyncio.get_running_loop().time() >= deadline:
                return False
            await asyncio.sleep(0.25)

    async def _adopt_new_colab_tab(self, previous_pages: list[Any], *, timeout_seconds: float = 3) -> None:
        if self.context is None:
            return
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        while True:
            for page in self.context.pages:
                if page not in previous_pages and urlparse(page.url).hostname == "colab.research.google.com":
                    self.page = page
                    await page.wait_for_load_state("domcontentloaded")
                    return
            if asyncio.get_running_loop().time() >= deadline:
                return
            await asyncio.sleep(0.25)

    async def open_upload_picker(self) -> bool:
        upload = ("Upload", "上传")
        opener = ("Open file picker", "Open notebook", "Open notebook…", "Open notebook...", "Open Colab", "打开文件选择器", "打开笔记本", "打开 notebook", "打开笔记本…", "打开笔记本...")
        if await self.click_observed(("Sign in", "登录", "登入")):
            raise ManualColabAction(GOOGLE_SIGN_IN_REASON)
        if await self.click_when_visible(upload, timeout_seconds=2):
            return True
        if await self.click_observed(opener):
            if await self.click_when_visible(upload):
                return True
        if await self.click_observed(("File", "文件")):
            if await self.click_when_visible(opener, timeout_seconds=3):
                if await self.click_when_visible(upload):
                    return True
        previous_pages = list(self.context.pages) if self.context else []
        if await self.click_observed(("New notebook", "New notebook…", "New notebook...", "新建笔记本", "新建 notebook")):
            await self._adopt_new_colab_tab(previous_pages)
            if await self.click_when_visible(("File", "文件")):
                if await self.click_when_visible(opener, timeout_seconds=3):
                    if await self.click_when_visible(upload):
                        return True
        # The home-page file picker can show only Recent/Search controls. The
        # official empty notebook opens the editor with its standard import
        # dialog; it contains no example code to execute or runtime to allocate.
        await self.click_observed(("Cancel", "取消"))
        await self.page.goto(COLAB_EMPTY_NOTEBOOK_URL, wait_until="domcontentloaded")
        if await self.click_observed(("Sign in", "登录", "登入")):
            raise ManualColabAction(GOOGLE_SIGN_IN_REASON)
        if await self.click_when_visible(("File", "文件")):
            if await self.click_when_visible(opener, timeout_seconds=3):
                return await self.click_when_visible(upload)
        return False

    def _official_mcp_location(self) -> tuple[Path, Path] | None:
        """Enable the installed bridge only for the application's own profile."""
        profile = self.options.profile_dir.resolve()
        if len(profile.parents) < 3 or tuple(part.casefold() for part in profile.parts[-3:]) != ("data", "colab-browser", "profile"):
            return None
        root = profile.parents[2]
        state = root / "data" / "colab-mcp" / "speaker-session"
        # This integration is optional; a custom profile or missing isolated
        # environment keeps the standard notebook upload flow available.
        if not state.is_dir() or state.resolve() != state:
            return None
        try:
            sidecar_python(root)
        except RuntimeError:
            return None
        return root, state

    async def _current_notebook_mcp_dialog(self) -> Any | None:
        dialogs = [dialog for dialog in await self.page.get_by_role("dialog").all() if await dialog.is_visible()]
        if len(dialogs) != 1:
            return None
        dialog = dialogs[0]
        markers = [control for control in await dialog.get_by_role("button", name="Connect separate tabs", exact=True).all() if await control.is_visible()]
        if len(markers) == 1:
            return dialog
        titles = {"connect to a local colab mcp server", "connect to local colab mcp server", "connect to mcp server", "connect to an mcp server"}
        # Current Colab labels this dialog itself, without a heading. Its tab
        # selector's accessible name can include the expand icon, so neither
        # a child heading nor that selector alone is a reliable identifier.
        if callable(getattr(dialog, "evaluate", None)):
            title = await dialog.evaluate("""element => element.getAttribute('aria-label') ||
                (element.getAttribute('aria-labelledby') || '').split(/\\s+/).map(id =>
                    element.getRootNode().getElementById?.(id)?.textContent || document.getElementById(id)?.textContent || '').join(' ')""")
            if control_label(str(title or "")) in titles:
                return dialog
        for heading in await dialog.get_by_role("heading").all():
            if await heading.is_visible() and control_label(await heading.inner_text()) in titles:
                return dialog
        return None

    async def _observed_editable_controls(self, scope: Any) -> list[tuple[str, str, Any]]:
        candidates = []
        for item in await self._dom_controls(scope):
            if item["role"] not in {"textbox", "combobox"}:
                continue
            controls = [control for control in await scope.get_by_role(item["role"], name=item["name"], exact=True).all() if await control.is_visible()]
            if not controls:
                # An observed placeholder can label a search field while its
                # accessibility name is empty. Match that exact visible value.
                controls = [control for control in await scope.get_by_placeholder(item["name"], exact=True).all() if await control.is_visible()]
            if len(controls) == 1 and not await controls[0].is_disabled():
                candidates.append((item["role"], item["name"], controls[0]))
        return candidates

    async def open_saved_notebook_mcp_command(self) -> None:
        async def palette_search():
            fields = await self._observed_editable_controls(self.page)
            search = [field for _,name,field in fields if re.search(r'command|palette|search|命令|搜索',name,re.IGNORECASE)]
            if len(search)>1:
                raise ManualColabAction('The notebook command search is ambiguous; select its MCP connection command manually')
            return search[0] if search else None
        search = await palette_search()
        if search is None and not await self.click_when_visible(("Commands Show command palette Ctrl+Shift+P", "Commands Show command palette", "Show command palette", "Command palette"), timeout_seconds=20):
            if not await self.click_observed(("Tools", "工具")) or not await self.click_when_visible(("Command palette", "Command palette…", "命令面板"), timeout_seconds=3):
                raise ManualColabAction("Open this saved notebook's command palette and select Connect to a local Colab MCP server")
        deadline = asyncio.get_running_loop().time() + 5
        while True:
            search = search or await palette_search()
            if search is not None:
                await search.fill("MCP")
                break
            if asyncio.get_running_loop().time() >= deadline:
                raise ManualColabAction("The command palette search field could not be identified; search MCP in this notebook's command palette")
            await asyncio.sleep(0.25)
        labels=('Connect to a local Colab MCP server','Connect to local Colab MCP server','Connect to MCP server','Connect to an MCP server')
        if await self.click_when_visible(labels,timeout_seconds=1):
            return
        # Some command rows expose text without an accessibility role. Exact
        # public text in this page is sufficient only when visibly unique.
        deadline=asyncio.get_running_loop().time()+5
        while True:
            visible=[]
            for label in labels:
                visible.extend([item for item in await self.page.get_by_text(label,exact=True).all() if await item.is_visible()])
            if len(visible)==1:
                await visible[0].click()
                return
            if len(visible)>1:
                raise ManualColabAction('The MCP connect command is ambiguous; select it manually')
            for label in ('Disconnect from the local Colab MCP server','Disconnect from a local Colab MCP server'):
                if any([await item.is_visible() for item in await self.page.get_by_text(label,exact=True).all()]):
                    # The UI is already connected or completing its bridge
                    # handshake. Let the caller poll/verify native ownership.
                    return
            if asyncio.get_running_loop().time()>=deadline:
                raise ManualColabAction('Select Connect to a local Colab MCP server in this notebook command palette')
            await asyncio.sleep(0.25)

    async def connect_current_notebook_mcp_dialog(self, *, connection_token: str | None = None, connection_port: str | None = None) -> bool:
        """Accept only the observed MCP dialog for this notebook.

        The current project is authorized to use this local bridge. Other
        dialogs, runtime Connect buttons and the separate-tabs control are
        deliberately outside this action's scope.
        """
        dialog = await self._current_notebook_mcp_dialog()
        if dialog is None:
            return False
        if connection_token is not None and connection_port is not None:
            fields = await self._observed_editable_controls(dialog)
            combined = [field for _, name, field in fields if re.search(r"connection\s*(?:code|string)|proxy\s*(?:code|string)|mcp\s*(?:code|string)|连接(?:码|代码|字符串)", name, re.IGNORECASE)]
            tokens = [field for _, name, field in fields if re.search(r"token|令牌", name, re.IGNORECASE)]
            ports = [field for _, name, field in fields if re.search(r"port|端口", name, re.IGNORECASE)]
            # A saved-notebook URL can prefill the bridge parameters before
            # its consent dialog renders. That UI has no editable fields.
            try:
                current_fragment=parse_qs(urlparse(validate_colab_url(self.page.url)).fragment)
                prefilled=(current_fragment.get('mcpProxyToken')==[connection_token]
                           and current_fragment.get('mcpProxyPort')==[connection_port])
            except (ValueError,AttributeError):
                prefilled=False
            try:
                if not fields and prefilled:
                    pass
                elif len(fields) == 1 and (len(combined) == 1 or len(tokens) == 1):
                    await fields[0][2].fill(connection_token + "&" + connection_port)
                elif len(fields) == 2 and len(tokens) == len(ports) == 1 and tokens[0] is not ports[0]:
                    await tokens[0].fill(connection_token)
                    await ports[0].fill(connection_port)
                else:
                    raise ManualColabAction("The MCP connection fields could not be identified; enter the current local bridge connection code in this notebook's MCP dialog")
            except ManualColabAction:
                raise
            except Exception as exc:
                if is_browser_closed_error(exc):
                    raise
                raise ManualColabAction("The local MCP connection code could not be entered; complete this notebook's MCP dialog") from None
        buttons = [control for control in await dialog.get_by_role("button", name="Connect", exact=True).all() if await control.is_visible()]
        if len(buttons) != 1 or await buttons[0].is_disabled():
            return False
        await buttons[0].click()
        return True

    async def ensure_mcp_registered_notebook(self, response: Mapping[str, Any], *, timeout_seconds: float = 5) -> None:
        """Bind the MCP cells to the notebook receiving the runtime UI action."""
        value = mcp_tool_result(response)
        cells = value.get("cells") if isinstance(value, dict) else None
        if not isinstance(cells, list) or not cells:
            raise ColabBrowserError("The official notebook bridge did not identify this notebook's cells")
        identifiers = [cell.get("id") for cell in cells if isinstance(cell, dict)]
        if len(identifiers) != len(cells) or any(not isinstance(identifier, str) or not identifier for identifier in identifiers):
            raise ColabBrowserError("The official notebook bridge returned invalid cell identifiers")
        deadline = asyncio.get_running_loop().time() + min(max(timeout_seconds, 0), 5)
        while True:
            # Inspect only DOM attributes of the public notebook cell hosts.
            # No application internals or hidden notebook state are accessed.
            rendered_identifiers = await self.page.locator('colab-code-cell, colab-text-cell, [id^="cell-"]').evaluate_all(
                "elements => elements.flatMap(element => Array.from(element.attributes, attribute => attribute.value))"
            )
            if all(identifier in rendered_identifiers or f"cell-{identifier}" in rendered_identifiers for identifier in identifiers):
                return
            if asyncio.get_running_loop().time() >= deadline:
                raise ManualColabAction("The official bridge is connected to another notebook or its cells could not be verified in this tab; reconnect this notebook before running")
            await asyncio.sleep(0.25)

    async def _connect_official_notebook_mcp(self, root: Path, state: Path, *, expected_cells: list[tuple[str, str]] | None = None, notebook_url: str | None = None) -> ColabMcpSidecarClient:
        # Import at runtime: the CLI imports BrowserSession during startup.
        from ..colab_browser_cli import _process_identity

        client = ColabMcpSidecarClient(state)
        try:
            connection = read_private_json(state / "connection.private.json")
            live = _process_identity(int(connection["pid"])) is not None
        except (OSError, ValueError, TypeError, KeyError):
            live = False
        if not live:
            # A single browser worker owns this profile's OS lock. It may
            # restart a dead bridge, but never launches alongside a live PID.
            launch_colab_mcp_sidecar(root, state)
        deadline = asyncio.get_running_loop().time() + min(self.options.timeout_ms / 1000, 45)
        while True:
            try:
                connection = read_private_json(state / "connection.private.json")
                if _process_identity(int(connection["pid"])) is not None:
                    break
            except (OSError, ValueError, TypeError, KeyError):
                pass
            if asyncio.get_running_loop().time() >= deadline:
                raise ManualColabAction("The official Colab notebook bridge did not start; resume after checking the local bridge")
            await asyncio.sleep(0.25)
        private_url = str(connection.get("url") or "")
        parsed = urlparse(validate_colab_url(private_url))
        fragment = parse_qs(parsed.fragment, strict_parsing=True)
        if (
            parsed.path != "/notebooks/empty.ipynb" or parsed.query
            or set(fragment) != {"mcpProxyToken", "mcpProxyPort"}
            or any(len(values) != 1 for values in fragment.values())
            or not fragment["mcpProxyToken"][0]
            or not fragment["mcpProxyPort"][0].isdigit()
            or not 1 <= int(fragment["mcpProxyPort"][0]) <= 65535
            or connection.get("official_commit") != OFFICIAL_MCP_COMMIT
        ):
            raise ColabBrowserError("The official notebook bridge connection is invalid")
        if notebook_url is not None:
            saved = urlparse(validate_colab_url(notebook_url))
            if saved.fragment or not re.fullmatch(r"/drive/[A-Za-z0-9_-]{1,200}", saved.path):
                raise ColabBrowserError("A saved notebook attachment requires a public Colab Drive URL")
            # The token is composed only in memory for this authorized saved
            # notebook. Job state and user-facing links keep the public URL.
            private_url = urlunparse(saved._replace(fragment=parsed.fragment))
        # Credentials remain in the private IPC file and this browser's URL.
        # Observations persist only the public URL without its fragment.
        reuse_current_page = self.page.url == private_url
        if not reuse_current_page:
            await self.page.goto(private_url, wait_until="domcontentloaded")
        deadline = asyncio.get_running_loop().time() + min(self.options.timeout_ms / 1000, 45)
        current_dialog_accepted = False
        saved_command_opened = False
        while True:
            # Colab requires its normal connection consent. Never click the
            # 'Connect separate tabs' control or another permission dialog.
            if notebook_url is not None and not client.status().get("connected"):
                current_dialog_accepted = await self.connect_current_notebook_mcp_dialog(
                    connection_token=fragment["mcpProxyToken"][0], connection_port=fragment["mcpProxyPort"][0]
                ) or current_dialog_accepted
            else:
                current_dialog_accepted = await self.connect_current_notebook_mcp_dialog() or current_dialog_accepted
            if notebook_url is not None and not client.status().get("connected") and not current_dialog_accepted:
                if not saved_command_opened and await self._current_notebook_mcp_dialog() is None:
                    try:
                        await self.open_saved_notebook_mcp_command()
                    except ManualColabAction:
                        # The URL can open its own consent dialog while the
                        # command palette is still being located. Continue
                        # polling that observed dialog instead of pausing.
                        if await self._current_notebook_mcp_dialog() is None and not client.status().get('connected'):
                            raise
                    saved_command_opened = True
                current_dialog_accepted = await self.connect_current_notebook_mcp_dialog(
                    connection_token=fragment["mcpProxyToken"][0], connection_port=fragment["mcpProxyPort"][0]
                ) or current_dialog_accepted
            # The sidecar may still report an older tab's closing connection.
            # A fresh notebook's own consent must precede trusting that flag.
            if client.status().get("connected"):
                if current_dialog_accepted:
                    break
                # Colab can reuse an existing user grant without showing its
                # consent dialog again. Bind the live bridge to this tab's
                # actual native cell IDs before trusting that connection.
                # This proof also rejects a still-connected different tab.
                response = await client.request("call_tool", name="get_cells", arguments={"includeOutputs": False}, timeout=30)
                actual = mcp_notebook_cells(response)
                blank = len(actual) == 1 and actual[0][0] == "code" and not actual[0][1].strip()
                if not blank and (expected_cells is None or not mcp_cells_match(actual, expected_cells)):
                    raise ManualColabAction("The connected notebook contains existing content that does not match this prepared project; open a fresh empty notebook before continuing")
                try:
                    await self.ensure_mcp_registered_notebook(response, timeout_seconds=0)
                except ManualColabAction:
                    # A stale connection can outlive navigation briefly.
                    # Wait for this tab's consent or verified cell ownership;
                    # neither a connected flag nor a blank source is enough.
                    pass
                else:
                    break
            if classify_colab_page(self.page.url, await self.visible_text()) == "auth":
                raise ManualColabAction(GOOGLE_SIGN_IN_REASON)
            if asyncio.get_running_loop().time() >= deadline:
                raise ManualColabAction("Open the official Colab notebook bridge in this browser and resume once connected")
            await asyncio.sleep(0.25)
        self._notebook_mcp = client
        return client

    async def author_prepared_notebook_with_mcp(self, job: Mapping[str, Any]) -> bool:
        location = self._official_mcp_location()
        if location is None:
            return False
        expected = prepared_notebook_cells(job)
        if self._notebook_mcp is not None and self._notebook_mcp.status().get("connected"):
            actual = mcp_notebook_cells(await self._notebook_mcp.request("call_tool", name="get_cells", arguments={"includeOutputs": False}, timeout=30))
            if mcp_cells_match(actual, expected):
                return True
        client = await self._connect_official_notebook_mcp(*location, expected_cells=expected)
        tool_response = await client.request("list_tools", timeout=30)
        tools = tool_response.get("result") if tool_response.get("ok") is True else None
        if not isinstance(tools, list):
            raise ColabBrowserError("The official notebook bridge tools could not be read")
        observed = {tool.get("name"): tool for tool in tools if isinstance(tool, dict)}
        required = {
            "get_cells": {"includeOutputs": "boolean"},
            "add_code_cell": {"cellIndex": "integer", "language": "string", "code": "string"},
            "add_text_cell": {"cellIndex": "integer", "content": "string"},
        }
        for name, fields in required.items():
            properties = observed.get(name, {}).get("inputSchema", {}).get("properties", {})
            if any(properties.get(key, {}).get("type") != kind for key, kind in fields.items()):
                raise ManualColabAction("The official notebook editing interface changed; import the prepared notebook through Colab's Upload dialog")
        response = await client.request("call_tool", name="get_cells", arguments={"includeOutputs": False}, timeout=30)
        actual = mcp_notebook_cells(response)
        await self.ensure_mcp_registered_notebook(response)
        if mcp_cells_match(actual, expected):
            return True
        if len(actual) > 1 or actual and (actual[0][0] != "code" or actual[0][1].strip()):
            raise ManualColabAction("The official notebook bridge contains existing content; open a fresh empty notebook before importing this project")
        offset = len(actual)
        for index, (kind, source) in enumerate(expected, start=offset):
            name = "add_code_cell" if kind == "code" else "add_text_cell"
            arguments = {"cellIndex": index, "language": "python", "code": source} if kind == "code" else {"cellIndex": index, "content": source}
            response = await client.request("call_tool", name=name, arguments=arguments, timeout=30)
            result = response.get("result")
            if response.get("ok") is not True or not isinstance(result, dict) or result.get("isError") or result.get("is_error"):
                raise ColabBrowserError("Prepared notebook authoring did not complete; verify its source before running")
        actual = mcp_notebook_cells(await client.request("call_tool", name="get_cells", arguments={"includeOutputs": False}, timeout=30))
        if not mcp_cells_match(actual, expected):
            raise ColabBrowserError("The authored notebook does not match the prepared artifact; execution was stopped")
        return True

    async def open_notebook(self, job: Mapping[str, Any]) -> None:
        notebook_url = str(job.get("notebook_url") or "")
        browser_url = str(job.get("browser_url") or "")
        if not notebook_url and urlparse(browser_url).path.startswith("/drive/"):
            notebook_url = browser_url
        if notebook_url:
            location = self._official_mcp_location()
            if location is not None and urlparse(notebook_url).path.startswith("/drive/"):
                expected = prepared_notebook_cells(job)
                client = await self._connect_official_notebook_mcp(*location, expected_cells=expected, notebook_url=notebook_url)
                response = await client.request("call_tool", name="get_cells", arguments={"includeOutputs": False}, timeout=30)
                if not mcp_cells_match(mcp_notebook_cells(response), expected):
                    self._notebook_mcp = None
                    raise ManualColabAction("The saved notebook content does not match this prepared project; select the project's saved notebook before continuing")
                await self.ensure_mcp_registered_notebook(response)
                return
            await self.page.goto(validate_colab_url(notebook_url), wait_until="domcontentloaded")
            return
        if await self.author_prepared_notebook_with_mcp(job):
            return
        # Preserve a manually imported notebook when the user resumes this worker.
        path = urlparse(self.page.url).path
        if urlparse(self.page.url).hostname == "colab.research.google.com" and path.startswith("/drive/"):
            if job.get("project_id") and f"AutoResearch_{job['project_id']}.ipynb" in await self.page.title():
                return
        await self.page.goto(COLAB_URL, wait_until="domcontentloaded")
        state = classify_colab_page(self.page.url, await self.visible_text())
        if state:
            raise ManualColabAction(f"Colab requires attention: {state}")
        await self.import_prepared_notebook(job)

    async def import_prepared_notebook(self, job: Mapping[str, Any]) -> None:
        if not await self.open_upload_picker():
            raise ManualColabAction(f"Import the notebook using File > Open notebook > Upload: {job['notebook_path']}")
        inputs = []
        for frame in getattr(self.page, "frames", None) or [self.page]:
            if frame is getattr(self.page, "main_frame", self.page) or is_trusted_picker_frame(frame.url):
                inputs.extend(await frame.locator('input[type="file"]').all())
        if len(inputs) != 1:
            raise ManualColabAction(f"Select this notebook in the standard Upload dialog: {job['notebook_path']}")
        notebook = Path(str(job["notebook_path"])).resolve(strict=True)
        if notebook.suffix != ".ipynb":
            raise ColabBrowserError("Prepared Colab notebook must be an .ipynb file")
        await inputs[0].set_input_files(str(notebook))
        await self.page.wait_for_url("**/drive/**", timeout=self.options.timeout_ms)

    async def rendered_markdown(self) -> str:
        return await self.page.locator("body").evaluate(
            """body => {
                const parts = [];
                const walk = node => {
                    if (node.nodeType === Node.TEXT_NODE) {
                        const parent = node.parentElement;
                        if (parent && parent.getClientRects().length && getComputedStyle(parent).visibility !== 'hidden') parts.push(node.textContent);
                        return;
                    }
                    if (node.nodeType !== Node.ELEMENT_NODE) return;
                    if (node.matches('colab-code-cell, colab-output-area, .cell.code, .codecell, .inputarea, .input_area, .output_subarea, .output-area, .CodeMirror, .monaco-editor')) return;
                    for (const child of node.childNodes) walk(child);
                    if (node.shadowRoot) for (const child of node.shadowRoot.childNodes) walk(child);
                };
                walk(body);
                return parts.join(' ');
            }""", timeout=10_000
        )

    async def ensure_prepared_notebook(self, job: Mapping[str, Any]) -> None:
        expected = str(job.get("bundle_sha256") or "")
        if not re.fullmatch(r"[a-f0-9]{64}", expected):
            raise ColabBrowserError("The prepared job has no valid source checksum")
        if self._notebook_mcp is not None:
            response = await self._notebook_mcp.request("call_tool", name="get_cells", arguments={"includeOutputs": False}, timeout=30)
            actual = mcp_notebook_cells(response)
            if mcp_cells_match(actual, prepared_notebook_cells(job)):
                if not source_stamp_matches(await self.rendered_markdown(), expected):
                    raise ManualColabAction("This browser tab does not display the prepared notebook source checksum; reconnect the official bridge to this notebook before running")
                await self.ensure_mcp_registered_notebook(response)
                return
            raise ColabBrowserError("The official Colab notebook source changed after import; execution was stopped")
        if source_stamp_matches(await self.rendered_markdown(), expected):
            return
        if await self.author_prepared_notebook_with_mcp(job):
            return
        # Import this prepared artifact through the same normal picker; do not
        # execute an old saved Drive notebook that embeds an earlier source zip.
        await self.import_prepared_notebook(job)
        deadline = asyncio.get_running_loop().time() + self.options.timeout_ms / 1000
        while True:
            if source_stamp_matches(await self.rendered_markdown(), expected):
                return
            if classify_colab_page(self.page.url, await self.visible_text()) == "auth":
                raise ManualColabAction("Complete Google sign-in before importing the newly prepared notebook")
            if asyncio.get_running_loop().time() >= deadline:
                raise ManualColabAction(f"Import the newly prepared notebook; its visible source checksum must match this job: {job['notebook_path']}")
            await asyncio.sleep(0.25)

    async def _click_unique_main_control(self, labels: tuple[str, ...]) -> bool:
        """Click a unique observed control in this notebook's own main page."""
        allowed = {control_label(label) for label in labels}
        candidates = {(item['role'],item['name']) for item in await self._dom_controls(self.page)
                      if control_label(item['name']) in allowed}
        if len(candidates) > 1:
            raise ManualColabAction('The notebook save controls are ambiguous; save this prepared notebook to Drive manually')
        if not candidates:
            return False
        role,name = next(iter(candidates))
        visible = [item for item in await self.page.get_by_role(role,name=name,exact=True).all() if await item.is_visible()]
        if len(visible) != 1 or await visible[0].is_disabled():
            return False
        await visible[0].click()
        return True

    async def save_prepared_notebook(self, job: Mapping[str,Any]) -> str:
        """Save only our verified scratchpad, before requesting any compute.

        Existing Drive notebooks are never overwritten. If a runtime was
        started manually, leave every tab open so its GPU remains intact.
        """
        await self.ensure_prepared_notebook(job)
        parsed = urlparse(validate_colab_url(self.page.url))
        if re.fullmatch(r'/drive/[A-Za-z0-9_-]{1,200}',parsed.path):
            return urlunparse(parsed._replace(query='',fragment=''))
        if parsed.path != '/notebooks/empty.ipynb' or self._notebook_mcp is None:
            raise ManualColabAction('Save this prepared notebook to Drive before requesting a GPU')
        response = await self._notebook_mcp.request('call_tool',name='get_cells',arguments={'includeOutputs':True},timeout=30)
        if not mcp_cells_match(mcp_notebook_cells(response),prepared_notebook_cells(job)):
            raise ColabBrowserError('The official Colab notebook source changed after import; execution was stopped')
        await self.ensure_mcp_registered_notebook(response)
        if any(item['status'] in {'available','running'} for item in parse_status_markers(
                mcp_stream_output_text(response),str(job['project_id']))):
            raise ManualColabAction('This unsaved notebook already has a GPU; save its current session to Drive manually before continuing')
        original = self.page
        previous_pages = list(self.context.pages) if self.context else [original]
        copied = await self._click_unique_main_control(('Copy to Drive','复制到云端硬盘','复制到 Google 云端硬盘'))
        if not copied:
            if not await self._click_unique_main_control(('File','文件')):
                raise ManualColabAction('Save a copy of this prepared notebook to Drive using its File menu')
            copied = await self._click_unique_main_control(('Save a copy in Drive','Save a copy to Drive',
                '在云端硬盘中保存副本','在云端硬盘保存副本','将副本保存到云端硬盘','在云端硬盘中保存一份副本'))
        if not copied:
            raise ManualColabAction('Save a copy of this prepared notebook to Drive using its File menu')
        deadline = asyncio.get_running_loop().time() + self.options.timeout_ms/1000
        while True:
            pages = list(self.context.pages) if self.context else [original]
            candidates = []
            for page in pages:
                location = urlparse(page.url)
                if (page is original or page not in previous_pages) and location.hostname == 'colab.research.google.com' and re.fullmatch(r'/drive/[A-Za-z0-9_-]{1,200}',location.path):
                    candidates.append(page)
            if len(candidates) > 1:
                raise ManualColabAction('More than one saved notebook appeared; select the newly saved project copy manually')
            if candidates:
                saved = candidates[0]
                break
            if classify_colab_page(original.url,await self.visible_text()) == 'auth':
                raise ManualColabAction('Complete Google sign-in and save this prepared notebook to Drive')
            if asyncio.get_running_loop().time() >= deadline:
                raise ManualColabAction('The notebook copy has no Drive URL yet; finish saving it before continuing')
            await asyncio.sleep(0.25)
        if saved is not original:
            # Recheck our exact source and runtime immediately before closing
            # the scratchpad we just copied; other tabs remain untouched.
            await self.ensure_prepared_notebook(job)
            if any(item['status'] in {'available','running'} for item in parse_status_markers(
                    await self.output_text(),str(job['project_id']))):
                raise ManualColabAction('The original notebook acquired a GPU while saving; keep it open and save its active session manually')
            await original.close()
        self.page = saved
        self._notebook_mcp = None
        location = urlparse(validate_colab_url(saved.url))
        return urlunparse(location._replace(query='',fragment=''))

    async def upgrade_registered_notebook(self, old_job: Mapping[str,Any], new_job: Mapping[str,Any], public_url: str) -> str:
        """Explicitly replace only an exact, owned, autogenerated old artifact.

        Never called implicitly on unknown source. The caller supplies a retained
        old export, and binds the returned URL only after this method succeeds.
        """
        old_expected,new_expected=prepared_notebook_cells(old_job),prepared_notebook_cells(new_job)
        if (old_job.get('project_id')!=new_job.get('project_id') or len(old_expected)!=len(new_expected)
                or [kind for kind,_ in old_expected]!=[kind for kind,_ in new_expected]):
            raise ManualColabAction('Notebook upgrade requires the same project and an unchanged cell layout')
        parsed=urlparse(validate_colab_url(public_url))
        if parsed.fragment or not re.fullmatch(r'/drive/[A-Za-z0-9_-]{1,200}',parsed.path):
            raise ValueError('Notebook upgrade requires its public saved Drive URL')
        url=urlunparse(parsed._replace(query='',fragment=''))
        location=self._official_mcp_location()
        if location is None:
            raise ManualColabAction('Connect the official bridge to this saved notebook before upgrading it')
        if self._notebook_mcp is None or urlparse(self.page.url).path!=parsed.path:
            await self._connect_official_notebook_mcp(*location,expected_cells=old_expected,notebook_url=url)
        client=self._notebook_mcp
        response=await client.request('call_tool',name='get_cells',arguments={'includeOutputs':True},timeout=30)
        if not mcp_cells_match(mcp_notebook_cells(response),old_expected):
            raise ManualColabAction('The saved notebook differs from the retained old export; no source was updated')
        await self.ensure_mcp_registered_notebook(response)
        if not source_stamp_matches(await self.rendered_markdown(),str(old_job['bundle_sha256'])):
            raise ManualColabAction('This browser tab does not display the retained old source identity')
        active=any(item['status'] in {'available','running'} for item in parse_status_markers(
            mcp_stream_output_text(response),str(old_job['project_id'])))
        if active and not await self.has_disconnected_runtime_control():
            raise ManualColabAction('The old notebook may still own a GPU; finish its runtime before upgrading its source')
        cells=mcp_tool_result(response)['cells']
        offset=1 if cells and cells[0]['cell_type']=='code' and not mcp_notebook_cells(response)[0][1].strip() else 0
        current=list(old_expected)
        for index,((kind,old_source),(_,new_source)) in enumerate(zip(old_expected,new_expected)):
            if old_source==new_source:
                continue
            response=await client.request('call_tool',name='get_cells',arguments={'includeOutputs':False},timeout=30)
            if not mcp_cells_match(mcp_notebook_cells(response),current):
                raise ManualColabAction('The notebook changed during upgrade; remaining source was left untouched')
            await self.ensure_mcp_registered_notebook(response)
            cell_id=mcp_tool_result(response)['cells'][index+offset]['id']
            changed=await client.request('call_tool',name='update_cell',arguments={'cellId':cell_id,'content':new_source},timeout=30)
            result=changed.get('result')
            if changed.get('ok') is not True or not isinstance(result,dict) or result.get('isError') or result.get('is_error'):
                raise ManualColabAction('Notebook upgrade was interrupted; verify its source before running')
            current[index]=(kind,new_source)
        response=await client.request('call_tool',name='get_cells',arguments={'includeOutputs':False},timeout=30)
        if not mcp_cells_match(mcp_notebook_cells(response),new_expected):
            raise ManualColabAction('Notebook upgrade did not match the new prepared export; execution remains stopped')
        await self.ensure_mcp_registered_notebook(response)
        await self.page.keyboard.press('Control+s')
        return url

    async def authorization_continuation_snapshot(self, job: Mapping[str,Any]) -> list[dict[str,Any]]:
        await self.ensure_prepared_notebook(job)
        if self._notebook_mcp is None:
            raise ManualColabAction('Reconnect this saved notebook bridge to continue after Drive authorization')
        response = await self._notebook_mcp.request('call_tool',name='get_cells',arguments={'includeOutputs':True},timeout=30)
        if not mcp_cells_match(mcp_notebook_cells(response),prepared_notebook_cells(job)):
            raise ColabBrowserError('The official Colab notebook source changed after import; execution was stopped')
        await self.ensure_mcp_registered_notebook(response)
        return mcp_tool_result(response)['cells']

    async def run_continuation_cell(self, job: Mapping[str,Any], cell_id: str) -> None:
        cells = await self.authorization_continuation_snapshot(job)
        matches = [cell for cell in cells if cell.get('id')==cell_id and cell.get('cell_type')=='code']
        if len(matches)!=1 or 'RUNTIME_ID =' in ''.join(matches[0]['source']):
            raise ManualColabAction('Continuation cannot replay notebook runtime initialization')
        response = await self._notebook_mcp.request('call_tool',name='run_code_cell',arguments={'cellId':cell_id},timeout=7200)
        if response.get('ok') is not True:
            raise ColabBrowserError('The official Colab notebook request failed')

    async def select_free_gpu(self) -> None:
        if not await self.click_when_visible(("Runtime", "运行时", "代码执行程序")):
            raise ManualColabAction("Use Runtime > Change runtime type and select T4 GPU in the visible notebook")
        if not await self.click_when_visible(("Change runtime type", "更改运行时类型", "更改运行时类型…", "更改运行时类型...")):
            raise ManualColabAction("Open Change runtime type and select the free T4 GPU")
        if not await self.click_when_visible(("T4 GPU", "GPU"), timeout_seconds=3):
            if await self.click_observed(("Hardware accelerator", "硬件加速器", "Accelerator", "加速器")):
                if await self.click_when_visible(("T4 GPU", "GPU"), timeout_seconds=3):
                    if not await self.click_when_visible(("Save", "保存")):
                        raise ManualColabAction("Save the selected T4 GPU runtime type in Colab")
                    return
            selects = await self.page.locator("select").all()
            matched = []
            for select in selects:
                if not await select.is_visible():
                    continue
                options = await select.locator("option").all_text_contents()
                label = next((label for label in options if label.strip() in {"T4 GPU", "GPU"}), None)
                if label:
                    matched.append((select, label))
            if len(matched) != 1:
                raise ManualColabAction("Select T4 GPU manually in the runtime type dialog; paid GPU types are not selected")
            await matched[0][0].select_option(label=matched[0][1])
        if not await self.click_when_visible(("Save", "保存")):
            raise ManualColabAction("Save the selected T4 GPU runtime type in Colab")

    async def run_all(self) -> None:
        if not await self.click_when_visible(("Runtime", "运行时", "代码执行程序")):
            raise ManualColabAction("Use Runtime > Run all to start this prepared experiment notebook")
        if not await self.click_when_visible(("Run all", "Run all cells", "全部运行", "运行所有单元格")):
            raise ManualColabAction("Select Runtime > Run all in the visible notebook")

    async def has_disconnected_runtime_control(self) -> bool:
        labels = {control_label(value) for value in (
            "Connect", "Reconnect", "Disconnected", "Reconnect to runtime", "Connect to a hosted runtime",
            "Connect T4 Connect to a new runtime",
            "连接", "重新连接", "已断开连接", "连接到托管运行时", "连接到代码执行程序",
        )}
        matched = [
            item for item in await self.controls() if
            item["role"] in {"button", "combobox"} and control_label(re.sub(r'\s+',' ',item["name"])) in labels
        ]
        return len(matched) == 1

    async def upload_source_bundle(self, bundle_path: Path) -> bool:
        bundle_path = bundle_path.resolve(strict=True)
        if bundle_path.suffix != ".zip":
            raise ColabBrowserError("Prepared source bundle must be a .zip file")
        candidates = []
        for frame in self.page.frames:
            if frame == self.page.main_frame:
                continue
            host = urlparse(frame.url).hostname or ""
            if not (host.endswith(".googleusercontent.com") or host == "colab.research.google.com"):
                continue
            candidates.extend(await frame.locator('input[type="file"]').all())
        candidates = [field for field in candidates if not await field.is_disabled()]
        if len(candidates) != 1:
            return False
        await candidates[0].set_input_files(str(bundle_path))
        return True

    async def upload_data_file(self, path: Path) -> bool:
        """Send one already-verified data shard to Colab's active upload form."""
        path = Path(path)
        if not path.is_file() or path.stat().st_size > 512 * 1024 * 1024:
            raise ColabBrowserError('Approved data shard is missing or exceeds its upload limit')
        candidates = []
        for frame in self.page.frames:
            if frame == self.page.main_frame:
                continue
            host = urlparse(frame.url).hostname or ''
            if not (host.endswith('.googleusercontent.com') or host == 'colab.research.google.com'):
                continue
            for field in await frame.locator('input[type="file"]').all():
                if not await field.is_disabled():
                    candidates.append(field)
        if len(candidates) != 1:
            return False
        await candidates[0].set_input_files(str(path))
        return True


class ColabBrowserWorker:
    def __init__(self, client: ColabJobClient, browser: ColabBrowserSession) -> None:
        self.client = client
        self.browser = browser
        self.runtime_id: str | None = None
        self.attempt_started: datetime | None = None
        self.running_reported = False
        self.available_reported = False
        self.bundle_uploaded = False
        self.inputs_uploaded = False
        self._manual_reason = ""
        self._ignored_runtime_ids: set[str] = set()
        self._disconnect_seen_at: datetime | None = None
        self._last_ui_diagnostics: str = ""
        self._last_ui_diagnostics_at: datetime | None = None
        self._monitor_attach_pending = False
        self._recovering_runtime = False
        self._data_files_uploaded: set[tuple[str, str, str]] = set()
        self._authorization_continuation: dict[str,Any] | None = None
        self._authorization_resumed = False
        self._continuation_task: asyncio.Task[Any] | None = None
        self._input_submission: dict[str,Any] | None = None
        self._data_submissions: dict[tuple[str,str,str],dict[str,Any]] = {}

    async def _tick_authorization_continuation(self, job: Mapping[str,Any]) -> None:
        state = self._authorization_continuation
        if not state or not self._authorization_resumed or state.get('runtime_id') != self.runtime_id:
            return
        if self._continuation_task is not None:
            if not self._continuation_task.done():
                return
            try:
                self._continuation_task.result()
            except Exception as exc:
                self.log('Continuation dispatch finished with '+type(exc).__name__+'; verifying native cell state before any further action')
            self._continuation_task = None
        try:
            cells = await self.browser.authorization_continuation_snapshot(job)
            plan = plan_authorization_continuation(cells,prepared_notebook_cells(job),project_id=self.client.project_id,
                runtime_id=self.runtime_id,expected_started_at=state['started_at'],auth_interrupted=True,
                inflight_cell_id=state.get('inflight_cell_id'))
            if plan.action=='wait':
                return
            if plan.action=='blocked':
                self._authorization_resumed = False
                await self._manual_observation(plan.reason,preserve_runtime=True)
                return
            if plan.action=='complete':
                self._authorization_resumed = False
                return
            state['inflight_cell_id'] = plan.cell_id
            await self._observation('available','Continuing the verified notebook after Drive authorization',
                runtime_id=self.runtime_id,started_at=state['started_at'],runtime_active=True,
                authorization_continuation=dict(state))
            self._continuation_task = asyncio.create_task(self.browser.run_continuation_cell(job,plan.cell_id))
            self.log('Continuing one prepared notebook cell without restarting its GPU runtime')
        except (ManualColabAction,ColabBrowserError,ValueError) as exc:
            self._authorization_resumed = False
            await self._manual_observation('Reconnect the verified notebook and review its interrupted authorization step before resuming',preserve_runtime=True)

    async def _confirm_input_submission(self, job: Mapping[str,Any], outputs: str) -> None:
        pending = self._input_submission
        if not pending or pending['runtime_id'] != self.runtime_id:
            return
        for line in outputs.splitlines():
            if not line.strip().startswith(INPUTS_READY_PREFIX):
                continue
            try:
                record = json.loads(line.strip()[len(INPUTS_READY_PREFIX):])
            except (ValueError,TypeError):
                continue
            if (isinstance(record,dict) and record.get('project_id')==self.client.project_id
                    and record.get('runtime_id')==self.runtime_id and record.get('sha256')==pending['sha256']):
                self.inputs_uploaded = True
                self._input_submission = None
                self.log('Notebook confirmed the prepared input bundle identity and required inputs')
                return
        # Earlier saved notebooks have no explicit ACK. A subsequent same-runtime
        # data request proves their metadata upload/validation cell returned.
        if any(request['runtime_id']==self.runtime_id for request in parse_data_upload_requests(outputs,self.client.project_id)):
            self.inputs_uploaded = True
            self._input_submission = None
            self.log('Notebook advanced to its approved data request; input metadata is confirmed')
            return
        elapsed=(datetime.now(timezone.utc)-pending['submitted_at']).total_seconds()
        size=job.get('input_bundle_bytes')
        size=size if type(size) is int and size>0 else 0
        timeout=max(120,min(1800,size/(100*1024)+60))
        if elapsed>=timeout and not pending.get('paused'):
            pending['paused']=True
            await self._manual_observation('The input file was submitted but Colab has not confirmed receipt; inspect the active upload progress before resuming',preserve_runtime=True)

    async def _confirm_data_submissions(self, job: Mapping[str,Any], outputs: str) -> None:
        requests=[item for item in parse_data_upload_requests(outputs,self.client.project_id) if item['runtime_id']==self.runtime_id]
        latest=(self.runtime_id,requests[-1]['path'],requests[-1]['sha256']) if requests else None
        receipts=parse_data_upload_requests(outputs,self.client.project_id,prefix=DATA_READY_PREFIX)
        try:
            explicit=any(DATA_READY_PREFIX in source for _,source in prepared_notebook_cells(job))
        except (OSError,ValueError,ColabBrowserError):
            explicit=True
        for key,pending in list(self._data_submissions.items()):
            if key[0]!=self.runtime_id:
                self._data_submissions.pop(key,None)
                continue
            acknowledged=any(item['runtime_id']==key[0] and item['path']==key[1] and item['sha256']==key[2]
                             and item['size_bytes']==pending['size_bytes'] for item in receipts)
            advanced=not explicit and latest is not None and latest not in pending['seen_requests']
            if acknowledged or advanced:
                self._data_files_uploaded.add(key)
                self._data_submissions.pop(key,None)
                self.log('Notebook confirmed an approved data shard: '+key[1])
                continue
            elapsed=(datetime.now(timezone.utc)-pending['submitted_at']).total_seconds()
            timeout=max(120,min(7200,pending['size_bytes']/(100*1024)+60))
            if elapsed>=timeout and not pending.get('paused'):
                pending['paused']=True
                await self._manual_observation('A data shard was submitted but Colab has not confirmed its checksum; inspect upload progress before resuming',preserve_runtime=True)

    def log(self, message: str) -> None:
        print(f"{datetime.now(timezone.utc).isoformat()} {message}", flush=True)

    async def _observation(self, status: str, reason: str = "", *, retry_after_minutes: int | None = None, **details: Any) -> None:
        self.log(f"Colab observation: {status}: {reason}")
        try:
            details["browser_url"] = public_colab_url(self.browser.page.url)
        except ValueError:
            pass
        await self.client.observe(status, reason=reason, retry_after_minutes=retry_after_minutes, **details)

    async def _provider_observation(self, status: str, text: str) -> None:
        cooldown = provider_cooldown_minutes(text)
        if cooldown is not None:
            cooldown = max(30, cooldown)
        if status == "quota" and cooldown is None:
            # Quotas are different from a temporarily busy GPU pool. A long
            # default pause avoids repeated requests against an explicit limit.
            cooldown = 12 * 60
        details: dict[str, Any] = {}
        if status in {"quota", "unavailable", "network"}:
            # The caller supplies visible_text(), which excludes notebook
            # source and historical outputs. Keep the provider's actual UI
            # wording without private authentication or account details.
            details["provider_ui_text"] = sanitize_provider_ui_text(text)
        if status == "auth":
            details.update(runtime_active=self.runtime_id is not None, runtime_id=self.runtime_id)
        await self._observation(status, "Colab displayed a provider or authentication message", retry_after_minutes=cooldown, **details)

    async def _manual_observation(self, reason: str, *, preserve_runtime: bool = False) -> None:
        try:
            diagnostics = await self.browser.ui_diagnostics()
        except Exception as exc:
            if is_browser_closed_error(exc):
                raise
            diagnostics = {"ui_controls": [], "ui_role_tree": "", "ui_frames": []}
        parsed = urlparse(self.browser.page.url)
        diagnostics.update(ui_host=parsed.hostname, ui_path=parsed.path)
        if preserve_runtime and self.runtime_id is not None and self.attempt_started is not None:
            diagnostics.update(runtime_active=True, runtime_id=self.runtime_id, started_at=self.attempt_started.isoformat())
        self._last_ui_diagnostics = json.dumps(diagnostics, sort_keys=True, ensure_ascii=False)
        self._last_ui_diagnostics_at = datetime.now(timezone.utc)
        self.log("Visible Colab controls: " + json.dumps(diagnostics, ensure_ascii=False))
        await self._observation("auth", reason, manual_action=True, **diagnostics)

    async def _ensure_runtime_monitor_attachment(self, job: Mapping[str, Any]) -> bool:
        if hasattr(self.browser, "_notebook_mcp") and self.browser._official_mcp_location() is not None:
            bridge = self.browser._notebook_mcp
            if bridge is None or not bridge.status().get("connected"):
                self._monitor_attach_pending = True
        if not self._monitor_attach_pending:
            return True
        try:
            # open_notebook attaches a saved project's bridge read-only. This
            # path never selects hardware or starts notebook cells again.
            await self.browser.open_notebook(job)
        except ManualColabAction as exc:
            await self._manual_observation(str(exc), preserve_runtime=True)
            return False
        except Exception as exc:
            if is_browser_closed_error(exc):
                raise
            details = {"runtime_active": True, "runtime_id": self.runtime_id}
            if self.attempt_started is not None:
                details["started_at"] = self.attempt_started.isoformat()
            await self._observation("network", safe_local_error_reason(exc, "Could not reopen the notebook for monitoring"), **details)
            return False
        self._monitor_attach_pending = False
        return True

    async def _refresh_import_diagnostics(self, job: Mapping[str, Any]) -> None:
        details = ((job.get("last_observation") or {}).get("details") or {})
        reason = str(details.get("reason") or "")
        if not details.get("manual_action") or not reason.startswith(("Import the notebook", "Select this notebook", "Import the newly prepared notebook")):
            return
        now = datetime.now(timezone.utc)
        if self._last_ui_diagnostics_at and (now - self._last_ui_diagnostics_at).total_seconds() < 30:
            return
        self._last_ui_diagnostics_at = now
        diagnostics = await self.browser.ui_diagnostics()
        parsed = urlparse(self.browser.page.url)
        diagnostics.update(ui_host=parsed.hostname, ui_path=parsed.path)
        signature = json.dumps(diagnostics, sort_keys=True, ensure_ascii=False)
        if signature != self._last_ui_diagnostics:
            self._last_ui_diagnostics = signature
            self.log("Visible Colab controls changed: " + signature)
            await self._observation("auth", reason, manual_action=True, **diagnostics)

    async def _adopt_manual_runtime(self) -> None:
        if self.runtime_id is not None or self.attempt_started is None:
            return
        for item in parse_status_markers(await self.browser.output_text(), self.client.project_id):
            if item["status"] == "starting" and item["runtime_id"] not in self._ignored_runtime_ids and marker_is_current(item, self.attempt_started):
                self.runtime_id = item["runtime_id"]
                self.log("Observed a manually started instance of this prepared notebook")
                return

    async def _resume_after_google_login(self, job: Mapping[str, Any]) -> bool:
        observation = job.get("last_observation") or {}
        details = observation.get("details") or {}
        if (
            job.get("status") != "auth_required"
            or observation.get("status") != "auth"
            or details.get("reason") != GOOGLE_SIGN_IN_REASON
            or details.get("manual_action") is not True
            or urlparse(self.browser.page.url).hostname != "colab.research.google.com"
        ):
            return False
        try:
            if classify_colab_page(self.browser.page.url, await self.browser.visible_text()) is not None:
                return False
            controls = await self.browser.controls()
        except Exception as exc:
            if is_browser_closed_error(exc):
                raise
            # A loading page is not evidence of completed authentication.
            return False
        names = {control_label(item["name"]) for item in controls}
        if names.intersection({"sign in", "登录", "登入"}):
            return False
        normal_controls = {"file", "runtime", "new notebook", "open notebook", "open file picker", "upload", "文件", "运行时", "新建笔记本", "打开文件选择器", "打开笔记本", "上传"}
        if not names.intersection(normal_controls):
            return False
        if not await self.client.resume():
            return False
        self.log("Google sign-in completed in this dedicated browser; resuming the prepared job")
        self._manual_reason = ""
        return True

    async def _record_closed_browser(self) -> None:
        latest = await self.client.status()
        if latest.get("status") in {"completed", "failed", "cancelled"}:
            self.log(f"Worker stopped: job is {latest['status']}")
            return
        if latest.get("status") == "waiting_for_gpu":
            try:
                retry_at = datetime.fromisoformat(str(latest.get("next_attempt_at") or "").replace("Z", "+00:00"))
                cooling_down = retry_at.tzinfo is not None and retry_at > datetime.now(timezone.utc)
            except (ValueError, TypeError):
                cooling_down = False
            if cooling_down:
                self.log("Worker stopped after browser closure; the saved GPU retry deadline and provider observation remain unchanged")
                return
        try:
            await self._observation(
                "network", "The dedicated Colab browser was closed; the saved retry queue and Drive checkpoints remain available",
                browser_closed=True, runtime_id=self.runtime_id,
                runtime_active=self.runtime_id is not None and self.attempt_started is not None,
                started_at=self.attempt_started.isoformat() if self.attempt_started is not None else None,
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 409:
                raise
            latest = await self.client.status()
            if latest.get("status") not in {"completed", "failed", "cancelled"}:
                raise
        self.log("Worker stopped after browser closure")

    def _page_is_closed(self) -> bool:
        checker = getattr(self.browser.page, "is_closed", None)
        return bool(checker()) if callable(checker) else False

    async def run(self) -> None:
        try:
            await self._run()
        except Exception as exc:
            if not is_browser_closed_error(exc):
                raise
            await self._record_closed_browser()
        finally:
            if self._continuation_task is not None:
                self._continuation_task.cancel()
                await asyncio.gather(self._continuation_task,return_exceptions=True)

    async def _run(self) -> None:
        initial = await self.client.status()
        self._authorization_continuation = initial.get('authorization_continuation')
        self._authorization_resumed = bool(self._authorization_continuation and initial.get('status')!='auth_required')
        previous_auth = (initial.get("last_observation") or {}).get("details") or {}
        recovery = initial.get("recovery") or {}
        recovering = (recovery.get("mode") == "attach" or
                      (initial.get("status") == "waiting_for_connection" and previous_auth.get("runtime_active") is True))
        active_id = recovery.get("runtime_id") or initial.get("runtime_id") or previous_auth.get("runtime_id")
        if active_id and (recovering or initial.get("status") in {"running", "gpu_available"} or previous_auth.get("runtime_active") is True):
            self.runtime_id = str(active_id)
            try:
                preserved_started_at = recovery.get("started_at") if recovering else None
                preserved_started_at = preserved_started_at or (previous_auth.get("started_at") if previous_auth.get("runtime_id") == self.runtime_id else None)
                self.attempt_started = datetime.fromisoformat(str(preserved_started_at or initial["last_attempt_at"]).replace("Z", "+00:00"))
                if self.attempt_started.tzinfo is None:
                    raise ValueError("Runtime start timestamp must have a timezone")
            except (KeyError, ValueError, TypeError):
                self.attempt_started = datetime.now(timezone.utc)
            self.running_reported = initial.get("status") == "running" or recovery.get("previous_status") == "running"
            self.available_reported = True
            self._recovering_runtime = recovering
            self._monitor_attach_pending = True
            await self._ensure_runtime_monitor_attachment(initial)
        while True:
            job = await self.client.status()
            status = job.get("status")
            if status in {"completed", "failed", "cancelled"}:
                self.log(f"Worker stopped: job is {status}")
                return
            if self._page_is_closed():
                await self._record_closed_browser()
                return
            if status == "auth_required":
                # The UI's explicit Resume action re-enables the next claim.
                if not self._manual_reason:
                    if self.browser.page.url in {"about:blank", ""}:
                        await self.browser.page.goto(
                            validate_colab_url(str(job.get("notebook_url") or COLAB_URL)),
                            wait_until="domcontentloaded",
                        )
                    pause_reason = ((job.get("last_observation") or {}).get("details") or {}).get("reason")
                    if pause_reason == GOOGLE_SIGN_IN_REASON:
                        self.log("Complete Google sign-in in the visible browser; the worker will continue after this page returns to Colab")
                    else:
                        self.log("Complete Drive permission or the indicated manual action in the visible browser, then Resume in AutoResearch")
                    self._manual_reason = "paused"
                await self._adopt_manual_runtime()
                if await self._resume_after_google_login(job):
                    continue
                await self._refresh_import_diagnostics(job)
                await asyncio.sleep(self.browser.options.poll_seconds)
                continue
            self._manual_reason = ""
            if status=='waiting_for_connection':
                # An allocation dialog can appear after the API attempt lease
                # expired. Observe it read-only while waiting; never claim a
                # new runtime just to discover that the old request hit quota.
                late_text=await self.browser.visible_text()
                late_status=classify_colab_page(self.browser.page.url,late_text)
                if late_status in {'quota','auth'}:
                    await self._provider_observation(late_status,late_text)
                    await asyncio.sleep(self.browser.options.poll_seconds)
                    continue
            if job.get("can_attempt"):
                if not await self.client.claim():
                    continue
                await self._adopt_manual_runtime()
                # A Drive permission pause can leave this notebook executing
                # after the user grants access. Resume monitoring the known
                # runtime rather than interrupting it with another Run all.
                if self.runtime_id and self.attempt_started is not None:
                    if not await self._ensure_runtime_monitor_attachment(job):
                        continue
                    self.log("Resuming observation of the existing notebook runtime")
                    self.available_reported = False
                    self._authorization_resumed = self._authorization_continuation is not None
                    await self._monitor(job)
                    await asyncio.sleep(self.browser.options.poll_seconds)
                    continue
                self.attempt_started = datetime.now(timezone.utc)
                self.runtime_id = None
                self.running_reported = False
                self.available_reported = False
                self.bundle_uploaded = False
                self.inputs_uploaded = False
                self._input_submission = None
                self._data_submissions.clear()
                self._data_files_uploaded.clear()
                self._authorization_continuation = None
                self._authorization_resumed = False
                if self._continuation_task is not None:
                    self._continuation_task.cancel()
                    await asyncio.gather(self._continuation_task,return_exceptions=True)
                    self._continuation_task = None
                self._disconnect_seen_at = None
                try:
                    await self.browser.open_notebook(job)
                    text = await self.browser.visible_text()
                    page_status = classify_colab_page(self.browser.page.url, text)
                    if page_status:
                        await self._provider_observation(page_status, text)
                        continue
                    await self.browser.ensure_prepared_notebook(job)
                    saved_url = await self.browser.save_prepared_notebook(job)
                    await self.client.bind_notebook(notebook_url=saved_url,bundle_sha256=str(job['bundle_sha256']))
                    job = {**job,'notebook_url':saved_url,'browser_url':saved_url}
                    # A Drive copy is a different notebook. Reconnect and verify
                    # its exact source/native cell ownership before allocation.
                    await self.browser.open_notebook(job)
                    await self.browser.ensure_prepared_notebook(job)
                    previous_output = await self.browser.output_text()
                    self._ignored_runtime_ids = {
                        item["runtime_id"] for item in parse_status_markers(previous_output, self.client.project_id)
                    }
                    await self.browser.select_free_gpu()
                    await self.browser.run_all()
                    self.log("Requested Run all through the standard Colab interface; waiting for actual notebook output")
                except ManualColabAction as exc:
                    await self._manual_observation(str(exc))
                    continue
                except Exception as exc:
                    if is_browser_closed_error(exc):
                        raise
                    await self._observation("network", safe_local_error_reason(exc, "Could not load or operate the Colab page"))
                    continue
            if self.attempt_started is not None and status not in {"waiting_for_gpu", "waiting_for_connection", "quota_exceeded"}:
                try:
                    if not await self._ensure_runtime_monitor_attachment(job):
                        continue
                    await self._monitor(job)
                except Exception as exc:
                    if is_browser_closed_error(exc):
                        raise
                    details = {}
                    if self.runtime_id is not None and self.attempt_started is not None:
                        details = dict(runtime_active=True, runtime_id=self.runtime_id,
                                       started_at=self.attempt_started.isoformat())
                        self._monitor_attach_pending = True
                        self._recovering_runtime = True
                    else:
                        self.attempt_started = None
                    await self._observation("network", f"Colab output could not be read ({type(exc).__name__})", **details)
            await asyncio.sleep(self.browser.options.poll_seconds)

    async def _monitor(self, job: Mapping[str, Any]) -> None:
        await self._tick_authorization_continuation(job)
        text = await self.browser.visible_text()
        outputs = await self.browser.output_text()
        await self._confirm_input_submission(job,outputs)
        await self._confirm_data_submissions(job,outputs)
        markers = [item for item in parse_status_markers(outputs, self.client.project_id)
                   if item["runtime_id"] not in self._ignored_runtime_ids]
        if self._recovering_runtime:
            # Historical output alone is not proof that the remote runtime
            # survived a local crash. Reattach without running any cell, and
            # wait for a stable disconnect control before requesting compute.
            terminal = any(item['runtime_id'] == self.runtime_id and item['status'] in {'completed', 'failed', 'disconnected'}
                           and self.attempt_started is not None and marker_is_current(item, self.attempt_started)
                           for item in markers)
            if not terminal and await self.browser.has_disconnected_runtime_control():
                now = datetime.now(timezone.utc)
                if self._disconnect_seen_at is None:
                    self._disconnect_seen_at = now
                elif (now - self._disconnect_seen_at).total_seconds() >= 60:
                    await self._observation('disconnected', 'The recovered notebook confirms its previous runtime is disconnected', runtime_id=self.runtime_id)
                    self.runtime_id = None
                    self.attempt_started = None
                    self._recovering_runtime = False
                return
            self._disconnect_seen_at = None
            # Current notebook connection or a matching terminal result has
            # now been observed. Republish its actual markers to clear recovery.
            self.available_reported = False
            self.running_reported = self.running_reported and terminal
            self._recovering_runtime = False
        # Mount can open an OAuth dialog before the next poll. Establish this
        # runtime before pausing so Resume will not interrupt its active cells.
        for marker in markers:
            if marker["status"] == "starting" and self.runtime_id is None and self.attempt_started is not None and marker_is_current(marker, self.attempt_started):
                self.runtime_id = marker["runtime_id"]
            if (
                marker["status"] == "available" and not self.available_reported
                and marker["runtime_id"] == self.runtime_id
                and self.attempt_started is not None and marker_is_current(marker, self.attempt_started)
            ):
                marker_details = {key: value for key, value in marker.items() if key not in {"status", "reason", "project_id"}}
                await self._observation("available", "Notebook verified a usable CUDA GPU", **marker_details)
                self.available_reported = True
        state = classify_colab_page(self.browser.page.url, text)
        if state:
            await self._provider_observation(state, text)
            if state != "auth":
                self.attempt_started = None
            return
        # This notebook is generated locally and its execution was authorized.
        await self.browser.click_observed(("Run anyway", "仍然运行"))
        for marker in markers:
            if self.attempt_started is None or not marker_is_current(marker, self.attempt_started):
                continue
            marker_status = marker["status"]
            marker_details = {k: v for k, v in marker.items() if k not in {"status", "reason", "project_id"}}
            if marker_status == "starting" and self.runtime_id is None:
                self.runtime_id = marker["runtime_id"]
            if marker["runtime_id"] != self.runtime_id:
                continue
            if marker_status == "available" and not self.available_reported:
                await self._observation("available", "Notebook verified a usable CUDA GPU", **marker_details)
                self.available_reported = True
            if marker_status == "auth":
                if (self._authorization_resumed and self._authorization_continuation
                        and self._authorization_continuation.get('runtime_id')==self.runtime_id):
                    continue
                reason = "Required experiment inputs are missing in Drive" if marker.get("reason") == "missing_inputs" else "Notebook requires manual authorization or inputs"
                self._authorization_continuation = {'runtime_id':self.runtime_id,'started_at':marker['started_at'],'inflight_cell_id':None}
                self._authorization_resumed = False
                await self._observation("auth", reason, runtime_active=True,
                    authorization_continuation=dict(self._authorization_continuation), **marker_details)
                return
            if marker_status == "unavailable":
                await self._observation("unavailable", "Notebook GPU check reported no usable CUDA GPU", **marker_details)
                self.attempt_started = None
                return
            if marker_status == "disconnected":
                await self._observation("disconnected", "Experiment saved its checkpoint at the configured session budget", **marker_details)
                self.attempt_started = None
                self.runtime_id = None
                return
            if marker_status == "running" and not self.running_reported:
                self.inputs_uploaded = True
                self._input_submission = None
                self._data_files_uploaded.update(self._data_submissions)
                self._data_submissions.clear()
                await self._observation("running", "The experiment subprocess started", **marker_details)
                self.running_reported = True
            if marker_status == "failed":
                await self._observation("failed", "Notebook execution failed; inspect Drive results and saved checkpoints", **marker_details)
                self.attempt_started = None
                return
            if marker_status == "completed":
                if marker.get("returncode") != 0 or not self.running_reported:
                    await self._observation("failed", "Invalid completion marker; successful execution was not established", **marker_details)
                else:
                    await self._observation("completed", "Notebook reported experiment exit code 0", **marker_details)
                self.attempt_started = None
                return
        if self.runtime_id and DATA_UPLOAD_PREFIX in outputs:
            from .colab_data import data_upload_path

            current_requests=[item for item in parse_data_upload_requests(outputs,self.client.project_id)
                              if item['runtime_id']==self.runtime_id]
            for request in current_requests[-1:]:
                if request['runtime_id'] != self.runtime_id:
                    continue
                key = (self.runtime_id, request['path'], request['sha256'])
                if key in self._data_files_uploaded or key in self._data_submissions:
                    continue
                try:
                    path = data_upload_path({**job, 'runtime_id': self.runtime_id}, request)
                    uploaded = await self.browser.upload_data_file(path)
                except (OSError, ValueError) as exc:
                    await self._observation('failed', 'The requested data shard failed its approved input integrity check', error_type=type(exc).__name__)
                    return
                if not uploaded:
                    await self._manual_observation('Select the approved data shard in this notebook file picker: ' + request['path'], preserve_runtime=True)
                    return
                self._data_submissions[key]={'size_bytes':request['size_bytes'],'submitted_at':datetime.now(timezone.utc),
                    'seen_requests':{(self.runtime_id,item['path'],item['sha256']) for item in current_requests}}
                self.log('Submitted an approved data shard; waiting for notebook checksum confirmation: ' + request['path'])
                return
        if self.runtime_id and UPLOAD_PREFIX in outputs and not self.bundle_uploaded:
            bundle = job.get("bundle_path")
            if not bundle or not await self.browser.upload_source_bundle(Path(str(bundle))):
                await self._observation("auth", f"Choose the prepared source bundle in Colab's file upload: {bundle}", manual_action=True)
                return
            self.bundle_uploaded = True
            self.log("Submitted the prepared source bundle to the active notebook file picker")
        if self.runtime_id and not self.inputs_uploaded and self._input_submission is None:
            for line in outputs.splitlines():
                if not line.strip().startswith(INPUTS_PREFIX):
                    continue
                try:
                    request = json.loads(line.strip()[len(INPUTS_PREFIX):])
                except (ValueError, TypeError):
                    continue
                if request.get("project_id") != self.client.project_id or request.get("runtime_id") != self.runtime_id:
                    continue
                bundle = job.get("input_bundle_path")
                if not bundle or not await self.browser.upload_source_bundle(Path(str(bundle))):
                    await self._observation("auth", f"Choose the prepared real-input bundle in Colab's file upload: {bundle}", manual_action=True)
                    return
                self._input_submission = {'runtime_id':self.runtime_id,'sha256':str(job.get('input_bundle_sha256') or ''),
                                          'submitted_at':datetime.now(timezone.utc)}
                self.log("Submitted the prepared input bundle; waiting for notebook receipt confirmation")
                break
        if self.runtime_id and (self.available_reported or self.running_reported):
            if await self.browser.has_disconnected_runtime_control():
                now = datetime.now(timezone.utc)
                if self._disconnect_seen_at is None:
                    self._disconnect_seen_at = now
                elif (now - self._disconnect_seen_at).total_seconds() >= 60:
                    await self._observation("disconnected", "The known active runtime consistently displays its reconnect control", runtime_id=self.runtime_id)
                    self.attempt_started = None
                    self.runtime_id = None
            else:
                self._disconnect_seen_at = None
