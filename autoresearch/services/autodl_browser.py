from __future__ import annotations

import asyncio
import inspect
import math
import os
import random
import shutil
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence, TypeVar
from urllib.parse import urljoin, urlparse


AUTODL_LOGIN_URL = "https://www.autodl.com/login"
AUTODL_INSTANCE_LIST_URL = "https://www.autodl.com/console/instance/list"
AUTODL_INSTANCE_MARKET_URL = "https://www.autodl.com/market/list"
_TRUSTED_AUTODL_HOSTS = frozenset({"autodl.com", "www.autodl.com", "api.autodl.com"})


class AutoDLBrowserError(RuntimeError):
    """Base error for the optional AutoDL browser adapter."""


class BrowserDependencyError(AutoDLBrowserError):
    """Raised when Playwright, Pillow, or a supported browser is unavailable."""


class AutoDLLoginError(AutoDLBrowserError):
    """Raised when the browser login workflow cannot reach an authenticated page."""


class PuzzleSolveError(AutoDLBrowserError):
    """Raised when an Aliyun puzzle challenge cannot be solved safely."""


@dataclass(slots=True, repr=False)
class AutoDLCredentials:
    """Credentials kept only in the caller's process memory."""

    phone: str
    password: str = field(repr=False)

    def __post_init__(self) -> None:
        self.phone = self.phone.strip()
        if not self.phone:
            raise ValueError("An AutoDL phone number is required")
        if not self.password:
            raise ValueError("An AutoDL password is required")


@dataclass(frozen=True, slots=True)
class BrowserLaunchOptions:
    browser: str = "auto"
    executable_path: Path | None = None
    headless: bool = False
    slow_mo_ms: int = 0
    timeout_ms: int = 30_000
    viewport_width: int = 1280
    viewport_height: int = 900

    def __post_init__(self) -> None:
        if self.browser not in {"auto", "chrome", "edge"}:
            raise ValueError("browser must be one of: auto, chrome, edge")
        if self.timeout_ms < 1_000:
            raise ValueError("timeout_ms must be at least 1000")
        if self.viewport_width < 800 or self.viewport_height < 600:
            raise ValueError("The browser viewport is too small for the AutoDL console")


@dataclass(frozen=True, slots=True)
class AutoDLBrowserSelectors:
    """Stable attributes from the AutoDL login and Aliyun CAPTCHA components."""

    phone_inputs: tuple[str, ...] = (
        "input[name='phone']",
        "input[autocomplete='tel']",
    )
    password_inputs: tuple[str, ...] = (
        "input[name='password']",
        "input[autocomplete='current-password']",
        "input[type='password']",
    )
    otp_inputs: tuple[str, ...] = (
        ".verify input[autocomplete='one-time-code']",
        ".verify input:not([name='phone']):not([type='password'])",
    )
    submit_buttons: tuple[str, ...] = (
        ".content > button.el-button--primary",
        ".content button.el-button--primary",
        "form button.el-button--primary",
    )
    send_code_buttons: tuple[str, ...] = (
        ".verify button:not([disabled])",
        ".verify .el-button:not([disabled])",
    )
    captcha_modal: str = "#aliyunCaptcha-window-popup"
    captcha_background: str = "#aliyunCaptcha-img"
    captcha_piece: str = "#aliyunCaptcha-puzzle"
    captcha_image_box: str = "#aliyunCaptcha-img-box"
    captcha_track: str = "#aliyunCaptcha-sliding-body"
    captcha_thumb: str = "#aliyunCaptcha-sliding-slider"
    captcha_refresh: str = "#aliyunCaptcha-btn-refresh"


@dataclass(frozen=True, slots=True)
class PuzzleMatch:
    offset: float
    score: float
    confidence: float
    background_width: int
    piece_width: int


@dataclass(frozen=True, slots=True)
class MouseStep:
    x: float
    y: float
    delay_seconds: float


@dataclass(frozen=True, slots=True)
class LoginResult:
    status: str
    url: str


OTPProvider = Callable[[], str | Awaitable[str]]
T = TypeVar("T")
AuthenticatedHook = Callable[["AutoDLBrowserSession"], T | Awaitable[T]]
PixelMatrix = Sequence[Sequence[int]]


def discover_browser_executable(browser: str = "auto") -> Path | None:
    """Find an installed Chrome or Edge executable without a bundled browser."""

    if browser not in {"auto", "chrome", "edge"}:
        raise ValueError("browser must be one of: auto, chrome, edge")

    order = ("chrome", "edge") if browser == "auto" else (browser,)
    command_names = {
        "chrome": ("chrome", "chrome.exe", "google-chrome", "google-chrome-stable"),
        "edge": ("msedge", "msedge.exe", "microsoft-edge"),
    }
    relative_paths = {
        "chrome": Path("Google/Chrome/Application/chrome.exe"),
        "edge": Path("Microsoft/Edge/Application/msedge.exe"),
    }

    for name in order:
        for command in command_names[name]:
            found = shutil.which(command)
            if found:
                return Path(found).resolve()

        if os.name == "nt":
            roots = (
                os.environ.get("PROGRAMFILES"),
                os.environ.get("PROGRAMFILES(X86)"),
                os.environ.get("LOCALAPPDATA"),
            )
            for root in roots:
                if not root:
                    continue
                candidate = Path(root) / relative_paths[name]
                if candidate.is_file():
                    return candidate.resolve()
    return None


def _matrix_size(name: str, matrix: PixelMatrix) -> tuple[int, int]:
    height = len(matrix)
    if height == 0:
        raise ValueError(f"{name} is empty")
    width = len(matrix[0])
    if width == 0 or any(len(row) != width for row in matrix):
        raise ValueError(f"{name} must be a nonempty rectangular matrix")
    return width, height


def _alpha_contour(alpha: PixelMatrix, threshold: int = 32) -> list[tuple[int, int]]:
    width, height = _matrix_size("piece_alpha", alpha)
    active = [[value >= threshold for value in row] for row in alpha]
    if not any(any(row) for row in active):
        raise ValueError("piece_alpha contains no visible puzzle shape")

    contour: list[tuple[int, int]] = []
    for y in range(height):
        for x in range(width):
            if not active[y][x]:
                continue
            if (
                x == 0
                or y == 0
                or x == width - 1
                or y == height - 1
                or not active[y][x - 1]
                or not active[y][x + 1]
                or not active[y - 1][x]
                or not active[y + 1][x]
            ):
                contour.append((x, y))
    if len(contour) < 8:
        raise ValueError("piece_alpha does not contain a usable puzzle contour")
    return contour


def _gradient(gray: PixelMatrix, x: int, y: int) -> float:
    width = len(gray[0])
    height = len(gray)
    left = gray[y][max(0, x - 1)]
    right = gray[y][min(width - 1, x + 1)]
    top = gray[max(0, y - 1)][x]
    bottom = gray[min(height - 1, y + 1)][x]
    return math.hypot(float(right - left), float(bottom - top))


def solve_puzzle_gap(
    background_gray: PixelMatrix,
    piece_alpha: PixelMatrix,
    *,
    search_start: int = 0,
    local_radius: int = 2,
) -> PuzzleMatch:
    """Estimate the horizontal puzzle offset from the piece contour and image edges."""

    background_width, background_height = _matrix_size("background_gray", background_gray)
    piece_width, piece_height = _matrix_size("piece_alpha", piece_alpha)
    if piece_height != background_height:
        raise ValueError("The background and piece images must have the same height")
    if piece_width >= background_width:
        raise ValueError("The puzzle piece must be narrower than the background")
    if local_radius < 0 or local_radius > 8:
        raise ValueError("local_radius must be between 0 and 8")

    contour = _alpha_contour(piece_alpha)
    maximum_offset = background_width - piece_width
    first_offset = max(0, min(search_start, maximum_offset))
    scores: list[tuple[int, float]] = []

    for offset in range(first_offset, maximum_offset + 1):
        edge_strengths: list[float] = []
        for piece_x, y in contour:
            background_x = offset + piece_x
            local = max(
                _gradient(background_gray, candidate_x, y)
                for candidate_x in range(
                    max(0, background_x - local_radius),
                    min(background_width - 1, background_x + local_radius) + 1,
                )
            )
            edge_strengths.append(local)

        edge_strengths.sort(reverse=True)
        retained = edge_strengths[: max(8, int(len(edge_strengths) * 0.72))]
        score = sum(retained) / len(retained)
        scores.append((offset, score))

    best_index = max(range(len(scores)), key=lambda index: scores[index][1])
    best_offset, best_score = scores[best_index]

    refined_offset = float(best_offset)
    if 0 < best_index < len(scores) - 1:
        left_score = scores[best_index - 1][1]
        right_score = scores[best_index + 1][1]
        denominator = left_score - 2.0 * best_score + right_score
        if abs(denominator) > 1e-9:
            refined_offset += max(-0.5, min(0.5, 0.5 * (left_score - right_score) / denominator))

    exclusion_radius = max(3, piece_width // 8)
    alternatives = [
        score
        for offset, score in scores
        if abs(offset - best_offset) > exclusion_radius
    ]
    runner_up = max(alternatives, default=0.0)
    confidence = best_score / max(runner_up, 1e-9)
    return PuzzleMatch(
        offset=refined_offset,
        score=best_score,
        confidence=confidence,
        background_width=background_width,
        piece_width=piece_width,
    )


def slider_distance_for_gap(
    match: PuzzleMatch,
    *,
    rendered_background_width: float,
    rendered_piece_width: float,
    track_width: float,
    thumb_width: float,
) -> float:
    """Convert the source-image gap offset to its rendered horizontal offset."""

    values = (
        rendered_background_width,
        rendered_piece_width,
        track_width,
        thumb_width,
    )
    if any(value <= 0 for value in values):
        raise ValueError("Rendered CAPTCHA dimensions must be positive")
    slider_travel = track_width - thumb_width
    source_travel = match.background_width - match.piece_width
    if rendered_background_width <= rendered_piece_width or slider_travel <= 0 or source_travel <= 0:
        raise ValueError("CAPTCHA geometry has no usable horizontal travel")

    rendered_offset = match.offset * rendered_background_width / match.background_width
    return max(0.0, min(slider_travel, rendered_offset))


def next_slider_distance(
    current_distance: float,
    *,
    desired_piece_left: float,
    actual_piece_left: float,
    maximum_distance: float,
    tolerance: float = 1.15,
) -> float | None:
    """Return the next pointer distance for closed-loop puzzle correction."""

    if maximum_distance <= 0 or tolerance <= 0:
        raise ValueError("Slider correction geometry and tolerance must be positive")
    error = desired_piece_left - actual_piece_left
    if abs(error) <= tolerance:
        return None
    return max(0.0, min(maximum_distance, current_distance + error))


def generate_mouse_trajectory(
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    rng: random.Random | None = None,
    steps: int = 36,
) -> list[MouseStep]:
    """Create a curved path with acceleration, deceleration, and a small correction."""

    if steps < 12:
        raise ValueError("steps must be at least 12")
    generator = rng or random.Random()
    start_x, start_y = start
    end_x, end_y = end
    distance = math.hypot(end_x - start_x, end_y - start_y)
    if distance < 1.0:
        raise ValueError("Mouse trajectory distance must be at least one pixel")

    direction = 1.0 if end_x >= start_x else -1.0
    overshoot = direction * min(max(distance * 0.012, 1.4), 3.6)
    curve = generator.choice((-1.0, 1.0)) * generator.uniform(1.8, 4.8)
    main_end_x = end_x + overshoot
    main_end_y = end_y + generator.uniform(-0.6, 0.6)
    main_steps = steps - 3
    points: list[MouseStep] = []

    for index in range(1, main_steps + 1):
        unit = index / main_steps
        eased = unit * unit * (3.0 - 2.0 * unit)
        inverse = 1.0 - eased
        control_1 = (start_x + (main_end_x - start_x) * 0.24, start_y + curve)
        control_2 = (start_x + (main_end_x - start_x) * 0.78, main_end_y - curve * 0.45)
        x = (
            inverse**3 * start_x
            + 3.0 * inverse**2 * eased * control_1[0]
            + 3.0 * inverse * eased**2 * control_2[0]
            + eased**3 * main_end_x
        )
        y = (
            inverse**3 * start_y
            + 3.0 * inverse**2 * eased * control_1[1]
            + 3.0 * inverse * eased**2 * control_2[1]
            + eased**3 * main_end_y
        )
        delay = generator.uniform(0.007, 0.014) * (1.0 + 0.7 * abs(2.0 * unit - 1.0))
        points.append(MouseStep(x=x, y=y, delay_seconds=delay))

    points.extend(
        (
            MouseStep(end_x + overshoot * 0.48, end_y + curve * 0.08, generator.uniform(0.025, 0.045)),
            MouseStep(end_x - direction * 0.35, end_y, generator.uniform(0.035, 0.060)),
            MouseStep(end_x, end_y, generator.uniform(0.025, 0.045)),
        )
    )
    return points


def decode_captcha_images(
    background_bytes: bytes,
    piece_bytes: bytes,
) -> tuple[list[list[int]], list[list[int]]]:
    """Decode CAPTCHA images lazily so the base AutoResearch install stays small."""

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on optional installation
        raise BrowserDependencyError(
            'Pillow is required for CAPTCHA image analysis; install AutoResearch with ".[browser]"'
        ) from exc

    try:
        with Image.open(BytesIO(background_bytes)) as image:
            background = image.convert("L")
            background_width, background_height = background.size
            background_pixels = list(background.getdata())
            background_gray = [
                background_pixels[row * background_width : (row + 1) * background_width]
                for row in range(background_height)
            ]

        with Image.open(BytesIO(piece_bytes)) as image:
            rgba = image.convert("RGBA")
            if rgba.height != background_height:
                raise PuzzleSolveError("CAPTCHA source images have different heights")
            pixels = list(rgba.getdata())
            alpha_values = [pixel[3] for pixel in pixels]
            if min(alpha_values) == max(alpha_values):
                corner = pixels[0][:3]
                alpha_values = [
                    255 if sum(abs(channel - corner[index]) for index, channel in enumerate(pixel[:3])) >= 30 else 0
                    for pixel in pixels
                ]
            piece_alpha = [
                alpha_values[row * rgba.width : (row + 1) * rgba.width]
                for row in range(rgba.height)
            ]
    except PuzzleSolveError:
        raise
    except Exception as exc:
        raise PuzzleSolveError("Could not decode the CAPTCHA source images") from exc

    return background_gray, piece_alpha


class AutoDLBrowserSession:
    """A reusable Playwright session for AutoDL login and authenticated UI hooks."""

    def __init__(
        self,
        options: BrowserLaunchOptions | None = None,
        selectors: AutoDLBrowserSelectors | None = None,
        *,
        rng: random.Random | None = None,
    ) -> None:
        self.options = options or BrowserLaunchOptions()
        self.selectors = selectors or AutoDLBrowserSelectors()
        self._rng = rng or random.Random()
        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None
        self._authenticated = False

    @property
    def page(self) -> Any:
        if self._page is None:
            raise AutoDLBrowserError("The browser session has not been started")
        return self._page

    @property
    def authenticated(self) -> bool:
        return self._authenticated

    async def __aenter__(self) -> "AutoDLBrowserSession":
        return await self.start()

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def start(self) -> "AutoDLBrowserSession":
        if self._page is not None:
            return self
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:
            raise BrowserDependencyError(
                'Playwright is required; install AutoResearch with ".[browser]"'
            ) from exc

        executable = self.options.executable_path or discover_browser_executable(
            self.options.browser
        )
        if executable is None or not Path(executable).is_file():
            raise BrowserDependencyError(
                "No installed Chrome or Edge executable was found; pass --executable-path"
            )

        self._playwright = await async_playwright().start()
        try:
            self._browser = await self._playwright.chromium.launch(
                executable_path=str(executable),
                headless=self.options.headless,
                slow_mo=self.options.slow_mo_ms,
                args=["--disable-blink-features=AutomationControlled", "--lang=zh-CN"],
            )
            self._context = await self._browser.new_context(
                viewport={
                    "width": self.options.viewport_width,
                    "height": self.options.viewport_height,
                },
                locale="zh-CN",
                timezone_id="Asia/Shanghai",
            )
            await self._context.add_init_script(
                "Object.defineProperty(navigator, 'webdriver', { get: () => undefined });"
            )
            self._page = await self._context.new_page()
            self._page.set_default_timeout(self.options.timeout_ms)
            self._page.set_default_navigation_timeout(self.options.timeout_ms)
        except Exception:
            await self.close()
            raise
        return self

    async def close(self) -> None:
        browser, playwright = self._browser, self._playwright
        self._page = None
        self._context = None
        self._browser = None
        self._playwright = None
        self._authenticated = False
        if browser is not None:
            await browser.close()
        if playwright is not None:
            await playwright.stop()

    async def login(
        self,
        credentials: AutoDLCredentials,
        otp_provider: OTPProvider,
        *,
        login_url: str = AUTODL_LOGIN_URL,
    ) -> LoginResult:
        """Authenticate while keeping the same page alive during the OTP callback."""

        if self._page is None:
            await self.start()
        await self.goto_autodl(login_url, wait_until="domcontentloaded")

        phone_input = await self._find_visible(self.selectors.phone_inputs)
        password_input = await self._find_visible(self.selectors.password_inputs)
        await phone_input.fill(credentials.phone)
        await password_input.fill(credentials.password)
        await self._click_submit()

        state = await self._wait_for_login_state()
        if state == "captcha":
            await self.solve_puzzle_captcha()
            state = await self._wait_for_login_state()

        if state == "otp":
            await self._request_sms_code()
            otp = otp_provider()
            if inspect.isawaitable(otp):
                otp = await otp
            normalized = str(otp).strip()
            if not normalized.isdigit() or not 4 <= len(normalized) <= 8:
                raise AutoDLLoginError("The SMS one-time code must contain 4 to 8 digits")
            otp_input = await self._find_visible(self.selectors.otp_inputs)
            await otp_input.fill(normalized)
            await self._click_submit()
            state = await self._wait_for_login_state(allow_otp=False)
            if state == "captcha":
                await self.solve_puzzle_captcha()
                state = await self._wait_for_login_state(allow_otp=False)

        if state != "authenticated":
            raise AutoDLLoginError("AutoDL login did not reach an authenticated page")
        self._authenticated = True
        return LoginResult(status="authenticated", url=self.page.url)

    async def run_authenticated(
        self,
        credentials: AutoDLCredentials,
        otp_provider: OTPProvider,
        hook: AuthenticatedHook[T],
    ) -> T:
        """Run a caller-supplied instance workflow in this authenticated session."""

        await self.login(credentials, otp_provider)
        result = hook(self)
        return await result if inspect.isawaitable(result) else result

    async def open_instance_list(self) -> Any:
        self._require_authenticated()
        return await self.goto_autodl(AUTODL_INSTANCE_LIST_URL)

    async def open_instance_market(self) -> Any:
        self._require_authenticated()
        return await self.goto_autodl(AUTODL_INSTANCE_MARKET_URL)

    async def goto_autodl(self, url: str, **kwargs: Any) -> Any:
        parsed = urlparse(urljoin(AUTODL_LOGIN_URL, url))
        if parsed.scheme != "https" or parsed.hostname not in _TRUSTED_AUTODL_HOSTS:
            raise ValueError("Browser navigation is restricted to trusted AutoDL HTTPS hosts")
        return await self.page.goto(parsed.geturl(), **kwargs)

    async def solve_puzzle_captcha(self, *, max_attempts: int = 3) -> PuzzleMatch:
        if max_attempts < 1 or max_attempts > 5:
            raise ValueError("max_attempts must be between 1 and 5")

        last_match: PuzzleMatch | None = None
        for attempt in range(max_attempts):
            background = await self._find_visible((self.selectors.captcha_background,))
            piece = await self._find_visible((self.selectors.captcha_piece,))
            track = await self._find_visible((self.selectors.captcha_track,))
            thumb = await self._find_visible((self.selectors.captcha_thumb,))

            background_bytes, piece_bytes = await asyncio.gather(
                self._image_bytes(background), self._image_bytes(piece)
            )
            background_gray, piece_alpha = decode_captcha_images(
                background_bytes, piece_bytes
            )
            match = solve_puzzle_gap(background_gray, piece_alpha, search_start=20)
            last_match = match
            if match.confidence < 1.005:
                await self._refresh_captcha()
                continue

            background_box, piece_box, track_box, thumb_box = await asyncio.gather(
                background.bounding_box(),
                piece.bounding_box(),
                track.bounding_box(),
                thumb.bounding_box(),
            )
            if not all((background_box, piece_box, track_box, thumb_box)):
                raise PuzzleSolveError("The CAPTCHA geometry is not visible")

            distance = slider_distance_for_gap(
                match,
                rendered_background_width=float(background_box["width"]),
                rendered_piece_width=float(piece_box["width"]),
                track_width=float(track_box["width"]),
                thumb_width=float(thumb_box["width"]),
            )
            start = (
                float(thumb_box["x"] + thumb_box["width"] / 2.0),
                float(thumb_box["y"] + thumb_box["height"] / 2.0),
            )
            end = (start[0] + distance, start[1])
            initial_source = await background.get_attribute("src")
            await self._drag_human_path(
                start,
                end,
                piece=piece,
                desired_piece_left=distance,
                maximum_distance=max(
                    1.0,
                    float(track_box["width"] - thumb_box["width"] - 1.0),
                ),
            )
            if await self._wait_for_captcha_success(initial_source):
                return match
            if attempt + 1 < max_attempts:
                await self._refresh_captcha()

        detail = "" if last_match is None else f" (last confidence {last_match.confidence:.3f})"
        raise PuzzleSolveError(f"Aliyun puzzle verification failed after {max_attempts} attempts{detail}")

    async def _request_sms_code(self) -> None:
        button = await self._find_visible(self.selectors.send_code_buttons)
        await button.click()
        await self._find_visible((self.selectors.captcha_background,))
        await self.solve_puzzle_captcha()

        deadline = asyncio.get_running_loop().time() + self.options.timeout_ms / 1000.0
        while asyncio.get_running_loop().time() < deadline:
            try:
                if await button.is_disabled():
                    return
            except Exception:
                return
            await asyncio.sleep(0.15)
        raise AutoDLLoginError("The CAPTCHA closed but AutoDL did not confirm SMS delivery")

    async def _wait_for_login_state(self, *, allow_otp: bool = True) -> str:
        deadline = asyncio.get_running_loop().time() + self.options.timeout_ms / 1000.0
        while asyncio.get_running_loop().time() < deadline:
            if not self._is_login_url(self.page.url):
                return "authenticated"
            if await self._is_visible(self.selectors.captcha_background):
                return "captcha"
            if allow_otp and await self._any_visible(self.selectors.otp_inputs):
                return "otp"
            await asyncio.sleep(0.15)
        raise AutoDLLoginError("AutoDL login did not advance before the timeout")

    async def _click_submit(self) -> None:
        button = await self._find_visible(self.selectors.submit_buttons)
        await button.click()

    async def _find_visible(self, selectors: Sequence[str]) -> Any:
        deadline = asyncio.get_running_loop().time() + self.options.timeout_ms / 1000.0
        while asyncio.get_running_loop().time() < deadline:
            for selector in selectors:
                locator = self.page.locator(selector)
                try:
                    count = min(await locator.count(), 8)
                    for index in range(count):
                        candidate = locator.nth(index)
                        if await candidate.is_visible():
                            return candidate
                except Exception:
                    continue
            await asyncio.sleep(0.1)
        raise AutoDLBrowserError("A required AutoDL page element did not become visible")

    async def _any_visible(self, selectors: Sequence[str]) -> bool:
        for selector in selectors:
            if await self._is_visible(selector):
                return True
        return False

    async def _is_visible(self, selector: str) -> bool:
        try:
            locator = self.page.locator(selector)
            count = min(await locator.count(), 8)
            return any([await locator.nth(index).is_visible() for index in range(count)])
        except Exception:
            return False

    async def _image_bytes(self, locator: Any) -> bytes:
        source = await locator.get_attribute("src")
        if source and self._context is not None:
            try:
                response = await self._context.request.get(urljoin(self.page.url, source))
                if response.ok:
                    return await response.body()
            except Exception:
                pass
        return await locator.screenshot(type="png")

    async def _drag_human_path(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        piece: Any | None = None,
        desired_piece_left: float | None = None,
        maximum_distance: float | None = None,
    ) -> None:
        path = generate_mouse_trajectory(start, end, rng=self._rng)
        await self.page.mouse.move(*start)
        await self.page.mouse.down()
        try:
            for point in path:
                await self.page.mouse.move(point.x, point.y)
                await asyncio.sleep(point.delay_seconds)
            if (
                piece is not None
                and desired_piece_left is not None
                and maximum_distance is not None
            ):
                current_distance = end[0] - start[0]
                for _ in range(18):
                    try:
                        actual_piece_left = float(
                            await piece.evaluate(
                                "element => parseFloat(element.style.left || getComputedStyle(element).left) || 0"
                            )
                        )
                    except Exception:
                        break
                    next_distance = next_slider_distance(
                        current_distance,
                        desired_piece_left=desired_piece_left,
                        actual_piece_left=actual_piece_left,
                        maximum_distance=maximum_distance,
                    )
                    if next_distance is None:
                        break
                    if abs(next_distance - current_distance) < 0.05:
                        next_distance = max(0.0, current_distance - 2.2)
                    current_distance = next_distance
                    await self.page.mouse.move(
                        start[0] + current_distance,
                        start[1] + self._rng.uniform(-0.35, 0.35),
                    )
                    await asyncio.sleep(self._rng.uniform(0.024, 0.045))
        finally:
            await self.page.mouse.up()

    async def _wait_for_captcha_success(self, initial_source: str | None) -> bool:
        deadline = asyncio.get_running_loop().time() + 8.0
        while asyncio.get_running_loop().time() < deadline:
            if not await self._is_visible(self.selectors.captcha_modal):
                return True
            background = self.page.locator(self.selectors.captcha_background).first
            try:
                current_source = await background.get_attribute("src")
                if initial_source and current_source and current_source != initial_source:
                    return False
            except Exception:
                return True
            await asyncio.sleep(0.15)
        return False

    async def _refresh_captcha(self) -> None:
        old_source: str | None = None
        background = self.page.locator(self.selectors.captcha_background).first
        try:
            old_source = await background.get_attribute("src")
        except Exception:
            pass
        refresh = self.page.locator(self.selectors.captcha_refresh).first
        if await refresh.is_visible():
            await refresh.click()
        deadline = asyncio.get_running_loop().time() + 5.0
        while asyncio.get_running_loop().time() < deadline:
            try:
                new_source = await background.get_attribute("src")
                if new_source and new_source != old_source:
                    return
            except Exception:
                pass
            await asyncio.sleep(0.1)

    def _require_authenticated(self) -> None:
        if not self._authenticated:
            raise AutoDLLoginError("Authenticate before opening an AutoDL instance page")

    @staticmethod
    def _is_login_url(url: str) -> bool:
        path = urlparse(url).path.rstrip("/").lower()
        return path in {"/login", "/admin/login", "/subaccountlogin"}
