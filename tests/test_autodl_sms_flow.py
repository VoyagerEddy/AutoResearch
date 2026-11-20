from unittest.mock import AsyncMock

import pytest

from autoresearch.services.autodl_browser import AutoDLBrowserSession


class _CaptchaLocator:
    def __init__(self, *, visible=True, sources=()):
        self.visible = visible
        self.sources = list(sources)

    @property
    def first(self):
        return self

    async def count(self):
        return 1

    async def is_visible(self):
        if isinstance(self.visible, list):
            if len(self.visible) > 1:
                return self.visible.pop(0)
            return self.visible[0]
        return self.visible

    async def get_attribute(self, _name):
        value = self.sources.pop(0) if self.sources else "new"
        if isinstance(value, Exception):
            raise value
        return value


class _CaptchaPage:
    def __init__(self, modal, background):
        self.modal = modal
        self.background = background

    def locator(self, selector):
        if selector == "#aliyunCaptcha-window-popup":
            return self.modal
        if selector == "#aliyunCaptcha-img":
            return self.background
        raise AssertionError(selector)


@pytest.mark.asyncio
async def test_sms_is_requested_before_asking_for_code_and_submitted_once():
    session = AutoDLBrowserSession()
    session._page = type("Page", (), {"url": "https://www.autodl.com/console/instance/list"})()
    phone, button, code = AsyncMock(), AsyncMock(), AsyncMock()
    button.inner_text.return_value = "59s"
    session.goto_autodl = AsyncMock()
    session._ensure_sms_login_mode = AsyncMock()
    session._find_visible = AsyncMock(side_effect=[phone, button, code])
    session._is_visible = AsyncMock(return_value=False)
    session._click_submit = AsyncMock()
    session._wait_for_login_state = AsyncMock(return_value="authenticated")

    async def provide_code():
        button.click.assert_awaited_once()
        session._click_submit.assert_not_awaited()
        return "123456"

    result = await session.login_sms("test-phone", provide_code)
    code.fill.assert_awaited_once_with("123456")
    session._click_submit.assert_awaited_once()
    assert result.status == "authenticated"


@pytest.mark.asyncio
async def test_invalid_code_is_not_submitted():
    session = AutoDLBrowserSession()
    session._page = object()
    phone, button = AsyncMock(), AsyncMock()
    button.inner_text.return_value = "59s"
    session.goto_autodl = AsyncMock()
    session._ensure_sms_login_mode = AsyncMock()
    session._find_visible = AsyncMock(side_effect=[phone, button])
    session._is_visible = AsyncMock(return_value=False)
    session._click_submit = AsyncMock()
    with pytest.raises(RuntimeError, match="4 to 8 digits"):
        await session.login_sms("test-phone", lambda: "invalid")
    button.click.assert_awaited_once()
    session._click_submit.assert_not_awaited()


@pytest.mark.asyncio
async def test_sms_switches_mode_and_automates_captcha():
    session = AutoDLBrowserSession()
    session._page = type("Page", (), {"url": "https://www.autodl.com/login"})()
    phone, button, code = AsyncMock(), AsyncMock(), AsyncMock()
    button.inner_text.return_value = "59s"
    session.goto_autodl = AsyncMock()
    session._ensure_sms_login_mode = AsyncMock()
    session._find_visible = AsyncMock(side_effect=[phone, button, code])
    session._is_visible = AsyncMock(side_effect=[True, False])
    session._complete_captcha = AsyncMock()
    session._click_submit = AsyncMock()
    session._wait_for_login_state = AsyncMock(return_value="authenticated")

    result = await session.login_sms(
        "test-phone", lambda: "123456", automate_captcha=True
    )

    session._ensure_sms_login_mode.assert_awaited_once()
    session._complete_captcha.assert_awaited_once_with(True)
    assert result.status == "authenticated"


@pytest.mark.asyncio
async def test_sms_mode_waits_for_toggle_then_send_button():
    session = AutoDLBrowserSession()
    toggle = AsyncMock()
    session._any_visible = AsyncMock(side_effect=[False, True, True])
    session._find_visible = AsyncMock(return_value=toggle)

    await session._ensure_sms_login_mode()

    toggle.click.assert_awaited_once()


@pytest.mark.asyncio
async def test_captcha_refresh_then_modal_close_is_success():
    session = AutoDLBrowserSession()
    session._page = _CaptchaPage(
        _CaptchaLocator(visible=[True, False]), _CaptchaLocator(sources=["new"])
    )

    assert await session._wait_for_captcha_success(
        "old", timeout_seconds=0.5, refresh_grace_seconds=0.3
    ) == "success"


@pytest.mark.asyncio
async def test_captcha_refresh_with_modal_open_is_rejected():
    session = AutoDLBrowserSession()
    session._page = _CaptchaPage(
        _CaptchaLocator(visible=True), _CaptchaLocator(sources=["new"])
    )

    assert await session._wait_for_captcha_success(
        "old", timeout_seconds=0.5, refresh_grace_seconds=0.01
    ) == "refreshed"


@pytest.mark.asyncio
async def test_captcha_source_error_does_not_count_as_success():
    session = AutoDLBrowserSession()
    session._page = _CaptchaPage(
        _CaptchaLocator(visible=True),
        _CaptchaLocator(sources=[RuntimeError("detached"), "old", "old"]),
    )

    assert await session._wait_for_captcha_success(
        "old", timeout_seconds=0.01
    ) == "timeout"
