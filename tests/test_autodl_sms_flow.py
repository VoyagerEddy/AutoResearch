from unittest.mock import AsyncMock

import pytest

from autoresearch.services.autodl_browser import AutoDLBrowserSession


@pytest.mark.asyncio
async def test_sms_is_requested_before_asking_for_code_and_submitted_once():
    session = AutoDLBrowserSession()
    session._page = type("Page", (), {"url": "https://www.autodl.com/console/instance/list"})()
    phone, button, code = AsyncMock(), AsyncMock(), AsyncMock()
    button.inner_text.return_value = "59s"
    session.goto_autodl = AsyncMock()
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
    session._find_visible = AsyncMock(side_effect=[phone, button])
    session._is_visible = AsyncMock(return_value=False)
    session._click_submit = AsyncMock()
    with pytest.raises(RuntimeError, match="4 to 8 digits"):
        await session.login_sms("test-phone", lambda: "invalid")
    button.click.assert_awaited_once()
    session._click_submit.assert_not_awaited()
