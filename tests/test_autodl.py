import json
from pathlib import Path

import httpx
import pytest

from autoresearch.config import DEFAULT_AUTODL_IMAGE_UUID, Settings
from autoresearch.domain import AutoDLCreateRequest
from autoresearch.services import autodl
from autoresearch.services.autodl import AutoDLClient, AutoDLCreateUncertain, AutoDLError, extract_ssh


@pytest.mark.asyncio
async def test_gpu_priority_falls_back_only_for_verified_capacity_code(tmp_path: Path, monkeypatch) -> None:
    # NoResource is a synthetic fixture, not a documented provider guarantee.
    monkeypatch.setattr(autodl, "VERIFIED_CAPACITY_ERROR_CODES", frozenset({"NoResource"}))
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        seen.append(payload["gpu_spec_uuid"])
        if payload["gpu_spec_uuid"] == "v-48g":
            return httpx.Response(200, json={"code": "NoResource", "msg": "sold out"})
        return httpx.Response(200, json={"code": "Success", "data": "pro-test"})

    settings = Settings.load(tmp_path).with_overrides(
        autodl_token="token", autodl_image_uuid="image-test", autodl_gpu_specs=("v-48g", "5090-p")
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = AutoDLClient(settings, http)
        choice = await client.create_preferred(AutoDLCreateRequest())
    assert seen == ["v-48g", "5090-p"]
    assert choice.instance_uuid == "pro-test"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["timeout", "http_503", "http_401", "invalid_json", "missing_uuid", "bad_uuid", "unknown_code", "unverified_capacity"])
async def test_uncertain_creation_never_retries_or_exposes_response(tmp_path, failure) -> None:
    seen = []

    def handler(request):
        seen.append(request.url.path)
        if failure == "timeout":
            raise httpx.ReadTimeout("private-token", request=request)
        if failure.startswith("http_"):
            return httpx.Response(int(failure.split("_")[1]), text="private-token")
        if failure == "invalid_json":
            return httpx.Response(200, text="private-token")
        if failure in {"missing_uuid", "bad_uuid"}:
            return httpx.Response(200, json={"code": "Success", "data": None if failure == "missing_uuid" else {"password": "private-token"}})
        return httpx.Response(200, json={"code": "NoResource" if failure == "unverified_capacity" else "Unknown", "msg": "private-token"})

    settings = Settings.load(tmp_path).with_overrides(autodl_token="private-token", autodl_image_uuid="image-test")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(AutoDLCreateUncertain) as error:
            await AutoDLClient(settings, http).create_preferred(AutoDLCreateRequest(confirm_billable=True))
    assert seen == ["/api/v1/dev/instance/pro/create"]
    assert "provisioning outcome is unknown" in str(error.value).lower()
    assert "private-token" not in str(error.value)


def test_extract_ssh_handles_nested_snapshot() -> None:
    snapshot = {"connection": {"ssh_host": "host.example", "ssh_port": "3022", "ssh_password": "pw"}}
    assert extract_ssh(snapshot) == {
        "host": "host.example", "port": 3022, "password": "pw", "username": "root"
    }


@pytest.mark.asyncio
async def test_preflight_reports_all_missing_settings_without_network(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail("Missing configuration must not issue an AutoDL API request")

    settings = Settings.load(tmp_path).with_overrides(
        autodl_token="", autodl_image_uuid="", autodl_gpu_specs=("v-48g", "5090-p")
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = AutoDLClient(settings, http)
        result = await client.preflight(verify_api=True)
        assert result["configured"] is False
        assert result["ready"] is False
        assert result["verified"] is False
        assert result["network_checked"] is False
        assert {issue["code"] for issue in result["blocking_issues"]} == {
            "missing_token", "missing_image",
        }
        with pytest.raises(AutoDLError) as error:
            await client.create_preferred(AutoDLCreateRequest(confirm_billable=True))
        assert "AUTODL_TOKEN" in str(error.value)
        assert "AUTODL_IMAGE_UUID" in str(error.value)


@pytest.mark.asyncio
async def test_preflight_local_configuration_does_not_claim_verified(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail("Local preflight must not make network requests")

    settings = Settings.load(tmp_path).with_overrides(
        autodl_token="private-token", autodl_image_uuid="private-image", autodl_gpu_specs=("v-48g",)
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = await AutoDLClient(settings, http).preflight()
    assert result["ready"] is True
    assert result["configured"] is True
    assert result["verified"] is False
    assert result["network_checked"] is False
    assert result["balance_verified"] is False
    assert result["availability_verified"] is False
    assert "private-token" not in json.dumps(result)
    assert "private-image" not in json.dumps(result)


@pytest.mark.asyncio
async def test_preflight_network_check_reads_wallet_and_redacts_records(tmp_path: Path) -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path, json.loads(request.content) if request.content else None))
        if request.url.path == "/api/v1/dev/wallet/balance":
            return httpx.Response(200, json={"code": "Success", "data": {
                "assets": 12567, "voucher_balance": 2345, "root_password": "private-password",
            }})
        return httpx.Response(200, json={"code": "Success", "data": {"list": [
            {"uuid": "private-instance", "root_password": "private-password"}
        ]}})

    settings = Settings.load(tmp_path).with_overrides(
        autodl_token="private-token", autodl_image_uuid="", autodl_gpu_specs=("v-48g",)
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = await AutoDLClient(settings, http).preflight(
            verify_api=True, image_uuid="override-image"
        )
    assert seen == [("POST", "/api/v1/dev/instance/pro/list", {"page_index": 1, "page_size": 1}),
                    ("POST", "/api/v1/dev/wallet/balance", None)]
    assert result["ready"] is True
    assert result["verified"] is True
    assert result["network_checked"] is True
    assert result["balance_verified"] is True
    assert result["balance"]["assets_cny"] == "12.567"
    assert result["balance"]["voucher_balance_cny"] == "2.345"
    assert result["availability_verified"] is False
    for private in ("private-token", "private-instance", "private-password", "override-image"):
        assert private not in json.dumps(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure, expected_code", [
    ("unauthorized", "api_auth_failed"),
    ("http_error", "api_http_error"),
    ("api_rejected", "api_rejected"),
    ("timeout", "api_unreachable"),
    ("invalid_json", "api_invalid_response"),
    ("invalid_body", "api_invalid_response"),
    ("missing_list", "api_invalid_response"),
])
async def test_preflight_failure_does_not_expose_upstream_secrets(
    tmp_path: Path, failure: str, expected_code: str
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if failure == "unauthorized":
            return httpx.Response(401, text="private-token")
        if failure == "http_error":
            return httpx.Response(503, text="private-token")
        if failure == "api_rejected":
            return httpx.Response(200, json={"code": "Denied", "msg": "private-token"})
        if failure == "timeout":
            raise httpx.ReadTimeout("private-token", request=request)
        if failure == "invalid_json":
            return httpx.Response(200, text="private-token")
        if failure == "invalid_body":
            return httpx.Response(200, json=["private-token"])
        return httpx.Response(200, json={"code": "Success", "data": None})

    settings = Settings.load(tmp_path).with_overrides(
        autodl_token="private-token", autodl_image_uuid="image", autodl_gpu_specs=("v-48g",)
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = await AutoDLClient(settings, http).preflight(verify_api=True)
    assert result["configured"] is True
    assert result["ready"] is False
    assert result["verified"] is False
    assert result["network_checked"] is True
    assert result["blocking_issues"][0]["code"] == expected_code
    assert "private-token" not in json.dumps(result)


@pytest.mark.asyncio
async def test_preflight_retains_missing_image_after_successful_auth(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"code": "Success", "data": {"list": []}})

    settings = Settings.load(tmp_path).with_overrides(
        autodl_token="token", autodl_image_uuid="", autodl_gpu_specs=("v-48g",)
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = await AutoDLClient(settings, http).preflight(verify_api=True)
    assert result["ready"] is False
    assert result["configured"] is False
    assert result["verified"] is False
    assert result["checks"][-1]["status"] == "passed"
    assert result["blocking_issues"][0]["code"] == "missing_image"


@pytest.mark.asyncio
async def test_preflight_warns_about_explicit_legacy_gpu_spec(tmp_path: Path) -> None:
    settings = Settings.load(tmp_path).with_overrides(
        autodl_token="token", autodl_image_uuid="image", autodl_gpu_specs=("v-48g", "5090")
    )
    result = await AutoDLClient(settings).preflight()
    assert "legacy_gpu_spec" in {warning["code"] for warning in result["warnings"]}
    assert settings.autodl_gpu_specs == ("v-48g", "5090")


def test_gpu_default_uses_documented_5090_spec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AUTODL_GPU_SPECS", raising=False)
    assert Settings.load(tmp_path).autodl_gpu_specs == ("v-48g", "5090-p")


def test_blank_image_setting_uses_documented_public_base_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AUTODL_IMAGE_UUID", raising=False)
    (tmp_path / ".env").write_text("AUTODL_IMAGE_UUID=\n", encoding="utf-8")
    settings = Settings.load(tmp_path)
    assert settings.autodl_image_uuid == DEFAULT_AUTODL_IMAGE_UUID


@pytest.mark.asyncio
async def test_preflight_reports_public_image_fallback_without_exposing_uuid(
    tmp_path: Path,
) -> None:
    settings = Settings.load(tmp_path).with_overrides(autodl_token="token")
    result = await AutoDLClient(settings).preflight()
    assert result["ready"] is True
    assert result["image_source"] == "public_default"
    assert "public base image" in result["checks"][1]["message"]
    assert DEFAULT_AUTODL_IMAGE_UUID not in json.dumps(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("data", [None, [], {}, {"assets": True, "voucher_balance": 1},
                                   {"assets": "1000", "voucher_balance": 1},
                                   {"assets": 1, "voucher_balance": 1.5},
                                   {"assets": 2**100, "voucher_balance": 1}])
async def test_malformed_wallet_does_not_invalidate_instance_auth(tmp_path, data):
    def handler(request):
        return httpx.Response(200, json={"code": "Success", "data":
            {"list": []} if request.url.path.endswith("/pro/list") else data})

    settings = Settings.load(tmp_path).with_overrides(autodl_token="private-token", autodl_image_uuid="image")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = await AutoDLClient(settings, http).preflight(verify_api=True)
    assert result["verified"] and result["ready"]
    assert result["balance_verified"] is False and result["balance"] is None
    assert "balance_not_verified" in {warning["code"] for warning in result["warnings"]}
    assert "private-token" not in json.dumps(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["timeout", "rejected", "unauthorized"])
async def test_wallet_errors_are_sanitized_and_not_billable(tmp_path, failure):
    seen = []
    def handler(request):
        seen.append(request.url.path)
        if request.url.path.endswith("/pro/list"):
            return httpx.Response(200, json={"code": "Success", "data": {"list": []}})
        if failure == "timeout":
            raise httpx.ReadTimeout("private-token", request=request)
        return httpx.Response(401 if failure == "unauthorized" else 200,
                             json={"code": "Denied", "msg": "private-token"})

    settings = Settings.load(tmp_path).with_overrides(autodl_token="private-token", autodl_image_uuid="image")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = await AutoDLClient(settings, http).preflight(verify_api=True)
    assert seen == ["/api/v1/dev/instance/pro/list", "/api/v1/dev/wallet/balance"]
    assert result["verified"] and result["balance_verified"] is False
    assert "private-token" not in json.dumps(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("assets,voucher", [(0, 0), (-1234, 0), (0, 1234)])
async def test_wallet_does_not_infer_payment_eligibility(tmp_path, assets, voucher):
    def handler(request):
        return httpx.Response(200, json={"code": "Success", "data":
            {"list": []} if request.url.path.endswith("/pro/list")
            else {"assets": assets, "voucher_balance": voucher}})

    settings = Settings.load(tmp_path).with_overrides(autodl_token="token", autodl_image_uuid="image")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = await AutoDLClient(settings, http).preflight(verify_api=True)
    assert result["ready"] and result["balance_verified"]
    assert result["balance"]["assets_milli_cny"] == assets
    assert result["balance"]["voucher_balance_milli_cny"] == voucher
    assert ("wallet_nonpositive" in {warning["code"] for warning in result["warnings"]}) == (assets <= 0 and voucher <= 0)
    assert not result["blocking_issues"]
