from __future__ import annotations

import asyncio
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import httpx

from ..config import DEFAULT_AUTODL_IMAGE_UUID, Settings
from ..domain import AutoDLCreateRequest


class AutoDLError(RuntimeError):
    pass


class AutoDLCreateUncertain(AutoDLError):
    """A create request may have reached the provider; never retry it automatically."""


class _AutoDLResponseError(AutoDLError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__("AutoDL API rejected the request; check configuration and permissions")


# The public Pro API reference currently documents Success, but does not give
# a contractual no-instance-created guarantee for failure codes. Keep empty
# until a capacity rejection code is verified with the provider.
VERIFIED_CAPACITY_ERROR_CODES: frozenset[str] = frozenset()


@dataclass(slots=True)
class InstanceChoice:
    instance_uuid: str
    gpu_spec: str
    response: dict[str, Any]


class AutoDLClient:
    """Client for AutoDL's documented container instance Pro API."""

    base_url = "https://api.autodl.com/api/v1/dev/instance/pro"
    docs_url = "https://www.autodl.com/docs/instance_pro_api/"

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self._client = client
        self._owns_client = client is None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(45.0, connect=15.0))
        return self._client

    def _headers(self) -> dict[str, str]:
        if not self.settings.autodl_token:
            raise AutoDLError("AutoDL developer token is not configured")
        return {"Authorization": self.settings.autodl_token, "Content-Type": "application/json"}

    async def close(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()

    def _image_selection(self, requested: str | None = None) -> tuple[str, str]:
        requested = (requested or "").strip()
        if requested:
            return requested, "request"
        configured = self.settings.autodl_image_uuid.strip()
        if configured == DEFAULT_AUTODL_IMAGE_UUID:
            return configured, "public_default"
        return configured, "configured" if configured else "missing"

    async def _request(self, method: str, path: str, payload: dict[str, Any]) -> Any:
        return await self._request_url(method, f"{self.base_url}/{path.lstrip('/')}", payload)

    async def _request_url(self, method: str, url: str, payload: dict[str, Any] | None = None) -> Any:
        response = await self._http().request(
            method,
            url,
            headers=self._headers(),
            json=payload,
        )
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            raise ValueError("Invalid AutoDL API response")
        if body.get("code") != "Success":
            raise _AutoDLResponseError(str(body.get("code", "")))
        return body.get("data")

    async def wallet_balance(self) -> dict[str, Any]:
        """Read official wallet amounts, whose integer unit is 1/1000 CNY.

        Voucher applicability and future experiment cost are not established by
        this endpoint. Never sum the fields into a claimed spendable budget.
        """
        data = await self._request_url("POST", "https://api.autodl.com/api/v1/dev/wallet/balance")
        if not isinstance(data, dict) or any(type(data.get(key)) is not int
                                             or not -(2**63) <= data[key] < 2**63
                                             for key in ("assets", "voucher_balance")):
            raise ValueError("Invalid AutoDL wallet response")
        return {
            "currency": "CNY",
            "assets_milli_cny": data["assets"],
            "voucher_balance_milli_cny": data["voucher_balance"],
            "assets_cny": format(Decimal(data["assets"]) / 1000, ".3f"),
            "voucher_balance_cny": format(Decimal(data["voucher_balance"]) / 1000, ".3f"),
        }

    async def list_instances(self, page_size: int = 50) -> list[dict[str, Any]]:
        data = await self._request("POST", "list", {"page_index": 1, "page_size": page_size})
        if not isinstance(data, dict) or not isinstance(data.get("list"), list):
            raise ValueError("Invalid AutoDL instance-list response")
        instances = data["list"]
        if not all(isinstance(item, dict) for item in instances):
            raise ValueError("Invalid AutoDL instance-list record")
        return instances

    async def snapshot(self, instance_uuid: str) -> dict[str, Any]:
        data = await self._request("GET", "snapshot", {"instance_uuid": instance_uuid})
        return dict(data or {})

    async def status(self, instance_uuid: str) -> str:
        return str(
            await self._request("GET", "status", {"instance_uuid": instance_uuid})
        )

    async def preflight(
        self, *, verify_api: bool = False, image_uuid: str | None = None
    ) -> dict[str, Any]:
        """Check local prerequisites and optionally read instance list and wallet.

        Wallet balance uses the documented common API, separate from Pro API.
        A successful check never promises that a billable creation succeeds.
        Raw API responses and exception text must not enter this public result.
        """
        checks: list[dict[str, str]] = []
        issues: list[dict[str, str]] = []
        warnings: list[dict[str, str]] = [
            {
                "code": "creation_not_verified",
                "message": "GPU inventory and image availability are unverified; provisioning may still fail after this check passes.",
            }
        ]
        selected_image, image_source = self._image_selection(image_uuid)
        image_message = (
            "The documented AutoDL public base image fallback is selected."
            if image_source == "public_default"
            else "An image UUID is configured for provisioning."
        )
        prerequisites = (
            (
                "token", bool(self.settings.autodl_token.strip()), "missing_token",
                "AutoDL developer token is configured.",
                "AutoDL developer token is missing; set AUTODL_TOKEN in the web settings.",
            ),
            (
                "image", bool(selected_image),
                "missing_image", image_message,
                "AutoDL image UUID is missing; set AUTODL_IMAGE_UUID in the web settings or pass image_uuid when provisioning.",
            ),
            (
                "gpu_specs", bool(self.settings.autodl_gpu_specs)
                and all(spec.strip() for spec in self.settings.autodl_gpu_specs),
                "missing_gpu_specs", "GPU specification priority is configured.",
                "No valid GPU specification is configured; set AUTODL_GPU_SPECS in the web settings.",
            ),
        )
        for name, present, code, success, failure in prerequisites:
            checks.append({
                "name": name, "status": "passed" if present else "blocked",
                "message": success if present else failure,
            })
            if not present:
                issues.append({"code": code, "message": failure})
        configured = not issues
        if "5090" in self.settings.autodl_gpu_specs:
            warnings.append({
                "code": "legacy_gpu_spec",
                "message": "GPU spec '5090' does not match the official Pro API appendix; the 5090-32G performance spec ID is '5090-p'.",
            })

        network_checked = verify_api and bool(self.settings.autodl_token.strip())
        api_verified = False
        api_check = {
            "name": "api", "status": "skipped",
            "message": "AutoDL API has not been checked online; use verify_api=true for a free read-only check.",
        }
        if verify_api and not network_checked:
            api_check["message"] = "Online check skipped because no token is configured."
        if network_checked:
            failure_code = "api_rejected"
            failure_message = "AutoDL API rejected the request; check the token, identity verification, and Pro API access."
            try:
                # Listing existing instances neither creates nor powers on a resource.
                # Do not return instance records: they can contain private metadata.
                await self.list_instances(page_size=1)
                api_verified = True
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in {401, 403}:
                    failure_code = "api_auth_failed"
                    failure_message = "AutoDL authentication or permission check failed; verify the developer token, identity status, and Pro API access."
                else:
                    failure_code = "api_http_error"
                    failure_message = f"AutoDL API returned HTTP {exc.response.status_code}; try again later."
            except httpx.RequestError:
                failure_code = "api_unreachable"
                failure_message = "Could not reach AutoDL API or the request timed out; check the local network and retry."
            except AutoDLError:
                pass
            except (ValueError, TypeError, AttributeError):
                failure_code = "api_invalid_response"
                failure_message = "AutoDL API returned an unrecognized response; retry later or check API compatibility."
            if api_verified:
                api_check.update(status="passed", message="Read-only AutoDL instance-list request succeeded; API authentication is valid.")
            else:
                api_check.update(status="failed", message=failure_message)
                issues.append({"code": failure_code, "message": failure_message})
        balance = None
        balance_check = {
            "name": "balance", "status": "skipped",
            "message": "Wallet balance is unverified; it can be read after read-only API authentication succeeds.",
        }
        if api_verified:
            try:
                balance = await self.wallet_balance()
            except (httpx.HTTPError, AutoDLError, ValueError, TypeError, AttributeError):
                balance_check.update(status="failed", message="Wallet query failed; the completed instance API authentication check remains valid.")
            else:
                balance_check.update(status="passed", message="Cash and voucher balances were read; voucher eligibility and experiment budget sufficiency remain unverified.")
                if balance["assets_milli_cny"] <= 0 and balance["voucher_balance_milli_cny"] <= 0:
                    warnings.append({
                        "code": "wallet_nonpositive",
                        "message": "Cash and voucher balances are both nonpositive; this endpoint does not establish payment eligibility for provisioning.",
                    })
        if balance is None:
            warnings.append({"code": "balance_not_verified", "message": "Wallet balance is unverified."})
        checks.extend([balance_check, api_check])
        return {
            "ready": configured and (not verify_api or api_verified),
            "configured": configured,
            "verified": configured and api_verified,
            "network_checked": network_checked,
            "checks": checks,
            "blocking_issues": issues,
            "warnings": warnings,
            "balance_verified": balance is not None,
            "balance": balance,
            "availability_verified": False,
            "image_source": image_source,
            "docs_url": self.docs_url,
        }

    async def create_preferred(self, request: AutoDLCreateRequest) -> InstanceChoice:
        readiness = await self.preflight(image_uuid=request.image_uuid)
        if not readiness["configured"]:
            raise AutoDLError("; ".join(issue["message"] for issue in readiness["blocking_issues"]))
        image_uuid, _ = self._image_selection(request.image_uuid)
        exhausted: list[str] = []
        uncertain_message = (
            "AutoDL provisioning outcome is unknown and automatic retries have stopped; a billable instance may exist. "
            "Inspect the AutoDL console and retry only after confirming that no new instance was created."
        )
        for gpu_spec in self.settings.autodl_gpu_specs:
            payload: dict[str, Any] = {
                "req_gpu_amount": request.gpu_amount,
                "expand_system_disk_by_gb": request.disk_gb,
                "gpu_spec_uuid": gpu_spec,
                "image_uuid": image_uuid,
                "cuda_v_from": self.settings.autodl_cuda_from,
                "instance_name": request.instance_name,
                "start_command": "sleep 1",
            }
            if request.data_centers:
                payload["data_center_list"] = request.data_centers
            try:
                instance_uuid = await self._request("POST", "create", payload)
                if (
                    not isinstance(instance_uuid, str) or not instance_uuid
                    or len(instance_uuid) > 200
                    or not all(char.isascii() and (char.isalnum() or char in "-_") for char in instance_uuid)
                ):
                    raise ValueError("AutoDL did not return a valid instance identifier")
                return InstanceChoice(instance_uuid, gpu_spec, payload)
            except _AutoDLResponseError as exc:
                if exc.code in VERIFIED_CAPACITY_ERROR_CODES:
                    exhausted.append(gpu_spec)
                    continue
                raise AutoDLCreateUncertain(uncertain_message) from None
            except (httpx.HTTPError, ValueError, TypeError, AttributeError):
                raise AutoDLCreateUncertain(uncertain_message) from None
        raise AutoDLError("Verified insufficient capacity for all configured GPU specifications: " + ", ".join(exhausted))

    async def power_on(self, instance_uuid: str) -> None:
        await self._request(
            "POST", "power_on", {"instance_uuid": instance_uuid, "payload": "gpu"}
        )

    async def power_off(self, instance_uuid: str) -> None:
        await self._request("POST", "power_off", {"instance_uuid": instance_uuid})

    async def release(self, instance_uuid: str) -> None:
        await self._request("POST", "release", {"instance_uuid": instance_uuid})

    async def save_image(self, instance_uuid: str, name: str) -> str:
        data = await self._request(
            "POST", "image/save", {"instance_uuid": instance_uuid, "image_name": name}
        )
        return str((data or {}).get("image_uuid", ""))

    async def list_images(self, page_size: int = 50) -> list[dict[str, Any]]:
        data = await self._request(
            "POST", "image/private/list", {"page_index": 1, "page_size": page_size}
        )
        return list((data or {}).get("list", []))

    async def wait_image(self, image_uuid: str, timeout: int = 1800) -> None:
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            images = await self.list_images()
            image = next(
                (item for item in images if item.get("image_uuid") == image_uuid), None
            )
            if image and image.get("status") == "finished":
                return
            if image and image.get("status") in {"failed", "error"}:
                raise AutoDLError("Failed to save the clone image")
            await asyncio.sleep(15)
        raise AutoDLError("Timed out waiting for the clone image")

    async def wait_running(self, instance_uuid: str, timeout: int = 600) -> dict[str, Any]:
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            state = await self.status(instance_uuid)
            if state == "running":
                return await self.snapshot(instance_uuid)
            if state in {"failed", "released", "error"}:
                raise AutoDLError(f"Instance entered an error state: {state}")
            await asyncio.sleep(8)
        raise AutoDLError("Timed out waiting for the AutoDL instance to start")


def extract_ssh(snapshot: dict[str, Any]) -> dict[str, Any] | None:
    """Tolerate small response-shape changes in AutoDL snapshot payloads."""

    aliases = {
        "host": ("ssh_host", "host", "host_name", "proxy_host"),
        "port": ("ssh_port", "port", "proxy_port"),
        "username": ("ssh_user", "username", "user"),
        "password": ("ssh_password", "password", "root_password"),
    }

    def walk(value: Any) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        if isinstance(value, dict):
            found.append(value)
            for child in value.values():
                found.extend(walk(child))
        elif isinstance(value, list):
            for child in value:
                found.extend(walk(child))
        return found

    for candidate in walk(snapshot):
        normalized: dict[str, Any] = {}
        for target, keys in aliases.items():
            for key in keys:
                if candidate.get(key) not in (None, ""):
                    normalized[target] = candidate[key]
                    break
        if "host" in normalized and "port" in normalized:
            normalized.setdefault("username", "root")
            normalized["port"] = int(normalized["port"])
            return normalized
    return None
