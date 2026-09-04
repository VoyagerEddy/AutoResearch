from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import httpx

from ..config import Settings
from ..domain import AutoDLCreateRequest


class AutoDLError(RuntimeError):
    pass


@dataclass(slots=True)
class InstanceChoice:
    instance_uuid: str
    gpu_spec: str
    response: dict[str, Any]


class AutoDLClient:
    """Client for AutoDL's documented container instance Pro API."""

    base_url = "https://api.autodl.com/api/v1/dev/instance/pro"

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
            raise AutoDLError("尚未配置 AutoDL 开发者 Token")
        return {"Authorization": self.settings.autodl_token, "Content-Type": "application/json"}

    async def close(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()

    async def _request(self, method: str, path: str, payload: dict[str, Any]) -> Any:
        response = await self._http().request(
            method,
            f"{self.base_url}/{path.lstrip('/')}",
            headers=self._headers(),
            json=payload,
        )
        response.raise_for_status()
        body = response.json()
        if body.get("code") != "Success":
            raise AutoDLError(body.get("msg") or f"AutoDL API 错误：{body.get('code')}")
        return body.get("data")

    async def list_instances(self, page_size: int = 50) -> list[dict[str, Any]]:
        data = await self._request("POST", "list", {"page_index": 1, "page_size": page_size})
        return list((data or {}).get("list", []))

    async def snapshot(self, instance_uuid: str) -> dict[str, Any]:
        data = await self._request("GET", "snapshot", {"instance_uuid": instance_uuid})
        return dict(data or {})

    async def status(self, instance_uuid: str) -> str:
        return str(
            await self._request("GET", "status", {"instance_uuid": instance_uuid})
        )

    async def create_preferred(self, request: AutoDLCreateRequest) -> InstanceChoice:
        image_uuid = request.image_uuid or self.settings.autodl_image_uuid
        if not image_uuid:
            raise AutoDLError("创建实例前必须配置 AutoDL 镜像 UUID")
        errors: list[str] = []
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
                instance_uuid = str(await self._request("POST", "create", payload))
                return InstanceChoice(instance_uuid, gpu_spec, payload)
            except (AutoDLError, httpx.HTTPError) as exc:
                errors.append(f"{gpu_spec}: {exc}")
        raise AutoDLError("首选 GPU 均无法创建实例；" + "；".join(errors))

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
                raise AutoDLError("保存克隆镜像失败")
            await asyncio.sleep(15)
        raise AutoDLError("等待克隆镜像完成超时")

    async def wait_running(self, instance_uuid: str, timeout: int = 600) -> dict[str, Any]:
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            state = await self.status(instance_uuid)
            if state == "running":
                return await self.snapshot(instance_uuid)
            if state in {"failed", "released", "error"}:
                raise AutoDLError(f"实例进入异常状态：{state}")
            await asyncio.sleep(8)
        raise AutoDLError("等待 AutoDL 实例启动超时")


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
