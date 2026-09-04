import json
from pathlib import Path

import httpx
import pytest

from autoresearch.config import Settings
from autoresearch.domain import AutoDLCreateRequest
from autoresearch.services.autodl import AutoDLClient, extract_ssh


@pytest.mark.asyncio
async def test_gpu_priority_falls_back_to_second_spec(tmp_path: Path) -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        seen.append(payload["gpu_spec_uuid"])
        if payload["gpu_spec_uuid"] == "v-48g":
            return httpx.Response(200, json={"code": "NoResource", "msg": "sold out"})
        return httpx.Response(200, json={"code": "Success", "data": "pro-test"})

    settings = Settings.load(tmp_path).with_overrides(
        autodl_token="token", autodl_image_uuid="image-test", autodl_gpu_specs=("v-48g", "5090")
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = AutoDLClient(settings, http)
        choice = await client.create_preferred(AutoDLCreateRequest())
    assert seen == ["v-48g", "5090"]
    assert choice.instance_uuid == "pro-test"


def test_extract_ssh_handles_nested_snapshot() -> None:
    snapshot = {"connection": {"ssh_host": "host.example", "ssh_port": "3022", "ssh_password": "pw"}}
    assert extract_ssh(snapshot) == {
        "host": "host.example", "port": 3022, "password": "pw", "username": "root"
    }

