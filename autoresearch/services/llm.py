from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import httpx

from ..config import Settings


class LLMError(RuntimeError):
    pass


class OpenRouterClient:
    base_url = "https://openrouter.ai/api/v1"

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self._client = client
        self._owns_client = client is None
        self._free_model_cache: list[str] | None = None

    async def __aenter__(self) -> "OpenRouterClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=15.0))
        return self._client

    def _headers(self) -> dict[str, str]:
        if not self.settings.openrouter_api_key:
            raise LLMError("尚未配置 OpenRouter API Key")
        return {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.settings.openrouter_site_url,
            "X-Title": "AutoResearch",
        }

    async def free_models(self) -> list[str]:
        if self._free_model_cache is not None:
            return self._free_model_cache
        response = await self._http().get(f"{self.base_url}/models", headers=self._headers())
        response.raise_for_status()
        models: list[str] = []
        for item in response.json().get("data", []):
            pricing = item.get("pricing") or {}
            model_id = str(item.get("id", ""))
            if model_id.endswith(":free") or (
                str(pricing.get("prompt")) == "0"
                and str(pricing.get("completion")) == "0"
            ):
                models.append(model_id)
        self._free_model_cache = models
        return models

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 6000,
    ) -> str:
        selected = model or self.settings.openrouter_model or "openrouter/free"
        payload = {
            "model": selected,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = await self._http().post(
                    f"{self.base_url}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                )
                if response.status_code == 429:
                    await asyncio.sleep(2**attempt)
                    continue
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                if not content:
                    raise LLMError("模型返回了空内容")
                return str(content)
            except (httpx.HTTPError, KeyError, IndexError, LLMError) as exc:
                last_error = exc
                if attempt < 2:
                    await asyncio.sleep(1.5 * (attempt + 1))
        raise LLMError(f"OpenRouter 请求失败：{last_error}")

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        max_tokens: int = 8000,
    ) -> dict[str, Any]:
        content = await self.chat(messages, model=model, max_tokens=max_tokens)
        return extract_json_object(content)


def extract_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMError(f"模型没有返回有效 JSON：{exc}") from exc
    if not isinstance(value, dict):
        raise LLMError("模型 JSON 顶层必须是对象")
    return value

