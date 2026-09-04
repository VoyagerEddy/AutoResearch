from __future__ import annotations

import asyncio
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote_plus, urlparse

import httpx

from ..config import Settings
from ..domain import Source
from .artifacts import safe_slug


class ResearchSearch:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self._client = client
        self._owns_client = client is None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(25.0, connect=10.0),
                follow_redirects=True,
                headers={"User-Agent": "AutoResearch/0.1 (local research tool)"},
            )
        return self._client

    async def close(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()

    async def search(self, queries: list[str], limit: int = 20) -> list[Source]:
        clean_queries = [q.strip() for q in queries if q.strip()][:5]
        if not clean_queries:
            return []
        per_provider = max(2, min(8, limit // max(1, len(clean_queries))))
        tasks = []
        for query in clean_queries:
            tasks.extend(
                [
                    self._guard(self.arxiv(query, per_provider)),
                    self._guard(self.openalex(query, per_provider)),
                    self._guard(self.semantic_scholar(query, per_provider)),
                    self._guard(self.github(query, min(5, per_provider))),
                ]
            )
        groups = await asyncio.gather(*tasks)
        unique: dict[str, Source] = {}
        for source in (item for group in groups for item in group):
            key = re.sub(r"\W+", "", source.title.lower()) or source.url
            previous = unique.get(key)
            if previous is None or source.citation_count > previous.citation_count:
                unique[key] = source
        ranked = sorted(
            unique.values(),
            key=lambda item: (item.kind == "paper", item.citation_count, item.year or 0),
            reverse=True,
        )
        return ranked[:limit]

    async def _guard(self, awaitable: object) -> list[Source]:
        try:
            return await awaitable  # type: ignore[misc]
        except (httpx.HTTPError, ET.ParseError, KeyError, TypeError, ValueError):
            return []

    async def arxiv(self, query: str, limit: int) -> list[Source]:
        url = (
            "https://export.arxiv.org/api/query?search_query=all:"
            f"{quote_plus(query)}&start=0&max_results={limit}&sortBy=relevance"
        )
        response = await self._http().get(url)
        response.raise_for_status()
        root = ET.fromstring(response.text)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        results: list[Source] = []
        for entry in root.findall("a:entry", ns):
            page_url = (entry.findtext("a:id", default="", namespaces=ns) or "").strip()
            pdf_url = ""
            for link in entry.findall("a:link", ns):
                if link.attrib.get("title") == "pdf":
                    pdf_url = link.attrib.get("href", "")
            published = entry.findtext("a:published", default="", namespaces=ns)
            authors = [
                author.findtext("a:name", default="", namespaces=ns)
                for author in entry.findall("a:author", ns)
            ]
            results.append(
                Source(
                    provider="arXiv",
                    title=" ".join(
                        (entry.findtext("a:title", default="", namespaces=ns) or "").split()
                    ),
                    url=page_url,
                    authors=[a for a in authors if a],
                    year=int(published[:4]) if published[:4].isdigit() else None,
                    abstract=" ".join(
                        (entry.findtext("a:summary", default="", namespaces=ns) or "").split()
                    ),
                    pdf_url=pdf_url,
                )
            )
        return results

    async def openalex(self, query: str, limit: int) -> list[Source]:
        params = {"search": query, "per-page": limit, "sort": "relevance_score:desc"}
        response = await self._http().get("https://api.openalex.org/works", params=params)
        response.raise_for_status()
        results: list[Source] = []
        for item in response.json().get("results", []):
            authors = [
                auth.get("author", {}).get("display_name", "")
                for auth in item.get("authorships", [])[:12]
            ]
            location = item.get("best_oa_location") or item.get("primary_location") or {}
            results.append(
                Source(
                    provider="OpenAlex",
                    title=item.get("display_name") or "Untitled",
                    url=item.get("doi") or item.get("id") or "",
                    authors=[a for a in authors if a],
                    year=item.get("publication_year"),
                    abstract=_rebuild_abstract(item.get("abstract_inverted_index")),
                    citation_count=int(item.get("cited_by_count") or 0),
                    pdf_url=location.get("pdf_url") or "",
                )
            )
        return results

    async def semantic_scholar(self, query: str, limit: int) -> list[Source]:
        params = {
            "query": query,
            "limit": limit,
            "fields": "title,url,abstract,authors,year,citationCount,openAccessPdf",
        }
        response = await self._http().get(
            "https://api.semanticscholar.org/graph/v1/paper/search", params=params
        )
        response.raise_for_status()
        results: list[Source] = []
        for item in response.json().get("data", []):
            pdf = item.get("openAccessPdf") or {}
            results.append(
                Source(
                    provider="Semantic Scholar",
                    title=item.get("title") or "Untitled",
                    url=item.get("url") or "",
                    authors=[a.get("name", "") for a in item.get("authors", []) if a.get("name")],
                    year=item.get("year"),
                    abstract=item.get("abstract") or "",
                    citation_count=int(item.get("citationCount") or 0),
                    pdf_url=pdf.get("url") or "",
                )
            )
        return results

    async def github(self, query: str, limit: int) -> list[Source]:
        headers = {"Accept": "application/vnd.github+json"}
        if self.settings.github_token:
            headers["Authorization"] = f"Bearer {self.settings.github_token}"
        response = await self._http().get(
            "https://api.github.com/search/repositories",
            params={"q": f"{query} in:name,description,readme", "per_page": limit},
            headers=headers,
        )
        response.raise_for_status()
        return [
            Source(
                provider="GitHub",
                kind="code",
                title=item.get("full_name") or item.get("name") or "repository",
                url=item.get("html_url") or "",
                abstract=item.get("description") or "",
                citation_count=int(item.get("stargazers_count") or 0),
                year=int((item.get("updated_at") or "0000")[:4]) or None,
            )
            for item in response.json().get("items", [])
        ]

    async def download_pdfs(
        self, sources: list[Source], destination: Path, limit: int = 3
    ) -> list[Path]:
        destination.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for source in sources:
            if len(written) >= limit or not source.pdf_url:
                continue
            parsed = urlparse(source.pdf_url)
            if parsed.scheme != "https" or not (
                parsed.hostname == "arxiv.org" or parsed.hostname == "export.arxiv.org"
            ):
                continue
            try:
                async with self._http().stream("GET", source.pdf_url) as response:
                    response.raise_for_status()
                    content_length = int(response.headers.get("content-length", "0") or 0)
                    if content_length > 25_000_000:
                        continue
                    data = bytearray()
                    async for chunk in response.aiter_bytes():
                        data.extend(chunk)
                        if len(data) > 25_000_000:
                            data.clear()
                            break
                if data.startswith(b"%PDF"):
                    target = destination / f"{safe_slug(source.title)}.pdf"
                    target.write_bytes(data)
                    written.append(target)
            except (httpx.HTTPError, ValueError):
                continue
        return written


def _rebuild_abstract(index: dict[str, list[int]] | None) -> str:
    if not index:
        return ""
    positions = [(position, word) for word, values in index.items() for position in values]
    return " ".join(word for _, word in sorted(positions))

