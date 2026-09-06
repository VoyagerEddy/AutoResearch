from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from .config import Settings
from .db import Database
from .domain import Project, ResearchRequest, Source
from .services.artifacts import ArtifactStore, fallback_code_bundle
from .services.llm import LLMError, OpenRouterClient
from .services.search import ResearchSearch


class ResearchOrchestrator:
    def __init__(self, db: Database, settings: Settings) -> None:
        self.db = db
        self.settings = settings
        self._tasks: dict[str, asyncio.Task[None]] = {}

    def update_settings(self, settings: Settings) -> None:
        self.settings = settings

    def create_and_start(self, request: ResearchRequest) -> Project:
        project = self.db.create_project(request, self.settings.workspace_dir)
        task = asyncio.create_task(self.run(project.id, request), name=f"research-{project.id}")
        self._tasks[project.id] = task
        task.add_done_callback(lambda _: self._tasks.pop(project.id, None))
        return project

    def _phase(self, project_id: str, phase: str, progress: int, message: str) -> None:
        self.db.update_project(
            project_id, status="running", phase=phase, progress=progress, error=""
        )
        self.db.add_event(project_id, phase, message)

    async def run(self, project_id: str, request: ResearchRequest) -> None:
        project = self.db.get_project(project_id)
        if not project:
            return
        workspace = Path(project.workspace)
        store = ArtifactStore(workspace)
        searcher = ResearchSearch(self.settings)
        try:
            self._phase(project_id, "planning", 8, "Breaking the research idea into searchable questions")
            plan = await self._plan(request)
            store.write_json("research/plan.json", plan)
            queries = [str(q)[:300] for q in plan.get("queries", []) if str(q).strip()]
            if not queries:
                queries = [request.topic]

            self._phase(project_id, "searching", 25, "Searching papers and open-source code")
            sources = await searcher.search(queries, request.max_sources)
            self.db.replace_sources(project_id, sources)
            store.write_json(
                "research/sources.json", [source.model_dump() for source in sources]
            )
            self.db.add_event(
                project_id,
                "searching",
                f"Deduplicated and saved {len(sources)} paper/code sources",
                details={"source_count": len(sources)},
            )
            if request.download_papers:
                downloaded = await searcher.download_pdfs(
                    sources, workspace / "research" / "papers"
                )
                self.db.add_event(
                    project_id, "searching", f"Downloaded {len(downloaded)} open-access PDFs"
                )

            self._phase(project_id, "synthesizing", 52, "Synthesizing evidence into an algorithm plan")
            report = await self._synthesize(request, plan, sources)
            store.write_text("RESEARCH.md", report)

            self._phase(project_id, "generating", 74, "Generating reproducible experiment code")
            bundle = await self._generate(request, report, sources)
            files = bundle.get("files")
            if not isinstance(files, list) or not files:
                bundle = fallback_code_bundle(request.topic)
                files = bundle["files"]
            written = store.materialize_files(files, "generated")
            store.write_json("generated/experiment_manifest.json", bundle.get("experiment", {}))
            store.write_json(
                "research/run.json",
                {
                    "topic": request.topic,
                    "model": request.model or self.settings.openrouter_model,
                    "queries": queries,
                    "source_count": len(sources),
                    "generated_files": [str(path.relative_to(workspace)) for path in written],
                },
            )
            summary = str(bundle.get("summary") or plan.get("objective") or request.topic)[:2000]
            self.db.update_project(
                project_id,
                status="ready",
                phase="ready",
                progress=100,
                summary=summary,
                model=request.model or self.settings.openrouter_model,
            )
            self.db.add_event(
                project_id,
                "ready",
                f"The research plan and {len(written)} experiment files are ready",
                details={"workspace": str(workspace)},
            )
        except Exception as exc:
            self.db.update_project(
                project_id,
                status="failed",
                phase="failed",
                progress=100,
                error=str(exc)[:4000],
            )
            self.db.add_event(
                project_id, "failed", f"Research workflow failed: {exc}", level="error"
            )
        finally:
            await searcher.close()

    async def _plan(self, request: ResearchRequest) -> dict[str, Any]:
        fallback = {
            "objective": request.topic,
            "hypotheses": ["Establish a reproducible baseline and compare it with one improvement"],
            "queries": [request.topic, f"{request.topic} benchmark", f"{request.topic} github"],
            "evaluation": ["Define the data split", "Report mean and variance", "Fix random seeds"],
        }
        if not self.settings.openrouter_api_key:
            return fallback
        prompt = f"""You are a rigorous research planner. Turn the topic into a testable research plan.
Topic: {request.topic}
Additional requirements: {request.notes or 'None'}
Return only a JSON object with these fields: objective (string), hypotheses (string[]), queries (string[] with 3 to 5 English search queries), and evaluation (string[])."""
        try:
            async with OpenRouterClient(self.settings) as llm:
                return await llm.chat_json(
                    [
                        {"role": "system", "content": "Your output must be strict JSON. Do not fabricate experimental results."},
                        {"role": "user", "content": prompt},
                    ],
                    model=request.model,
                )
        except LLMError:
            return fallback

    async def _synthesize(
        self, request: ResearchRequest, plan: dict[str, Any], sources: list[Source]
    ) -> str:
        digest = _source_digest(sources)
        if self.settings.openrouter_api_key:
            prompt = f"""Write an English research plan based on the topic and retrieved sources below.
Topic: {request.topic}
Plan: {json.dumps(plan, ensure_ascii=False)}
Sources:
{digest}

Requirements: include the problem definition, related work with [S1]-style citations, falsifiable hypotheses, algorithm design, data and baselines, evaluation metrics, ablation studies, risks, and reproduction steps. Do not claim results from experiments that have not been run."""
            try:
                async with OpenRouterClient(self.settings) as llm:
                    return await llm.chat(
                        [
                            {
                                "role": "system",
                                "content": "You are the research lead. Use only the supplied sources and clearly distinguish facts, inferences, and hypotheses that still require testing.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        model=request.model,
                        max_tokens=7000,
                    )
            except LLMError:
                pass
        bibliography = "\n".join(
            f"- [S{index}] [{source.title}]({source.url}) — {source.provider}"
            for index, source in enumerate(sources, 1)
        ) or "- No sources were retrieved. Add literature manually before running the study."
        return f"""# {request.topic}

## Research Objective

{plan.get('objective', request.topic)}

## Hypotheses and Evaluation

""" + "\n".join(f"- {item}" for item in plan.get("hypotheses", [])) + f"""

## Implementation Principles

- Run a deterministic baseline first, then change one variable for comparison.
- Fix random seeds and save the configuration, environment, and metrics.
- This document is an experiment plan; it does not represent completed findings.

## Retrieved Sources

{bibliography}
"""

    async def _generate(
        self, request: ResearchRequest, report: str, sources: list[Source]
    ) -> dict[str, Any]:
        fallback = fallback_code_bundle(request.topic)
        if not self.settings.openrouter_api_key:
            return fallback
        prompt = f"""Generate a minimal, runnable, and reproducible Python experiment project from the research plan.
Research plan:
{report[:24000]}

Return strict JSON only:
{{
  "summary": "Implementation summary",
  "files": [{{"path": "relative/path", "content": "complete file content"}}],
  "experiment": {{
    "setup_commands": ["safe, non-interactive installation commands"],
    "dataset_commands": ["idempotent dataset download commands"],
    "run_command": "main experiment command",
    "metrics_file": "results/metrics.json"
  }}
}}
Include README.md, requirements.txt, experiment.py or an equivalent entry point, a test or validation script, and .gitignore. Do not write secrets, use absolute paths, or fabricate experiment values."""
        try:
            async with OpenRouterClient(self.settings) as llm:
                return await llm.chat_json(
                    [
                        {
                            "role": "system",
                            "content": "You are a machine-learning engineer. Return complete files, not patches or Markdown code fences.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    model=request.model,
                    max_tokens=12000,
                )
        except LLMError:
            return fallback

    async def analyze_and_improve(
        self, project_id: str, experiment_id: str, result: dict[str, Any], iteration: int
    ) -> bool:
        project = self.db.get_project(project_id)
        if not project:
            return False
        store = ArtifactStore(Path(project.workspace))
        if not self.settings.openrouter_api_key:
            store.write_text(
                f"reports/{experiment_id}-iteration-{iteration}.md",
                "# Experiment Analysis\n\nNo OpenRouter key is configured. Logs and metrics were preserved for manual analysis.\n",
            )
            return False
        generated = Path(project.workspace) / "generated"
        inventory = []
        for path in generated.rglob("*"):
            if path.is_file() and path.stat().st_size < 200_000:
                inventory.append(str(path.relative_to(generated)))
        prompt = f"""Analyze the experiment results and propose one conservative algorithm improvement.
Topic: {project.topic}
Iteration: {iteration}
Results: {json.dumps(result, ensure_ascii=False)[:16000]}
Current files: {inventory}

Return JSON only: {{"analysis":"Markdown analysis","should_continue":true/false,"files":[{{"path":"relative/path","content":"complete replacement file content"}}]}}.
Set should_continue=false if the results are insufficient or the experiment failed. Do not modify the system environment beyond dependencies, and do not write secrets."""
        try:
            async with OpenRouterClient(self.settings) as llm:
                answer = await llm.chat_json(
                    [
                        {"role": "system", "content": "Change only a small number of variables at a time and avoid overfitting to noise from a single run."},
                        {"role": "user", "content": prompt},
                    ],
                    model=project.model or None,
                    max_tokens=10000,
                )
        except LLMError as exc:
            self.db.add_event(project_id, "analyzing", f"Model analysis failed: {exc}", level="warning")
            return False
        store.write_text(
            f"reports/{experiment_id}-iteration-{iteration}.md",
            str(answer.get("analysis", "No analysis was returned.")),
        )
        files = answer.get("files")
        if not answer.get("should_continue") or not isinstance(files, list) or not files:
            return False
        store.materialize_files(files, "generated")
        store.materialize_files(files, f"iterations/iteration-{iteration + 1}")
        self.db.add_event(project_id, "improving", "Generated the next algorithm improvement from the results")
        return True


def _source_digest(sources: list[Source]) -> str:
    chunks: list[str] = []
    for index, source in enumerate(sources, 1):
        chunks.append(
            f"[S{index}] {source.title}\n"
            f"URL: {source.url}\nProvider: {source.provider}; Year: {source.year}; "
            f"Citations/Stars: {source.citation_count}\n"
            f"Abstract: {source.abstract[:1200]}"
        )
    return "\n\n".join(chunks)[:36000]
