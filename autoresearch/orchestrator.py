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
            self._phase(project_id, "planning", 8, "正在把科研想法拆成可检索的问题")
            plan = await self._plan(request)
            store.write_json("research/plan.json", plan)
            queries = [str(q)[:300] for q in plan.get("queries", []) if str(q).strip()]
            if not queries:
                queries = [request.topic]

            self._phase(project_id, "searching", 25, "正在检索论文与开源代码")
            sources = await searcher.search(queries, request.max_sources)
            self.db.replace_sources(project_id, sources)
            store.write_json(
                "research/sources.json", [source.model_dump() for source in sources]
            )
            self.db.add_event(
                project_id,
                "searching",
                f"已去重并保存 {len(sources)} 条论文/代码来源",
                details={"source_count": len(sources)},
            )
            if request.download_papers:
                downloaded = await searcher.download_pdfs(
                    sources, workspace / "research" / "papers"
                )
                self.db.add_event(
                    project_id, "searching", f"已下载 {len(downloaded)} 篇开放 PDF"
                )

            self._phase(project_id, "synthesizing", 52, "正在综合证据并形成算法方案")
            report = await self._synthesize(request, plan, sources)
            store.write_text("RESEARCH.md", report)

            self._phase(project_id, "generating", 74, "正在生成可复现实验代码")
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
                f"研究方案和 {len(written)} 个实验文件已经就绪",
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
                project_id, "failed", f"研究流程失败：{exc}", level="error"
            )
        finally:
            await searcher.close()

    async def _plan(self, request: ResearchRequest) -> dict[str, Any]:
        fallback = {
            "objective": request.topic,
            "hypotheses": ["建立可复现基线并与一个改进方法比较"],
            "queries": [request.topic, f"{request.topic} benchmark", f"{request.topic} github"],
            "evaluation": ["明确数据划分", "报告均值和方差", "固定随机种子"],
        }
        if not self.settings.openrouter_api_key:
            return fallback
        prompt = f"""你是严谨的科研规划员。将题目转成可验证的研究计划。
题目：{request.topic}
补充要求：{request.notes or '无'}
只返回 JSON 对象，字段：objective(string)、hypotheses(string[])、queries(string[]，3到5条英文检索式)、evaluation(string[])。"""
        try:
            async with OpenRouterClient(self.settings) as llm:
                return await llm.chat_json(
                    [
                        {"role": "system", "content": "你的输出必须是严格 JSON，不编造实验结果。"},
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
            prompt = f"""围绕下面的研究题目和真实检索结果撰写中文研究方案。
题目：{request.topic}
计划：{json.dumps(plan, ensure_ascii=False)}
来源：
{digest}

要求：包含问题定义、相关工作（用 [S1] 形式引用）、可证伪假设、算法设计、数据与基线、评估指标、消融实验、风险和复现步骤。不得声称尚未运行的结果已经发生。"""
            try:
                async with OpenRouterClient(self.settings) as llm:
                    return await llm.chat(
                        [
                            {
                                "role": "system",
                                "content": "你是科研负责人，只基于提供的来源，清楚区分事实、推断与待验证假设。",
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
        ) or "- 暂未检索到来源；运行前需要人工补充文献。"
        return f"""# {request.topic}

## 研究目标

{plan.get('objective', request.topic)}

## 假设与评估

""" + "\n".join(f"- {item}" for item in plan.get("hypotheses", [])) + f"""

## 实施原则

- 先运行确定性基线，再改变一个变量进行对照。
- 固定随机种子，保存配置、环境和指标。
- 当前文档是实验计划，不代表已有实验结论。

## 检索来源

{bibliography}
"""

    async def _generate(
        self, request: ResearchRequest, report: str, sources: list[Source]
    ) -> dict[str, Any]:
        fallback = fallback_code_bundle(request.topic)
        if not self.settings.openrouter_api_key:
            return fallback
        prompt = f"""根据研究方案生成一个最小但可运行、可复现的 Python 实验项目。
研究方案：
{report[:24000]}

只返回严格 JSON：
{{
  "summary": "实现摘要",
  "files": [{{"path": "相对路径", "content": "完整文件内容"}}],
  "experiment": {{
    "setup_commands": ["安全、非交互的安装命令"],
    "dataset_commands": ["可重复执行的数据下载命令"],
    "run_command": "主实验命令",
    "metrics_file": "results/metrics.json"
  }}
}}
必须包含 README.md、requirements.txt、experiment.py 或等价入口、测试/校验脚本和 .gitignore。不得写入密钥，不得使用绝对路径，不得伪造实验数值。"""
        try:
            async with OpenRouterClient(self.settings) as llm:
                return await llm.chat_json(
                    [
                        {
                            "role": "system",
                            "content": "你是机器学习工程师。输出完整文件，不输出补丁或 Markdown 代码围栏。",
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
                "# 实验分析\n\n没有配置 OpenRouter Key；已保留日志与指标，请人工分析。\n",
            )
            return False
        generated = Path(project.workspace) / "generated"
        inventory = []
        for path in generated.rglob("*"):
            if path.is_file() and path.stat().st_size < 200_000:
                inventory.append(str(path.relative_to(generated)))
        prompt = f"""分析实验结果并提出一次保守的算法改进。
题目：{project.topic}
迭代：{iteration}
结果：{json.dumps(result, ensure_ascii=False)[:16000]}
当前文件：{inventory}

只返回 JSON：{{"analysis":"Markdown 分析","should_continue":true/false,"files":[{{"path":"相对路径","content":"需要替换的完整文件"}}]}}。
如果结果不充分或实验失败，should_continue=false。不能修改依赖以外的系统环境，不能写密钥。"""
        try:
            async with OpenRouterClient(self.settings) as llm:
                answer = await llm.chat_json(
                    [
                        {"role": "system", "content": "一次只改变少量变量，避免根据单次噪声过拟合。"},
                        {"role": "user", "content": prompt},
                    ],
                    model=project.model or None,
                    max_tokens=10000,
                )
        except LLMError as exc:
            self.db.add_event(project_id, "analyzing", f"模型分析失败：{exc}", level="warning")
            return False
        store.write_text(
            f"reports/{experiment_id}-iteration-{iteration}.md",
            str(answer.get("analysis", "没有返回分析。")),
        )
        files = answer.get("files")
        if not answer.get("should_continue") or not isinstance(files, list) or not files:
            return False
        store.materialize_files(files, "generated")
        store.materialize_files(files, f"iterations/iteration-{iteration + 1}")
        self.db.add_event(project_id, "improving", "已根据结果生成下一轮算法改进")
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

