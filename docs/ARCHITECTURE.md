# 架构

AutoResearch 借鉴 agent harness 的可替换能力设计，但保持单机 MVP 的部署复杂度。核心编排器只负责阶段推进和持久化，各外部能力放在 `autoresearch/services/`：

```text
ChatGPT native model → Secure MCP Tunnel → /mcp → ChatGPTBridge
                                                   ├→ project notes / code
                                                   ├→ ResearchSearch
                                                   ├→ OpenRouter (delegated code only)
                                                   ├→ ExperimentManager → AutoDL / SSH
                                                   └→ SQLite event log

Chrome UI → FastAPI → ResearchOrchestrator → OpenRouter
                            ├──────────────→ Search providers
                            ├──────────────→ ArtifactStore
                            └──────────────→ SQLite event log

ExperimentManager → SSHRunner → AutoDL instance
                 ├→ AutoDL Pro API (provision/recovery)
                 ├→ result analysis / bounded improvement
                 └→ GitSync (code only)
```

## 持久事实

SQLite 的 `projects`、`events`、`sources` 和 `experiments` 是 UI 和状态恢复的事实来源。事件是只追加日志，项目表保留当前投影。这与 harness 的“模型可见事实应可重建”原则一致。

ChatGPT 和网页共享同一个 `AppState`、SQLite 数据库和工作区。MCP 工具的结果返回项目 ID 与网页深链接；网页每隔 2.5 秒重新读取事件和实验表，所以通过 ChatGPT 发起的操作也会显示在现有工作台中。

## 能力边界

- `OpenRouterClient`：模型请求、免费模型发现、JSON 提取和有限重试。
- `ResearchSearch`：论文与代码检索、去重、受限 PDF 下载。
- `ArtifactStore`：模型文件落盘，阻止绝对路径、`..`、敏感文件名和过大输出。
- `AutoDLClient`：官方 Pro API；GPU 规格顺序由配置注入。
- `SSHRunner`：GPU 检查、受控目录上传、守护进程和指标读取。
- `GitSync`：仅允许 github.com 远端，拒绝 URL 内嵌 Token。
- `DesktopBridge`：明确寻找 Chrome，调用 VS Code CLI。
- `ChatGPTBridge`：把 ChatGPT 工具调用映射到既有数据库、产物、检索、AutoDL 与实验服务；不保存 ChatGPT 对话，也不调用 OpenAI API。
- `MCPServer`：在同一 FastAPI 进程的 `/mcp` 暴露 Streamable HTTP 工具，使用准确的只读、写入、开放网络和远程执行注解。

服务均可通过构造器替换或在测试中模拟。后续可把搜索器、模型提供商和算力提供商升级成入口点插件，而无需改动工作流数据模型。

## 状态机

```text
queued → planning → searching → synthesizing → generating → ready
                                                        ↓
                  completed ← analyzing ← experiment ←─┘

chatgpt_thinking → searching → chatgpt_thinking → generating/ready
                                                        ↓
                                chatgpt_analysis ← experiment
```

任何阶段异常进入 `failed` 或 `experiment_failed`，错误同时写入项目快照和事件日志。
