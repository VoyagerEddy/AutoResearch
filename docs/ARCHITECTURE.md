# Architecture

AutoResearch borrows the replaceable-capability structure of an agent harness while keeping the deployment complexity of a single-machine MVP. The core orchestrator advances and persists phases. External capabilities live under `autoresearch/services/`:

```text
ChatGPT native model → Secure MCP Tunnel → /mcp → ChatGPTBridge
                                                   ├→ project notes / code
                                                   ├→ ResearchSearch
                                                   ├→ OpenRouter (delegated code only)
                                                   ├→ ExperimentManager → AutoDL / SSH
                                                   └→ SQLite event log

Web UI → FastAPI → ResearchOrchestrator → OpenRouter
                          ├──────────────→ Search providers
                          ├──────────────→ ArtifactStore
                          └──────────────→ SQLite event log

ExperimentManager → SSHRunner → AutoDL instance
                 ├→ AutoDL Pro API (provisioning/recovery)
                 ├→ result analysis / bounded improvement
                 └→ GitSync (code only)
```

## Durable state

SQLite tables `projects`, `events`, `sources`, and `experiments` are the source of truth for the UI and state recovery. Events form an append-only log; the projects table retains the current projection.

ChatGPT and the web UI share one `AppState`, SQLite database, and workspace. MCP results include a project ID and dashboard deep link. The UI reloads events and experiments every 2.5 seconds, so operations started through ChatGPT appear in the same workbench.

## Capability boundaries

- `OpenRouterClient`: model requests, free-model discovery, JSON extraction, and bounded retry.
- `ResearchSearch`: paper and code search, deduplication, and restricted PDF downloads.
- `ArtifactStore`: safe model-output writes with rejection of absolute paths, traversal, sensitive filenames, conflicts, and oversized bundles.
- `AutoDLClient`: official Pro API with configuration-driven GPU preference.
- `SSHRunner`: GPU inspection, controlled directory upload, daemonized execution, and metric reads.
- `GitSync`: GitHub remotes only; embedded URL tokens are rejected.
- `DesktopBridge`: explicit Chrome discovery and VS Code CLI invocation.
- `ChatGPTBridge`: maps ChatGPT tool calls to the existing database, artifacts, search, AutoDL, and experiment services. It stores no ChatGPT conversation and calls no OpenAI API.
- `MCPServer`: exposes Streamable HTTP tools at `/mcp` in the FastAPI process, with accurate read, write, open-network, and remote-execution annotations.

Services are constructor-injected or mockable in tests. Search, model, and compute providers can later become entry-point plugins without changing the workflow data model.

## State machines

```text
queued → planning → searching → synthesizing → generating → ready
                                                        ↓
                  completed ← analyzing ← experiment ←─┘

chatgpt_thinking → searching → chatgpt_thinking → generating/ready
                                                        ↓
                                chatgpt_analysis ← experiment
```

Any phase error transitions to `failed` or `experiment_failed`, and the error is persisted in both the project snapshot and event log.
