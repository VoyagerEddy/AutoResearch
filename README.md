# AutoResearch

AutoResearch 是一个运行在本机的科研执行工作台。推荐工作方式是：在 ChatGPT 桌面端使用当前对话和所选 ChatGPT 模型进行科研思考；需要保存研究笔记或代码、检索资源、创建 AutoDL 实例、启动实验、读取状态与结果时，再由 ChatGPT 调用 AutoResearch 的本地 MCP 工具。项目状态、来源、日志、指标和分析报告持续显示在 AutoResearch 网页中。

原有的 OpenRouter 全自动研究模式仍然保留。协作模式下，`generate_experiment_code` 才会调用 AutoResearch 配置的 OpenRouter 模型；`save_experiment_code` 只是安全保存 ChatGPT 已生成的代码，不产生第二次模型调用。

> 当前版本是可运行的 MVP，不应在无人监管的生产机器上执行模型生成代码。任何 AutoDL 计费实例创建、SSH 执行和可能释放实例的操作，都需要在界面中明确确认。

## 快速开始（Windows）

要求：Python 3.11+、Git；建议安装 Chrome 与 VS Code。

1. 双击 `start.cmd`。首次启动会创建 `.venv`、安装依赖并生成本地 `.env`。
2. Chrome 会打开 `http://127.0.0.1:8765`。
3. 默认选择「ChatGPT 协作」，创建项目后把项目 ID 用在 ChatGPT 对话中。
4. 首次连接 ChatGPT 时，点击首页「首次配置连接」或双击 `setup-chatgpt.cmd`；以后双击 `start-chatgpt.cmd` 即可同时保持本机服务和 Tunnel 可用。
5. 进入「设置与连接」，填写 OpenRouter Key（仅委托 AutoResearch 生成代码或使用全自动模式时需要）。
5. 如需 AutoDL，在 AutoDL 控制台的「账号 → 设置」获取开发者 Token，并填写镜像 UUID。

也可以在 PowerShell 中运行：

```powershell
./start.ps1
```

只启动服务、不打开浏览器：

```powershell
.venv/Scripts/python.exe -m autoresearch --no-open
```

## 工作流

### ChatGPT 协作模式（推荐）

1. ChatGPT 使用当前软件内的模型讨论问题、提出假设并分析结果。
2. `create_research_project` 在 AutoResearch 创建协作项目。
3. 按需调用 `save_research_note`、`search_research_sources`、`save_experiment_code` 或 `generate_experiment_code`。
4. 用户确认后，调用 `create_autodl_instance` 和 `start_autodl_experiment`。
5. ChatGPT 用 `get_experiment_result` 读取指标与日志并完成分析，再用 `record_chatgpt_analysis` 保存结论。
6. AutoResearch 网页通过项目 ID 深链接显示全部阶段、来源、实验状态、指标、日志和报告。

AutoResearch 不需要 OpenAI API Key。ChatGPT 的对话与推理由 ChatGPT 软件本身提供；OpenRouter Key 只由 AutoResearch 在明确委托代码生成或原有全自动流程时使用。

一份完整的本地连接步骤见 [`docs/CHATGPT.md`](docs/CHATGPT.md)。本机 MCP 地址为：

```text
http://127.0.0.1:8765/mcp
```

ChatGPT 不能直接连接环回地址，需要在开发者模式下使用 Secure MCP Tunnel（或自行部署的公开 HTTPS MCP 地址）。Tunnel 是向外连接，不要求把本机端口暴露到公网。项目附带 Windows 配置向导：它会检查/启动本机 MCP、调用 `tunnel-client init` 和 `doctor`、用当前 Windows 用户的 DPAPI 加密保存运行时 key，并打开 ChatGPT Plugins 页面。创建 Tunnel、启用 Developer mode 和首次添加连接仍必须由登录账号本人完成。

### Windows 一键连接

- 首次：双击 `setup-chatgpt.cmd`，按提示从 OpenAI Platform 下载 `tunnel-client.exe`，输入 `tunnel_id` 和运行时 API key。Key 输入不可见，且不会写入仓库或 `.env`。
- 日常：双击 `start-chatgpt.cmd`。脚本会在需要时启动 AutoResearch，并在当前窗口持续运行 Tunnel；使用 ChatGPT 期间不要关闭该窗口。
- 诊断：运行 `powershell -NoProfile -ExecutionPolicy Bypass -File .\chatgpt-tunnel.ps1 -Action Doctor`。
- 状态：网页会分别显示 MCP、客户端安装、Tunnel 配置和 Tunnel 运行状态。

### AutoResearch 全自动模式

1. **规划**：把题目转成目标、假设、检索式和评估计划。
2. **检索**：查询多个论文/代码来源，按标题去重并保存真实 URL。
3. **综合**：生成带 `[S1]` 等来源编号的研究方案；明确区分事实、推断与待验证假设。
4. **生成**：将完整实验项目写入 `workspaces/<项目ID>/generated/`。
5. **实验**：用户确认命令后，检查远端 GPU、上传代码并用 `nohup` 启动。
6. **迭代**：定期读取状态、日志和 `results/metrics.json`，在设定轮数内分析和小步改进。
7. **同步**：只同步代码目录；`.env`、论文、数据和实验结果默认被排除。

如果 OpenRouter 未配置或暂时不可用，系统仍会生成一个确定性的离线基线，让项目流程可以检查；这不等同于完成领域算法设计。

## AutoDL

AutoResearch 使用官方[容器实例 Pro API](https://www.autodl.com/docs/instance_pro_api/)，不保存 AutoDL 账号密码，也不依赖易碎的网页模拟登录。默认 GPU 规格顺序为：

1. `v-48g`（官方 API 当前列出的 4090-48G 规格）
2. `5090`（预留的可配置规格 ID；如官方实际 ID 不同，请在设置中替换）

如果 SSH 检查发现 GPU 上已有计算进程，恢复策略是：保存原实例镜像并创建克隆实例；克隆仍忙时，仅在用户勾选授权后释放这个替代实例并用基础镜像全新创建。为避免无限计费，最多走一轮克隆和一轮重建。AutoDL 官方也建议使用守护进程运行训练，参见[SSH 文档](https://api.autodl.com/docs/ssh/)和[VS Code 文档](https://www.autodl.com/docs/vscode/)。

## 配置

复制 `.env.example` 为 `.env`，或在网页设置中填写。关键项：

```dotenv
OPENROUTER_API_KEY=
OPENROUTER_MODEL=openrouter/free
AUTODL_TOKEN=
AUTODL_IMAGE_UUID=
AUTODL_GPU_SPECS=v-48g,5090
GITHUB_REMOTE_URL=https://github.com/VoyagerEddy/AutoResearch.git
```

真实密钥绝不能提交。设置 API 只返回“是否已配置”，不会回显 Key 或 Token。

## 实验项目约定

生成项目的 `experiment_manifest.json` 描述安装、数据下载、运行命令和指标位置。当前远程监控读取：

```text
results/metrics.json
```

实验入口应返回非零状态表示失败。数据下载命令和依赖安装命令会生成在清单中供检查；当前 UI 只执行用户确认后的“实验命令”，不会静默执行清单里的额外命令。

通过 ChatGPT 工具启动 AutoDL 实验时，AutoResearch 会直接从 AutoDL API 取得 SSH 连接，不把 SSH 密码、AutoDL Token 或 OpenRouter Key 返回给 ChatGPT。创建计费实例需要 `confirm_billable=true`，上传和运行代码需要 `confirm_execute=true`；这两个值只能在用户当前对话明确同意后设置。

## 开发与测试

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"
.venv/Scripts/python.exe -m pytest
```

结构说明见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)，安全边界见 [`docs/SECURITY.md`](docs/SECURITY.md)。

## 已知边界

- AutoDL Pro API 需要实名认证、开发者 Token 和可用镜像；普通网页市场上的所有 GPU 不一定都有 Pro API 规格。
- SSH 密码只保存在当前进程内存；服务重启后，正在监控的远程任务不会自动恢复连接。
- GitHub 仓库需预先创建，本机 Git 需已经通过凭据管理器或 SSH 完成认证。
- 生成代码运行前仍应人工阅读；沙箱和确认机制不能消除模型生成代码的全部风险。
