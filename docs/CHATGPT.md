# 在 ChatGPT 中使用 AutoResearch

## 分工

- ChatGPT 桌面端：科研讨论、问题拆解、假设判断、方案取舍、结果分析。
- AutoResearch：项目持久化、文献与代码检索、代码落盘、OpenRouter 代码生成、AutoDL 实例和远程实验、状态与结果展示。
- OpenRouter：只在调用 `generate_experiment_code` 或使用网页中的全自动模式时使用。

这套方式不会把 ChatGPT 对话改成 OpenAI API 请求，也不需要在 AutoResearch 中填写 OpenAI API Key。

## 一次性连接

1. 双击项目根目录的 `start.cmd`，确认网页显示“本地服务与 ChatGPT 工具已就绪”。
2. 点击首页“首次配置连接”，或双击根目录的 `setup-chatgpt.cmd`。向导会检查本机服务并打开 [OpenAI Platform 的 tunnel settings](https://platform.openai.com/settings/organization/tunnels)。
3. 在登录账号中创建 Secure MCP Tunnel，取得 `tunnel_id` 与运行时 API key，并从 Platform 页面下载 Windows 版 `tunnel-client.exe`。这三项属于账号权限操作，不能由本地软件代替。
4. 回到向导，输入下载文件路径、`tunnel_id` 和运行时 API key。向导会自动执行以下 HTTP MCP 配置，并运行 `doctor`：

   ```text
   --mcp-server-url http://127.0.0.1:8765/mcp
   ```

5. 运行时 API key 会通过 Windows DPAPI 加密保存到当前 Windows 用户的本地应用配置目录；原始 key 不会写入本项目、`.env` 或 Git。
6. 在 ChatGPT 打开“设置 → Security and login”，启用 Developer mode。开发者模式是否可用取决于账号和工作区策略。
7. 前往 [ChatGPT Plugins](https://chatgpt.com/plugins)，点击加号，连接方式选择 Tunnel，再选择或填写 `tunnel_id`，创建 AutoResearch 连接。
8. 新建 ChatGPT 对话，从工具菜单启用 AutoResearch 连接。
9. 以后双击 `start-chatgpt.cmd`；它会检查并启动 AutoResearch，再保持 Tunnel 运行。不要在使用期间关闭这个窗口。

也可以在 AutoResearch 首页或“设置与连接”窗口里直接启动首次配置向导和日常连接窗口。网页每 5 秒更新 Tunnel 状态。

本机 MCP 服务地址仍是 `http://127.0.0.1:8765/mcp`。这个地址可用于 MCP Inspector 本地检查，但不能直接填入 ChatGPT 的公网 URL 连接框。

官方参考：[Secure MCP Tunnel](https://developers.openai.com/api/docs/guides/secure-mcp-tunnels)、[连接并测试插件](https://developers.openai.com/plugins/deploy/connect-chatgpt)。

## 建议的对话流程

可以在 ChatGPT 中直接说：

> 我们研究“小样本医学图像分割中的不确定性采样”。你负责科研思考；请创建 AutoResearch 协作项目，把关键研究计划保存进去。需要实验代码时委托 AutoResearch 的 OpenRouter 模型生成，先不要创建 AutoDL 实例。

代码生成后：

> 读取项目状态和实验清单，说明即将执行的代码与命令。得到我确认后，再创建 AutoDL 实例并启动实验。

实验完成后：

> 读取实验指标和日志，分析失败原因、可信结论与下一轮变量控制，并把分析保存回 AutoResearch。

## 工具清单

| 工具 | 用途 | 外部副作用 |
|---|---|---|
| `list_research_projects` | 列出项目和网页地址 | 无 |
| `create_research_project` | 创建 ChatGPT 协作项目 | 本地写入 |
| `get_research_status` | 读取状态、事件、来源、产物和实验 | 无 |
| `read_project_artifact` | 读取非敏感文本产物 | 无 |
| `save_research_note` | 保存 ChatGPT 的研究计划或阶段结论 | 本地写入 |
| `search_research_sources` | 跨论文数据库和 GitHub 检索并保存来源 | 网络请求、本地写入 |
| `save_experiment_code` | 保存 ChatGPT 已生成的完整代码文件 | 本地写入 |
| `generate_experiment_code` | 调用 AutoResearch 的 OpenRouter 模型生成并保存代码 | 模型 API 请求、本地写入 |
| `create_autodl_instance` | 创建计费 AutoDL Pro 实例 | 可能立即计费，必须确认 |
| `get_autodl_instance_status` | 检查实例和 SSH 是否就绪，不返回凭据 | AutoDL 只读请求 |
| `start_autodl_experiment` | 内部取得 SSH、上传代码并运行命令 | 远程执行，必须确认 |
| `get_experiment_result` | 读取状态、退出码、指标和日志末尾 | 无 |
| `record_chatgpt_analysis` | 保存 ChatGPT 的结果分析和建议 | 本地写入 |

## 安全说明

- 不要在 ChatGPT 对话中粘贴 OpenRouter Key、AutoDL Token 或 SSH 密码。
- ChatGPT 工具拿不到这些密钥；AutoResearch 只从本机 `.env` 读取。
- 创建 AutoDL 必须在当前对话明确确认费用，远程执行必须先审阅代码和命令并再次确认。
- Secure MCP Tunnel 的运行时 API key 应只配置在 `tunnel-client` 的安全运行环境中，不要写进本项目或聊天。
- 配置向导只在进程环境中把解密后的 key 提供给 `tunnel-client`；磁盘上的副本由 Windows 当前用户 DPAPI 保护。
- AutoResearch 网页和 MCP 服务默认只监听 `127.0.0.1`。不要改成 `0.0.0.0`，除非已经配置认证与防火墙。
