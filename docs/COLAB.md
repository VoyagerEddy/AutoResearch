# Colab 免费 GPU

在 ChatGPT 中使用 AutoResearch 插件保存研究计划和实验代码。ChatGPT 调用
`prepare_colab_experiment` 生成 notebook，再用已获得的执行授权调用
`start_colab_experiment`。Windows 桌面的 `AutoResearch` 快捷方式启动或复用本地服务，
打开只读仪表板，显示项目进度、GPU 状态和等待原因。

工作器使用独立的可见 Chrome/Edge 配置并复用已登录的会话。启动时会校验绑定的 Drive
notebook 与准备产物的源码校验和，然后通过 Colab 界面选择免费 GPU 并运行。

如果 Google 要求重新登录、验证或同意 Drive 授权，相关操作在专用浏览器中完成。
检测到登录成功后，工作器会继续已授权的作业。如果 Google 要求普通浏览器登录，
调度器会打开使用同一专用配置的普通 Chrome/Edge，登录窗口关闭后重新检查会话。
ChatGPT 也可以调用 `open_colab_login` 打开专用登录窗口。

Drive 授权暂停会保留已连接的 notebook。工作器在确认同一运行时已经继续执行后
自动更新状态。仍处于授权暂停时，ChatGPT 可以使用原执行授权调用
`resume_colab_experiment`，继续观察现有运行时。

作业启动后，AutoResearch 会保存自动运行授权。明确的 GPU 不可用、连接故障或会话
断开会进入持久重试队列，默认随机等待 5–10 分钟。账号达到 GPU 配额时等待更久：
优先采用 Colab 明确提供的冷却时间；没有时间提示时程序保守默认等待 12 小时，这不是 Google 保证的额度重置时间。
额度等待按默认登录浏览器配置共享，跨项目、重新准备和本地程序重启仍有效；重复出现同一额度提示不会不断推迟截止时间，明确更长的官方等待会优先采用。
连接请求超时后，工作器仍只读检查迟到的额度或登录提示，不会为此再次申请 GPU。
只有比额度提示更新、足够新鲜且带真实 GPU 名称与显存信息的当前运行时检查，才能清除已过时的共享等待；历史输出和空的“可用”记录不会清除它。

保持本地 AutoResearch 服务运行，调度器会在保存的时间到达后自动启动缺失的浏览器
工作器；无需手动保持工作器窗口打开。关闭或重启本地服务不会清除授权和重试时间，
下次启动服务后会继续检查队列。等待 GPU 时工作器释放浏览器配置，资源 Agent 继续展示
状态，调度器到点再启动工作器。ChatGPT 调用 `cancel_colab_retries` 可取消后续重试。
如需停止仍在训练的 Colab 运行时，请在 Colab 页面停止。

等待流程包含一个专用的“Colab 资源检查 Agent”。启动已授权的任务后，服务会自动
维持这个独立后台进程；它每 10 秒跟踪申请队列和工作器的真实结果，并刷新项目面板。
实际 GPU 连接检查仍按 5–10 分钟的随机重试安排执行；账号额度冷却优先。
Agent 不依赖浏览器窗口持续打开，也不会因页面刷新额外申请 GPU。

面板显示当前账号“有 / 暂不可用 / 待确认”、Agent 状态和最近心跳、最近一次真实检查
结果及时间、下次实际检查时间。Agent 心跳只表示监测程序活着，不能当作新的 GPU
检查；历史结果、连接异常或额度冷却期间会显示“待确认”，并保留最近一次结果。
成功分配 GPU 后，既有流程继续启动试验及 checkpoint 保存；停止重试或任务结束后
Agent 自动退出。它只观察当前账号的连接结果，不推断 Colab 全局空闲显卡库存。

也可以单独启动当前项目的监测进程（API 地址使用正在运行的 AutoResearch 服务）：

```powershell
.venv\Scripts\python.exe -m autoresearch.colab_resource_monitor_cli --launch --project "项目ID" --api-url "http://127.0.0.1:8767"
```

Notebook 通过正常 Drive 挂载保存 `inputs/`、`checkpoints/`、`results/` 和 `session_status.json`。
项目需要声明完整 checkpoint 与恢复命令；恢复训练会读取已有 checkpoint。
Colab 断开时仍可能丢失最后一次 checkpoint 之后的训练进度。
工作器通过当前 notebook 输出的运行 ID 和退出码判断结果，不把历史输出当成当前成功。
全量上传前，notebook 会检查 Drive 写入和临时盘容量，并执行项目声明的预检命令。

ChatGPT 使用 `get_colab_status` 查看登录、授权或排队状态，使用 `get_colab_resources`
查看 Agent 的最近一次检测。完成后调用 `get_colab_experiment_result` 获取当前运行的指标，
再按 `get_research_workflow` 返回的实验覆盖情况记录科研决策。

排查工作器时，可从命令行运行已由插件准备并授权的项目。

```powershell
.\colab-browser.ps1 -ProjectId "项目ID" -Browser chrome
```

后台辅助器日志位于 `data/colab-browser/<项目ID>.log`；Google 登录状态只留在
`data/colab-browser/profile/`。不要共享该目录。
首次安装可选浏览器依赖：`.venv\Scripts\python.exe -m pip install -e ".[browser]"`。

可选的官方 [Colab Notebook MCP](https://github.com/googlecolab/colab-mcp) 辅助进程使用
独立 Python 3.13 环境 `data/colab-mcp/venv/`，不改变主程序的 Python 环境。
常规工作器通过标准 notebook 界面导入和运行。需要逐单元格接管时，可用
`--notebook-bridge` 启动 CLI，或在启动本地服务前设置
`AUTORESEARCH_COLAB_NOTEBOOK_BRIDGE=1`。启用后，工作器检测到已安装的环境和
`data/colab-mcp/speaker-session/` 时，通过 notebook 的
正常连接界面使用它编写、执行单元格并读取当前输出。它不负责申请 GPU；运行时类型
选择和连接仍通过 Colab 界面完成。未安装该辅助进程时可继续使用 notebook 上传流程。
MCP 临时连接信息保存在本地私有文件中，不要共享 `data/colab-mcp/`。

免费 GPU 与运行时上限由 [Colab 官方规则](https://research.google.com/colaboratory/faq.html) 决定。
此模式不使用 SSH、付费 GPU、后台保活、内部申请 API 或多账号绕过限制。
