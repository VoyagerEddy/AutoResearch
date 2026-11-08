# Colab 免费 GPU

双击 Windows 桌面的 `AutoResearch` 快捷方式打开本地应用。启动器在后台启动服务；
再次点击会复用当前服务。选择项目后，将计算平台切换到“Google Colab · 免费 GPU”，
即可启动资源检查 Agent，并查看当前账号的 GPU 状态。已准备的任务会按保存的重试
规则检查资源；登录或手动授权暂停会显示需要完成的操作。

尚未准备的项目先在“Colab 免费 GPU”面板中准备 notebook，再启动浏览器工作器。
工作器使用独立的可见 Chrome/Edge 配置，不读取你日常浏览器的 Cookie，也不保存 Google 密码。
首次登录、验证码和 Google Drive 授权由你在该浏览器中完成；出现暂停时，完成面板显示的操作后点击“继续”。

如果 Google 提示自动化浏览器“不安全”，先停止工作器，再用面板“Google 登录”打开普通
Chrome/Edge 完成登录。普通登录窗口使用同一个专用配置，不带自动化或远程调试参数。
登录完成后关闭该普通窗口，再启动工作器；登录状态会被复用。
也可以运行 `.\colab-browser.ps1 -Action Login -Browser chrome`。

如果 Colab 界面改变，工作器会暂停，并显示需要手动导入或选择的步骤。你可以通过
**文件 → 打开 notebook → 上传** 导入面板提供的 `.ipynb`，再选择
**运行时 → 更改运行时类型 → T4 GPU**。工作器只选择免费 GPU 类型。

点击“启动”后，AutoResearch 会保存自动运行授权。明确的 GPU 不可用、连接故障或会话
断开会进入持久重试队列，默认随机等待 5–10 分钟。账号达到 GPU 配额时等待更久：
优先采用 Colab 明确提供的冷却时间；没有时间提示时程序保守默认等待 12 小时，这不是 Google 保证的额度重置时间。
额度等待按默认登录浏览器配置共享，跨项目、重新准备和本地程序重启仍有效；重复出现同一额度提示不会不断推迟截止时间，明确更长的官方等待会优先采用。
连接请求超时后，工作器仍只读检查迟到的额度或登录提示，不会为此再次申请 GPU。
只有比额度提示更新、足够新鲜且带真实 GPU 名称与显存信息的当前运行时检查，才能清除已过时的共享等待；历史输出和空的“可用”记录不会清除它。

保持本地 AutoResearch 服务运行，调度器会在保存的时间到达后自动启动缺失的浏览器
工作器；无需手动保持工作器窗口打开。关闭或重启本地服务不会清除授权和重试时间，
下次启动服务后会继续检查队列。登录、验证码和 Drive 授权会暂停队列，等待你完成
操作并点击“继续”，不会自动重试。“停止重试”会取消队列，关闭该工作器的浏览器，
并停止后续自动启动；若 Colab 运行时仍在训练，请在 Colab 页面停止它。

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

手动运行辅助器（先在网页中准备并启动队列）：

```powershell
.\colab-browser.ps1 -ProjectId "项目ID" -Browser chrome
```

后台辅助器日志位于 `data/colab-browser/<项目ID>.log`；Google 登录状态只留在
`data/colab-browser/profile/`。不要共享该目录。
首次安装可选浏览器依赖：`.venv\Scripts\python.exe -m pip install -e ".[browser]"`。

可选的官方 [Colab Notebook MCP](https://github.com/googlecolab/colab-mcp) 辅助进程使用
独立 Python 3.13 环境 `data/colab-mcp/venv/`，不改变主程序的 Python 环境。
工作器检测到已安装的环境和 `data/colab-mcp/speaker-session/` 后，通过 notebook 的
正常连接界面使用它编写、执行单元格并读取当前输出。它不负责申请 GPU；运行时类型
选择和连接仍通过 Colab 界面完成。未安装该辅助进程时可继续使用 notebook 上传流程。
MCP 临时连接信息保存在本地私有文件中，不要共享 `data/colab-mcp/`。

免费 GPU 与运行时上限由 [Colab 官方规则](https://research.google.com/colaboratory/faq.html) 决定。
此模式不使用 SSH、付费 GPU、后台保活、内部申请 API 或多账号绕过限制。
