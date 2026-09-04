# 安全边界

AutoResearch 会处理密钥、访问网络、生成代码并可在远端执行命令，因此默认采用以下边界：

- `.env`、账号资料、数据、论文 PDF 和实验结果不进入 Git。
- 浏览器设置接口永不回显密钥；SSH 密码不写数据库。
- 模型只能在项目工作区创建文件，绝对路径、路径穿越和常见敏感文件名会被拒绝。
- PDF 自动下载只允许 HTTPS 的 arXiv 主机，单文件最多 25 MB。
- 远程上传排除 `.git`、`.env`、数据、结果和虚拟环境。
- AutoDL 创建实例必须确认计费；释放替代实例还需要单独勾选授权。
- GPU 恢复次数有上限，不会不断创建计费实例。
- Git 远端仅接受标准 GitHub HTTPS/SSH 地址，不接受 URL 中携带 Token。
- MCP 默认只监听 `127.0.0.1`；ChatGPT 通过向外建立连接的 Secure MCP Tunnel 使用，不要求开放入站端口。
- MCP 工具不会返回 OpenRouter Key、AutoDL Token 或 SSH 密码；通过 AutoDL 启动实验时，SSH 信息只在 AutoResearch 进程内转换为连接对象。
- MCP 的计费实例创建和远程命令执行分别要求 `confirm_billable` 与 `confirm_execute`。工具描述和安全注解要求 ChatGPT 只在当前对话得到明确同意后设置这些值。

这不是强隔离沙箱。开始远程实验代表授权执行界面中显示的命令；执行前应审阅 `generated/`。建议只在可丢弃的 AutoDL 实例中运行，并为重要数据保留备份。

如果密钥曾出现在聊天、截图或明文需求文档中，应在服务商控制台轮换，并把新的值只填入本机设置。
