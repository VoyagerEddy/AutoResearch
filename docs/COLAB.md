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

Notebook 的第一格运行轻量 GPU 探针，探针通过后才挂载 Drive。控制台按尝试编号显示
GPU 不可用、额度限制、Notebook 启动超时和 Drive 授权。旧尝试的迟到回执保留编号
与时间，当前状态沿用最新尝试。已导出的旧版 Notebook 可以重新准备以启用第一格探针和新版回执。

作业启动后，AutoResearch 会保存自动运行授权。明确的 GPU 不可用、连接故障或会话
断开会进入持久重试队列，默认随机等待 5–10 分钟。连续第三次及以后的 GPU 不可用
会把间隔延长到 15–30 分钟。账号达到 GPU 配额时按 Colab 给出的时间等待。
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
实际 GPU 检查按分级随机间隔调度。账号额度冷却优先。
Agent 不依赖浏览器窗口持续打开，也不会因页面刷新额外申请 GPU。

面板显示当前账号“有 / 暂不可用 / 待确认”、Agent 状态和最近心跳、最近一次真实检查
结果及时间、下次实际检查时间。Agent 心跳只表示监测程序活着，不能当作新的 GPU
检查；历史结果、连接异常或额度冷却期间会显示“待确认”，并保留最近一次结果。
成功分配 GPU 后，既有流程继续启动试验及 checkpoint 保存；停止重试或任务结束后
Agent 自动退出。它观察当前账号的连接结果。

### 数据、断点和备用 GPU

等待期间分批验证已登记的数据清单和本地分片 SHA-256。取得 GPU 后，已声明音频环境
的项目先运行 CUDA 运算和 TorchAudio FBank 预检。随后 Notebook 从已挂载 Drive 的
项目目录 `inputs/verified-shards/` 按 SHA-256 恢复分片，浏览器补传缺失或校验失败的
分片。每个新分片以原子替换写入缓存并回读校验。Drive 写入受限时，已验证的缓存仍可
复用，当前运行时可使用浏览器已传输的分片。检查点正常落盘后记录文件大小和 SHA-256。

等候 GPU 一小时后，调度器可以检查已由用户在 ChatGPT 中批准的本机 GPU 或正在运行的
AutoDL 实例。ChatGPT 用 `authorize_colab_backup` 记录目标、可访问的共享项目路径、
可选的已准备数据路径和 Python 解释器。批准绑定当前准备的源码和数据清单。备用目标
使用同样的 GPU 预检和训练命令，可按清单从共享分片缓存恢复本机暂存目录。已有训练要求
共享路径包含最新 Colab 回执对应的断点
文件。备用执行与 Colab 申请共用这项任务的调度状态。AutoDL 只检查和使用获准的现有
实例。需要新增 Google 授权或备用算力授权时，ChatGPT 通过 `claim_colab_notice`
领取一次提示。控制台展示状态与回执。

也可以单独启动当前项目的监测进程（API 地址使用正在运行的 AutoResearch 服务）：

```powershell
.venv\Scripts\python.exe -m autoresearch.colab_resource_monitor_cli --launch --project "项目ID" --api-url "http://127.0.0.1:8767"
```

Notebook 通过正常 Drive 挂载保存 `inputs/`、`checkpoints/`、`results/` 和 `session_status.json`。
项目需要声明完整 checkpoint 与恢复命令；恢复训练会读取已有 checkpoint。
Colab 断开时仍可能丢失最后一次 checkpoint 之后的训练进度。
工作器通过当前 notebook 输出的运行 ID 和退出码判断结果，不把历史输出当成当前成功。
全量上传前，notebook 会检查 Drive 写入和临时盘容量，并执行项目声明的预检命令。
PyTorch 项目可以在 `runtime_requirements` 声明精确的 `torch_version` 和 `torchaudio_version`，并在
`setup_commands` 安装对应的 wheel。预检在分配到的 GPU 上执行同步的 CUDA 运算，要求 TorchAudio
时还会用合成音频运行重采样和 Kaldi FBank。版本、预检错误及当前阶段可在状态中查看。

以下片段是 CUDA 12.6 的配套环境配置，可并入 `experiment_manifest.json` 的 `colab` 对象。

```json
{
  "setup_commands": [
    "python -m pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.9.1+cu126 torchvision==0.24.1+cu126 torchaudio==2.9.1+cu126"
  ],
  "runtime_requirements": {
    "framework": "pytorch",
    "require_cuda": true,
    "match_torchaudio": true,
    "torch_version": "2.9.1+cu126",
    "torchaudio_version": "2.9.1+cu126"
  }
}
```

使用已绑定的 Drive notebook，并启用官方 Colab Notebook MCP 辅助进程时，工作器收到已安全
保存的终态后，会通过 Colab 界面释放它自己启动并核验过的运行时。已有项目需要重新导出
Notebook 才能使用新版预检和释放标记。
结果先写入作业状态，释放回执独立保存。状态 `released` 表示界面已确认断开，`unknown` 表示需要
核对 Colab 会话。重新接管的运行时、手工启动的运行时和未确认持久保存的失败会话由原会话管理。
状态诊断会区分活跃会话占满和 GPU 用量限制，并保留 Colab 的重试等待。

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

## RSI 私有 Hugging Face Dataset

1. 在 Hugging Face 新建一个可由数据授权范围内成员访问的 **Private Dataset**。上传端需要对该仓库有写权限的 token。已有仓库也可使用，上传工具会在提交前核对其私有状态。[Hugging Face token 权限](https://huggingface.co/docs/hub/en/security-tokens)
2. 从仓库根目录执行下列 PowerShell 命令。把数据根目录、仓库名和项目 ID 换成你的值。上传器会在终端遮盖 token 输入。中断后重新执行同一命令，会逐批跳过远端大小和 SHA-256 一致的文件。[Hugging Face 上传文档](https://huggingface.co/docs/huggingface_hub/en/guides/upload)

   ```powershell
   .venv/Scripts/python.exe -m pip install -e ".[hf]"
   .venv/Scripts/python.exe -m autoresearch.hf_dataset_upload --data-root "D:\datasets\autoresearch-full-data" --repo-id "your-namespace/tidyvoicex-full-data-private" --output "workspaces/<project-id>/generated/inputs/hf_dataset_pointer.json"
   ```

3. 在项目的 `generated/experiment_manifest.json` 中配置以下字段。Prepare 会读取指针中的 `repo_id`、`revision`（完整 commit SHA）、`manifest_path` 和 `manifest_sha256`。指针就位后重新 Prepare 并导入新版 notebook。

   ```json
   {
     "colab": {
       "hf_dataset_transfer": {
         "pointer": "inputs/hf_dataset_pointer.json",
         "online_augmentation": true,
         "window_shards": 4,
         "window_bytes": 1610612736
       }
     }
   }
   ```

4. 在 HF 创建一个只可读取该私有 Dataset 的 fine-grained token。在 Colab 打开左侧钥匙图标 **Secrets**，新增名称为 `HF_TOKEN` 的条目，粘贴只读 token 并打开 **Notebook access**，然后运行全部单元格。使用 AutoResearch 续跑时，在授权后点击 Resume。[Colab Secrets 设置示例](https://huggingface.co/agents-course/notebooks/commit/9fb4a1001b3f3cb67d70bc2dd03cd16cffa84ba4)

预检通过后，Colab 在 `/content/autoresearch-full-data` 校验并准备元数据和 trials。在线增强启用时逐块重组 MUSAN 和 RIRS。关闭 `online_augmentation` 会改变训练配置，应对 B0 和 VQ 使用同一设置。训练读取有界的临时分片，评测逐归档提取 embedding。Drive 保存源代码、预训练输入、日志、checkpoint、embedding 及结果。RSI 要求 HF 指针。若需恢复兼容模式，可移除 `colab.hf_dataset_transfer` 并把 `required_inputs` 的本地指针改回 `inputs/full-data-pointer.json`。通过 `AUTORESEARCH_COLAB_DATA_ROOT` 可配置浏览器上传模式的本地绝对数据根目录。

RSI 的 `colab.rsi_screening_gate` 配置跨语言 EER 最小降幅、minDCF 最大增幅、codebook 活跃比例和困惑度下限。候选还须降低整体 EER。若没有候选通过，`results/rsi/metrics.json` 记录每个候选未通过的原因，确认阶段停止。

梯度累积通过在 `prepare_command`、`run_command` 和 `checkpoint.resume_command` 同时附加 `--grad-accum-steps 32` 启用。微批大小为 8 时，每次完整更新处理 256 条训练样本，BatchNorm 仍以 8 条计算。`--screen-steps` 以优化器更新为单位。HF 训练跨 epoch 会重新下载分片，中途恢复可能重新读取同一 epoch 的早期分片以重放确定性采样顺序。
