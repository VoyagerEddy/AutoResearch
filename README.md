# AutoResearch

AutoResearch is a local research execution workbench. The recommended workflow is to use the current conversation and selected model in the ChatGPT desktop app for research reasoning, then call AutoResearch's local MCP tools to persist notes and code, search sources, provision AutoDL instances, run experiments, and inspect results. Project state, sources, logs, metrics, and analysis remain visible in the web dashboard.

The original fully autonomous OpenRouter workflow is still available. In collaborative mode, only `generate_experiment_code` invokes the configured OpenRouter model. `save_experiment_code` safely stores code already written by ChatGPT and does not make another model call.

> This release is a working MVP. Do not run model-generated code unattended on production systems. The UI requires explicit confirmation before provisioning billable AutoDL instances, running SSH commands, or releasing replacement instances.

## Quick start on Windows

Requirements: Python 3.11+ and Git. Chrome and VS Code are recommended.

1. Double-click `start.cmd`. The first run creates `.venv`, installs dependencies, and creates a local `.env` file.
2. Chrome opens `http://127.0.0.1:8765`.
3. The default workflow is **ChatGPT collaboration**. Create a project and use its project ID in your ChatGPT conversation.
4. For the first ChatGPT connection, select **Set up connection** on the home page or run `setup-chatgpt.cmd`. On later runs, use `start-chatgpt.cmd` to keep both the local service and Tunnel available.
5. Open **Settings and connections** and enter an OpenRouter key only if you delegate code generation to AutoResearch or use autonomous mode.
6. To use AutoDL, obtain a developer token from **Account → Settings** in the AutoDL console and enter an image UUID.

PowerShell alternative:

```powershell
./start.ps1
```

Start the service without opening a browser:

```powershell
.venv/Scripts/python.exe -m autoresearch --no-open
```

## Workflows

### ChatGPT collaboration (recommended)

1. Use the current ChatGPT model to discuss the problem, develop hypotheses, and analyze results.
2. Call `create_research_project` to create a collaborative AutoResearch project.
3. Use `save_research_note`, `search_research_sources`, `save_experiment_code`, or `generate_experiment_code` as needed.
4. After confirmation, call `create_autodl_instance` and `start_autodl_experiment`.
5. Read metrics and logs with `get_experiment_result`, analyze them in ChatGPT, and save conclusions with `record_chatgpt_analysis`.
6. Open the project deep link to see every phase, source, experiment, metric, log, and report.

AutoResearch does not require an OpenAI API key. The ChatGPT app provides the conversation and reasoning. AutoResearch uses the OpenRouter key only for explicitly delegated code generation or the autonomous workflow.

Complete connection instructions are in [`docs/CHATGPT.md`](docs/CHATGPT.md). The local MCP endpoint is:

```text
http://127.0.0.1:8765/mcp
```

ChatGPT cannot connect directly to loopback addresses. Developer mode uses Secure MCP Tunnel, or you can deploy an HTTPS MCP endpoint. The tunnel connects outbound and does not require exposing a local inbound port. The Windows setup assistant checks the local MCP service, runs `tunnel-client init` and `doctor`, encrypts the runtime key with Windows DPAPI for the current user, and opens the ChatGPT Plugins page. Creating the tunnel, enabling Developer mode, and adding the initial connection still require the signed-in account.

### Windows connection helpers

- First use: run `setup-chatgpt.cmd`, download `tunnel-client.exe` from OpenAI Platform when prompted, and enter the `tunnel_id` and runtime API key. Key input is hidden and is never written to the repository or `.env`.
- Normal use: run `start-chatgpt.cmd`. It starts AutoResearch when needed and keeps Tunnel running in the current window. Keep the window open while using ChatGPT.
- Diagnostics: run `powershell -NoProfile -ExecutionPolicy Bypass -File .\chatgpt-tunnel.ps1 -Action Doctor`.
- Status: the web UI reports MCP, client installation, Tunnel configuration, and Tunnel runtime status separately.

### Autonomous AutoResearch mode

1. **Plan:** convert the topic into an objective, hypotheses, search queries, and an evaluation plan.
2. **Search:** query several paper and code sources, deduplicate by title, and persist real URLs.
3. **Synthesize:** generate a research plan with source markers such as `[S1]`, clearly separating facts, inferences, and untested hypotheses.
4. **Generate:** write a complete experiment project under `workspaces/<project-id>/generated/`.
5. **Experiment:** after command confirmation, inspect the remote GPU, upload code, and launch it with `nohup`.
6. **Iterate:** poll state, logs, and `results/metrics.json`, then make bounded improvements within the configured iteration limit.
7. **Sync:** sync code only. Secrets, papers, data, and experiment results are excluded by default.

If OpenRouter is not configured or temporarily unavailable, AutoResearch creates a deterministic offline baseline so the workflow can be inspected. This does not constitute a completed domain algorithm.

## AutoDL

AutoResearch uses AutoDL's official [Container Instance Pro API](https://www.autodl.com/docs/instance_pro_api/). It does not store AutoDL account passwords or depend on browser login automation. The default GPU preference is:

1. `v-48g`, the documented 4090-48G general-purpose specification.
2. `5090-p`, the documented 5090-32G performance specification.

If a create request times out, returns an unexpected response, or lacks a valid instance UUID, AutoResearch stops without retrying and asks you to inspect the instance list. This prevents duplicate billable resources. It only falls back to another GPU specification for provider errors that contractually guarantee no instance was created. The public API currently documents no such error code, so the fallback allowlist is empty.

Before provisioning, call `check_autodl_readiness` or `GET /api/autodl/preflight`. Local checks verify the token, image, and GPU configuration. With `verify_api=true`, AutoResearch performs a free read-only instance-list request and then reads the wallet balance without returning instance records or secrets. `configured` means required local settings exist, `verified` means the instance API authenticated successfully, and `balance_verified` separately means the wallet response was valid. Cash and voucher balances remain separate exact decimal strings based on the official milli-CNY fields. AutoResearch does not infer voucher eligibility or a spendable experiment budget. Inventory and image availability remain unverified.

If an SSH check finds an active GPU workload, recovery saves the current image and creates one clone. If the clone is also busy, AutoResearch may release only that replacement and provision from the base image when explicitly authorized. This is bounded to one clone and one rebuild.

## Configuration

Copy `.env.example` to `.env`, or use the web settings dialog:

```dotenv
OPENROUTER_API_KEY=
OPENROUTER_MODEL=openrouter/free
AUTODL_TOKEN=
AUTODL_IMAGE_UUID=
AUTODL_GPU_SPECS=v-48g,5090-p
GITHUB_REMOTE_URL=https://github.com/VoyagerEddy/AutoResearch.git
```

Never commit real secrets. The settings API returns only whether a value is configured and never echoes keys or tokens.

## Experiment package contract

See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for input validation, manifest fields, and paired execution. `check_experiment_readiness` and the project `/readiness` API report blockers in code entry points, data, and metric declarations. Without the UI or Tunnel, run `.venv/Scripts/python.exe -m autoresearch.preflight --project <project-id>` for a local diagnostic that does not provision or execute anything.

`autoresearch/experiment_runner.py` uses only the Python standard library and can be copied to a remote host. It preflights by default and executes steps only with `--execute`. Each step has a timeout, separate logs, and strict metric validation. Historical and current results are stored separately. A successful preflight is never treated as a successful experiment.

Generated `experiment_manifest.json` files describe setup, data downloads, the run command, and metric locations. Remote monitoring reads:

```text
results/metrics.json
```

Experiment entry points must return nonzero on failure. Setup and download commands are recorded for review; the UI executes only the confirmed experiment command. Through ChatGPT tools, AutoResearch retrieves SSH credentials internally and never returns the SSH password, AutoDL token, or OpenRouter key. Billable provisioning requires `confirm_billable=true`, and upload plus execution requires `confirm_execute=true` after explicit consent in the current conversation.

## Development and testing

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"
.venv/Scripts/python.exe -m pytest
node --test tests/frontend_preflight.test.cjs
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and [`docs/SECURITY.md`](docs/SECURITY.md).

## Known limitations

- AutoDL Pro API requires identity verification, a developer token, and an available image. Not every GPU shown in the regular marketplace has a Pro API specification.
- SSH passwords live only in process memory. Monitoring an existing remote job is not automatically restored after a service restart.
- The GitHub repository must already exist, and Git must be authenticated through a credential manager or SSH.
- Generated code still requires review before execution. Sandboxing and confirmation gates cannot eliminate every risk.
