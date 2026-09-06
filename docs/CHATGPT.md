# Using AutoResearch from ChatGPT

## Responsibilities

- ChatGPT desktop: research discussion, problem decomposition, hypothesis evaluation, design choices, and result analysis.
- AutoResearch: project persistence, paper and code search, artifact writes, delegated OpenRouter code generation, AutoDL instances and remote experiments, and status/result display.
- OpenRouter: used only by `generate_experiment_code` and autonomous web workflows.

This workflow does not convert the ChatGPT conversation into OpenAI API requests, and AutoResearch needs no OpenAI API key.

## One-time connection

1. Run `start.cmd` from the repository root and confirm that the web page reports the local service and ChatGPT tools as ready.
2. Select **Set up connection** on the home page, or run `setup-chatgpt.cmd`. The assistant checks the local service and opens [OpenAI Platform tunnel settings](https://platform.openai.com/settings/organization/tunnels).
3. In the signed-in account, create a Secure MCP Tunnel, obtain the `tunnel_id` and runtime API key, and download `tunnel-client.exe` for Windows. These are account-level operations and cannot be performed by the local app.
4. Return to the assistant and enter the downloaded file path, `tunnel_id`, and runtime API key. It configures the HTTP MCP endpoint and runs `doctor`:

   ```text
   --mcp-server-url http://127.0.0.1:8765/mcp
   ```

5. The runtime API key is encrypted with Windows DPAPI for the current Windows user. Plaintext is never written to this project, `.env`, or Git.
6. In ChatGPT, open **Settings → Security and login** and enable Developer mode. Availability depends on account and workspace policy.
7. Open [ChatGPT Plugins](https://chatgpt.com/plugins), select the plus button, choose Tunnel, select or enter the `tunnel_id`, and create the AutoResearch connection.
8. Start a new ChatGPT conversation and enable AutoResearch from the tools menu.
9. On later runs, start `start-chatgpt.cmd`. It checks and starts AutoResearch, then keeps Tunnel running. Leave that window open while using ChatGPT.

The same setup and connection actions are available from the AutoResearch home page and settings dialog. The page refreshes Tunnel status every five seconds.

The local endpoint remains `http://127.0.0.1:8765/mcp`. It works with MCP Inspector for local checks but cannot be entered directly as a public ChatGPT connector URL.

Official references: [Secure MCP Tunnel](https://developers.openai.com/api/docs/guides/secure-mcp-tunnels) and [Connect and test a plugin](https://developers.openai.com/plugins/deploy/connect-chatgpt).

## Suggested conversation flow

Initial request:

> We are studying uncertainty sampling for few-shot medical image segmentation. Handle the scientific reasoning, create an AutoResearch collaboration project, and save the key research plan. Delegate experiment code to AutoResearch's OpenRouter model when needed, but do not provision an AutoDL instance yet.

After code generation:

> Read the project state and experiment manifest, then explain the code and command to be executed. After I confirm them, provision an AutoDL instance and start the experiment.

After completion:

> Read the experiment metrics and logs. Analyze failure causes, supported conclusions, and controls for the next iteration, then save the analysis to AutoResearch.

## Tool reference

| Tool | Purpose | External side effects |
|---|---|---|
| `list_research_projects` | List projects and dashboard links | None |
| `create_research_project` | Create a ChatGPT collaboration project | Local write |
| `get_research_status` | Read state, events, sources, artifacts, and experiments | None |
| `read_project_artifact` | Read a nonsensitive text artifact | None |
| `save_research_note` | Save a ChatGPT research plan or conclusion | Local write |
| `search_research_sources` | Search paper databases and GitHub, then persist sources | Network requests and local writes |
| `save_experiment_code` | Store complete code files written by ChatGPT | Local write |
| `generate_experiment_code` | Use the AutoResearch OpenRouter model to generate and store code | Model API request and local writes |
| `check_autodl_readiness` | Check token, image, and GPU settings; optionally authenticate read-only | Local read by default; network read when `verify_api` is set; never provisions |
| `check_experiment_readiness` | Check entry points, data paths, and metric declarations | Local read only; never executes |
| `create_autodl_instance` | Provision a billable AutoDL Pro instance | May bill immediately; requires confirmation |
| `get_autodl_instance_status` | Check instance and SSH readiness without returning credentials | Read-only AutoDL request |
| `start_autodl_experiment` | Retrieve SSH internally, upload code, and execute a command | Remote execution; requires confirmation |
| `get_experiment_result` | Read status, exit code, metrics, and log tail | None |
| `record_chatgpt_analysis` | Save ChatGPT result analysis and recommendations | Local write |

## Safety notes

- Code paths passed to `save_experiment_code` are relative to the project's `generated/` directory, such as `tidyvoice/wespeaker/models/example.py`; do not prefix them with `generated/`. The complete bundle is validated before any write, so invalid paths, conflicts, or oversized content cannot cause partial replacement.
- A saved code package has not necessarily been executed. After `check_autodl_readiness`, still verify data, weights, dependencies, and the experiment entry point.
- Connection status checks process existence and Tunnel health/readiness separately. A running process is not reported as connected until readiness passes. Windows process probes do not send termination signals.
- If provisioning times out or returns an invalid response, inspect the AutoDL instance list first. AutoResearch does not automatically retry with another GPU when creation status is uncertain.
- Do not paste OpenRouter keys, AutoDL tokens, or SSH passwords into ChatGPT conversations.
- ChatGPT tools cannot retrieve these secrets; AutoResearch reads them from local `.env` configuration.
- AutoDL provisioning requires explicit cost approval in the current conversation. Remote execution requires review of the code and command followed by explicit approval.
- Keep the Secure MCP Tunnel runtime key only in the secure `tunnel-client` environment, never in this repository or a chat.
- The setup assistant passes the decrypted key to `tunnel-client` only through process environment. The stored copy is protected by Windows DPAPI for the current user.
- AutoResearch listens on `127.0.0.1` by default. Do not bind to `0.0.0.0` without authentication and firewall controls.
