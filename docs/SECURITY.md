# Security boundaries

AutoResearch handles secrets, accesses networks, generates code, and can execute remote commands. It applies these boundaries by default:

- `.env`, account data, datasets, paper PDFs, and experiment results are excluded from Git.
- The browser settings API never echoes secrets, and SSH passwords are never stored in the database.
- Generated files are confined to the project workspace. Absolute paths, traversal, common sensitive filenames, conflicts, and oversized output are rejected before any file is written.
- Automatic PDF downloads allow only HTTPS arXiv hosts and at most 25 MB per file.
- Remote uploads exclude `.git`, `.env`, datasets, results, and virtual environments.
- AutoDL provisioning requires billable-action confirmation. Releasing a replacement instance requires separate authorization.
- GPU recovery attempts are bounded and cannot continuously provision billable instances.
- Git remotes accept only standard GitHub HTTPS or SSH URLs and reject embedded tokens.
- MCP listens on `127.0.0.1` by default. ChatGPT connects through outbound Secure MCP Tunnel without opening an inbound port.
- MCP tools never return OpenRouter keys, AutoDL tokens, or SSH passwords. AutoDL SSH data is converted directly into an in-process connection object.
- Billable provisioning and remote execution require `confirm_billable` and `confirm_execute`, respectively. Tool descriptions require explicit consent in the current conversation before those fields are set.

This is not a strongly isolated sandbox. Starting a remote experiment authorizes the command displayed in the UI. Review `generated/` before execution, use disposable AutoDL instances, and back up important data.

Rotate any secret that has appeared in a chat, screenshot, or plaintext requirements document. Store the replacement only in local settings.
