# Experiment preparation and bounded execution

Check compute configuration before checking the experiment package. These diagnostics work without a GPU:

- MCP: `check_autodl_readiness` and `check_experiment_readiness(project_id)`.
- API: `GET /api/autodl/preflight` and `GET /api/projects/{project_id}/readiness`.
- Local diagnostic without the UI or Tunnel: `.venv/Scripts/python.exe -m autoresearch.preflight --project <project-id>`. It returns JSON and exits 2 on blockers without provisioning or executing code.

Diagnostics inspect declarations and file metadata only. Remote inputs are explicitly reported as unknown. Even `ready=true` does not prove CUDA compatibility, data semantics, or algorithm effectiveness. Restart the web service after adding backend routes.

## Manifest contract

`generated/experiment_manifest.json` uses `schema_version: 1` and declares `run_command`, `working_directory`, `required_files`, `required_inputs`, and `success_criteria`. Code paths are relative to `generated`; input paths are relative to the working directory or explicit absolute data paths. Remote inputs must be rechecked on the execution host.

`success_criteria.required_metrics` lists scientific metric names. A smoke test that validates only process completion may explicitly set `validation: "process_exit"` and `required_metrics: []`; the diagnostic then states that no scientific measurement was requested. Legacy manifests without success criteria receive an improvement warning.

## Portable runner

`autoresearch/experiment_runner.py` uses only the Python standard library and can be uploaded as `experiment_runner.py`. A runnable manifest also has a `steps` array. Each step contains a unique `name`, a `command` argument array, and a positive integer `timeout_seconds`; optional fields are `env`, `metrics_file`, and `required_metrics`. Commands run directly with `shell=False`; setup or download commands are never run implicitly. Metric files are relative to and confined within the working directory.

```sh
python experiment_runner.py --manifest experiment_manifest.json --output results
python experiment_runner.py --manifest experiment_manifest.json --output results --execute
```

The default is preflight only. Execution is sequential and stops on failure or timeout. Every step records stdout, stderr, exit code, and duration. Each run gets a unique `run-*` directory, while `results/metrics.json` is published atomically for SSH monitoring. An exclusive lock prevents concurrent use of one output directory. After an abnormal exit, inspect the PID stored in the lock before removing it. Use a fresh metric path or working copy for the next experiment so old measurements cannot be mistaken for new ones.

Declared metrics must be produced as fresh JSON. Missing metrics, nonnumeric values, NaN, and Infinity fail validation. Only a summary from executing all steps successfully is treated as completed execution. A zero-exit preflight cannot trigger experiment success or automatic improvement. Legacy exit-code handling remains compatible, but scientific validity still requires separate analysis.

Project-specific TidyVoice preparation lives in that project's `generated/README_STAGE1.md`. Install YAML support with `pip install -e ".[research]"`. PyTorch and the target GPU runtime must be validated separately and are never installed implicitly.
