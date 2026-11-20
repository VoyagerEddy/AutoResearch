from __future__ import annotations

import ast
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import uuid

import pytest

from autoresearch.colab_notebook import build_notebook


def notebook_cells(config):
    notebook = build_notebook('preflight-test', config, drive_root='AutoResearch',
                              bundle_sha256='a' * 64, bundle_bytes=b'not-executed')
    return [''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code']


def gate_source(config):
    return next(cell for cell in notebook_cells(config) if 'AUTORESEARCH_PREFLIGHT_GATE' in cell)


def context(tmp_path, config, monkeypatch):
    drive = tmp_path / 'drive'
    source = tmp_path / 'source'
    scratch = tmp_path / 'scratch'
    for path in (drive / 'results', drive / 'checkpoints', source, scratch):
        path.mkdir(parents=True, exist_ok=True)
    events = []
    namespace = dict(SETTINGS={'config': config}, config=config, DRIVE_PROJECT=drive,
        SOURCE=source, RUNTIME_ID='runtime-preflight', PROJECT_ID='preflight-test',
        datetime=datetime, json=json, os=os, shutil=shutil, hashlib=hashlib,
        subprocess=subprocess, uuid=uuid,
        pathlib=SimpleNamespace(Path=lambda value: scratch if str(value) == '/content/autoresearch-full-data' else Path(value)),
        estimate_ephemeral_disk_budget=lambda path, transfer: {'enough': True, 'required_bytes': 1, 'free_bytes': 100},
        file_sha256=lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        relative_data_path=lambda path: path,
        reconstruct_artifacts=lambda *args: None,
        safe_extract_dev_archive=lambda *args, **kwargs: None)
    def report(status, **details):
        namespace['_last_report_status'] = status
        events.append({'status': status, **json.loads(json.dumps(details))})
    namespace['report'] = report
    monkeypatch.delenv('AUTORESEARCH_SESSION_DEADLINE_UTC', raising=False)
    monkeypatch.setenv('AUTORESEARCH_CHECKPOINT_DIR', str(drive / 'checkpoints'))
    monkeypatch.setenv('AUTORESEARCH_RESULTS_DIR', str(drive / 'results'))
    return namespace, events


def real_child(tmp_path, monkeypatch, content):
    """Execute a real local Python child, adapting only the Colab Bash transport."""
    script = tmp_path / 'preflight_child.py'
    script.write_text(content, encoding='utf-8')
    original = subprocess.Popen
    calls = []
    def launch(command, **kwargs):
        assert command == 'python preflight_child.py'
        assert kwargs.pop('shell') is True
        assert kwargs.pop('executable') == '/bin/bash'
        calls.append({'command': command, 'environment': kwargs['env']})
        return original([sys.executable, str(script)], shell=False, **kwargs)
    monkeypatch.setattr(subprocess, 'Popen', launch)
    return calls


def test_generated_preflight_precedes_every_bulk_upload_and_all_cells_parse():
    config = {'preflight_command': 'python preflight_child.py'}
    cells = notebook_cells(config)
    gate = next(index for index, source in enumerate(cells) if 'AUTORESEARCH_PREFLIGHT_GATE' in source)
    bulk = next(index for index, source in enumerate(cells) if 'AUTORESEARCH_COLAB_UPLOAD_DATA ' in source)
    setup = next(index for index, source in enumerate(cells) if 'Installing project dependencies:' in source)
    assert setup < gate < bulk
    for source in cells:
        ast.parse(source)


def test_legacy_project_needs_no_framework_or_preflight_command(tmp_path, monkeypatch):
    namespace, events = context(tmp_path, {}, monkeypatch)
    import importlib
    monkeypatch.setattr(importlib, 'import_module', lambda name: pytest.fail('Undeclared framework import: ' + name))
    exec(gate_source({}), namespace)
    evidence = namespace['RUNTIME_PREFLIGHT']
    assert evidence['status'] == 'passed'
    assert evidence['framework']['status'] == 'not_requested'
    assert evidence['project_command']['status'] == 'not_configured'
    assert evidence['drive']['write_verified'] is True
    assert evidence['drive']['cloud_quota_status'] == 'unknown'
    assert evidence['drive']['filesystem_available_bytes'] >= 0
    assert evidence['scientific_metrics_produced'] is False
    assert not list(namespace['DRIVE_PROJECT'].glob('.preflight_*.tmp'))
    assert all(event['status'] == 'starting' for event in events)


def test_project_preflight_really_runs_and_isolates_checkpoint_and_metrics(tmp_path, monkeypatch):
    config = {'preflight_command': 'python preflight_child.py', 'preflight_timeout_seconds': 20}
    namespace, events = context(tmp_path, config, monkeypatch)
    calls = real_child(tmp_path, monkeypatch, '''import json, os
from pathlib import Path
checkpoints = Path(os.environ['AUTORESEARCH_CHECKPOINT_DIR'])
results = Path(os.environ['AUTORESEARCH_RESULTS_DIR'])
checkpoints.mkdir(parents=True)
results.mkdir(parents=True)
(checkpoints / 'resume_latest.pt').write_bytes(b'preflight-only')
(results / 'metrics.json').write_text('{"campaign_complete": true}')
print(json.dumps({'mode': os.environ['AUTORESEARCH_PREFLIGHT'], 'checkpoint_dir': str(checkpoints), 'results_dir': str(results)}))
''')
    exec(gate_source(config), namespace)
    assert len(calls) == 1
    evidence = namespace['RUNTIME_PREFLIGHT']
    output = json.loads(evidence['project_command']['output_tail'])
    assert output['mode'] == '1'
    assert Path(output['checkpoint_dir']).is_relative_to(namespace['SOURCE'])
    assert Path(output['results_dir']).is_relative_to(namespace['SOURCE'])
    assert not (namespace['DRIVE_PROJECT'] / 'checkpoints/resume_latest.pt').exists()
    assert not (namespace['DRIVE_PROJECT'] / 'results/metrics.json').exists()
    assert evidence['status'] == 'passed' and evidence['project_command']['returncode'] == 0
    assert all(event['status'] != 'completed' for event in events)


def test_drive_space_can_be_unknown_but_write_is_still_verified(tmp_path, monkeypatch):
    namespace, events = context(tmp_path, {}, monkeypatch)
    original = shutil.disk_usage
    def usage(path):
        if Path(path) == namespace['DRIVE_PROJECT']:
            raise OSError('Drive mount has no quota statistics')
        return original(path)
    monkeypatch.setattr(shutil, 'disk_usage', usage)
    exec(gate_source({}), namespace)
    evidence = events[-1]['preflight']['drive']
    assert evidence['write_verified'] is True
    assert evidence['filesystem_available_bytes'] is None
    assert evidence['filesystem_space_status'] == 'unknown'
    assert evidence['cloud_quota_status'] == 'unknown'


def test_unwritable_drive_blocks_preflight_child(tmp_path, monkeypatch):
    config = {'preflight_command': 'must not run'}
    namespace, events = context(tmp_path, config, monkeypatch)
    import builtins
    original = builtins.open
    def open_file(path, mode='r', *args, **kwargs):
        if mode == 'xb' and Path(path).name.startswith('.preflight_'):
            raise PermissionError('Drive is not writable')
        return original(path, mode, *args, **kwargs)
    monkeypatch.setattr(builtins, 'open', open_file)
    monkeypatch.setattr(subprocess, 'Popen', lambda *args, **kwargs: pytest.fail('Unwritable Drive executed a child'))
    with pytest.raises(PermissionError, match='Drive is not writable'):
        exec(gate_source(config), namespace)
    assert events[-1]['status'] == 'failed'


def test_nonzero_preflight_blocks_bulk_requests_and_preserves_failure_output(tmp_path, monkeypatch, capsys):
    config = {'preflight_command': 'python preflight_child.py'}
    namespace, events = context(tmp_path, config, monkeypatch)
    real_child(tmp_path, monkeypatch, "import sys\nprint('real model cannot fit')\nsys.exit(7)\n")
    reached_bulk = False
    with pytest.raises(subprocess.CalledProcessError) as error:
        exec(gate_source(config), namespace)
        reached_bulk = True
    assert not reached_bulk and error.value.returncode == 7
    failed = events[-1]
    assert failed['status'] == 'failed' and failed['preflight']['status'] == 'failed'
    assert 'real model cannot fit' in failed['preflight']['project_command']['output_tail']
    assert 'AUTORESEARCH_COLAB_UPLOAD_DATA' not in capsys.readouterr().out
    assert not any(event['status'] == 'completed' for event in events)


def test_silent_preflight_times_out_and_is_terminated(tmp_path, monkeypatch):
    config = {'preflight_command': 'python preflight_child.py', 'preflight_timeout_seconds': 1}
    namespace, events = context(tmp_path, config, monkeypatch)
    real_child(tmp_path, monkeypatch, 'import time\ntime.sleep(30)\n')
    with pytest.raises(subprocess.TimeoutExpired):
        exec(gate_source(config), namespace)
    assert events[-1]['preflight']['project_command']['timed_out'] is True
    assert events[-1]['status'] == 'failed'


def test_expired_session_never_starts_project_preflight(tmp_path, monkeypatch):
    config = {'preflight_command': 'python preflight_child.py'}
    namespace, events = context(tmp_path, config, monkeypatch)
    monkeypatch.setenv('AUTORESEARCH_SESSION_DEADLINE_UTC', datetime.datetime.now(datetime.timezone.utc).isoformat())
    monkeypatch.setattr(subprocess, 'Popen', lambda *args, **kwargs: pytest.fail('Expired session executed a child'))
    with pytest.raises(RuntimeError, match='Session budget'):
        exec(gate_source(config), namespace)
    assert events[-1]['status'] == 'failed'


@pytest.mark.parametrize('failure', ['cuda', 'version'])
def test_explicit_pytorch_contract_fails_before_project_execution(tmp_path, monkeypatch, failure):
    config = {'preflight_command': 'must not run', 'runtime_requirements':
        {'framework': 'pytorch', 'require_cuda': True, 'match_torchaudio': True}}
    namespace, events = context(tmp_path, config, monkeypatch)
    fake_torch = SimpleNamespace(__version__='2.6.0+cpu', cuda=SimpleNamespace(is_available=lambda: False))
    fake_audio = SimpleNamespace(__version__='2.5.0' if failure == 'version' else '2.6.0')
    import importlib
    monkeypatch.setattr(importlib, 'import_module', lambda name: fake_torch if name == 'torch' else fake_audio)
    monkeypatch.setattr(subprocess, 'Popen', lambda *args, **kwargs: pytest.fail('Incompatible framework executed a child'))
    with pytest.raises(RuntimeError, match='versions must match' if failure == 'version' else 'No actual CUDA GPU'):
        exec(gate_source(config), namespace)
    assert events[-1]['status'] == 'failed'


def test_explicit_pytorch_contract_records_verified_device_and_versions(tmp_path, monkeypatch):
    config = {'runtime_requirements': {'framework': 'pytorch', 'require_cuda': True, 'match_torchaudio': True}}
    namespace, events = context(tmp_path, config, monkeypatch)
    fake_torch = SimpleNamespace(__version__='2.6.0+cu124', cuda=SimpleNamespace(
        is_available=lambda: True, get_device_name=lambda index: 'fixture GPU',
        get_device_properties=lambda index: SimpleNamespace(total_memory=12345)))
    fake_audio = SimpleNamespace(__version__='2.6.0+cu124')
    import importlib
    monkeypatch.setattr(importlib, 'import_module', lambda name: fake_torch if name == 'torch' else fake_audio)
    exec(gate_source(config), namespace)
    evidence = events[-1]['preflight']['framework']
    assert evidence['status'] == 'passed' and evidence['cuda_available'] is True
    assert evidence['cuda_total_memory_bytes'] == 12345
    assert evidence['torch_version'] == evidence['torchaudio_version'] == '2.6.0+cu124'


def test_insufficient_scratch_space_blocks_before_project_preflight(tmp_path, monkeypatch):
    config = {'full_data_transfer': {'declared': 'large corpus'}, 'preflight_command': 'must not run'}
    namespace, events = context(tmp_path, config, monkeypatch)
    namespace['estimate_ephemeral_disk_budget'] = lambda *args: {'enough': False, 'free_bytes': 1, 'required_bytes': 100}
    monkeypatch.setattr(subprocess, 'Popen', lambda *args, **kwargs: pytest.fail('Insufficient disk executed a child'))
    with pytest.raises(RuntimeError, match='Insufficient Colab ephemeral disk'):
        exec(gate_source(config), namespace)
    assert events[-1]['preflight']['scratch']['corpus_budget']['enough'] is False


@pytest.mark.parametrize('old_evidence', [
    {'status': 'failed', 'runtime_id': 'runtime-preflight'},
    {'status': 'passed', 'runtime_id': 'previous-runtime'},
])
def test_bulk_cell_cannot_bypass_failed_or_stale_preflight(tmp_path, monkeypatch, old_evidence):
    transfer = {'metadata_directory': 'full-data', 'manifest_path': 'data_manifest.json',
                'manifest_sha256': hashlib.sha256(b'{}').hexdigest()}
    config = {'full_data_transfer': transfer}
    namespace, events = context(tmp_path, config, monkeypatch)
    metadata = namespace['DRIVE_PROJECT'] / 'inputs/full-data'
    metadata.mkdir(parents=True)
    (metadata / 'data_manifest.json').write_bytes(b'{}')
    namespace['RUNTIME_PREFLIGHT'] = old_evidence
    namespace['files'] = SimpleNamespace(upload=lambda: pytest.fail('Failed preflight uploaded data'))
    bulk = next(source for source in notebook_cells(config) if 'AUTORESEARCH_COLAB_UPLOAD_DATA ' in source)
    with pytest.raises(RuntimeError, match='current-runtime preflight'):
        exec(bulk, namespace)
    assert events[-1]['status'] == 'failed'


def test_experiment_cell_cannot_treat_preflight_artifacts_as_completion(tmp_path, monkeypatch):
    config = {'preflight_command': 'already failed', 'run_command': 'must not run'}
    namespace, events = context(tmp_path, config, monkeypatch)
    namespace['RUNTIME_PREFLIGHT'] = {'status': 'failed', 'runtime_id': namespace['RUNTIME_ID']}
    (namespace['DRIVE_PROJECT'] / 'results/metrics.json').write_text('{"campaign_complete": true}')
    monkeypatch.setattr(subprocess, 'Popen', lambda *args, **kwargs: pytest.fail('Failed preflight executed experiment'))
    with pytest.raises(RuntimeError, match='current-runtime preflight'):
        exec(notebook_cells(config)[-1], namespace)
    assert events[-1]['status'] == 'failed'


def test_passed_preflight_is_required_before_real_bulk_cell_executes(tmp_path, monkeypatch):
    payload = b'small corpus shard'
    manifest = b'{}'
    transfer = {'metadata_directory': 'full-data', 'manifest_path': 'data_manifest.json',
        'manifest_sha256': hashlib.sha256(manifest).hexdigest(),
        'files': [{'kind': 'train_shard', 'path': 'train/shards/000000.tar', 'size_bytes': len(payload),
                   'sha256': hashlib.sha256(payload).hexdigest()}],
        'augmentation': {'musan': {'path': 'augmentation/musan/lmdb/data.mdb', 'parts': [], 'size_bytes': 1, 'sha256': 'a' * 64}},
        'assemblies': []}
    config = {'full_data_transfer': transfer}
    namespace, events = context(tmp_path, config, monkeypatch)
    metadata = namespace['DRIVE_PROJECT'] / 'inputs/full-data'
    metadata.mkdir(parents=True)
    (metadata / 'data_manifest.json').write_bytes(manifest)
    def upload():
        assert namespace['RUNTIME_PREFLIGHT']['status'] == 'passed'
        assert events[-1]['phase'] == 'preflight'
        return {'000000.tar': payload}
    namespace['files'] = SimpleNamespace(upload=upload)
    cells = notebook_cells(config)
    bulk = next(source for source in cells if 'AUTORESEARCH_COLAB_UPLOAD_DATA ' in source)
    before = Path.cwd()
    try:
        exec(gate_source(config), namespace)
        exec(bulk, namespace)
    finally:
        os.chdir(before)
    assert (tmp_path / 'scratch/train/shards/000000.tar').read_bytes() == payload
    assert not any(event['status'] in {'running', 'completed'} for event in events)
