import json
import subprocess
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from autoresearch.colab_notebook import build_notebook
from autoresearch.config import Settings
from autoresearch.db import Database
from autoresearch.domain import ColabFallbackTarget, ColabObservationRequest, ColabPrepareRequest, ResearchRequest
from autoresearch.services import colab, colab_data, colab_fallback, colab_fallback_worker
from autoresearch.services.colab_browser import provider_retry_deadline


@pytest.fixture
def queued(tmp_path, monkeypatch):
    now = [datetime(2026, 11, 24, tzinfo=timezone.utc)]
    monkeypatch.setattr(colab, '_utc', lambda: now[0])
    settings = Settings.load(tmp_path)
    db = Database(settings.data_dir / 'repo.sqlite3')
    project = db.create_project(ResearchRequest(topic='GPU retries'), settings.workspace_dir)
    generated = Path(project.workspace) / 'generated'
    generated.mkdir(parents=True)
    (generated / 'train.py').write_text('pass\n', encoding='utf-8')
    (generated / 'experiment_manifest.json').write_text(json.dumps({'colab': {
        'run_command': 'python train.py', 'checkpoint': {
            'directory': 'checkpoints', 'files': ['state.pt'], 'interval_seconds': 60,
            'resume_command': 'python train.py --resume', 'resume_supported': True,
            'full_training_state': True
        }
    }}), encoding='utf-8')
    service = colab.ColabService(db, settings)
    assert service.prepare(project.id, ColabPrepareRequest())['prepared']
    return service, project.id, now


def test_first_cell_reports_gpu_denial_before_drive(monkeypatch, capsys):
    notebook = build_notebook('example', {}, drive_root='AutoResearch', bundle_sha256='a' * 64,
                              bundle_bytes=b'')
    assert notebook['cells'][0]['cell_type'] == 'code'
    first = ''.join(notebook['cells'][0]['source'])
    assert 'drive.mount' not in first
    monkeypatch.setattr(subprocess, 'run', lambda *args, **kwargs: subprocess.CompletedProcess(args, 1, '', ''))
    scope = {}
    with pytest.raises(RuntimeError, match='GPU 不可用'):
        exec(first, scope)
    rows = [json.loads(line.split(' ', 1)[1]) for line in capsys.readouterr().out.splitlines()]
    assert [(row['phase'], row['status']) for row in rows] == [('gpu_probe', 'starting'), ('gpu_probe', 'unavailable')]
    mount = next(''.join(cell['source']) for cell in notebook['cells'] if "drive.mount(" in ''.join(cell['source']))
    with pytest.raises(RuntimeError, match='before mounting Drive'):
        exec(mount, scope)


def test_late_drive_receipt_cannot_change_new_attempt(queued):
    service, project, now = queued
    service.attempt(project)
    service.record_stage(project, 'starting_notebook', 1)
    now[0] += timedelta(minutes=11)
    timeout = service.status(project)
    assert timeout['attempt_receipts'][-1]['code'] == 'notebook_timeout'
    now[0] = datetime.fromisoformat(timeout['next_attempt_at'])
    service.attempt(project)
    service.observe(project, ColabObservationRequest(status='unavailable', attempt_number=2,
                                                     details={'phase': 'gpu_probe'}))
    result = service.observe(project, ColabObservationRequest(status='auth', attempt_number=1,
                                                               details={'auth_kind': 'drive_consent'}))
    assert result['status'] == 'waiting_for_gpu'
    assert result['last_observation']['attempt_number'] == 2
    assert [(item['attempt_number'], item['code'], item['late']) for item in result['attempt_receipts']] == [
        (1, 'notebook_timeout', False), (2, 'gpu_unavailable', False), (1, 'drive_authorization', True)]


def test_third_gpu_denial_slows_retries_and_explicit_quota_takes_precedence(queued):
    service, project, now = queued
    for number in range(1, 4):
        service.attempt(project)
        request = ColabObservationRequest(status='unavailable', attempt_number=number)
        result = service.observe(project, request)
        assert result['consecutive_gpu_denials'] == number
        assert service.observe(project, request)['next_attempt_at'] == result['next_attempt_at']
        interval = datetime.fromisoformat(result['next_attempt_at']) - now[0]
        low, high = (15, 30) if number == 3 else (5, 10)
        assert timedelta(minutes=low) <= interval <= timedelta(minutes=high)
        now[0] += interval
    service.attempt(project)
    service.observe(project, ColabObservationRequest(status='quota', attempt_number=4))
    deadline = now[0] + timedelta(minutes=4)
    explicit = provider_retry_deadline('Please try again at ' + deadline.isoformat(), now=now[0])
    assert explicit == deadline
    result = service.observe(project, ColabObservationRequest(status='quota', attempt_number=4,
                                                             retry_not_before=explicit))
    assert datetime.fromisoformat(result['next_attempt_at']) == deadline
    assert not result['can_attempt']


def test_verified_shard_can_be_reused_and_corrupt_copy_cannot_replace_target(tmp_path):
    source, cached, fresh = (tmp_path / name for name in ('source.tar', 'cached.tar', 'fresh.tar'))
    source.write_bytes(b'authorized-shard')
    item = {'size_bytes': source.stat().st_size, 'sha256': colab_data.file_sha256(source)}
    colab_data.copy_verified_shard(source, cached, item)
    colab_data.copy_verified_shard(cached, fresh, item)
    assert colab_data.shard_matches(fresh, item)
    cached.write_bytes(b'Damaged-data')
    with pytest.raises(ValueError, match='checksum'):
        colab_data.copy_verified_shard(cached, fresh, item)
    assert colab_data.shard_matches(fresh, item)


def test_backup_restores_missing_data_from_the_shared_verified_cache(tmp_path, monkeypatch):
    inputs = tmp_path / 'inputs'
    metadata, cache, scratch = inputs / 'full-data', inputs / 'verified-shards', tmp_path / 'runtime'
    for directory in (metadata, cache, scratch):
        directory.mkdir(parents=True)
    manifest, stored = metadata / 'data_manifest.json', cache / 'temporary'
    manifest.write_bytes(b'{}')
    stored.write_bytes(b'cached-data')
    digest = colab_data.file_sha256(stored)
    stored.rename(cache / digest)
    record = {'path': 'train/shards/first.tar', 'kind': 'train_shard',
              'sha256': digest, 'size_bytes': len(b'cached-data')}
    transfer = {'metadata_directory': 'full-data', 'manifest_path': 'data_manifest.json',
                'manifest_sha256': colab_data.file_sha256(manifest), 'files': [record], 'augmentation': {}}
    monkeypatch.setattr(colab_fallback_worker, 'estimate_ephemeral_disk_budget',
                       lambda *args: {'enough': True})
    restored = colab_fallback_worker.restore_backup_data(inputs, scratch, transfer)
    assert restored.read_bytes() == manifest.read_bytes()
    assert colab_data.shard_matches(scratch / record['path'], record)


def test_backup_approval_notice_is_one_time_and_only_one_target_is_claimed(queued, tmp_path):
    service, project, now = queued
    service.enable_autorun(project, 'auto')
    service.attempt(project)
    service.observe(project, ColabObservationRequest(status='unavailable', attempt_number=1))
    now[0] += timedelta(minutes=59)
    assert service.reserve_backup(project) is None
    assert not service.status(project)['pending_notice']
    now[0] += timedelta(minutes=2)
    assert service.reserve_backup(project) is None
    notice = service.status(project)['pending_notice']
    assert service.claim_notice(project, notice['id']) == notice
    assert service.claim_notice(project, notice['id']) is None
    service.authorize_backup(project, [ColabFallbackTarget(provider='local', state_dir=str(tmp_path))])
    reserved = service.reserve_backup(project)
    assert reserved['backup']['provider'] == 'local'
    assert service.reserve_backup(project) is None
    assert not service.status(project)['can_attempt']
    service.record_backup(project, reserved['backup']['run_id'], {'status': 'not_ready'}, launched=True)
    assert service.status(project)['can_attempt']


def test_backup_worker_runs_the_resume_command_only_with_verified_state(tmp_path, monkeypatch):
    project = tmp_path.resolve()
    run_id = uuid4().hex
    work = project / '.autoresearch-fallback' / run_id
    work.mkdir(parents=True)
    checkpoints = project / 'checkpoints'
    checkpoints.mkdir()
    saved = checkpoints / 'state.pt'
    saved.write_bytes(b'training state')
    archive = work / 'source.zip'
    with zipfile.ZipFile(archive, 'w') as bundle:
        bundle.writestr('train.py', 'import os, sys\nfrom pathlib import Path\n'
                        'assert sys.argv[1] == "--resume"\n'
                        'Path(os.environ["AUTORESEARCH_RESULTS_DIR"], "resumed").touch()\n')
    bundle_digest = colab_data.file_sha256(archive)
    receipt = colab_data.write_checkpoint_receipt(checkpoints, [saved], bundle_digest, None)
    contract = {'files': ['state.pt'], 'resume_command': f'"{sys.executable}" train.py --resume'}
    payload = {'run_id': run_id, 'bundle_sha256': bundle_digest, 'checkpoint_receipt': receipt,
               'require_resume': True, 'target': {'state_dir': str(project)},
               'config': {'checkpoint': contract, 'run_command': 'exit 1'}}
    compile(colab_fallback_worker.backup_worker_source(payload), 'backup-worker.py', 'exec')
    monkeypatch.setattr(colab_fallback_worker, '__file__', str(work / 'worker.py'))
    monkeypatch.setattr(colab_fallback_worker, 'verify_backup_cuda', lambda requirements: None)
    assert colab_fallback_worker.backup_main(payload) == 0
    assert (project / 'results' / 'resumed').is_file()
    assert json.loads((work / 'receipt.json').read_text(encoding='utf-8'))['resumed'] is True
    saved.write_bytes(b'corrupted state')
    with pytest.raises(ValueError, match='SHA-256'):
        colab_fallback_worker.verify_backup_checkpoint(checkpoints, contract, receipt, bundle_digest, None, True)


def test_private_dataset_backup_reports_not_ready_before_running(tmp_path, monkeypatch):
    project = tmp_path.resolve()
    run_id = uuid4().hex
    work = project / '.autoresearch-fallback' / run_id
    work.mkdir(parents=True)
    archive = work / 'source.zip'
    with zipfile.ZipFile(archive, 'w'):
        pass
    monkeypatch.setattr(colab_fallback_worker, '__file__', str(work / 'worker.py'))
    payload = {'run_id': run_id, 'bundle_sha256': colab_data.file_sha256(archive),
               'config': {'hf_dataset_transfer': {'repo_id': 'fixture/private-corpus'}},
               'target': {'state_dir': str(project)}}
    assert colab_fallback_worker.backup_main(payload) == 1
    saved = json.loads((work / 'receipt.json').read_text(encoding='utf-8'))
    assert saved['status'] == 'not_ready'
    assert 'private Dataset streaming in Colab' in saved['reason']


@pytest.mark.asyncio
async def test_backup_only_checks_the_approved_running_autodl_instance(queued, monkeypatch):
    service, _, _ = queued

    class ExplicitInstance:
        def __init__(self, settings):
            pass

        async def status(self, instance):
            assert instance == 'existing-instance'
            return 'stopped'

        async def close(self):
            pass

    monkeypatch.setattr(colab_fallback, 'AutoDLClient', ExplicitInstance)
    job = {'backup': {'provider': 'autodl', 'target': {'instance_uuid': 'existing-instance'}}}
    receipt = await colab_fallback.run_backup_step(service.settings, job, launch=True)
    assert receipt['status'] == 'not_ready'
    assert 'not running' in receipt['reason']
