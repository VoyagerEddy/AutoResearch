from __future__ import annotations

import ast
import asyncio
import base64
import hashlib
import io
import json
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from autoresearch.api import AppState, create_app
from autoresearch.config import Settings
from autoresearch.db import Database
from autoresearch.domain import ColabNotebookBindingRequest, ColabObservationRequest, ColabPrepareRequest, ResearchRequest
from autoresearch.services.colab import ColabConflict, ColabService


def config() -> dict:
    return {'schema_version': 1, 'setup_commands': [], 'prepare_command': '',
            'run_command': 'python train.py', 'required_drive_paths': [],
            'checkpoint': {'resume_supported': True, 'full_training_state': True,
                           'interval_seconds': 300, 'files': ['resume_latest.pt'],
                           'resume_command': 'python train.py --resume auto'}}


def project_service(tmp_path: Path):
    settings = Settings.load(tmp_path)
    db = Database(settings.data_dir / 'test.sqlite3')
    project = db.create_project(ResearchRequest(topic='Colab speaker verification'), settings.workspace_dir)
    generated = Path(project.workspace) / 'generated'
    generated.mkdir(parents=True)
    (generated / 'train.py').write_text('print("real project runner")\n', encoding='utf-8')
    (generated / 'experiment_manifest.json').write_text(json.dumps({'colab': config()}), encoding='utf-8')
    return ColabService(db, settings), project.id, generated


def test_preparation_is_not_an_attempt_and_source_export_excludes_inputs_and_secrets(tmp_path):
    service, project_id, generated = project_service(tmp_path)
    (generated / '.env').write_text('PRIVATE=never-export-me')
    (generated / 'inputs').mkdir()
    (generated / 'inputs' / 'metadata.json').write_text('{"private_data": true}')
    (generated / 'inputs' / 'weights.pt').write_bytes(b'weights')
    (generated / 'checkpoints').mkdir()
    (generated / 'checkpoints' / 'state.json').write_text('{"state": true}')
    job = service.prepare(project_id, ColabPrepareRequest())
    assert job['status'] == 'queued' and job['can_attempt'] is True
    assert job['attempt_count'] == 0 and job['last_observation'] is None
    with zipfile.ZipFile(service.export_path(project_id, 'bundle')) as bundle:
        assert set(bundle.namelist()) == {'train.py', 'experiment_manifest.json'}
    notebook = json.loads(service.export_path(project_id, 'notebook').read_text(encoding='utf-8'))
    code = '\n'.join(''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code')
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']))
    assert 'AUTORESEARCH_COLAB_STATUS' in code and 'drive.mount' in code
    assert 'checkpoint_dir' in code and 'resume_command' in code
    assert 'http://127.0.0.1' not in code
    assert 'ssh' not in code.lower()
    embedded_cell = next(''.join(cell['source']) for cell in notebook['cells'] if 'EMBEDDED_SOURCE =' in ''.join(cell['source']))
    namespace = {}
    exec(embedded_cell, namespace)
    assert hashlib.sha256(namespace['EMBEDDED_SOURCE']).hexdigest() == job['bundle_sha256']
    markdown = '\n'.join(''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'markdown')
    assert f'AUTORESEARCH_SOURCE_SHA256 {job["bundle_sha256"]}' in markdown
    assert "'_' + RUNTIME_ID" in code


@pytest.mark.parametrize('change', ['source', 'contract', 'drive_root'])
def test_reprepare_changed_export_invalidates_saved_notebook_url(tmp_path, change):
    service, project_id, generated = project_service(tmp_path)
    first = service.prepare(project_id, ColabPrepareRequest())
    url = 'https://colab.research.google.com/drive/saved-notebook'
    service.observe(project_id, ColabObservationRequest(status='auth', details={'browser_url': url}))
    assert service.status(project_id)['notebook_url'] == url
    request = ColabPrepareRequest()
    if change == 'source':
        (generated / 'train.py').write_text('print("updated runner")\n')
    elif change == 'contract':
        spec = config()
        spec['run_command'] = 'python train.py --updated'
        (generated / 'experiment_manifest.json').write_text(json.dumps({'colab': spec}))
    else:
        request = ColabPrepareRequest(drive_root='OtherAutoResearch')
    updated = service.prepare(project_id, request)
    assert updated['notebook_url'] == updated['browser_url'] == ''
    if change != 'drive_root':
        assert updated['bundle_sha256'] != first['bundle_sha256']
    notebook = json.loads(service.export_path(project_id, 'notebook').read_text(encoding='utf-8'))
    markdown = '\n'.join(''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'markdown')
    assert f'AUTORESEARCH_SOURCE_SHA256 {updated["bundle_sha256"]}' in markdown


def test_reprepare_identical_source_preserves_hash_and_saved_url(tmp_path):
    service, project_id, _ = project_service(tmp_path)
    first = service.prepare(project_id, ColabPrepareRequest())
    url = 'https://colab.research.google.com/drive/saved-notebook'
    service.observe(project_id, ColabObservationRequest(status='auth', details={'browser_url': url}))
    updated = service.prepare(project_id, ColabPrepareRequest())
    assert updated['bundle_sha256'] == first['bundle_sha256']
    assert updated['notebook_url'] == updated['browser_url'] == url


def test_reprepare_changed_source_retains_explicit_url_for_worker_validation(tmp_path):
    service, project_id, generated = project_service(tmp_path)
    first = service.prepare(project_id, ColabPrepareRequest())
    (generated / 'train.py').write_text('print("updated runner")\n')
    url = 'https://colab.research.google.com/drive/explicit-notebook'
    updated = service.prepare(project_id, ColabPrepareRequest(notebook_url=url))
    assert updated['bundle_sha256'] != first['bundle_sha256']
    assert updated['notebook_url'] == url


@pytest.mark.parametrize('change', ['source','drive_root','terminal'])
def test_reprepare_new_execution_scope_drops_old_runtime_markers_but_keeps_provider_cooldown(tmp_path, monkeypatch, change):
    service, project_id, generated = project_service(tmp_path)
    now = datetime(2026,11,8,12,0,tzinfo=timezone.utc)
    monkeypatch.setattr('autoresearch.services.colab._utc',lambda:now)
    service.prepare(project_id,ColabPrepareRequest())
    previous = service._read(project_id)
    cooldown = (now+timedelta(minutes=9)).isoformat()
    previous.update(status='completed' if change=='terminal' else 'auth_required',runtime_id='old-smoke',
                    runtime_started_at=(now-timedelta(hours=2)).isoformat(),
                    last_attempt_at=(now-timedelta(hours=2)).isoformat(),
                    last_observation={'status':'auth','observed_at':now.isoformat(),
                        'details':{'runtime_active':True,'runtime_id':'old-smoke','started_at':now.isoformat()}},
                    recovery={'mode':'attach','runtime_id':'old-smoke'},provider_retry_not_before=cooldown)
    service._write(project_id,previous)
    request = ColabPrepareRequest()
    if change=='source':
        (generated/'train.py').write_text('print("full campaign")\n')
    elif change=='drive_root':
        request = ColabPrepareRequest(drive_root='FullCampaign')
    updated = service.prepare(project_id,request)
    assert updated['last_observation'] is None and updated['last_attempt_at'] is None
    assert 'runtime_id' not in updated and 'runtime_started_at' not in updated and 'recovery' not in updated
    assert updated['provider_retry_not_before'] == cooldown
    assert updated['next_attempt_at'] == cooldown and not updated['can_attempt']


def test_reprepare_identical_active_scope_retains_runtime_for_authentication_resume(tmp_path):
    service, project_id, _ = project_service(tmp_path)
    service.prepare(project_id,ColabPrepareRequest())
    previous = service._read(project_id)
    previous.update(status='auth_required',runtime_id='active-full',runtime_started_at=datetime.now(timezone.utc).isoformat(),
                    last_observation={'status':'auth','details':{'runtime_active':True,'runtime_id':'active-full'}})
    service._write(project_id,previous)
    updated = service.prepare(project_id,ColabPrepareRequest())
    assert updated['runtime_id'] == 'active-full'
    assert updated['last_observation']['details']['runtime_active'] is True


def test_missing_checkpoint_contract_pauses_export_without_claiming_resume(tmp_path):
    service, project_id, generated = project_service(tmp_path)
    (generated / 'experiment_manifest.json').write_text('{"run_command":"python train.py"}')
    job = service.prepare(project_id, ColabPrepareRequest())
    assert job['status'] == 'blocked' and not job['prepared'] and not job['can_attempt']
    assert 'checkpoint/resume' in job['blocked_reason']
    with pytest.raises(ColabConflict):
        service.export_path(project_id, 'notebook')


@pytest.mark.parametrize('mode', ['campaign_complete', 'acceptance_passed'])
def test_service_completion_enforces_prepared_scientific_contract(tmp_path, mode):
    service, project_id, generated = project_service(tmp_path)
    spec = config()
    spec['success_criteria'] = {'validation':'scientific_metrics','metrics_file':'campaign/metrics.json',
                                'status_value':mode,'required_metrics':['B0_eer_percent','VQ_min_dcf']}
    (generated/'experiment_manifest.json').write_text(json.dumps({'colab':spec}))
    service.prepare(project_id, ColabPrepareRequest())
    service.observe(project_id, ColabObservationRequest(status='running', details={'runtime_id':'current'}))
    minimal = {'runtime_id':'current','returncode':0}
    with pytest.raises(ColabConflict, match='prepared scientific'):
        service.observe(project_id, ColabObservationRequest(status='completed',details=minimal))
    valid = {**minimal, 'completion_kind':mode,'scientific_metrics_produced':mode=='campaign_complete',
             'metrics_file':'campaign/metrics.json','metrics':{'B0_eer_percent':5.0,'VQ_min_dcf':0.4}}
    for key, bad in [('completion_kind','process_exit'), ('scientific_metrics_produced', mode!='campaign_complete'),
                     ('metrics_file','other.json'), ('metrics',{'B0_eer_percent':5.0,'VQ_min_dcf':float('nan')})]:
        with pytest.raises(ColabConflict):
            service.observe(project_id, ColabObservationRequest(status='completed',details={**valid,key:bad}))
        assert service.status(project_id)['status'] == 'running'
    result = service.observe(project_id, ColabObservationRequest(status='completed',details=valid))
    assert result['completion']['completion_kind'] == mode
    assert result['completion']['scientific_metrics_produced'] is (mode=='campaign_complete')


def test_invalid_scientific_success_contract_is_blocked_before_allocation(tmp_path):
    service, project_id, generated = project_service(tmp_path)
    spec = config()
    spec['success_criteria'] = {'validation':'scientific_metrics', 'metrics_file':'metrics.json', 'required_metrics':[]}
    (generated/'experiment_manifest.json').write_text(json.dumps({'colab':spec}))
    result = service.prepare(project_id,ColabPrepareRequest())
    assert result['status'] == 'blocked' and not result['can_attempt']


def test_unavailable_retry_is_persisted_jitter_and_reprepare_keeps_cooldown(tmp_path, monkeypatch):
    service, project_id, _ = project_service(tmp_path)
    now = datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr('autoresearch.services.colab._utc', lambda: now)
    service.prepare(project_id, ColabPrepareRequest())
    assert service.attempt(project_id)['attempt_count'] == 1
    with pytest.raises(ColabConflict):
        service.attempt(project_id)
    job = service.observe(project_id, ColabObservationRequest(status='unavailable', details={'reason': 'GPU unavailable in Colab UI'}))
    next_at = datetime.fromisoformat(job['next_attempt_at'])
    assert timedelta(minutes=5) <= next_at - now <= timedelta(minutes=10)
    assert job['status'] == 'waiting_for_gpu' and job['can_attempt'] is False
    restarted = ColabService(service.db, service.settings)
    assert restarted.status(project_id)['next_attempt_at'] == job['next_attempt_at']
    assert restarted.prepare(project_id, ColabPrepareRequest())['next_attempt_at'] == job['next_attempt_at']
    with pytest.raises(ColabConflict):
        restarted.resume(project_id)
    monkeypatch.setattr('autoresearch.services.colab._utc', lambda: next_at + timedelta(seconds=1))
    assert restarted.status(project_id)['can_attempt'] is True


def test_quota_retry_respects_requested_longer_cooldown(tmp_path):
    service, project_id, _ = project_service(tmp_path)
    service.prepare(project_id, ColabPrepareRequest())
    before = datetime.now(timezone.utc)
    job = service.observe(project_id, ColabObservationRequest(status='quota', retry_after_minutes=120))
    assert datetime.fromisoformat(job['next_attempt_at']) - before >= timedelta(minutes=120)


def test_auth_is_paused_and_network_is_not_reported_as_gpu_shortage(tmp_path):
    service, project_id, _ = project_service(tmp_path)
    service.prepare(project_id, ColabPrepareRequest())
    auth = service.observe(project_id, ColabObservationRequest(status='auth'))
    assert auth['status'] == 'auth_required' and auth['next_attempt_at'] is None
    assert service.resume(project_id)['can_attempt']
    network = service.observe(project_id, ColabObservationRequest(status='network', details={'reason': 'Connection timed out'}))
    assert network['status'] == 'waiting_for_connection'
    assert network['last_observation']['status'] == 'network'


def test_attempt_lease_recovers_after_worker_crash_and_cancel_stops_retries(tmp_path, monkeypatch):
    service, project_id, _ = project_service(tmp_path)
    now = datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr('autoresearch.services.colab._utc', lambda: now)
    service.prepare(project_id, ColabPrepareRequest())
    service.attempt(project_id)
    monkeypatch.setattr('autoresearch.services.colab._utc', lambda: now + timedelta(minutes=11))
    recovered = service.status(project_id)
    assert recovered['status'] == 'waiting_for_connection' and recovered['next_attempt_at']
    cancelled = service.cancel(project_id)
    assert cancelled['status'] == 'cancelled' and not cancelled['can_attempt']
    assert cancelled['next_attempt_at'] is None


@pytest.mark.parametrize('observed,expected_status', [('running', 'running'), ('available', 'gpu_available')])
def test_dead_worker_recovery_preserves_runtime_and_first_reattaches(tmp_path, monkeypatch, observed, expected_status):
    from autoresearch import colab_browser_cli

    service, project_id, _ = project_service(tmp_path)
    clock = datetime(2026, 11, 1, 8, 0, tzinfo=timezone.utc)
    monkeypatch.setattr('autoresearch.services.colab._utc', lambda: clock)
    monkeypatch.setattr(colab_browser_cli, 'worker_is_running', lambda *_: None)
    service.prepare(project_id, ColabPrepareRequest())
    service.enable_autorun(project_id, 'chrome')
    service.attempt(project_id)
    url = 'https://colab.research.google.com/drive/saved-owned-notebook'
    active = service.observe(project_id, ColabObservationRequest(status=observed, details={
        'runtime_id': 'known-runtime', 'started_at': clock.isoformat(), 'browser_url': url}))
    assert active['status'] == expected_status  # Child PID handoff has a short grace.
    clock += timedelta(seconds=61)
    restarted = ColabService(service.db, service.settings)
    recovered = restarted.status(project_id)
    assert recovered['status'] == 'disconnected' and recovered['can_attempt']
    assert not recovered['worker_running']
    assert recovered['recovery'] == {
        'mode': 'attach', 'runtime_id': 'known-runtime', 'started_at': active['runtime_started_at'],
        'previous_status': expected_status, 'detected_at': clock.isoformat()}
    assert recovered['notebook_url'] == url
    assert recovered['last_observation'] == active['last_observation']
    assert recovered['attempt_count'] == active['attempt_count']
    assert restarted.status(project_id)['recovery'] == recovered['recovery']


def test_recovery_of_unknown_runtime_waits_random_interval(tmp_path, monkeypatch):
    from autoresearch import colab_browser_cli

    service, project_id, _ = project_service(tmp_path)
    clock = datetime(2026, 11, 1, 8, 0, tzinfo=timezone.utc)
    monkeypatch.setattr('autoresearch.services.colab._utc', lambda: clock)
    monkeypatch.setattr(colab_browser_cli, 'worker_is_running', lambda *_: None)
    service.prepare(project_id, ColabPrepareRequest())
    service.enable_autorun(project_id, 'chrome')
    service.observe(project_id, ColabObservationRequest(status='running'))
    clock += timedelta(minutes=2)
    job = service.status(project_id)
    assert job['status'] == 'disconnected' and job['recovery']['mode'] == 'retry'
    delay = datetime.fromisoformat(job['next_attempt_at']) - clock
    assert timedelta(minutes=5) <= delay <= timedelta(minutes=10)
    assert not job['can_attempt']


def test_live_worker_and_manual_pauses_do_not_enter_recovery(tmp_path, monkeypatch):
    from autoresearch import colab_browser_cli

    service, project_id, _ = project_service(tmp_path)
    clock = datetime(2026, 11, 1, 8, 0, tzinfo=timezone.utc)
    monkeypatch.setattr('autoresearch.services.colab._utc', lambda: clock)
    running = {'pid': 1234}
    monkeypatch.setattr(colab_browser_cli, 'worker_is_running', lambda *_: running['pid'])
    service.prepare(project_id, ColabPrepareRequest())
    service.enable_autorun(project_id, 'chrome')
    service.observe(project_id, ColabObservationRequest(status='running', details={'runtime_id': 'known-runtime'}))
    clock += timedelta(days=1)
    assert service.status(project_id)['status'] == 'running'
    assert 'recovery' not in service.status(project_id)
    paused = service.observe(project_id, ColabObservationRequest(status='auth', details={'reason': 'Drive authorization'}))
    running['pid'] = None
    clock += timedelta(days=1)
    assert service.status(project_id)['status'] == 'auth_required'
    assert service.status(project_id)['last_observation'] == paused['last_observation']
    assert service.status(project_id)['next_attempt_at'] is None


def test_recovery_respects_provider_barrier_and_clears_after_confirmed_disconnect(tmp_path, monkeypatch):
    from autoresearch import colab_browser_cli

    service, project_id, _ = project_service(tmp_path)
    clock = datetime(2026, 11, 1, 8, 0, tzinfo=timezone.utc)
    monkeypatch.setattr('autoresearch.services.colab._utc', lambda: clock)
    monkeypatch.setattr(colab_browser_cli, 'worker_is_running', lambda *_: None)
    service.prepare(project_id, ColabPrepareRequest())
    service.enable_autorun(project_id, 'chrome')
    quota = service.observe(project_id, ColabObservationRequest(status='quota'))
    service.observe(project_id, ColabObservationRequest(status='running', details={'runtime_id': 'known-runtime'}))
    clock += timedelta(minutes=2)
    recovered = service.status(project_id)
    assert recovered['status'] == 'waiting_for_gpu' and not recovered['can_attempt']
    assert recovered['next_attempt_at'] == quota['provider_retry_not_before']
    assert recovered['recovery']['mode'] == 'attach'
    clock = datetime.fromisoformat(quota['provider_retry_not_before']) + timedelta(seconds=1)
    confirmed = service.observe(project_id, ColabObservationRequest(status='disconnected', details={'runtime_id': 'known-runtime'}))
    assert 'recovery' not in confirmed
    assert not confirmed['can_attempt']


def test_two_service_instances_cannot_claim_the_same_attempt(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    service, project_id, _ = project_service(tmp_path)
    service.prepare(project_id, ColabPrepareRequest())
    other = ColabService(service.db, service.settings)
    def claim(instance):
        try:
            return instance.attempt(project_id)['attempt_count']
        except ColabConflict:
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(claim, (service, other)))
    assert sorted(value for value in outcomes if value is not None) == [1]
    assert service.status(project_id)['attempt_count'] == 1


def test_two_api_processes_cannot_claim_the_same_attempt(tmp_path):
    import subprocess
    import sys

    service, project_id, _ = project_service(tmp_path)
    service.prepare(project_id, ColabPrepareRequest())
    program = '''
import sys
from pathlib import Path
from autoresearch.config import Settings
from autoresearch.db import Database
from autoresearch.services.colab import ColabService, ColabConflict
s = Settings.load(Path(sys.argv[1]))
c = ColabService(Database(s.data_dir / "test.sqlite3"), s)
try:
    c.attempt(sys.argv[2])
    print("claimed")
except ColabConflict:
    print("waiting")
'''
    processes = [subprocess.Popen([sys.executable, '-c', program, str(tmp_path), project_id],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(2)]
    outcomes = [process.communicate(timeout=20) for process in processes]
    assert all(process.returncode == 0 for process in processes), outcomes
    assert sorted(output.strip() for output, _ in outcomes) == ['claimed', 'waiting']
    assert service.status(project_id)['attempt_count'] == 1


@pytest.mark.asyncio
async def test_scheduler_dispatches_recovery_once_without_claiming_new_gpu(tmp_path, monkeypatch):
    from autoresearch import colab_browser_cli

    service, project_id, _ = project_service(tmp_path)
    clock = datetime(2026, 11, 1, 8, 0, tzinfo=timezone.utc)
    monkeypatch.setattr('autoresearch.services.colab._utc', lambda: clock)
    live = {}
    monkeypatch.setattr(colab_browser_cli, 'worker_is_running', lambda root, pid: live.get(pid))
    service.prepare(project_id, ColabPrepareRequest())
    service.enable_autorun(project_id, 'chrome')
    service.attempt(project_id)
    active = service.observe(project_id, ColabObservationRequest(status='running', details={'runtime_id': 'known-runtime'}))
    clock += timedelta(seconds=61)
    calls = []
    def launch(root, pid, **kwargs):
        calls.append(pid)
        live[pid] = 4321
        return 4321
    monkeypatch.setattr(colab_browser_cli, 'launch_colab_worker', launch)
    state = scheduler_state(ColabService(service.db, service.settings))
    results = await asyncio.gather(state.launch_due_colab_jobs(), state.launch_due_colab_jobs())
    assert sum(len(result) for result in results) == 1 and calls == [project_id]
    recovered = service.status(project_id)
    assert recovered['recovery']['mode'] == 'attach'
    assert recovered['attempt_count'] == active['attempt_count']
    assert recovered['runtime_id'] == 'known-runtime'


def test_completion_requires_current_runtime_success(tmp_path):
    service, project_id, _ = project_service(tmp_path)
    service.prepare(project_id, ColabPrepareRequest())
    service.observe(project_id, ColabObservationRequest(status='running', details={'runtime_id': 'current'}))
    with pytest.raises(ColabConflict):
        service.observe(project_id, ColabObservationRequest(status='completed', details={'runtime_id': 'old', 'returncode': 0}))
    completed = service.observe(project_id, ColabObservationRequest(status='completed', details={'runtime_id': 'current', 'returncode': 0}))
    assert completed['status'] == 'completed' and not completed['can_attempt']


def test_embedded_secret_refuses_source_export_without_returning_secret(tmp_path):
    service, project_id, generated = project_service(tmp_path)
    secret = 'sk-' + 'A' * 32
    (generated / 'train.py').write_text(f'API_KEY = "{secret}"')
    job = service.prepare(project_id, ColabPrepareRequest())
    assert job['status'] == 'blocked'
    assert 'possible embedded secret' in job['blocked_reason']
    assert secret not in json.dumps(job)


def test_linked_source_is_refused(tmp_path, monkeypatch):
    service, project_id, _ = project_service(tmp_path)
    from autoresearch.services import colab
    original = colab._linked
    monkeypatch.setattr(colab, '_linked', lambda path: path.name == 'train.py' or original(path))
    job = service.prepare(project_id, ColabPrepareRequest())
    assert job['status'] == 'blocked'
    assert 'linked' in job['blocked_reason']


def input_archive(generated, members):
    inputs = generated / 'inputs'
    inputs.mkdir(exist_ok=True)
    archive = inputs / 'approved.zip'
    with zipfile.ZipFile(archive, 'w') as bundle:
        for member in members:
            bundle.writestr(member, b'approved audio fixture')
    return {'path': 'inputs/approved.zip', 'members': members,
            'sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}


def test_input_artifact_only_exports_allowlisted_members(tmp_path):
    service, project_id, generated = project_service(tmp_path)
    spec = config()
    spec['input_archive'] = input_archive(generated, ['colab-smoke/train-smoke.tar'])
    (generated / 'experiment_manifest.json').write_text(json.dumps({'colab': spec}))
    job = service.prepare(project_id, ColabPrepareRequest())
    assert job['input_bundle_path'] and job['input_bundle_url'].endswith('/inputs.zip')
    assert service.export_path(project_id, 'inputs').is_file()
    with zipfile.ZipFile(service.export_path(project_id, 'bundle')) as source:
        assert not any(path.startswith('inputs/') for path in source.namelist())


@pytest.mark.parametrize('members', [['../outside.txt'], ['colab-smoke/.env'], ['colab-smoke/key.pem']])
def test_unsafe_input_artifact_is_refused(tmp_path, members):
    service, project_id, generated = project_service(tmp_path)
    spec = config()
    spec['input_archive'] = input_archive(generated, members)
    (generated / 'experiment_manifest.json').write_text(json.dumps({'colab': spec}))
    assert service.prepare(project_id, ColabPrepareRequest())['status'] == 'blocked'


def test_colab_api_exports_and_observations(tmp_path):
    with TestClient(create_app(Settings.load(tmp_path))) as client:
        created = client.post('/api/chatgpt/projects', json={'topic': 'Colab API project'}).json()
        project_id = created['project']['id']
        client.post(f'/api/chatgpt/projects/{project_id}/code', json={
            'files': [{'path': 'train.py', 'content': 'print("train")'}],
            'experiment_manifest': {'colab': config()}})
        assert client.get('/api/projects/missing/colab').status_code == 404
        prepared = client.post(f'/api/projects/{project_id}/colab/prepare', json={})
        assert prepared.status_code == 200 and prepared.json()['prepared']
        assert client.get(f'/api/projects/{project_id}/colab/notebook.ipynb').status_code == 200
        assert client.get(f'/api/projects/{project_id}/colab/bundle.zip').status_code == 200
        assert client.post(f'/api/projects/{project_id}/colab/attempt', json={}).status_code == 200
        assert client.post(f'/api/projects/{project_id}/colab/attempt', json={}).status_code == 409
        observed = client.post(f'/api/projects/{project_id}/colab/observations', json={'status': 'unavailable'})
        assert observed.status_code == 200 and observed.json()['next_attempt_at']
        assert client.post(f'/api/projects/{project_id}/colab/cancel', json={}).json()['status'] == 'cancelled'


def test_notebook_binding_persists_public_drive_url_without_changing_observations_or_retry(tmp_path):
    service, project_id, _ = project_service(tmp_path)
    prepared = service.prepare(project_id,ColabPrepareRequest())
    quota = service.observe(project_id,ColabObservationRequest(status='quota'))
    before = service._read(project_id)
    url = 'https://colab.research.google.com/drive/owned-notebook'
    result = service.bind_notebook(project_id,ColabNotebookBindingRequest(notebook_url=url,bundle_sha256=prepared['bundle_sha256']))
    assert result['notebook_url'] == result['browser_url'] == url
    for name in ('status','attempt_count','last_observation','next_attempt_at','provider_retry_not_before','lease_until'):
        assert result.get(name) == before.get(name)
    assert result['next_attempt_at'] == quota['next_attempt_at']
    with pytest.raises(ColabConflict,match='source does not match'):
        service.bind_notebook(project_id,ColabNotebookBindingRequest(notebook_url=url,bundle_sha256='0'*64))
    service.cancel(project_id)
    with pytest.raises(ColabConflict,match='active Colab job'):
        service.bind_notebook(project_id,ColabNotebookBindingRequest(notebook_url=url,bundle_sha256=prepared['bundle_sha256']))


@pytest.mark.parametrize('url', ['https://colab.research.google.com/notebooks/empty.ipynb',
    'https://colab.research.google.com/drive/owned#private-connection',
    'https://colab.research.google.com/drive/owned?token=private',
    'https://colab.research.google.com:8443/drive/owned', 'https://drive.google.com/drive/owned'])
def test_notebook_binding_rejects_unsaved_or_private_connection_url(tmp_path,url):
    service, project_id, _ = project_service(tmp_path)
    prepared = service.prepare(project_id,ColabPrepareRequest())
    with pytest.raises(ValueError):
        service.bind_notebook(project_id,ColabNotebookBindingRequest(notebook_url=url,bundle_sha256=prepared['bundle_sha256']))
    assert service._read(project_id)['notebook_url'] == ''


def test_notebook_binding_api_updates_metadata_only(tmp_path):
    with TestClient(create_app(Settings.load(tmp_path))) as client:
        project_id = client.post('/api/chatgpt/projects',json={'topic':'Save verified notebook'}).json()['project']['id']
        client.post(f'/api/chatgpt/projects/{project_id}/code',json={
            'files':[{'path':'train.py','content':'print("train")'}],'experiment_manifest':{'colab':config()}})
        prepared = client.post(f'/api/projects/{project_id}/colab/prepare',json={}).json()
        payload = {'notebook_url':'https://colab.research.google.com/drive/new-owned-copy','bundle_sha256':prepared['bundle_sha256']}
        bound = client.post(f'/api/projects/{project_id}/colab/notebook-binding',json=payload)
        assert bound.status_code == 200 and bound.json()['status']=='queued'
        assert bound.json()['attempt_count']==0 and bound.json()['last_observation'] is None
        assert client.post(f'/api/projects/{project_id}/colab/notebook-binding',json={**payload,'bundle_sha256':'0'*64}).status_code == 409
        assert client.post('/api/projects/missing/colab/notebook-binding',json=payload).status_code == 404


def test_authorization_continuation_is_durable_and_cannot_follow_another_runtime(tmp_path):
    service,project_id,_ = project_service(tmp_path)
    service.prepare(project_id,ColabPrepareRequest())
    started = datetime.now(timezone.utc).isoformat()
    service.observe(project_id,ColabObservationRequest(status='available',details={'runtime_id':'live','started_at':started}))
    state = {'runtime_id':'live','started_at':started,'inflight_cell_id':None}
    service.observe(project_id,ColabObservationRequest(status='auth',details={
        'runtime_id':'live','runtime_active':True,'authorization_continuation':state}))
    assert ColabService(service.db,service.settings).status(project_id)['authorization_continuation']==state
    service.resume(project_id)
    state['inflight_cell_id']='native-cell-1'
    updated = service.observe(project_id,ColabObservationRequest(status='available',details={
        'runtime_id':'live','started_at':started,'authorization_continuation':state}))
    assert updated['authorization_continuation']['inflight_cell_id']=='native-cell-1'
    with pytest.raises(ColabConflict):
        service.observe(project_id,ColabObservationRequest(status='auth',details={
            'runtime_id':'other','authorization_continuation':state}))
    changed = service.observe(project_id,ColabObservationRequest(status='available',details={'runtime_id':'new','started_at':started}))
    assert 'authorization_continuation' not in changed


def test_login_api_opens_normal_browser_without_allocating_gpu(tmp_path, monkeypatch):
    from autoresearch import colab_browser_cli
    calls = []
    monkeypatch.setattr(colab_browser_cli, 'worker_is_running', lambda *args: None)
    monkeypatch.setattr(colab_browser_cli, 'launch_colab_login_browser',
                        lambda root, **kwargs: calls.append(kwargs) or 321)
    app = create_app(Settings.load(tmp_path))
    with TestClient(app) as client:
        created = client.post('/api/chatgpt/projects', json={'topic': 'Colab normal login'}).json()
        project_id = created['project']['id']
        result = client.post(f'/api/projects/{project_id}/colab/login', json={'browser': 'chrome'})
        assert result.status_code == 202 and result.json()['pid'] == 321
        job = client.get(f'/api/projects/{project_id}/colab').json()
        assert job['status'] == 'not_prepared' and job['attempt_count'] == 0
        assert calls == [{'browser': 'chrome'}]


def scheduler_state(service):
    state = AppState(service.settings)
    state.db, state.colab = service.db, service
    return state


def test_provider_quota_survives_browser_closure_and_restart(tmp_path, monkeypatch):
    service, project_id, _ = project_service(tmp_path)
    now = datetime(2026, 10, 9, 1, 55, 58, tzinfo=timezone.utc)
    monkeypatch.setattr('autoresearch.services.colab._utc', lambda: now)
    service.prepare(project_id, ColabPrepareRequest())
    quota = service.observe(project_id, ColabObservationRequest(status='quota', details={'reason': 'Colab GPU usage limit'}))
    assert datetime.fromisoformat(quota['next_attempt_at']) >= now + timedelta(minutes=720)
    closed = service.observe(project_id, ColabObservationRequest(status='network', details={
        'reason': 'Browser closed', 'browser_closed': True}))
    assert closed['status'] == 'waiting_for_gpu'
    assert closed['last_observation']['status'] == 'quota'
    assert closed['last_connection_observation']['details']['browser_closed']
    restarted = ColabService(service.db, service.settings).status(project_id)
    assert restarted['next_attempt_at'] == quota['next_attempt_at'] == restarted['provider_retry_not_before']
    assert not restarted['can_attempt']


def second_project(service):
    project=service.db.create_project(ResearchRequest(topic='Another project sharing the same Colab browser profile'),service.settings.workspace_dir)
    generated=Path(project.workspace)/'generated'
    generated.mkdir(parents=True)
    (generated/'train.py').write_text('print("second paired experiment")\n')
    (generated/'experiment_manifest.json').write_text(json.dumps({'colab':config()}))
    return project.id,generated


def test_quota_is_shared_across_projects_restart_and_reprepare_without_sliding(tmp_path,monkeypatch):
    service,project_id,_=project_service(tmp_path)
    now=datetime(2026,11,8,20,0,tzinfo=timezone.utc)
    monkeypatch.setattr('autoresearch.services.colab._utc',lambda:now)
    service.prepare(project_id,ColabPrepareRequest())
    first=service.observe(project_id,ColabObservationRequest(status='quota'))
    deadline=first['next_attempt_at']
    assert datetime.fromisoformat(deadline)==now+timedelta(hours=12)
    other_id,generated=second_project(service)
    second=service.prepare(other_id,ColabPrepareRequest())
    assert second['status']=='waiting_for_gpu' and not second['can_attempt']
    assert second['next_attempt_at']==deadline and second['shared_quota_retry_not_before']==deadline
    now+=timedelta(minutes=10)
    repeated=service.observe(project_id,ColabObservationRequest(status='quota'))
    assert repeated['next_attempt_at']==deadline
    restarted=ColabService(service.db,service.settings)
    (generated/'train.py').write_text('print("new source cannot reset account quota")\n')
    assert restarted.prepare(other_id,ColabPrepareRequest())['next_attempt_at']==deadline
    with pytest.raises(ColabConflict):
        restarted.attempt(other_id)
    longer=restarted.observe(project_id,ColabObservationRequest(status='quota',retry_after_minutes=1440))
    longer_deadline=longer['next_attempt_at']
    assert datetime.fromisoformat(longer_deadline)==now+timedelta(days=1)
    now+=timedelta(minutes=10)
    assert restarted.observe(project_id,ColabObservationRequest(status='quota',retry_after_minutes=1440))['next_attempt_at']==longer_deadline
    assert restarted.status(other_id)['next_attempt_at']==longer_deadline


def test_plain_unavailability_keeps_random_project_retry_and_cannot_shorten_a_later_quota(tmp_path,monkeypatch):
    service,project_id,_=project_service(tmp_path)
    now=datetime(2026,11,8,20,0,tzinfo=timezone.utc)
    monkeypatch.setattr('autoresearch.services.colab._utc',lambda:now)
    service.prepare(project_id,ColabPrepareRequest())
    unavailable=service.observe(project_id,ColabObservationRequest(status='unavailable'))
    assert 300<=(datetime.fromisoformat(unavailable['next_attempt_at'])-now).total_seconds()<=600
    assert not (service.settings.root/'data/colab-browser/gpu-quota.json').exists()
    other_id,_=second_project(service)
    assert service.prepare(other_id,ColabPrepareRequest())['can_attempt']
    quota=service.observe(project_id,ColabObservationRequest(status='quota'))
    assert datetime.fromisoformat(quota['next_attempt_at'])==now+timedelta(hours=12)
    assert service.status(other_id)['next_attempt_at']==quota['next_attempt_at']


def test_legacy_recorded_quota_migrates_with_its_original_deadline(tmp_path,monkeypatch):
    service,project_id,_=project_service(tmp_path)
    now=datetime(2026,11,8,20,0,tzinfo=timezone.utc)
    monkeypatch.setattr('autoresearch.services.colab._utc',lambda:now)
    service.prepare(project_id,ColabPrepareRequest())
    first=service.observe(project_id,ColabObservationRequest(status='quota'))
    (service.settings.root/'data/colab-browser/gpu-quota.json').unlink()
    now+=timedelta(hours=1)
    restarted=ColabService(service.db,service.settings)
    assert restarted.status(project_id)['next_attempt_at']==first['next_attempt_at']
    other_id,_=second_project(service)
    assert restarted.prepare(other_id,ColabPrepareRequest())['next_attempt_at']==first['next_attempt_at']


@pytest.mark.parametrize('proof',['fresh','empty','stale','before_quota','missing_runtime','bad_gpu'])
def test_only_new_fresh_real_gpu_proof_can_clear_shared_quota(tmp_path,monkeypatch,proof):
    service,project_id,_=project_service(tmp_path)
    now=datetime(2026,11,8,20,0,tzinfo=timezone.utc)
    monkeypatch.setattr('autoresearch.services.colab._utc',lambda:now)
    service.prepare(project_id,ColabPrepareRequest())
    quota=service.observe(project_id,ColabObservationRequest(status='quota'))
    other_id,_=second_project(service)
    service.prepare(other_id,ColabPrepareRequest())
    now+=timedelta(hours=1)
    details={'runtime_id':'43217198-789b-48c0-9b86-94849c0c1e33','started_at':now.isoformat(),'gpu':'Tesla T4, 15360 MiB'}
    if proof=='empty':
        details={}
    elif proof=='stale':
        details['started_at']=(now-timedelta(minutes=10)).isoformat()
    elif proof=='before_quota':
        details['started_at']=(now-timedelta(hours=2)).isoformat()
    elif proof=='missing_runtime':
        details.pop('runtime_id')
    elif proof=='bad_gpu':
        details['gpu']='GPU available'
    result=service.observe(project_id,ColabObservationRequest(status='available',details=details))
    if proof=='fresh':
        assert 'provider_retry_not_before' not in result
        other=service.status(other_id)
        assert 'provider_retry_not_before' not in other
        assert 300<=(datetime.fromisoformat(other['next_attempt_at'])-now).total_seconds()<=600
    else:
        assert result['provider_retry_not_before']==quota['next_attempt_at']
        assert service.status(other_id)['next_attempt_at']==quota['next_attempt_at']


def test_cancel_and_reprepare_cannot_bypass_provider_quota(tmp_path):
    service, project_id, _ = project_service(tmp_path)
    service.prepare(project_id, ColabPrepareRequest())
    quota = service.observe(project_id, ColabObservationRequest(status='quota'))
    service.cancel(project_id)
    prepared = service.prepare(project_id, ColabPrepareRequest())
    assert prepared['status'] == 'waiting_for_gpu'
    assert prepared['next_attempt_at'] == quota['next_attempt_at']
    assert not prepared['can_attempt']


@pytest.mark.asyncio
async def test_scheduler_launches_due_authorized_job_once_after_restart(tmp_path, monkeypatch):
    from autoresearch import colab_browser_cli
    service, project_id, _ = project_service(tmp_path)
    now = datetime(2026, 10, 9, 1, 55, 58, tzinfo=timezone.utc)
    monkeypatch.setattr('autoresearch.services.colab._utc', lambda: now)
    running, calls = {}, []
    monkeypatch.setattr(colab_browser_cli, 'worker_is_running', lambda root, pid: running.get(pid))
    def launch(root, pid, **kwargs):
        calls.append((pid, kwargs))
        running[pid] = 4321
        return 4321
    monkeypatch.setattr(colab_browser_cli, 'launch_colab_worker', launch)
    service.prepare(project_id, ColabPrepareRequest())
    service.enable_autorun(project_id, 'chrome')
    quota = service.observe(project_id, ColabObservationRequest(status='quota', retry_after_minutes=720))
    restarted = ColabService(service.db, service.settings)
    state = scheduler_state(restarted)
    assert await state.launch_due_colab_jobs() == []
    assert not calls
    due = datetime.fromisoformat(quota['next_attempt_at']) + timedelta(seconds=1)
    monkeypatch.setattr('autoresearch.services.colab._utc', lambda: due)
    responses = await asyncio.gather(state.launch_due_colab_jobs(), state.launch_due_colab_jobs())
    assert sum(len(result) for result in responses) == 1
    assert calls == [(project_id, {'api_url': f'http://127.0.0.1:{service.settings.port}', 'browser': 'chrome'})]
    job = restarted.status(project_id)
    assert job['last_worker_pid'] == 4321
    # Launching a helper is not proof that Google allocated a runtime.
    assert job['last_observation']['status'] == 'quota' and job['status'] == 'waiting_for_gpu'


@pytest.mark.asyncio
@pytest.mark.parametrize('pause', ['unauthorized', 'auth', 'failed', 'completed', 'cancelled'])
async def test_scheduler_skips_unauthorized_and_paused_or_terminal_jobs(tmp_path, monkeypatch, pause):
    from autoresearch import colab_browser_cli
    service, project_id, _ = project_service(tmp_path)
    calls = []
    monkeypatch.setattr(colab_browser_cli, 'worker_is_running', lambda *args: None)
    monkeypatch.setattr(colab_browser_cli, 'launch_colab_worker', lambda *args, **kwargs: calls.append(args) or 123)
    service.prepare(project_id, ColabPrepareRequest())
    if pause != 'unauthorized':
        service.enable_autorun(project_id, 'chrome')
    if pause in {'auth', 'failed'}:
        service.observe(project_id, ColabObservationRequest(status=pause))
    elif pause == 'completed':
        service.observe(project_id, ColabObservationRequest(status='running', details={'runtime_id': 'verified'}))
        service.observe(project_id, ColabObservationRequest(status='completed', details={'runtime_id': 'verified', 'returncode': 0}))
    elif pause == 'cancelled':
        assert service.cancel(project_id)['autorun_enabled'] is False
    assert await scheduler_state(service).launch_due_colab_jobs() == []
    assert not calls


def test_launch_api_persists_authorization_without_shortening_cooldown(tmp_path, monkeypatch):
    from autoresearch import colab_browser_cli
    calls = []
    monkeypatch.setattr(colab_browser_cli, 'worker_is_running', lambda *args: None)
    monkeypatch.setattr(colab_browser_cli, 'launch_colab_worker', lambda *args, **kwargs: calls.append(args) or 123)
    app = create_app(Settings.load(tmp_path))
    with TestClient(app) as client:
        created = client.post('/api/chatgpt/projects', json={'topic': 'Colab authorized retry'}).json()
        project_id = created['project']['id']
        client.post(f'/api/chatgpt/projects/{project_id}/code', json={
            'files': [{'path': 'train.py', 'content': 'print("train")'}],
            'experiment_manifest': {'colab': config()}})
        client.post(f'/api/projects/{project_id}/colab/prepare', json={})
        quota = client.post(f'/api/projects/{project_id}/colab/observations', json={'status': 'quota'}).json()
        result = client.post(f'/api/projects/{project_id}/colab/launch', json={'browser': 'chrome'})
        assert result.status_code == 202
        assert result.json()['status'] == 'retry_scheduled'
        assert result.json()['next_attempt_at'] == quota['next_attempt_at']
        persisted = client.get(f'/api/projects/{project_id}/colab').json()
        assert persisted['autorun_enabled'] is True and persisted['browser'] == 'chrome'
        assert not calls
        client.post(f'/api/projects/{project_id}/colab/cancel', json={})
        assert client.get(f'/api/projects/{project_id}/colab').json()['autorun_enabled'] is False


def test_lifespan_starts_and_stops_colab_scheduler(tmp_path, monkeypatch):
    lifecycle = []
    async def scheduler(self):
        lifecycle.append('started')
        try:
            await asyncio.Future()
        finally:
            lifecycle.append('stopped')
    monkeypatch.setattr(AppState, 'run_colab_scheduler', scheduler)
    with TestClient(create_app(Settings.load(tmp_path))) as client:
        assert client.get('/api/health').status_code == 200
        assert lifecycle == ['started']
    assert lifecycle == ['started', 'stopped']


@pytest.mark.asyncio
@pytest.mark.parametrize('observation', ['quota', 'auth'])
async def test_resource_agent_stays_alive_during_wait_without_altering_gpu_deadline(tmp_path, monkeypatch, observation):
    from autoresearch.services import colab_resource_monitor

    service, project_id, _ = project_service(tmp_path)
    service.prepare(project_id, ColabPrepareRequest())
    service.enable_autorun(project_id, 'chrome')
    before = service.observe(project_id, ColabObservationRequest(status=observation))
    calls = []
    monkeypatch.setattr(colab_resource_monitor, 'launch_resource_monitor',
                        lambda root, pid, **kwargs: calls.append((pid, kwargs)) or 9876)
    state = scheduler_state(service)
    result = await state.ensure_colab_resource_monitors()
    assert result == [{'project_id': project_id, 'pid': 9876}]
    assert calls == [(project_id, {'api_url': f'http://127.0.0.1:{service.settings.port}', 'browser': 'chrome'})]
    after = service.status(project_id)
    assert after['next_attempt_at'] == before['next_attempt_at']
    assert after['last_observation'] == before['last_observation']
    assert after['attempt_count'] == before['attempt_count']


@pytest.mark.asyncio
@pytest.mark.parametrize('stop', ['unauthorized', 'cancelled', 'failed', 'completed'])
async def test_resource_agent_does_not_start_for_unauthorized_or_finished_jobs(tmp_path, monkeypatch, stop):
    from autoresearch.services import colab_resource_monitor

    service, project_id, _ = project_service(tmp_path)
    service.prepare(project_id, ColabPrepareRequest())
    if stop != 'unauthorized':
        service.enable_autorun(project_id, 'chrome')
    if stop == 'cancelled':
        service.cancel(project_id)
    elif stop == 'failed':
        service.observe(project_id, ColabObservationRequest(status='failed'))
    elif stop == 'completed':
        service.observe(project_id, ColabObservationRequest(status='running', details={'runtime_id': 'live'}))
        service.observe(project_id, ColabObservationRequest(status='completed', details={'runtime_id': 'live', 'returncode': 0}))
    calls = []
    monkeypatch.setattr(colab_resource_monitor, 'launch_resource_monitor', lambda *args, **kwargs: calls.append(args) or 9876)
    assert await scheduler_state(service).ensure_colab_resource_monitors() == []
    assert not calls


@pytest.mark.parametrize('phase', ['queued', 'quota', 'auth', 'unprepared', 'cancelled', 'failed', 'completed'])
def test_select_colab_check_starts_monitor_once_and_preserves_pause(tmp_path, monkeypatch, phase):
    from autoresearch import colab_browser_cli
    from autoresearch.services import colab_resource_monitor

    running, allocations, monitors = {}, [], []
    monkeypatch.setattr(colab_browser_cli, 'worker_is_running', lambda root, pid: running.get(pid))

    def launch(root, project_id, **kwargs):
        allocations.append(project_id)
        running[project_id] = 4567
        return 4567

    monkeypatch.setattr(colab_browser_cli, 'launch_colab_worker', launch)
    monkeypatch.setattr(colab_resource_monitor, 'launch_resource_monitor',
                        lambda root, pid, **kwargs: monitors.append(pid) or 9876)
    app = create_app(Settings.load(tmp_path))
    with TestClient(app) as client:
        assert client.get('/api/health').json()['application_root'] == str(tmp_path)
        assert client.post('/api/projects/missing/colab/check', json={}).status_code == 404
        project_id = client.post('/api/chatgpt/projects', json={'topic': 'Colab selection'}).json()['project']['id']
        client.post(f'/api/chatgpt/projects/{project_id}/code', json={
            'files': [{'path': 'train.py', 'content': 'print("train")'}],
            'experiment_manifest': {'colab': config()}})
        if phase != 'unprepared':
            client.post(f'/api/projects/{project_id}/colab/prepare', json={})
        if phase in {'quota', 'auth', 'failed'}:
            client.post(f'/api/projects/{project_id}/colab/observations', json={
                'status': phase, 'details': {'manual_action': phase == 'auth'}})
        elif phase == 'cancelled':
            client.post(f'/api/projects/{project_id}/colab/cancel', json={})
        elif phase == 'completed':
            client.post(f'/api/projects/{project_id}/colab/observations', json={
                'status': 'running', 'details': {'runtime_id': 'verified'}})
            client.post(f'/api/projects/{project_id}/colab/observations', json={
                'status': 'completed', 'details': {'runtime_id': 'verified', 'returncode': 0}})
        before = client.get(f'/api/projects/{project_id}/colab').json()
        results = [client.post(f'/api/projects/{project_id}/colab/check', json={'browser': 'chrome'}) for _ in range(2)]
        assert all(result.status_code == 202 for result in results)
        after = client.get(f'/api/projects/{project_id}/colab').json()
        assert after['last_observation'] == before['last_observation']
        assert after['next_attempt_at'] == before['next_attempt_at']
        assert after['attempt_count'] == before['attempt_count']
        assert allocations == ([project_id] if phase == 'queued' else [])
        if phase in {'queued', 'quota', 'auth'}:
            assert after['autorun_enabled'] is True
            assert all(result.json()['resource_agent_pid'] == 9876 for result in results)
            assert monitors == [project_id, project_id]
        else:
            assert not monitors and after.get('autorun_enabled', False) is False
        if phase == 'auth':
            assert all(result.json()['status'] == 'auth_required' for result in results)
            assert after['status'] == 'auth_required'
