from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from autoresearch import colab_browser_cli
from autoresearch.api import AppState
from autoresearch.config import Settings
from autoresearch.domain import ColabObservationRequest, ColabPrepareRequest, ResearchRequest
from autoresearch.services import colab


@pytest.fixture
def prepared_job(tmp_path):
    settings = Settings.load(tmp_path)
    state = AppState(settings)
    project = state.db.create_project(ResearchRequest(topic='Colab login handoff'), settings.workspace_dir)
    generated = Path(project.workspace) / 'generated'
    generated.mkdir(parents=True)
    (generated / 'train.py').write_text('print("test only")\n', encoding='utf-8')
    (generated / 'experiment_manifest.json').write_text(json.dumps({'colab': {
        'run_command': 'python train.py',
        'checkpoint': {
            'resume_supported': True, 'full_training_state': True, 'interval_seconds': 300,
            'files': ['resume_latest.pt'], 'resume_command': 'python train.py --resume auto',
        },
    }}), encoding='utf-8')
    assert state.colab.prepare(project.id, ColabPrepareRequest())['prepared']
    state.colab.enable_autorun(project.id, 'chrome')
    return state, project.id


@pytest.fixture
def browser_processes(monkeypatch):
    """Use real job persistence with simulated, isolated browser ownership."""
    running = {'login': None, 'worker': {}}
    calls = []

    def launch_login(root, **kwargs):
        assert running['login'] is None
        assert not running['worker']
        calls.append(('login', kwargs['browser']))
        running['login'] = 1234
        return 1234

    def launch_worker(root, project_id, **kwargs):
        assert running['login'] is None
        assert not running['worker']
        calls.append(('worker', project_id))
        running['worker'][project_id] = 5678
        return 5678

    monkeypatch.setattr(colab_browser_cli, 'login_browser_is_running', lambda root: running['login'])
    monkeypatch.setattr(colab_browser_cli, 'worker_is_running', lambda root, pid: running['worker'].get(pid))
    monkeypatch.setattr(colab_browser_cli, 'launch_colab_login_browser', launch_login)
    monkeypatch.setattr(colab_browser_cli, 'launch_colab_worker', launch_worker)
    return running, calls


def google_requires_an_ordinary_browser(state, project_id):
    return state.colab.observe(project_id, ColabObservationRequest(status='auth', details={
        'auth_kind': 'google_sign_in', 'login_browser': 'required', 'manual_action': True,
        'reason': 'Google requires an ordinary browser for sign-in',
    }))


@pytest.mark.asyncio
async def test_scheduler_rechecks_login_after_ordinary_window_closes_even_after_restart(prepared_job, browser_processes):
    state, project_id = prepared_job
    running, calls = browser_processes
    google_requires_an_ordinary_browser(state, project_id)

    assert await state.launch_due_colab_jobs() == [
        {'status': 'login_browser_opened', 'pid': 1234, 'project_id': project_id},
    ]
    assert await state.launch_due_colab_jobs() == []
    assert state.colab.status(project_id)['attempt_count'] == 0
    assert calls == [('login', 'chrome')]

    restarted = AppState(state.settings)
    running['login'] = None
    assert (await restarted.launch_due_colab_jobs())[0]['status'] == 'worker_launched'
    job = restarted.colab.status(project_id)
    assert job['ordinary_login_recheck'] and 'ordinary_login' not in job
    assert job['attempt_count'] == 0 and job['status'] == 'queued'
    assert calls == [('login', 'chrome'), ('worker', project_id)]
    assert await restarted.launch_due_colab_jobs() == []


@pytest.mark.asyncio
async def test_ordinary_login_does_not_shorten_account_gpu_cooldown(prepared_job, browser_processes, monkeypatch):
    state, project_id = prepared_job
    running, calls = browser_processes
    times = [datetime.now(timezone.utc)]
    monkeypatch.setattr(colab, '_utc', lambda: times[0])
    quota = state.colab.observe(project_id, ColabObservationRequest(status='quota', retry_after_minutes=60))
    google_requires_an_ordinary_browser(state, project_id)
    assert (await state.launch_due_colab_jobs())[0]['status'] == 'login_browser_opened'

    running['login'] = None
    assert await state.launch_due_colab_jobs() == []
    waiting = state.colab.status(project_id)
    deadline = datetime.fromisoformat(quota['next_attempt_at'])
    assert waiting['status'] == 'waiting_for_gpu' and waiting['next_attempt_at'] == quota['next_attempt_at']
    assert calls == [('login', 'chrome')] and waiting['attempt_count'] == 0

    times[0] = deadline + timedelta(seconds=1)
    assert (await state.launch_due_colab_jobs())[0]['status'] == 'worker_launched'
    assert calls == [('login', 'chrome'), ('worker', project_id)]


@pytest.mark.asyncio
async def test_rejected_recheck_does_not_reopen_ordinary_login_until_user_action(prepared_job, browser_processes):
    state, project_id = prepared_job
    running, calls = browser_processes
    google_requires_an_ordinary_browser(state, project_id)
    await state.launch_due_colab_jobs()
    running['login'] = None
    await state.launch_due_colab_jobs()
    running['worker'].clear()

    google_requires_an_ordinary_browser(state, project_id)
    assert await state.launch_due_colab_jobs() == []
    assert await state.launch_due_colab_jobs() == []
    assert state.chatgpt.colab_status(project_id)['next_action']['action'] == 'open_colab_login'
    assert calls == [('login', 'chrome'), ('worker', project_id)]


@pytest.mark.asyncio
async def test_shared_profile_race_preserves_due_authorized_job(prepared_job, browser_processes, monkeypatch):
    state, project_id = prepared_job
    _, calls = browser_processes
    available_launcher = colab_browser_cli.launch_colab_worker

    def busy(*args, **kwargs):
        raise colab_browser_cli.ColabProfileBusy('Another browser owns the profile')

    monkeypatch.setattr(colab_browser_cli, 'launch_colab_worker', busy)
    old = state.colab.status(project_id)
    assert await state.launch_due_colab_jobs() == []
    waiting = state.colab.status(project_id)
    assert waiting['status'] == 'queued' and waiting['can_attempt']
    assert waiting['next_attempt_at'] == old['next_attempt_at']
    assert waiting['last_observation'] is None and waiting['attempt_count'] == 0

    monkeypatch.setattr(colab_browser_cli, 'launch_colab_worker', available_launcher)
    assert (await state.launch_due_colab_jobs())[0]['status'] == 'worker_launched'
    assert calls == [('worker', project_id)]
