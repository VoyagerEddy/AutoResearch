from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from autoresearch.config import Settings
from autoresearch.db import Database
from autoresearch.domain import ColabPrepareRequest, ResearchRequest
from autoresearch.services.colab import ColabConflict, ColabService


def prepared(tmp_path):
    settings = Settings.load(tmp_path)
    database = Database(settings.data_dir / 'test.sqlite3')
    project = database.create_project(ResearchRequest(topic='Current Colab code'), settings.workspace_dir)
    generated = Path(project.workspace) / 'generated'
    generated.mkdir(parents=True)
    (generated / 'train.py').write_text('print("training")\n', encoding='utf-8')
    contract = {'run_command': 'python train.py', 'checkpoint': {
        'resume_supported': True, 'full_training_state': True, 'interval_seconds': 300,
        'files': ['latest.pt'], 'resume_command': 'python train.py --resume'}}
    (generated / 'experiment_manifest.json').write_text(json.dumps({'colab': contract}), encoding='utf-8')
    service = ColabService(database, settings)
    service.prepare(project.id, ColabPrepareRequest())
    return service, project.id, generated, contract


def test_source_edits_block_running_an_old_notebook_without_rewriting_snapshot(tmp_path):
    service, project, generated, _ = prepared(tmp_path)
    before = (service._directory(project) / 'job.json').read_bytes()
    assert service.source_matches_prepared_bundle(project) is True
    assert (service._directory(project) / 'job.json').read_bytes() == before
    (generated / 'train.py').write_text('print("changed hypothesis")\n')
    state = service.status(project)
    assert state['source_current'] is False and not state['can_attempt']
    with pytest.raises(ColabConflict, match='changed after preparation'):
        service.enable_autorun(project, 'auto')
    assert service.prepare(project, ColabPrepareRequest())['source_current'] is True


@pytest.mark.parametrize('change', ['added', 'removed', 'renamed'])
def test_source_inventory_changes_invalidate_prepared_code(tmp_path, change):
    service, project, generated, _ = prepared(tmp_path)
    if change == 'added':
        (generated / 'new_model.py').write_text('VERSION = 2\n')
    elif change == 'removed':
        (generated / 'train.py').unlink()
    else:
        (generated / 'train.py').rename(generated / 'renamed.py')
    assert service.source_matches_prepared_bundle(project) is False


def test_legacy_jobs_use_verified_zip_and_generated_runtime_outputs_are_not_source(tmp_path):
    service, project, generated, _ = prepared(tmp_path)
    job = service._read(project)
    job.pop('source_manifest_sha256')
    service._write(project, job)
    for folder in ('inputs', 'checkpoints', 'results', 'colab_prepared'):
        (generated / folder).mkdir()
        (generated / folder / 'runtime.json').write_text('{"runtime":true}')
    assert service.source_matches_prepared_bundle(project) is True
    (service._directory(project) / 'source.zip').write_bytes(b'corrupted snapshot')
    assert service.source_matches_prepared_bundle(project) is False


def test_declared_exclusions_keep_unused_private_download_scripts_out_of_export(tmp_path):
    service, project, generated, contract = prepared(tmp_path)
    (generated / 'private-download.sh').write_text('PRIVATE_DOWNLOAD_CONFIGURATION=local-only\n')
    contract['source_exclude_paths'] = ['private-download.sh']
    (generated / 'experiment_manifest.json').write_text(json.dumps({'colab': contract}))
    assert service.prepare(project, ColabPrepareRequest())['prepared']
    with zipfile.ZipFile(service.export_path(project, 'bundle')) as archive:
        assert 'private-download.sh' not in archive.namelist()
    (generated / 'private-download.sh').write_text('PRIVATE_DOWNLOAD_CONFIGURATION=changed\n')
    assert service.source_matches_prepared_bundle(project) is True


@pytest.mark.parametrize('field,value', [
    ('source_exclude_paths', ['../escape']),
    ('source_exclude_paths', ['experiment_manifest.json']),
    ('preflight_command', ['python check.py']),
    ('preflight_timeout_seconds', True),
    ('preflight_timeout_seconds', 3601),
    ('runtime_requirements', {'framework': 'unknown'}),
    ('runtime_requirements', {'require_cuda': 'true'}),
    ('runtime_requirements', {'require_cuda': True}),
])
def test_invalid_preflight_or_export_declarations_block_preparation(tmp_path, field, value):
    service, project, generated, contract = prepared(tmp_path)
    contract[field] = value
    (generated / 'experiment_manifest.json').write_text(json.dumps({'colab': contract}))
    assert service.prepare(project, ColabPrepareRequest())['status'] == 'blocked'


def test_unprepared_project_has_no_claim_of_source_freshness(tmp_path):
    service, project, _, _ = prepared(tmp_path)
    job = service._read(project)
    job['prepared'] = False
    service._write(project, job)
    assert service.source_matches_prepared_bundle(project) is None
