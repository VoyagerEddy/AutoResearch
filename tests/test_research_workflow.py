"""Isolated evidence/iteration tests; fabricated fixtures are never Colab acceptance."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from autoresearch.config import Settings
from autoresearch.db import Database
from autoresearch.domain import ResearchRequest
from autoresearch.services.artifacts import ArtifactStore
from autoresearch.services.colab_data import validate_completion_details
from autoresearch.services.research_workflow import ResearchWorkflowConflict, ResearchWorkflowService


@pytest.fixture
def study(tmp_path):
    settings = Settings.load(tmp_path)
    db = Database(tmp_path / 'test.sqlite3')
    project = db.create_project(ResearchRequest(topic='Evidence-bound research'), settings.workspace_dir)
    store = ArtifactStore(Path(project.workspace))
    store.write_text('generated/train.py', 'def train(): return None\n')
    declaration = {'id': 'h1-full', 'hypothesis_ids': ['H1'], 'dataset_ids': ['train', 'dev'],
                   'scope': 'full', 'implementation_files': ['train.py']}
    manifest = {'schema_version': 1, 'run_command': 'python train.py', 'required_files': ['train.py'],
                'research_coverage': {'experiments': [declaration]}}
    store.write_json('generated/experiment_manifest.json', manifest)
    plan = {'title': 'Fixture research plan', 'hypotheses': [{'id': 'H1', 'description': 'Fixture hypothesis'}],
            'datasets': [{'id': 'train', 'scope': 'full', 'labels': 'available'},
                         {'id': 'dev', 'scope': 'evaluation', 'labels': 'available'}],
            'experiments': [{'id': 'h1-full', 'hypothesis_ids': ['H1'], 'dataset_ids': ['train', 'dev'],
                             'scope': 'full', 'implementation_files': ['train.py'], 'seed': 42}],
            'success_criteria': {'required_metrics': ['fixture_score', 'fixture_loss']}}
    return ResearchWorkflowService(db, settings), project.id, store, plan, manifest


def complete_fixture(service, pid, store, *, acceptance=False):
    _, files, _ = service._source(store)
    bundle = store.workspace / 'colab/source.zip'
    bundle.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.writestr(path, (store.workspace / 'generated' / path).read_bytes())
    kind = 'acceptance_passed' if acceptance else 'campaign_complete'
    criteria = {'validation': 'scientific_metrics', 'status_value': kind,
                'metrics_file': 'fixture/metrics.json', 'required_metrics': ['fixture_score', 'fixture_loss']}
    details = {'runtime_id': 'runtime-current', 'returncode': 0, 'completion_kind': kind,
               'scientific_metrics_produced': not acceptance, 'metrics_file': criteria['metrics_file'],
               'metrics': {'fixture_score': 4.0, 'fixture_loss': 3.8}}
    job = {'project_id': pid, 'prepared': True, 'status': 'completed', 'runtime_id': 'runtime-current',
           'success_criteria': criteria, 'completion': validate_completion_details(details, criteria),
           'bundle_sha256': hashlib.sha256(bundle.read_bytes()).hexdigest(),
           'last_observation': {'status': 'completed', 'details': details}}
    store.write_json('colab/job.json', job)
    return job


def decision(status, action='accept', **extra):
    return {**{key: status[key] for key in ('plan_id', 'plan_sha256', 'code_sha256', 'evidence_id')},
            'action': action, 'analysis': 'ChatGPT reviewed the current evidence.', **extra}


def test_unplanned_status_is_read_only_and_does_not_change_ready_project(study):
    service, pid, store, _, _ = study
    before = service.db.get_project(pid).model_dump()
    status = service.get_status(pid)
    assert status['status'] == 'unplanned'
    assert not (store.workspace / 'research/workflow').exists()
    assert service.db.get_project(pid).model_dump() == before


def test_code_coverage_is_not_experimental_completion_or_all_hypotheses(study):
    service, pid, _, plan, _ = study
    plan['hypotheses'].append({'id': 'H2', 'description': 'Temporal bottleneck'})
    plan['experiments'].append({'id': 'h2-full', 'hypothesis_ids': ['H2'], 'dataset_ids': ['train', 'dev'], 'scope': 'full'})
    status = service.save_plan(pid, plan)
    assert status['reasoning_owner'] == 'chatgpt'
    assert [row['status'] for row in status['coverage']['hypotheses']] == ['implemented', 'planned']
    assert not status['all_required_experiments_completed']
    assert status['status'] == 'implementation_required'


def test_cannot_advance_or_accept_without_measured_results(study):
    service, pid, _, plan, _ = study
    status = service.save_plan(pid, plan)
    for action in ('advance', 'accept'):
        with pytest.raises(ResearchWorkflowConflict, match='full-data'):
            service.record_decision(pid, decision(status, action, next_experiment_ids=['h1-full']))


def test_acceptance_only_result_is_tested_not_full_data_completed(study):
    service, pid, store, plan, _ = study
    service.save_plan(pid, plan)
    complete_fixture(service, pid, store, acceptance=True)
    status = service.get_status(pid)
    assert status['coverage']['experiments'][0]['status'] == 'tested'
    assert not status['all_required_experiments_completed']
    with pytest.raises(ResearchWorkflowConflict, match='acceptance'):
        service.record_decision(pid, decision(status))


def test_full_result_requires_chatgpt_analysis_before_plan_completion(study):
    service, pid, store, plan, _ = study
    service.save_plan(pid, plan)
    complete_fixture(service, pid, store)
    status = service.get_status(pid)
    assert status['status'] == 'analysis_required'
    assert status['all_required_experiments_completed']
    assert status['analysis_required']
    accepted = service.record_decision(pid, decision(status))
    assert accepted['status'] == 'completed'
    assert not accepted['research_effect_validated']
    assert not accepted['latest_decision']['execution_authorized']


def test_partial_coverage_cannot_accept_whole_research_plan(study):
    service, pid, store, plan, _ = study
    plan['hypotheses'].append({'id': 'H4', 'description': 'Cross-language consistency'})
    plan['experiments'].append({'id': 'h4-full', 'hypothesis_ids': ['H4'], 'dataset_ids': ['train', 'dev'], 'scope': 'full'})
    service.save_plan(pid, plan)
    complete_fixture(service, pid, store)
    status = service.get_status(pid)
    with pytest.raises(ResearchWorkflowConflict, match='partially'):
        service.record_decision(pid, decision(status))


@pytest.mark.parametrize(('scope', 'expected'), [
    (['h1-full'], ['completed', 'implemented']),
    (['h2-full'], ['implemented', 'completed']),
    (None, ['implemented', 'implemented']),
    ([], ['implemented', 'implemented']),
    (['h1-full', 'unknown'], ['implemented', 'implemented']),
    (['h1-full', 'h1-full'], ['implemented', 'implemented']),
])
def test_current_colab_result_proves_only_experiments_named_in_prepared_contract(study, scope, expected):
    service, pid, store, plan, manifest = study
    second = {'id': 'h2-full', 'hypothesis_ids': ['H2'], 'dataset_ids': ['train', 'dev'],
              'scope': 'full', 'implementation_files': ['h2.py']}
    plan['hypotheses'].append({'id': 'H2', 'description': 'Temporal bottleneck'})
    plan['experiments'].append(second.copy())
    manifest['research_coverage']['experiments'].append(second)
    if scope is not None:
        manifest['colab'] = {'research_experiment_ids': scope}
    store.write_text('generated/h2.py', 'def train(): return 2\n')
    store.write_json('generated/experiment_manifest.json', manifest)
    service.save_plan(pid, plan)
    complete_fixture(service, pid, store)

    result = service.get_status(pid)
    assert [item['status'] for item in result['coverage']['experiments']] == expected
    assert result['evidence']['research_experiment_ids'] == (scope if 'completed' in expected else [])
    assert not result['all_required_experiments_completed']
    with pytest.raises(ResearchWorkflowConflict):
        service.record_decision(pid, decision(result))


def test_two_plan_experiments_cannot_share_attribution_through_one_manifest_identifier(study):
    service, pid, store, plan, _ = study
    alternate = {**plan['experiments'][0], 'id': 'second-seed', 'manifest_id': 'h1-full', 'seed': 43}
    plan['experiments'].append(alternate)
    service.save_plan(pid, plan)
    complete_fixture(service, pid, store)
    status = service.get_status(pid)
    assert status['evidence']['research_experiment_ids'] == ['h1-full']
    assert [item['status'] for item in status['coverage']['experiments']] == ['implemented', 'implemented']
    assert not status['all_required_experiments_completed']


@pytest.mark.parametrize('labels', ['unknown', 'unavailable'])
def test_missing_labels_cannot_be_completed_even_with_numeric_metrics(study, labels):
    service, pid, store, plan, _ = study
    plan['datasets'][1]['labels'] = labels
    service.save_plan(pid, plan)
    complete_fixture(service, pid, store)
    status = service.get_status(pid)
    assert status['coverage']['experiments'][0]['unknown_or_missing_labels'] == ['dev']
    assert not status['all_required_experiments_completed']
    with pytest.raises(ResearchWorkflowConflict):
        service.record_decision(pid, decision(status))


def test_current_source_changes_invalidate_old_results_and_decisions(study):
    service, pid, store, plan, _ = study
    service.save_plan(pid, plan)
    complete_fixture(service, pid, store)
    before = service.get_status(pid)
    service.record_decision(pid, decision(before))
    store.write_text('generated/train.py', 'def train(): return 2\n')
    after = service.get_status(pid)
    assert after['status'] != 'completed'
    assert after['code_changed_since_plan_save']
    assert not after['evidence']['full_data_completed']
    assert not after['latest_decision_is_current']
    with pytest.raises(ResearchWorkflowConflict, match='code_sha256'):
        service.record_decision(pid, decision(before))


def test_plan_revisions_preserve_immutable_history_and_reject_stale_parent(study):
    service, pid, store, plan, _ = study
    first = service.save_plan(pid, plan)
    old = store.workspace / 'research/workflow/plans' / (first['plan_id'] + '.json')
    old_bytes = old.read_bytes()
    revised = copy.deepcopy(plan)
    revised.update(title='Second hypothesis cycle', parent_plan_id=first['plan_id'])
    second = service.save_plan(pid, revised)
    assert second['plan_id'] != first['plan_id']
    assert second['parent_plan_id'] == first['plan_id']
    assert old.read_bytes() == old_bytes
    with pytest.raises(ResearchWorkflowConflict, match='Parent plan'):
        service.save_plan(pid, revised)
    with pytest.raises(ResearchWorkflowConflict, match='plan_id'):
        service.record_decision(pid, decision(first, 'stop'))


def test_decision_chain_preserves_stop_and_revision_without_execution(study):
    service, pid, store, plan, _ = study
    initial = service.save_plan(pid, plan)
    stopped = service.record_decision(pid, decision(initial, 'stop'))
    assert stopped['status'] == 'stopped'
    revised = service.record_decision(pid, decision(stopped, 'revise'))
    assert revised['status'] == 'revision_required'
    assert revised['latest_decision']['parent_decision_id'] == stopped['latest_decision']['decision_id']
    assert len(list((store.workspace / 'research/workflow/decisions').glob('*.json'))) == 2
    assert not revised['executes_experiments']


@pytest.mark.parametrize('path', ['../../outside.py', '.env', 'D:/outside.py'])
def test_plan_implementation_paths_cannot_escape_generated(study, path):
    service, pid, _, plan, _ = study
    plan['experiments'][0]['implementation_files'] = [path]
    with pytest.raises(ValueError):
        service.save_plan(pid, plan)


def test_manifest_scope_or_data_mismatch_does_not_claim_implementation(study):
    service, pid, store, plan, manifest = study
    manifest['research_coverage']['experiments'][0]['scope'] = 'acceptance'
    store.write_json('generated/experiment_manifest.json', manifest)
    status = service.save_plan(pid, plan)
    assert not status['coverage']['experiments'][0]['implemented']


def test_duplicate_plan_ids_and_supplied_completion_claims_are_rejected(study):
    service, pid, _, plan, _ = study
    original = copy.deepcopy(plan)
    plan['hypotheses'].append(copy.deepcopy(plan['hypotheses'][0]))
    with pytest.raises(ValueError, match='Duplicate'):
        service.save_plan(pid, plan)
    original['experiments'][0]['status'] = 'completed'
    with pytest.raises(ValueError, match='derived from evidence'):
        service.save_plan(pid, original)


def test_full_experiment_cannot_use_acceptance_dataset(study):
    service, pid, _, plan, _ = study
    plan['datasets'][0]['scope'] = 'acceptance'
    with pytest.raises(ValueError, match='acceptance-only'):
        service.save_plan(pid, plan)


def test_runtime_and_source_zip_are_required_current_evidence(study):
    service, pid, store, plan, _ = study
    service.save_plan(pid, plan)
    job = complete_fixture(service, pid, store)
    before = service.get_status(pid)
    job['runtime_id'] = 'new-runtime'
    store.write_json('colab/job.json', job)
    assert not service.get_status(pid)['evidence']['tested']
    with pytest.raises(ResearchWorkflowConflict, match='evidence_id'):
        service.record_decision(pid, decision(before))
    job['runtime_id'] = 'runtime-current'
    store.write_json('colab/job.json', job)
    (store.workspace / 'colab/source.zip').write_bytes(b'corrupted upload')
    assert not service.get_status(pid)['evidence']['tested']


def test_current_result_must_satisfy_plan_metric_declarations(study):
    service, pid, store, plan, _ = study
    plan['success_criteria']['required_metrics'].append('cross_language_gain')
    service.save_plan(pid, plan)
    complete_fixture(service, pid, store)
    status = service.get_status(pid)
    assert status['missing_required_metrics'] == ['cross_language_gain']
    with pytest.raises(ResearchWorkflowConflict, match='incomplete'):
        service.record_decision(pid, decision(status))


def test_advance_records_next_experiment_but_never_starts_it(study):
    service, pid, store, plan, _ = study
    service.save_plan(pid, plan)
    complete_fixture(service, pid, store)
    current = service.get_status(pid)
    updated = service.record_decision(pid, decision(current, 'advance', next_experiment_ids=['h1-full']))
    assert updated['latest_decision']['next_experiment_ids'] == ['h1-full']
    assert not updated['latest_decision']['executes_experiments']
    assert service.db.list_experiments(pid) == []


def test_workflow_uses_the_same_source_exclusions_as_the_colab_exporter(study):
    service, pid, store, plan, manifest = study
    manifest['colab'] = {'source_exclude_paths': ['private-download.sh']}
    store.write_json('generated/experiment_manifest.json', manifest)
    store.write_text('generated/private-download.sh', 'local-only settings\n')
    service.save_plan(pid, plan)
    complete_fixture(service, pid, store)
    first = service.get_status(pid)
    store.write_text('generated/private-download.sh', 'changed local-only settings\n')
    store.write_text('generated/colab_prepared/runtime.json', '{"temporary":true}')
    second = service.get_status(pid)
    assert first['code_sha256'] == second['code_sha256']
    assert second['evidence']['full_data_completed']
