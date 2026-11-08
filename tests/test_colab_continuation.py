from __future__ import annotations

import copy
import json

import pytest

from autoresearch.services.colab_continuation import plan_authorization_continuation


STARTED = '2026-11-08T12:00:00+00:00'
EXPECTED = [('markdown','# prepared source'),
            ('code','RUNTIME_ID = generate_runtime_identity()'),
            ('code','EMBEDDED_SOURCE = source_zip'),
            ('code',"from google.colab import drive, files\ndrive.mount('/content/drive')"),
            ('code','define_helpers()'),('code','upload_verified_data()'),('code','run_experiment()')]


def stream(status):
    return {'output_type':'stream','name':'stdout','text':'AUTORESEARCH_COLAB_STATUS '+json.dumps({
        'project_id':'speaker','runtime_id':'runtime-1','started_at':STARTED,'status':status})+'\n'}


def cells():
    result = [{'id':f'cell-{index}','cell_type':kind,'source':source,'execution_count':index if index<4 else None,'outputs':[]}
              for index,(kind,source) in enumerate(EXPECTED)]
    result[1]['outputs'] = [stream('starting'),stream('available')]
    result[3]['outputs'] = [stream('auth'),{'output_type':'error','ename':'ValueError','evalue':'mount failed'}]
    return result


def plan(actual,**kwargs):
    return plan_authorization_continuation(actual,EXPECTED,project_id='speaker',runtime_id='runtime-1',
        expected_started_at=STARTED,auth_interrupted=True,**kwargs)


def test_mount_retry_preserves_initialized_identity_and_advances_one_cell_after_success():
    actual = cells()
    first = plan(actual)
    assert (first.action,first.cell_id) == ('run','cell-3')
    actual[3]['execution_count'] = 8
    actual[3]['outputs'] = []
    assert (plan(actual).action,plan(actual).cell_id) == ('run','cell-4')
    actual[4]['execution_count'] = 9
    assert plan(actual,inflight_cell_id='cell-4').cell_id == 'cell-5'


@pytest.mark.parametrize('evidence',['inflight','stream','native_running'])
def test_pending_or_running_cell_is_never_dispatched_twice(evidence):
    actual = cells()
    actual[3]['outputs'] = []
    kwargs = {}
    if evidence == 'inflight':
        kwargs['inflight_cell_id'] = 'cell-4'
    elif evidence == 'stream':
        actual[4]['outputs'] = [{'output_type':'stream','name':'stdout','text':'uploading'}]
    else:
        actual[4]['metadata'] = {'executionInfo':{'status':'running'}}
    result = plan(actual,**kwargs)
    assert result.action == 'wait' and result.cell_id == 'cell-4'


@pytest.mark.parametrize('execution_count',[None,3])
def test_repeated_mount_failure_and_unrelated_later_error_require_user_action(execution_count):
    actual = cells()
    actual[3]['execution_count'] = execution_count
    assert plan(actual,inflight_cell_id='cell-3').action == 'blocked'
    actual[3]['outputs'] = []
    actual[3]['execution_count'] = 8
    actual[4].update(execution_count=10,outputs=[{'output_type':'error','ename':'ValueError','evalue':'bad inputs'}])
    assert plan(actual).action == 'blocked'


@pytest.mark.parametrize('change',['source','runtime','start_time','duplicate_id','unfinished_start'])
def test_source_or_runtime_identity_mismatch_cannot_continue(change):
    actual = cells()
    if change == 'source':
        actual[5]['source'] = 'unreviewed code'
    elif change in {'runtime','start_time'}:
        marker = json.loads(actual[1]['outputs'][1]['text'].split(' ',1)[1])
        marker['runtime_id' if change=='runtime' else 'started_at'] = 'other' if change=='runtime' else '2026-11-07T12:00:00+00:00'
        actual[1]['outputs'][1]['text'] = 'AUTORESEARCH_COLAB_STATUS '+json.dumps(marker)
    elif change == 'duplicate_id':
        actual[5]['id'] = actual[4]['id']
    else:
        actual[2]['execution_count'] = None
    with pytest.raises(ValueError):
        plan(actual)


def test_terminal_result_stops_continuation_and_complete_cells_only_monitor_results():
    actual = cells()
    actual[-1]['outputs'] = [stream('completed')]
    assert plan(actual).action == 'blocked'
    for cell in actual:
        cell.update(execution_count=10,outputs=[])
    actual[1]['outputs'] = [stream('starting'),stream('available')]
    assert plan(actual).action == 'complete'


def test_normal_run_all_is_never_rescheduled_by_authorization_planner():
    assert plan_authorization_continuation(cells(),EXPECTED,project_id='speaker',runtime_id='runtime-1',
        expected_started_at=STARTED,auth_interrupted=False).action == 'wait'
