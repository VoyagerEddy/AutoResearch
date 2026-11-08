"""Pure continuation planning for a verified notebook interrupted by Drive auth.

The browser must still bind the returned native cell ID to its current tab and
dispatch it through Google's normal notebook interface. This module neither
executes code nor allocates a runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class AuthorizationContinuation:
    action: str
    cell_id: str | None
    reason: str


def _source(cell: Mapping[str, Any]) -> str:
    source = cell.get('source')
    if isinstance(source, list) and all(isinstance(line, str) for line in source):
        source = ''.join(source)
    if not isinstance(source, str):
        raise ValueError('Unrecognized notebook source during authorization continuation')
    return source


def _stream_markers(cell: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = []
    for output in cell.get('outputs') or []:
        if not isinstance(output, dict) or output.get('output_type') != 'stream' or output.get('name') not in {'stdout','stderr'}:
            continue
        text = output.get('text')
        if isinstance(text,list) and all(isinstance(line,str) for line in text):
            text = ''.join(text)
        if not isinstance(text,str):
            continue
        for line in text.splitlines():
            prefix = 'AUTORESEARCH_COLAB_STATUS '
            if not line.strip().startswith(prefix):
                continue
            try:
                marker = json.loads(line.strip()[len(prefix):])
            except (ValueError,TypeError):
                continue
            if isinstance(marker,dict):
                result.append(marker)
    return result


def plan_authorization_continuation(
    actual_cells: Sequence[Mapping[str, Any]], expected_cells: Sequence[tuple[str,str]], *,
    project_id: str, runtime_id: str, expected_started_at: str,
    auth_interrupted: bool, inflight_cell_id: str | None = None,
) -> AuthorizationContinuation:
    """Select at most one pending cell; never replay runtime initialization.

    Activate this only after a recorded Drive auth failure halted Run all. Keep
    ``inflight_cell_id`` until that cell finishes; a poll with no output is not
    permission to dispatch it again. An unrelated failure requires user review.
    """
    if not auth_interrupted:
        return AuthorizationContinuation('wait',None,'No recorded authorization interruption')
    cells = list(actual_cells)
    if cells and cells[0].get('cell_type') == 'code' and not _source(cells[0]).strip():
        cells = cells[1:]
    observed = [(cell.get('cell_type'),_source(cell)) for cell in cells]
    if observed != list(expected_cells):
        raise ValueError('Notebook source differs from the prepared authorization continuation')
    identifiers = [cell.get('id') for cell in cells]
    if (any(not isinstance(cell_id,str) or not cell_id or len(cell_id)>200 for cell_id in identifiers)
            or len(set(identifiers)) != len(identifiers)):
        raise ValueError('Notebook continuation requires unique native cell identifiers')
    try:
        started = datetime.fromisoformat(expected_started_at.replace('Z','+00:00'))
        if started.tzinfo is None:
            raise ValueError('Missing timezone')
    except (AttributeError,TypeError,ValueError):
        raise ValueError('Notebook continuation requires its original runtime start time') from None
    markers = []
    for cell in cells:
        for marker in _stream_markers(cell):
            if marker.get('project_id') != project_id or marker.get('runtime_id') != runtime_id:
                continue
            try:
                timestamp = datetime.fromisoformat(marker['started_at'].replace('Z','+00:00'))
            except (KeyError,AttributeError,TypeError,ValueError):
                continue
            if timestamp.tzinfo is not None and timestamp == started:
                markers.append(marker)
    if not any(marker.get('status') == 'starting' for marker in markers) or not any(marker.get('status') == 'available' for marker in markers):
        raise ValueError('The original GPU runtime could not be established for continuation')
    if any(marker.get('status') in {'completed','failed','disconnected','unavailable'} for marker in markers):
        return AuthorizationContinuation('blocked',None,'The runtime has already reported a terminal result')
    mount_indices = [index for index,(_,source) in enumerate(observed)
                     if 'from google.colab import drive, files' in source and "drive.mount('/content/drive')" in source]
    if len(mount_indices) != 1:
        raise ValueError('Prepared notebook must contain one recognizable Drive mount cell')
    mount_index = mount_indices[0]
    for cell in cells[:mount_index]:
        if cell.get('cell_type') == 'code' and (cell.get('execution_count') is None or any(
                isinstance(output,dict) and output.get('output_type') == 'error' for output in cell.get('outputs') or [])):
            raise ValueError('Runtime initialization did not finish before Drive authorization')
    for index,cell in enumerate(cells[mount_index:],start=mount_index):
        if cell.get('cell_type') != 'code':
            continue
        cell_id = cell['id']
        outputs = cell.get('outputs') or []
        failed = any(isinstance(output,dict) and output.get('output_type') == 'error' for output in outputs)
        running = (cell.get('execution_state') in {'running','busy','queued'} or
                   (cell.get('metadata') or {}).get('executionInfo',{}).get('status') in {'running','busy','queued'})
        if running:
            return AuthorizationContinuation('wait',cell_id,'The pending cell is already executing')
        if failed:
            if index != mount_index:
                return AuthorizationContinuation('blocked',cell_id,'A later experiment cell failed; do not replay it automatically')
            if not any(marker.get('status') == 'auth' and marker.get('project_id') == project_id
                       and marker.get('runtime_id') == runtime_id for marker in _stream_markers(cell)):
                return AuthorizationContinuation('blocked',cell_id,'Drive setup failed outside its authorization step')
            if cell_id == inflight_cell_id:
                return AuthorizationContinuation('blocked',cell_id,'Drive authorization retry failed; user action is required')
            return AuthorizationContinuation('run',cell_id,'Retry the interrupted Drive mount without resetting runtime identity')
        if cell.get('execution_count') is None and (outputs or cell_id == inflight_cell_id):
            return AuthorizationContinuation('wait',cell_id,'The pending cell is already executing')
        if cell.get('execution_count') is None:
            return AuthorizationContinuation('run',cell_id,'Continue with the next unexecuted prepared cell')
    return AuthorizationContinuation('complete',None,'All prepared code cells have finished; observe their scientific result')
