"""Launch and observe an approved backup host without provisioning cloud compute."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import sys
import tempfile
from typing import Any

from ..config import Settings
from ..domain import ColabFallbackTarget, SSHConnection
from .autodl import AutoDLClient, extract_ssh
from .colab_data import file_sha256
from .colab_fallback_worker import backup_worker_source


ACTIVE_BACKUP = {'launching', 'checking', 'running', 'unknown'}


def validate_target(target: ColabFallbackTarget) -> dict[str, str]:
    path_type = Path if target.provider == 'local' else PurePosixPath
    for value in (target.state_dir, target.data_root):
        if value and (not path_type(value).is_absolute() or '..' in path_type(value).parts
                      or any(character in value for character in ('\0', '\r', '\n'))):
            raise ValueError('Backup paths must be absolute directories on the approved GPU host')
    if bool(target.instance_uuid) != (target.provider == 'autodl'):
        raise ValueError('AutoDL backup requires an explicitly approved existing instance')
    if target.python and (not path_type(target.python).is_absolute()
                          or any(character in target.python for character in ('\0', '\r', '\n'))):
        raise ValueError('A custom backup interpreter must be an absolute executable path')
    return target.model_dump()


def _package(job: dict[str, Any], destination: Path) -> None:
    backup = job['backup']
    payload = {
        'run_id': backup['run_id'], 'bundle_sha256': job['bundle_sha256'],
        'config': job['prepared_config'], 'target': backup['target'],
        'checkpoint_receipt': job.get('checkpoint_receipt'),
        'require_resume': bool(job.get('training_started') or job.get('running_at') or job.get('checkpoint_receipt')),
    }
    archive = destination / 'source.zip'
    shutil.copyfile(job['bundle_path'], archive)
    if file_sha256(archive) != job['bundle_sha256']:
        raise ValueError('Backup source archive changed after approval')
    (destination / 'worker.py').write_text(backup_worker_source(payload), encoding='utf-8')


def _local_directory(backup: dict[str, Any], *, create: bool = False) -> Path:
    root = Path(backup['target']['state_dir']).resolve(strict=True)
    parent = root / '.autoresearch-fallback'
    if parent.is_symlink() or not parent.resolve().is_relative_to(root):
        raise ValueError('Unsafe local backup work directory')
    directory = parent / backup['run_id']
    if create:
        directory.mkdir(parents=True, exist_ok=False)
    if directory.is_symlink() or not directory.resolve().is_relative_to(root):
        raise ValueError('Unsafe local backup run directory')
    return directory


def _launch_local(job: dict[str, Any]) -> dict[str, Any]:
    state = job['backup']
    try:
        directory = _local_directory(state, create=True)
    except FileExistsError:
        return {'status': 'unknown', 'reason': 'Backup work directory already exists. Waiting for its original receipt.'}
    _package(job, directory)
    flags = getattr(subprocess, 'DETACHED_PROCESS', 0) | getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0)
    with (directory / 'worker.log').open('ab') as log:
        process = subprocess.Popen([state['target']['python'] or sys.executable, str(directory / 'worker.py')],
            cwd=directory, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            creationflags=flags, start_new_session=os.name != 'nt')
    return {'status': 'checking', 'pid': process.pid}


def _read_local(backup: dict[str, Any]) -> dict[str, Any]:
    receipt = _local_directory(backup) / 'receipt.json'
    if not receipt.is_file():
        return {}
    if receipt.is_symlink() or receipt.stat().st_size > 100_000:
        raise ValueError('Unsafe backup worker receipt')
    return json.loads(receipt.read_text(encoding='utf-8'))


async def _autodl_connection(settings: Settings, target: dict[str, str]) -> SSHConnection:
    client = AutoDLClient(settings)
    try:
        instance = target['instance_uuid']
        if await client.status(instance) != 'running':
            raise ValueError('Approved AutoDL instance is not running')
        connection = extract_ssh(await client.snapshot(instance))
        if not connection:
            raise ValueError('Approved AutoDL instance is not ready for SSH')
        return SSHConnection(**connection)
    finally:
        await client.close()


def _remote_directory(backup: dict[str, Any]) -> str:
    return str(PurePosixPath(backup['target']['state_dir']) / '.autoresearch-fallback' / backup['run_id'])


def _launch_remote(job: dict[str, Any], connection: SSHConnection) -> dict[str, Any]:
    from .ssh_runner import SSHRunner

    backup = job['backup']
    destination = _remote_directory(backup)
    state_dir = shlex.quote(backup['target']['state_dir'])
    directory = shlex.quote(destination)
    python = shlex.quote(backup['target']['python'] or 'python3')
    with tempfile.TemporaryDirectory(prefix='autoresearch-backup-') as folder:
        _package(job, Path(folder))
        with SSHRunner(connection) as runner:
            exit_code, output, _ = runner.execute(
                f"test -d {state_dir} && test ! -L {state_dir}/.autoresearch-fallback"
                f" && mkdir -p {state_dir}/.autoresearch-fallback"
                f" && (mkdir {directory} || (test -d {directory} && printf existing))")
            if exit_code == 0 and output == 'existing':
                return {'status': 'unknown', 'reason': 'Backup work directory already exists. Waiting for its original receipt.'}
            if exit_code:
                raise ValueError('Approved AutoDL project state directory is unavailable')
            if runner.gpu_processes():
                raise ValueError('Approved AutoDL instance GPU is occupied')
            runner.upload_tree(Path(folder), destination)
            try:
                pid = runner.start(destination, f'{python} worker.py')
            except Exception:
                # A disconnected SSH response is ambiguous after dispatch.
                return {'status': 'unknown', 'reason': 'Backup launch has no return receipt yet'}
            return {'status': 'checking', 'pid': pid}


def _read_remote(backup: dict[str, Any], connection: SSHConnection) -> dict[str, Any]:
    from .ssh_runner import SSHRunner

    with SSHRunner(connection) as runner:
        return runner.read_json(_remote_directory(backup) + '/receipt.json', max_bytes=100_000)


async def run_backup_step(settings: Settings, job: dict[str, Any], *, launch: bool) -> dict[str, Any]:
    backup = job['backup']
    try:
        if backup['provider'] == 'local':
            return await asyncio.to_thread(_launch_local, job) if launch else await asyncio.to_thread(_read_local, backup)
        connection = await _autodl_connection(settings, backup['target'])
        if launch:
            return await asyncio.to_thread(_launch_remote, job, connection)
        return await asyncio.to_thread(_read_remote, backup, connection)
    except Exception as exc:
        # Connection errors may include machine-specific paths or credentials.
        reason = str(exc) if type(exc) is ValueError else type(exc).__name__
        return {'status': 'not_ready' if launch else 'unknown', 'reason': reason[:200]}
