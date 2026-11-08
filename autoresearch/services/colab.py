"""Durable Colab preparation and observation queue, without an allocation API.

Only the standard Colab browser worker can request a runtime. Scheduling a job
or exporting a notebook does not establish availability or execution.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import stat
import threading
import tempfile
import time
import zipfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ..colab_notebook import build_notebook
from ..config import Settings
from ..db import Database
from ..domain import ColabNotebookBindingRequest, ColabObservationRequest, ColabPrepareRequest
from .artifacts import ArtifactStore


class ColabConflict(ValueError):
    pass


SOURCE_SUFFIXES = {'.py', '.yaml', '.yml', '.json', '.toml', '.md', '.txt', '.sh', '.cfg', '.ini'}
EXCLUDED_DIRS = {
    '.git', '.venv', 'venv', '__pycache__', '.pytest_cache', '.mypy_cache', '.cache',
    'cache', 'data', 'datasets', 'inputs', 'checkpoints', 'weights', 'results', 'logs',
    'wandb', 'node_modules', 'stage1_prepared', 'stage1_runs', 'artifacts', '.ssh',
}
EXCLUDED_NAMES = ArtifactStore.BLOCKED_NAMES | {
    'credentials.json', 'token.json', 'tokens.json', 'secrets.json', 'service-account.json',
    'preparation_inputs.json', 'input_sources.json',
}
SECRET_PATTERN = re.compile(r'(?:sk-[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9]{30,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)')
RETRY_STATUSES = {'queued', 'waiting_for_gpu', 'waiting_for_connection', 'disconnected'}
TERMINAL_STATUSES = {'completed', 'failed', 'cancelled', 'blocked'}


def _utc() -> datetime:
    return datetime.now(timezone.utc)


def _stamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _parse(value: str) -> datetime:
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError('Colab state timestamps must include a timezone')
    return result.astimezone(timezone.utc)


def _relative(value: str) -> str:
    return '/'.join(ArtifactStore._relative_parts(object.__new__(ArtifactStore), value))


def _linked(path: Path) -> bool:
    info = path.lstat()
    return path.is_symlink() or bool(getattr(info, 'st_file_attributes', 0) & 0x400)


def _colab_url(value: str) -> str:
    if not value:
        return ''
    parsed = urlparse(value)
    if parsed.scheme != 'https' or parsed.hostname not in {'colab.research.google.com', 'colab.google'} or parsed.username or parsed.password:
        raise ValueError('notebook_url must be an HTTPS Colab URL')
    return value


def _sensitive_name(value: str) -> bool:
    folded = value.casefold()
    return folded in EXCLUDED_NAMES or folded.split('.', 1)[0] in {
        'secret', 'secrets', 'credential', 'credentials', 'token', 'tokens',
        'service-account', 'service_account', 'client_secret',
    }


class ColabService:
    def __init__(self, db: Database, settings: Settings) -> None:
        self.db = db
        self.settings = settings
        self._lock = threading.RLock()
        self._transactions = threading.local()
        self._random = random.SystemRandom()

    def update_settings(self, settings: Settings) -> None:
        self.settings = settings

    def _directory(self, project_id: str) -> Path:
        project = self.db.get_project(project_id)
        if not project:
            raise LookupError('Research project not found')
        workspace = Path(project.workspace)
        if not workspace.is_absolute():
            raise ValueError('Project workspace must be absolute')
        if workspace.exists() and _linked(workspace):
            raise ValueError('Project workspace must not be a symlink or junction')
        directory = workspace / 'colab'
        if directory.exists() and _linked(directory):
            raise ValueError('Colab state directory must not be a symlink or junction')
        return directory

    def _read(self, project_id: str) -> dict[str, Any]:
        path = self._directory(project_id) / 'job.json'
        if not path.exists():
            return {'project_id': project_id, 'status': 'not_prepared', 'prepared': False,
                    'attempt_count': 0, 'next_attempt_at': None, 'last_observation': None,
                    'notebook_url': '', 'browser_url': '', 'blocked_reason': '',
                    'notebook_path': '', 'bundle_path': '', 'notebook_download_url': '',
                    'bundle_url': '', 'drive_project_dir': '',
                    'autorun_enabled': False, 'browser': 'auto'}
        if _linked(path) or path.stat().st_size > 2_000_000:
            raise ValueError('Unsafe Colab job state file')
        value = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(value, dict) or value.get('project_id') != project_id:
            raise ValueError('Invalid Colab job state')
        # Older prepared jobs are never implicitly authorized to launch a browser.
        value.setdefault('autorun_enabled', False)
        value.setdefault('browser', 'auto')
        return value

    @contextmanager
    def _transaction(self, project_id: str):
        """Serialize read/modify/write across API processes as well as threads."""
        with self._lock:
            depths = getattr(self._transactions, 'depths', None)
            if depths is None:
                depths = self._transactions.depths = {}
            if depths.get(project_id, 0):
                depths[project_id] += 1
                try:
                    yield
                finally:
                    depths[project_id] -= 1
                return
            directory = self._directory(project_id)
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / 'job.lock'
            if path.exists() and _linked(path):
                raise ValueError('Colab state lock must not be linked')
            with path.open('a+b') as handle:
                deadline = time.monotonic() + 30
                while True:
                    handle.seek(0)
                    try:
                        if os.name == 'nt':
                            import msvcrt
                            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                        else:
                            import fcntl
                            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise ColabConflict('Another local process is updating this Colab task; retry shortly') from None
                        time.sleep(0.05)
                if path.stat().st_size == 0:
                    handle.write(b'0')
                    handle.flush()
                depths[project_id] = 1
                try:
                    yield
                finally:
                    depths.pop(project_id, None)
                    handle.seek(0)
                    if os.name == 'nt':
                        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        fcntl.flock(handle, fcntl.LOCK_UN)

    def _write(self, project_id: str, job: dict[str, Any]) -> None:
        directory = self._directory(project_id)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / 'job.json'
        if path.exists() and _linked(path):
            raise ValueError('Colab state files must not be linked')
        job['updated_at'] = _stamp(_utc())
        descriptor, filename = tempfile.mkstemp(prefix='.job-', suffix='.tmp', dir=directory)
        try:
            with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
                json.dump(job, handle, ensure_ascii=False, indent=2)
                handle.write('\n')
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(filename, path)
        finally:
            Path(filename).unlink(missing_ok=True)

    def _schedule(self, job: dict[str, Any], *, minimum: int | None = None) -> None:
        low = max(job.get('retry_min_minutes', 5), minimum or 5)
        high = max(job.get('retry_max_minutes', 10), low)
        seconds = self._random.randint(low * 60, high * 60)
        next_at = _utc() + timedelta(seconds=seconds)
        provider_until = job.get('provider_retry_not_before')
        if provider_until:
            next_at = max(next_at, _parse(provider_until))
        job['next_attempt_at'] = _stamp(next_at)
        job['lease_until'] = None

    def _read_profile_quota(self) -> dict[str,Any] | None:
        directory=self.settings.root/'data'/'colab-browser'
        path=directory/'gpu-quota.json'
        if directory.exists() and _linked(directory):
            raise ValueError('Colab profile quota directory must not be linked')
        if not path.exists():
            return None
        if _linked(path) or path.stat().st_size>16000:
            raise ValueError('Unsafe Colab profile quota state')
        value=json.loads(path.read_text(encoding='utf-8'))
        if (not isinstance(value,dict) or value.get('scope')!='default_browser_profile'
                or type(value.get('retry_after_minutes')) is not int or value['retry_after_minutes']<=0):
            raise ValueError('Invalid Colab profile quota state')
        _parse(value['retry_not_before'])
        return value

    def _publish_profile_quota(self, project_id: str, retry_after_minutes: int | None, job: dict[str,Any]) -> dict[str,Any]:
        """Keep a single account-profile cooldown; repeated denials never slide it."""
        from ..colab_browser_cli import WorkerLock

        directory=self.settings.root/'data'/'colab-browser'
        directory.mkdir(parents=True,exist_ok=True)
        if _linked(directory):
            raise ValueError('Colab profile quota directory must not be linked')
        lock=directory/'gpu-quota.lock'
        if lock.exists() and _linked(lock):
            raise ValueError('Colab profile quota lock must not be linked')
        with WorkerLock(lock,wait_seconds=10):
            now=_utc()
            current=self._read_profile_quota()
            # Preserve a real denial already recorded before this version was
            # loaded, rather than restarting its default 12-hour wait.
            observation=job.get('last_observation') or {}
            until=job.get('provider_retry_not_before')
            if (current is None and observation.get('status')=='quota' and until and _parse(until)>now):
                observed=observation.get('observed_at') or _stamp(now)
                duration=max(1,int((_parse(until)-_parse(observed)).total_seconds()/60))
                current={'scope':'default_browser_profile','observed_at':observed,'retry_not_before':until,
                         'retry_after_minutes':duration,'source_project_id':project_id}
            active=bool(current and _parse(current['retry_not_before'])>now)
            if not active:
                minutes=retry_after_minutes if retry_after_minutes is not None else 720
                current={'scope':'default_browser_profile','observed_at':_stamp(now),
                         'retry_not_before':_stamp(now+timedelta(minutes=minutes)),
                         'retry_after_minutes':minutes,'source_project_id':project_id}
            elif retry_after_minutes is not None and retry_after_minutes>current['retry_after_minutes']:
                # Only a newly observed, explicitly longer provider duration
                # can extend an active barrier. Repeating it cannot extend it.
                current={**current,'retry_not_before':_stamp(max(_parse(current['retry_not_before']),now+timedelta(minutes=retry_after_minutes))),
                         'observed_at':_stamp(now),'retry_after_minutes':retry_after_minutes,'source_project_id':project_id}
            path=directory/'gpu-quota.json'
            if path.exists() and _linked(path):
                raise ValueError('Colab profile quota state must not be linked')
            descriptor,filename=tempfile.mkstemp(prefix='.gpu-quota-',suffix='.tmp',dir=directory)
            try:
                with os.fdopen(descriptor,'w',encoding='utf-8') as handle:
                    json.dump(current,handle,ensure_ascii=False,indent=2)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(filename,path)
            finally:
                Path(filename).unlink(missing_ok=True)
            return current

    def _clear_profile_quota_on_fresh_gpu(self, job: dict[str,Any], details: dict[str,Any]) -> bool:
        """Only a newly started, real-SMI runtime can supersede an older denial."""
        runtime_id,gpu=details.get('runtime_id'),details.get('gpu')
        if (runtime_id!=job.get('runtime_id') or not isinstance(gpu,str)
                or not re.fullmatch(r'[A-Za-z0-9 ._()/+-]{1,120},\s*[0-9]+(?:\.[0-9]+)?\s*(?:MiB|GiB|MB|GB)',gpu.strip())):
            return False
        try:
            uuid.UUID(runtime_id)
            started=_parse(details.get('started_at'))
        except (ValueError,TypeError,AttributeError):
            return False
        now=_utc()
        if not -30<=(now-started).total_seconds()<=120:
            return False
        from ..colab_browser_cli import WorkerLock
        directory=self.settings.root/'data'/'colab-browser'
        if not directory.exists():
            return False
        if _linked(directory) or ((directory/'gpu-quota.lock').exists() and _linked(directory/'gpu-quota.lock')):
            raise ValueError('Colab profile quota paths must not be linked')
        with WorkerLock(directory/'gpu-quota.lock',wait_seconds=10):
            current=self._read_profile_quota()
            if (not current or _parse(current['retry_not_before'])<=now
                    or started<=_parse(current['observed_at'])):
                return False
            current={**current,'superseded_retry_not_before':current['retry_not_before'],
                     'retry_not_before':_stamp(now),'cleared_at':_stamp(now),'cleared_by_runtime_id':runtime_id}
            descriptor,filename=tempfile.mkstemp(prefix='.gpu-quota-',suffix='.tmp',dir=directory)
            try:
                with os.fdopen(descriptor,'w',encoding='utf-8') as handle:
                    json.dump(current,handle,ensure_ascii=False,indent=2)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(filename,directory/'gpu-quota.json')
            finally:
                Path(filename).unlink(missing_ok=True)
            return True

    def status(self, project_id: str) -> dict[str, Any]:
        with self._transaction(project_id):
            job = self._read(project_id)
            now = _utc()
            shared=self._read_profile_quota()
            if (shared and shared.get('cleared_at') and job.get('shared_quota_retry_not_before')
                    and job.get('provider_retry_not_before')==job['shared_quota_retry_not_before']
                    and job['shared_quota_retry_not_before']==shared.get('superseded_retry_not_before')):
                previous_deadline=job.pop('shared_quota_retry_not_before')
                job.pop('provider_retry_not_before',None)
                if job['status'] in RETRY_STATUSES and job.get('next_attempt_at')==previous_deadline:
                    self._schedule(job)
                self._write(project_id,job)
            last_observation=job.get('last_observation') or {}
            if (shared is None and last_observation.get('status')=='quota' and job.get('provider_retry_not_before')
                    and _parse(job['provider_retry_not_before'])>now):
                shared=self._publish_profile_quota(project_id,None,job)
            if shared and _parse(shared['retry_not_before'])>now:
                deadline=shared['retry_not_before']
                changed=False
                if not job.get('provider_retry_not_before') or _parse(job['provider_retry_not_before'])<_parse(deadline):
                    job['provider_retry_not_before']=deadline
                    changed=True
                if job.get('shared_quota_retry_not_before')!=deadline:
                    job['shared_quota_retry_not_before']=deadline
                    changed=True
                if changed:
                    self._write(project_id,job)
            try:
                from ..colab_browser_cli import worker_is_running

                worker_pid = worker_is_running(self.settings.root, project_id)
            except ImportError:
                worker_pid = None
            if (job.get('prepared') and job.get('autorun_enabled')
                    and job['status'] in {'running', 'gpu_available'} and worker_pid is None):
                # A new child needs time to replace the Windows venv-launcher
                # PID. Fresh observations must not race that identity handoff.
                observation = job.get('last_observation') or {}
                recent = []
                for value in (observation.get('observed_at'), job.get('last_worker_launch_at'),
                              job.get('running_at'), job.get('last_attempt_at')):
                    if value:
                        try:
                            recent.append(_parse(value))
                        except (ValueError, TypeError):
                            pass
                if not recent or (now - max(recent)).total_seconds() >= 60:
                    previous_status = job['status']
                    runtime_id = job.get('runtime_id')
                    details = observation.get('details') or {}
                    started_at = job.get('runtime_started_at') or (
                        details.get('started_at') if details.get('runtime_id') == runtime_id else None
                    ) or job.get('last_attempt_at')
                    job['recovery'] = {
                        'mode': 'attach' if runtime_id else 'retry',
                        'runtime_id': runtime_id, 'started_at': started_at,
                        'previous_status': previous_status, 'detected_at': _stamp(now),
                    }
                    job.update(status='disconnected', lease_until=None,
                               blocked_reason='The local worker stopped; reconnecting to the saved notebook before requesting new compute.')
                    if runtime_id:
                        # This dispatch only reattaches the known notebook. The
                        # browser must prove that runtime was lost before Run all.
                        job['next_attempt_at'] = _stamp(now)
                    else:
                        self._schedule(job)
                    self._write(project_id, job)
                    self.db.add_event(project_id, 'colab',
                                      'Recovering a Colab task after its local worker exited.',
                                      details={'status': job['status'], 'mode': job['recovery']['mode'],
                                               'next_attempt_at': job['next_attempt_at']})
            if job.get('status') == 'attempting' and job.get('lease_until') and _parse(job['lease_until']) <= now:
                job['status'] = 'waiting_for_connection'
                job['blocked_reason'] = 'Browser attempt expired without an observation; a new attempt is scheduled.'
                self._schedule(job)
                self._write(project_id, job)
            provider_until = job.get('provider_retry_not_before')
            if (job['status'] in RETRY_STATUSES and provider_until and _parse(provider_until) > now
                    and (not job.get('next_attempt_at') or _parse(job['next_attempt_at']) < _parse(provider_until))):
                job.update(status='waiting_for_gpu', next_attempt_at=provider_until,
                           blocked_reason='Colab provider GPU cooldown remains in force.')
                self._write(project_id, job)
            job['can_attempt'] = bool(job.get('prepared') and job['status'] in RETRY_STATUSES
                and job.get('next_attempt_at') and _parse(job['next_attempt_at']) <= now)
            job['queue_only'] = job['status'] in RETRY_STATUSES
            job['rules_url'] = 'https://research.google.com/colaboratory/faq.html'
            job['worker_pid'] = worker_pid
            job['worker_running'] = worker_pid is not None
            return job

    def due_jobs(self) -> list[dict[str, Any]]:
        return [job for project in self.db.list_projects(limit=10000)
                if (job := self.status(project.id))['can_attempt']]

    def enable_autorun(self, project_id: str, browser: str) -> dict[str, Any]:
        if browser not in {'auto', 'chrome', 'edge'}:
            raise ValueError('browser must be auto, chrome, or edge')
        with self._transaction(project_id):
            job = self._read(project_id)
            if not job.get('prepared') or job['status'] in TERMINAL_STATUSES:
                raise ColabConflict('Prepare an active Colab job before enabling its browser worker')
            job.update(autorun_enabled=True, browser=browser)
            self._write(project_id, job)
            return self.status(project_id)

    def worker_launched(self, project_id: str, pid: int) -> dict[str, Any]:
        with self._transaction(project_id):
            job = self._read(project_id)
            job.update(last_worker_launch_at=_stamp(_utc()), last_worker_pid=pid)
            self._write(project_id, job)
            self.db.add_event(project_id, 'colab', 'Colab browser worker launched; GPU availability is still determined by its UI.',
                              details={'pid': pid, 'status': job['status']})
            return self.status(project_id)

    def _contract(self, generated: Path) -> dict[str, Any]:
        manifest = generated / 'experiment_manifest.json'
        if not manifest.is_file() or _linked(manifest) or manifest.stat().st_size > 2_000_000:
            raise ValueError('A regular generated/experiment_manifest.json is required')
        value = json.loads(manifest.read_text(encoding='utf-8-sig'))
        config = value.get('colab') if isinstance(value, dict) else None
        if not isinstance(config, dict):
            raise ValueError('Project needs an explicit experiment_manifest.colab checkpoint/resume contract')
        checkpoint = config.get('checkpoint', {})
        if not isinstance(checkpoint, dict) or checkpoint.get('resume_supported') is not True or checkpoint.get('full_training_state') is not True:
            raise ValueError('Colab requires verified full training state and resume_supported=true')
        for name, command in [('run_command', config.get('run_command')), ('resume_command', checkpoint.get('resume_command'))]:
            if not isinstance(command, str) or not command.strip() or len(command) > 12000:
                raise ValueError(f'Colab {name} must be an explicit nonempty command')
        if type(checkpoint.get('interval_seconds')) is not int or not 1 <= checkpoint['interval_seconds'] <= 900:
            raise ValueError('Colab checkpoint interval_seconds must be between 1 and 900')
        paths = checkpoint.get('files')
        if not isinstance(paths, list) or not paths or not all(isinstance(path, str) for path in paths):
            raise ValueError('Colab checkpoint.files must declare saved checkpoint paths')
        for path in paths:
            _relative(path)
        setup = config.get('setup_commands', [])
        if not isinstance(setup, list) or len(setup) > 30 or any(not isinstance(command, str) or len(command) > 12000 for command in setup):
            raise ValueError('Invalid Colab setup_commands')
        prepare = config.get('prepare_command', '')
        if not isinstance(prepare, str) or len(prepare) > 12000:
            raise ValueError('Invalid Colab prepare_command')
        required = config.get('required_drive_paths', [])
        if not isinstance(required, list) or len(required) > 100:
            raise ValueError('Invalid Colab required_drive_paths')
        for path in required:
            _relative(path)
        from .colab_data import validate_success_criteria

        validate_success_criteria(config.get('success_criteria'))
        return config

    def _input_bundle(self, generated: Path, config: dict[str, Any], target: Path) -> dict[str, Any]:
        archive_config = config.get('input_archive')
        if archive_config is None:
            return {}
        if not isinstance(archive_config, dict):
            raise ValueError('Colab input_archive must be an explicit archive contract')
        relative = _relative(archive_config.get('path'))
        archive = generated.joinpath(*relative.split('/'))
        for candidate in (archive, *archive.parents):
            if candidate == generated.parent:
                break
            if candidate.exists() and _linked(candidate):
                raise ValueError('Colab input archive path must not contain links')
        if not archive.is_file() or not archive.resolve().is_relative_to(generated.resolve()):
            raise ValueError('Declared Colab input archive is missing or escapes generated')
        if archive.stat().st_size > 250_000_000:
            raise ValueError('Colab input archive exceeds 250 MB')
        members = archive_config.get('members')
        if not isinstance(members, list) or not members or len(members) > 1000:
            raise ValueError('Colab input_archive.members must explicitly allowlist each member')
        approved = {_relative(path) for path in members}
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest != archive_config.get('sha256'):
            raise ValueError('Colab input archive SHA256 differs from the declared artifact')
        with zipfile.ZipFile(archive) as bundle:
            actual = set()
            infos = bundle.infolist()
            if sum(info.file_size for info in infos) > 250_000_000:
                raise ValueError('Expanded Colab input archive exceeds 250 MB')
            for info in infos:
                path = _relative(info.filename)
                if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16) or path not in approved or path in actual:
                    raise ValueError('Colab input archive contains an unapproved, linked, or duplicate member')
                if any(_sensitive_name(part) for part in path.split('/')) or Path(path).suffix.casefold() in {'.pem', '.key', '.p12', '.pfx'}:
                    raise ValueError('Colab input archive contains a sensitive filename')
                actual.add(path)
            if actual != approved:
                raise ValueError('Colab input archive does not contain the full declared member set')
        temporary = target.with_suffix('.zip.tmp')
        for candidate in (target, temporary):
            if candidate.exists() and _linked(candidate):
                raise ValueError('Input archive export target must not be linked')
        shutil.copyfile(archive, temporary)
        os.replace(temporary, target)
        return {'input_bundle_path': str(target), 'input_bundle_url': '',
                'input_bundle_sha256': digest, 'input_bundle_bytes': target.stat().st_size}

    def _bundle(self, generated: Path, target: Path) -> dict[str, Any]:
        if not generated.is_dir() or _linked(generated):
            raise ValueError('Generated source directory must be regular and unlinked')
        root = generated.resolve()
        pending: list[tuple[str, bytes]] = []
        total_bytes = 0
        for directory, dirs, files in os.walk(generated, followlinks=False):
            parent = Path(directory)
            for name in list(dirs):
                if name.casefold() in EXCLUDED_DIRS or name.startswith('.'):
                    dirs.remove(name)
                elif _linked(parent / name):
                    raise ValueError('Source bundle must not contain linked directories')
            for name in sorted(files):
                path = parent / name
                if name.startswith('.') or _sensitive_name(name) or path.suffix.casefold() not in SOURCE_SUFFIXES:
                    continue
                if _linked(path) or not path.resolve().is_relative_to(root):
                    raise ValueError('Source bundle must not contain linked or escaping files')
                relative = _relative(path.relative_to(generated).as_posix())
                size = path.stat().st_size
                if size > 5_000_000:
                    raise ValueError(f'Source file exceeds 5 MB: {relative}')
                total_bytes += size
                if total_bytes > 100_000_000 or len(pending) >= 10000:
                    raise ValueError('Source bundle exceeds the export size limit')
                content = path.read_bytes()
                decoded = content.decode('utf-8', errors='replace')
                if SECRET_PATTERN.search(decoded):
                    raise ValueError(f'Source export refused a possible embedded secret in {relative}')
                pending.append((relative, content))
        if not pending:
            raise ValueError('No generated source files available for Colab export')
        temp = target.with_suffix('.zip.tmp')
        for path in (target, temp):
            if path.exists() and _linked(path):
                raise ValueError('Export target must not be a symlink or junction')
        with zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for relative, content in sorted(pending):
                # The source identity must depend on content, not the time
                # Prepare was clicked. A stable archive hash also identifies
                # the exact embedded source in the imported notebook.
                info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, content)
        os.replace(temp, target)
        return {'source_file_count': len(pending), 'source_bytes': total_bytes,
                'bundle_sha256': hashlib.sha256(target.read_bytes()).hexdigest()}

    def prepare(self, project_id: str, request: ColabPrepareRequest) -> dict[str, Any]:
        if request.retry_max_minutes < request.retry_min_minutes:
            raise ValueError('retry_max_minutes must be >= retry_min_minutes')
        drive_root = _relative(request.drive_root)
        url = _colab_url(request.notebook_url)
        with self._transaction(project_id):
            job = self._read(project_id)
            if job['status'] in {'running', 'attempting', 'gpu_available'}:
                raise ColabConflict('A Colab attempt or experiment is active; cancel before preparing again')
            project = self.db.get_project(project_id)
            generated = Path(project.workspace) / 'generated'
            directory = self._directory(project_id)
            directory.mkdir(parents=True, exist_ok=True)
            now = _stamp(_utc())
            previous_bundle_sha256 = job.get('bundle_sha256')
            previous_drive_root = job.get('drive_root')
            job.update(project_id=project_id, retry_min_minutes=request.retry_min_minutes,
                       retry_max_minutes=request.retry_max_minutes, drive_root=drive_root,
                       drive_project_dir=f'/content/drive/MyDrive/{drive_root}/{project_id}',
                       notebook_url=url or job.get('notebook_url', ''), created_at=job.get('created_at', now))
            try:
                config = self._contract(generated)
                from .colab_data import prepare_full_data_transfer

                transfer = prepare_full_data_transfer(generated, config.get('full_data_transfer'), project_id=project_id)
                if transfer['full_data_transfer'] is not None:
                    config = {**config, 'full_data_transfer': transfer['full_data_transfer'],
                              'input_archive': transfer['input_archive'],
                              'required_drive_paths': transfer['required_drive_paths']}
                stats = self._bundle(generated, directory / 'source.zip')
                input_stats = {'input_bundle_path': '', 'input_bundle_url': '',
                               'input_bundle_sha256': '', 'input_bundle_bytes': 0}
                input_stats.update(self._input_bundle(generated, config, directory / 'inputs.zip'))
                if input_stats['input_bundle_path']:
                    input_stats['input_bundle_url'] = f'/api/projects/{project_id}/colab/inputs.zip'
                notebook = build_notebook(project_id, config, drive_root=drive_root,
                                          bundle_sha256=stats['bundle_sha256'],
                                          bundle_bytes=(directory / 'source.zip').read_bytes())
                notebook_path = directory / f'AutoResearch_{project_id}.ipynb'
                if notebook_path.exists() and _linked(notebook_path):
                    raise ValueError('Notebook export target must not be linked')
                notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            except (OSError, ValueError, zipfile.BadZipFile) as exc:
                job.update(status='blocked', prepared=False, next_attempt_at=None,
                           blocked_reason=str(exc), checkpoint_contract=None)
                self._write(project_id, job)
                return self.status(project_id)
            previous_status = job['status']
            previous_next = job.get('next_attempt_at')
            changed_source = previous_bundle_sha256 and previous_bundle_sha256 != stats['bundle_sha256']
            changed_drive_root = previous_drive_root and previous_drive_root != drive_root
            if not url and (changed_source or changed_drive_root):
                # An existing Drive notebook embeds its own source/settings;
                # opening its old URL would ignore this newly prepared export.
                job['notebook_url'] = ''
                job['browser_url'] = ''
            keep_cooldown = previous_status in {'waiting_for_gpu', 'waiting_for_connection', 'disconnected'} and previous_next and _parse(previous_next) > _utc()
            job.update(prepared=True, status=previous_status if keep_cooldown else 'queued',
                       next_attempt_at=previous_next if keep_cooldown else now, lease_until=None,
                       blocked_reason='', notebook_path=str(notebook_path), bundle_path=str(directory / 'source.zip'),
                       notebook_download_url=f'/api/projects/{project_id}/colab/notebook.ipynb',
                       bundle_url=f'/api/projects/{project_id}/colab/bundle.zip',
                       checkpoint_contract=config['checkpoint'], success_criteria=config.get('success_criteria'),
                       **stats, **input_stats)
            job.pop('completion', None)
            job['data_upload_files'] = transfer['data_upload_files']
            if changed_source or changed_drive_root or previous_status in TERMINAL_STATUSES:
                # A runtime belongs to its embedded source and Drive scope.
                # Reusing its auth/connection marker could attach an old smoke
                # notebook after the project was prepared for a full campaign.
                # Provider cooldowns remain independent of that source identity.
                for key in ('recovery', 'authorization_continuation', 'runtime_id', 'runtime_started_at', 'running_at', 'finished_at',
                            'last_connection_observation'):
                    job.pop(key, None)
                job['last_observation'] = None
                job['last_attempt_at'] = None
            self._write(project_id, job)
            self.db.add_event(project_id, 'colab', 'Colab notebook exported; GPU availability has not been checked.', details={'status': 'queued'})
            return self.status(project_id)

    def attempt(self, project_id: str) -> dict[str, Any]:
        with self._transaction(project_id):
            job = self.status(project_id)
            if not job['can_attempt']:
                raise ColabConflict('Colab job is not due for an attempt')
            job.update(status='attempting', attempt_count=job['attempt_count'] + 1,
                       last_attempt_at=_stamp(_utc()), lease_until=_stamp(_utc() + timedelta(minutes=10)),
                       next_attempt_at=None, blocked_reason='')
            self._write(project_id, job)
            return self.status(project_id)

    def bind_notebook(self, project_id: str, request: ColabNotebookBindingRequest) -> dict[str, Any]:
        """Persist the source-verified Drive copy without asserting compute state.

        The browser calls this only after exact prepared-cell and native-tab
        ownership checks. Binding never executes cells or replaces a notebook.
        """
        url = _colab_url(request.notebook_url)
        parsed = urlparse(url)
        if (parsed.hostname != 'colab.research.google.com' or parsed.port not in (None,443)
                or not re.fullmatch(r'/drive/[A-Za-z0-9_-]{1,200}',parsed.path)
                or parsed.fragment or parsed.query):
            raise ValueError('Notebook binding requires a public Colab Drive URL without connection credentials')
        with self._transaction(project_id):
            job = self._read(project_id)
            if not job.get('prepared') or job['status'] in TERMINAL_STATUSES:
                raise ColabConflict('Prepare an active Colab job before binding its notebook')
            if request.bundle_sha256 != job.get('bundle_sha256'):
                raise ColabConflict('The saved notebook source does not match the current prepared artifact')
            job.update(notebook_url=url,browser_url=url,notebook_bundle_sha256=request.bundle_sha256,
                       notebook_bound_at=_stamp(_utc()))
            self._write(project_id,job)
            self.db.add_event(project_id,'colab','Saved the verified Colab notebook URL for recovery.',
                              details={'bundle_sha256':request.bundle_sha256})
            # Reading status can recover a dead worker and change scheduling.
            # This operation deliberately changes only the notebook metadata.
            return dict(job)

    def observe(self, project_id: str, request: ColabObservationRequest) -> dict[str, Any]:
        serialized = json.dumps(request.details, ensure_ascii=False)
        if len(serialized) > 20_000:
            raise ValueError('Observation details exceed 20 KB')
        with self._transaction(project_id):
            job = self._read(project_id)
            if not job.get('prepared') or job['status'] in TERMINAL_STATUSES:
                raise ColabConflict('The Colab job is not active')
            previous_quota_job=dict(job)
            if request.status == 'completed':
                runtime_id = request.details.get('runtime_id')
                if request.details.get('returncode') != 0 or not runtime_id or runtime_id != job.get('runtime_id'):
                    raise ColabConflict('Completion requires a successful result from the currently observed runtime')
                from .colab_data import validate_completion_details

                try:
                    completion = validate_completion_details(request.details, job.get('success_criteria'))
                except ValueError as exc:
                    raise ColabConflict(str(exc)) from None
                job['completion'] = completion
            observed_at = _stamp(_utc())
            # A local browser/network failure must not replace a provider GPU
            # cooldown with a shorter connection retry, including after restart.
            provider_until = job.get('provider_retry_not_before')
            if (request.status in {'network', 'disconnected'} and job['status'] == 'waiting_for_gpu'
                    and provider_until and _parse(provider_until) > _utc()):
                job['last_connection_observation'] = {'status': request.status, 'details': request.details,
                                                       'observed_at': observed_at}
                self._write(project_id, job)
                return self.status(project_id)
            job['last_observation'] = {'status': request.status, 'details': request.details, 'observed_at': observed_at}
            browser_url = request.details.get('browser_url')
            if isinstance(browser_url, str) and browser_url:
                job['browser_url'] = _colab_url(browser_url)
                if '/drive/' in browser_url:
                    job['notebook_url'] = browser_url
            job['lease_until'] = None
            job['next_attempt_at'] = None
            job['blocked_reason'] = str(request.details.get('reason', ''))[:2000]
            statuses = {'available': 'gpu_available', 'unavailable': 'waiting_for_gpu',
                        'quota': 'waiting_for_gpu', 'auth': 'auth_required',
                        'network': 'waiting_for_connection', 'disconnected': 'disconnected',
                        'running': 'running', 'completed': 'completed', 'failed': 'failed'}
            job['status'] = statuses[request.status]
            if request.status in {'unavailable', 'quota', 'network', 'disconnected'}:
                minimum = request.retry_after_minutes
                if request.status=='quota':
                    quota=self._publish_profile_quota(project_id,minimum,previous_quota_job)
                    job.update(next_attempt_at=quota['retry_not_before'],provider_retry_not_before=quota['retry_not_before'],
                               shared_quota_retry_not_before=quota['retry_not_before'])
                else:
                    self._schedule(job, minimum=minimum)
                if request.status == 'unavailable':
                    job['provider_retry_not_before'] = job['next_attempt_at']
            if request.status in {'available', 'running'} and request.details.get('runtime_id'):
                previous_runtime = job.get('runtime_id')
                job['runtime_id'] = request.details['runtime_id']
                if previous_runtime != job['runtime_id'] or not job.get('runtime_started_at'):
                    value = request.details.get('started_at') or job.get('last_attempt_at')
                    if value:
                        try:
                            job['runtime_started_at'] = _stamp(_parse(value))
                        except (TypeError, ValueError):
                            job.pop('runtime_started_at', None)
            continuation = request.details.get('authorization_continuation')
            if request.status=='available' and self._clear_profile_quota_on_fresh_gpu(job,request.details):
                job.pop('provider_retry_not_before',None)
                job.pop('shared_quota_retry_not_before',None)
            if isinstance(continuation,dict):
                runtime_id,started_at = continuation.get('runtime_id'),continuation.get('started_at')
                if runtime_id != job.get('runtime_id') or request.details.get('runtime_id') != runtime_id:
                    raise ColabConflict('Authorization continuation must belong to the currently observed runtime')
                try:
                    _parse(started_at)
                except (TypeError,ValueError):
                    raise ColabConflict('Authorization continuation requires its original runtime start time') from None
                cell_id = continuation.get('inflight_cell_id')
                if cell_id is not None and (not isinstance(cell_id,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,200}',cell_id)):
                    raise ValueError('Invalid authorization continuation cell identifier')
                job['authorization_continuation'] = {'runtime_id':runtime_id,'started_at':started_at,'inflight_cell_id':cell_id}
            persisted_continuation = job.get('authorization_continuation') or {}
            if request.status in {'completed','failed','disconnected','unavailable','quota'} or (
                    persisted_continuation and persisted_continuation.get('runtime_id') != job.get('runtime_id')):
                job.pop('authorization_continuation',None)
            if request.status in {'available', 'running', 'disconnected', 'completed', 'failed', 'unavailable', 'quota'}:
                job.pop('recovery', None)
            if request.status == 'running':
                job['running_at'] = observed_at
            if request.status in {'completed', 'failed'}:
                job['finished_at'] = observed_at
            self._write(project_id, job)
            self.db.add_event(project_id, 'colab', f'Colab browser observation: {request.status}',
                              details={'status': job['status'], 'next_attempt_at': job.get('next_attempt_at')})
            return self.status(project_id)

    def resume(self, project_id: str) -> dict[str, Any]:
        with self._transaction(project_id):
            job = self._read(project_id)
            if job['status'] != 'auth_required':
                raise ColabConflict('Only an authentication pause can be resumed immediately; GPU retry cooldown remains in force')
            provider_until = job.get('provider_retry_not_before')
            cooling_down = bool(provider_until and _parse(provider_until) > _utc())
            job.update(status='waiting_for_gpu' if cooling_down else 'queued',
                       next_attempt_at=provider_until if cooling_down else _stamp(_utc()),
                       blocked_reason='Colab provider GPU cooldown remains in force.' if cooling_down else '', lease_until=None)
            self._write(project_id, job)
            return self.status(project_id)

    def cancel(self, project_id: str) -> dict[str, Any]:
        with self._transaction(project_id):
            job = self._read(project_id)
            job.update(status='cancelled', next_attempt_at=None, lease_until=None, autorun_enabled=False,
                       blocked_reason='Local retry queue stopped. Stop an active Colab runtime in its browser UI.')
            job.pop('recovery', None)
            job.pop('authorization_continuation',None)
            self._write(project_id, job)
            return self.status(project_id)

    def export_path(self, project_id: str, kind: str) -> Path:
        job = self.status(project_id)
        if not job.get('prepared'):
            raise ColabConflict('Prepare a Colab notebook with a checkpoint/resume contract first')
        names = {'notebook': f'AutoResearch_{project_id}.ipynb', 'bundle': 'source.zip', 'inputs': 'inputs.zip'}
        name = names[kind]
        if kind == 'inputs' and not job.get('input_bundle_path'):
            raise ColabConflict('Project has no approved Colab input archive')
        path = self._directory(project_id) / name
        if not path.is_file() or _linked(path):
            raise ValueError('Colab export file is missing or linked')
        return path
