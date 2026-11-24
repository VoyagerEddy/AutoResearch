"""Entry point shipped with a prepared source archive to an approved backup GPU."""
from __future__ import annotations

import importlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

from .colab_data import (copy_verified_shard, estimate_ephemeral_disk_budget, file_sha256,
                         notebook_helpers_source, preflight_cuda, preflight_torchaudio,
                         reconstruct_artifacts, relative_data_path, safe_extract_dev_archive,
                         shard_matches, validate_scientific_result, validate_screening_stop)


def contained_path(root, relative):
    path = root / relative_data_path(relative)
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError('An experiment input escapes its authorized directory')
    return path


def verify_backup_checkpoint(root, requirements, expected, bundle_sha256, data_sha256, require_resume):
    paths = {}
    for name in requirements['files']:
        for path in root.rglob(relative_data_path(name)):
            if path.is_file() and path.stat().st_size:
                relative = path.relative_to(root).as_posix()
                paths[relative] = contained_path(root, relative)
    if not paths and not require_resume:
        return False
    receipt = contained_path(root, 'autoresearch-checkpoint-receipt.json')
    if not expected or not receipt.is_file() or receipt.stat().st_size > 2_000_000:
        raise ValueError('The latest Colab checkpoint receipt is not accessible on this GPU host')
    if file_sha256(receipt) != expected['sha256']:
        raise ValueError('The mirrored checkpoint receipt is older than the latest Colab receipt')
    recorded = json.loads(receipt.read_text(encoding='utf-8'))
    files = recorded['files']
    if (recorded.get('schema_version') != 1 or recorded.get('bundle_sha256') != bundle_sha256
            or recorded.get('data_manifest_sha256') != data_sha256
            or len(files) != expected['file_count'] or {item['path'] for item in files} != set(paths)):
        raise ValueError('Checkpoint files do not match the approved source and data')
    for item in files:
        if not shard_matches(paths[item['path']], item):
            raise ValueError('A mirrored checkpoint failed its SHA-256 check')
    return True


def verify_backup_cuda(requirements):
    torch = importlib.import_module('torch')
    if not torch.cuda.is_available():
        raise RuntimeError('Backup host has no usable CUDA GPU')
    if requirements.get('torch_version') and str(torch.__version__) != requirements['torch_version']:
        raise RuntimeError('Backup GPU host does not have the declared PyTorch version')
    preflight_cuda(torch)
    if requirements.get('match_torchaudio'):
        audio = importlib.import_module('torchaudio')
        if (requirements.get('torchaudio_version') not in (None, str(audio.__version__))
                or str(audio.__version__).split('+')[0] != str(torch.__version__).split('+')[0]):
            raise RuntimeError('Backup host does not have the declared TorchAudio version')
        preflight_torchaudio(torch, audio)


def verify_backup_data(root, transfer):
    manifest = contained_path(root, transfer['manifest_path'])
    if file_sha256(manifest) != transfer['manifest_sha256']:
        raise ValueError('Backup host has a different data manifest')
    assembled_parts = set()
    for item in list(transfer['augmentation'].values()) + transfer.get('assemblies', []):
        if not shard_matches(contained_path(root, item['path']), item):
            raise ValueError('An assembled artifact on the backup GPU host is missing or corrupt')
        assembled_parts.update(item['parts'])
    for item in transfer['files']:
        if item['path'] not in assembled_parts and not shard_matches(contained_path(root, item['path']), item):
            raise ValueError('A data shard on the backup GPU host is missing or corrupt')
    return manifest


def restore_backup_data(inputs, scratch, transfer):
    metadata = contained_path(inputs, transfer['metadata_directory'])
    cache = contained_path(inputs, 'verified-shards')
    manifest = contained_path(metadata, transfer['manifest_path'])
    if file_sha256(manifest) != transfer['manifest_sha256']:
        raise ValueError('Shared data manifest does not match the approved backup experiment')
    if not estimate_ephemeral_disk_budget(scratch, transfer)['enough']:
        raise ValueError('Backup host has insufficient scratch space for the verified data')
    artifact_entries = list(transfer['augmentation'].values()) + transfer.get('assemblies', [])
    part_owners = {part: entry for entry in artifact_entries for part in entry['parts']}
    records = sorted(transfer['files'], key=lambda item: (
        0, part_owners[item['path']]['path'], item['path']) if item['path'] in part_owners else (1, '', item['path']))
    sources = {item['path']: contained_path(metadata, item['path']) if item['kind'] == 'metadata'
               else contained_path(cache, item['sha256']) for item in records}
    for item in records:
        source = sources[item['path']]
        if not source.is_file() or source.stat().st_size != item['size_bytes']:
            raise ValueError('The authorized shared cache does not yet contain all required shards')
    for item in records:
        relative = item['path']
        source = sources[relative]
        target = contained_path(scratch, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        copy_verified_shard(source, target, item)
        owner = part_owners.get(relative)
        if owner and relative == owner['parts'][-1]:
            reconstruct_artifacts(scratch, [owner])
        if item['kind'] == 'dev_archive':
            audio = 'acceptance/dev/audio' if relative.startswith('acceptance/') else 'dev/audio'
            safe_extract_dev_archive(target, scratch / audio, expected_audio_bytes=item['audio_bytes'],
                                     expected_utterances=item['utterances'])
    target = contained_path(scratch, transfer['manifest_path'])
    shutil.copyfile(manifest, target)
    return verify_backup_data(scratch, transfer)


def backup_main(payload):
    private = Path(__file__).resolve().parent
    config = payload['config']
    target = payload['target']
    project = Path(target['state_dir']).resolve(strict=True)

    def report(status, **details):
        record = dict(run_id=payload['run_id'], bundle_sha256=payload['bundle_sha256'],
                      status=status, **details)
        temporary = private / '.receipt.tmp'
        with temporary.open('w', encoding='utf-8') as handle:
            json.dump(record, handle, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, private / 'receipt.json')

    executed = False
    scratch = None
    try:
        report('checking')
        if private != project / '.autoresearch-fallback' / payload['run_id']:
            raise ValueError('Backup worker directory does not match the approved project')
        if file_sha256(private / 'source.zip') != payload['bundle_sha256']:
            raise ValueError('Backup source archive does not match its approval')
        if config.get('hf_dataset_transfer'):
            raise ValueError('The prepared job uses private Dataset streaming in Colab; this backup cannot stage its data')
        verify_backup_cuda(config.get('runtime_requirements') or {})
        source = private / 'source'
        source.mkdir()
        with zipfile.ZipFile(private / 'source.zip') as archive:
            for member in archive.infolist():
                contained_path(source, member.filename)
            archive.extractall(source)
        checkpoints = contained_path(project, 'checkpoints')
        results = contained_path(project, 'results')
        inputs = contained_path(project, 'inputs')
        checkpoints.mkdir(exist_ok=True)
        results.mkdir(exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=project) as probe:
            probe.write(b'AutoResearch')
            probe.flush()
            os.fsync(probe.fileno())
        env = os.environ.copy()
        env.update(AUTORESEARCH_DRIVE_PROJECT=str(project), AUTORESEARCH_DATA_DIR=str(inputs),
                   AUTORESEARCH_CHECKPOINT_DIR=str(checkpoints), AUTORESEARCH_RESULTS_DIR=str(results),
                   PATH=str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', ''))
        env.pop('AUTORESEARCH_SESSION_DEADLINE_UTC', None)
        env.pop('AUTORESEARCH_SESSION_SECONDS', None)
        transfer = config.get('full_data_transfer')
        checkpoint = config['checkpoint']
        resumed = verify_backup_checkpoint(checkpoints, checkpoint, payload['checkpoint_receipt'],
            payload['bundle_sha256'], (transfer or {}).get('manifest_sha256'), payload['require_resume'])
        if transfer:
            if target['data_root']:
                data_root = Path(target['data_root']).resolve(strict=True)
                manifest = verify_backup_data(data_root, transfer)
            else:
                scratch = tempfile.TemporaryDirectory(prefix='autoresearch-backup-data-')
                data_root = Path(scratch.name).resolve()
                manifest = restore_backup_data(inputs, data_root, transfer)
            env.update(AUTORESEARCH_FULL_DATA_MANIFEST=str(manifest),
                AUTORESEARCH_AUGMENTATION_DIR=str((data_root / transfer['augmentation']['musan']['path']).parents[2]))
        for required in config.get('required_drive_paths', []):
            if transfer and required.startswith('inputs/' + transfer['metadata_directory'] + '/'):
                continue  # These paths were checked against the approved data root above.
            if not contained_path(project, required).exists():
                raise FileNotFoundError('A declared project input is absent from the backup host')
        preflight = config.get('preflight_command')
        if preflight:
            with tempfile.TemporaryDirectory(dir=private) as probe:
                probe_env = {**env, 'AUTORESEARCH_PREFLIGHT': '1',
                             'AUTORESEARCH_CHECKPOINT_DIR': str(Path(probe) / 'checkpoints'),
                             'AUTORESEARCH_RESULTS_DIR': str(Path(probe) / 'results')}
                subprocess.run(preflight, cwd=source, env=probe_env, shell=True, check=True,
                               timeout=config.get('preflight_timeout_seconds', 900))
        executed = True
        report('running', resumed=resumed)
        criteria = config.get('success_criteria') or {}
        metrics_path = contained_path(results, criteria['metrics_file']) if criteria.get('metrics_file') else None
        previous_metrics = metrics_path.stat().st_mtime_ns if metrics_path and metrics_path.is_file() else None
        command = checkpoint['resume_command'] if resumed else config['run_command']
        for step in filter(None, (config.get('prepare_command'), command)):
            subprocess.run(step, cwd=source, env=env, shell=True, check=True)
        if metrics_path and metrics_path.is_file() and metrics_path.stat().st_mtime_ns == previous_metrics:
            raise ValueError('Backup run did not produce a new scientific result file')
        screening = validate_screening_stop(results, criteria, (transfer or {}).get('manifest_sha256'))
        if screening:
            report('failed', resumed=resumed, failure_kind='screening_failed',
                   reason='RSI screening ended because no healthy VQ candidate passed the B0 comparison',
                   **screening)
            return 1
        validated = validate_scientific_result(results, criteria)
        report('completed', resumed=resumed, result=validated)
    except Exception as exc:
        report('failed' if executed else 'not_ready', error_type=type(exc).__name__,
               reason=str(exc)[:400])
        return 1
    finally:
        if scratch is not None:
            scratch.cleanup()
    return 0


def backup_worker_source(payload: dict) -> str:
    imports = 'import importlib, sys, subprocess, zipfile\n'
    functions = '\n'.join(inspect.getsource(function) for function in (
        contained_path, verify_backup_checkpoint, verify_backup_cuda, verify_backup_data,
        restore_backup_data, backup_main,
    ))
    return imports + notebook_helpers_source() + '\n' + functions + '\nraise SystemExit(backup_main(' + repr(payload) + '))\n'
