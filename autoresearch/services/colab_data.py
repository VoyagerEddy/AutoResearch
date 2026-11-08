"""Bounded, allowlisted full-corpus transfers through normal Colab uploads."""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import shutil
import tarfile
import tempfile
from typing import Any
import zipfile


APPROVED_DATA_ROOT = Path('C:/Users/lenovo/Documents/datasets/TidyVoiceX_ASV/autoresearch-full-data')
MAX_FILE_BYTES = 512 * 1024 * 1024
MAX_FILES = 2048
METADATA_PATHS = {'train/shard.list', 'train/utt2spk', 'train/utt2lang', 'dev/shard.list',
                  'dev/wav.scp', 'dev/utt2spk', 'dev/utt2lang'}
MANIFEST_KINDS = {'data_manifest.json': 'tidyvoicex_full_train_dev',
                  'acceptance-manifest.json': 'tidyvoicex_acceptance_train_dev'}


def relative_data_path(value):
    if not isinstance(value, str) or not value or len(value) > 240 or '\\' in value:
        raise ValueError('Invalid relative data path')
    parts = value.split('/')
    blocked = {'credentials', 'credentials.json', 'secrets', 'secrets.json', 'token', 'token.json',
               'tokens', 'tokens.json', 'client_secret.json', 'service-account.json'}
    for part in parts:
        if (not part or part in {'.', '..'} or part.startswith('.') or part.endswith((' ', '.'))
                or part.casefold() in blocked or any(ord(c) < 32 or c in ':<>"|?*' for c in part)
                or Path(part).suffix.casefold() in {'.pem', '.key', '.p12', '.pfx'}):
            raise ValueError('Unsafe relative data path')
    return '/'.join(parts)


def file_sha256(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(4_194_304), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def _no_links(path: Path, boundary: Path) -> None:
    for candidate in (path, *path.parents):
        if candidate == boundary.parent:
            break
        if candidate.exists() and (candidate.is_symlink() or bool(getattr(candidate, 'is_junction', lambda: False)())):
            raise ValueError('Full-data transfer paths must not contain links or junctions')


def _safe_file(root: Path, relative: str) -> Path:
    path = root.joinpath(*relative_data_path(relative).split('/'))
    _no_links(path, root)
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Full-data transfer file is missing or escapes its approved root')
    return path


def _read_object(path: Path, maximum: int = 2_000_000) -> dict[str, Any]:
    if path.stat().st_size > maximum:
        raise ValueError('Full-data metadata exceeds its size limit')
    result = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(result, dict):
        raise ValueError('Full-data metadata must be an object')
    return result


def prepare_full_data_transfer(generated: Path, declaration: Any, *, project_id: str) -> dict[str, Any]:
    if declaration is None:
        return {'data_upload_files': [], 'full_data_transfer': None}
    if not isinstance(declaration, dict) or set(declaration) != {'pointer'}:
        raise ValueError('full_data_transfer must name its explicit local pointer')
    generated = generated.resolve(strict=True)
    pointer_path = _safe_file(generated, declaration['pointer'])
    pointer = _read_object(pointer_path, 16_000)
    root = Path(pointer.get('data_root', ''))
    if (pointer.get('schema_version') != 1 or pointer.get('project_id') != project_id
            or not root.is_absolute() or root.absolute() != APPROVED_DATA_ROOT.absolute()):
        raise ValueError('Full-data pointer does not name this project and the approved corpus root')
    _no_links(root, root)
    if not root.is_dir():
        raise ValueError('Approved full-data corpus directory is missing')
    manifest_relative = relative_data_path(pointer.get('manifest'))
    if manifest_relative not in MANIFEST_KINDS:
        raise ValueError('Full-data pointer must pin an approved corpus or acceptance manifest')
    manifest_path = _safe_file(root, manifest_relative)
    manifest_sha = pointer.get('manifest_sha256')
    if not isinstance(manifest_sha, str) or not re.fullmatch('[0-9a-f]{64}', manifest_sha) or file_sha256(manifest_path) != manifest_sha:
        raise ValueError('Full-data manifest SHA256 does not match the approved pointer')
    manifest = _read_object(manifest_path)
    if manifest.get('schema_version') != 1 or manifest.get('kind') != MANIFEST_KINDS[manifest_relative] or manifest.get('ready') is not True:
        raise ValueError('Full train/dev data preparation is not complete')
    records = manifest.get('files')
    if not isinstance(records, list) or not records or len(records) > MAX_FILES:
        raise ValueError('Full-data manifest needs a bounded, explicit file inventory')
    prefix = 'acceptance/' if manifest_relative == 'acceptance-manifest.json' else ''
    public, private, metadata, seen = [], [], [('data_manifest.json', manifest_path)], set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError('Invalid full-data file record')
        relative = relative_data_path(record.get('path'))
        if relative.casefold() in seen:
            raise ValueError('Duplicate full-data file inventory path')
        seen.add(relative.casefold())
        if not relative.startswith(prefix) or (not prefix and relative.startswith('acceptance/')):
            raise ValueError('Corpus artifact belongs to another experiment scope')
        local = relative[len(prefix):]
        if re.fullmatch(r'train/shards/train-[0-9]+\.tar(?:\.gz)?', local):
            kind = 'train_shard'
        elif re.fullmatch(r'dev/shards/dev-[0-9]+\.tar', local):
            kind = 'dev_archive'
        elif re.fullmatch(r'augmentation/(?:musan|rirs)/data\.mdb\.part-[0-9]{6}', local):
            kind = 'augmentation_chunk'
        elif re.fullmatch(r'dev/trials\.kaldi\.part-[0-9]{6}', local):
            kind = 'trials_chunk'
        elif (local in METADATA_PATHS or (prefix and local == 'dev/trials.kaldi')
              or (local.startswith('source_metadata/') and Path(local).suffix == '.txt')):
            kind = 'metadata'
        else:
            raise ValueError('Unapproved artifact in full-data transfer manifest')
        if 'kind' in record and record['kind'] != kind:
            raise ValueError('Full-data transfer kind differs from its approved path')
        size, checksum = record.get('size_bytes'), record.get('sha256')
        if type(size) is not int or not 0 < size <= MAX_FILE_BYTES or not isinstance(checksum, str) or not re.fullmatch('[0-9a-f]{64}', checksum):
            raise ValueError('Invalid full-data transfer size or checksum')
        path = _safe_file(root, relative)
        if path.stat().st_size != size:
            raise ValueError('Full-data transfer file size changed after preparation')
        item = {'path': relative, 'size_bytes': size, 'sha256': checksum, 'kind': kind}
        if kind == 'metadata':
            if file_sha256(path) != checksum:
                raise ValueError('Full-data metadata checksum mismatch')
            metadata.append((relative, path))
        else:
            private.append({**item, 'local_path': str(path), 'approved_root': str(root)})
            if kind == 'dev_archive':
                for key in ('audio_bytes', 'utterances'):
                    value = record.get(key)
                    if type(value) is not int or value <= 0:
                        raise ValueError('Dev archives require exact audio-byte and utterance bounds')
                    item[key] = value
        public.append(item)
    if not {prefix + name for name in METADATA_PATHS}.issubset(seen) or not any(r['kind'] == 'train_shard' for r in public) or not any(r['kind'] == 'dev_archive' for r in public):
        raise ValueError('Full-data corpus is missing required train/dev artifacts')
    augmentation = validate_augmentation_inventory(manifest.get('augmentation'), public,
        require_full=manifest_relative == 'data_manifest.json', prefix=prefix)
    assemblies = validate_assemblies(manifest.get('assemblies', []), public, prefix=prefix)
    if sum(path.stat().st_size for _, path in metadata) > 250_000_000:
        raise ValueError('Full-data metadata ZIP would exceed the 250 MB input limit')
    archive_name = 'full-data-metadata.zip' if manifest_relative == 'data_manifest.json' else 'acceptance-data-metadata.zip'
    archive = generated / 'inputs' / archive_name
    archive.parent.mkdir(parents=True, exist_ok=True)
    _no_links(archive, generated)
    descriptor, temporary = tempfile.mkstemp(prefix='.full-metadata-', suffix='.zip', dir=archive.parent)
    os.close(descriptor)
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
            for relative, path in sorted(metadata):
                info = zipfile.ZipInfo('full-data/' + relative, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                bundle.writestr(info, path.read_bytes())
        os.replace(temporary, archive)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return {
        'data_upload_files': private,
        'full_data_transfer': {'schema_version': 1, 'mode': 'browser_upload', 'files': public,
                               'manifest_path': 'data_manifest.json', 'manifest_sha256': manifest_sha,
                               'metadata_directory': 'full-data', 'max_file_bytes': MAX_FILE_BYTES,
                               'augmentation': augmentation, 'assemblies': assemblies},
        'input_archive': {'path': archive.relative_to(generated).as_posix(),
                          'sha256': file_sha256(archive),
                          'members': ['full-data/' + relative for relative, _ in sorted(metadata)]},
        'required_drive_paths': ['inputs/full-data/' + relative for relative, _ in sorted(metadata)],
    }


def validate_augmentation_inventory(augmentation, records, *, require_full, prefix=''):
    if not isinstance(augmentation, dict) or set(augmentation) != {'musan', 'rirs'}:
        raise ValueError('Corpus must declare both MUSAN and RIRS augmentation inventories')
    chunks = {item['path']: item for item in records if item['kind'] == 'augmentation_chunk'}
    used, result = set(), {}
    for name in ('musan', 'rirs'):
        entry = augmentation[name]
        if not isinstance(entry, dict) or entry.get('path') != f'{prefix}augmentation/{name}/lmdb/data.mdb':
            raise ValueError('Invalid augmentation LMDB destination')
        if entry.get('kind') not in ({'full'} if require_full else {'full', 'acceptance', 'acceptance_subset', 'smoke'}):
            raise ValueError('Formal campaign requires full augmentation databases')
        size, checksum, parts = entry.get('size_bytes'), entry.get('sha256'), entry.get('parts')
        if (type(size) is not int or not 0 < size <= 64 * 1024**3
                or not isinstance(checksum, str) or not re.fullmatch('[0-9a-f]{64}', checksum)
                or not isinstance(parts, list) or not parts or len(parts) > 256):
            raise ValueError('Invalid augmentation size, checksum or chunk inventory')
        expected_parts = [f'{prefix}augmentation/{name}/data.mdb.part-{index:06d}' for index in range(len(parts))]
        if parts != expected_parts or any(part not in chunks or part in used for part in parts):
            raise ValueError('Augmentation chunks must be complete, ordered and unique')
        if sum(chunks[part]['size_bytes'] for part in parts) != size:
            raise ValueError('Augmentation chunk sizes do not equal the declared LMDB size')
        used.update(parts)
        result[name] = {key: entry[key] for key in ('path', 'kind', 'size_bytes', 'sha256', 'parts')}
    if used != set(chunks):
        raise ValueError('Unclaimed augmentation chunks in corpus inventory')
    return result


def validate_assemblies(assemblies, records, *, prefix=''):
    if not isinstance(assemblies, list) or len(assemblies) > 1:
        raise ValueError('Invalid official trials assembly inventory')
    chunks = {item['path']: item for item in records if item['kind'] == 'trials_chunk'}
    direct = any(item['path'] == prefix + 'dev/trials.kaldi' and item['kind'] == 'metadata' for item in records)
    if direct and prefix and not assemblies and not chunks:
        return []
    if len(assemblies) != 1:
        raise ValueError('Corpus must declare its complete official trials assembly')
    entry = assemblies[0]
    if not isinstance(entry, dict) or entry.get('kind') != 'official_dev_trials' or entry.get('path') != prefix + 'dev/trials.kaldi':
        raise ValueError('Invalid official trials assembly destination')
    size, checksum, parts = entry.get('size_bytes'), entry.get('sha256'), entry.get('parts')
    if (type(size) is not int or not 0 < size <= 2 * 1024**3
            or not isinstance(checksum, str) or not re.fullmatch('[0-9a-f]{64}', checksum)
            or not isinstance(parts, list) or not 1 <= len(parts) <= 16):
        raise ValueError('Invalid official trials assembly size or checksum')
    expected = [f'{prefix}dev/trials.kaldi.part-{index:06d}' for index in range(len(parts))]
    if parts != expected or set(parts) != set(chunks) or sum(chunks[part]['size_bytes'] for part in parts) != size:
        raise ValueError('Official trials chunks must be complete and ordered')
    return [{key: entry[key] for key in ('kind', 'path', 'size_bytes', 'sha256', 'parts')}]


def reconstruct_artifacts(data_root, entries):
    """Join individually verified chunks without loading an entire artifact."""
    root = Path(data_root).resolve()
    for entry in entries:
        relative = relative_data_path(entry['path'])
        if not re.fullmatch(r'(?:acceptance/)?(?:augmentation/(?:musan|rirs)/lmdb/data\.mdb|dev/trials\.kaldi)', relative):
            raise ValueError('Invalid reconstructed artifact destination')
        target = root / relative_data_path(entry['path'])
        for path in (target, *target.parents):
            if path == root.parent:
                break
            if path.is_symlink() or bool(getattr(path, 'is_junction', lambda: False)()):
                raise ValueError('Artifact destination must not contain links')
        if not target.resolve().is_relative_to(root):
            raise ValueError('Artifact destination escapes corpus directory')
        if target.is_file() and target.stat().st_size == entry['size_bytes'] and file_sha256(target) == entry['sha256']:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        required_bytes = entry['size_bytes'] + 1024**3
        if shutil.disk_usage(root).free < required_bytes:
            raise ValueError('Insufficient ephemeral disk space to reconstruct the artifact')
        descriptor, temporary = tempfile.mkstemp(prefix='.lmdb-', suffix='.tmp', dir=target.parent)
        total, checksum = 0, hashlib.sha256()
        try:
            with os.fdopen(descriptor, 'wb') as output:
                for relative in entry['parts']:
                    path = root / relative_data_path(relative)
                    if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
                        raise ValueError('Artifact chunk is missing or unsafe')
                    with path.open('rb') as source:
                        for block in iter(lambda: source.read(4_194_304), b''):
                            total += len(block)
                            if total > entry['size_bytes']:
                                raise ValueError('Artifact exceeds its declared size')
                            checksum.update(block)
                            output.write(block)
                output.flush()
                os.fsync(output.fileno())
            if total != entry['size_bytes'] or checksum.hexdigest() != entry['sha256']:
                raise ValueError('Reconstructed artifact size or checksum mismatch')
            os.replace(temporary, target)
            # Whole-artifact identity supersedes the transport chunks.
            # Releasing them keeps corpus transfers within a free runtime's disk.
            for relative in entry['parts']:
                (root / relative_data_path(relative)).unlink()
        finally:
            Path(temporary).unlink(missing_ok=True)
    return [str(root / entry['path']) for entry in entries]


def estimate_ephemeral_disk_budget(data_root, transfer, *, free_bytes=None):
    """Bound staging space using only the prepared inventory and extraction receipts.

    Existing files receive size credit only; the normal transfer subsequently
    verifies every SHA256. Chunks are released one artifact at a time, so their
    peak is compared with the final corpus rather than added to every artifact.
    """
    root = Path(data_root).resolve()

    def existing_bytes(relative, expected):
        path = root / relative_data_path(relative)
        if type(expected) is not int or expected <= 0:
            raise ValueError('Invalid disk-budget file size')
        for parent in (path, *path.parents):
            if parent == root.parent:
                break
            if parent.is_symlink() or bool(getattr(parent, 'is_junction', lambda: False)()):
                raise ValueError('Disk-budget paths must not contain links')
        if not path.resolve().is_relative_to(root):
            raise ValueError('Disk-budget file escapes the approved corpus directory')
        return expected if path.is_file() and path.stat().st_size == expected else 0

    final_bytes, retained_bytes = 0, 0
    records = transfer['files']
    for item in records:
        if item['kind'] not in {'train_shard', 'dev_archive', 'metadata'}:
            continue
        final_bytes += item['size_bytes']
        retained_bytes += existing_bytes(item['path'], item['size_bytes'])
        if item['kind'] != 'dev_archive':
            continue
        final_bytes += item['audio_bytes']
        prefix = 'acceptance/' if item['path'].startswith('acceptance/') else ''
        receipt_relative = prefix + 'dev/extraction-receipts/' + item['sha256'] + '.json'
        receipt = root / relative_data_path(receipt_relative)
        if not receipt.is_file() or receipt.is_symlink() or receipt.stat().st_size > 2_000_000:
            continue
        # Only the receipt named by this declared archive can supply WAV paths.
        existing_bytes(receipt_relative, receipt.stat().st_size)
        try:
            value = json.loads(receipt.read_text(encoding='utf-8'))
            if not isinstance(value, dict):
                continue
            waves = value.get('files')
            if (value.get('archive_sha256') != item['sha256'] or value.get('audio_bytes') != item['audio_bytes']
                    or not isinstance(waves, list) or len(waves) != item['utterances']):
                continue
            total, credit, seen = 0, 0, set()
            for wave in waves:
                name, size = relative_data_path(wave['path']), wave['size_bytes']
                if name in seen or not name.endswith('.wav') or type(size) is not int or size <= 0:
                    raise ValueError('Invalid dev extraction disk receipt')
                seen.add(name)
                total += size
                credit += existing_bytes(prefix + 'dev/audio/' + name, size)
            if total == item['audio_bytes']:
                retained_bytes += credit
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            pass
    entries = list(transfer['augmentation'].values()) + transfer.get('assemblies', [])
    assembly_peak, prior_assemblies = 0, 0
    by_path = {item['path']: item for item in records}
    for entry in sorted(entries, key=lambda value: value['path']):
        final_bytes += entry['size_bytes']
        present = existing_bytes(entry['path'], entry['size_bytes'])
        retained_bytes += present
        if present:
            continue
        present_parts = sum(existing_bytes(part, by_path[part]['size_bytes']) for part in entry['parts'])
        # Both all chunks and the temporary assembled output coexist briefly.
        assembly_peak = max(assembly_peak, prior_assemblies + 2 * entry['size_bytes'] - present_parts)
        prior_assemblies += entry['size_bytes']
    missing_final = max(0, final_bytes - retained_bytes)
    reserve_bytes = 10 * 1024**3
    required_bytes = max(missing_final, assembly_peak) + reserve_bytes
    available = shutil.disk_usage(root).free if free_bytes is None else free_bytes
    return {'final_bytes': final_bytes, 'existing_bytes': retained_bytes,
            'missing_final_bytes': missing_final, 'assembly_peak_bytes': assembly_peak,
            'reserve_bytes': reserve_bytes, 'required_bytes': required_bytes,
            'free_bytes': available, 'enough': available >= required_bytes}


def data_upload_path(job: dict[str, Any], request: dict[str, Any]) -> Path:
    if request.get('project_id') != job.get('project_id') or request.get('runtime_id') != job.get('runtime_id'):
        raise ValueError('Full-data upload request belongs to another project or runtime')
    relative = relative_data_path(request.get('path'))
    records = job.get('data_upload_files')
    matches = [item for item in records or [] if isinstance(item, dict) and item.get('path') == relative]
    if len(matches) != 1:
        raise ValueError('Full-data upload path is not in the prepared registry')
    item = matches[0]
    if (request.get('sha256') != item.get('sha256') or request.get('size_bytes') != item.get('size_bytes')
            or type(item.get('size_bytes')) is not int or not 0 < item['size_bytes'] <= MAX_FILE_BYTES):
        raise ValueError('Full-data upload request differs from the prepared registry')
    root = Path(item.get('approved_root', ''))
    if not root.is_absolute() or root.absolute() != APPROVED_DATA_ROOT.absolute():
        raise ValueError('Full-data registry root is not approved')
    path = _safe_file(root, relative)
    if str(path) != item.get('local_path') or path.stat().st_size != item['size_bytes'] or file_sha256(path) != item['sha256']:
        raise ValueError('Full-data upload file changed since preparation')
    return path


def safe_extract_dev_archive(archive, destination, *, expected_audio_bytes, expected_utterances):
    """Extract bounded WAV members, rejecting links/traversal/duplicate ownership."""
    archive, destination = Path(archive), Path(destination)
    for parent in (destination, *destination.parents):
        if parent.is_symlink() or bool(getattr(parent, 'is_junction', lambda: False)()):
            raise ValueError('Dev extraction destination must not contain links')
    if type(expected_audio_bytes) is not int or type(expected_utterances) is not int or expected_audio_bytes <= 0 or expected_utterances <= 0:
        raise ValueError('Invalid dev extraction bounds')
    if archive.stat().st_size > 512 * 1024 * 1024 or expected_audio_bytes > archive.stat().st_size:
        raise ValueError('Dev archive exceeds its extraction budget')
    destination.mkdir(parents=True, exist_ok=True)
    archive_id = file_sha256(archive)
    owners_path = destination.parent / 'dev-extraction-owners.json'
    if owners_path.is_symlink() or (owners_path.exists() and owners_path.stat().st_size > 20_000_000):
        raise ValueError('Unsafe dev extraction ownership registry')
    owners = json.loads(owners_path.read_text()) if owners_path.exists() else {}
    if not isinstance(owners, dict) or len(owners) > 1_000_000:
        raise ValueError('Invalid dev extraction ownership registry')
    stage = Path(tempfile.mkdtemp(prefix='.dev-extract-', dir=destination.parent))
    names, waves, total = set(), [], 0
    try:
        with tarfile.open(archive, 'r:') as bundle:
            for member in bundle:
                relative = relative_data_path(member.name)
                if relative.casefold() in names or not member.isfile() or not relative.endswith(('.wav', '.spk')):
                    raise ValueError('Unsafe or duplicate dev archive member')
                names.add(relative.casefold())
                if len(names) > expected_utterances * 2:
                    raise ValueError('Too many dev archive members')
                if relative.endswith('.spk'):
                    if member.size > 512:
                        raise ValueError('Dev speaker record is too large')
                    continue
                if owners.get(relative) not in (None, archive_id):
                    raise ValueError('Duplicate WAV across dev archives')
                total += member.size
                if member.size <= 0 or total > expected_audio_bytes or len(waves) >= expected_utterances:
                    raise ValueError('Dev audio exceeds its declared extraction budget')
                target = stage / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.extractfile(member) as source, target.open('wb') as output:
                    shutil.copyfileobj(source, output, length=4_194_304)
                waves.append(relative)
        if total != expected_audio_bytes or len(waves) != expected_utterances:
            raise ValueError('Dev archive does not match its declared audio counts')
        for relative in waves:
            target = destination / relative
            if not target.resolve().is_relative_to(destination.resolve()) or target.is_symlink():
                raise ValueError('Dev extraction target escaped its destination')
            for parent in target.parents:
                if parent == destination.parent:
                    break
                if parent.is_symlink() or bool(getattr(parent, 'is_junction', lambda: False)()):
                    raise ValueError('Dev extraction target contains a linked directory')
            if target.exists() and relative not in owners:
                raise ValueError('Dev extraction would overwrite an unowned file')
            owners[relative] = archive_id
        temporary = owners_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(owners, sort_keys=True), encoding='utf-8')
        os.replace(temporary, owners_path)
        for relative in waves:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage / relative, target)
        receipts = destination.parent / 'extraction-receipts'
        if receipts.is_symlink() or bool(getattr(receipts, 'is_junction', lambda: False)()):
            raise ValueError('Dev extraction receipt directory must not be linked')
        receipts.mkdir(exist_ok=True)
        receipt = receipts / (archive_id + '.json')
        if receipt.is_symlink() or bool(getattr(receipt, 'is_junction', lambda: False)()):
            raise ValueError('Dev extraction receipt must not be linked')
        receipt_temporary = receipt.with_suffix('.json.tmp')
        if receipt_temporary.is_symlink() or bool(getattr(receipt_temporary, 'is_junction', lambda: False)()):
            raise ValueError('Dev extraction temporary receipt must not be linked')
        receipt_temporary.write_text(json.dumps({'archive_sha256':archive_id, 'audio_bytes':total,
            'files':[{'path':name,'size_bytes':(destination/name).stat().st_size} for name in waves]}), encoding='utf-8')
        os.replace(receipt_temporary, receipt)
        return {'utterances': len(waves), 'audio_bytes': total}
    finally:
        if stage.resolve().is_relative_to(destination.parent.resolve()):
            shutil.rmtree(stage)


def validate_success_criteria(criteria):
    if criteria is None:
        return
    if not isinstance(criteria, dict):
        raise ValueError('Experiment success criteria must be an object')
    validation = criteria.get('validation', 'process_exit')
    if validation == 'process_exit':
        return
    if validation != 'scientific_metrics':
        raise ValueError('Unknown experiment success validation')
    relative_data_path(criteria.get('metrics_file'))
    if criteria.get('status_value', 'campaign_complete') not in {'campaign_complete', 'acceptance_passed'}:
        raise ValueError('Unknown scientific completion status')
    required = criteria.get('required_metrics')
    if (not isinstance(required, list) or not 1 <= len(required) <= 100
            or any(not isinstance(name, str) or not re.fullmatch('[A-Za-z][A-Za-z0-9_]{0,99}', name) for name in required)
            or len(set(required)) != len(required)):
        raise ValueError('Scientific success requires explicit unique metric names')


def validate_completion_details(details, criteria):
    validate_success_criteria(criteria)
    if not criteria or criteria.get('validation', 'process_exit') == 'process_exit':
        return {'completion_kind': 'process_exit', 'scientific_metrics_produced': False}
    expected = criteria.get('status_value', 'campaign_complete')
    full = expected == 'campaign_complete'
    if details.get('completion_kind') != expected or details.get('scientific_metrics_produced') is not full:
        raise ValueError('Completion does not match the prepared scientific experiment contract')
    if details.get('metrics_file') != criteria['metrics_file']:
        raise ValueError('Completion metrics file differs from the prepared scientific contract')
    metrics = details.get('metrics')
    if not isinstance(metrics, dict):
        raise ValueError('Scientific completion requires measured metrics')
    measured = {}
    for name in criteria['required_metrics']:
        number = metrics.get(name)
        if not isinstance(number, (int, float)) or isinstance(number, bool) or not math.isfinite(number):
            raise ValueError('Scientific result is missing a finite required metric')
        if name.endswith('_eer_percent') and not name.startswith('delta_') and not 0 <= number <= 100:
            raise ValueError('EER metric is outside its valid range')
        if name.endswith('_min_dcf') and not name.startswith('delta_') and not 0 <= number <= 1:
            raise ValueError('minDCF metric is outside its valid range')
        measured[name] = number
    return {'completion_kind': expected, 'scientific_metrics_produced': full,
            'metrics_file': criteria['metrics_file'], 'metrics': measured}


def validate_scientific_result(results_dir, criteria):
    validate_success_criteria(criteria)
    if not criteria or criteria.get('validation', 'process_exit') == 'process_exit':
        return {'completion_kind': 'process_exit', 'scientific_metrics_produced': False}
    relative = relative_data_path(criteria.get('metrics_file'))
    base = Path(results_dir).resolve()
    path = base / relative
    for parent in (path, *path.parents):
        if parent == base.parent:
            break
        if parent.is_symlink() or bool(getattr(parent, 'is_junction', lambda: False)()):
            raise ValueError('Scientific results must not contain linked paths')
    if path.is_symlink() or not path.resolve().is_relative_to(base) or not path.is_file() or path.stat().st_size > 20_000_000:
        raise ValueError('Scientific result file is missing or unsafe')
    value = json.loads(path.read_text(encoding='utf-8'))
    expected = criteria.get('status_value', 'campaign_complete')
    if expected not in {'campaign_complete', 'acceptance_passed'} or value.get('status') != expected:
        raise ValueError('Scientific campaign has not reached its required completion status')
    full = expected == 'campaign_complete'
    if (value.get('campaign_complete') is not full or value.get('acceptance_complete') is not (not full)
            or value.get('scientific_metrics_produced') is not full):
        raise ValueError('Acceptance and full-campaign completion flags differ')
    return validate_completion_details({'completion_kind': expected, 'scientific_metrics_produced': full,
        'metrics_file': relative, 'metrics': value.get('metrics')}, criteria)


def notebook_helpers_source() -> str:
    imports = 'import hashlib, json, math, os, re, shutil, tarfile, tempfile\nfrom pathlib import Path\n'
    return imports + '\n\n'.join(inspect.getsource(function) for function in (
        relative_data_path, file_sha256, safe_extract_dev_archive, reconstruct_artifacts,
        estimate_ephemeral_disk_budget,
        validate_success_criteria, validate_completion_details, validate_scientific_result))
