"""Verify a local TidyVoiceX corpus and publish it to an existing private Dataset."""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
import sys

from .hf_dataset import (HUB, MAX_MANIFEST_BYTES, DatasetAuthError, DatasetConfigurationError,
                         DatasetIntegrityError, DatasetNetworkError, HubReader, atomic_write,
                         file_matches, read_token, relative_path, validate_hf_pointer, validate_repo_id,
                         validate_revision, validate_token)
from .services.colab_data import _safe_file, validate_corpus_manifest


def local_inventory(root):
    root = Path(root).absolute()
    manifest = _safe_file(root, 'data_manifest.json')
    if manifest.stat().st_size > MAX_MANIFEST_BYTES:
        raise DatasetConfigurationError('Local data_manifest.json exceeds its size limit')
    raw = manifest.read_bytes()
    records = validate_corpus_manifest(json.loads(raw), 'data_manifest.json')['files']
    records.append({'path': 'data_manifest.json', 'size_bytes': len(raw),
                    'sha256': hashlib.sha256(raw).hexdigest()})
    local = []
    for item in records:
        relative_path(item['path'])
        path = _safe_file(root, item['path'])
        if not file_matches(path, item):
            raise DatasetIntegrityError('Local dataset file size or SHA-256 mismatch: ' + item['path'])
        local.append((item, path))
    return local


def _hf_call(method, *args, **kwargs):
    try:
        return method(*args, **kwargs)
    except Exception as exc:
        status = getattr(getattr(exc, 'response', None), 'status_code', None)
        if status in {401, 403, 404}:
            raise DatasetAuthError('Use an existing private Dataset and a token with write access') from None
        if status in {409, 412}:
            raise DatasetConfigurationError('Dataset branch changed during upload. Rerun to resume.') from None
        raise DatasetNetworkError('Hugging Face API request failed. Rerun to resume from committed batches.') from None


def _private_revision(api, repo_id, revision='main'):
    info = _hf_call(api.repo_info, repo_id=repo_id, repo_type='dataset', revision=revision)
    if getattr(info, 'private', None) is not True:
        raise DatasetConfigurationError('Target Dataset must already be private. Upload aborted.')
    return validate_revision(getattr(info, 'sha', None))


def _missing(api, reader, repo_id, revision, batch):
    paths = [item['path'] for item, _ in batch]
    available = _hf_call(api.get_paths_info, repo_id=repo_id, repo_type='dataset',
                         revision=revision, paths=paths)
    remote = {entry.path: entry for entry in available}
    missing = []
    for item, path in batch:
        entry = remote.get(item['path'])
        if getattr(entry, 'size', None) == item['size_bytes']:
            lfs = getattr(entry, 'lfs', None)
            if lfs is None:
                # Ordinary Git files have a SHA-1 object ID, not a content SHA-256.
                url = reader._artifact_url(repo_id, revision, item['path'])
                try:
                    reader._read(url, limit=item['size_bytes'], expected_hash=item['sha256'],
                                 exact=True, capture=False)
                except DatasetIntegrityError:
                    pass
                else:
                    continue
            elif lfs.sha256 == item['sha256']:
                continue
        missing.append((item, path))
    return missing


def _batches(files):
    batch = []
    size = 0
    for item, path in files:
        if batch and (len(batch) == 16 or size + item['size_bytes'] > 1024**3):
            yield batch
            batch, size = [], 0
        batch.append((item, path))
        size += item['size_bytes']
    if batch:
        yield batch


def publish(root, repo_id, output, *, api, reader, operation, progress=print):
    repo_id = validate_repo_id(repo_id)
    output = Path(output)
    if output.name != 'hf_dataset_pointer.json':
        raise DatasetConfigurationError('Output path must name hf_dataset_pointer.json')
    # Check privacy before inspecting the local corpus, and again before every commit.
    revision = _private_revision(api, repo_id)
    files = local_inventory(root)
    progress(f'Verified {len(files)} local files.')
    for batch in _batches(files):
        revision = _private_revision(api, repo_id)
        missing = _missing(api, reader, repo_id, revision, batch)
        if not missing:
            progress(f'Verified {len(batch)} already committed files.')
            continue
        additions = []
        for item, path in missing:
            addition = operation(path_in_repo=item['path'], path_or_fileobj=path)
            if (addition.upload_info.size != item['size_bytes']
                    or addition.upload_info.sha256.hex() != item['sha256']):
                raise DatasetIntegrityError('Local dataset changed after verification: ' + item['path'])
            additions.append(addition)
        if _private_revision(api, repo_id) != revision:
            raise DatasetConfigurationError('Dataset branch changed during upload. Rerun to resume.')
        _hf_call(api.create_commit, repo_id=repo_id, repo_type='dataset', revision='main',
                 parent_commit=revision, operations=additions,
                 commit_message='Upload verified AutoResearch dataset files')
        progress(f'Committed {len(missing)} files.')
    revision = _private_revision(api, repo_id)
    for batch in _batches(files):
        if _missing(api, reader, repo_id, revision, batch):
            raise DatasetIntegrityError('Final Dataset revision does not match the verified local inventory')
    if _private_revision(api, repo_id, revision) != revision:
        raise DatasetConfigurationError('Hugging Face did not confirm the final immutable Dataset revision')
    pointer = validate_hf_pointer(dict(repo_id=repo_id, revision=revision, manifest_path='data_manifest.json',
                                      manifest_sha256=files[-1][0]['sha256']))
    raw = (json.dumps(pointer, indent=2) + '\n').encode()
    atomic_write(output, lambda handle: handle.write(raw))
    return pointer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--repo-id', required=True)
    parser.add_argument('--output', type=Path, default=Path('hf_dataset_pointer.json'))
    args = parser.parse_args(argv)
    try:
        os.environ.pop('HF_DEBUG', None)
        from huggingface_hub import CommitOperationAdd, HfApi
    except ImportError:
        print('Install huggingface_hub before running the uploader.', file=sys.stderr)
        return 1
    try:
        if 'HF_TOKEN' not in os.environ:
            if not sys.stdin.isatty():
                raise DatasetAuthError('Run interactively or supply HF_TOKEN using a secure environment')
            token = validate_token(getpass.getpass('Hugging Face Dataset write token: '))
        else:
            token = read_token()
        pointer = publish(args.data_root, args.repo_id, args.output, api=HfApi(endpoint=HUB, token=token),
                          reader=HubReader(token), operation=CommitOperationAdd)
    except (DatasetAuthError, DatasetConfigurationError, DatasetIntegrityError, DatasetNetworkError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError, KeyError, RuntimeError):
        print('Upload could not complete. Check the local dataset and rerun to resume.', file=sys.stderr)
        return 1
    print(json.dumps(pointer, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
