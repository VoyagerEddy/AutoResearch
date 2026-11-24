from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import sys
from threading import Event
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

import pytest

from autoresearch import hf_dataset as hf
from autoresearch.colab_notebook import build_notebook
from autoresearch.hf_dataset_upload import publish
from test_colab_data import prepared_corpus


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def pointer(raw=b'{}'):
    return {'repo_id': 'example/private-corpus', 'revision': 'a' * 40,
            'manifest_path': 'data_manifest.json', 'manifest_sha256': digest(raw)}


def record(relative, raw, kind):
    return {'path': relative, 'size_bytes': len(raw), 'sha256': digest(raw), 'kind': kind}


@pytest.mark.parametrize('changes', [
    {'revision': 'main'}, {'manifest_sha256': '0' * 40},
    {'manifest_path': '../data_manifest.json'}, {'repo_id': 'https://huggingface.co/example/corpus'},
    {'token': 'unexpected'}, {'repo_id': 'example/' + 'hf_' + 'a' * 24},
])
def test_pointer_requires_immutable_commit_and_safe_paths(changes):
    with pytest.raises(hf.DatasetConfigurationError):
        hf.validate_hf_pointer({**pointer(), **changes})


@pytest.mark.parametrize('value', [None, 'invalid', 'hf_' + 'x' * 24 + '\n'])
def test_missing_or_malformed_token_has_sanitized_auth_error(monkeypatch, value):
    monkeypatch.delenv('HF_TOKEN', raising=False)
    if value is not None:
        monkeypatch.setenv('HF_TOKEN', value)
    with pytest.raises(hf.DatasetAuthError, match='Configure a valid HF_TOKEN') as error:
        hf.read_token()
    if value:
        assert value not in str(error.value)


def test_http_auth_and_public_repository_are_rejected_without_raw_error():
    def forbidden(req, timeout):
        raise HTTPError(req.full_url, 403, 'untrusted raw error', {}, None)

    reader = hf.HubReader('test-fixture', opener=forbidden)
    with pytest.raises(hf.DatasetAuthError, match='inaccessible') as error:
        reader.manifest(pointer())
    assert 'untrusted raw error' not in str(error.value)
    reader = hf.HubReader('test-fixture', opener=lambda req, timeout: io.BytesIO(b'{"private":false}'))
    with pytest.raises(hf.DatasetConfigurationError, match='must be private'):
        reader.manifest(pointer())


def test_private_manifest_read_is_revision_bound_and_does_not_forward_token():
    raw = b'{"ready":true}'
    config = pointer(raw)
    observed = []

    def opener(req, timeout):
        observed.append(req.full_url)
        assert req.get_header('Authorization') == 'Bearer test-fixture'
        redirected = hf._HubRedirect().redirect_request(
            req, None, 302, 'redirect', {}, 'https://cdn.example.org/signed-object')
        assert redirected.get_header('Authorization') is None
        same_origin = hf._HubRedirect().redirect_request(
            req, None, 302, 'redirect', {}, 'https://huggingface.co/api/resolve-cache/artifact')
        assert same_origin.get_header('Authorization') == 'Bearer test-fixture'
        if '/api/' in req.full_url:
            return io.BytesIO(json.dumps(dict(private=True, sha=config['revision'])).encode())
        return io.BytesIO(raw)

    assert hf.HubReader('test-fixture', opener=opener).manifest(config) == raw
    assert len(observed) == 2
    assert all(config['revision'] in url for url in observed)
    with pytest.raises(hf.DatasetIntegrityError, match='SHA-256'):
        hf.HubReader('test-fixture', opener=opener).manifest(pointer())


def test_shard_http_retry_then_verification_and_finally_cleanup(tmp_path, monkeypatch):
    requests = []
    delays = []
    name = 'train/shards/train-000000.tar.gz'
    content = b'shard'

    def opener(req, timeout):
        requests.append(req.full_url)
        if len(requests) == 1:
            raise URLError('connection reset')
        return io.BytesIO(content)

    monkeypatch.setattr(hf.time, 'sleep', delays.append)
    reader = hf.HubReader('test-fixture', opener=opener)
    dataset = hf.Dataset(tmp_path, pointer(), [record(name, content, 'train_shard')], reader)
    with pytest.raises(RuntimeError, match='consumer interruption'):
        with dataset.shard(name) as path:
            assert path.read_bytes() == content
            raise RuntimeError('consumer interruption')
    assert delays == [1]
    assert requests == [requests[0]] * 2
    assert not path.exists()
    content = b'corpt'
    with pytest.raises(hf.DatasetIntegrityError, match='SHA-256'):
        with dataset.shard(name):
            pytest.fail('Unverified data reached a consumer')
    assert list((tmp_path / 'hf-shards').glob('*.tar')) == []


def test_window_enforces_file_and_byte_cap_across_readers(tmp_path):
    files = [f'train/shards/train-{n:06d}.tar.gz' for n in range(3)]
    ready = [Event() for _ in files]
    release = [Event() for _ in files]
    config = dict(pointer(), window_shards=2, window_bytes=8)
    records = [record(name, b'data', 'train_shard') for name in files]
    reader = hf.HubReader('test-fixture', opener=lambda req, timeout: io.BytesIO(b'data'))

    def consume(index):
        with hf.Dataset(tmp_path, config, records, reader).shard(files[index]) as path:
            ready[index].set()
            assert release[index].wait(5)
            assert path.read_bytes() == b'data'

    with ThreadPoolExecutor(max_workers=3) as pool:
        pending = [pool.submit(consume, i) for i in range(3)]
        try:
            assert ready[0].wait(3) or ready[1].wait(3)
            # Let the third worker reach the lease. At most two may proceed.
            for event in ready:
                event.wait(0.15)
            assert sum(event.is_set() for event in ready) == 2
            cached = list((tmp_path / 'hf-shards').glob('*.tar'))
            assert len(cached) == 2
            assert sum(path.stat().st_size for path in cached) == 8
        finally:
            for event in release:
                event.set()
        for future in pending:
            future.result()
    assert list((tmp_path / 'hf-shards').glob('*.tar')) == []


@pytest.mark.parametrize('online', [False, True])
def test_metadata_and_chunk_assembly_only_stage_declared_files(tmp_path, monkeypatch, online):
    from autoresearch.services.colab_data import validate_corpus_manifest

    root, _, _, manifest = prepared_corpus(tmp_path, monkeypatch)
    inventory = validate_corpus_manifest(manifest, 'data_manifest.json')
    raw = (root / 'data_manifest.json').read_bytes()
    downloads = []

    def opener(req, timeout):
        relative = req.full_url.split('/resolve/' + 'a' * 40 + '/')[1]
        downloads.append(relative)
        return io.BytesIO((root / relative).read_bytes())

    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    transfer = dict(inventory, online_augmentation=online)
    reader = hf.HubReader('test-fixture', opener=opener)
    hf.install_metadata(scratch, raw, transfer, pointer(raw), reader)
    assert (scratch / 'dev/trials.kaldi').read_bytes() == b'a b target\n'
    assert (scratch / 'data_manifest.json').read_bytes() == raw
    assert (scratch / 'augmentation/musan/lmdb/data.mdb').exists() is online
    assert not list(scratch.rglob('*.part-*'))
    assert not list(scratch.rglob('*.tar*'))
    assert all('/shards/' not in name for name in downloads)
    assert any(name.startswith('augmentation/') for name in downloads) is online
    downloads.clear()
    hf.install_metadata(scratch, raw, transfer, pointer(raw), reader)
    assert not downloads


def test_uploader_rejects_public_repo_before_any_file_access(tmp_path):
    api = SimpleNamespace(repo_info=lambda **kw: SimpleNamespace(private=False, sha='a' * 40))
    with pytest.raises(hf.DatasetConfigurationError, match='must already be private'):
        publish(tmp_path / 'missing', 'example/corpus', tmp_path / 'hf_dataset_pointer.json',
                api=api, reader=None, operation=None)


def test_uploader_resumes_and_binds_verified_revision(tmp_path, monkeypatch):
    root, _, _, _ = prepared_corpus(tmp_path, monkeypatch)
    output = tmp_path / 'hf_dataset_pointer.json'
    identities = {}
    revision = 'a' * 40
    commits = []

    def paths_info(paths, **kw):
        return [SimpleNamespace(path=name, size=len(identities[name]), lfs=SimpleNamespace(
            sha256=digest(identities[name]))) for name in paths if name in identities]

    def commit(operations, parent_commit, **kw):
        nonlocal revision
        assert parent_commit == revision
        for operation in operations:
            identities[operation.path] = operation.raw
        revision = 'b' * 40
        commits.append(operations)
        return SimpleNamespace(oid=revision)

    def operation(path_in_repo, path_or_fileobj):
        raw = Path(path_or_fileobj).read_bytes()
        return SimpleNamespace(path=path_in_repo, raw=raw,
                               upload_info=SimpleNamespace(size=len(raw), sha256=hashlib.sha256(raw).digest()))

    api = SimpleNamespace(repo_info=lambda **kw: SimpleNamespace(private=True, sha=revision),
                          get_paths_info=paths_info, create_commit=commit)
    reader = hf.HubReader('test-fixture')
    result = publish(root, 'example/corpus', output, api=api, reader=reader, operation=operation)
    assert result['revision'] == 'b' * 40
    assert result['manifest_sha256'] == digest(identities['data_manifest.json'])
    assert json.loads(output.read_text()) == result
    assert len(commits) == 1
    assert publish(root, 'example/corpus', output, api=api, reader=reader, operation=operation) == result
    assert len(commits) == 1
    (root / 'train/shards/train-000000.tar').write_bytes(b'corrupted!!')
    with pytest.raises(hf.DatasetIntegrityError, match='SHA-256 mismatch'):
        publish(root, 'example/corpus', output, api=api, reader=reader, operation=operation)
    assert len(commits) == 1


def test_notebook_hf_branch_is_preflight_gated_and_does_not_include_shard_upload():
    notebook = build_notebook('test', {'hf_dataset_transfer': pointer()}, drive_root='AutoResearch',
                              bundle_sha256='a' * 64, bundle_bytes=b'')
    cells = [''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code']
    for source in cells:
        compile(source, '<cell>', 'exec')
    source = '\n'.join(cells)
    assert source.index('AUTORESEARCH_PREFLIGHT_GATE') < source.index('hf_reader.manifest')
    assert 'AUTORESEARCH_COLAB_UPLOAD_DATA' not in source
    assert 'verified-shards' not in source
    first_cell = cells[0].split("report('starting'")[0]
    namespace = {}
    exec(first_cell, namespace)
    secret = 'hf_' + 'z' * 24
    assert namespace['safe_text']('failed ' + secret) == 'failed [redacted]'


@pytest.mark.parametrize('authorized', [False, True])
def test_notebook_hf_cell_stages_verified_inputs_in_ephemeral_root(tmp_path, monkeypatch, authorized):
    from autoresearch.services.colab_data import notebook_helpers_source

    local, _, _, _ = prepared_corpus(tmp_path, monkeypatch)
    raw = (local / 'data_manifest.json').read_bytes()
    config = {'hf_dataset_transfer': dict(pointer(raw), online_augmentation=False,
                                         window_shards=4, window_bytes=hf.MAX_WINDOW_BYTES)}
    notebook = build_notebook('speaker', config, drive_root='AutoResearch',
                              bundle_sha256='a' * 64, bundle_bytes=b'')
    cell = next(''.join(cell['source']) for cell in notebook['cells']
                if 'hf_manifest_bytes = hf_reader.manifest' in ''.join(cell['source']))
    source, scratch = tmp_path / 'source', tmp_path / 'scratch'
    source.mkdir()
    scratch.mkdir()
    reports, requested = [], []
    monkeypatch.setattr(sys, 'path', list(sys.path))
    monkeypatch.setitem(sys.modules, 'autoresearch_hf_dataset', hf)
    token = 'hf_' + 't' * 24
    monkeypatch.setenv('HF_TOKEN', token)
    for key in (hf.CONFIG_ENV, 'AUTORESEARCH_FULL_DATA_MANIFEST', 'AUTORESEARCH_AUGMENTATION_DIR'):
        monkeypatch.setenv(key, '')
    real_reader = hf.HubReader

    def opener(request, timeout):
        assert request.get_header('Authorization') == 'Bearer ' + token
        if not authorized:
            raise HTTPError(request.full_url, 401, token, {}, None)
        relative = request.full_url.rpartition('/resolve/' + 'a' * 40 + '/')[2]
        if '/api/' in request.full_url:
            return io.BytesIO(json.dumps(dict(private=True, sha='a' * 40)).encode())
        requested.append(relative)
        return io.BytesIO((local / relative).read_bytes())

    monkeypatch.setattr(hf, 'HubReader', lambda supplied: real_reader(supplied, opener=opener))
    namespace = {}
    exec(notebook_helpers_source(), namespace)
    namespace.update(config=config, SOURCE=source, RUNTIME_ID='test',
                     RUNTIME_PREFLIGHT=dict(status='passed', runtime_id='test', scratch={}),
                     pathlib=SimpleNamespace(Path=lambda _: scratch), preflight_path=tmp_path / 'preflight.json',
                     save_preflight_evidence=lambda _: None, safe_text=str,
                     report=lambda status, **details: reports.append((status, details)))
    namespace['shutil'] = SimpleNamespace(disk_usage=lambda _: SimpleNamespace(free=32 * 1024**3))
    if authorized:
        exec(cell, namespace)
        assert reports[-1][1]['phase'] == 'data_ready'
        assert json.loads(namespace['os'].environ[hf.CONFIG_ENV])['root'] == str(scratch)
        assert (scratch / 'dev/trials.kaldi').is_file()
        assert not any(name.startswith('augmentation/') or '/shards/' in name for name in requested)
    else:
        # The real report() sets this global before the cell wrapper runs.
        def report(status, **details):
            namespace['_last_report_status'] = status
            reports.append((status, details))

        namespace['report'] = report
        with pytest.raises(RuntimeError, match='Enable notebook access') as error:
            exec(cell, namespace)
        assert token not in str(error.value)
        assert reports[-1] == ('auth', {'phase': 'preparing_data', 'reason': 'hf_dataset_auth'})
        assert not (scratch / 'data_manifest.json').exists()
