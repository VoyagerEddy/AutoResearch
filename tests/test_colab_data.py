from __future__ import annotations

import ast
import copy
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
from types import SimpleNamespace
import zipfile

import pytest

from autoresearch.colab_notebook import build_notebook
from autoresearch.services import colab_data as data


def prepared_corpus(tmp_path, monkeypatch):
    root, generated = tmp_path / 'approved-corpus', tmp_path / 'generated'
    root.mkdir()
    (generated / 'inputs').mkdir(parents=True)
    monkeypatch.setattr(data, 'APPROVED_DATA_ROOT', root)
    inventory = []
    values = {name: b'official metadata\n' for name in data.METADATA_PATHS}
    values.update({'source_metadata/official-dev-trials.txt': b'a b target\n',
                   'train/shards/train-000000.tar': b'train-shard', 'dev/shards/dev-000000.tar': b'dev-shard',
                   'dev/trials.kaldi.part-000000': b'a b target\n',
                   'augmentation/musan/data.mdb.part-000000': b'full-musan',
                   'augmentation/rirs/data.mdb.part-000000': b'full-rirs'})
    for name, content in values.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        kind = ('train_shard' if name.startswith('train/shards/') else 'dev_archive' if name.startswith('dev/shards/')
                else 'augmentation_chunk' if name.startswith('augmentation/')
                else 'trials_chunk' if name.startswith('dev/trials.kaldi.part-') else 'metadata')
        record = {'path': name, 'kind': kind, 'size_bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}
        if kind == 'dev_archive':
            record.update(audio_bytes=3, utterances=1)
        inventory.append(record)
    augmentation = {name: {'kind':'full', 'path':f'augmentation/{name}/lmdb/data.mdb',
                          'size_bytes':len(values[f'augmentation/{name}/data.mdb.part-000000']),
                          'sha256':hashlib.sha256(values[f'augmentation/{name}/data.mdb.part-000000']).hexdigest(),
                          'parts':[f'augmentation/{name}/data.mdb.part-000000']} for name in ('musan','rirs')}
    assemblies = [{'kind':'official_dev_trials', 'path':'dev/trials.kaldi',
                   'size_bytes':len(values['dev/trials.kaldi.part-000000']),
                   'sha256':hashlib.sha256(values['dev/trials.kaldi.part-000000']).hexdigest(),
                   'parts':['dev/trials.kaldi.part-000000']}]
    manifest = {'schema_version': 1, 'kind': 'tidyvoicex_full_train_dev', 'ready': True, 'files': inventory,
                'augmentation':augmentation,'assemblies':assemblies}
    (root / 'data_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    pointer = {'schema_version': 1, 'project_id': 'speaker', 'data_root': str(root),
               'manifest': 'data_manifest.json', 'manifest_sha256': data.file_sha256(root / 'data_manifest.json')}
    (generated / 'inputs/full-data-pointer.json').write_text(json.dumps(pointer), encoding='utf-8')
    return root, generated, pointer, manifest


def test_metadata_export_and_upload_registry_are_separate(tmp_path, monkeypatch):
    root, generated, _, _ = prepared_corpus(tmp_path, monkeypatch)
    result = data.prepare_full_data_transfer(generated, {'pointer': 'inputs/full-data-pointer.json'}, project_id='speaker')
    assert len(result['data_upload_files']) == 5
    assert all(item['local_path'].startswith(str(root)) for item in result['data_upload_files'])
    assert str(root) not in json.dumps(result['full_data_transfer'])
    with zipfile.ZipFile(generated / result['input_archive']['path']) as bundle:
        assert set(bundle.namelist()) == set(result['input_archive']['members'])
        assert 'full-data/data_manifest.json' in bundle.namelist()
        assert not any('/shards/' in name for name in bundle.namelist())
    record = result['data_upload_files'][0]
    request = {'project_id': 'speaker', 'runtime_id': 'r1', **{k:record[k] for k in ('path', 'size_bytes', 'sha256')}}
    job = {'project_id': 'speaker', 'runtime_id': 'r1', 'data_upload_files': result['data_upload_files']}
    assert data.data_upload_path(job, request) == Path(record['local_path'])
    Path(record['local_path']).write_bytes(b'x' * record['size_bytes'])
    with pytest.raises(ValueError, match='changed'):
        data.data_upload_path(job, request)


@pytest.mark.parametrize('suffix',['.tar.gz','.tar.zip','.tar.gz.exe'])
def test_train_upload_allows_only_declared_uncompressed_or_gzip_shards(tmp_path,monkeypatch,suffix):
    root,generated,pointer,manifest = prepared_corpus(tmp_path,monkeypatch)
    record = next(item for item in manifest['files'] if item['kind']=='train_shard')
    record['path']='train/shards/train-000000'+suffix
    content=gzip.compress(b'approved lossless training shard')
    (root/record['path']).write_bytes(content)
    record.update(size_bytes=len(content),sha256=hashlib.sha256(content).hexdigest())
    (root/'data_manifest.json').write_text(json.dumps(manifest))
    pointer['manifest_sha256']=data.file_sha256(root/'data_manifest.json')
    (generated/'inputs/full-data-pointer.json').write_text(json.dumps(pointer))
    if suffix!='.tar.gz':
        with pytest.raises(ValueError,match='Unapproved artifact'):
            data.prepare_full_data_transfer(generated,{'pointer':'inputs/full-data-pointer.json'},project_id='speaker')
    else:
        result=data.prepare_full_data_transfer(generated,{'pointer':'inputs/full-data-pointer.json'},project_id='speaker')
        gzip_record=next(item for item in result['data_upload_files'] if item['kind']=='train_shard')
        assert gzip_record['path'].endswith('.tar.gz') and gzip_record['size_bytes']==len(content)


@pytest.mark.parametrize('change', ['root', 'hash', 'not_ready', 'duplicate', 'secret', 'oversize'])
def test_unapproved_manifest_artifacts_are_refused(tmp_path, monkeypatch, change):
    root, generated, pointer, manifest = prepared_corpus(tmp_path, monkeypatch)
    if change == 'root':
        pointer['data_root'] = str(tmp_path / 'outside')
    elif change == 'hash':
        pointer['manifest_sha256'] = '0' * 64
    else:
        if change == 'not_ready':
            manifest['ready'] = False
        elif change == 'duplicate':
            manifest['files'].append(manifest['files'][0])
        elif change == 'secret':
            manifest['files'][0]['path'] = 'credentials.json'
        else:
            manifest['files'][0]['size_bytes'] = data.MAX_FILE_BYTES + 1
        (root / 'data_manifest.json').write_text(json.dumps(manifest))
        pointer['manifest_sha256'] = data.file_sha256(root / 'data_manifest.json')
    (generated / 'inputs/full-data-pointer.json').write_text(json.dumps(pointer))
    with pytest.raises(ValueError):
        data.prepare_full_data_transfer(generated, {'pointer': 'inputs/full-data-pointer.json'}, project_id='speaker')


def test_metadata_hash_and_symlink_escape_are_refused(tmp_path, monkeypatch):
    root, generated, _, _ = prepared_corpus(tmp_path, monkeypatch)
    (root / 'dev/wav.scp').write_bytes(b'changed metadata\n')
    with pytest.raises(ValueError, match='size changed|checksum'):
        data.prepare_full_data_transfer(generated, {'pointer': 'inputs/full-data-pointer.json'}, project_id='speaker')
    monkeypatch.setattr(data, '_no_links', lambda *args: (_ for _ in ()).throw(ValueError('linked path')))
    with pytest.raises(ValueError, match='linked'):
        data.prepare_full_data_transfer(generated, {'pointer': 'inputs/full-data-pointer.json'}, project_id='speaker')


@pytest.mark.parametrize('field,value', [('project_id','other'), ('runtime_id','old'), ('path','../outside'), ('sha256','0'*64), ('size_bytes',1)])
def test_upload_request_must_match_current_runtime_and_exact_record(tmp_path, monkeypatch, field, value):
    _, generated, _, _ = prepared_corpus(tmp_path, monkeypatch)
    result = data.prepare_full_data_transfer(generated, {'pointer': 'inputs/full-data-pointer.json'}, project_id='speaker')
    record = result['data_upload_files'][0]
    request = {'project_id': 'speaker', 'runtime_id': 'r1', **{k:record[k] for k in ('path', 'size_bytes', 'sha256')}}
    request[field] = value
    with pytest.raises(ValueError):
        data.data_upload_path({'project_id':'speaker','runtime_id':'r1','data_upload_files':result['data_upload_files']}, request)


def archive_file(path, members):
    with tarfile.open(path, 'w') as bundle:
        for name, content, kind in members:
            item = tarfile.TarInfo(name)
            if kind == 'file':
                item.size = len(content)
                bundle.addfile(item, io.BytesIO(content))
            else:
                item.type = tarfile.SYMTYPE if kind == 'symlink' else tarfile.LNKTYPE
                item.linkname = 'outside.wav'
                bundle.addfile(item)
    return path


def test_dev_extraction_is_bounded_replayable_and_rejects_cross_archive_duplicates(tmp_path):
    archive = archive_file(tmp_path / 'dev.tar', [('speaker/en/a.wav',b'PCM','file'),('speaker/en/a.spk',b'speaker','file')])
    destination = tmp_path / 'data/dev/audio'
    assert data.safe_extract_dev_archive(archive, destination, expected_audio_bytes=3, expected_utterances=1) == {'utterances':1,'audio_bytes':3}
    assert (destination / 'speaker/en/a.wav').read_bytes() == b'PCM'
    assert not (destination / 'speaker/en/a.spk').exists()
    data.safe_extract_dev_archive(archive, destination, expected_audio_bytes=3, expected_utterances=1)
    other = archive_file(tmp_path / 'other.tar', [('speaker/en/a.wav',b'NEW','file')])
    with pytest.raises(ValueError, match='Duplicate WAV'):
        data.safe_extract_dev_archive(other, destination, expected_audio_bytes=3, expected_utterances=1)


@pytest.mark.parametrize('members,budget,count', [
    ([('../outside.wav',b'PCM','file')],3,1),
    ([('/absolute.wav',b'PCM','file')],3,1),
    ([('a.wav',b'','symlink')],3,1),
    ([('a.wav',b'','hardlink')],3,1),
    ([('a.wav',b'PCM','file'),('a.wav',b'PCM','file')],6,2),
    ([('a.wav',b'PCM','file')],2,1),
    ([('a.wav',b'PCM','file')],3,2),
    ([('a.wav',b'PCM','file'),('a.spk',b'x'*513,'file')],3,1),
])
def test_dev_archive_rejects_traversal_links_duplicates_and_wrong_bounds(tmp_path,members,budget,count):
    archive = archive_file(tmp_path / 'dev.tar', members)
    with pytest.raises(ValueError):
        data.safe_extract_dev_archive(archive, tmp_path / 'dev/audio', expected_audio_bytes=budget, expected_utterances=count)
    assert not (tmp_path / 'outside.wav').exists()


def criteria(mode='campaign_complete'):
    return {'validation':'scientific_metrics','metrics_file':'full-campaign/metrics.json','status_value':mode,
            'required_metrics':['B0_eer_percent','VQ_eer_percent','B0_min_dcf','VQ_min_dcf']}


def metric_report(mode='campaign_complete'):
    full = mode == 'campaign_complete'
    return {'status':mode,'campaign_complete':full,'acceptance_complete':not full,'scientific_metrics_produced':full,
            'metrics':{'B0_eer_percent':4.0,'VQ_eer_percent':3.9,'B0_min_dcf':0.4,'VQ_min_dcf':0.39}}


def test_scientific_success_requires_metrics_and_does_not_promote_acceptance(tmp_path):
    target = tmp_path / 'full-campaign/metrics.json'
    target.parent.mkdir()
    with pytest.raises(ValueError, match='missing'):
        data.validate_scientific_result(tmp_path, criteria())
    target.write_text(json.dumps(metric_report('acceptance_passed')))
    with pytest.raises(ValueError, match='completion status'):
        data.validate_scientific_result(tmp_path, criteria())
    assert data.validate_scientific_result(tmp_path, criteria('acceptance_passed'))['scientific_metrics_produced'] is False
    target.write_text(json.dumps(metric_report()))
    assert data.validate_scientific_result(tmp_path, criteria())['completion_kind'] == 'campaign_complete'
    value = metric_report()
    value['metrics']['VQ_min_dcf'] = float('nan')
    target.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='finite'):
        data.validate_scientific_result(tmp_path, criteria())


def test_generated_notebook_is_standalone_and_zero_exit_cannot_fake_scientific_completion(tmp_path):
    config = {'run_command':'test runner','checkpoint':{'files':[],'resume_command':'test resume'},'success_criteria':criteria()}
    notebook = build_notebook('speaker', config, drive_root='AutoResearch', bundle_sha256='a'*64, bundle_bytes=b'zip')
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']))
    source = '\n'.join(''.join(cell['source']) for cell in notebook['cells'])
    assert 'AUTORESEARCH_COLAB_UPLOAD_DATA ' in source
    assert 'import autoresearch' not in source
    namespace = {}
    exec(data.notebook_helpers_source(), namespace)
    assert namespace['relative_data_path']('dev/audio/a.wav') == 'dev/audio/a.wav'
    (tmp_path/'checkpoints').mkdir()
    (tmp_path/'results').mkdir()
    reports = []
    class Process:
        stdout = []
        def wait(self):
            return 0
    namespace.update(DRIVE_PROJECT=tmp_path,config=config,RUNTIME_ID='r1',report=lambda status,**values:reports.append((status,values)),
                         subprocess=SimpleNamespace(Popen=lambda *args,**kwargs:Process(), PIPE=subprocess.PIPE,
                                                    STDOUT=subprocess.STDOUT, CalledProcessError=subprocess.CalledProcessError))
    with pytest.raises(ValueError, match='missing'):
        exec(''.join(notebook['cells'][-1]['source']), namespace)
    assert reports[-1][0] == 'failed'
    assert not any(status == 'completed' for status,_ in reports)


def test_augmentation_and_large_trials_are_chunked_and_reconstructed_without_metadata_bloat(tmp_path, monkeypatch):
    root, generated, _, manifest = prepared_corpus(tmp_path, monkeypatch)
    result = data.prepare_full_data_transfer(generated, {'pointer':'inputs/full-data-pointer.json'}, project_id='speaker')
    with zipfile.ZipFile(generated / result['input_archive']['path']) as bundle:
        assert not any('.part-' in name for name in bundle.namelist())
    entries = list(manifest['augmentation'].values()) + manifest['assemblies']
    expected = {entry['path']: b''.join((root/part).read_bytes() for part in entry['parts']) for entry in entries}
    data.reconstruct_artifacts(root, entries)
    for entry in entries:
        assert (root/entry['path']).read_bytes() == expected[entry['path']]
        assert all(not (root/part).exists() for part in entry['parts'])
    # A resumed runtime validates each whole artifact and needs no transport chunks.
    data.reconstruct_artifacts(root, entries)


def test_corrupt_artifact_does_not_replace_valid_output_or_remove_transport_chunks(tmp_path):
    root = tmp_path / 'corpus'
    part = root/'augmentation/musan/data.mdb.part-000000'
    part.parent.mkdir(parents=True)
    part.write_bytes(b'corrupt')
    entry = {'path':'augmentation/musan/lmdb/data.mdb','size_bytes':7,'sha256':'0'*64,
             'parts':['augmentation/musan/data.mdb.part-000000']}
    with pytest.raises(ValueError, match='checksum mismatch'):
        data.reconstruct_artifacts(root, [entry])
    assert part.read_bytes() == b'corrupt'
    assert not (root/entry['path']).exists()


def test_acceptance_manifest_is_scoped_and_full_cannot_use_tiny_augmentation(tmp_path, monkeypatch):
    root, generated, pointer, manifest = prepared_corpus(tmp_path, monkeypatch)
    for entry in manifest['augmentation'].values():
        entry['kind'] = 'acceptance_subset'
    (root/'data_manifest.json').write_text(json.dumps(manifest))
    pointer['manifest_sha256'] = data.file_sha256(root/'data_manifest.json')
    (generated/'inputs/full-data-pointer.json').write_text(json.dumps(pointer))
    with pytest.raises(ValueError, match='full augmentation'):
        data.prepare_full_data_transfer(generated, {'pointer':'inputs/full-data-pointer.json'}, project_id='speaker')
    manifest['kind'] = 'tidyvoicex_acceptance_train_dev'
    for record in manifest['files']:
        before = root/record['path']
        record['path'] = 'acceptance/' + record['path']
        after = root/record['path']
        after.parent.mkdir(parents=True,exist_ok=True)
        after.write_bytes(before.read_bytes())
    for entry in [*manifest['augmentation'].values(), *manifest['assemblies']]:
        entry['path'] = 'acceptance/' + entry['path']
        entry['parts'] = ['acceptance/' + part for part in entry['parts']]
    (root/'acceptance-manifest.json').write_text(json.dumps(manifest))
    pointer.update(manifest='acceptance-manifest.json',manifest_sha256=data.file_sha256(root/'acceptance-manifest.json'))
    (generated/'inputs/full-data-pointer.json').write_text(json.dumps(pointer))
    result = data.prepare_full_data_transfer(generated, {'pointer':'inputs/full-data-pointer.json'}, project_id='speaker')
    assert result['input_archive']['path'] == 'inputs/acceptance-data-metadata.zip'
    assert all(item['path'].startswith('acceptance/') for item in result['full_data_transfer']['files'])


def test_disk_budget_counts_final_corpus_and_individual_assembly_peak_not_all_transport_chunks(tmp_path, monkeypatch):
    _, generated, _, _ = prepared_corpus(tmp_path, monkeypatch)
    transfer = data.prepare_full_data_transfer(generated, {'pointer':'inputs/full-data-pointer.json'}, project_id='speaker')['full_data_transfer']
    runtime = tmp_path/'runtime'
    runtime.mkdir()
    files = transfer['files']
    final = (sum(item['size_bytes'] for item in files if item['kind'] in {'train_shard','dev_archive','metadata'})
             + sum(item['audio_bytes'] for item in files if item['kind']=='dev_archive')
             + sum(item['size_bytes'] for item in transfer['augmentation'].values())
             + sum(item['size_bytes'] for item in transfer['assemblies']))
    result = data.estimate_ephemeral_disk_budget(runtime, transfer, free_bytes=10**15)
    assert result['final_bytes'] == final
    assert result['existing_bytes'] == 0
    assert result['required_bytes'] == max(final,result['assembly_peak_bytes']) + 10*1024**3
    assert data.estimate_ephemeral_disk_budget(runtime,transfer,free_bytes=result['required_bytes'])['enough'] is True
    assert data.estimate_ephemeral_disk_budget(runtime,transfer,free_bytes=result['required_bytes']-1)['enough'] is False
    train = next(item for item in files if item['kind']=='train_shard')
    target = runtime/train['path']
    target.parent.mkdir(parents=True)
    target.write_bytes(b'x'*train['size_bytes'])
    reused = data.estimate_ephemeral_disk_budget(runtime,transfer,free_bytes=10**15)
    assert reused['existing_bytes'] == train['size_bytes']
    # Size credit is an estimate only; transfer still detects this wrong checksum.
    assert data.file_sha256(target) != train['sha256']


def test_disk_budget_credits_only_receipted_dev_audio_and_size_matched_files(tmp_path):
    archive = tmp_path/'dev/shards/dev-000000.tar'
    archive.parent.mkdir(parents=True)
    archive_file(archive,[('speaker/a.wav',b'PCM','file')])
    item = {'path':'dev/shards/dev-000000.tar','kind':'dev_archive','size_bytes':archive.stat().st_size,
            'sha256':data.file_sha256(archive),'audio_bytes':3,'utterances':1}
    transfer = {'files':[item],'augmentation':{},'assemblies':[]}
    before = data.estimate_ephemeral_disk_budget(tmp_path,transfer,free_bytes=10**15)
    assert before['existing_bytes'] == archive.stat().st_size
    data.safe_extract_dev_archive(archive,tmp_path/'dev/audio',expected_audio_bytes=3,expected_utterances=1)
    after = data.estimate_ephemeral_disk_budget(tmp_path,transfer,free_bytes=10**15)
    assert after['existing_bytes'] == before['existing_bytes'] + 3
    (tmp_path/'dev/audio/speaker/a.wav').write_bytes(b'bad-size')
    assert data.estimate_ephemeral_disk_budget(tmp_path,transfer,free_bytes=10**15)['existing_bytes'] == before['existing_bytes']


def test_notebook_insufficient_disk_fails_before_any_bulk_upload(tmp_path, monkeypatch):
    _, generated, _, _ = prepared_corpus(tmp_path, monkeypatch)
    transfer_result = data.prepare_full_data_transfer(generated, {'pointer':'inputs/full-data-pointer.json'}, project_id='speaker')
    config = {'full_data_transfer':transfer_result['full_data_transfer']}
    notebook = build_notebook('speaker',config,drive_root='AutoResearch',bundle_sha256='a'*64,bundle_bytes=b'zip')
    drive = tmp_path/'drive'
    (drive/'inputs').mkdir(parents=True)
    with zipfile.ZipFile(generated/transfer_result['input_archive']['path']) as bundle:
        bundle.extractall(drive/'inputs')
    namespace, reports, uploads = {}, [], []
    exec(data.notebook_helpers_source(),namespace)
    def report(status,**values):
        namespace['_last_report_status'] = status
        reports.append((status,values))
    namespace.update(config=config,DRIVE_PROJECT=drive,pathlib=SimpleNamespace(Path=lambda value:tmp_path/'runtime'),
                     report=report,
                     estimate_ephemeral_disk_budget=lambda *args:{'required_bytes':20*1024**3,'free_bytes':5*1024**3,'enough':False},
                     files=SimpleNamespace(upload=lambda:uploads.append(True)))
    with pytest.raises(RuntimeError,match='Insufficient Colab ephemeral disk'):
        exec(''.join(notebook['cells'][-2]['source']),namespace)
    assert uploads == []
    assert reports == [('failed',{'reason':'Insufficient Colab ephemeral disk space for the declared full corpus; stop before bulk upload.',
                                 'disk_budget':{'required_bytes':20*1024**3,'free_bytes':5*1024**3,'enough':False}})]
