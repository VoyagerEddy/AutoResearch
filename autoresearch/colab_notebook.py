"""Build a normal Colab notebook; allocation stays in Colab's interactive UI."""
from __future__ import annotations

import base64
import json
import textwrap
from typing import Any

from .services.colab_data import notebook_helpers_source


def build_notebook(project_id: str, config: dict[str, Any], *, drive_root: str,
                   bundle_sha256: str, bundle_bytes: bytes) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []

    def markdown(value: str) -> None:
        cells.append({"cell_type": "markdown", "metadata": {}, "source": value.splitlines(True)})

    def code(value: str) -> None:
        value = textwrap.dedent(value).strip() + "\n"
        if 'def report(' not in value:
            value = ('try:\n' + textwrap.indent(value, '    ') +
                     "except BaseException as notebook_error:\n"
                     "    if globals().get('_last_report_status') not in {'auth', 'failed', 'unavailable', 'disconnected', 'completed'}:\n"
                     "        report('failed', reason='Notebook cell failed: ' + str(notebook_error))\n"
                     "    raise\n")
        cells.append({"cell_type": "code", "metadata": {}, "source": value.splitlines(True),
                      "execution_count": None, "outputs": []})

    markdown(f"""# AutoResearch · {project_id} · Colab 免费 GPU

AUTORESEARCH_SOURCE_SHA256 {bundle_sha256}

在 Colab 的 **运行时 → 更改运行时类型** 中选择免费的 GPU，再运行全部单元格。
免费 GPU 不保证可用，会话时间和额度由 Colab 决定；本 notebook 不建立 SSH、后台保活或付费资源。
首次运行需要正常登录并授权挂载自己的 Google Drive。代码已安全嵌入此 notebook；
把实验数据和预训练权重放到 Drive 的 `{drive_root}/{project_id}/inputs/`。
checkpoint 写入 Drive 的 `checkpoints/`；重连后运行此 notebook 会执行项目声明的恢复命令。
中断仍可能丢失最近一次 checkpoint 之后的步骤。

规则：[Colab 官方 FAQ](https://research.google.com/colaboratory/faq.html)。
""")
    payload = {"project_id": project_id, "drive_root": drive_root, "config": config,
               "bundle_sha256": bundle_sha256}
    code(f"""
        import datetime, hashlib, json, os, pathlib, subprocess, time, uuid, zipfile
        SETTINGS = json.loads({json.dumps(json.dumps(payload, ensure_ascii=False))})
        PROJECT_ID = SETTINGS['project_id']
        RUNTIME_ID = str(uuid.uuid4())
        STARTED_AT = datetime.datetime.now(datetime.timezone.utc).isoformat()
        DRIVE_PROJECT = None
        def report(status, **details):
            global _last_report_status
            _last_report_status = status
            record = dict(project_id=PROJECT_ID, runtime_id=RUNTIME_ID,
                          started_at=STARTED_AT, status=status, **details)
            marker = 'AUTORESEARCH_COLAB_STATUS ' + json.dumps(record)
            release_safe = details.get('runtime_release_safe') is True
            if not release_safe:
                print(marker, flush=True)
            if DRIVE_PROJECT is not None:
                target = DRIVE_PROJECT / 'session_status.json'
                temp = target.with_suffix('.json.tmp')
                with open(temp, 'w', encoding='utf-8') as handle:
                    json.dump(record, handle, ensure_ascii=False, indent=2)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp, target)
            elif release_safe:
                raise RuntimeError('A release-safe report requires persistent Drive evidence')
            if release_safe:
                print(marker, flush=True)
        report('starting')
        try:
            gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                                 capture_output=True, text=True)
        except FileNotFoundError:
            gpu = subprocess.CompletedProcess(['nvidia-smi'], 1, '', 'nvidia-smi missing')
        if gpu.returncode != 0 or not gpu.stdout.strip():
            report('unavailable', reason='This Colab runtime has no GPU')
            raise RuntimeError('GPU 不可用，请通过 Colab 标准 UI 重试。')
        print(gpu.stdout)
        report('available', gpu=gpu.stdout.strip())
    """)
    encoded = base64.b64encode(bundle_bytes).decode('ascii')
    code(f"import base64\nEMBEDDED_SOURCE = base64.b64decode({encoded!r})")
    code("""
        from google.colab import drive, files
        try:
            drive.mount('/content/drive')
        except Exception as exc:
            report('auth', reason='Drive authorization/mount requires attention: ' + str(exc))
            raise
        DRIVE_PROJECT = pathlib.Path('/content/drive/MyDrive') / SETTINGS['drive_root'] / PROJECT_ID
        DRIVE_PROJECT.mkdir(parents=True, exist_ok=True)
        for name in ('inputs', 'checkpoints', 'results'):
            (DRIVE_PROJECT / name).mkdir(exist_ok=True)
        archive = DRIVE_PROJECT / 'source.zip'
        if not archive.exists() or hashlib.sha256(archive.read_bytes()).hexdigest() != SETTINGS['bundle_sha256']:
            if hashlib.sha256(EMBEDDED_SOURCE).hexdigest() != SETTINGS['bundle_sha256']:
                report('failed', reason='embedded source checksum mismatch')
                raise RuntimeError('Notebook embedded source checksum mismatch')
            temporary = archive.with_suffix('.zip.tmp')
            temporary.write_bytes(EMBEDDED_SOURCE)
            os.replace(temporary, archive)
        # Each execution gets fresh ephemeral source. Persistent data,
        # checkpoints and results stay in DRIVE_PROJECT and are never removed.
        SOURCE = pathlib.Path('/content') / ('autoresearch_' + PROJECT_ID + '_' + RUNTIME_ID)
        SOURCE.mkdir()
        with zipfile.ZipFile(archive) as bundle:
            members = bundle.infolist()
            if sum(item.file_size for item in members) > 100_000_000:
                raise RuntimeError('Source archive is too large')
            for item in members:
                target = (SOURCE / item.filename).resolve()
                if not target.is_relative_to(SOURCE.resolve()) or (item.external_attr >> 16) & 0o170000 == 0o120000:
                    raise RuntimeError('Unsafe source archive member')
            bundle.extractall(SOURCE)
        os.chdir(SOURCE)
        os.environ.update(AUTORESEARCH_DRIVE_PROJECT=str(DRIVE_PROJECT),
                          AUTORESEARCH_CHECKPOINT_DIR=str(DRIVE_PROJECT / 'checkpoints'),
                          AUTORESEARCH_DATA_DIR=str(DRIVE_PROJECT / 'inputs'),
                          AUTORESEARCH_RESULTS_DIR=str(DRIVE_PROJECT / 'results'),
                          AUTORESEARCH_SESSION_SECONDS='36000',
                          AUTORESEARCH_SESSION_DEADLINE_UTC=(datetime.datetime.fromisoformat(STARTED_AT) + datetime.timedelta(hours=10)).isoformat())
        print('持久化目录:', DRIVE_PROJECT)
    """)
    code(notebook_helpers_source())
    code("""
        config = SETTINGS['config']
        missing = [path for path in config.get('required_drive_paths', [])
                   if not (DRIVE_PROJECT / path).exists()]
        full_transfer = config.get('full_data_transfer')
        if full_transfer:
            pinned_manifest = DRIVE_PROJECT / 'inputs' / full_transfer['metadata_directory'] / full_transfer['manifest_path']
            if pinned_manifest.is_file() and file_sha256(pinned_manifest) != full_transfer['manifest_sha256']:
                missing.append(str(pinned_manifest.relative_to(DRIVE_PROJECT)))
            for item in full_transfer['files']:
                if item['kind'] == 'metadata':
                    current = DRIVE_PROJECT / 'inputs' / full_transfer['metadata_directory'] / item['path']
                    if current.is_file() and (current.stat().st_size != item['size_bytes'] or file_sha256(current) != item['sha256']):
                        missing.append(str(current.relative_to(DRIVE_PROJECT)))
        input_contract = config.get('input_archive')
        if missing and input_contract:
            print('AUTORESEARCH_COLAB_UPLOAD_INPUTS ' + json.dumps(dict(project_id=PROJECT_ID, runtime_id=RUNTIME_ID)), flush=True)
            print('请上传项目明确导出的 inputs.zip（经过校验的试验输入或完整数据元数据）。')
            uploaded = files.upload()
            candidates = [data for name, data in uploaded.items() if name.endswith('.zip')
                          and hashlib.sha256(data).hexdigest() == input_contract['sha256']]
            if len(candidates) != 1:
                report('failed', reason='uploaded input archive checksum mismatch')
                raise RuntimeError('Inputs archive does not match the project manifest')
            import io, stat
            with zipfile.ZipFile(io.BytesIO(candidates[0])) as inputs:
                infos = inputs.infolist()
                expected = set(input_contract['members'])
                if len(infos) != len(expected) or set(info.filename for info in infos) != expected:
                    raise RuntimeError('Unexpected input archive members')
                destination = DRIVE_PROJECT / 'inputs'
                if sum(info.file_size for info in infos) > 250_000_000:
                    raise RuntimeError('Input archive is too large')
                for info in infos:
                    if not (destination / info.filename).resolve().is_relative_to(destination.resolve()) or stat.S_ISLNK(info.external_attr >> 16):
                        raise RuntimeError('Unsafe input archive member')
                inputs.extractall(destination)
            missing = [path for path in config.get('required_drive_paths', [])
                       if not (DRIVE_PROJECT / path).exists()]
        if missing:
            report('auth', reason='missing_inputs', missing_inputs=missing)
            raise FileNotFoundError('请先将真实实验输入上传至 Google Drive: ' + ', '.join(missing))
        if full_transfer:
            pinned_manifest = DRIVE_PROJECT / 'inputs' / full_transfer['metadata_directory'] / full_transfer['manifest_path']
            if file_sha256(pinned_manifest) != full_transfer['manifest_sha256']:
                raise RuntimeError('Uploaded metadata manifest checksum mismatch')
            for item in full_transfer['files']:
                if item['kind']=='metadata':
                    current = DRIVE_PROJECT / 'inputs' / full_transfer['metadata_directory'] / item['path']
                    if current.stat().st_size!=item['size_bytes'] or file_sha256(current)!=item['sha256']:
                        raise RuntimeError('Uploaded metadata file checksum mismatch')
        if input_contract:
            print('AUTORESEARCH_COLAB_INPUTS_READY ' + json.dumps(dict(project_id=PROJECT_ID,
                runtime_id=RUNTIME_ID,sha256=input_contract['sha256'])),flush=True)
        try:
            for command in config.get('setup_commands', []):
                print('Installing project dependencies:', command)
                subprocess.run(command, shell=True, check=True, executable='/bin/bash')
        except Exception as exc:
            report('failed', reason='dependency setup failed: ' + str(exc),
                   returncode=getattr(exc, 'returncode', None))
            raise
    """)
    code("""
        # AUTORESEARCH_PREFLIGHT_GATE: no bulk data request precedes this cell.
        import importlib, math, signal, tempfile
        RUNTIME_PREFLIGHT = dict(schema_version=1, project_id=PROJECT_ID, runtime_id=RUNTIME_ID,
                                 phase='preflight', status='checking',
                                 scientific_metrics_produced=False)

        def preflight_disk(path):
            try:
                usage = shutil.disk_usage(path)
            except OSError as exc:
                return dict(filesystem_available_bytes=None, filesystem_space_status='unknown',
                            filesystem_error_type=type(exc).__name__, cloud_quota_status='unknown')
            return dict(filesystem_total_bytes=usage.total,
                        filesystem_used_bytes=usage.used,
                        filesystem_available_bytes=usage.free,
                        filesystem_space_status='measured', cloud_quota_status='unknown')

        def preflight_cuda(torch):
            sample = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda:0', dtype=torch.float32)
            product = sample @ sample
            torch.cuda.synchronize()
            if not torch.isfinite(product).all().item() or product.tolist() != [[7.0, 10.0], [15.0, 22.0]]:
                raise RuntimeError('CUDA matrix multiplication failed its finite-value and expected-result check')
            return dict(status='passed', operation='float32_matmul', synchronized=True)

        def preflight_torchaudio(torch, torchaudio):
            samples = torch.arange(800, dtype=torch.float32)
            waveform = (torch.sin(samples * (2.0 * math.pi * 440.0 / 8000)) * 0.1).unsqueeze(0)
            waveform = torchaudio.functional.resample(waveform, 8000, 16000)
            if list(waveform.shape) != [1, 1600] or not torch.isfinite(waveform).all().item():
                raise RuntimeError('TorchAudio resample produced unexpected or non-finite samples')
            features = torchaudio.compliance.kaldi.fbank(waveform * (1 << 15),
                num_mel_bins=80, frame_length=25.0, frame_shift=10.0, dither=0.0,
                sample_frequency=16000.0, window_type='hamming', use_energy=False)
            if list(features.shape) != [8, 80] or not torch.isfinite(features).all().item():
                raise RuntimeError('TorchAudio Kaldi fbank produced unexpected or non-finite features')
            return dict(status='passed', source_rate=8000, input_rate=16000, feature_shape=list(features.shape))

        def save_preflight_evidence(path):
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                                prefix='.' + path.name, delete=False) as handle:
                    temporary = pathlib.Path(handle.name)
                    json.dump(RUNTIME_PREFLIGHT, handle, ensure_ascii=False, indent=2)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)

        def run_preflight_command(command, timeout_seconds, log_path, environment):
            # Use the same shell and durable log transport as project execution.
            # stdout goes directly to the log so a silent child cannot bypass
            # the wall-clock timeout by blocking a pipe reader.
            with open(log_path, 'a', encoding='utf-8', buffering=1) as log:
                process = subprocess.Popen(command, shell=True, executable='/bin/bash',
                    stdout=log, stderr=subprocess.STDOUT, text=True, bufsize=1,
                    env=environment, start_new_session=True)
                try:
                    returncode = process.wait(timeout=timeout_seconds)
                except BaseException:
                    if process.poll() is None:
                        if os.name == 'posix':
                            try:
                                os.killpg(process.pid, signal.SIGTERM)
                            except ProcessLookupError:
                                pass
                        else:
                            process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            if os.name == 'posix':
                                try:
                                    os.killpg(process.pid, signal.SIGKILL)
                                except ProcessLookupError:
                                    pass
                            else:
                                process.kill()
                            process.wait(timeout=5)
                    raise
            with open(log_path, 'rb') as saved:
                saved.seek(0, os.SEEK_END)
                saved.seek(max(0, saved.tell() - 8192))
                tail = saved.read().decode('utf-8', errors='replace')
            if tail:
                print(tail, end='' if tail.endswith('\\n') else '\\n', flush=True)
            if returncode:
                raise subprocess.CalledProcessError(returncode, command, output=tail)
            return tail

        preflight_log = DRIVE_PROJECT / 'results' / ('preflight_' + RUNTIME_ID + '.log')
        preflight_path = preflight_log.with_suffix('.json')
        try:
            drive_evidence = preflight_disk(DRIVE_PROJECT)
            probe = DRIVE_PROJECT / ('.preflight_' + RUNTIME_ID + '.tmp')
            probe_created = False
            try:
                with open(probe, 'xb') as handle:
                    probe_created = True
                    handle.write(RUNTIME_ID.encode('utf-8'))
                    handle.flush()
                    os.fsync(handle.fileno())
                if probe.read_bytes() != RUNTIME_ID.encode('utf-8'):
                    raise RuntimeError('Drive preflight write/read verification failed')
                drive_evidence['write_verified'] = True
            finally:
                if probe_created:
                    probe.unlink(missing_ok=True)
            RUNTIME_PREFLIGHT['drive'] = drive_evidence
            transfer = config.get('full_data_transfer')
            scratch = pathlib.Path('/content/autoresearch-full-data') if transfer else SOURCE.parent
            if scratch.is_symlink():
                raise RuntimeError('Preflight scratch directory must not be linked')
            scratch.mkdir(parents=True, exist_ok=True)
            RUNTIME_PREFLIGHT['scratch'] = preflight_disk(scratch)
            if transfer:
                budget = estimate_ephemeral_disk_budget(scratch, transfer)
                RUNTIME_PREFLIGHT['scratch']['corpus_budget'] = budget
                if not budget['enough']:
                    raise RuntimeError('Insufficient Colab ephemeral disk space for the declared full corpus; stop before bulk upload.')
            requirements = config.get('runtime_requirements') or {}
            framework = requirements.get('framework')
            RUNTIME_PREFLIGHT['framework'] = dict(status='not_requested')
            if framework is None and any(requirements.get(key) for key in
                                         ('require_cuda', 'match_torchaudio', 'torch_version', 'torchaudio_version')):
                raise ValueError('Runtime framework must be declared for CUDA, TorchAudio or version requirements')
            if requirements.get('torchaudio_version') and not requirements.get('match_torchaudio'):
                raise ValueError('TorchAudio version pin requires match_torchaudio')
            if framework is not None:
                if framework != 'pytorch':
                    raise ValueError('Unsupported explicit runtime framework requirement')
                framework_evidence = dict(name='pytorch', status='checking')
                RUNTIME_PREFLIGHT['framework'] = framework_evidence
                torch = importlib.import_module('torch')
                framework_evidence.update(torch_version=str(torch.__version__),
                                          cuda_available=bool(torch.cuda.is_available()))
                pinned_torch = requirements.get('torch_version')
                if pinned_torch and framework_evidence['torch_version'] != pinned_torch:
                    raise RuntimeError('Expected Torch ' + pinned_torch + ', found ' + framework_evidence['torch_version'])
                if requirements.get('match_torchaudio'):
                    torchaudio = importlib.import_module('torchaudio')
                    framework_evidence['torchaudio_version'] = str(torchaudio.__version__)
                    pinned_audio = requirements.get('torchaudio_version')
                    if pinned_audio and framework_evidence['torchaudio_version'] != pinned_audio:
                        raise RuntimeError('Expected TorchAudio ' + pinned_audio + ', found ' + framework_evidence['torchaudio_version'])
                    if framework_evidence['torch_version'].split('+')[0] != framework_evidence['torchaudio_version'].split('+')[0]:
                        raise RuntimeError('Torch and TorchAudio versions must match before bulk upload')
                if requirements.get('require_cuda') and not framework_evidence['cuda_available']:
                    raise RuntimeError('No actual CUDA GPU has been allocated for the declared PyTorch project')
                if framework_evidence['cuda_available']:
                    framework_evidence.update(cuda_name=torch.cuda.get_device_name(0),
                        cuda_total_memory_bytes=int(torch.cuda.get_device_properties(0).total_memory))
                if requirements.get('require_cuda'):
                    framework_evidence['cuda_smoke'] = dict(status='checking')
                    framework_evidence['cuda_smoke'] = preflight_cuda(torch)
                if requirements.get('match_torchaudio'):
                    framework_evidence['torchaudio_smoke'] = dict(status='checking')
                    framework_evidence['torchaudio_smoke'] = preflight_torchaudio(torch, torchaudio)
                framework_evidence['status'] = 'passed'
            preflight_command = config.get('preflight_command') or ''
            if preflight_command:
                timeout_seconds = config.get('preflight_timeout_seconds', 900)
                if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 3600:
                    raise ValueError('Preflight timeout must be an integer between 1 and 3600 seconds')
                deadline_text = os.environ.get('AUTORESEARCH_SESSION_DEADLINE_UTC')
                if deadline_text:
                    deadline = datetime.datetime.fromisoformat(deadline_text.replace('Z', '+00:00'))
                    remaining = (deadline - datetime.datetime.now(datetime.timezone.utc)).total_seconds() - 180
                    if remaining <= 0:
                        raise RuntimeError('Session budget is exhausted before project preflight')
                    timeout_seconds = min(timeout_seconds, remaining)
                # Project preflight may inspect inputs and populate pinned model
                # inputs, but its checkpoints/results have a separate namespace.
                # They can never satisfy the formal experiment's resume/metrics.
                probe_root = pathlib.Path(tempfile.mkdtemp(prefix='preflight_', dir=SOURCE))
                environment = os.environ.copy()
                environment.update(AUTORESEARCH_PREFLIGHT='1',
                    AUTORESEARCH_CHECKPOINT_DIR=str(probe_root / 'checkpoints'),
                    AUTORESEARCH_RESULTS_DIR=str(probe_root / 'results'))
                RUNTIME_PREFLIGHT['project_command'] = dict(status='running',
                    timeout_seconds=timeout_seconds, log_path=str(preflight_log))
                report('starting', phase='preflight', preflight=RUNTIME_PREFLIGHT)
                output_tail = run_preflight_command(preflight_command, timeout_seconds, preflight_log, environment)
                RUNTIME_PREFLIGHT['project_command'].update(status='passed', returncode=0,
                    output_tail=output_tail)
            else:
                RUNTIME_PREFLIGHT['project_command'] = dict(status='not_configured')
            RUNTIME_PREFLIGHT['status'] = 'passed'
            save_preflight_evidence(preflight_path)
            report('starting', phase='preflight', preflight=RUNTIME_PREFLIGHT,
                   preflight_path=str(preflight_path))
        except BaseException as exc:
            RUNTIME_PREFLIGHT.update(status='failed', error_type=type(exc).__name__, reason=str(exc))
            framework_evidence = RUNTIME_PREFLIGHT.get('framework', {})
            if framework_evidence.get('status') == 'checking':
                framework_evidence['status'] = 'failed'
                for probe in ('cuda_smoke', 'torchaudio_smoke'):
                    if framework_evidence.get(probe, {}).get('status') == 'checking':
                        framework_evidence[probe]['status'] = 'failed'
            if RUNTIME_PREFLIGHT.get('project_command', {}).get('status') == 'running':
                RUNTIME_PREFLIGHT['project_command'].update(status='failed',
                    returncode=getattr(exc, 'returncode', None), timed_out=isinstance(exc, subprocess.TimeoutExpired))
                if preflight_log.is_file():
                    with open(preflight_log, 'rb') as saved:
                        saved.seek(0, os.SEEK_END)
                        saved.seek(max(0, saved.tell() - 8192))
                        RUNTIME_PREFLIGHT['project_command']['output_tail'] = saved.read().decode('utf-8', errors='replace')
            release = {}
            if RUNTIME_PREFLIGHT.get('drive', {}).get('write_verified') is True:
                try:
                    save_preflight_evidence(preflight_path)
                except OSError as save_error:
                    RUNTIME_PREFLIGHT['persistence_error'] = type(save_error).__name__
                else:
                    release = dict(runtime_release_safe=True, preflight_path=str(preflight_path))
            report('failed', reason='Runtime preflight failed before bulk upload: ' + str(exc),
                   preflight=RUNTIME_PREFLIGHT, log_path=str(preflight_log), **release)
            raise
    """)
    code("""
        transfer = config.get('full_data_transfer')
        if transfer:
            import gc
            metadata_root = DRIVE_PROJECT / 'inputs' / transfer['metadata_directory']
            manifest_source = metadata_root / transfer['manifest_path']
            if file_sha256(manifest_source) != transfer['manifest_sha256']:
                raise RuntimeError('Full-data manifest checksum differs from the prepared notebook')
            data_root = pathlib.Path('/content/autoresearch-full-data')
            if data_root.is_symlink():
                raise RuntimeError('Ephemeral data directory must not be linked')
            data_root.mkdir(parents=True, exist_ok=True)
            disk_budget = estimate_ephemeral_disk_budget(data_root, transfer)
            print('Colab 临时磁盘预算: required={:.2f} GiB, free={:.2f} GiB (含 10 GiB 余量)'.format(
                disk_budget['required_bytes'] / 1024**3, disk_budget['free_bytes'] / 1024**3), flush=True)
            if not disk_budget['enough']:
                reason = 'Insufficient Colab ephemeral disk space for the declared full corpus; stop before bulk upload.'
                report('failed', reason=reason, disk_budget=disk_budget)
                raise RuntimeError(reason)
            if (globals().get('RUNTIME_PREFLIGHT', {}).get('status') != 'passed'
                    or RUNTIME_PREFLIGHT.get('runtime_id') != RUNTIME_ID):
                raise RuntimeError('A successful current-runtime preflight is required before bulk upload')
            report('starting', phase='preparing_data', preflight=RUNTIME_PREFLIGHT)
            for item in transfer['files']:
                relative = relative_data_path(item['path'])
                if item['kind'] == 'metadata':
                    original = metadata_root / relative
                    if original.stat().st_size != item['size_bytes'] or file_sha256(original) != item['sha256']:
                        raise RuntimeError('Full-data metadata file checksum differs')
                    target = data_root / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(original, target)
            shutil.copyfile(manifest_source, data_root / 'data_manifest.json')
            artifact_entries = list(transfer['augmentation'].values()) + transfer.get('assemblies', [])
            part_owners = {part: entry for entry in artifact_entries for part in entry['parts']}
            assembled = {entry['path'] for entry in artifact_entries
                         if (data_root / entry['path']).is_file()
                         and (data_root / entry['path']).stat().st_size == entry['size_bytes']
                         and file_sha256(data_root / entry['path']) == entry['sha256']}
            # Assemble and release large augmentation/trial chunks before speech
            # arrives, so a free runtime need not retain two copies of each LMDB.
            ordered_files = sorted(transfer['files'], key=lambda item:
                (0, part_owners[item['path']]['path'], item['path'])
                if item['path'] in part_owners else (1, '', item['path']))
            for item in ordered_files:
                if item['kind'] == 'metadata':
                    continue
                relative = relative_data_path(item['path'])
                owner = part_owners.get(relative)
                if owner and owner['path'] in assembled:
                    continue
                target = data_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.is_symlink() or not target.resolve().is_relative_to(data_root.resolve()):
                    raise RuntimeError('Unsafe full-data upload destination')
                matches = target.is_file() and target.stat().st_size == item['size_bytes'] and file_sha256(target) == item['sha256']
                if not matches:
                    request = dict(project_id=PROJECT_ID, runtime_id=RUNTIME_ID, path=relative,
                                   size_bytes=item['size_bytes'], sha256=item['sha256'])
                    print('AUTORESEARCH_COLAB_UPLOAD_DATA ' + json.dumps(request), flush=True)
                    print('Uploading one approved full-data shard:', relative, flush=True)
                    incoming = SOURCE / ('upload_' + uuid.uuid4().hex)
                    incoming.mkdir()
                    os.chdir(incoming)
                    try:
                        uploaded = files.upload()
                        if len(uploaded) != 1:
                            raise RuntimeError('Full-data upload must contain exactly one declared file')
                        uploaded_name, uploaded_bytes = next(iter(uploaded.items()))
                        if (pathlib.Path(uploaded_name).name != pathlib.Path(relative).name
                                or len(uploaded_bytes) != item['size_bytes']
                                or hashlib.sha256(uploaded_bytes).hexdigest() != item['sha256']):
                            raise RuntimeError('Full-data uploaded shard size or checksum mismatch')
                        temporary = target.with_suffix('.tar.uploading')
                        with open(temporary, 'wb') as output:
                            output.write(uploaded_bytes)
                            output.flush()
                            os.fsync(output.fileno())
                        os.replace(temporary, target)
                        del uploaded_bytes, uploaded
                        gc.collect()
                    finally:
                        os.chdir(SOURCE)
                        if incoming.resolve().is_relative_to(SOURCE.resolve()):
                            shutil.rmtree(incoming)
                print('AUTORESEARCH_COLAB_DATA_READY ' + json.dumps(dict(project_id=PROJECT_ID,
                    runtime_id=RUNTIME_ID,path=relative,size_bytes=item['size_bytes'],sha256=item['sha256'])),flush=True)
                if owner and relative == owner['parts'][-1]:
                    reconstruct_artifacts(data_root, [owner])
                    assembled.add(owner['path'])
                if item['kind'] == 'dev_archive':
                    audio_directory = 'acceptance/dev/audio' if relative.startswith('acceptance/') else 'dev/audio'
                    safe_extract_dev_archive(target, data_root / audio_directory,
                        expected_audio_bytes=item['audio_bytes'], expected_utterances=item['utterances'])
            os.environ['AUTORESEARCH_AUGMENTATION_DIR'] = str(
                (data_root / transfer['augmentation']['musan']['path']).parents[2])
            os.environ['AUTORESEARCH_FULL_DATA_MANIFEST'] = str(data_root / 'data_manifest.json')
            print('Full train/dev corpus transferred and verified:', data_root, flush=True)
    """)
    code("""
        checkpoint_dir = DRIVE_PROJECT / 'checkpoints'
        results_dir = DRIVE_PROJECT / 'results'
        preflight_evidence = globals().get('RUNTIME_PREFLIGHT', dict(status='not_run'))
        if (config.get('preflight_command') or config.get('runtime_requirements') or config.get('full_data_transfer')):
            if preflight_evidence.get('status') != 'passed' or preflight_evidence.get('runtime_id') != RUNTIME_ID:
                raise RuntimeError('A successful current-runtime preflight is required before experiment execution')
        drive_verified = (preflight_evidence.get('status') == 'passed'
                          and preflight_evidence.get('runtime_id') == RUNTIME_ID
                          and preflight_evidence.get('drive', {}).get('write_verified') is True)
        # Checkpoint implementation belongs to the project. No generic model
        # state is invented here; export requires an explicit full-resume contract.
        checkpoint = config['checkpoint']
        found_checkpoint = any(any(checkpoint_dir.rglob(name)) for name in checkpoint['files'])
        command = checkpoint['resume_command'] if found_checkpoint else config['run_command']
        session_exit_code = checkpoint.get('session_exit_code')
        check_resume_safety = (drive_verified and checkpoint.get('full_training_state') is True
                               and checkpoint.get('resume_supported') is True
                               and type(session_exit_code) is int and session_exit_code > 0)

        def checkpoint_stamps():
            root = checkpoint_dir.resolve()
            stamps = {}
            if checkpoint_dir.is_symlink() or not root.is_relative_to(DRIVE_PROJECT.resolve()):
                return stamps
            for name in checkpoint['files']:
                for path in checkpoint_dir.rglob(name):
                    if path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root):
                        info = path.stat()
                        if info.st_size:
                            stamps[path] = (info.st_mtime_ns, info.st_size)
            return stamps

        prepare_command = config.get('prepare_command', '')
        steps = ([prepare_command] if prepare_command else []) + [command]
        report('running', checkpoint_dir=str(checkpoint_dir), results_dir=str(results_dir),
               resumed=found_checkpoint, preflight=preflight_evidence)
        log_path = results_dir / ('session_' + RUNTIME_ID + '.log')
        run_started = False
        try:
            with open(log_path, 'a', encoding='utf-8', buffering=1) as log:
                for index, step in enumerate(steps):
                    if index == len(steps) - 1:
                        checkpoint_before_run = checkpoint_stamps() if check_resume_safety else {}
                        run_started = True
                    print('Executing:', step, flush=True)
                    process = subprocess.Popen(step, shell=True, executable='/bin/bash',
                                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                               text=True, bufsize=1)
                    for line in process.stdout:
                        print(line, end='', flush=True)
                        log.write(line)
                    returncode = process.wait()
                    log.flush()
                    os.fsync(log.fileno())
                    if returncode:
                        raise subprocess.CalledProcessError(returncode, step)
            validated = validate_scientific_result(results_dir, config.get('success_criteria'))
            if drive_verified and validated.get('metrics_file'):
                validated['runtime_release_safe'] = True
            report('completed', returncode=0, checkpoint_dir=str(checkpoint_dir),
                   results_dir=str(results_dir), log_path=str(log_path), preflight=preflight_evidence, **validated)
        except BaseException as exc:
            exitcode = getattr(exc, 'returncode', None)
            resumable = exitcode is not None and exitcode == session_exit_code
            release = {}
            if resumable and run_started and check_resume_safety:
                try:
                    current_checkpoints = checkpoint_stamps()
                except OSError:
                    current_checkpoints = {}
                if any(checkpoint_before_run.get(path) != stamp for path, stamp in current_checkpoints.items()):
                    release = dict(runtime_release_safe=True)
            report('disconnected' if resumable else 'failed', returncode=exitcode,
                   reason=str(exc), checkpoint_dir=str(checkpoint_dir), log_path=str(log_path),
                   preflight=preflight_evidence, **release)
            raise
    """)
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "colab": {"name": f"AutoResearch_{project_id}.ipynb", "provenance": []},
        "accelerator": "GPU", "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"}}, "cells": cells}
