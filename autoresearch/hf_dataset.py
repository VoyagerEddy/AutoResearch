"""Private, pinned HF Dataset downloads without a persistent Hub cache.

This module is also embedded at the root of the Colab source bundle.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


HUB = 'https://huggingface.co'
CONFIG_ENV = 'AUTORESEARCH_HF_DATASET_CONFIG'
MAX_MANIFEST_BYTES = 2_000_000
DEFAULT_WINDOW_SHARDS = 4
MAX_WINDOW_BYTES = 1536 * 1024**2
BLOCK_BYTES = 4 * 1024**2
POINTER_FIELDS = {'repo_id', 'revision', 'manifest_path', 'manifest_sha256'}


class DatasetConfigurationError(ValueError):
    pass


class DatasetAuthError(RuntimeError):
    pass


class DatasetNetworkError(RuntimeError):
    pass


class DatasetIntegrityError(ValueError):
    pass


def validate_repo_id(repo_id):
    if (not isinstance(repo_id, str) or not re.fullmatch(
            r'[A-Za-z0-9_][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9_][A-Za-z0-9_.-]{0,95}', repo_id)
            or '--' in repo_id or '..' in repo_id
            or any(part.endswith(('.', '-', '.git')) for part in repo_id.split('/'))
            or contains_token(repo_id)):
        raise DatasetConfigurationError('HF Dataset repo_id must be a namespace/name')
    return repo_id


def contains_token(value):
    return re.search(r'hf_[A-Za-z0-9]{16,}', value) is not None


def relative_path(value):
    if (not isinstance(value, str) or len(value) > 240 or contains_token(value)
            or any(not re.fullmatch(r'[A-Za-z0-9_-][A-Za-z0-9_./-]*', part)
                   or part.endswith('.') for part in value.split('/'))):
        raise DatasetConfigurationError('Invalid HF Dataset artifact path')
    return value


def validate_hf_pointer(pointer):
    if not isinstance(pointer, dict) or set(pointer) != POINTER_FIELDS:
        raise DatasetConfigurationError('HF Dataset pointer requires repo_id, revision, manifest_path and manifest_sha256')
    validate_repo_id(pointer['repo_id'])
    validate_revision(pointer['revision'])
    checksum = pointer['manifest_sha256']
    if not isinstance(checksum, str) or not re.fullmatch(r'[0-9a-f]{64}', checksum):
        raise DatasetConfigurationError('HF Dataset manifest_sha256 must be a full SHA-256 digest')
    path = relative_path(pointer['manifest_path'])
    if Path(path).name != 'data_manifest.json':
        raise DatasetConfigurationError('HF Dataset manifest_path must name data_manifest.json')
    return dict(pointer)


def validate_revision(revision):
    if not isinstance(revision, str) or not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise DatasetConfigurationError('HF Dataset revision must be a full commit SHA')
    return revision


def validate_token(token):
    if not isinstance(token, str) or not re.fullmatch(r'hf_[A-Za-z0-9]{16,}', token):
        raise DatasetAuthError('Configure a valid HF_TOKEN with access to the private Dataset')
    return token


def read_token(*, colab_secrets=False):
    token = os.environ.get('HF_TOKEN')
    if not token and colab_secrets:
        try:
            from google.colab import userdata
            token = userdata.get('HF_TOKEN')
        except Exception:
            # Colab may deny access or attach details to its exception.
            token = None
    return validate_token(token)


class _HubRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urlsplit(newurl)
        if target.scheme != 'https' or target.username is not None or not target.hostname:
            raise DatasetConfigurationError('HF Dataset returned an unsafe download redirect')
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is None:
            return None
        if target.netloc == urlsplit(HUB).netloc and req.has_header('Authorization'):
            redirected.add_unredirected_header('Authorization', req.get_header('Authorization'))
        else:
            redirected.remove_header('Authorization')
        return redirected


class HubReader:
    def __init__(self, token, *, opener=None, retries=3):
        self._token = token
        self._open = opener or build_opener(_HubRedirect()).open
        self._retries = retries

    def _read(self, url, *, limit, output=None, expected_hash=None, exact=False, capture=True):
        for attempt in range(self._retries + 1):
            try:
                request = Request(url)
                # The redirect handler explicitly preserves authorization only on the Hub origin.
                request.add_unredirected_header('Authorization', 'Bearer ' + self._token)
                if output is not None:
                    output.seek(0)
                    output.truncate()
                body = bytearray() if output is None and capture else None
                checksum = hashlib.sha256()
                size = 0
                with self._open(request, timeout=60) as response:
                    while block := response.read(BLOCK_BYTES):
                        size += len(block)
                        if size > limit:
                            raise DatasetIntegrityError('HF Dataset artifact exceeds its declared size bound')
                        checksum.update(block)
                        if body is not None:
                            body.extend(block)
                        elif output is not None:
                            output.write(block)
                if exact and size != limit:
                    # A dropped connection can surface as early EOF without an exception.
                    if attempt < self._retries:
                        time.sleep(2**attempt)
                        continue
                    raise DatasetIntegrityError('HF Dataset artifact size mismatch')
                if expected_hash is not None and checksum.hexdigest() != expected_hash:
                    raise DatasetIntegrityError('HF Dataset artifact SHA-256 mismatch')
                return bytes(body) if body is not None else size
            except HTTPError as exc:
                status = exc.code
                exc.close()
                if status in {401, 403, 404}:
                    raise DatasetAuthError('HF Dataset or pinned artifact is inaccessible with the supplied HF_TOKEN') from None
                if status not in {408, 429, 500, 502, 503, 504}:
                    raise DatasetConfigurationError(f'HF Dataset request rejected (HTTP {status})') from None
            except (URLError, TimeoutError, ConnectionError, http.client.HTTPException):
                pass
            if attempt < self._retries:
                time.sleep(2**attempt)
        raise DatasetNetworkError('HF Dataset download failed after network retries')

    def require_private(self, repo_id, revision):
        validate_repo_id(repo_id)
        validate_revision(revision)
        url = f'{HUB}/api/datasets/{repo_id}/revision/{revision}'
        try:
            info = json.loads(self._read(url, limit=MAX_MANIFEST_BYTES))
        except (ValueError, UnicodeError) as exc:
            if isinstance(exc, (DatasetConfigurationError, DatasetIntegrityError)):
                raise
            raise DatasetConfigurationError('HF Dataset repository metadata is invalid') from None
        if not isinstance(info, dict) or info.get('private') is not True:
            raise DatasetConfigurationError('HF Dataset repository must be private')
        if info.get('sha') != revision:
            raise DatasetConfigurationError('HF Dataset API did not confirm the pinned commit')

    def manifest(self, pointer):
        pointer = validate_hf_pointer(pointer)
        self.require_private(pointer['repo_id'], pointer['revision'])
        url = self._artifact_url(pointer['repo_id'], pointer['revision'], pointer['manifest_path'])
        return self._read(url, limit=MAX_MANIFEST_BYTES, expected_hash=pointer['manifest_sha256'])

    @staticmethod
    def _artifact_url(repo_id, revision, relative):
        return f'{HUB}/datasets/{repo_id}/resolve/{revision}/{quote(relative, safe="/")}'

    def download(self, pointer, item, output):
        relative = relative_path(item['path'])
        # A manifest in a repository subdirectory roots the inventory there.
        base = pointer['manifest_path'].rpartition('/')[0]
        if base:
            relative = base + '/' + relative
        url = self._artifact_url(pointer['repo_id'], pointer['revision'], relative)
        return self._read(url, limit=item['size_bytes'], output=output,
                          expected_hash=item['sha256'], exact=True, capture=False)


def file_matches(path, item):
    if path.is_symlink() or not path.is_file() or path.stat().st_size != item['size_bytes']:
        return False
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest() == item['sha256']


def atomic_write(path, write):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w+b', dir=path.parent, prefix='.hf-install-', delete=False) as output:
        temporary = Path(output.name)
        try:
            write(output)
            output.flush()
            os.fsync(output.fileno())
            output.close()
            os.replace(temporary, path)
        finally:
            output.close()
            temporary.unlink(missing_ok=True)


def safe_target(root, relative):
    target = root / relative_path(relative)
    for part in (target, *target.parents):
        if part.is_symlink() or getattr(part, 'is_junction', lambda: False)():
            raise DatasetConfigurationError('HF Dataset local paths must be unlinked')
        if part == root:
            break
    return target


def install_metadata(root, raw, transfer, pointer, reader):
    root = Path(root)
    records = {item['path']: item for item in transfer['files']}
    for item in transfer['files']:
        if item['kind'] != 'metadata':
            continue
        target = safe_target(root, item['path'])
        if not file_matches(target, item):
            target.unlink(missing_ok=True)
            atomic_write(target, lambda output: reader.download(pointer, item, output))

    def assemble(entry, output):
        for part in entry['parts']:
            # Spool one piece so a retried connection never duplicates bytes
            # already assembled. The piece is unlinked as soon as it is copied.
            with tempfile.TemporaryFile(dir=root) as chunk:
                reader.download(pointer, records[part], chunk)
                chunk.seek(0)
                shutil.copyfileobj(chunk, output, length=BLOCK_BYTES)
        output.seek(0)
        if (os.fstat(output.fileno()).st_size != entry['size_bytes']
                or hashlib.file_digest(output, 'sha256').hexdigest() != entry['sha256']):
            raise DatasetIntegrityError('HF Dataset assembled artifact checksum mismatch')

    entries = list(transfer['assemblies'])
    if transfer['online_augmentation']:
        entries += list(transfer['augmentation'].values())
    for entry in entries:
        target = safe_target(root, entry['path'])
        if not file_matches(target, entry):
            target.unlink(missing_ok=True)
            atomic_write(target, lambda output: assemble(entry, output))
    manifest = safe_target(root, 'data_manifest.json')
    if not file_matches(manifest, {'size_bytes': len(raw), 'sha256': pointer['manifest_sha256']}):
        atomic_write(manifest, lambda output: output.write(raw))


@contextmanager
def _slot_lock(path):
    with path.open('a+b') as handle:
        if os.name == 'nt':
            import msvcrt

            def lock():
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)

            def unlock():
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            def lock():
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

            def unlock():
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

        try:
            lock()
        except (BlockingIOError, PermissionError):
            yield False
        except OSError as exc:
            if os.name != 'nt' or exc.errno != 13:
                raise
            yield False
        else:
            try:
                yield True
            finally:
                unlock()


class Dataset:
    def __init__(self, root, config, records, reader):
        self.root = Path(root)
        self.pointer = validate_hf_pointer({key: config.get(key) for key in POINTER_FIELDS})
        self.online_augmentation = config.get('online_augmentation', True)
        self.window_shards = config.get('window_shards', DEFAULT_WINDOW_SHARDS)
        self.window_bytes = config.get('window_bytes', MAX_WINDOW_BYTES)
        if (type(self.online_augmentation) is not bool
                or type(self.window_shards) is not int or not 1 <= self.window_shards <= 8
                or type(self.window_bytes) is not int or not 0 < self.window_bytes <= MAX_WINDOW_BYTES):
            raise DatasetConfigurationError('Invalid private Dataset shard-window configuration')
        self.records = {}
        for item in records:
            relative = item['path']
            kind = item.get('kind')
            if kind is None:
                if re.fullmatch(r'train/shards/train-[0-9]+\.tar(?:\.gz)?', relative):
                    kind = 'train_shard'
                elif re.fullmatch(r'dev/shards/dev-[0-9]+\.tar', relative):
                    kind = 'dev_archive'
            self.records[relative] = {**item, 'kind': kind}
        self._reader = reader
        self._cache = safe_target(self.root, 'hf-shards')
        self._cache.mkdir(parents=True, exist_ok=True)

    def clean_abandoned_slots(self):
        for index in range(8):
            with _slot_lock(self._cache / f'slot-{index}.lock') as locked:
                if locked:
                    (self._cache / f'slot-{index}.tar').unlink(missing_ok=True)

    @classmethod
    def from_environment(cls):
        value = os.environ.get(CONFIG_ENV)
        if not value:
            return None
        try:
            config = json.loads(value)
            root = Path(config['root'])
            source = safe_target(root, 'data_manifest.json')
            if source.stat().st_size > MAX_MANIFEST_BYTES:
                raise DatasetIntegrityError('Local HF Dataset manifest exceeds its size limit')
            raw = source.read_bytes()
            if hashlib.sha256(raw).hexdigest() != config['manifest_sha256']:
                raise DatasetIntegrityError('Local HF Dataset manifest SHA-256 mismatch')
            records = json.loads(raw)['files']
        except DatasetIntegrityError:
            raise
        except (OSError, TypeError, ValueError, KeyError):
            raise DatasetConfigurationError('Invalid staged HF Dataset configuration or manifest') from None
        return cls(root, config, records, HubReader(read_token()))

    @contextmanager
    def shard(self, relative):
        item = self.records.get(relative_path(relative))
        if not item or item.get('kind') not in {'train_shard', 'dev_archive'}:
            raise DatasetConfigurationError('Requested archive is absent from the HF Dataset inventory')
        # Train and dev run in separate phases. The dev reader takes every
        # slot, allowing one larger archive while excluding training readers.
        shared = item['kind'] == 'train_shard'
        limit = self.window_bytes // self.window_shards if shared else self.window_bytes
        if item['size_bytes'] > limit:
            raise DatasetConfigurationError('HF Dataset archive exceeds its shard-window allocation')
        with self._lease(shared) as target:
            try:
                with target.open('wb') as output:
                    self._reader.download(self.pointer, item, output)
                yield target
            finally:
                target.unlink(missing_ok=True)

    @contextmanager
    def _lease(self, shared):
        while True:
            with ExitStack() as locks:
                slots = []
                for index in range(self.window_shards):
                    lock = self._cache / f'slot-{index}.lock'
                    if locks.enter_context(_slot_lock(lock)):
                        slot = self._cache / f'slot-{index}.tar'
                        # A killed worker may leave the prior slot file behind.
                        slot.unlink(missing_ok=True)
                        slots.append(slot)
                        if shared:
                            break
                    elif not shared:
                        break
                if len(slots) == (1 if shared else self.window_shards):
                    yield slots[0]
                    return
            time.sleep(0.05)
