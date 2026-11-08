"""Pinned headless ImageJ runtime, independent of any external Fiji installation."""
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

if __package__:
    from .platform_support import exclusive_file_lock
else:
    from platform_support import exclusive_file_lock

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'source/imagej/runtime.json'
RUNTIME = ROOT / 'runtime/imagej'
BOOTSTRAP = ROOT / 'source/imagej/HeadlessImageJ.java'
REPAIR = 'Run python scripts/setup_imagej.py from the source app folder, or reinstall the packaged app.'


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def platform_key():
    machine = {'aarch64': 'arm64', 'amd64': 'x86_64'}.get(platform.machine().lower(), platform.machine().lower())
    key = {'Darwin': 'macos', 'Windows': 'windows'}.get(platform.system(), platform.system().lower()) + '-' + machine
    if key not in ('macos-arm64', 'macos-x86_64', 'windows-x86_64'):
        raise RuntimeError(f'Unsupported ImageJ platform: {key}')
    if platform.system() == 'Darwin':
        minimum = manifest()['minimum_macos'][machine]
        actual = tuple(int(n) for n in platform.mac_ver()[0].split('.')[:2])
        if actual < tuple(int(n) for n in minimum.split('.')):
            raise RuntimeError(f'The pinned CLIJ2-FFT runtime requires macOS {minimum} or newer on {machine}.')
    return key


def manifest():
    return json.loads(MANIFEST.read_text(encoding='utf-8'))


def runtime_info(directory=RUNTIME):
    """Validate every locked artifact before constructing a JVM classpath."""
    directory = Path(directory)
    data = manifest()
    key = platform_key()
    try:
        if (directory / 'manifest.json').read_bytes() != MANIFEST.read_bytes():
            raise ValueError('manifest does not match this app version')
        for artifact in data['artifacts']:
            path = directory / artifact['path']
            if sha256(path) != artifact['sha256']:
                raise ValueError(f"checksum mismatch: {artifact['path']}")
    except (OSError, ValueError) as error:
        raise RuntimeError(f'Missing or corrupt app-local ImageJ runtime: {error}. {REPAIR}') from error
    return {'manifest_sha256': sha256(MANIFEST), 'platform': key,
            'versions': data['versions'],
            'artifacts': {a['path']: a['sha256'] for a in data['artifacts']}}


def cache_directory():
    base = (Path(os.environ.get('LOCALAPPDATA', Path.home())) / 'Batch Deconvolution/cache'
            if os.name == 'nt' else Path.home() / 'Library/Caches/Batch Deconvolution')
    return base / sha256(MANIFEST)


def native_directory():
    artifact = next(a for a in manifest()['artifacts'] if '/clij2-fft_' in a['path'])
    native_platform = platform_key().replace('macos-', 'macosx-')
    return cache_directory() / 'javacpp' / Path(artifact['path']).name / 'net/haesleinhuepf/clijx/plugins' / native_platform


def prepare_native_cache(directory):
    # JavaCPP 1.5.10 does not extract clFFT's versioned dylibs or StatTimer.dll.
    # Populate its expected directory from the verified JAR before it loads JNI.
    target = native_directory()
    target.mkdir(parents=True, exist_ok=True)
    artifact = next(a for a in manifest()['artifacts'] if '/clij2-fft_' in a['path'])
    with exclusive_file_lock(target.parent / '.extract.lock', 120, lambda: None):
        with zipfile.ZipFile(Path(directory) / artifact['path']) as jar:
            for entry in artifact['native_libraries']:
                if Path(entry).parent.name != target.name:
                    continue
                data = jar.read(entry)
                output = target / Path(entry).name
                expected = hashlib.sha256(data).hexdigest()
                if output.is_file() and sha256(output) == expected:
                    continue
                with tempfile.NamedTemporaryFile(dir=target, delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(data)
                try:
                    temporary.replace(output)
                finally:
                    temporary.unlink(missing_ok=True)
    return target


def java_environment():
    environment = os.environ.copy()
    for name in ('JDK_JAVA_OPTIONS', 'JAVA_TOOL_OPTIONS', '_JAVA_OPTIONS', 'CLASSPATH'):
        environment.pop(name, None)
    if os.name == 'nt':
        environment['PATH'] = str(native_directory()) + os.pathsep + environment.get('PATH', '')
    return environment


def java_executable(*, allow_external=True):
    name = 'java.exe' if os.name == 'nt' else 'java'
    prefix = Path(sys.prefix)
    candidates = [prefix / p / name for p in ('lib/jvm/bin', 'Library/lib/jvm/bin', 'Library/bin', 'bin')]
    if allow_external and os.environ.get('JAVA_HOME'):
        candidates.append(Path(os.environ['JAVA_HOME']) / 'bin' / name)
    found = shutil.which(name)
    if allow_external and found:
        candidates.append(Path(found))
    for candidate in candidates:
        if candidate.is_file():
            try:
                result = subprocess.run([str(candidate), '-version'], capture_output=True, text=True,
                                        env=java_environment(), timeout=15)
            except (OSError, subprocess.TimeoutExpired):
                continue
            version = result.stderr + result.stdout
            if result.returncode == 0 and re.search(r'version "21(?:[.\"]|$)', version):
                return candidate.resolve()
    if (prefix / 'conda-meta').is_dir():
        raise RuntimeError(
            f'Java 21 was not found for Python environment {prefix}. '
            'Run python scripts/setup_imagej.py in that environment; setup installs Java automatically. '
            'Then restart the app.')
    raise RuntimeError(
        f'Java 21 was not found for Python {sys.executable}. '
        'Activate the Conda environment created from environment.yml, or reinstall the packaged app.')


def launch_command(memory, worker, parameters, directory=RUNTIME):
    """Return an argument vector; never consult Fiji or expand wildcard JAR paths."""
    runtime_info(directory)
    directory = Path(directory).resolve()
    classpath = os.pathsep.join(str(directory / a['path']) for a in manifest()['artifacts'])
    if not re.fullmatch(r'[1-9][0-9]*[mMgG]', str(memory)):
        raise ValueError('ImageJ memory must be a positive integer followed by m or g.')
    native = prepare_native_cache(directory)
    cache = cache_directory()
    command = [str(java_executable()), f'-Xmx{memory}', '-Djava.awt.headless=true',
               f'-Dorg.bytedeco.javacpp.cachedir={cache / "javacpp"}',
               f'-Djava.library.path={native}',
               '-Dimagej.updater.disableAutocheck=true', '-Dpython.cachedir.skip=true',
               f'-Dij.dir={directory}', f'-Dplugins.dir={directory / "plugins"}',
               '-cp', classpath, str(BOOTSTRAP), str(worker)]
    for key, value in parameters.items():
        command.extend((key, str(value).lower() if isinstance(value, bool) else str(value)))
    return command
