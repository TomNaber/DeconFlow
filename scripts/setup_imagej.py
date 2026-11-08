"""Install the locked ImageJ runtime. No Maven, Fiji, or compiler setup required."""
import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import sys
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from source.imagej_runtime import MANIFEST, RUNTIME, platform_key, runtime_info, sha256, java_executable
from source.platform_support import exclusive_file_lock
from source.bioformats_runtime import MANIFEST as BF_MANIFEST, runtime_info as bf_runtime_info


def ensure_java():
    """Provision Java inside the active environment so releases can bundle it."""
    try:
        return java_executable(allow_external=False)
    except RuntimeError:
        pass
    prefix = Path(sys.prefix)
    if not (prefix / 'conda-meta').is_dir():
        raise RuntimeError('Run source setup in the Conda environment from environment.yml. '
                           'Packaged apps already include Python and Java.')
    candidates = [Path(value) for value in (os.environ.get('CONDA_EXE'), shutil.which('conda')) if value]
    roots = [prefix, *prefix.parents]
    # Environments installed with --prefix may live outside the Conda installation.
    for record in (prefix / 'conda-meta').glob('python-*.json'):
        source = (json.loads(record.read_text()).get('link') or {}).get('source')
        if source:
            roots.append(Path(source).parent.parent)
    for root in roots:
        candidates.extend(root / path for path in ('bin/conda', 'Scripts/conda.exe', '_conda.exe'))
    conda = next((p for p in candidates if p.is_file() and os.access(p, os.X_OK)), None)
    if conda is None:
        raise RuntimeError('Conda could not be located to install Java. Run setup from a Conda terminal.')
    print(f'Installing Java 21 into {prefix}...', flush=True)
    subprocess.run([str(conda), 'install', '--prefix', str(prefix), '--yes',
                    '--override-channels', '-c', 'conda-forge', '--freeze-installed', 'openjdk=21'], check=True)
    return java_executable(allow_external=False)


def install(destination=RUNTIME):
    platform_key()
    ensure_java()
    imagej = install_artifacts(destination, MANIFEST, runtime_info)
    install_artifacts(Path(destination).parent / 'bioformats', BF_MANIFEST, bf_runtime_info)
    return imagej


def install_artifacts(destination, lockfile, verify):
    data = json.loads(lockfile.read_text())
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup = destination.with_name(destination.name + '.previous')
    with exclusive_file_lock(destination.with_name(destination.name + '.install.lock'), 300, lambda: None):
        # Recover an interrupted rename before doing network work.
        if backup.exists() and not destination.exists():
            backup.rename(destination)
        try:
            verify(destination)
        except RuntimeError:
            pass
        else:
            print(f'Runtime verified: {destination}')
            return destination
        with tempfile.TemporaryDirectory(prefix='imagej-install-', dir=destination.parent) as temporary:
            stage = Path(temporary) / 'imagej'
            (stage / 'jars').mkdir(parents=True)
            (stage / 'plugins').mkdir()

            def fetch(artifact):
                target = stage / artifact['path']
                cached = destination / artifact['path']
                if cached.is_file() and sha256(cached) == artifact['sha256']:
                    shutil.copyfile(cached, target)
                else:
                    request = urllib.request.Request(artifact['url'], headers={'User-Agent': 'Batch-Deconvolution-runtime-installer'})
                    with urllib.request.urlopen(request, timeout=90) as response, target.open('wb') as stream:
                        shutil.copyfileobj(response, stream)
                if sha256(target) != artifact['sha256']:
                    raise RuntimeError(f"Runtime download checksum mismatch: {artifact['path']}")

            with ThreadPoolExecutor(max_workers=6) as pool:
                list(pool.map(fetch, data['artifacts']))
            shutil.copyfile(lockfile, stage / 'manifest.json')
            verify(stage)
            if backup.exists():
                shutil.rmtree(backup)
            if destination.exists():
                destination.rename(backup)
            try:
                stage.rename(destination)
            except BaseException:
                if backup.exists():
                    backup.rename(destination)
                raise
            if backup.exists():
                shutil.rmtree(backup)
    print(f'Installed pinned runtime: {destination}')
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=RUNTIME)
    args = parser.parse_args()
    try:
        install(args.destination)
    except Exception as error:
        parser.exit(1, f'ImageJ setup failed: {error}\n')
