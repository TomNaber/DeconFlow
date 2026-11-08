"""Shared native builder; invoked by build_macos.sh or build_windows.ps1."""

import argparse
import hashlib
from pathlib import Path
import platform
import plistlib
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app_version import __version__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--platform', choices=('macos', 'windows'), required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'dist')
    args = parser.parse_args()
    expected = 'Darwin' if args.platform == 'macos' else 'Windows'
    if platform.system() != expected:
        parser.error(f'{args.platform} deliverables must be built on {expected}.')
    machine = platform.machine().lower()
    arch = {'amd64': 'x86_64', 'x86_64': 'x86_64', 'arm64': 'arm64'}.get(machine)
    if arch is None or (args.platform == 'windows' and arch != 'x86_64'):
        parser.error(f'Unsupported architecture: {machine}')
    prefix = Path(sys.prefix)
    if sys.version_info[:2] != (3, 13) or not (prefix / 'conda-meta').is_dir():
        parser.error('Activate the Python 3.13 environment created from environment.yml first.')
    try:
        import conda_pack
    except ImportError:
        parser.error('Install the build dependency: python -m pip install -r requirements-build.txt')
    subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
    subprocess.run([sys.executable, '-c',
                    'import sys; sys.path.insert(0, "source"); '
                    'import tkinter, psfmodels, napari_vipp_imaris; '
                    'import source.preview, source.batch_deconvolve_ims; '
                    'import source.generate_gibson_lanni_psfs'], cwd=ROOT, check=True)
    java_name = 'java.exe' if args.platform == 'windows' else 'java'
    java_candidates = sorted(prefix.rglob(java_name))
    java = next((p for p in java_candidates if p.parent.name == 'bin' and p.is_file()), None)
    if java is None:
        parser.error('OpenJDK is missing; recreate the environment from environment.yml.')
    subprocess.run([str(java), '-version'], check=True)
    java_home = java.parent.parent.relative_to(prefix)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    name = f'Batch-Deconvolution-{__version__}-{args.platform}-{arch}'
    archive = output / f'{name}.zip'
    checksum = output / f'{name}.zip.sha256'
    if archive.exists() or checksum.exists():
        parser.error(f'Release output already exists: {archive}. Move it or use --output.')
    with tempfile.TemporaryDirectory(prefix='batch-release-') as staging:
        stage = Path(staging)
        bundle = stage / ('Batch Deconvolution.app' if args.platform == 'macos' else name)
        resources = bundle / 'Contents/Resources' if args.platform == 'macos' else bundle
        app = resources / 'app'
        app.mkdir(parents=True)
        for filename in ('main.py', 'app_version.py', 'README.md', 'environment.yml', 'requirements.txt'):
            shutil.copy2(ROOT / filename, app / filename)
        shutil.copytree(ROOT / 'source', app / 'source',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.DS_Store'))
        shutil.copytree(ROOT / 'docs', app / 'docs')
        # Preserve native dependency notices without distributing unused wheels.
        shutil.copytree(ROOT / 'vendor', app / 'vendor',
                        ignore=shutil.ignore_patterns('*.whl', '__pycache__', '*.pyc', '.DS_Store'))
        runtime = resources / ('runtime.tar.gz' if args.platform == 'macos' else 'runtime.zip')
        conda_pack.pack(prefix=str(prefix), output=str(runtime), verbose=True)
        # Content-addressed caches allow the app itself to move after first launch.
        with runtime.open('rb') as stream:
            build_id = f'{__version__}-{arch}-{hashlib.file_digest(stream, "sha256").hexdigest()[:16]}'
        extension = 'sh' if args.platform == 'macos' else 'ps1'
        template = ROOT / 'scripts/packaging' / f'launch_{args.platform}.{extension}'
        launcher = template.read_text().replace('@BUILD_ID@', build_id)
        launcher = launcher.replace('@JAVA_HOME@', str(java_home))
        if args.platform == 'macos':
            executable = bundle / 'Contents/MacOS/Batch Deconvolution'
            executable.parent.mkdir(parents=True)
            executable.write_text(launcher)
            executable.chmod(0o755)
            with (bundle / 'Contents/Info.plist').open('wb') as stream:
                plistlib.dump({
                    'CFBundleName': 'Batch Deconvolution',
                    'CFBundleDisplayName': 'Batch Deconvolution',
                    'CFBundleIdentifier': 'org.batch-deconvolution.app',
                    'CFBundleExecutable': executable.name,
                    'CFBundlePackageType': 'APPL',
                    'CFBundleShortVersionString': __version__,
                    'CFBundleVersion': __version__,
                    'LSMinimumSystemVersion': '12.0',
                    'NSHighResolutionCapable': True,
                }, stream)
        else:
            (bundle / 'Launch.ps1').write_text(launcher, encoding='utf-8-sig')
            (bundle / 'Launch.cmd').write_bytes(
                b'@echo off\r\npowershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Launch.ps1" %*\r\n')
        shutil.copy2(ROOT / 'docs/deliverables.md', bundle / 'READ-ME.md')
        temporary_archive = stage / 'release.zip'
        if args.platform == 'macos':
            subprocess.run(['ditto', '-c', '-k', '--sequesterRsrc', '--keepParent',
                            str(bundle), str(temporary_archive)], check=True)
        else:
            shutil.make_archive(str(stage / 'release'), 'zip', stage, bundle.name)
        shutil.move(str(temporary_archive), archive)
    with archive.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    checksum.write_text(f'{digest}  {archive.name}\n')
    print(f'Created {archive}\nSHA-256: {digest}')


if __name__ == '__main__':
    main()
