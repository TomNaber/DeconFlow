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
from app_version import __version__, NATIVE_VERSION, BUILD_VERSION
from scripts.setup_imagej import install as install_imagej
from scripts.minimal_runtime import build_runtime
from source.imagej_runtime import manifest as imagej_manifest, java_executable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--platform', choices=('macos', 'windows'), required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'dist')
    parser.add_argument('--installer', action='store_true', help=argparse.SUPPRESS)  # Backward-compatible; always installer-only.
    args = parser.parse_args()
    expected = 'Darwin' if args.platform == 'macos' else 'Windows'
    if platform.system() != expected:
        parser.error(f'{args.platform} deliverables must be built on {expected}.')
    machine = platform.machine().lower()
    arch = {'amd64': 'x86_64', 'x86_64': 'x86_64', 'arm64': 'arm64'}.get(machine)
    if arch is None or (args.platform == 'windows' and arch != 'x86_64'):
        parser.error(f'Unsupported architecture: {machine}')
    prefix = Path(sys.prefix).resolve()
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
    imagej = install_imagej()  # Also installs Java into this prefix when missing.
    java = java_executable(allow_external=False)
    java_home = java.parent.parent.relative_to(prefix)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    name = f'DeconFlow-{__version__}-{args.platform}-{arch}'
    installer = output / (f'{name}.dmg' if args.platform == 'macos' else f'{name}-Setup.exe')
    if installer.exists() or installer.with_suffix(installer.suffix + '.sha256').exists():
        parser.error(f'Installer already exists: {installer}. Move it or use --output.')
    with tempfile.TemporaryDirectory(prefix='batch-release-') as staging:
        stage = Path(staging)
        bundle = stage / ('DeconFlow.app' if args.platform == 'macos' else name)
        resources = bundle / 'Contents/Resources' if args.platform == 'macos' else bundle
        app = resources / 'app'
        app.mkdir(parents=True)
        for filename in ('main.py', 'app_version.py', 'README.md', 'environment.yml', 'requirements.txt'):
            shutil.copy2(ROOT / filename, app / filename)
        shutil.copytree(ROOT / 'source', app / 'source',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.DS_Store'))
        shutil.copytree(ROOT / 'docs', app / 'docs',
                        ignore=shutil.ignore_patterns('assets', 'images', '.DS_Store'))
        shutil.copytree(imagej, app / 'runtime/imagej')
        shutil.copytree(imagej.parent / 'bioformats', app / 'runtime/bioformats')
        # Preserve native dependency notices without distributing unused wheels.
        shutil.copytree(ROOT / 'vendor', app / 'vendor',
                        ignore=shutil.ignore_patterns('*.whl', '__pycache__', '*.pyc', '.DS_Store'))
        runtime = resources / ('runtime.tar.gz' if args.platform == 'macos' else 'runtime.zip')
        build_runtime(prefix, java_home, runtime, stage, app, imagej)
        # Content-addressed caches allow the app itself to move after first launch.
        with runtime.open('rb') as stream:
            build_id = f'{__version__}-{arch}-{hashlib.file_digest(stream, "sha256").hexdigest()[:16]}'
        extension = 'sh' if args.platform == 'macos' else 'ps1'
        template = ROOT / 'scripts/packaging' / f'launch_{args.platform}.{extension}'
        launcher = template.read_text().replace('@BUILD_ID@', build_id)
        launcher = launcher.replace('@JAVA_HOME@', str(java_home))
        if args.platform == 'macos':
            executable = bundle / 'Contents/MacOS/DeconFlow'
            executable.parent.mkdir(parents=True)
            executable.write_text(launcher)
            executable.chmod(0o755)
            with (bundle / 'Contents/Info.plist').open('wb') as stream:
                plistlib.dump({
                    'CFBundleName': 'DeconFlow',
                    'CFBundleDisplayName': 'DeconFlow',
                    'CFBundleIdentifier': 'org.batch-deconvolution.app',
                    'CFBundleExecutable': executable.name,
                    'CFBundlePackageType': 'APPL',
                    'CFBundleShortVersionString': NATIVE_VERSION,
                    'CFBundleVersion': NATIVE_VERSION,
                    'LSMinimumSystemVersion': imagej_manifest()['minimum_macos'][arch],
                    'NSHighResolutionCapable': True,
                }, stream)
        else:
            (bundle / 'Launch.ps1').write_text(launcher, encoding='utf-8-sig')
            (bundle / 'Launch.cmd').write_bytes(
                b'@echo off\r\npowershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Launch.ps1" %*\r\n')
        shutil.copy2(ROOT / 'docs/deliverables.md', bundle / 'READ-ME.md')
        if args.platform == 'macos':
            disk = stage / 'disk'
            disk.mkdir()
            shutil.move(str(bundle), disk / bundle.name)
            (disk / 'Applications').symlink_to('/Applications')
            subprocess.run(['hdiutil', 'create', '-volname', f'DeconFlow {__version__}',
                            '-srcfolder', str(disk), '-format', 'UDZO', str(installer)], check=True)
        else:
            compiler = shutil.which('ISCC.exe')
            if compiler is None:
                compiler = str(Path('C:/Program Files (x86)/Inno Setup 6/ISCC.exe'))
            subprocess.run([compiler, f'/DAppVersion={__version__}',
                            f'/DNativeVersion={BUILD_VERSION}', f'/DSourceDir={bundle}',
                            f'/DOutputDir={output}', f'/DOutputName={installer.stem}',
                            str(ROOT / 'scripts/packaging/windows.iss')], check=True)
    with installer.open('rb') as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    installer.with_suffix(installer.suffix + '.sha256').write_text(f'{digest}  {installer.name}\n')
    print(f'Created {installer}\nSHA-256: {digest}')



if __name__ == '__main__':
    main()
