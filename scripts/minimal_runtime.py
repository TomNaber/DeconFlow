"""Build a relocatable Python runtime plus a linked Java 21 runtime.

Filtering affects the deliverable only, never the source development environment.
"""
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import zipfile

# Includes Java APIs used by ImageJ, Jython, Bio-Formats and JPype. Desktop is
# necessary even in headless mode (ImageIO/AWT); the compiler is not shipped.
JAVA_MODULES = (
    'java.desktop', 'java.logging', 'java.management', 'java.naming',
    'java.prefs', 'java.scripting', 'java.sql', 'java.xml', 'jdk.unsupported',
    'jdk.crypto.ec', 'jdk.zipfs',
)


def build_runtime(prefix, java_home, destination, stage, app, imagej):
    import conda_pack

    suffix = '.exe' if destination.suffix == '.zip' else ''
    jdk = prefix / java_home
    linked = stage / 'java-runtime'
    subprocess.run([str(jdk / 'bin' / ('jlink' + suffix)),
                    '--add-modules', ','.join(JAVA_MODULES), '--strip-debug',
                    '--no-header-files', '--no-man-pages', '--compress=zip-9',
                    '--output', str(linked)], check=True)
    # Java source-file launch requires a compiler. Compile once while building,
    # then run the class with the smaller runtime on the user's machine.
    classes = app / 'source/imagej'
    classpath = os.pathsep.join(str(p) for p in sorted(imagej.rglob('*.jar')))
    subprocess.run([str(jdk / 'bin' / ('javac' + suffix)), '--release', '21', '-proc:none',
                    '-cp', classpath, '-d', str(classes),
                    str(classes / 'HeadlessImageJ.java')], check=True)
    smoke = stage / 'verify-headless.py'
    smoke.write_text('print("DeconFlow headless runtime ready")\n')
    subprocess.run([str(linked / 'bin' / ('java' + suffix)),
                    '-Djava.awt.headless=true', '-cp', str(classes) + os.pathsep + classpath,
                    'HeadlessImageJ', str(smoke)], check=True, timeout=120)
    env = conda_pack.CondaEnv.from_prefix(str(prefix))
    before = sum(Path(f.source).stat().st_size for f in env.files if Path(f.source).is_file())
    filters = [
        java_home.as_posix() + '/*', 'include/*', 'Library/include/*',
        'share/man/*', 'share/doc/*', 'share/gtk-doc/*',
        '*.a', '*.la', '*.pdb', '*.pyc', '*/__pycache__/*',
        'lib/python*/test/*', 'Lib/test/*', '*/site-packages/*/tests/*',
        '*/site-packages/*/test/*', '*/site-packages/PyObjCTest/*',
        '*/site-packages/pip/*', '*/site-packages/conda_pack/*',
        '*/site-packages/wheel/*', 'bin/pip*', 'Scripts/pip*',
        'bin/conda-pack', 'Scripts/conda-pack*',
    ]
    for pattern in filters:
        env = env.exclude(pattern)
    # Preserve licenses, distribution metadata and all runtime libraries.
    after = sum(Path(f.source).stat().st_size for f in env.files if Path(f.source).is_file())
    if destination.suffix == '.zip':
        env.pack(output=str(destination), verbose=True, compress_level=9)
        with zipfile.ZipFile(destination, 'a', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for p in sorted(linked.rglob('*')):
                if p.is_file():
                    archive.write(p, (java_home / p.relative_to(linked)).as_posix())
    else:
        raw = stage / 'python-runtime.tar'
        env.pack(output=str(raw), format='tar', verbose=True)
        with tarfile.open(raw, 'a') as archive:
            archive.add(linked, arcname=java_home.as_posix())
        with raw.open('rb') as source, destination.open('wb') as target:
            with gzip.GzipFile(filename='', mode='wb', fileobj=target, compresslevel=9, mtime=0) as compressed:
                shutil.copyfileobj(source, compressed)
        raw.unlink()
    java_bytes = sum(p.stat().st_size for p in linked.rglob('*') if p.is_file())
    report = {'python_environment_before_bytes': before,
              'python_environment_filtered_bytes': after,
              'linked_java_bytes': java_bytes, 'java_modules': list(JAVA_MODULES),
              'runtime_archive_bytes': destination.stat().st_size, 'excluded_patterns': filters}
    (app / 'runtime/packaging.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
