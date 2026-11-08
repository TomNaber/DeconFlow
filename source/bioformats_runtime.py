"""Use the bundled Bio-Formats reader without Maven resolution or Java downloads."""
from functools import cache
import json
import os
from pathlib import Path

if __package__:
    from .imagej_runtime import ROOT, sha256, java_executable
else:
    from imagej_runtime import ROOT, sha256, java_executable

MANIFEST = ROOT / 'source/bioformats/runtime.json'
RUNTIME = ROOT / 'runtime/bioformats'


def runtime_info(directory=RUNTIME):
    directory = Path(directory)
    data = json.loads(MANIFEST.read_text())
    try:
        if (directory / 'manifest.json').read_bytes() != MANIFEST.read_bytes():
            raise ValueError('manifest does not match this app')
        for artifact in data['artifacts']:
            if sha256(directory / artifact['path']) != artifact['sha256']:
                raise ValueError(f"checksum mismatch: {artifact['path']}")
    except (OSError, ValueError) as error:
        raise RuntimeError(f'Missing or corrupt bundled Bio-Formats: {error}. '
                           'Run python scripts/setup_imagej.py or reinstall the packaged app.') from error
    return data


@cache
def configure_bioformats():
    data = runtime_info()
    os.environ['JAVA_HOME'] = str(java_executable(allow_external=False).parent.parent)
    # Import first: bffile registers its normal online Maven endpoints at import time.
    import bffile._java_stuff
    import scyjava.config
    scyjava.config.set_java_constraints(version='21', fetch='never')
    scyjava.config.endpoints.clear()
    scyjava.config.add_classpath(*(str(RUNTIME / a['path']) for a in data['artifacts']))
