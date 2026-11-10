"""Durable, incremental batch provenance using only the standard library."""

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
from io import StringIO
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
import re
import runpy
import sys
import xml.etree.ElementTree as ET


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def xml_text(value):
    """Render XML 1.0-forbidden characters as readable Python-style escapes."""
    return re.sub(r'[^\x09\x0a\x0d\x20-\ud7ff\ue000-\ufffd\U00010000-\U0010ffff]',
                  lambda match: ascii(match[0])[1:-1], str(value))


def append_value(parent, name, value):
    node = ET.SubElement(parent, name)
    if isinstance(value, dict):
        for key, item in value.items():
            entry = ET.SubElement(node, 'entry', key=xml_text(key))
            append_value(entry, 'value', item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            append_value(node, 'item', item)
    elif value is None:
        node.set('null', 'true')
    else:
        node.text = xml_text(value)
    return node


class Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, text):
        self.log.write(text)
        self.log.flush()
        return self.stream.write(text)

    def flush(self):
        self.log.flush()
        self.stream.flush()


class BatchReport:
    def __init__(self, output_dir, settings):
        started = datetime.now().astimezone()
        name = started.strftime('batch deconvolution report %Y%m%d%H-%M%S')
        self.path = Path(output_dir).resolve() / f'{name}.xml'
        # Preserve earlier reports if two batches start within the same second.
        suffix = 1
        while True:
            try:
                self.path.touch(exist_ok=False)
                break
            except FileExistsError:
                suffix += 1
                self.path = Path(output_dir).resolve() / f'{name} ({suffix}).xml'
        self.log = StringIO()
        packages = {}
        for package in ('numpy', 'tifffile', 'h5py', 'bioio', 'bioio-bioformats',
                        'psfmodels', 'napari-vipp-imaris'):
            try:
                packages[package] = version(package)
            except PackageNotFoundError:
                packages[package] = 'unavailable'
        self.data = dict(started_utc=started.astimezone(timezone.utc).isoformat(),
                         started_local=started.isoformat(), status='running', settings=settings,
                         environment=dict(python=sys.version, platform=platform.platform(), packages=packages,
                                          application_version=runpy.run_path(str(Path(__file__).resolve().parent.parent / 'app_version.py'))['__version__']),
                         channel_index_convention='zero-based',
                         algorithm='ImageJ CLIJx DeconvolveRichardsonLucyFFT (RL-TV)',
                         input_conversion='float32; nonfinite and negative values set to zero; signed int8 reinterpreted as uint8',
                         psf_conversion='float32; nonfinite and negative values set to zero; normalized to unit sum',
                         output_conversion='uint16; nonfinite and negative values set to zero; round to nearest; clip at 65535',
                         console_log='', psfs=[], files=[])
        self.save()

    def save(self):
        self.data['console_log'] = self.log.getvalue()
        root = ET.Element('deconvolution_batch', schema_version='1')
        for key, value in self.data.items():
            append_value(root, key, value)
        ET.indent(root)
        temporary = self.path.with_suffix('.xml.tmp')
        ET.ElementTree(root).write(temporary, encoding='utf-8', xml_declaration=True)
        temporary.replace(self.path)

    def record_psfs(self, channels, psfs):
        for index, (channel, volume) in enumerate(zip(channels, psfs)):
            if volume is None:
                continue
            source = Path(channel['psf_file'])
            self.data['psfs'].append(dict(assignment_position=index, original_path=str(source.resolve()),
                                         sha256=sha256(source), shape_zyx=list(volume.shape),
                                         channel_metadata=channel))
        self.save()

    @contextmanager
    def capture(self):
        with redirect_stdout(Tee(sys.stdout, self.log)), redirect_stderr(Tee(sys.stderr, self.log)):
            try:
                yield
            finally:
                self.save()
