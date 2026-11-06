"""Session-only PSF cache shared by batch, preview, and PSF viewing."""
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp

from source.psf_channels import read_psf_channels


def memory_argument(gigabytes):
    value = float(gigabytes)
    if not math.isfinite(value) or value <= 0 or value * 1024 < 1:
        raise ValueError("FIJI max memory must be a positive number of GB (at least 1 MB).")
    return f"{round(value * 1024)}m"


def memory_gb(value):
    text = str(value).strip().lower()
    if text.endswith('g'):
        return float(text[:-1])
    if text.endswith('m'):
        return float(text[:-1]) / 1024
    return float(text)


class PSFSession:
    def __init__(self):
        self.directory = TemporaryDirectory(prefix="batch-psfs-")
        self.entries = {}

    @staticmethod
    def key(settings):
        return json.dumps(settings, sort_keys=True, separators=(',', ':'), allow_nan=False)

    def get(self, settings):
        key = self.key(settings)
        catalog = self.entries.get(key)
        if catalog and all(Path(entry['psf_file']).is_file() for entry in catalog):
            return catalog
        self.entries.pop(key, None)
        return None

    def worker_settings(self, settings):
        return dict(settings, module="generate_gibson_lanni_psfs",
                    OUTPUT_DIR=mkdtemp(prefix="generation-", dir=self.directory.name),
                    OUTPUT_IS_TEMPORARY=True, OVERWRITE=True)

    def publish(self, settings, folder):
        catalog = read_psf_channels(folder, validate=True)
        catalog.sort(key=lambda entry: entry['index'] if entry['index'] is not None else -1)
        definitions = sorted((dict(channel,index=channel.get('index',position))
                              for position,channel in enumerate(settings['CHANNELS'])),key=lambda channel:channel['index'])
        if [entry['index'] for entry in catalog] != [channel['index'] for channel in definitions]:
            raise ValueError("Generation did not produce all requested PSFs.")
        for entry, definition in zip(catalog, definitions):
            entry['name'] = definition['name']
        self.entries[self.key(settings)] = catalog
        return catalog

    def close(self):
        self.directory.cleanup()
