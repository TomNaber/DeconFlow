"""Session-only PSF cache shared by batch, preview, and PSF viewing."""
import json
import math
import shutil
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
        self.scope = None

    def set_scope(self, path):
        scope = str(Path(path).resolve()) if path is not None else None
        if scope == self.scope:
            return
        for _, entry in self.entries.values():
            shutil.rmtree(Path(entry['psf_file']).parent)
        self.entries.clear()
        self.scope = scope

    @staticmethod
    def key(settings, channel):
        # Channel selection and display names do not alter the PSF's voxels.
        numerical = {key: value for key, value in settings.items() if key != 'CHANNELS'}
        numerical['CHANNEL'] = {key: value for key, value in channel.items() if key != 'name'}
        return json.dumps(numerical, sort_keys=True, separators=(',', ':'), allow_nan=False)

    @staticmethod
    def definitions(settings):
        return sorted((dict(channel, index=channel.get('index', position))
                       for position, channel in enumerate(settings['CHANNELS'])),
                      key=lambda channel: channel['index'])

    def cached(self, settings, channel):
        previous = self.entries.get(channel['index'])
        if previous is not None:
            key, entry = previous
            if key == self.key(settings, channel) and Path(entry['psf_file']).is_file():
                return dict(entry, name=channel['name'])
        return None

    def get(self, settings):
        catalog = [self.cached(settings, channel) for channel in self.definitions(settings)]
        return catalog if catalog and all(catalog) else None

    def worker_settings(self, settings):
        missing = [channel for channel in self.definitions(settings)
                   if self.cached(settings, channel) is None]
        return dict(settings, CHANNELS=missing, module="generate_vectorial_psfs",
                    OUTPUT_DIR=mkdtemp(prefix="generation-", dir=self.directory.name),
                    OUTPUT_IS_TEMPORARY=True, OVERWRITE=True)

    def publish(self, settings, folder, generated=None):
        catalog = read_psf_channels(folder, validate=True)
        catalog.sort(key=lambda entry: entry['index'] if entry['index'] is not None else -1)
        definitions = self.definitions(generated if generated is not None else settings)
        if [entry['index'] for entry in catalog] != [channel['index'] for channel in definitions]:
            raise ValueError("Generation did not produce all requested PSFs.")
        for entry, definition in zip(catalog, definitions):
            index = definition['index']
            channel_dir = Path(self.directory.name) / f'channel-{index}'
            channel_dir.mkdir(exist_ok=True)
            destination = channel_dir / Path(entry['psf_file']).name
            Path(entry['psf_file']).replace(destination)
            previous = self.entries.get(index)
            if previous is not None and Path(previous[1]['psf_file']) != destination:
                Path(previous[1]['psf_file']).unlink(missing_ok=True)
            entry = dict(entry, name=definition['name'], psf_file=str(destination))
            self.entries[index] = (self.key(settings, definition), entry)
        result = self.get(settings)
        if result is None:
            raise ValueError("Generated and cached PSFs do not cover the requested channels.")
        return result

    def close(self):
        self.entries.clear()
        self.directory.cleanup()
