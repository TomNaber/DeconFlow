from contextlib import redirect_stdout, redirect_stderr
import os
from pathlib import Path
import signal
import sys
import traceback

import h5py
import numpy as np

from source.channel_metadata import read_channels, text


def load_volume(path):
    channels = read_channels(path)
    if not channels:
        raise ValueError("No image channels found.")
    volumes = []
    projections = []
    with h5py.File(path, "r") as handle:
        image = handle["DataSetInfo/Image"].attrs
        shape = tuple(int(text(image[axis])) for axis in "ZYX")
        if any(size <= 0 for size in shape):
            raise ValueError("Invalid image dimensions.")
        spacing = tuple(
            (float(text(image[f"ExtMax{axis}"])) - float(text(image[f"ExtMin{axis}"]))) / size
            for axis, size in zip((2, 1, 0), shape)
        )
        if not all(np.isfinite(value) and value > 0 for value in spacing):
            raise ValueError("Invalid image voxel spacing.")
        for channel in channels:
            index = channel["index"]
            dataset = handle[f"DataSet/ResolutionLevel 0/TimePoint 0/Channel {index}/Data"]
            if dataset.ndim != 3 or any(actual < required for actual, required in zip(dataset.shape, shape)):
                raise ValueError(f"Invalid image dimensions for C{index + 1}.")
            volume = dataset[tuple(slice(0, size) for size in shape)]
            if volume.dtype == np.int8:
                volume = volume.view(np.uint8)
            if np.issubdtype(volume.dtype, np.floating):
                np.nan_to_num(volume, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            attrs = handle[f"DataSetInfo/Channel {index}"].attrs
            try:
                limits = tuple(float(value) for value in text(attrs.get("ColorRange", "")).split())
                if len(limits) != 2 or not all(np.isfinite(limits)) or limits[1] <= limits[0]:
                    raise ValueError()
            except ValueError:
                low, high = float(volume.min()), float(volume.max())
                limits = (low, high if high > low else low + 1)
            try:
                color = tuple(float(value) for value in text(attrs.get("Color", "1 1 1")).split())
                if len(color) != 3 or not all(np.isfinite(color)):
                    raise ValueError()
                color = tuple(np.clip(color, 0, 1))
            except ValueError:
                color = (1.0, 1.0, 1.0)
            volumes.append(volume)
            projections.append({
                **channel,
                "xy": volume.max(axis=0),
                "yz": volume.max(axis=2).T,
                "range": limits,
                "color": color,
            })
    return volumes, {"path": str(path), "shape": shape, "spacing": spacing, "channels": projections}


class Log:
    def __init__(self, connection):
        self.connection = connection

    def write(self, value):
        if value:
            self.connection.send(("log", value))
        return len(value)

    def flush(self):
        pass


def deconvolve_volume(volumes, original, settings, directory):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import batch_deconvolve_ims as batch

    for name, value in settings.items():
        if isinstance(getattr(batch, name), Path):
            value = Path(value).expanduser()
        setattr(batch, name, value)
    batch.TEMP_DIR = Path(directory)
    if not batch.FIJI_EXECUTABLE.is_file():
        raise FileNotFoundError(batch.FIJI_EXECUTABLE)
    channels, psfs = batch.load_psfs()
    if len(psfs) != len(volumes):
        raise ValueError("Select one PSF for each loaded channel.")
    output = []
    projections = []
    for volume, channel, psf in zip(volumes, original["channels"], psfs):
        image = np.array(volume, dtype=np.float32, copy=True, order="C")
        np.nan_to_num(image, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.maximum(image, 0.0, out=image)
        result = batch.run_fiji(image, psf, f"Preview C{channel['index'] + 1} {channel['name']}")
        if result.size != volume.size:
            raise ValueError("Deconvolution returned an unexpected image size.")
        stored = batch.store_result(result.reshape(volume.shape))
        output.append(stored)
        projections.append({**channel, "xy": stored.max(axis=0), "yz": stored.max(axis=2).T})
        del image, result
    return output, {**original, "channels": projections, "settings": settings}


def run(connection, path):
    os.setsid()

    def stop(signum, frame):
        raise SystemExit(130)

    signal.signal(signal.SIGTERM, stop)
    try:
        volumes, projection = load_volume(path)
        connection.send(("loaded", projection))
        output = None
        while True:
            request = connection.recv()
            if request == "close":
                break
            revision, settings, directory = request
            try:
                output = None
                with redirect_stdout(Log(connection)), redirect_stderr(Log(connection)):
                    output, result = deconvolve_volume(volumes, projection, settings, directory)
                connection.send(("deconvolved", (revision, result)))
                del result
            except Exception:
                connection.send(("failed", (revision, traceback.format_exc())))
    except (EOFError, BrokenPipeError):
        pass
    except Exception:
        connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()
