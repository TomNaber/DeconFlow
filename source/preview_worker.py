from contextlib import redirect_stdout, redirect_stderr
import os
import json
from pathlib import Path
import signal
import sys
import traceback
from time import perf_counter
import tifffile

import h5py
import numpy as np

from source.channel_metadata import read_channels, text


def display_range(volume):
    """Use the full integer range, or the data range for floating-point images."""
    if np.issubdtype(volume.dtype, np.integer):
        limits = np.iinfo(volume.dtype)
        return float(limits.min), float(limits.max)
    low, high = float(volume.min()), float(volume.max())
    if low == high:
        return (0.0, high) if high > 0 else (low, low + 1.0)
    return low, high


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
                "range": display_range(volume),
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


def channel_settings(settings, index):
    indices = settings.get("CHANNEL_INDICES", list(range(len(settings["PSF_FILES"]))))
    filename = dict(zip(indices, settings["PSF_FILES"]))[index]
    path = Path(settings["PSF_DIR"]).expanduser() / filename
    try:
        stat = path.stat()
        fingerprint = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    except OSError:
        fingerprint = (str(path), None)
    return {
        **{key: value for key, value in settings.items()
           if key not in ("TEMP_DIR", "CHANNEL_INDICES", "PSF_FILES", "FIJI_SESSION", "FORCE", "OUTPUT_CHANNEL_NAMES")},
        "PSF_FILE": filename,
        "PSF_FINGERPRINT": fingerprint,
    }


class PreviewCache:
    """One crop per source channel and one PSF/result per selected channel."""
    def __init__(self, directory, volumes, original):
        self.directory = Path(directory)
        self.volumes = {c["index"]: (v, c) for v, c in zip(volumes, original["channels"])}
        self.original = original
        self.inputs = {}
        self.psfs = {}
        self.results = {}

    def bounds(self, settings):
        bounds = settings.get("CROP_BOUNDS", {a: [0, n] for a, n in zip("zyx", self.original["shape"])})
        for axis, size in zip("zyx", self.original["shape"]):
            low, high = bounds[axis]
            if not isinstance(low, int) or not isinstance(high, int) or not 0 <= low < high <= size:
                raise ValueError(f"Invalid {axis.upper()} crop bounds.")
        return bounds

    def project_original(self, bounds, axes):
        """Limit each projection's viewing axis, retaining the plane for crop editing."""
        bounds = self.bounds({"CROP_BOUNDS": bounds})
        projections = []
        for index, (volume, _) in self.volumes.items():
            channel = {"index": index}
            if "xy" in axes:
                channel["xy"] = volume[slice(*bounds["z"]), :, :].max(axis=0)
            if "yz" in axes:
                channel["yz"] = volume[:, :, slice(*bounds["x"])].max(axis=2).T
            projections.append(channel)
        return projections

    def prepare(self, settings, stale=lambda: False):
        bounds = self.bounds(settings)
        crop_key = tuple(tuple(bounds[a]) for a in "zyx")
        selected = settings["CHANNEL_INDICES"]
        # Release deselected channels, keeping cache size independent of interaction history.
        for cache in (self.inputs, self.psfs):
            for index in list(cache):
                if index not in selected:
                    cache.pop(index)[1].unlink(missing_ok=True)
        for index in list(self.results):
            if index not in selected:
                del self.results[index]
        for index in selected:
            if stale():
                return
            volume, _ = self.volumes[index]
            if self.inputs.get(index, (None,))[0] != crop_key:
                image = np.array(volume[tuple(slice(*bounds[a]) for a in "zyx")],
                                 dtype=np.float32, copy=True, order="C")
                np.nan_to_num(image, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                np.maximum(image, 0.0, out=image)
                path = self.directory / f"input-{index}.tif"
                tifffile.imwrite(path, image, photometric="minisblack", metadata={"axes": "ZYX"}, bigtiff=True)
                self.inputs[index] = (crop_key, path)
                del image
            fingerprint = channel_settings(settings, index)["PSF_FINGERPRINT"]
            if self.psfs.get(index, (None,))[0] != fingerprint:
                psf = np.asarray(tifffile.imread(fingerprint[0]), dtype=np.float32).squeeze()
                np.nan_to_num(psf, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                np.maximum(psf, 0.0, out=psf)
                psf /= psf.sum(dtype=np.float64)
                path = self.directory / f"psf-{index}.tif"
                tifffile.imwrite(path, np.ascontiguousarray(psf), photometric="minisblack",
                                 metadata={"axes": "ZYX"}, bigtiff=True)
                self.psfs[index] = (fingerprint, path)


def deconvolve_volume(cache, settings, directory, emit, stale, initialized):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import batch_deconvolve_ims as batch

    for name in ("ITERATIONS", "RLTV_REGULARIZATION", "RLTV_NON_CIRCULANT", "FIJI_EXECUTABLE", "FIJI_MEMORY"):
        value = settings[name]
        if isinstance(getattr(batch, name), Path):
            value = Path(value).expanduser()
        setattr(batch, name, value)
    batch.TEMP_DIR = Path(directory)
    selected = settings["CHANNEL_INDICES"]
    if not selected or len(settings["PSF_FILES"]) != len(selected):
        raise ValueError("Select one PSF for each selected channel.")
    cache.prepare(settings, stale)
    from fiji_session import FijiClient
    if stale():
        return cache.original
    bounds = cache.bounds(settings)
    crop_xyz = [bounds[axis][1] - bounds[axis][0] for axis in "xyz"]
    geometries = []
    for index in selected:
        with tifffile.TiffFile(cache.psfs[index][1]) as psf:
            psf_xyz = list(reversed(psf.series[0].shape))
        geometries.append({"crop": crop_xyz, "psf": psf_xyz})
    geometries = FijiClient(settings["FIJI_SESSION"]).inspect_geometries(
        geometries, settings["RLTV_NON_CIRCULANT"])
    initialized()

    def dimensions(field):
        shapes = list(dict.fromkeys(tuple(item[field]) for item in geometries))
        return " or ".join(" x ".join(str(n) for n in shape) for shape in shapes)

    cache_status = "new" if any(item["cache"] == "new" for item in geometries) else "reused"
    print(f"Crop XYZ {dimensions('crop')}. PSF XYZ {dimensions('psf')}. "
          f"Padded FFT XYZ {dimensions('fft')}. FFT cache {cache_status}.", flush=True)
    projections = []
    public_settings = {key: value for key, value in settings.items() if key not in ("FIJI_SESSION", "FORCE")}
    for position, index in enumerate(selected, start=1):
        if stale():
            break
        _, channel = cache.volumes[index]
        output_name = settings.get("OUTPUT_CHANNEL_NAMES", {}).get(str(index), channel["name"])
        label = f"Preview C{position}/{len(selected)} {output_name}"
        processing = channel_settings(settings, index)
        processing["PSF_FINGERPRINT"] = cache.psfs[index][0]
        previous = cache.results.get(index)
        if not settings.get("FORCE", False) and previous is not None and previous["processing"] == processing:
            projection = previous
            print(f"{label} completed in 0.0s (cached).", flush=True)
        else:
            result = batch.run_fiji(None, None, label,
                                   session_directory=settings["FIJI_SESSION"],
                                   input_path=cache.inputs[index][1], psf_path=cache.psfs[index][1],
                                   report_geometry=False)
            shape = tuple(bounds[a][1] - bounds[a][0] for a in "zyx")
            if result.size != np.prod(shape):
                raise ValueError("Deconvolution returned an unexpected image size.")
            started = perf_counter()
            stored = batch.store_result(result.reshape(shape))
            projection = {**channel, "xy": stored.max(axis=0), "yz": stored.max(axis=2).T,
                          # Stretch each result channel to its brightest voxel.
                          "range": (0.0, max(1.0, float(stored.max()))),
                          "processing": processing}
            cache.results[index] = projection
            del result, stored
        projections.append(projection)
        if not stale():
            emit({**cache.original, "channels": list(projections), "settings": public_settings, "crop": bounds})
    return {**cache.original, "channels": projections, "settings": public_settings, "crop": bounds}


def run(connection, path, cache_directory):
    if os.name != "nt":
        os.setsid()

    def stop(signum, frame):
        raise SystemExit(130)

    signal.signal(signal.SIGTERM, stop)
    revision_path = Path(cache_directory) / "revision"

    def stale(revision):
        try:
            return json.loads(revision_path.read_text()) != str(revision)
        except OSError:
            return True

    try:
        started = perf_counter()
        volumes, projection = load_volume(path)
        cache = PreviewCache(cache_directory, volumes, projection)
        connection.send(("loaded", projection))
        while True:
            request = connection.recv()
            if request == "close":
                break
            if request == "clear_result":
                cache.results.clear()
                continue
            if request[0] == "project_original":
                _, bounds, axes = request
                connection.send(("original_projection", (bounds, axes, cache.project_original(bounds, axes))))
                continue
            if request[0] == "prepare":
                _, revision, settings = request
                if not stale(revision):
                    try:
                        with redirect_stdout(Log(connection)), redirect_stderr(Log(connection)):
                            cache.prepare(settings, lambda: stale(revision))
                    except Exception as error:
                        connection.send(("log", f"Preview preparation deferred: {error}\n"))
                continue
            revision, settings, directory = request
            try:
                with redirect_stdout(Log(connection)), redirect_stderr(Log(connection)):
                    result = deconvolve_volume(
                        cache, settings, directory,
                        lambda result: connection.send(("channel", (revision, result))),
                        lambda: stale(revision),
                        lambda: connection.send(("initialized", revision)),
                    )
                connection.send(("deconvolved", (revision, result)))
            except Exception:
                connection.send(("failed", (revision, traceback.format_exc())))
    except (EOFError, BrokenPipeError):
        pass
    except Exception:
        connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()
