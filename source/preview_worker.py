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
from source.psf_channels import load_psf_volume
from source.console_progress import ProgressLine


def display_range(volume):
    """Use 0–255 for 8-bit inputs and the image data range otherwise."""
    dtype = volume.dtype
    if np.issubdtype(dtype, np.integer) and dtype.itemsize == 1:
        return 0.0, 255.0
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
                "yz": volume.max(axis=2),
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
    fingerprint = None
    if filename is not None:
        path = Path(filename).expanduser()
        if not path.is_absolute():
            path = Path(settings.get("PSF_DIR", ".")).expanduser() / path
        try:
            stat = path.stat()
            fingerprint = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        except OSError:
            fingerprint = (str(path), None)
    return {
        **{key: value for key, value in settings.items()
           if key not in ("TEMP_DIR", "CHANNEL_INDICES", "PSF_FILES", "FIJI_SESSION", "FORCE", "OUTPUT_CHANNEL_NAMES", "CHANNEL_ITERATIONS", "CHANNEL_REGULARIZATION", "CHANNEL_POST_PROCESSING")},
        "RLTV_REGULARIZATION": settings.get("CHANNEL_REGULARIZATION", {}).get(str(index), settings.get("RLTV_REGULARIZATION", .0001)),
        "ITERATIONS": settings.get("CHANNEL_ITERATIONS", {}).get(str(index), settings.get("ITERATIONS", 40)),
        "OUTPUT_CHANNEL_NAME": settings.get("OUTPUT_CHANNEL_NAMES", {}).get(str(index)),
        "CHANNEL_POST_PROCESSING": settings.get("CHANNEL_POST_PROCESSING", {}).get(str(index), {}),
        "PSF_FILE": filename,
        "PSF_FINGERPRINT": fingerprint,
    }


def deconvolution_key(processing):
    # Only numerical inputs belong in the key; runtime/UI settings do not.
    keys = ("CROP_BOUNDS", "PSF_FINGERPRINT")
    if processing["PSF_FILE"] is not None:
        keys += ("ITERATIONS", "RLTV_REGULARIZATION", "RLTV_NON_CIRCULANT")
    return {key: processing.get(key) for key in keys}


def processed_key(processing):
    post = processing.get("CHANNEL_POST_PROCESSING", {})
    if not processing.get("POST_PROCESSING", False) or post.get("filter", "None") == "None":
        post = {}
    return {**deconvolution_key(processing), "post": post}


class PreviewCache:
    """Bounded per-channel caches for inputs, PSFs, deconvolution and final previews."""
    def __init__(self, directory, volumes, original):
        self.directory = Path(directory)
        self.volumes = {c["index"]: (v, c) for v, c in zip(volumes, original["channels"])}
        self.original = original
        self.inputs = {}
        self.psfs = {}
        self.results = {}
        self.deconvolved = {}

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
                channel["yz"] = volume[:, :, slice(*bounds["x"])].max(axis=2)
            projections.append(channel)
        return projections

    def prepare(self, settings, stale=lambda: False):
        bounds = self.bounds(settings)
        crop_key = tuple(tuple(bounds[a]) for a in "zyx")
        selected = settings["CHANNEL_INDICES"]
        # One slot per loaded channel, retained even while that channel is disabled.
        # The worker and its entire cache are discarded when the stack is unloaded.
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
            if fingerprint is None:
                previous = self.psfs.pop(index, None)
                if previous is not None:
                    previous[1].unlink(missing_ok=True)
            elif self.psfs.get(index, (None,))[0] != fingerprint:
                psf = load_psf_volume(fingerprint[0])
                path = self.directory / f"psf-{index}.tif"
                tifffile.imwrite(path, np.ascontiguousarray(psf), photometric="minisblack",
                                 metadata={"axes": "ZYX"}, bigtiff=True)
                self.psfs[index] = (fingerprint, path)


def deconvolve_volume(cache, settings, directory, stale, initialized):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import batch_deconvolve_ims as batch

    for name in ("ITERATIONS", "RLTV_REGULARIZATION", "RLTV_NON_CIRCULANT", "FIJI_MEMORY"):
        value = settings.get(name, 40) if name == "ITERATIONS" else settings[name]
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
    processing_by_channel = {index: channel_settings(settings, index) for index in selected}
    missing = [index for index in selected
               if settings.get("FORCE", False) or index not in cache.deconvolved
               or cache.deconvolved[index][0] != deconvolution_key(processing_by_channel[index])]
    deconvolving = [index for index in missing if processing_by_channel[index]["PSF_FILE"] is not None]
    if deconvolving:
        crop_xyz = [bounds[axis][1] - bounds[axis][0] for axis in "xyz"]
        geometries = []
        for index in deconvolving:
            with tifffile.TiffFile(cache.psfs[index][1]) as psf:
                psf_xyz = list(reversed(psf.series[0].shape))
            geometries.append({"crop": crop_xyz, "psf": psf_xyz})
        geometries = FijiClient(settings["FIJI_SESSION"]).inspect_geometries(
            geometries, settings["RLTV_NON_CIRCULANT"])

        def dimensions(field):
            shapes = list(dict.fromkeys(tuple(item[field]) for item in geometries))
            return " or ".join(" x ".join(str(n) for n in shape) for shape in shapes)

        if any(item["cache"] == "new" for item in geometries):
            print(f"\nGenerating FFT cache. Crop XYZ {dimensions('crop')}. "
                  f"PSF XYZ {dimensions('psf')}. Padded FFT XYZ {dimensions('fft')}.", flush=True)
    initialized()
    projections = []
    public_settings = {key: value for key, value in settings.items() if key not in ("FIJI_SESSION", "FORCE")}
    for position, index in enumerate(selected, start=1):
        if stale():
            break
        volume, channel = cache.volumes[index]
        output_name = settings.get("OUTPUT_CHANNEL_NAMES", {}).get(str(index), channel["name"])
        label = f"Preview C{position}/{len(selected)} {output_name}"
        channel_started = perf_counter()
        timings = {}
        processing = processing_by_channel[index]
        previous = cache.results.get(index)
        if not settings.get("FORCE", False) and previous is not None and processed_key(previous["processing"]) == processed_key(processing):
            projection = {**previous, "processing": processing}
            cache.results[index] = projection
            print(f"{label}: reusing cached preview.", flush=True)
        else:
            if index in missing and processing["PSF_FILE"] is None:
                result = np.array(volume[tuple(slice(*bounds[a]) for a in "zyx")], dtype=np.float32, copy=True)
                cache.deconvolved[index] = (deconvolution_key(processing), result)
                cache.results.pop(index, None)
            elif index in missing:
                result = batch.run_fiji(None, None, label,
                                       session_directory=settings["FIJI_SESSION"],
                                       input_path=cache.inputs[index][1], psf_path=cache.psfs[index][1],
                                       timings=timings, total_started=channel_started,
                                       report_geometry=False, iterations=processing["ITERATIONS"],
                                       regularization=processing["RLTV_REGULARIZATION"])
                shape = tuple(bounds[a][1] - bounds[a][0] for a in "zyx")
                if result.size != np.prod(shape):
                    raise ValueError("Deconvolution returned an unexpected image size.")
                cache.deconvolved[index] = (deconvolution_key(processing), result.reshape(shape))
                cache.results.pop(index, None)
            else:
                print(f"{label}: updating post-processing (cached deconvolution).", flush=True)
            # Quantization sanitizes in place; preserve the unfiltered floating-point cache.
            post = processing["CHANNEL_POST_PROCESSING"]
            activity = f"Post-processing ({post['filter']})" if settings.get('POST_PROCESSING', False) and post.get('filter', 'None') != 'None' else 'Copying result'
            with ProgressLine(label, activity, delay=0 if activity.startswith('Post-processing') else 1,
                              timings=timings, total_started=channel_started).running():
                result = cache.deconvolved[index][1].copy()
                if settings.get('POST_PROCESSING', False):
                    result = batch.post_process(result, post, spacing_zyx=cache.original["spacing"])
            with ProgressLine(label, 'Converting output and building preview projections',
                              timings=timings, total_started=channel_started).running():
                stored = batch.store_result(result)
                projection = {**channel, "xy": stored.max(axis=0), "yz": stored.max(axis=2),
                              "range": display_range(stored), "processing": processing}
                cache.results[index] = projection
                del result, stored
            elapsed = perf_counter() - channel_started
            print(f'{label} channel completed in {elapsed:.1f}s total.', flush=True)
        projections.append({**projection, "name": output_name})
    return {**cache.original, "channels": projections, "settings": public_settings, "crop": bounds}


def import_batch_sample(directory, manifest, cache_directory, previous):
    """Build a replacement cache off to the side; never damage the displayed one."""
    path = manifest["path"]
    if previous is not None and previous.original["path"] == path:
        volumes = [v for v, _ in previous.volumes.values()]
        original = previous.original
    else:
        volumes, original = load_volume(path)
    cache = PreviewCache(cache_directory, volumes, original)
    if manifest["kind"] == "original":
        return cache, None
    settings = dict(manifest["settings"])
    settings["CROP_BOUNDS"] = {a: [0, n] for a, n in zip("zyx", original["shape"])}
    selected = settings["CHANNEL_INDICES"]
    if selected is None:
        selected = list(cache.volumes)
        settings["CHANNEL_INDICES"] = selected
    settings["OUTPUT_CHANNEL_NAMES"] = {
        str(i): settings.get("CHANNEL_EDITS", {}).get(str(i), {}).get("name", c["name"])
        for i, (_, c) in cache.volumes.items()}
    projections = []
    for index in cache.volumes:
        channel_config = settings
        if index not in selected:
            channel_config = {**settings, "CHANNEL_INDICES": [index], "PSF_FILES": [None],
                              "CHANNEL_POST_PROCESSING": {}}
        processing = channel_settings(channel_config, index)
        raw = np.load(Path(directory) / "sample" / f"raw-{index}.npy", allow_pickle=False)
        if raw.shape != tuple(original["shape"]) or raw.dtype != np.float32:
            raise ValueError("Batch preview volume does not match the original image")
        cache.deconvolved[index] = (deconvolution_key(processing), raw)
        channel = cache.volumes[index][1]
        with np.load(Path(directory) / "sample" / f"projection-{index}.npz", allow_pickle=False) as data:
            projection = {**channel, "xy": data["xy"], "yz": data["yz"],
                          "range": tuple(data["range"]), "processing": processing}
        cache.results[index] = projection
        projections.append({**projection, "name": settings["OUTPUT_CHANNEL_NAMES"][str(index)]})
    return cache, {**original, "batch_result": True, "channels": projections, "settings": settings,
                   "crop": settings["CROP_BOUNDS"]}


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
        cache = None
        if path is not None:
            volumes, projection = load_volume(path)
            cache = PreviewCache(cache_directory, volumes, projection)
            connection.send(("loaded", projection))
        while True:
            request = connection.recv()
            if request == "close":
                break
            if request[0] == "batch_import":
                _, directory, manifest = request
                try:
                    replacement, result = import_batch_sample(directory, manifest, cache_directory, cache)
                    for temporary in Path(cache_directory).glob("*.tif"):
                        temporary.unlink(missing_ok=True)
                    cache = replacement
                    connection.send(("batch_imported", (cache.original, result)))
                except Exception:
                    connection.send(("batch_failed", traceback.format_exc()))
                continue
            if cache is None:
                continue
            if request == "clear_result":
                cache.results.clear()
                cache.deconvolved.clear()
                continue
            if request[0] == "project_original":
                _, bounds, axes = request
                with redirect_stdout(Log(connection)), redirect_stderr(Log(connection)):
                    with ProgressLine('Preview', 'Building original projections', delay=1).running():
                        projections = cache.project_original(bounds, axes)
                connection.send(("original_projection", (bounds, axes, projections)))
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
