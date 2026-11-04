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


def run(connection, path):
    try:
        volumes, projection = load_volume(path)
        connection.send(("loaded", projection))
        while connection.recv() != "close":
            pass
    except (EOFError, BrokenPipeError):
        pass
    except Exception:
        connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()
