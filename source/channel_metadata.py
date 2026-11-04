"""Read channel descriptions without loading image pixels or Bio-Formats."""

import h5py
import numpy as np


def text(value):
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"S", "a"}:
            return value.tobytes().decode("utf-8", errors="replace").rstrip("\x00")
        if value.ndim == 0:
            return text(value.item())
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").rstrip("\x00")
    return str(value).rstrip("\x00")


def read_channels(path):
    with h5py.File(path, "r") as handle:
        data = handle["DataSet/ResolutionLevel 0/TimePoint 0"]
        indices = sorted(int(name.split()[1]) for name in data if name.startswith("Channel "))
        channels = []
        for index in indices:
            section = handle.get(f"DataSetInfo/Channel {index}")
            attrs = section.attrs if section is not None else {}
            channels.append({
                "index": index,
                "name": text(attrs.get("Name", f"Channel {index + 1}")),
            })
        return channels
