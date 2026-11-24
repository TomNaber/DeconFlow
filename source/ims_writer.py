#!/usr/bin/env python3

import re
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path

import h5py
import numpy as np
from napari_vipp_imaris import PyImarisWriter as writer


AXES = ("t", "c", "z", "y", "x")
MAX_BLOCK = {"t": 1, "c": 1, "z": 16, "y": 256, "x": 256}
DISPLAY_PARAMETERS = {
    "Color",
    "ColorMode",
    "ColorOpacity",
    "ColorRange",
    "ColorTable",
    "ColorTableLength",
    "GammaCorrection",
}


def _text(value):
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"S", "a"}:
            return value.tobytes().decode("utf-8", errors="replace").rstrip("\x00")
        if value.ndim == 0:
            return _text(value.item())
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").rstrip("\x00")
    return str(value).rstrip("\x00")


def _read_metadata(path):
    sections = {}
    with h5py.File(path, "r") as handle:
        for section_name, section in handle["DataSetInfo"].items():
            if isinstance(section, h5py.Group):
                sections[str(section_name)] = {
                    str(name): _text(value) for name, value in section.attrs.items()
                }
    return sections


def _number(value):
    return format(float(value), ".15g")


def _parameter_sections(
    source,
    sizes,
    channel_names,
    excitation_wavelengths_nm,
    emission_wavelengths_nm,
    dtype,
):
    sections = {
        name: dict(values)
        for name, values in source.items()
        if name != "TimeInfo" and re.fullmatch(r"Channel\s+\d+", name) is None
    }
    for index in range(sizes["c"]):
        values = {
            name: value
            for name, value in source.get(f"Channel {index}", {}).items()
            if name not in DISPLAY_PARAMETERS
        }
        values["Name"] = str(channel_names[index])
        if excitation_wavelengths_nm[index] is not None:
            values["LSMExcitationWavelength"] = _number(
                excitation_wavelengths_nm[index]
            )
        if emission_wavelengths_nm[index] is not None:
            values["LSMEmissionWavelength"] = _number(emission_wavelengths_nm[index])
        sections[f"Channel {index}"] = values

    image = sections.setdefault("Image", {})
    image["NumberOfChannels"] = str(sizes["c"])
    image["Unit"] = "um"

    processing = sections.setdefault("RedLionfish", {})
    processing.update(
        {
            "ApplicationName": "Fiji CLIJ2-FFT RL-TV batch",
            "ApplicationVersion": "1.4",
            "DataType": dtype,
            "AxisOrder": "TCZYX",
        }
    )
    return sections


def _parameters(sections):
    parameters = writer.Parameters()
    for section, values in sections.items():
        for name, value in values.items():
            parameters.set_value(str(section), str(name), str(value))
    return parameters


def _extents(source):
    image = source["Image"]
    names = ("ExtMin0", "ExtMin1", "ExtMin2", "ExtMax0", "ExtMax1", "ExtMax2")
    return writer.ImageExtents(*(float(image[name]) for name in names))


def _times(source, count):
    time_info = source.get("TimeInfo", {})
    fallback = time_info.get("TimePoint1") or source["Image"].get("RecordingDate", "1970-01-01")
    return [
        datetime.fromisoformat(
            time_info.get(f"TimePoint{index + 1}", fallback).replace("Z", "+00:00")
        ).replace(tzinfo=None)
        for index in range(count)
    ]


def _colors(source, channel_ranges, channel_colors=None):
    colors = []
    for index, channel_range in enumerate(channel_ranges):
        values = source[f"Channel {index}"]
        override = channel_colors[index] if channel_colors is not None else None
        red, green, blue = override if override is not None else (
            float(value) for value in values["Color"].split()[:3]
        )
        info = writer.ColorInfo()
        info.set_base_color(writer.Color(red, green, blue, 1.0))
        info.mOpacity = float(values.get("ColorOpacity", 1.0))
        info.mGammaCorrection = float(values.get("GammaCorrection", 1.0))
        info.mRangeMin, info.mRangeMax = map(float, channel_range)
        colors.append(info)
    return colors


def _block(data, starts, sizes, blocks):
    valid = {key: min(blocks[key], sizes[key] - starts[key]) for key in AXES}
    slices = tuple(slice(starts[key], starts[key] + valid[key]) for key in AXES)
    chunk = np.asarray(data[slices])
    padded = np.zeros(tuple(blocks[key] for key in AXES), dtype=data.dtype)
    padded[tuple(slice(0, valid[key]) for key in AXES)] = chunk
    return np.ascontiguousarray(padded)


def write_imaris(
    data,
    path,
    *,
    channel_names,
    channel_ranges,
    excitation_wavelengths_nm,
    emission_wavelengths_nm,
    source_ims=None,
    scene=None,
    channel_colors=None,
):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sizes = dict(zip(AXES, map(int, data.shape), strict=True))
    blocks = {key: min(sizes[key], MAX_BLOCK[key]) for key in AXES}
    if source_ims is not None:
        source = _read_metadata(source_ims)
    else:
        spacing = dict(zip("zyx", scene.spacing))
        origin = dict(zip("zyx", scene.origin))
        # The native API requires extents and absolute timestamps.
        date = datetime.fromisoformat(scene.acquired) if scene.acquired else datetime(1970,1,1)
        image = {"RecordingDate": date.isoformat()}
        for i, axis in enumerate("xyz"):
            image[f"ExtMin{i}"] = _number(origin[axis])
            image[f"ExtMax{i}"] = _number(origin[axis] + sizes[axis] * (spacing[axis] or 1.0))
        source = {"Image": image, "TimeInfo": {
            f"TimePoint{i+1}": (date + timedelta(seconds=elapsed)).isoformat()
            for i, elapsed in enumerate(scene.times)}}
        for i, ch in enumerate(scene.channels):
            source[f"Channel {i}"] = {"Color": " ".join(map(str,ch["color"]))}
    sections = _parameter_sections(
        source,
        sizes,
        channel_names,
        excitation_wavelengths_nm,
        emission_wavelengths_nm,
        data.dtype.name,
    )

    class Progress(writer.CallbackClass):
        def RecordProgress(self, progress, total_bytes_written):
            print(
                f"\r    IMS export: {100 * float(progress):5.1f}% "
                f"({int(total_bytes_written) / 1024**2:.1f} MiB written)",
                end="",
                flush=True,
            )

    converter = writer.ImageConverter(
        data.dtype.name,
        writer.ImageSize(**sizes),
        writer.ImageSize(**dict.fromkeys(AXES, 1)),
        writer.DimensionSequence("x", "y", "z", "c", "t"),
        writer.ImageSize(**blocks),
        str(output_path),
        writer.Options(),
        "Fiji CLIJ2-FFT RL-TV batch",
        "1.4",
        Progress(),
    )
    try:
        ranges = [range(0, sizes[key], blocks[key]) for key in AXES]
        for offsets in product(*ranges):
            starts = dict(zip(AXES, offsets, strict=True))
            block_index = writer.ImageSize(
                **{key: starts[key] // blocks[key] for key in AXES}
            )
            if converter.NeedCopyBlock(block_index):
                converter.CopyBlock(_block(data, starts, sizes, blocks), block_index)

        converter.Finish(
            _extents(source),
            _parameters(sections),
            _times(source, sizes["t"]),
            _colors(source, channel_ranges, channel_colors),
            True,
        )
        print()
    finally:
        converter.Destroy()
    # The SDK replaces RecordingDate with the first frame timestamp in Finish.
    # Retain the acquisition origin for time series that start at nonzero DeltaT.
    recording_date = source["Image"].get("RecordingDate")
    if recording_date:
        from source.edit_ims_channels import _set_text_attribute
        with h5py.File(output_path, "r+") as handle:
            image_info = handle["DataSetInfo/Image"]
            if _text(image_info.attrs.get("RecordingDate")) != recording_date:
                _set_text_attribute(image_info, "RecordingDate", recording_date)
    return output_path
