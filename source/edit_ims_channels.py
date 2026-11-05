#!/usr/bin/env python3

"""Batch-rename and recolor channels without rewriting IMS image data."""

import shutil
import traceback
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "examples"
OUTPUT_DIR = ROOT / "channel-edited"
OVERWRITE = False

CHANNEL_EDITS = {
     0: {"name": "CtBP2-488", "color": "magenta"},
     1: {"name": "MyoVIIA", "color": "white"},
     2: {"name": "Synapsin-6", "color": "cyan"},
}

NAMED_COLORS = {
    "Red": (1.0, 0.0, 0.0),
    "Green": (0.0, 1.0, 0.0),
    "Blue": (0.0, 0.0, 1.0),
    "Cyan": (0.0, 1.0, 1.0),
    "Yellow": (1.0, 1.0, 0.0),
    "Magenta": (1.0, 0.0, 1.0),
    "White": (1.0, 1.0, 1.0),
}


def parse_color(value):
    if isinstance(value, str):
        color = value.strip().lower()
        if color in NAMED_COLORS:
            return NAMED_COLORS[color]
        if len(color) == 7 and color.startswith("#"):
            try:
                return tuple(
                    int(color[index : index + 2], 16) / 255
                    for index in (1, 3, 5)
                )
            except ValueError:
                pass
        raise ValueError(
            f"Unknown color {value!r}; use a named color, #RRGGBB, or RGB tuple"
        )

    try:
        red, green, blue = (float(component) for component in value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid RGB color: {value!r}") from None
    if not all(0 <= component <= 255 for component in (red, green, blue)):
        raise ValueError(f"RGB components must be between 0 and 255: {value!r}")
    if max(red, green, blue) > 1:
        red, green, blue = red / 255, green / 255, blue / 255
    return red, green, blue


def _attribute_value(current, text):
    """Match the source IMS string encoding when replacing an attribute."""
    if current is None:
        return np.frombuffer(text.encode("utf-8"), dtype="S1")
    if isinstance(current, np.ndarray) and current.dtype.kind in {"S", "a"}:
        if current.ndim == 1 and current.dtype.itemsize == 1:
            return np.frombuffer(text.encode("utf-8"), dtype="S1")
        return np.bytes_(text)
    if isinstance(current, (bytes, np.bytes_)):
        return np.bytes_(text)
    return text


def _set_text_attribute(group, name, text):
    current = group.attrs.get(name)
    value = _attribute_value(current, str(text))
    if name in group.attrs:
        del group.attrs[name]
    group.attrs.create(name, value)


def edit_channels(path):
    with h5py.File(path, "r+") as handle:
        try:
            dataset_info = handle["DataSetInfo"]
        except KeyError:
            raise ValueError(f"Not an Imaris IMS file: {path}") from None

        for index, edit in CHANNEL_EDITS.items():
            channel_name = f"Channel {index}"
            if channel_name not in dataset_info:
                raise ValueError(f"{path.name} has no {channel_name}")
            channel = dataset_info[channel_name]

            if "name" in edit:
                _set_text_attribute(channel, "Name", edit["name"])
            if "color" in edit:
                color = parse_color(edit["color"])
                _set_text_attribute(
                    channel, "Color", " ".join(format(value, ".15g") for value in color)
                )
                _set_text_attribute(channel, "ColorMode", "BaseColor")


def process_file(input_path):
    output_path = OUTPUT_DIR / input_path.name
    partial_path = OUTPUT_DIR / f".{input_path.name}.partial"
    if output_path.exists() and not OVERWRITE:
        print(f"Skipping existing: {output_path.name}")
        return

    partial_path.unlink(missing_ok=True)
    try:
        shutil.copy2(input_path, partial_path)
        edit_channels(partial_path)
        output_path.unlink(missing_ok=True)
        partial_path.replace(output_path)
        print(f"Saved: {output_path}")
    finally:
        partial_path.unlink(missing_ok=True)


def main():
    if not CHANNEL_EDITS:
        raise ValueError("Configure at least one entry in CHANNEL_EDITS")

    input_files = sorted(INPUT_DIR.glob("*.ims"))
    if not input_files:
        raise FileNotFoundError(f"No .ims files found in {INPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(input_files)} IMS files")
    print(f"Output: {OUTPUT_DIR}")
    failed = []
    for input_path in input_files:
        try:
            process_file(input_path)
        except Exception:
            print(f"FAILED: {input_path.name}")
            traceback.print_exc()
            failed.append(input_path.name)
    if failed:
        raise SystemExit(f"Batch finished with {len(failed)} failed files: {failed}")
    print("Batch complete")


if __name__ == "__main__":
    main()
