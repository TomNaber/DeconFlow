"""Discover PSFs and normalize legacy C0 and current C1 filenames."""

import json
import re
from pathlib import Path


def read_psf_channels(folder):
    folder = Path(folder)
    manifest_path = folder / "psf_manifest.json"
    if manifest_path.is_file():
        entries = json.loads(manifest_path.read_text(encoding="utf-8"))["channels"]
    else:
        entries = [{"psf_file": path.name} for path in sorted(folder.glob("psf_C*.tif"))]
    parsed = []
    for entry in entries:
        filename = entry["psf_file"]
        match = re.fullmatch(r"psf_C(\d+)_(.+)\.tiff?", filename, re.IGNORECASE)
        if match is None:
            raise ValueError(f"Expected a PSF filename like psf_C1_Alexa488.tif: {filename}")
        if not (folder / filename).is_file():
            raise ValueError(f"PSF file not found: {filename}")
        parsed.append({**entry, "number": int(match[1]), "name": entry.get("name", match[2])})
    if not parsed:
        raise ValueError("No PSF files found in the selected PSF folder.")
    base = 0 if any(entry["number"] == 0 for entry in parsed) else 1
    indices = set()
    for entry in parsed:
        entry["index"] = entry.pop("number") - base
        if entry["index"] in indices:
            raise ValueError("Multiple PSFs have the same channel number.")
        indices.add(entry["index"])
    return sorted(parsed, key=lambda entry: entry["index"])
