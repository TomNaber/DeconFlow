"""Discover TIFF PSFs without a manifest and validate their volume data."""

import re
from collections import Counter
from pathlib import Path


def load_psf_volume(path):
    import numpy as np
    import tifffile

    psf = np.asarray(tifffile.imread(path), dtype=np.float32).squeeze()
    if psf.ndim != 3 or not all(psf.shape):
        raise ValueError(f"PSF must be a 3D TIFF volume: {path}")
    np.nan_to_num(psf, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.maximum(psf, 0.0, out=psf)
    total = psf.sum(dtype=np.float64)
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"PSF must contain positive intensity: {path}")
    psf /= total
    return np.ascontiguousarray(psf)


def read_psf_channels(files, *, validate=False):
    if isinstance(files, (str, Path)):
        files = sorted(path for path in Path(files).iterdir()
                       if path.suffix.lower() in (".tif", ".tiff"))
    paths = list(dict.fromkeys(Path(path).expanduser().resolve() for path in files))
    parsed = []
    for path in paths:
        if not path.is_file() or path.suffix.lower() not in (".tif", ".tiff"):
            raise ValueError(f"PSF TIFF file not found: {path}")
        if validate:
            load_psf_volume(path)
        match = re.fullmatch(r"psf_C(\d+)_(.+)\.tiff?", path.name, re.IGNORECASE)
        parsed.append({"psf_file": str(path), "name": match[2] if match else path.stem,
                       "number": int(match[1]) if match else None})
    base = 0 if any(entry["number"] == 0 for entry in parsed) else 1
    counts = Counter(entry["number"] for entry in parsed)
    for entry in parsed:
        number = entry.pop("number")
        entry["index"] = number - base if number is not None and counts[number] == 1 else None
    return parsed
