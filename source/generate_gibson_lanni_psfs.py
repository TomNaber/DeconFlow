#!/usr/bin/env python3

import re
from pathlib import Path
from time import monotonic

import numpy as np
import psfmodels as psfm
import tifffile

from console_progress import ProgressLine


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "psfs/1000"
OUTPUT_IS_TEMPORARY = False
SCAN_DIRECTION = "up"

XY_PIXEL_SIZE_UM = 0.0801669
Z_STEP_UM = 0.3002
CHANNELS = [
    {"name": "Alexa488", "excitation_nm": 488.0, "emission_nm": 525.0},
    {"name": "Alexa568", "excitation_nm": 552.0, "emission_nm": 603.0},
    {"name": "Alexa647", "excitation_nm": 638.0, "emission_nm": 671.0},
]

NUMERICAL_APERTURE = 1.4
IMMERSION_RI_DESIGN = 1.4811
IMMERSION_RI_ACTUAL = 1.4811
SPECIMEN_RI = 1.45
COVERSLIP_RI_DESIGN = 1.5255
COVERSLIP_RI_ACTUAL = 1.5255
COVERSLIP_THICKNESS_DESIGN_UM = 145.0
COVERSLIP_THICKNESS_ACTUAL_UM = 145.0
OBJECTIVE_WORKING_DISTANCE_UM = 140.0
IMAGING_DEPTH_UM = 400.0
PINHOLE_AIRY_UNITS = 1.0
PSF_SIZE_Z = 209
PSF_SIZE_XY = 119
OVERSAMPLE_FACTOR = 3
OVERWRITE = True


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = monotonic()

    for index, channel in enumerate(CHANNELS):
        name = channel["name"]
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or f"channel{index + 1}"
        input_index = channel.get("index", index)
        filename = f"psf_C{input_index + 1}_{safe_name}.tif"
        output_path = OUTPUT_DIR / filename

        label = f"Channel {index + 1}/{len(CHANNELS)} ({name})"
        if output_path.exists() and not OVERWRITE:
            print(f"{label} skipped (already exists).", flush=True)
            continue

        with ProgressLine(label, activity="PSF generation").running():
            psf = psfm.confocal_psf(
                z=PSF_SIZE_Z,
                nx=PSF_SIZE_XY,
                dxy=XY_PIXEL_SIZE_UM,
                dz=Z_STEP_UM,
                pz=IMAGING_DEPTH_UM,
                pinhole_au=PINHOLE_AIRY_UNITS,
                NA=NUMERICAL_APERTURE,
                ex_wvl=channel["excitation_nm"] / 1000.0,
                em_wvl=channel["emission_nm"] / 1000.0,
                ns=SPECIMEN_RI,
                ni=IMMERSION_RI_ACTUAL,
                ni0=IMMERSION_RI_DESIGN,
                tg=COVERSLIP_THICKNESS_ACTUAL_UM,
                tg0=COVERSLIP_THICKNESS_DESIGN_UM,
                ng=COVERSLIP_RI_ACTUAL,
                ng0=COVERSLIP_RI_DESIGN,
                ti0=OBJECTIVE_WORKING_DISTANCE_UM,
                oversample_factor=OVERSAMPLE_FACTOR,
                model="scalar",
            )
            psf = np.asarray(psf, dtype=np.float32)
            np.nan_to_num(psf, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            np.maximum(psf, 0.0, out=psf)
            psf /= psf.sum(dtype=np.float64)
            if SCAN_DIRECTION == "down":
                psf = np.ascontiguousarray(psf[::-1])

            tifffile.imwrite(
                output_path,
                psf,
                imagej=True,
                metadata={"axes": "ZYX", "spacing": Z_STEP_UM, "unit": "um"},
                resolution=(1.0 / XY_PIXEL_SIZE_UM, 1.0 / XY_PIXEL_SIZE_UM),
            )

    message = f"PSF generation completed in {monotonic() - started:.1f}s."
    if not OUTPUT_IS_TEMPORARY:
        message += f" Output: {OUTPUT_DIR}"
    print(message, flush=True)


if __name__ == "__main__":
    main()
