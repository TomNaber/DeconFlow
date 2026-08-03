#!/usr/bin/env python3

import json
import re
from pathlib import Path

import numpy as np
import psfmodels as psfm
import tifffile


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "psfs/1000"
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
    print(
        f"PSF sampling: dz={Z_STEP_UM:g} µm, "
        f"dy={XY_PIXEL_SIZE_UM:g} µm, dx={XY_PIXEL_SIZE_UM:g} µm"
    )

    manifest = {
        "model": "confocal scalar Gibson-Lanni",
        "axes": "ZYX",
        "scan_direction": SCAN_DIRECTION,
        "voxel_size_um": {
            "z": Z_STEP_UM,
            "y": XY_PIXEL_SIZE_UM,
            "x": XY_PIXEL_SIZE_UM,
        },
        "imaging_depth_um": IMAGING_DEPTH_UM,
        "channels": [],
    }

    for index, channel in enumerate(CHANNELS):
        name = channel["name"]
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
        filename = f"psf_C{index}_{safe_name}.tif"
        output_path = OUTPUT_DIR / filename

        manifest["channels"].append(
            {
                "index": index,
                "name": name,
                "excitation_nm": channel["excitation_nm"],
                "emission_nm": channel["emission_nm"],
                "psf_file": filename,
            }
        )

        if output_path.exists() and not OVERWRITE:
            print(f"Skipping existing: {filename}")
            continue

        print(f"Generating channel {index + 1}/{len(CHANNELS)}: {name}")
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
        print(f"  Saved {filename}")

    manifest_path = OUTPUT_DIR / "psf_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
