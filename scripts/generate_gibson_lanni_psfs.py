#!/usr/bin/env python3
"""Generate channel-specific confocal Gibson-Lanni PSFs."""

import json
import re
from pathlib import Path

import numpy as np
import psfmodels as psfm
import tifffile


# =============================================================================
# SETTINGS TO EDIT
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent / "psfs/1000"

# The direction of the confocal scan.
SCAN_DIRECTION = "up"

# Image sampling in micrometres.
#
# These values must match the images that will later be deconvolved.
XY_PIXEL_SIZE_UM = 0.0801669
Z_STEP_UM = 0.3002

# Must match the channel order in the Imaris files.
CHANNELS = [
    {
        "name": "Alexa488",
        "excitation_nm": 488.0,
        "emission_nm": 525.0,
    },
    {
        "name": "Alexa568",
        "excitation_nm": 552.0,
        "emission_nm": 603.0,
    },
    {
        "name": "Alexa647",
        "excitation_nm": 638.0,
        "emission_nm": 671.0,
    },
]

# Objective and optical parameters.
NUMERICAL_APERTURE = 1.4

IMMERSION_RI_DESIGN = 1.4811
IMMERSION_RI_ACTUAL = 1.4811

SPECIMEN_RI = 1.45

COVERSLIP_RI_DESIGN = 1.5255
COVERSLIP_RI_ACTUAL = 1.5255

COVERSLIP_THICKNESS_DESIGN_UM = 145.0
COVERSLIP_THICKNESS_ACTUAL_UM = 145.0

OBJECTIVE_WORKING_DISTANCE_UM = 140.0

# Fixed fluorophore depth relative to the coverslip/sample interface.
IMAGING_DEPTH_UM = 1000.0

# Confocal pinhole diameter in Airy units.
PINHOLE_AIRY_UNITS = 1.0

# PSF dimensions must be odd.
PSF_SIZE_Z = 209
PSF_SIZE_XY = 119

# Pixel integration accuracy. Higher values are slower.
OVERSAMPLE_FACTOR = 3

OVERWRITE = True


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def safe_name(text):
    """Convert a channel name into a safe filename component."""
    return re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        text,
    ).strip("_")


def edge_to_peak_ratio(psf):
    """Estimate whether the PSF array is large enough."""

    edge_maximum = max(
        psf[0].max(),
        psf[-1].max(),
        psf[:, 0, :].max(),
        psf[:, -1, :].max(),
        psf[:, :, 0].max(),
        psf[:, :, -1].max(),
    )

    peak = float(psf.max())

    if peak <= 0:
        return float("inf")

    return float(edge_maximum / peak)


def validate_settings():
    """Check the manually entered PSF-generation parameters."""

    if XY_PIXEL_SIZE_UM <= 0:
        raise ValueError(
            "XY_PIXEL_SIZE_UM must be greater than zero."
        )

    if Z_STEP_UM <= 0:
        raise ValueError(
            "Z_STEP_UM must be greater than zero."
        )

    if PSF_SIZE_Z % 2 == 0 or PSF_SIZE_XY % 2 == 0:
        raise ValueError(
            "PSF_SIZE_Z and PSF_SIZE_XY must both be odd."
        )

    if NUMERICAL_APERTURE <= 0:
        raise ValueError(
            "NUMERICAL_APERTURE must be greater than zero."
        )

    if NUMERICAL_APERTURE >= IMMERSION_RI_ACTUAL:
        raise ValueError(
            "NUMERICAL_APERTURE must be lower than "
            "IMMERSION_RI_ACTUAL."
        )

    if not CHANNELS:
        raise ValueError(
            "CHANNELS must contain at least one channel."
        )

    if SCAN_DIRECTION not in {"up", "down"}:
        raise ValueError(
            'SCAN_DIRECTION must be either "up" or "down".'
        )


def main():

    validate_settings()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dx = float(XY_PIXEL_SIZE_UM)
    dy = float(XY_PIXEL_SIZE_UM)
    dz = float(Z_STEP_UM)
    dxy = float(XY_PIXEL_SIZE_UM)

    print(
        "PSF sampling: "
        f"dz={dz:g} µm, "
        f"dy={dy:g} µm, "
        f"dx={dx:g} µm"
    )

    manifest = {
        "model": "confocal scalar Gibson-Lanni",
        "axes": "ZYX",
        "scan_direction": SCAN_DIRECTION,
        "voxel_size_um": {
            "z": dz,
            "y": dy,
            "x": dx,
        },
        "imaging_depth_um": IMAGING_DEPTH_UM,
        "channels": [],
    }

    for channel_index, channel in enumerate(CHANNELS):

        channel_name = channel["name"]

        filename = (
            f"psf_C{channel_index}_"
            f"{safe_name(channel_name)}.tif"
        )

        output_path = OUTPUT_DIR / filename

        if output_path.exists() and not OVERWRITE:
            raise FileExistsError(
                f"Already exists: {output_path}\n"
                "Set OVERWRITE=True to replace it."
            )

        print(
            f"Generating channel {channel_index}: "
            f"{channel_name}"
        )

        psf = psfm.confocal_psf(
            z=PSF_SIZE_Z,
            nx=PSF_SIZE_XY,

            dxy=dxy,
            dz=dz,

            # Fluorophore depth relative to the coverslip.
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

            # The scalar model is the Gibson-Lanni model.
            model="scalar",
        )

        psf = np.asarray(
            psf,
            dtype=np.float32,
        )

        psf = np.nan_to_num(
            psf,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        np.maximum(
            psf,
            0.0,
            out=psf,
        )

        psf_sum = psf.sum(
            dtype=np.float64,
        )

        if not np.isfinite(psf_sum) or psf_sum <= 0:
            raise RuntimeError(
                f"Invalid PSF generated for {channel_name}."
            )

        # Normalize the PSF for use as a convolution kernel.
        psf /= psf_sum

        # Match the saved PSF orientation to the acquisition Z direction.
        #
        # Axis 0 is Z because the PSF uses ZYX ordering.
        if SCAN_DIRECTION == "down":
            psf = np.ascontiguousarray(
                psf[::-1, :, :]
            )

        edge_ratio = edge_to_peak_ratio(psf)

        tifffile.imwrite(
            str(output_path),
            np.asarray(psf, dtype=np.float32),
                imagej=True,
            metadata={
                "axes": "ZYX",
                "spacing": float(dz),
                "unit": "um",
            },
            resolution=(
                1.0 / float(dxy),
                1.0 / float(dxy),
            ),
        )

        manifest["channels"].append(
            {
                "index": channel_index,
                "name": channel_name,
                "excitation_nm": channel["excitation_nm"],
                "emission_nm": channel["emission_nm"],
                "psf_file": filename,
            }
        )

        print(
            f"  Saved {filename}; "
            f"boundary/peak={edge_ratio:.3g}"
        )

        if edge_ratio > 1e-3:
            print(
                "  WARNING: PSF signal reaches an array "
                "boundary. Increase PSF_SIZE_Z and/or "
                "PSF_SIZE_XY."
            )

    manifest_path = OUTPUT_DIR / "psf_manifest.json"

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Saved manifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()