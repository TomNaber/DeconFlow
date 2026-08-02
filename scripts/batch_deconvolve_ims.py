#!/usr/bin/env python3
"""Batch-deconvolve Imaris IMS files and write native uint16 IMS output."""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import re
import shutil
import sys
from collections import deque
from contextlib import contextmanager
from time import perf_counter
from pathlib import Path
from typing import Any

os.environ.setdefault("PYOPENCL_NO_CACHE", "1")

import bioio_bioformats
import numpy as np
import RedLionfishDeconv as rlf
import tifffile
from bioio import BioImage

# Allow direct execution from PROJECT_DIR/scripts while importing PROJECT_DIR/src.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.ims_writer import require_imaris_writer, write_imaris  # noqa: E402


# =============================================================================
# SETTINGS TO EDIT
# =============================================================================

INPUT_DIR = PROJECT_DIR / "examples" 
# INPUT_DIR = Path('/Volumes/TomNaberSSD/PhD Research Data/3 GABA in Tinnitus/3-1 Tinnitus classification/3-1-4 Tinnitus Synapse Analysis/Raw Z-stacks/Batch 2/ims')
PSF_DIR = PROJECT_DIR / "psfs"
OUTPUT_DIR = PROJECT_DIR / "dcv"
# OUTPUT_DIR = Path('/Volumes/TomNaberSSD/PhD Research Data/3 GABA in Tinnitus/3-1 Tinnitus classification/3-1-4 Tinnitus Synapse Analysis/Deconvolved Z-stacks/Batch 2')
TEMP_DIR = PROJECT_DIR / ".deconvolution_temp"

# Bio-Formats can expose more than one scene/series. For normal converted IMS
# files the desired full-resolution image is commonly scene 0. The script prints
# all discovered scenes so this can be checked before a full batch.
SCENE_INDEX = 0

ITERATIONS = 32

# Optional median filtering of the completed float32 Richardson-Lucy result,
# immediately before conversion to uint16.
#
# Kernel order is (Z, Y, X). With the current anisotropic sampling, (1, 3, 3)
# filters only within each XY plane and is a conservative starting point.
# Use (3, 3, 3) only when axial impulse noise also needs filtering.
APPLY_MEDIAN_FILTER = True
MEDIAN_FILTER_SIZE_ZYX: int | tuple[int, int, int] = (3, 3, 3)
MEDIAN_FILTER_MODE = "nearest"

# "gpu" uses OpenCL. On Apple Silicon, use "gpu" only after confirming that
# RedLionfish detects the Apple OpenCL device and does not fall back to CPU.
METHOD = "gpu"

# Useful for volumes that do not fit in GPU memory. RedLionfish warns that block
# processing can introduce block-boundary artefacts, so inspect a test result.
USE_BLOCK_ALGORITHM = True

OVERWRITE = False

# Continue processing later files if one file fails.
STOP_ON_ERROR = False

# Keep the temporary uint16 TCZYX memmap after an error for troubleshooting.
KEEP_TEMP_ON_ERROR = False

# Metadata written to the output IMS. Explicit values override source metadata.
OBJECTIVE_NA: float | None = 1.4
IMMERSION_REFRACTIVE_INDEX: float | None = 1.4811
APPLICATION_VERSION = "1.2"

# RedLionfish calculates in float32. Store those numeric intensities directly
# in uint16: 1.0 becomes 1, 255.0 becomes 255, and restored values above the
# original 8-bit range remain representable up to the true uint16 limit, 65535.
#
# Do not multiply by 256. Richardson-Lucy can legitimately create values above
# 255, so an x256 gain would saturate everything above approximately 256.

# Bio-Formats and its Java/native dependencies can print hundreds of INFO
# messages while opening an IMS file. Suppress those messages while retaining
# Python exceptions and this script's own status/progress output.
QUIET_BIOFORMATS = True


# =============================================================================
# LOGGING AND OUTPUT CONTROL
# =============================================================================


THIRD_PARTY_LOGGERS = (
    "bffile",
    "bioio",
    "bioio_base",
    "bioio_bioformats",
    "scyjava",
    "jgo",
    "jpype",
)


def configure_third_party_logging() -> None:
    """Suppress routine third-party logs without hiding raised exceptions."""

    for logger_name in THIRD_PARTY_LOGGERS:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.ERROR)
        logger.propagate = False

        # Some libraries install their own stream handler, which otherwise
        # produces a second copy in addition to the root logger.
        for handler in tuple(logger.handlers):
            logger.removeHandler(handler)

        logger.addHandler(logging.NullHandler())


@contextmanager
def suppress_process_output(enabled: bool = True):
    """Temporarily silence stdout/stderr at file-descriptor level.

    This also catches output written directly by Java/native libraries, which
    contextlib.redirect_stdout cannot intercept.
    """

    if not enabled:
        yield
        return

    try:
        sys.stdout.flush()
        sys.stderr.flush()

        stdout_fd = sys.stdout.fileno()
        stderr_fd = sys.stderr.fileno()
    except (AttributeError, OSError):
        # Fall back gracefully in environments without normal terminal FDs.
        yield
        return

    saved_stdout = os.dup(stdout_fd)
    saved_stderr = os.dup(stderr_fd)

    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), stdout_fd)
            os.dup2(devnull.fileno(), stderr_fd)
            yield
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            os.dup2(saved_stdout, stdout_fd)
            os.dup2(saved_stderr, stderr_fd)
            os.close(saved_stdout)
            os.close(saved_stderr)


configure_third_party_logging()


# =============================================================================
# PSF LOADING
# =============================================================================


def load_psfs() -> tuple[dict[str, Any], list[dict[str, Any]], list[np.ndarray]]:
    """Load the PSF manifest and all channel-specific PSFs."""

    manifest_path = PSF_DIR / "psf_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"PSF manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    channels = sorted(manifest["channels"], key=lambda channel: channel["index"])
    psfs: list[np.ndarray] = []

    for expected_index, channel in enumerate(channels):
        actual_index = int(channel["index"])
        if actual_index != expected_index:
            raise ValueError(
                "PSF channel indices must be consecutive and start at zero; "
                f"expected {expected_index}, found {actual_index}."
            )

        psf_path = PSF_DIR / channel["psf_file"]
        if not psf_path.is_file():
            raise FileNotFoundError(f"PSF not found: {psf_path}")

        psf = np.asarray(tifffile.imread(str(psf_path)), dtype=np.float32).squeeze()
        if psf.ndim != 3:
            raise ValueError(f"PSF is not 3D: {psf_path} has shape {psf.shape}")
        if any(size % 2 == 0 for size in psf.shape):
            raise ValueError(
                f"PSF dimensions should be odd, but {psf_path.name} "
                f"has shape {psf.shape}."
            )

        psf = np.nan_to_num(psf, nan=0.0, posinf=0.0, neginf=0.0)
        np.maximum(psf, 0.0, out=psf)
        psf_sum = psf.sum(dtype=np.float64)
        if not np.isfinite(psf_sum) or psf_sum <= 0:
            raise ValueError(f"Invalid PSF: {psf_path}")

        psf /= psf_sum
        psfs.append(np.ascontiguousarray(psf))

    if not psfs:
        raise ValueError("The PSF manifest contains no channels.")

    return manifest, channels, psfs


# =============================================================================
# BIOIO / BIO-FORMATS READING
# =============================================================================


def open_ims(input_path: Path) -> BioImage:
    """Open one IMS file explicitly with the Bio-Formats BioIO reader."""

    configure_third_party_logging()

    # Bio-Formats may write directly to Java/native stdout and stderr while
    # loading libraries and examining the IMS hierarchy. Silence that output,
    # then print only the useful scene information ourselves.
    with suppress_process_output(QUIET_BIOFORMATS):
        reader = BioImage(
            str(input_path),
            reader=bioio_bioformats.Reader,
        )
        scenes = tuple(reader.scenes)

    configure_third_party_logging()

    if not scenes:
        raise ValueError(f"No Bio-Formats scenes were found in {input_path.name}.")
    if not 0 <= SCENE_INDEX < len(scenes):
        raise IndexError(
            f"SCENE_INDEX={SCENE_INDEX} is invalid for {input_path.name}; "
            f"the file contains {len(scenes)} scene(s): {scenes}"
        )

    if len(scenes) > 1:
        print(f"  Bio-Formats scenes ({len(scenes)}):")
        for index, scene in enumerate(scenes):
            marker = "  <-- selected" if index == SCENE_INDEX else ""
            print(f"    [{index}] {scene!r}{marker}")

    with suppress_process_output(QUIET_BIOFORMATS):
        reader.set_scene(SCENE_INDEX)
        dimension_order = reader.dims.order

    configure_third_party_logging()

    # BioImage standardizes image access to TCZYX, adding singleton dimensions
    # where the source has no explicit T, C, or Z dimension.
    if dimension_order != "TCZYX":
        raise ValueError(
            f"Unexpected BioIO dimension order {dimension_order!r}; "
            "expected standardized TCZYX."
        )

    return reader


def get_voxel_size(reader: BioImage) -> tuple[float, float, float]:
    """Return physical voxel spacing as dz, dy, dx in micrometres."""

    with suppress_process_output(QUIET_BIOFORMATS):
        physical = reader.physical_pixel_sizes

    configure_third_party_logging()
    missing = [
        axis
        for axis, value in (("Z", physical.Z), ("Y", physical.Y), ("X", physical.X))
        if value is None
    ]
    if missing:
        raise ValueError(
            "Bio-Formats did not report physical pixel size for: "
            + ", ".join(missing)
        )

    dz = float(physical.Z)
    dy = float(physical.Y)
    dx = float(physical.X)

    if not all(np.isfinite(value) and value > 0 for value in (dz, dy, dx)):
        raise ValueError(
            f"Invalid physical pixel sizes: dz={dz}, dy={dy}, dx={dx} µm."
        )

    return dz, dy, dx


def read_channel_volume(
    reader: BioImage,
    time_index: int,
    channel_index: int,
) -> np.ndarray:
    """Read one selected channel as a non-negative contiguous float32 ZYX volume."""

    # This remains lazy until compute() and requests only the selected T/C volume.
    with suppress_process_output(QUIET_BIOFORMATS):
        lazy_volume = reader.get_image_dask_data(
            "ZYX",
            T=time_index,
            C=channel_index,
        )
        raw = np.asarray(lazy_volume.compute())

    configure_third_party_logging()

    if raw.ndim != 3:
        raise RuntimeError(
            "Expected a ZYX channel volume from BioIO, but obtained "
            f"shape {raw.shape}."
        )

    if raw.dtype == np.int8:
        print("      WARNING: reinterpreting signed int8 bytes as uint8.")
        raw = raw.view(np.uint8)

    print(
        f"      input dtype={raw.dtype}, "
        f"range={float(raw.min()):g}–{float(raw.max()):g}"
    )

    image = np.asarray(raw, dtype=np.float32)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    np.maximum(image, 0.0, out=image)
    return np.ascontiguousarray(image)


# =============================================================================
# COMPACT REDLIONFISH PROGRESS
# =============================================================================


def format_duration(seconds: float) -> str:
    """Format seconds as a compact human-readable duration."""

    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


class RedLionfishProgressHandler(logging.Handler):
    """Turn RedLionfish INFO records into block progress and ETA messages."""

    DATA_SHAPE_PATTERN = re.compile(r"data shape:\s*\(([^)]*)\)")
    BLOCK_STEP_PATTERN = re.compile(r"blockstep:\s*\[([^\]]*)\]")

    def __init__(
        self,
        *,
        label: str,
        remaining_volumes: int,
    ) -> None:
        super().__init__(level=logging.INFO)
        self.label = label
        self.remaining_volumes = remaining_volumes
        self.data_shape: tuple[int, ...] | None = None
        self.block_step: tuple[int, ...] | None = None
        self.total_blocks: int | None = None
        self.completed_blocks = 0
        self.block_started_at: float | None = None
        self.block_durations: deque[float] = deque(maxlen=5)
        self.device_reported = False

    @staticmethod
    def _parse_integer_list(text: str) -> tuple[int, ...]:
        return tuple(int(value.strip()) for value in text.split(","))

    def _update_total_blocks(self) -> None:
        if self.data_shape is None or self.block_step is None:
            return
        if len(self.data_shape) != len(self.block_step):
            return

        self.total_blocks = math.prod(
            math.ceil(size / step)
            for size, step in zip(
                self.data_shape,
                self.block_step,
                strict=True,
            )
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()

            shape_match = self.DATA_SHAPE_PATTERN.search(message)
            if shape_match:
                self.data_shape = self._parse_integer_list(shape_match.group(1))
                self._update_total_blocks()
                return

            step_match = self.BLOCK_STEP_PATTERN.search(message)
            if step_match:
                self.block_step = self._parse_integer_list(step_match.group(1))
                self._update_total_blocks()
                return

            if "OCL device selected" in message and not self.device_reported:
                device = message.split(
                    "OCL device selected",
                    maxsplit=1,
                )[-1].lstrip(" :")
                print(f"      GPU: {device}", flush=True)
                self.device_reported = True
                return

            if "Start RL deconvolution of this block" in message:
                self.block_started_at = perf_counter()
                return

            if "This block's RL deconvolution completed" in message:
                now = perf_counter()

                if self.block_started_at is None:
                    block_duration = float("nan")
                else:
                    block_duration = now - self.block_started_at
                    self.block_durations.append(block_duration)

                self.block_started_at = None
                self.completed_blocks += 1

                if self.total_blocks is None or not self.block_durations:
                    print(
                        f"      {self.label}: block "
                        f"{self.completed_blocks} completed",
                        flush=True,
                    )
                    return

                mean_duration = (
                    sum(self.block_durations) / len(self.block_durations)
                )
                remaining_channel_blocks = max(
                    self.total_blocks - self.completed_blocks,
                    0,
                )
                remaining_file_blocks = (
                    remaining_channel_blocks
                    + self.remaining_volumes * self.total_blocks
                )
                percentage = (
                    100.0 * self.completed_blocks / self.total_blocks
                )
                block_text = (
                    format_duration(block_duration)
                    if np.isfinite(block_duration)
                    else "unknown"
                )

                print(
                    f"      {self.label}: "
                    f"block {self.completed_blocks}/{self.total_blocks} "
                    f"({percentage:.0f}%) | "
                    f"block {block_text} | "
                    f"channel ETA "
                    f"{format_duration(remaining_channel_blocks * mean_duration)} | "
                    f"file ETA "
                    f"{format_duration(remaining_file_blocks * mean_duration)}",
                    flush=True,
                )
                return

            if "GPU calculation failed, falling back to CPU" in message:
                print(
                    f"      WARNING: {self.label}: GPU failed; "
                    "RedLionfish is switching to CPU.",
                    flush=True,
                )
        except Exception:
            self.handleError(record)


@contextmanager
def redlionfish_progress(
    *,
    label: str,
    remaining_volumes: int,
):
    """Attach compact RedLionfish progress reporting for one channel volume."""

    root_logger = logging.getLogger()
    handler = RedLionfishProgressHandler(
        label=label,
        remaining_volumes=remaining_volumes,
    )
    root_logger.addHandler(handler)

    try:
        yield
    finally:
        root_logger.removeHandler(handler)
        handler.close()


# =============================================================================
# OPTIONAL POST-DECONVOLUTION FILTERING
# =============================================================================


def validated_median_filter_size() -> tuple[int, int, int]:
    """Return the configured median-filter size as an odd positive ZYX tuple."""

    value = MEDIAN_FILTER_SIZE_ZYX
    if isinstance(value, (int, np.integer)):
        size = (int(value),) * 3
    else:
        try:
            size = tuple(int(component) for component in value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "MEDIAN_FILTER_SIZE_ZYX must be an integer or a "
                "three-element (Z, Y, X) sequence."
            ) from exc

    if len(size) != 3:
        raise ValueError(
            "MEDIAN_FILTER_SIZE_ZYX must contain exactly three values "
            "in (Z, Y, X) order."
        )
    if any(component <= 0 for component in size):
        raise ValueError("Median-filter dimensions must be greater than zero.")
    if any(component % 2 == 0 for component in size):
        raise ValueError(
            "Median-filter dimensions must be odd so the kernel has a "
            f"well-defined centre; got {size}."
        )

    return size


def require_scipy_median_filter():
    """Load scipy.ndimage.median_filter only when filtering is enabled."""

    if not APPLY_MEDIAN_FILTER:
        return None

    try:
        from scipy.ndimage import median_filter
    except ImportError as exc:
        python = sys.executable
        raise RuntimeError(
            "APPLY_MEDIAN_FILTER is enabled, but SciPy is unavailable. "
            "Install it into this exact environment with:\n"
            f'  "{python}" -m pip install scipy'
        ) from exc

    return median_filter


def apply_post_deconvolution_filter(result: np.ndarray) -> np.ndarray:
    """Optionally median-filter one completed float32 ZYX result."""

    if not APPLY_MEDIAN_FILTER:
        return result

    median_filter = require_scipy_median_filter()
    if median_filter is None:
        return result

    size = validated_median_filter_size()
    before_min = float(result.min())
    before_max = float(result.max())

    filtered = median_filter(
        result,
        size=size,
        mode=MEDIAN_FILTER_MODE,
    )
    filtered = np.asarray(filtered, dtype=np.float32)

    after_min = float(filtered.min())
    after_max = float(filtered.max())
    print(
        f"      median filter ZYX={size}, mode={MEDIAN_FILTER_MODE!r}; "
        f"range {before_min:g}–{before_max:g} -> "
        f"{after_min:g}–{after_max:g}"
    )

    return filtered


# =============================================================================
# DECONVOLUTION
# =============================================================================


def deconvolve_into_memmap(
    *,
    reader: BioImage,
    output: np.memmap,
    channels: list[dict[str, Any]],
    psfs: list[np.ndarray],
    n_timepoints: int,
    n_channels: int,
) -> list[tuple[float, float]]:
    """Deconvolve volumes and return non-clipping display ranges per channel."""

    channel_maxima = [0 for _ in range(n_channels)]

    for time_index in range(n_timepoints):
        for channel_index in range(n_channels):
            channel_name = str(channels[channel_index]["name"])
            print(
                f"    T {time_index + 1}/{n_timepoints}, "
                f"C {channel_index + 1}/{n_channels}: {channel_name}"
            )

            image = read_channel_volume(reader, time_index, channel_index)

            volume_index = time_index * n_channels + channel_index
            total_volumes = n_timepoints * n_channels
            remaining_volumes = total_volumes - volume_index - 1
            progress_label = (
                f"T{time_index + 1}/{n_timepoints} "
                f"C{channel_index + 1}/{n_channels} "
                f"{channel_name}"
            )

            with redlionfish_progress(
                label=progress_label,
                remaining_volumes=remaining_volumes,
            ):
                result = rlf.doRLDeconvolutionFromNpArrays(
                    image,
                    psfs[channel_index],
                    niter=ITERATIONS,
                    method=METHOD,
                    useBlockAlgorithm=USE_BLOCK_ALGORITHM,
                    resAsUint8=False,
                )

            if result is None:
                raise RuntimeError(
                    "RedLionfish returned no result for "
                    f"T={time_index}, C={channel_index}."
                )

            result = np.asarray(result, dtype=np.float32)
            if result.shape != image.shape:
                raise RuntimeError(
                    f"Unexpected result shape {result.shape}; expected {image.shape}."
                )

            result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
            np.maximum(result, 0.0, out=result)

            result = apply_post_deconvolution_filter(result)
            np.maximum(result, 0.0, out=result)

            float_min = float(result.min())
            float_max = float(result.max())

            # Preserve the Richardson-Lucy numeric range directly. The restored
            # result is not restricted to the source's 0..255 acquisition range.
            uint16_max = np.iinfo(np.uint16).max
            saturated_voxels = int(np.count_nonzero(result > uint16_max))
            total_voxels = int(result.size)

            stored = np.clip(
                np.rint(result),
                0,
                uint16_max,
            ).astype(np.uint16)

            stored_min = int(stored.min())
            stored_max = int(stored.max())
            channel_maxima[channel_index] = max(
                channel_maxima[channel_index],
                stored_max,
            )

            output[time_index, channel_index, :, :, :] = stored
            output.flush()

            print(
                f"      RL float range={float_min:g}–{float_max:g}; "
                f"stored 1:1 as uint16={stored_min}–{stored_max}"
            )

            if saturated_voxels:
                saturation_percent = (
                    100.0 * saturated_voxels / total_voxels
                )
                print(
                    "      WARNING: "
                    f"{saturated_voxels:,} voxels "
                    f"({saturation_percent:.6f}%) exceeded the true uint16 "
                    "limit of 65535 and were clipped."
                )

            # Release each full input/output channel before loading the next one.
            del image, result, stored
            gc.collect()

    # The native writer defaults every channel display range to 0..255. Use the
    # actual stored maximum so Imaris does not merely display valid uint16 data
    # as saturated white.
    return [
        (0.0, float(maximum if maximum > 0 else 1))
        for maximum in channel_maxima
    ]


def ensure_temp_space(shape: tuple[int, ...]) -> int:
    """Verify that TEMP_DIR has enough free space for one uint16 memmap."""

    required_bytes = int(np.prod(shape, dtype=np.int64)) * np.dtype(np.uint16).itemsize
    free_bytes = shutil.disk_usage(TEMP_DIR).free
    reserve_bytes = 2 * 1024**3

    if free_bytes < required_bytes + reserve_bytes:
        raise OSError(
            "Insufficient free disk space for the temporary uint16 array. "
            f"Required approximately {required_bytes / 1024**3:.2f} GiB plus "
            f"a 2 GiB reserve; only {free_bytes / 1024**3:.2f} GiB is free."
        )

    return required_bytes


def process_file(
    input_path: Path,
    manifest: dict[str, Any],
    channels: list[dict[str, Any]],
    psfs: list[np.ndarray],
) -> None:
    """Deconvolve one IMS file and write a native uint16 IMS result."""

    output_path = OUTPUT_DIR / f"{input_path.stem}_deconvolved.ims"
    partial_path = OUTPUT_DIR / f".{input_path.stem}_deconvolved.partial.ims"
    temporary_array_path = TEMP_DIR / f".{input_path.stem}.uint16.tczyx.dat"

    if output_path.exists() and not OVERWRITE:
        print(f"Skipping existing: {output_path.name}")
        return

    partial_path.unlink(missing_ok=True)
    temporary_array_path.unlink(missing_ok=True)

    reader: BioImage | None = None
    output_array: np.memmap | None = None
    succeeded = False

    try:
        reader = open_ims(input_path)

        n_t = int(reader.dims.T)
        n_c = int(reader.dims.C)
        n_z = int(reader.dims.Z)
        n_y = int(reader.dims.Y)
        n_x = int(reader.dims.X)
        shape = (n_t, n_c, n_z, n_y, n_x)

        dz, dy, dx = get_voxel_size(reader)

        psf_voxel = manifest["voxel_size_um"]
        expected_voxel = (
            float(psf_voxel["z"]),
            float(psf_voxel["y"]),
            float(psf_voxel["x"]),
        )

        if n_c != len(psfs):
            raise ValueError(
                f"IMS contains {n_c} channels, but there are {len(psfs)} PSFs."
            )

        if not np.allclose(
            (dz, dy, dx),
            expected_voxel,
            rtol=0.01,
            atol=1e-9,
        ):
            raise ValueError(
                "Voxel-size mismatch.\n"
                f"IMS: {(dz, dy, dx)} µm\n"
                f"PSF: {expected_voxel} µm\n"
                "Generate a separate matching PSF set."
            )

        print(f"\nProcessing {input_path.name}")
        print(f"  Selected scene: {reader.current_scene!r}")
        print(f"  Shape TCZYX: {shape}")
        print(f"  Voxel size ZYX: {(dz, dy, dx)} µm")
        print(f"  Source dtype: {reader.dtype}")
        print(f"  Source channels: {reader.channel_names}")

        required_bytes = ensure_temp_space(shape)
        print(f"  Temporary uint16 array: {required_bytes / 1024**3:.2f} GiB")

        output_array = np.memmap(
            temporary_array_path,
            mode="w+",
            dtype=np.uint16,
            shape=shape,
        )

        channel_display_ranges = deconvolve_into_memmap(
            reader=reader,
            output=output_array,
            channels=channels,
            psfs=psfs,
            n_timepoints=n_t,
            n_channels=n_c,
        )

        output_array.flush()

        # BioImage does not expose the old imaris-reader close() API. Drop the
        # object before native IMS writing so Java/Dask reader resources can be
        # reclaimed when possible.
        reader = None
        gc.collect()

        print(f"  Writing native IMS: {output_path.name}")
        write_imaris(
            output_array,
            partial_path,
            axes="TCZYX",
            voxel_size_um=(dz, dy, dx),
            channel_names=[str(channel["name"]) for channel in channels],
            channel_ranges=channel_display_ranges,
            excitation_wavelengths_nm=[
                _optional_float(channel.get("excitation_nm")) for channel in channels
            ],
            emission_wavelengths_nm=[
                _optional_float(channel.get("emission_nm")) for channel in channels
            ],
            source_ims=input_path,
            objective_na=OBJECTIVE_NA,
            refractive_index=IMMERSION_REFRACTIVE_INDEX,
            application_name="RedLionfish batch",
            application_version=APPLICATION_VERSION,
        )

        output_path.unlink(missing_ok=True)
        partial_path.replace(output_path)
        succeeded = True
        print(f"Finished: {output_path.name}")

    finally:
        reader = None

        if output_array is not None:
            output_array.flush()
            del output_array

        gc.collect()

        if succeeded or not KEEP_TEMP_ON_ERROR:
            temporary_array_path.unlink(missing_ok=True)
        if not succeeded:
            partial_path.unlink(missing_ok=True)


def _optional_float(value: Any) -> float | None:
    """Convert an optional manifest value to float."""

    if value is None:
        return None
    return float(value)


# =============================================================================
# PROGRAM ENTRY POINT
# =============================================================================


def main() -> None:
    """Run the complete batch."""

    if ITERATIONS <= 0:
        raise ValueError("ITERATIONS must be positive.")
    if METHOD not in {"gpu", "cpu"}:
        raise ValueError('METHOD must be "gpu" or "cpu".')
    if SCENE_INDEX < 0:
        raise ValueError("SCENE_INDEX must be zero or greater.")

    if APPLY_MEDIAN_FILTER:
        validated_median_filter_size()
        require_scipy_median_filter()
        if MEDIAN_FILTER_MODE not in {
            "reflect",
            "constant",
            "nearest",
            "mirror",
            "wrap",
        }:
            raise ValueError(
                "MEDIAN_FILTER_MODE must be one of: reflect, constant, "
                "nearest, mirror, or wrap."
            )

    # Fail before deconvolving anything if native IMS export is unavailable.
    require_imaris_writer()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        force=True,
    )

    # The root logger remains at INFO so the custom RedLionfish handler can
    # inspect block events. Its normal console handler prints only WARNING+.
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in root_logger.handlers:
        handler.setLevel(logging.WARNING)

    configure_third_party_logging()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    manifest, channels, psfs = load_psfs()
    input_files = sorted(INPUT_DIR.glob("*.ims"))
    if not input_files:
        raise FileNotFoundError(f"No .ims files found in {INPUT_DIR}")

    print(f"Loaded {len(psfs)} channel PSFs.")
    print(f"Found {len(input_files)} IMS files.")
    print("Reader: BioIO with bioio-bioformats")
    print(
        "Logging: compact RedLionfish block progress; "
        f"Bio-Formats chatter {'suppressed' if QUIET_BIOFORMATS else 'enabled'}"
    )
    print("Output dtype: uint16; RL intensities preserved 1:1 (no gain remapping)")
    if APPLY_MEDIAN_FILTER:
        print(
            "Post-filter: SciPy median filter, "
            f"ZYX={validated_median_filter_size()}, "
            f"mode={MEDIAN_FILTER_MODE!r}"
        )
    else:
        print("Post-filter: disabled")
    print(f"Output: {OUTPUT_DIR}")

    failed_files: list[str] = []

    for input_path in input_files:
        try:
            process_file(input_path, manifest, channels, psfs)
        except Exception as error:
            logging.exception("FAILED: %s: %s", input_path.name, error)
            failed_files.append(input_path.name)
            if STOP_ON_ERROR:
                raise

    if failed_files:
        raise SystemExit(
            f"Batch finished with {len(failed_files)} failed files: {failed_files}"
        )

    print("\nBatch complete.")


if __name__ == "__main__":
    main()
