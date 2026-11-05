#!/usr/bin/env python3

import gc
import os
import shutil
import subprocess
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import bioio_bioformats
import h5py
import numpy as np
import tifffile
from bioio import BioImage

from platform_support import default_fiji_executable
from ims_writer import write_imaris
from edit_ims_channels import parse_color
from psf_channels import read_psf_channels
from console_progress import ProgressLine
from channel_metadata import read_channels


SOURCE_DIR = Path(__file__).resolve().parent
ROOT = SOURCE_DIR.parent
INPUT_DIR = ROOT / "examples"
PSF_DIR = ROOT / "psfs"
OUTPUT_DIR = ROOT / "dcv"
TEMP_DIR = ROOT / ".deconvolution_temp"

ITERATIONS = 40
RLTV_REGULARIZATION = 0.0001
RLTV_NON_CIRCULANT = True
FIJI_EXECUTABLE = Path(default_fiji_executable())
FIJI_MEMORY = "12g"
OVERWRITE = False
CHANNEL_EDITS = {}
PSF_FILES = []
# None preserves automatic all-channel processing for command-line callers.
CHANNEL_INDICES = None


@contextmanager
def quiet_bioformats():
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    try:
        with open(os.devnull, "w") as sink:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
            yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(saved[0])
        os.close(saved[1])


def load_psfs():
    if CHANNEL_INDICES is not None:
        if (len(CHANNEL_INDICES) != len(set(CHANNEL_INDICES))
                or any(type(index) is not int or index < 0 for index in CHANNEL_INDICES)
                or len(CHANNEL_INDICES) != len(PSF_FILES)):
            raise ValueError("Select one PSF for each unique channel index")
        if not CHANNEL_INDICES:
            return [], []
    channels = read_psf_channels(PSF_DIR)
    if PSF_FILES:
        by_file = {channel["psf_file"]: channel for channel in channels}
        channels = [by_file[filename] for filename in PSF_FILES]
    elif [channel["index"] for channel in channels] != list(range(len(channels))):
        raise ValueError("PSF channel numbering must be consecutive for automatic assignment")
    psfs = []
    for channel in channels:
        path = PSF_DIR / channel["psf_file"]
        if not path.is_file():
            raise FileNotFoundError(path)
        psf = np.asarray(tifffile.imread(path), dtype=np.float32).squeeze()
        np.nan_to_num(psf, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.maximum(psf, 0.0, out=psf)
        psf /= psf.sum(dtype=np.float64)
        psfs.append(np.ascontiguousarray(psf))
    return channels, psfs


def open_ims(path):
    with quiet_bioformats():
        return BioImage(str(path), reader=bioio_bioformats.Reader)


def voxel_size(reader):
    with quiet_bioformats():
        size = reader.physical_pixel_sizes
    return float(size.Z), float(size.Y), float(size.X)


def read_volume(reader, time_index, channel_index, *, preserve=False):
    with quiet_bioformats():
        raw = np.asarray(
            reader.get_image_dask_data(
                "ZYX", T=time_index, C=channel_index
            ).compute()
        )
    if raw.dtype == np.int8:
        print("      WARNING: reinterpreting signed int8 bytes as uint8")
        raw = raw.view(np.uint8)
    if preserve:
        return np.ascontiguousarray(raw)
    image = np.asarray(raw, dtype=np.float32)
    np.nan_to_num(image, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.maximum(image, 0.0, out=image)
    return np.ascontiguousarray(image)


def duration(seconds):
    seconds = max(0, int(round(seconds)))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def scijava_string(value):
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def run_fiji_one_shot(image, psf, label):
    started = perf_counter()
    with TemporaryDirectory(prefix="fiji-rltv-", dir=TEMP_DIR) as directory:
        directory = Path(directory).resolve()
        (directory / "lib").mkdir()
        shutil.copyfile(SOURCE_DIR / "lib/totalvariationterm.cl", directory / "lib/totalvariationterm.cl")
        input_tif = directory / "input.tif"
        psf_tif = directory / "psf.tif"
        output_tif = directory / "output.tif"
        tifffile.imwrite(
            input_tif,
            np.ascontiguousarray(image, dtype=np.float32),
            photometric="minisblack",
            metadata={"axes": "ZYX"},
            bigtiff=True,
        )
        tifffile.imwrite(
            psf_tif,
            np.ascontiguousarray(psf, dtype=np.float32),
            photometric="minisblack",
            metadata={"axes": "ZYX"},
            bigtiff=True,
        )

        parameters = ",".join(
            (
                f"input_path={scijava_string(input_tif)}",
                f"psf_path={scijava_string(psf_tif)}",
                f"output_path={scijava_string(output_tif)}",
                f"iterations={ITERATIONS}",
                f"regularization={RLTV_REGULARIZATION:.17g}",
                f"non_circulant={'true' if RLTV_NON_CIRCULANT else 'false'}",
            )
        )
        command = [
            str(FIJI_EXECUTABLE),
            "--no-splash",
            "--headless",
            f"--mem={FIJI_MEMORY}",
            "-Dimagej.updater.disableAutocheck=true",
            "--",
            "--headless",
            "--run",
            str(SOURCE_DIR / "fiji_rltv_worker.py"),
            parameters,
        ]
        environment = os.environ.copy()
        for name in ("JAVA_HOME", "JDK_JAVA_OPTIONS", "_JAVA_OPTIONS", "CLASSPATH"):
            environment.pop(name, None)

        print(
            f"      Fiji RL-TV {label}: iterations={ITERATIONS}, "
            f"regularization={RLTV_REGULARIZATION:g}, "
            f"non_circulant={RLTV_NON_CIRCULANT}"
        )
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=environment,
            cwd=directory,
        )
        for line in process.stdout or ():
            if line := line.strip():
                print(f"        Fiji: {line}", flush=True)
        return_code = process.wait()
        if return_code:
            raise RuntimeError(f"Fiji exited with status {return_code}")
        if not output_tif.is_file():
            raise RuntimeError("Fiji did not create an output TIFF")
        result = np.asarray(tifffile.imread(output_tif), dtype=np.float32).squeeze()

    print(f"      Fiji RL-TV completed in {duration(perf_counter() - started)}")
    return np.ascontiguousarray(result)


_session = None


def run_fiji(image, psf, label, session_directory=None, input_path=None, psf_path=None,
             report_geometry=True):
    """Reuse Fiji; callers may supply immutable, already prepared TIFFs."""
    from fiji_session import FijiClient, FijiSession
    global _session
    if session_directory is None:
        key = (FIJI_EXECUTABLE.expanduser().resolve(), str(FIJI_MEMORY))
        if (_session is None or _session.closed or _session.process.poll() is not None
                or (_session.directory / "failed.json").exists()
                or (_session.executable, _session.memory) != key):
            if _session is not None:
                _session.close()
            _session = FijiSession(*key)
        session_directory = _session.directory
    with TemporaryDirectory(prefix="fiji-rltv-", dir=TEMP_DIR) as directory:
        directory = Path(directory)
        if input_path is None:
            input_path = directory / "input.tif"
            tifffile.imwrite(input_path, np.ascontiguousarray(image, dtype=np.float32),
                             photometric="minisblack", metadata={"axes": "ZYX"}, bigtiff=True)
        if psf_path is None:
            psf_path = directory / "psf.tif"
            tifffile.imwrite(psf_path, np.ascontiguousarray(psf, dtype=np.float32),
                             photometric="minisblack", metadata={"axes": "ZYX"}, bigtiff=True)
        output_path = directory / "output.tif"
        progress = ProgressLine(label)
        progress.update()
        success = False
        try:
            FijiClient(session_directory).run(
                input_path, psf_path, output_path, ITERATIONS,
                RLTV_REGULARIZATION, RLTV_NON_CIRCULANT, progress=progress.update,
                geometry=progress.geometry if report_geometry else None)
            result = np.asarray(tifffile.imread(output_path), dtype=np.float32).squeeze()
            success = True
        finally:
            progress.finish(success)
    return np.ascontiguousarray(result)


def store_result(result):
    np.nan_to_num(result, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.maximum(result, 0.0, out=result)
    return np.clip(np.rint(result), 0, 65535).astype(np.uint16)


def deconvolve(reader, output, channels, psfs, n_timepoints, n_channels):
    channel_maxima = [0] * n_channels
    for time_index in range(n_timepoints):
        for channel_index in range(n_channels):
            name = str(channels[channel_index]["name"])
            label = (
                f"T{time_index + 1}/{n_timepoints} "
                f"C{channel_index + 1}/{n_channels} {name}"
            )
            print(f"{label}: preparing input...", flush=True)
            preserve = psfs[channel_index] is None
            image = read_volume(reader, time_index, channel_index, preserve=preserve)
            if preserve:
                print(f"{label}: keeping original channel", flush=True)
                # IMS output is uint16. Widen uint8 losslessly, but never silently
                # clip or round an excluded channel of an unsupported type.
                if not np.can_cast(image.dtype, np.uint16, casting="safe"):
                    raise ValueError(
                        f"Cannot preserve {label} with dtype {image.dtype} in uint16 output")
                result = image
                stored = image.astype(np.uint16, copy=False)
                saturated = 0
            else:
                result = run_fiji(image, psfs[channel_index], label)
                stored = store_result(result)
                saturated = int(np.count_nonzero(result > 65535))
            stored_max = int(stored.max())
            channel_maxima[channel_index] = max(
                channel_maxima[channel_index], stored_max
            )
            output[time_index, channel_index] = stored
            output.flush()
            if saturated:
                print(
                    f"      WARNING: {saturated:,} voxels "
                    f"({100 * saturated / result.size:.6f}%) clipped at 65535"
                )
            del image, result, stored
            gc.collect()
    return [(0.0, float(maximum or 1)) for maximum in channel_maxima]


def preserve_original_channel_metadata(input_path, output_path, indices):
    """Restore original names, colors, display ranges and wavelengths for copies."""
    if not indices:
        return
    with h5py.File(input_path, "r") as source, h5py.File(output_path, "r+") as output:
        for index in indices:
            name = f"Channel {index}"
            metadata = f"DataSetInfo/{name}"
            if metadata in source:
                if metadata in output:
                    del output[metadata]
                source.copy(metadata, output["DataSetInfo"], name=name)


def process_file(input_path, channels, psfs):
    tag = f"iter{ITERATIONS}_reg{format(RLTV_REGULARIZATION, '.10g')}"
    stem = f"{input_path.stem}_deconvolved_{tag}"
    output_path = OUTPUT_DIR / f"{stem}.ims"
    partial_path = OUTPUT_DIR / f".{stem}.partial.ims"
    array_path = TEMP_DIR / f".{stem}.uint16.tczyx.dat"

    if output_path.exists() and not OVERWRITE:
        print(f"Skipping existing: {output_path.name}")
        return
    partial_path.unlink(missing_ok=True)
    array_path.unlink(missing_ok=True)

    reader = None
    output = None
    try:
        if CHANNEL_INDICES == []:
            shutil.copy2(input_path, partial_path)
            partial_path.replace(output_path)
            print(f"Copied unchanged (no channels selected): {output_path.name}")
            return
        reader = open_ims(input_path)
        shape = tuple(int(getattr(reader.dims, axis)) for axis in "TCZYX")
        n_t, n_c, _, _, _ = shape
        selected = list(range(n_c)) if CHANNEL_INDICES is None else CHANNEL_INDICES
        if any(index >= n_c for index in selected):
            raise ValueError(f"Selected channel is absent from image with {n_c} channels")
        if len(selected) > len(psfs):
            raise ValueError(f"Image needs {len(selected)} PSFs but only {len(psfs)} are available")
        original_channels = read_channels(input_path)
        if [channel["index"] for channel in original_channels] != list(range(n_c)):
            raise ValueError("Image channel numbering must be consecutive")
        file_channels = [dict(channel) for channel in original_channels]
        file_psfs = [None] * n_c
        for position, index in enumerate(selected):
            file_channels[index] = channels[position]
            file_psfs[index] = psfs[position]
        output_names = [channel["name"] for channel in file_channels]
        output_colors = [None] * n_c
        for index in selected:
            edit = CHANNEL_EDITS.get(str(index), CHANNEL_EDITS.get(index, {}))
            if "name" in edit:
                output_names[index] = edit["name"]
            if "color" in edit:
                output_colors[index] = parse_color(edit["color"])
        dz, dy, dx = voxel_size(reader)
        print(f"\nProcessing {input_path.name}")
        print(f"  Shape TCZYX: {shape}")
        print(f"  Voxel size ZYX: {(dz, dy, dx)} µm")
        print(f"  Source dtype: {reader.dtype}")
        print(f"  Source channels: {reader.channel_names}")

        output = np.memmap(array_path, mode="w+", dtype=np.uint16, shape=shape)
        channel_ranges = deconvolve(reader, output, file_channels, file_psfs, n_t, n_c)
        output.flush()
        reader = None
        gc.collect()

        print(f"  Writing native IMS: {output_path.name}")
        write_imaris(
            output,
            partial_path,
            channel_names=output_names,
            channel_colors=output_colors,
            channel_ranges=channel_ranges,
            excitation_wavelengths_nm=[
                channel.get("excitation_nm") for channel in file_channels
            ],
            emission_wavelengths_nm=[
                channel.get("emission_nm") for channel in file_channels
            ],
            source_ims=input_path,
        )
        excluded = [index for index in range(n_c) if index not in selected]
        preserve_original_channel_metadata(input_path, partial_path, excluded)
        output_path.unlink(missing_ok=True)
        partial_path.replace(output_path)
        print(f"Finished: {output_path.name}")
    finally:
        reader = None
        if output is not None:
            output.flush()
            del output
        gc.collect()
        array_path.unlink(missing_ok=True)
        partial_path.unlink(missing_ok=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    for path in (() if CHANNEL_INDICES == [] else (
        FIJI_EXECUTABLE,
        SOURCE_DIR / "fiji_rltv_worker.py",
        SOURCE_DIR / "lib/totalvariationterm.cl",
    )):
        if not path.is_file():
            raise FileNotFoundError(path)

    channels, psfs = load_psfs()
    input_files = sorted(INPUT_DIR.glob("*.ims"))
    if not input_files:
        raise FileNotFoundError(f"No .ims files found in {INPUT_DIR}")

    print(f"Loaded {len(psfs)} channel PSFs")
    print(f"Found {len(input_files)} IMS files")
    print(
        f"Fiji RL-TV: iterations={ITERATIONS}, "
        f"regularization={RLTV_REGULARIZATION:g}, "
        f"non_circulant={RLTV_NON_CIRCULANT}"
    )
    print(f"Output: {OUTPUT_DIR}")

    failed = []
    for input_path in input_files:
        try:
            process_file(input_path, channels, psfs)
        except Exception:
            print(f"FAILED: {input_path.name}", file=sys.stderr)
            traceback.print_exc()
            failed.append(input_path.name)
    if failed:
        raise SystemExit(f"Batch finished with {len(failed)} failed files: {failed}")
    print("\nBatch complete")


if __name__ == "__main__":
    main()
