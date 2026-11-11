#!/usr/bin/env python3

import gc
import json
import os
import shutil
import subprocess
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir
from time import perf_counter

import bioio_bioformats
import h5py
import numpy as np
import tifffile
from bioio import BioImage

from imagej_runtime import launch_command, java_environment, runtime_info
from bioformats_runtime import configure_bioformats
from platform_support import process_session_options, stop_process_tree
from ims_writer import write_imaris
from edit_ims_channels import parse_color
from psf_channels import read_psf_channels, load_psf_volume
from console_progress import ProgressLine
from channel_metadata import read_channels
from batch_report import BatchReport, timestamp, sha256
from median_kernels import kernel_zyx


SOURCE_DIR = Path(__file__).resolve().parent
ROOT = SOURCE_DIR.parent
INPUT_DIR = ROOT / "examples"
PSF_DIR = ROOT / "psfs"
OUTPUT_DIR = ROOT / "dcv"
TEMP_DIR = Path(gettempdir())

ITERATIONS = 40
CHANNEL_ITERATIONS = {}
CHANNEL_REGULARIZATION = {}
POST_PROCESSING = False
CHANNEL_POST_PROCESSING = {}
RLTV_REGULARIZATION = 0.0001
RLTV_NON_CIRCULANT = True
FIJI_MEMORY = "12g"
OVERWRITE = False
CHANNEL_EDITS = {}
PSF_FILES = []
PSF_SOURCE = "external"
PSF_GENERATION_PARAMETERS = None
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
    if PSF_FILES and any(filename is None for filename in PSF_FILES):
        channels, psfs = [], []
        for filename in PSF_FILES:
            if filename is None:
                channels.append({})
                psfs.append(None)
            else:
                path = Path(filename).expanduser()
                path = path if path.is_absolute() else PSF_DIR / path
                channels.append(read_psf_channels([path])[0])
                psfs.append(load_psf_volume(path))
        return channels, psfs
    paths = [Path(filename).expanduser() for filename in PSF_FILES]
    channels = read_psf_channels([path if path.is_absolute() else PSF_DIR / path for path in paths]
                                 if paths else PSF_DIR)
    if paths:
        # Preserve repeated assignments when one PSF is used for multiple channels.
        by_file = {channel["psf_file"]: channel for channel in channels}
        channels = [by_file[str((path if path.is_absolute() else PSF_DIR / path).resolve())]
                    for path in paths]
    else:
        if any(channel["index"] is None for channel in channels):
            raise ValueError("Assign unnumbered or ambiguous PSFs explicitly")
        channels.sort(key=lambda channel: channel["index"])
        if not channels or [channel["index"] for channel in channels] != list(range(len(channels))):
            raise ValueError("PSF channel numbering must be consecutive for automatic assignment")
    psfs = [load_psf_volume(channel["psf_file"]) for channel in channels]
    return channels, psfs


def open_ims(path):
    configure_bioformats()
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


def read_fiji_result(path, shape):
    """Restore input axes even when ImageJ omits singleton TIFF dimensions."""
    result = np.asarray(tifffile.imread(path), dtype=np.float32)
    if result.size != np.prod(shape):
        raise ValueError("Deconvolution returned an unexpected image size.")
    return np.ascontiguousarray(result.reshape(shape))


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

        command = launch_command(FIJI_MEMORY, SOURCE_DIR / "fiji_rltv_worker.py", {
            'input_path': input_tif, 'psf_path': psf_tif, 'output_path': output_tif,
            'iterations': ITERATIONS, 'regularization': RLTV_REGULARIZATION,
            'non_circulant': RLTV_NON_CIRCULANT,
        })
        environment = java_environment()

        print(
            f"      ImageJ RL-TV {label}: iterations={ITERATIONS}, "
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
            **process_session_options(),
        )
        try:
            for line in process.stdout or ():
                if line := line.strip():
                    print(f"        ImageJ: {line}", flush=True)
            return_code = process.wait()
        finally:
            if process.poll() is None:
                stop_process_tree(process.pid, force=True)
                process.wait()
            process.stdout.close()
        if return_code:
            raise RuntimeError(f"ImageJ exited with status {return_code}")
        if not output_tif.is_file():
            raise RuntimeError("ImageJ did not create an output TIFF")
        result = read_fiji_result(output_tif, image.shape)

    print(f"      ImageJ RL-TV completed in {duration(perf_counter() - started)}")
    return np.ascontiguousarray(result)


_session = None


def run_fiji(image, psf, label, session_directory=None, input_path=None, psf_path=None,
             report_geometry=True, iterations=None, regularization=None):
    """Reuse ImageJ; callers may supply immutable, already prepared TIFFs."""
    iterations = ITERATIONS if iterations is None else iterations
    regularization = RLTV_REGULARIZATION if regularization is None else regularization
    from fiji_session import FijiClient, FijiSession
    global _session
    if session_directory is None:
        key = str(FIJI_MEMORY)
        if (_session is None or _session.closed or _session.process.poll() is not None
                or (_session.directory / "failed.json").exists()
                or _session.memory != key):
            if _session is not None:
                _session.close()
            _session = FijiSession(key)
        session_directory = _session.directory
    with TemporaryDirectory(prefix="fiji-rltv-", dir=TEMP_DIR) as directory:
        directory = Path(directory)
        if input_path is None:
            shape = image.shape
            input_path = directory / "input.tif"
            tifffile.imwrite(input_path, np.ascontiguousarray(image, dtype=np.float32),
                             photometric="minisblack", metadata={"axes": "ZYX"}, bigtiff=True)
        else:
            with tifffile.TiffFile(input_path) as input_tif:
                series = input_tif.series[0]
                # TIFF series omit a singleton Z even for prepared ZYX inputs.
                shape = (1, *series.shape) if series.axes == 'YX' else series.shape
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
                input_path, psf_path, output_path, iterations,
                regularization, RLTV_NON_CIRCULANT, progress=progress.update,
                geometry=progress.geometry if report_geometry else None)
            result = read_fiji_result(output_path, shape)
            success = True
        finally:
            progress.finish(success)
    return np.ascontiguousarray(result)


def post_process(result, settings, spacing_zyx=None):
    """Filter ZYX data, using its voxel spacing for physical XY widths.

    Legacy direct callers may omit spacing and supply xy_pixel_size_um instead.
    """
    method = settings.get('filter', 'None')
    if method == 'None':
        return result
    if method in ('Gaussian filter', 'Bilateral filter'):
        from scipy.ndimage import gaussian_filter
        if spacing_zyx is None:
            dy = dx = float(settings['xy_pixel_size_um'])
        else:
            _, dy, dx = (float(value) for value in spacing_zyx)
        sigma_um = float(settings['sigma_um'])
        if not all(np.isfinite(value) and value > 0 for value in (dy, dx, sigma_um)):
            raise ValueError('Filter width and XY pixel size must be finite and positive.')
        # Smooth XY independently in each Z plane; width is the physical standard deviation.
        sigma_yx = (sigma_um / dy, sigma_um / dx)
        if method == 'Bilateral filter':
            from bilateral_filter import bilateral_filter
            return bilateral_filter(result, sigma_yx, float(settings.get('sigma_intensity', 1000.0)))
        return gaussian_filter(np.asarray(result, dtype=np.float32),
                               sigma=(0,) * (result.ndim - 2) + sigma_yx, mode='reflect')
    if method != 'Median filter':
        raise ValueError('Select None, Median filter, Gaussian filter or Bilateral filter.')
    from scipy.ndimage import median_filter
    return median_filter(result, size=kernel_zyx(settings.get('kernel', 3)), mode='reflect')


def store_result(result):
    np.nan_to_num(result, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.maximum(result, 0.0, out=result)
    return np.clip(np.rint(result), 0, 65535).astype(np.uint16)


def deconvolve(reader, output, channels, psfs, n_timepoints, n_channels):
    spacing_zyx = voxel_size(reader)
    channel_maxima = [0] * n_channels
    for time_index in range(n_timepoints):
        for channel_index in range(n_channels):
            name = str(channels[channel_index]["name"])
            label = (
                f"T{time_index + 1}/{n_timepoints} "
                f"C{channel_index + 1}/{n_channels} {name}"
            )
            print(f"{label}: preparing input...", flush=True)
            post = CHANNEL_POST_PROCESSING.get(str(channel_index), {})
            filter_enabled = POST_PROCESSING and post.get('filter', 'None') != 'None'
            preserve = psfs[channel_index] is None and not filter_enabled
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
                result = image if psfs[channel_index] is None else run_fiji(image, psfs[channel_index], label,
                                  iterations=CHANNEL_ITERATIONS.get(str(channel_index), ITERATIONS),
                                  regularization=CHANNEL_REGULARIZATION.get(str(channel_index), RLTV_REGULARIZATION))
                if POST_PROCESSING:
                    result = post_process(result, post, spacing_zyx=spacing_zyx)
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


def process_file(input_path, channels, psfs, record=None):
    record = {} if record is None else record
    counts = {int(key): value for key, value in CHANNEL_ITERATIONS.items()}
    iteration_tag = (str(next(iter(counts.values()))) if counts and len(set(counts.values())) == 1
                     else "_".join(f"C{key + 1}-{value}" for key, value in sorted(counts.items()))
                     if counts else str(ITERATIONS))
    regularizations = {int(key): value for key,value in CHANNEL_REGULARIZATION.items()}
    regularization_tag = (format(next(iter(regularizations.values())), '.10g')
                          if regularizations and len(set(regularizations.values())) == 1
                          else "_".join(f"C{key+1}-{value:.10g}" for key,value in sorted(regularizations.items()))
                          if regularizations else format(RLTV_REGULARIZATION, '.10g'))
    tag = f"iter{iteration_tag}_reg{regularization_tag}"
    stem = f"{input_path.stem}_deconvolved_{tag}"
    output_path = OUTPUT_DIR / f"{stem}.ims"
    partial_path = OUTPUT_DIR / f".{stem}.partial.ims"
    array_path = TEMP_DIR / f".{stem}.uint16.tczyx.dat"

    record['save_location'] = str(output_path.resolve())
    try:
        originals = read_channels(input_path)
        selected_indices = list(range(len(originals))) if CHANNEL_INDICES is None else CHANNEL_INDICES
        record['channels'] = []
        for original in originals:
            index = original['index']
            selected_channel = index in selected_indices
            position = selected_indices.index(index) if selected_channel else None
            assigned = channels[position] if position is not None and position < len(channels) else {}
            edit = CHANNEL_EDITS.get(str(index), CHANNEL_EDITS.get(index, {})) if selected_channel else {}
            record['channels'].append(dict(
                index=index, input_name=original['name'],
                requested_output_name=edit.get('name', assigned.get('name', original['name'])),
                action=('deconvolve' if psfs[position] is not None else 'filter') if selected_channel else 'preserve',
                psf_assignment_position=position,
                iterations=CHANNEL_ITERATIONS.get(str(index), ITERATIONS) if selected_channel and psfs[position] is not None else None,
                regularization=CHANNEL_REGULARIZATION.get(str(index), RLTV_REGULARIZATION) if selected_channel and psfs[position] is not None else None,
                requested_color=edit.get('color', 'original')))
        record['channel_count'] = len(originals)
    except Exception as error:
        record['metadata_error'] = str(error)

    if output_path.exists() and not OVERWRITE:
        record.update(status='skipped', reason='Output exists; overwrite disabled. Requested settings were not applied.')
        try:
            record['existing_output_channels'] = read_channels(output_path)
        except Exception as error:
            record['existing_output_metadata_error'] = str(error)
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
            record.update(status="completed", action="copied unchanged")
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
            file_channels[index] = channels[position] or file_channels[index]
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
        record.update(shape_tczyx=shape, voxel_size_zyx_um=(dz, dy, dx),
                      source_dtype=str(reader.dtype), output_channel_names=output_names)
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
        record.update(status="completed")
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
    names = ('INPUT_DIR', 'OUTPUT_DIR', 'TEMP_DIR', 'PSF_DIR', 'PSF_FILES',
             'PSF_SOURCE', 'ITERATIONS', 'CHANNEL_ITERATIONS', 'CHANNEL_REGULARIZATION',
             'RLTV_REGULARIZATION', 'RLTV_NON_CIRCULANT',
             'FIJI_MEMORY', 'OVERWRITE', 'CHANNEL_EDITS', 'CHANNEL_INDICES',
             'POST_PROCESSING', 'CHANNEL_POST_PROCESSING')
    settings = {name: str(globals()[name].expanduser().resolve())
                if isinstance(globals()[name], Path) else globals()[name] for name in names}
    if PSF_SOURCE == 'internal':
        settings['PSF_GENERATION_PARAMETERS'] = PSF_GENERATION_PARAMETERS
    report = BatchReport(OUTPUT_DIR, settings)
    print(f"Batch report: {report.path}", flush=True)
    with report.capture():
        try:
            run_batch(report)
        except BaseException as error:
            interrupted = isinstance(error, KeyboardInterrupt) or isinstance(error, SystemExit) and error.code == 130
            report.data.update(status='interrupted' if interrupted else 'failed', error=str(error))
            traceback.print_exc()
            raise
        else:
            report.data['status'] = 'completed'
        finally:
            if _session is not None:
                ready = _session.directory / 'ready.json'
                if ready.is_file():
                    report.data.setdefault('imagej_runtime', {}).update(json.loads(ready.read_text()))
                report.data['imagej_diagnostics'] = list(_session.diagnostics)
            report.data['finished_utc'] = timestamp()
            report.save()


def run_batch(report):
    input_files = sorted(INPUT_DIR.glob("*.ims"))
    report.data['files'] = [dict(filename=path.name, input_path=str(path.resolve()),
                                 status='pending') for path in input_files]
    report.save()
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    for path in (() if CHANNEL_INDICES == [] else (
        SOURCE_DIR / "fiji_rltv_worker.py",
        SOURCE_DIR / "lib/totalvariationterm.cl",
    )):
        if not path.is_file():
            raise FileNotFoundError(path)
    report.data['source_sha256'] = {
        str(path.relative_to(ROOT)): sha256(path) for path in (
            Path(__file__), ROOT / 'app_version.py', SOURCE_DIR / 'generate_gibson_lanni_psfs.py',
            SOURCE_DIR / 'fiji_session_worker.py', SOURCE_DIR / 'imagej_runtime.py',
            SOURCE_DIR / 'imagej/HeadlessImageJ.java', SOURCE_DIR / 'ims_writer.py',
            SOURCE_DIR / 'lib/totalvariationterm.cl')}
    if CHANNEL_INDICES != [] and (not PSF_FILES or any(PSF_FILES)):
        report.data['imagej_runtime'] = runtime_info()
    channels, psfs = load_psfs()
    report.record_psfs(channels, psfs)
    if not input_files:
        raise FileNotFoundError(f"No .ims files found in {INPUT_DIR}")

    print(f"Loaded {len(psfs)} channel PSFs")
    print(f"Found {len(input_files)} IMS files")
    print(f"Output: {OUTPUT_DIR}")
    failed = []
    for input_path, record in zip(input_files, report.data['files']):
        record.update(status='running', started_utc=timestamp())
        report.save()
        try:
            stat = input_path.stat()
            record.update(input_size_bytes=stat.st_size, input_mtime_ns=stat.st_mtime_ns)
            process_file(input_path, channels, psfs, record)
        except Exception as error:
            record.update(status='failed', error=str(error), traceback=traceback.format_exc())
            print(f"FAILED: {input_path.name}", file=sys.stderr)
            traceback.print_exc()
            failed.append(input_path.name)
        except BaseException:
            record['status'] = 'interrupted'
            raise
        finally:
            record['finished_utc'] = timestamp()
            report.save()
    if failed:
        raise RuntimeError(f"Batch finished with {len(failed)} failed files: {failed}")
    print("\nBatch complete")


if __name__ == "__main__":
    main()
