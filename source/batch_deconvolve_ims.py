#!/usr/bin/env python3

import gc
import json
import shutil
import subprocess
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir
from time import perf_counter

import h5py
import numpy as np
import tifffile

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from source.scene_io import Dataset, discover
from source.scene_export import destination, encode, output_dtype, publish, validate_output, write_scene

from imagej_runtime import launch_command, java_environment, runtime_info
from platform_support import process_session_options, stop_process_tree
from edit_ims_channels import parse_color
from psf_channels import read_psf_channels, load_psf_volume
from console_progress import ProgressLine
from batch_report import BatchReport, timestamp, sha256
from median_kernels import kernel_zyx


SOURCE_DIR = Path(__file__).resolve().parent
ROOT = SOURCE_DIR.parent
INPUT_DIR = ROOT / "examples"
PREVIEW_HANDOFF = None
_preview = None
INPUT_FILES = None  # Explicit input paths, or None to process the entire input folder.
PSF_DIR = ROOT / "psfs"
OUTPUT_DIR = ROOT / "dcv"
OUTPUT_FORMAT = "Follow input"
SPACING_FALLBACK = (None,None,None)
TEMP_DIR = Path(gettempdir())

ITERATIONS = 20
CHANNEL_ITERATIONS = {}
CHANNEL_REGULARIZATION = {}
POST_PROCESSING = False
CHANNEL_POST_PROCESSING = {}
RLTV_REGULARIZATION = 0.002
RLTV_NON_CIRCULANT = True
FIJI_MEMORY = "12g"
FIJI_SESSION = None
OVERWRITE = False
CHANNEL_EDITS = {}
PSF_FILES = []
PSF_SOURCE = "external"
PSF_GENERATION_PARAMETERS = None
# None preserves automatic all-channel processing for command-line callers.
CHANNEL_INDICES = None


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


def read_volume(reader, time_index, channel_index, *, preserve=False):
    raw = reader.volume(time_index, channel_index)
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
             report_geometry=True, iterations=None, regularization=None, timings=None, total_started=None):
    """Reuse ImageJ; callers may supply immutable, already prepared TIFFs."""
    iterations = ITERATIONS if iterations is None else iterations
    regularization = RLTV_REGULARIZATION if regularization is None else regularization
    from fiji_session import FijiClient, FijiSession
    global _session
    if session_directory is None:
        session_directory = FIJI_SESSION
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
        with ProgressLine(label, 'Preparing ImageJ input', delay=1, timings=timings,
                          total_started=total_started).running():
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
            FijiClient(session_directory).wait_ready()
        progress = ProgressLine(label, timings=timings, total_started=total_started)
        with progress.running():
            FijiClient(session_directory).run(
                input_path, psf_path, output_path, iterations,
                regularization, RLTV_NON_CIRCULANT,
                geometry=progress.geometry if report_geometry else None)
        with ProgressLine(label, 'Reading deconvolved result', delay=1, timings=timings,
                          total_started=total_started).running():
            result = read_fiji_result(output_path, shape)
    return np.ascontiguousarray(result)


def post_process(result, settings, spacing_zyx=None, *, backend='simpleitk'):
    """Filter ZYX data, using its voxel spacing for physical XY widths.

    Legacy direct callers may omit spacing and supply xy_pixel_size_um instead.
    SimpleITK is the application default. Legacy is retained for reproducibility
    through the comparison utility's explicit backend argument.
    """
    if backend == 'simpleitk':
        from simpleitk_filters import post_process as simpleitk_post_process
        return simpleitk_post_process(result, settings, spacing_zyx)
    if backend != 'legacy':
        raise ValueError(f'Unknown post-processing backend: {backend}')
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


def deconvolve(reader, output, channels, psfs, n_timepoints, n_channels, record=None,
               *, file_index=1, n_files=1):
    spacing_zyx = tuple(v or fallback for v,fallback in zip(reader.scene.spacing, SPACING_FALLBACK))
    channel_maxima = [-np.inf] * n_channels
    for time_index in range(n_timepoints):
        for channel_index in range(n_channels):
            name = str(channels[channel_index]["name"])
            label = (
                f"File {file_index}/{n_files} "
                + (f"T{time_index + 1}/{n_timepoints} " if n_timepoints > 1 else "")
                + f"C{channel_index + 1}/{n_channels} {name}"
            )
            channel_started = perf_counter()
            timings = {}
            post = CHANNEL_POST_PROCESSING.get(str(channel_index), {})
            filter_enabled = (CHANNEL_INDICES is None or channel_index in CHANNEL_INDICES) and POST_PROCESSING and post.get('filter', 'None') != 'None'
            preserve = psfs[channel_index] is None and not filter_enabled
            with ProgressLine(label, 'Reading input', delay=1, timings=timings,
                              total_started=channel_started).running():
                image = read_volume(reader, time_index, channel_index, preserve=preserve)
            if preserve:
                print(f"{label}: keeping original channel", flush=True)
                result = image
            else:
                result = image if psfs[channel_index] is None else run_fiji(image, psfs[channel_index], label,
                                  timings=timings, total_started=channel_started,
                                  iterations=CHANNEL_ITERATIONS.get(str(channel_index), ITERATIONS),
                                  regularization=CHANNEL_REGULARIZATION.get(str(channel_index), RLTV_REGULARIZATION))
                if _preview is not None and time_index == 0:
                    _preview.attempt(_preview.capture_raw, channel_index, result)
                if filter_enabled:
                    with ProgressLine(label, f"Post-processing ({post['filter']})",
                                      timings=timings, total_started=channel_started).running():
                        result = post_process(result, post, spacing_zyx=spacing_zyx)
                result = np.asarray(result, dtype=np.float32)
                np.nan_to_num(result, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            if preserve and _preview is not None and time_index == 0:
                _preview.attempt(_preview.capture_raw, channel_index, result)
            with ProgressLine(label, 'Writing channel output', delay=1, timings=timings,
                              total_started=channel_started).running():
                if np.issubdtype(result.dtype, np.floating) and not np.isfinite(result).all():
                    finite = result[np.isfinite(result)]
                    maximum = float(finite.max()) if finite.size else 0.0
                else:
                    maximum = float(result.max())
                channel_maxima[channel_index] = max(channel_maxima[channel_index], maximum)
                output[time_index, channel_index] = result
            del image, result
            gc.collect()
            elapsed = perf_counter() - channel_started
            timings['Other / cleanup'] = max(0.0, elapsed - sum(timings.values()))
            print(f'{label} channel completed in {elapsed:.1f}s total.', flush=True)
            if record is not None:
                record.setdefault('channel_timings', []).append(dict(
                    time_index=time_index, channel_index=channel_index,
                    total_seconds=elapsed, stages_seconds=timings))
    return channel_maxima


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


def process_file(input_path, channels, psfs, record=None, *, file_index=1, n_files=1,
                 scene_index=0, disambiguate=False):
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
    with Dataset(input_path) as reader:
        scene = reader.select(scene_index)
        family, name = destination(scene, OUTPUT_FORMAT, tag, disambiguate=disambiguate)
        output_path = OUTPUT_DIR / name
        record.update(
            save_location=str(output_path.resolve()),
            output_format=family,
            scene_index=scene.index,
            scene_name=scene.name,
            shape_tczyx=scene.shape,
            source_dtype=scene.dtype,
            voxel_size_zyx_um=scene.spacing,
        )
        if output_path.exists() and not OVERWRITE:
            record.update(status="skipped", reason="Output exists and overwrite is disabled.")
            return
        selected = list(range(scene.shape[1])) if CHANNEL_INDICES is None else CHANNEL_INDICES
        if any(i < 0 for i in selected) or len(selected) > len(psfs):
            raise ValueError(f"Channel or PSF selection does not match {scene.label}")
        file_psfs = [None] * scene.shape[1]
        originals = [dict(ch) for ch in scene.channels]
        for position, i in enumerate(selected):
            if i >= scene.shape[1]:
                if n_files == 1:
                    raise ValueError(f"Channel {i + 1} is absent from {scene.label}")
                record.setdefault("absent_selected_channels", []).append(i)
                continue
            file_psfs[i] = psfs[position]
            edit = CHANNEL_EDITS.get(str(i), CHANNEL_EDITS.get(i, {}))
            if edit.get("name"):
                scene.channels[i]["name"] = edit["name"]
            if edit.get("color") and edit["color"] != "original":
                scene.channels[i]["color"] = parse_color(edit["color"])
            for attr in ("excitation_nm", "emission_nm"):
                if channels[position].get(attr) is not None:
                    scene.channels[i][attr] = channels[position][attr]
        record["channels"] = []
        for i, ch in enumerate(scene.channels):
            deconvolving = file_psfs[i] is not None
            post_filter = CHANNEL_POST_PROCESSING.get(str(i), {}).get("filter", "None")
            filtering = i in selected and POST_PROCESSING and post_filter != "None"
            edit = CHANNEL_EDITS.get(str(i), CHANNEL_EDITS.get(i, {}))
            record["channels"].append(
                dict(
                    index=i,
                    input_name=originals[i]["name"],
                    requested_output_name=ch["name"],
                    requested_color=edit.get("color", "original"),
                    action="deconvolve" if deconvolving else "filter" if filtering else "preserve",
                    psf_assignment_position=selected.index(i) if i in selected else None,
                    iterations=CHANNEL_ITERATIONS.get(str(i), ITERATIONS) if deconvolving else None,
                    regularization=CHANNEL_REGULARIZATION.get(str(i), RLTV_REGULARIZATION)
                    if deconvolving
                    else None,
                )
            )
        record.update(
            channel_count=len(scene.channels), output_channel_names=[ch["name"] for ch in scene.channels]
        )
        dtype = output_dtype(scene.dtype)
        # A float64 staging buffer exactly represents untreated uint32 pixels and
        # processed float32 values in the same scene.
        work_dtype = (
            np.float64 if np.dtype(scene.dtype).itemsize >= 4 and scene.dtype != "float32" else np.float32
        )
        if _preview is not None:
            _preview.attempt(_preview.start, input_path, scene.index)
        try:
            with (
                TemporaryDirectory(dir=TEMP_DIR) as work,
                TemporaryDirectory(dir=OUTPUT_DIR, prefix=".deconflow-") as stage,
            ):
                staged_path = Path(stage) / name
                with mapped_volume(Path(work) / "processed", work_dtype, scene.shape) as staged:
                    maxima = deconvolve(
                        reader,
                        staged,
                        scene.channels,
                        file_psfs,
                        *scene.shape[:2],
                        record=record,
                        file_index=file_index,
                        n_files=n_files,
                    )
                    with mapped_volume(Path(work) / "encoded", dtype, scene.shape) as output:
                        factor = encode(staged, output, max(maxima))
                        record.update(
                            output_dtype=dtype.name, pre_scale_maximum=max(maxima), scale_factor=factor
                        )
                        print(
                            f"{scene.label}: {dtype.name}, maximum {max(maxima):g}, scale factor {factor:.12g}"
                        )
                        ranges = [(0.0, max(1.0, m * factor)) for m in maxima]
                        if _preview is not None:
                            for i in range(scene.shape[1]):
                                _preview.attempt(_preview.capture_result, i, output[0, i])
                        write_scene(
                            output,
                            staged_path,
                            family,
                            scene,
                            ranges,
                            record,
                            follow=OUTPUT_FORMAT == "Follow input",
                        )
                        if family == scene.family == "Imaris":
                            excluded = [i for i in range(scene.shape[1]) if i not in selected and factor == 1]
                            preserve_original_channel_metadata(input_path, staged_path, excluded)
                        validate_output(staged_path, output, scene)
                publish(staged_path, output_path, OVERWRITE)
            record["status"] = "completed"
            if _preview is not None:
                _preview.attempt(_preview.finish)
        finally:
            if _preview is not None:
                _preview.discard()


@contextmanager
def mapped_volume(path, dtype, shape):
    array = np.memmap(path, mode="w+", dtype=dtype, shape=shape)
    try:
        yield array
    finally:
        array._mmap.close()


def main():
    global _preview
    batch_started = perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    names = ('INPUT_DIR', 'INPUT_FILES', 'OUTPUT_FORMAT', 'OUTPUT_DIR', 'TEMP_DIR', 'PSF_DIR', 'PSF_FILES',
             'PSF_SOURCE', 'ITERATIONS', 'CHANNEL_ITERATIONS', 'CHANNEL_REGULARIZATION',
             'RLTV_REGULARIZATION', 'RLTV_NON_CIRCULANT',
             'FIJI_MEMORY', 'OVERWRITE', 'CHANNEL_EDITS', 'CHANNEL_INDICES',
             'POST_PROCESSING', 'CHANNEL_POST_PROCESSING')
    settings = {name: str(globals()[name].expanduser().resolve())
                if isinstance(globals()[name], Path) else globals()[name] for name in names}
    if PSF_SOURCE == 'internal':
        settings['PSF_GENERATION_PARAMETERS'] = PSF_GENERATION_PARAMETERS
    if PREVIEW_HANDOFF:
        from batch_preview import BatchPreview
        _preview = BatchPreview(PREVIEW_HANDOFF, settings)
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
            report.data['total_seconds'] = perf_counter() - batch_started
            print(f"Batch total: {report.data['total_seconds']:.1f}s.", flush=True)
            report.data['finished_utc'] = timestamp()
            report.save()


def run_batch(report):
    disambiguate = set()
    if INPUT_FILES is None:
        scenes = [(Path(s.path),s.index) for s in discover(INPUT_DIR)]
    else:
        scenes = []
        for entry in INPUT_FILES:
            if isinstance(entry, dict):
                key = Path(entry["path"]), int(entry["index"])
                scenes.append(key)
                if entry.get("disambiguate"):
                    disambiguate.add(key)
            else:
                path = Path(entry).expanduser()
                try:
                    with Dataset(path) as ds:
                        scenes.extend((path,i) for i in ds.indices)
                except Exception as error:
                    print(f"Cannot read {path.name}: {error}")
                    scenes.append((path,0))  # Record the failure and continue with other inputs.
    report.data['files'] = [dict(filename=p.name, input_path=str(p.resolve()), scene_index=i,
                                 status='pending') for p,i in scenes]
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
            Path(__file__), ROOT / 'app_version.py', SOURCE_DIR / 'generate_vectorial_psfs.py',
            SOURCE_DIR / 'fiji_session_worker.py', SOURCE_DIR / 'imagej_runtime.py',
            SOURCE_DIR / 'imagej/HeadlessImageJ.java', SOURCE_DIR / 'ims_writer.py', SOURCE_DIR / 'scene_io.py', SOURCE_DIR / 'scene_export.py',
            SOURCE_DIR / 'lib/totalvariationterm.cl')}
    if CHANNEL_INDICES != [] and (not PSF_FILES or any(PSF_FILES)):
        report.data['imagej_runtime'] = runtime_info()
    channels, psfs = load_psfs()
    report.record_psfs(channels, psfs)
    if not scenes:
        raise FileNotFoundError(f"No readable images found in {INPUT_DIR}")

    print(f"Loaded {len(psfs)} channel PSFs")
    print(f"Found {len({path for path,_ in scenes})} datasets containing {len(scenes)} images")
    print(f"Output: {OUTPUT_DIR}")
    failed = []
    for file_index, ((input_path, scene_index), record) in enumerate(zip(scenes, report.data['files']), start=1):
        file_started = perf_counter()
        record.update(status='running', started_utc=timestamp())
        report.save()
        try:
            stat = input_path.stat()
            record.update(input_size_bytes=stat.st_size, input_mtime_ns=stat.st_mtime_ns)
            process_file(input_path, channels, psfs, record,
                         file_index=file_index, n_files=len(scenes), scene_index=scene_index,
                         disambiguate=(input_path, scene_index) in disambiguate)
        except Exception as error:
            record.update(status='failed', error=str(error), traceback=traceback.format_exc())
            print(f"FAILED: {input_path.name}", file=sys.stderr)
            traceback.print_exc()
            failed.append(input_path.name)
        except BaseException:
            record['status'] = 'interrupted'
            raise
        finally:
            record['total_seconds'] = perf_counter() - file_started
            stages = record.setdefault('file_stages_seconds', {})
            stages['Channels'] = sum(item['total_seconds'] for item in record.get('channel_timings', []))
            stages['Other / cleanup'] = max(0.0, record['total_seconds'] - sum(stages.values()))
            print(f"{input_path.name} total: {record['total_seconds']:.1f}s ({record['status']}).", flush=True)
            record['finished_utc'] = timestamp()
            report.save()
    if failed:
        raise RuntimeError(f"Batch finished with {len(failed)} failed files: {failed}")
    print("\nBatch complete")


if __name__ == "__main__":
    main()
