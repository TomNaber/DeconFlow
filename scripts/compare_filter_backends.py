#!/usr/bin/env python3
"""Compare filters on a saved internal-PSF setup without changing app settings.

Use the application Python environment. Output must be a new directory.
This research comparison utility is not a repository test suite.
"""
import argparse
import configparser
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))
import numpy as np
import scipy
import SimpleITK as sitk
import tifffile
from PIL import Image, ImageDraw
import batch_deconvolve_ims as batch
import generate_gibson_lanni_psfs as generator


def metrics(old, new):
    delta = new.astype(np.float64) - old.astype(np.float64)
    changed = int(np.count_nonzero(delta))
    return dict(exact_equal=changed == 0, voxels=int(delta.size), changed_voxels=changed,
                changed_percent=100 * changed / delta.size,
                max_absolute_difference=float(np.abs(delta).max()),
                mean_absolute_difference=float(np.abs(delta).mean()),
                rmse=float(np.sqrt(np.mean(delta ** 2))), mean_difference=float(delta.mean()))


def comparison(old, new, margins):
    interior = tuple(slice(n, -n) if n else slice(None) for n in margins)
    result = dict(all=metrics(old, new))
    mask = np.ones(old.shape, dtype=bool)
    mask[interior] = False
    if old[interior].size:
        result['interior'] = metrics(old[interior], new[interior])
    if mask.any():
        result['boundary'] = metrics(old[mask], new[mask])
    return result


def figure(old, new, path, title):
    # Select by signal, not discrepancy; both images use the same display range.
    z = int(np.argmax(old.sum(axis=(1, 2), dtype=np.float64)))
    a, b = old[z].astype(float), new[z].astype(float)
    difference = np.abs(b - a)
    upper = max(float(np.percentile(np.concatenate((a.ravel(), b.ravel())), 99.5)), 1)
    dmax = float(difference.max())
    panels = [np.clip(a / upper, 0, 1), np.clip(b / upper, 0, 1), difference / (dmax or 1)]
    labels = [f'Legacy (0 .. {upper:.3g})', f'SimpleITK (0 .. {upper:.3g})', f'Absolute difference (0 .. {dmax:.3g})']
    h, w = a.shape
    canvas = Image.new('RGB', (3 * w, h + 65), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), f'{title}; Z={z}; image units; pooled 99.5th percentile display ceiling', fill='black')
    for index, (panel, label) in enumerate(zip(panels, labels)):
        canvas.paste(Image.fromarray(np.rint(panel * 255).astype('uint8')).convert('RGB'), (index * w, 60))
        draw.text((index * w + 8, 30), label, fill='black')
    canvas.save(path)
    return dict(z_index=z, image_display_max=upper, absolute_difference_display_max=dmax)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    text = args.settings.read_text()
    (args.output / 'settings.ini').write_text(text)
    config = configparser.ConfigParser(interpolation=None)
    config.read_string(text)
    if config.getboolean('PSF import', 'enabled', fallback=False):
        raise ValueError('This comparison expects the saved internal PSF setup.')
    for key, value in config.items('PSF generation'):
        name = key.upper()
        if hasattr(generator, name):
            setattr(generator, name, type(getattr(generator, name))(value))
    generator.SCAN_DIRECTION = {'up': 'down', 'down': 'up'}[config.get('PSF generation', 'scan_direction').lower()]
    definitions = json.loads(config.get('PSF generation', 'channel_rows'))
    generator.CHANNELS = sorted([dict(index=v['input_index'], name=v['name'],
                                    excitation_nm=float(v['excitation_nm']), emission_nm=float(v['emission_nm']))
                                 for v in definitions.values()], key=lambda c: c['index'])
    reader = batch.open_ims(args.input)
    spacing = batch.voxel_size(reader)
    if config.getboolean('PSF generation', 'xy_pixel_size_um_auto', fallback=True):
        generator.XY_PIXEL_SIZE_UM = spacing[2]
    if config.getboolean('PSF generation', 'z_step_um_auto', fallback=True):
        generator.Z_STEP_UM = spacing[0]
    generator.OUTPUT_DIR = args.output / 'psfs'
    generator.OUTPUT_IS_TEMPORARY = True
    generator.main()
    batch.FIJI_MEMORY = f"{round(config.getfloat('Deconvolution', 'fiji_memory') * 1024)}m"
    batch.RLTV_NON_CIRCULANT = config.getboolean('Deconvolution', 'rltv_non_circulant')
    report = dict(input=str(args.input.resolve()), input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                  spacing_zyx_um=spacing, python=sys.version, platform=platform.platform(),
                  numpy=np.__version__, scipy=scipy.__version__, simpleitk=sitk.Version_VersionString(),
                  itk=sitk.Version_ITKVersionString(), threads=sitk.ProcessObject.GetGlobalDefaultNumberOfThreads(),
                  gaussian=dict(maximum_error=0.01, maximum_kernel_width=32, use_image_spacing=True),
                  bilateral=dict(range_samples=100, automatic_kernel_size=True),
                  simpleitk_boundary='zero-flux Neumann / edge replication',
                  legacy_boundary='half-sample symmetric reflection', channels=[])
    methods = ('Gaussian filter', 'Median filter', 'Bilateral filter')
    try:
        for t in range(reader.dims.T):
            for c, method in enumerate(methods):
                pref = json.loads(config.get('Channel pseudocolor', str(c)))
                settings = dict(filter=method, kernel=pref['post_kernel'],
                                sigma_um=float(pref['post_sigma_pixels']) * generator.XY_PIXEL_SIZE_UM,
                                xy_pixel_size_um=generator.XY_PIXEL_SIZE_UM,
                                sigma_intensity=float(pref['post_sigma_intensity']))
                psf = tifffile.imread(next(generator.OUTPUT_DIR.glob(f'psf_C{c+1}_*.tif')))
                image = batch.read_volume(reader, t, c)
                started = perf_counter()
                deconvolved = batch.run_fiji(image, psf, f'Comparison C{c+1}',
                                            iterations=int(pref['iterations']), regularization=float(pref['regularization']))
                deconv_seconds = perf_counter() - started
                prefix = f'T{t+1}_C{c+1}'
                np.save(args.output / f'{prefix}_deconvolved.npy', deconvolved)
                outputs, timings = {}, {}
                for backend in ('legacy', 'simpleitk'):
                    started = perf_counter()
                    outputs[backend] = batch.post_process(deconvolved, settings, spacing, backend=backend)
                    timings[backend] = perf_counter() - started
                    np.save(args.output / f'{prefix}_{backend}_float.npy', outputs[backend])
                if method == 'Median filter':
                    margins = tuple((n - 1) // 2 for n in batch.kernel_zyx(settings['kernel']))
                else:
                    margins = (0, int(np.ceil(4 * settings['sigma_um'] / spacing[1])) + 2,
                               int(np.ceil(4 * settings['sigma_um'] / spacing[2])) + 2)
                result = dict(channel=c+1, timepoint=t+1, settings=settings, shape=list(deconvolved.shape),
                              iterations=int(pref['iterations']), regularization=float(pref['regularization']),
                              deconvolution_seconds=deconv_seconds, filter_seconds=timings, interior_margins_zyx=margins,
                              floating=comparison(outputs['legacy'], outputs['simpleitk'], margins))
                result['figure'] = figure(outputs['legacy'], outputs['simpleitk'], args.output / f'{prefix}_comparison.png', f'C{c+1} {method}')
                stored = {key: batch.store_result(value.copy()) for key, value in outputs.items()}
                result['uint16'] = comparison(stored['legacy'], stored['simpleitk'], margins)
                for backend, volume in stored.items():
                    tifffile.imwrite(args.output / f'{prefix}_{backend}_uint16.tif', volume, photometric='minisblack',
                                     metadata={'axes': 'ZYX', 'spacing': spacing[0], 'unit': 'um'},
                                     resolution=(1/spacing[2], 1/spacing[1]))
                report['channels'].append(result)
                (args.output / 'comparison.json').write_text(json.dumps(report, indent=2))
                print(json.dumps(result), flush=True)
    finally:
        if batch._session is not None:
            batch._session.close()


if __name__ == '__main__':
    main()
