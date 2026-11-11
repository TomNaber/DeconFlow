"""SimpleITK post-processing backend for preview and batch processing.

Spatial units are micrometres, matching the image spacing supplied by the app.
SimpleITK uses XYZ coordinates; NumPy stores these images in ZYX order.
Library-native zero-flux (edge-replicating) boundaries are used, as evaluated
in docs/simpleitk-filter-comparison.md.
"""

import numpy as np
import SimpleITK as sitk

from median_kernels import kernel_zyx


GAUSSIAN_MAXIMUM_ERROR = 0.01
GAUSSIAN_MAXIMUM_KERNEL_WIDTH = 32
BILATERAL_RANGE_SAMPLES = 100


def post_process(result, settings, spacing_zyx=None):
    """Apply the selected filter with the existing post-processing interface.

Gaussian uses ITK's discrete Gaussian, not SciPy's sampled continuous kernel.
Bilateral uses ITK's automatic spatial support and range-weight lookup table.
Neither method rescales image intensities. No filtering crosses Z for these
two methods; the median's selected neighbourhood may include Z.
"""
    method = settings.get('filter', 'None')
    if method == 'None':
        return result
    if method not in ('Gaussian filter', 'Median filter', 'Bilateral filter'):
        raise ValueError('Select None, Median filter, Gaussian filter or Bilateral filter.')
    data = np.asarray(result) if method == 'Median filter' else np.asarray(result, dtype=np.float32)
    if data.ndim not in (2, 3) or not data.size:
        raise ValueError('SimpleITK filtering requires a nonempty XY image or ZYX volume.')
    if method == 'Median filter':
        radius = tuple((size - 1) // 2 for size in kernel_zyx(settings.get('kernel', 3)))
        filt = sitk.MedianImageFilter()
        filt.SetRadius(list(reversed(radius[-data.ndim:])))
        return sitk.GetArrayFromImage(filt.Execute(sitk.GetImageFromArray(data)))

    if spacing_zyx is None:
        dy = dx = float(settings['xy_pixel_size_um'])
    else:
        _, dy, dx = map(float, spacing_zyx)
    sigma = float(settings['sigma_um'])
    if not all(np.isfinite(v) and v > 0 for v in (dy, dx, sigma)):
        raise ValueError('Filter width and XY pixel size must be finite and positive.')
    if method == 'Gaussian filter':
        filt = sitk.DiscreteGaussianImageFilter()
        filt.SetUseImageSpacing(True)
        filt.SetVariance(sigma ** 2)
        filt.SetMaximumError(GAUSSIAN_MAXIMUM_ERROR)
        filt.SetMaximumKernelWidth(GAUSSIAN_MAXIMUM_KERNEL_WIDTH)
    else:
        intensity = float(settings.get('sigma_intensity', 1000.0))
        if not np.isfinite(intensity) or intensity <= 0:
            raise ValueError('Bilateral intensity tolerance must be finite and positive.')
        data = np.nan_to_num(data, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
        filt = sitk.BilateralImageFilter()
        filt.SetDomainSigma(sigma)
        filt.SetRangeSigma(intensity)
        filt.SetNumberOfRangeGaussianSamples(BILATERAL_RANGE_SAMPLES)
    output = np.empty_like(data)
    height, width = data.shape[-2:]
    for plane, destination in zip(data.reshape(-1, height, width), output.reshape(-1, height, width)):
        image = sitk.GetImageFromArray(plane)
        image.SetSpacing((dx, dy))
        destination[:] = sitk.GetArrayFromImage(filt.Execute(image))
    return output
