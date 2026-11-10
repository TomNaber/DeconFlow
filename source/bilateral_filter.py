"""Edge-preserving XY smoothing without mixing Z planes."""

import numpy as np


def bilateral_filter(volume, sigma_spatial, sigma_intensity):
    """Use spatial sigma in pixels and intensity sigma in original image units."""
    if not np.isfinite(sigma_spatial) or not 0 < sigma_spatial <= 10:
        raise ValueError('Bilateral spatial width must be positive and at most 10 pixels.')
    if not np.isfinite(sigma_intensity) or sigma_intensity <= 0:
        raise ValueError('Bilateral intensity tolerance must be finite and positive.')
    data = np.array(volume, dtype=np.float32, copy=True)
    if data.ndim < 2 or not data.size:
        raise ValueError('Bilateral filtering requires nonempty XY images.')
    np.nan_to_num(data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    output = np.empty_like(data)
    radius = int(np.ceil(3 * sigma_spatial))
    height, width = data.shape[-2:]
    # Work plane by plane to keep temporary memory independent of stack depth.
    for plane, filtered in zip(data.reshape(-1, height, width), output.reshape(-1, height, width)):
        padded = np.pad(plane, radius, mode='symmetric')
        total = np.zeros_like(plane)
        weights = np.zeros_like(plane)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                neighbor = padded[radius + dy:radius + dy + height,
                                  radius + dx:radius + dx + width]
                weight = (neighbor - plane) / sigma_intensity
                np.square(weight, out=weight)
                weight *= -0.5
                weight -= (dx * dx + dy * dy) / (2 * sigma_spatial ** 2)
                np.exp(weight, out=weight)
                total += weight * neighbor
                weights += weight
        np.divide(total, weights, out=filtered)
    return output
