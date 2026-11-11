"""Edge-preserving XY smoothing without mixing Z planes."""

import numpy as np


def bilateral_filter(volume, sigma_spatial, sigma_intensity):
    """Use scalar or (Y, X) spatial sigmas in pixels, intensity in image units."""
    sigma_y, sigma_x = np.broadcast_to(np.asarray(sigma_spatial, dtype=float), (2,))
    # A fixed physical width can exceed the GUI's pixel range on finer images.
    if not all(np.isfinite(value) and value > 0 for value in (sigma_y, sigma_x)):
        raise ValueError('Bilateral spatial widths must be finite and positive.')
    if not np.isfinite(sigma_intensity) or sigma_intensity <= 0:
        raise ValueError('Bilateral intensity tolerance must be finite and positive.')
    data = np.array(volume, dtype=np.float32, copy=True)
    if data.ndim < 2 or not data.size:
        raise ValueError('Bilateral filtering requires nonempty XY images.')
    np.nan_to_num(data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    output = np.empty_like(data)
    radius_y = int(np.ceil(3 * sigma_y))
    radius_x = int(np.ceil(3 * sigma_x))
    height, width = data.shape[-2:]
    # Work plane by plane to keep temporary memory independent of stack depth.
    for plane, filtered in zip(data.reshape(-1, height, width), output.reshape(-1, height, width)):
        padded = np.pad(plane, ((radius_y, radius_y), (radius_x, radius_x)), mode='symmetric')
        total = np.zeros_like(plane)
        weights = np.zeros_like(plane)
        for dy in range(-radius_y, radius_y + 1):
            for dx in range(-radius_x, radius_x + 1):
                neighbor = padded[radius_y + dy:radius_y + dy + height,
                                  radius_x + dx:radius_x + dx + width]
                weight = (neighbor - plane) / sigma_intensity
                np.square(weight, out=weight)
                weight *= -0.5
                weight -= 0.5 * ((dx / sigma_x) ** 2 + (dy / sigma_y) ** 2)
                np.exp(weight, out=weight)
                total += weight * neighbor
                weights += weight
        np.divide(total, weights, out=filtered)
    return output
