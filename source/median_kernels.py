"""Median neighborhoods displayed in XYZ order and applied to ZYX volumes."""

MEDIAN_KERNELS = ('3x3x1', '3x3x3', '5x5x1', '5x5x5', '7x7x1', '7x7x7')


def kernel_label(value=3):
    """Normalize a kernel, including legacy scalar cubic sizes."""
    label = str(value)
    if label in ('3', '5', '7'):
        label = 'x'.join([label] * 3)
    if label not in MEDIAN_KERNELS:
        raise ValueError('Select a median kernel of ' + ', '.join(MEDIAN_KERNELS) + ' (X×Y×Z).')
    return label


def kernel_zyx(value=3):
    return tuple(int(size) for size in reversed(kernel_label(value).split('x')))
