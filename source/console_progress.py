"""Single-line elapsed-time progress for the terminal and application console."""
import sys
from time import monotonic


class ProgressLine:
    def __init__(self, label):
        self.label = label
        self.started = monotonic()
        self.last_update = None
        self.width = 0

    def update(self):
        now = monotonic()
        if self.last_update is not None and now - self.last_update < 1:
            return
        self.last_update = now
        # Native FFT setup has no completion callback, so elapsed time includes it.
        self.write(f'{self.label} Richardson-Lucy running. {int(now - self.started)}s elapsed.')

    def write(self, text, end=''):
        sys.stdout.write('\r' + text.ljust(self.width) + end)
        sys.stdout.flush()
        self.width = len(text)

    def geometry(self, value):
        dimensions = lambda name: ' x '.join(str(n) for n in value[name])
        self.write(f"{self.label}. Crop XYZ {dimensions('crop')}. "
                   f"PSF XYZ {dimensions('psf')}. Padded FFT XYZ {dimensions('fft')}. "
                   f"FFT cache {value['cache']}.", end='\n')
        self.width = 0
        self.last_update = None
        self.update()

    def finish(self, success):
        status = 'completed' if success else 'failed'
        self.write(f'{self.label} {status} in {monotonic() - self.started:.1f}s.', end='\n')
