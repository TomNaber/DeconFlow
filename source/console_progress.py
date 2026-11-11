"""Single-line elapsed-time progress for the terminal and application console."""
import sys
from contextlib import contextmanager
from threading import Event, Thread, RLock
from time import perf_counter as monotonic


class ProgressLine:
    def __init__(self, label, activity='Richardson-Lucy', *, delay=0, total_started=None, timings=None):
        self.label = label
        self.activity = activity
        self.started = monotonic()
        self.last_update = None
        self.width = 0
        self.delay = delay
        self.total_started = total_started
        self.timings = timings
        self.lock = RLock()

    def update(self):
        with self.lock:
            now = monotonic()
            if now - self.started < self.delay:
                return
            if self.last_update is not None and now - self.last_update < 1:
                return
            self.last_update = now
            total = f' Channel total {now - self.total_started:.1f}s.' if self.total_started is not None else ''
            self.write(f'{self.label} {self.activity}. {now - self.started:.0f}s elapsed.{total}')

    @contextmanager
    def running(self):
        """Keep elapsed time live for operations without progress callbacks."""
        stopped = Event()

        def refresh():
            while not stopped.wait(1):
                self.update()

        self.update()
        ticker = Thread(target=refresh, daemon=True)
        ticker.start()
        success = False
        try:
            yield
            success = True
        finally:
            stopped.set()
            ticker.join()
            self.finish(success)

    def write(self, text, end=''):
        sys.stdout.write('\r' + text.ljust(self.width) + end)
        sys.stdout.flush()
        self.width = len(text)

    def geometry(self, value):
        with self.lock:
            if value['cache'] != 'new':
                return
            dimensions = lambda name: ' x '.join(str(n) for n in value[name])
            self.write(f"Generating FFT cache. Crop XYZ {dimensions('crop')}. "
                       f"PSF XYZ {dimensions('psf')}. Padded FFT XYZ {dimensions('fft')}.", end='\n')
            self.width = 0
            self.last_update = None
            self.update()

    def finish(self, success):
        elapsed = monotonic() - self.started
        if self.timings is not None:
            self.timings[self.activity] = self.timings.get(self.activity, 0.0) + elapsed
        if success and not self.width and elapsed < self.delay:
            return
        status = 'completed' if success else 'failed'
        self.write(f'{self.label} {self.activity} {status} in {elapsed:.1f}s.', end='\n')
