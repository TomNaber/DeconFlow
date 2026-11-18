"""Display-paced Tk rendering and elapsed-time wheel zoom transitions."""
import math
import sys
from time import monotonic

from source.platform_support import display_refresh_rate


class DisplayClock:
    """Shared by a window; re-query on moves and once a second while active."""
    def __init__(self, window):
        self.window = window
        self.geometry = None
        self.checked = -math.inf
        self.hz = 60.0

    def interval(self):
        now = monotonic()
        geometry = (self.window.winfo_rootx(), self.window.winfo_rooty(),
                    self.window.winfo_width(), self.window.winfo_height())
        if geometry != self.geometry or now - self.checked >= 1:
            self.hz = display_refresh_rate(self.window)
            self.geometry, self.checked = geometry, now
        return 1 / self.hz


class FrameScheduler:
    """One pending frame, no catch-up queue, no fixed refresh-rate ceiling."""
    def __init__(self, widget, draw):
        self.widget, self.draw = widget, draw
        window = widget.winfo_toplevel()
        if not hasattr(window, '_display_clock'):
            window._display_clock = DisplayClock(window)
        self.clock = window._display_clock
        self.job = None
        self.deadline = 0.0
        self.closed = False
        widget.bind('<Destroy>', self._destroy, add='+')

    def request(self):
        if self.closed or self.job is not None:
            return
        interval = self.clock.interval()
        now = monotonic()
        # A newly faster display must not inherit a slower display's deadline.
        self.deadline = min(self.deadline, now + interval)
        delay = max(0, math.ceil((self.deadline - now) * 1000))
        self.job = self.widget.after(delay, self._run)

    def _run(self):
        self.job = None
        interval = self.clock.interval()
        now = monotonic()
        if self.deadline <= 0 or now - self.deadline > interval:
            self.deadline = now
        self.deadline += interval
        self.draw()
        # Retain fractional deadlines (e.g. 6.944 ms at 144 Hz). If rendering
        # overruns, the next request runs once, rather than replaying frames.

    def cancel(self):
        if self.job is not None:
            self.widget.after_cancel(self.job)
            self.job = None

    def close(self):
        self.cancel()
        self.closed = True

    def _destroy(self, event):
        if event.widget is self.widget:
            self.close()


class WheelZoom:
    """Logarithmic zoom interpolation; input accumulates against the target."""
    def __init__(self, get_value, apply, schedule, limits):
        self.get_value, self.apply = get_value, apply
        self.schedule, self.limits = schedule, limits
        self.target = None
        self.anchor = ()

    def cancel(self):
        self.target = None

    def add(self, steps, *anchor):
        low, high = self.limits()
        base = self.target if self.target is not None else self.get_value()
        self.start = self.get_value()
        self.target = min(high, max(low, base * 1.1 ** max(-10, min(10, steps))))
        self.anchor = anchor
        self.started = monotonic()
        self.schedule()

    def step(self):
        if self.target is None:
            return
        progress = min(1.0, (monotonic() - self.started) / .080)
        target = self.target
        eased = 1 - (1 - progress) ** 3
        value = target if progress == 1 else math.exp(
            math.log(self.start) + eased * math.log(target / self.start))
        if progress == 1:
            self.target = None
        self.apply(value, *self.anchor)

    def finish(self):
        if self.target is not None:
            target, self.target = self.target, None
            self.apply(target, *self.anchor)


def wheel_steps(event):
    number = getattr(event, 'num', None)
    if number in (4, 5):
        return 1.0 if number == 4 else -1.0
    delta = float(event.delta)
    return delta if sys.platform == 'darwin' else delta / 120


def precise_wheel(event):
    if sys.platform == 'darwin' and getattr(event, 'num', None) not in (4, 5):
        try:
            from AppKit import NSApplication
            native = NSApplication.sharedApplication().currentEvent()
            return bool(native and native.hasPreciseScrollingDeltas())
        except (ImportError, AttributeError):
            pass
    return False


def scroll_pixels(canvas, event):
    """Retain subpixel deltas without making scrollbar drags lag behind input."""
    region = canvas.tk.splitlist(canvas.cget('scrollregion'))
    if len(region) != 4:
        return
    top, bottom = float(region[1]), float(region[3])
    extent = bottom - top
    if extent <= 0:
        return
    first, last = canvas.yview()
    if first == 0 and last == 1:
        canvas._scroll_fraction = 0.0
        return
    # Tk reports smaller deltas on macOS, including precise trackpad events.
    pixels = -wheel_steps(event) * (2 if sys.platform == 'darwin' else 32)
    actual = first * extent
    previous = getattr(canvas, '_scroll_position', None)
    remainder = getattr(canvas, '_scroll_fraction', 0.0) if previous == first else 0.0
    target = min(max(0.0, extent * (1 - (last - first))), max(0.0, actual + pixels + remainder))
    position = round(target)
    canvas.yview_moveto(position / extent)
    canvas._scroll_position = canvas.yview()[0]
    canvas._scroll_fraction = target - position
