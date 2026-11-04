import sys


def install_pinch(widget, callback):
    if sys.platform != "darwin":
        return lambda: None
    from AppKit import NSEvent, NSEventMaskMagnify

    def handle(event):
        x, y = widget.winfo_pointerxy()
        target = widget.winfo_containing(x, y)
        if callback(target, x, y, float(event.magnification())):
            return None
        return event

    monitor = NSEvent.addLocalMonitorForEventsMatchingMask_handler_(NSEventMaskMagnify, handle)
    return lambda: NSEvent.removeMonitor_(monitor)
