import importlib
import json
from pathlib import Path
import signal
import sys

from app_version import __version__

ROOT = Path(__file__).resolve().parent


def run_worker():
    def stop(signum, frame):
        raise SystemExit(130)

    signal.signal(signal.SIGTERM, stop)
    sys.path.insert(0, str(ROOT / "source"))
    settings = json.load(sys.stdin)
    module = importlib.import_module(settings.pop("module"))
    for name, value in settings.items():
        if isinstance(getattr(module, name), Path):
            value = Path(value).expanduser()
        setattr(module, name, value)
    module.main()


def main():
    import tkinter as tk
    from source.application import Application

    window = tk.Tk()
    app = Application(window)
    window.update_idletasks()
    window.lift()
    if sys.platform == "darwin":
        from AppKit import NSApplication, NSApplicationActivateIgnoringOtherApps, NSRunningApplication
        for native_window in NSApplication.sharedApplication().windows():
            if native_window.title() == window.title() and native_window.screen() is not None:
                native_window.setFrame_display_(native_window.screen().visibleFrame(), True)
                break
        NSRunningApplication.currentApplication().activateWithOptions_(NSApplicationActivateIgnoringOtherApps)
    window.focus_force()
    window.mainloop()


if __name__ == "__main__":
    if "--version" in sys.argv:
        print(__version__)
    else:
        run_worker() if "--worker" in sys.argv else main()
