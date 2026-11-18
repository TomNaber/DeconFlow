"""Platform-specific file locks, and owned process cleanup."""

from contextlib import contextmanager
from functools import lru_cache
import errno
import os
from pathlib import Path
import signal
import subprocess
from time import monotonic, sleep


def process_session_options():
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def stop_process_tree(pid, *, force=False):
    """Stop an owned process tree; callers must check that its owner is still alive."""
    if os.name == "nt":
        # Windows has no SIGTERM equivalent for console-less Python/Java children.
        result = subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW,
        )
        if result.returncode not in (0, 128):  # 128: already exited
            raise OSError(result.stderr.decode(errors="replace").strip())
    else:
        try:
            os.killpg(pid, signal.SIGKILL if force else signal.SIGTERM)
        except ProcessLookupError:
            pass


@contextmanager
def exclusive_file_lock(path, timeout, check):
    """Serialize ImageJ clients across processes, releasing the lock on exit/crash."""
    if os.name == "nt":
        import msvcrt
    else:
        import fcntl
    with Path(path).open("a+b") as lock:
        if lock.seek(0, os.SEEK_END) == 0:
            lock.write(b"\0")
            lock.flush()
        deadline = monotonic() + timeout
        while True:
            check()
            try:
                if os.name == "nt":
                    lock.seek(0)
                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as error:
                if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    raise
                if monotonic() >= deadline:
                    raise TimeoutError("Timed out waiting for the application file lock.") from error
                sleep(0.025)
        try:
            yield
        finally:
            if os.name == "nt":
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock, fcntl.LOCK_UN)


def display_refresh_rate(window):
    """Current monitor mode; variable-rate Mac displays use their maximum rate.

    Tk's millisecond timers are best effort, not a display-link/vsync guarantee.
    Unknown platforms or unavailable display services fall back to 60 Hz.
    """
    import math
    import sys
    try:
        if sys.platform == 'darwin':
            rate = _mac_refresh_rate(window)
        elif sys.platform == 'win32':
            rate = _windows_refresh_rate(window)
        else:
            return 60.0
        return float(rate) if math.isfinite(rate) and rate > 1 else 60.0
    except (ImportError, AttributeError, OSError, ValueError, TypeError, RuntimeError):
        return 60.0


def _mac_refresh_rate(window):
    from AppKit import NSScreen

    screens = list(NSScreen.screens())
    if not screens:
        return 0
    x, y = window.winfo_rootx(), window.winfo_rooty()
    width, height = window.winfo_width(), window.winfo_height()
    # AppKit has a bottom-left origin; Tk's desktop origin is top-left.
    primary_top = screens[0].frame().origin.y + screens[0].frame().size.height
    y = primary_top - y - height

    def overlap(screen):
        frame = screen.frame()
        return (max(0, min(x + width, frame.origin.x + frame.size.width) - max(x, frame.origin.x)) *
                max(0, min(y + height, frame.origin.y + frame.size.height) - max(y, frame.origin.y)))

    screen = max(screens, key=overlap)
    display = int(screen.deviceDescription()['NSScreenNumber'])
    cg = _core_graphics()
    mode = cg.CGDisplayCopyDisplayMode(display)
    if not mode:
        return float(screen.maximumFramesPerSecond())
    try:
        rate = cg.CGDisplayModeGetRefreshRate(mode)
    finally:
        cg.CGDisplayModeRelease(mode)
    # ProMotion modes may report 0 rather than a fixed refresh frequency.
    return rate if rate > 1 else float(screen.maximumFramesPerSecond())


@lru_cache(maxsize=1)
def _core_graphics():
    import ctypes
    cg = ctypes.CDLL('/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics')
    cg.CGDisplayCopyDisplayMode.argtypes = [ctypes.c_uint32]
    cg.CGDisplayCopyDisplayMode.restype = ctypes.c_void_p
    cg.CGDisplayModeGetRefreshRate.argtypes = [ctypes.c_void_p]
    cg.CGDisplayModeGetRefreshRate.restype = ctypes.c_double
    cg.CGDisplayModeRelease.argtypes = [ctypes.c_void_p]
    cg.CGDisplayModeRelease.restype = None
    return cg


def _windows_refresh_rate(window):
    import ctypes
    from ctypes import wintypes as w

    class MonitorInfo(ctypes.Structure):
        _fields_ = [('size', w.DWORD), ('monitor', w.RECT), ('work', w.RECT),
                    ('flags', w.DWORD), ('device', w.WCHAR * 32)]

    class DevMode(ctypes.Structure):
        # The display arm of DEVMODEW's printer/display union.
        _fields_ = [('device', w.WCHAR * 32), ('spec', w.WORD), ('driver', w.WORD),
                    ('size', w.WORD), ('extra', w.WORD), ('fields', w.DWORD),
                    ('position', w.POINT), ('orientation', w.DWORD), ('fixed', w.DWORD),
                    ('color', w.SHORT), ('duplex', w.SHORT), ('resolution', w.SHORT),
                    ('tt', w.SHORT), ('collate', w.SHORT), ('form', w.WCHAR * 32),
                    ('pixels', w.WORD), ('bits', w.DWORD), ('width', w.DWORD),
                    ('height', w.DWORD), ('flags', w.DWORD), ('frequency', w.DWORD),
                    ('icm_method', w.DWORD), ('icm_intent', w.DWORD), ('media', w.DWORD),
                    ('dither', w.DWORD), ('reserved1', w.DWORD), ('reserved2', w.DWORD),
                    ('pan_width', w.DWORD), ('pan_height', w.DWORD)]

    user = ctypes.WinDLL('user32', use_last_error=True)
    user.MonitorFromWindow.argtypes = [w.HWND, w.DWORD]
    user.MonitorFromWindow.restype = w.HANDLE
    user.GetMonitorInfoW.argtypes = [w.HANDLE, ctypes.POINTER(MonitorInfo)]
    user.GetMonitorInfoW.restype = w.BOOL
    user.EnumDisplaySettingsW.argtypes = [w.LPCWSTR, w.DWORD, ctypes.POINTER(DevMode)]
    user.EnumDisplaySettingsW.restype = w.BOOL
    monitor = user.MonitorFromWindow(window.winfo_id(), 2)
    info = MonitorInfo()
    info.size = ctypes.sizeof(info)
    if not user.GetMonitorInfoW(monitor, ctypes.byref(info)):
        return 0
    mode = DevMode()
    mode.size = ctypes.sizeof(mode)
    if not user.EnumDisplaySettingsW(info.device, 0xffffffff, ctypes.byref(mode)):
        return 0
    return mode.frequency
