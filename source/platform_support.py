"""Platform-specific file locks, and owned process cleanup."""

from contextlib import contextmanager
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
