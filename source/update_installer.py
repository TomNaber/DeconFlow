"""Detached update helper, copied outside the installation before the GUI exits."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


def wait_for_exit(pid):
    if sys.platform == 'win32':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x00100000, False, pid)
        if not handle:
            if ctypes.get_last_error() == 87:  # Process already exited.
                return
            raise OSError('Cannot wait for DeconFlow to exit.')
        try:
            if kernel.WaitForSingleObject(handle, 120000) != 0:
                raise TimeoutError('DeconFlow did not exit; installation cancelled.')
        finally:
            kernel.CloseHandle(handle)
    else:
        for _ in range(240):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.5)
        raise TimeoutError('DeconFlow did not exit; installation cancelled.')


def install_macos(installer, destination):
    mount = Path(tempfile.mkdtemp(prefix='deconflow-mount-'))
    stage = Path(tempfile.mkdtemp(prefix='.deconflow-update-', dir=destination.parent))
    incoming, backup = stage / 'DeconFlow.app', stage / 'previous.app'
    mounted = False
    try:
        subprocess.run(['/usr/bin/hdiutil', 'attach', str(installer), '-readonly', '-nobrowse',
                        '-mountpoint', str(mount)], check=True)
        mounted = True
        source = mount / 'DeconFlow.app'
        if not (source / 'Contents/MacOS/DeconFlow').is_file():
            raise ValueError('The disk image contains no DeconFlow application.')
        subprocess.run(['/usr/bin/ditto', str(source), str(incoming)], check=True)
        destination.rename(backup)
        try:
            incoming.rename(destination)
            subprocess.run(['/usr/bin/open', '-n', str(destination)], check=True)
        except Exception:
            if destination.exists():
                shutil.rmtree(destination)
            backup.rename(destination)
            raise
    finally:
        if mounted:
            subprocess.run(['/usr/bin/hdiutil', 'detach', str(mount)], check=False)
        # Preserve the backup if restoration itself failed.
        if not backup.exists() or destination.exists():
            shutil.rmtree(stage, ignore_errors=True)
        try:
            mount.rmdir()
        except OSError:
            pass


def install_windows(installer, destination):
    subprocess.run([str(installer), '/VERYSILENT', '/SUPPRESSMSGBOXES', '/SP-',
                    '/NORESTART', '/NOCLOSEAPPLICATIONS', '/NORESTARTAPPLICATIONS',
                    f'/DIR={destination}', f'/LOG={installer.parent / "installer.log"}'], check=True)
    # Invoke PowerShell directly; avoid cmd.exe interpreting installation-path characters.
    subprocess.Popen(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass',
                      '-File', str(destination / 'Launch.ps1')], cwd=destination,
                     creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    pid, installer, destination = int(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    try:
        wait_for_exit(pid)
        if sys.platform == 'darwin':
            install_macos(installer, destination)
        elif sys.platform == 'win32':
            install_windows(installer, destination)
        else:
            raise RuntimeError('Unsupported update platform.')
        installer.unlink(missing_ok=True)
    except Exception as error:
        import traceback
        traceback.print_exc()
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror('DeconFlow update failed',
                f'{error}\n\nUpdate logs: {installer.parent}\n'
                'You can retry the downloaded installer or reopen DeconFlow.', parent=root)
            root.destroy()
        finally:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
