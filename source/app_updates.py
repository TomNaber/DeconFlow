"""GitHub release updates for native installations (Tk calls stay on the UI thread)."""
import hashlib
import json
import os
from pathlib import Path
import platform
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from tkinter import messagebox
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, build_opener, HTTPRedirectHandler

from app_version import REPOSITORY_URL, __version__


class SafeRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlparse(newurl).scheme != 'https':
            raise ValueError('Update redirects must use HTTPS.')
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and urlparse(newurl).netloc != urlparse(req.full_url).netloc:
            redirected.remove_header('Authorization')
        return redirected


def fetch(url, binary=False):
    if urlparse(url).scheme != 'https' or urlparse(url).netloc != 'api.github.com':
        raise ValueError('Invalid GitHub update URL.')
    headers = {'User-Agent': f'DeconFlow/{__version__}',
               'Accept': 'application/octet-stream' if binary else 'application/vnd.github+json'}
    token = os.environ.get('DECONFLOW_GITHUB_TOKEN')
    if token:
        headers['Authorization'] = f'Bearer {token}'
    return build_opener(SafeRedirect()).open(Request(url, headers=headers), timeout=30)


def version_tuple(value):
    if not re.fullmatch(r'v?\d+\.\d+\.\d+', value):
        raise ValueError('Expected a stable major.minor.patch version.')
    return tuple(map(int, value.removeprefix('v').split('.')))


def installation_root():
    root = Path(__file__).resolve().parents[1]
    if sys.platform == 'darwin' and root.parent.name == 'Resources':
        bundle = root.parents[2]
        if (bundle / 'Contents/MacOS/DeconFlow').is_file():
            return bundle
    if sys.platform == 'win32' and (root.parent / 'Launch.cmd').is_file():
        return root.parent
    return None


def latest_release():
    repo = REPOSITORY_URL.removeprefix('https://github.com/')
    with fetch(f'https://api.github.com/repos/{repo}/releases/latest') as response:
        release = json.load(response)
    if release.get('draft') or release.get('prerelease'):
        return None
    tag = release['tag_name']
    if version_tuple(tag) <= version_tuple(__version__):
        return None
    arch = platform.machine().lower().replace('amd64', 'x86_64')
    system = {'darwin': 'macos', 'win32': 'windows'}[sys.platform]
    suffix = '.dmg' if system == 'macos' else '-Setup.exe'
    name = f'DeconFlow-{tag.removeprefix("v")}-{system}-{arch}{suffix}'
    assets = {a['name']: a for a in release['assets']}
    if name not in assets or name + '.sha256' not in assets:
        raise ValueError(f'Release {tag} has no installer and checksum for {system}/{arch}.')
    return tag, assets[name], assets[name + '.sha256']


def download(release, directory):
    _, asset, checksum = release
    with fetch(checksum['url'], binary=True) as response:
        expected = response.read(4096).decode().split()[0].lower()
    if not re.fullmatch('[a-f0-9]{64}', expected):
        raise ValueError('Invalid release checksum.')
    target = directory / asset['name']
    digest = hashlib.sha256()
    with fetch(asset['url'], binary=True) as response, target.open('wb') as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            digest.update(chunk)
    if target.stat().st_size != asset['size'] or digest.hexdigest() != expected:
        target.unlink()
        raise ValueError('The installer download failed checksum verification. Please try again.')
    return target


class UpdateController:
    def __init__(self, app):
        self.app = app
        self.root = installation_root()
        self.events = queue.Queue()
        self.ready = None
        self.directory = None
        self.dialog = None
        if self.root is not None:
            self.run('release', latest_release)
            app.window.after(250, self.poll)

    def run(self, kind, action):
        def work():
            try:
                self.events.put((kind, action()))
            except Exception as error:
                self.events.put(('check_error' if kind == 'release' else 'error', error))
        threading.Thread(target=work, daemon=True).start()

    def busy(self):
        return bool(self.app.process or self.app.export_task or self.app.batch_locked)

    def poll(self):
        try:
            kind, value = self.events.get_nowait()
        except queue.Empty:
            kind, value = None, None
        if kind == 'check_error':
            explanation = ('No published release is accessible. Private repositories require '
                           'DECONFLOW_GITHUB_TOKEN with read-only Contents access.'
                           if isinstance(value, HTTPError) and value.code == 404 else str(value))
            self.app.log(f'Update check: {explanation}\n')
        elif kind == 'error':
            self.dismiss()
            if self.directory:
                shutil.rmtree(self.directory, ignore_errors=True)
            messagebox.showerror('Update failed', str(value), parent=self.app.window)
        elif kind == 'release' and value:
            self.ready = ('offer', value)
        elif kind == 'download':
            self.ready = ('install', value)
        if self.ready and not self.busy():
            stage, value = self.ready
            self.ready = None
            if stage == 'offer':
                if messagebox.askyesno('DeconFlow update',
                    f'DeconFlow {value[0].removeprefix("v")} is available (installed: {__version__}).\n\n'
                    'Download and install it now? DeconFlow will close and relaunch automatically.',
                    parent=self.app.window):
                    if not os.access(self.root.parent, os.W_OK) or not os.access(self.root, os.W_OK):
                        messagebox.showerror('Update unavailable',
                            'The installation is read-only. Move DeconFlow to a writable Applications folder '
                            'or ask your administrator to update it.', parent=self.app.window)
                    else:
                        self.directory = Path(tempfile.mkdtemp(prefix='deconflow-update-'))
                        self.show_progress()
                        self.run('download', lambda: download(value, self.directory))
            else:
                try:
                    self.app.save_settings()
                    helper = self.directory / 'update_installer.py'
                    shutil.copy2(Path(__file__).with_name('update_installer.py'), helper)
                    command = [sys.executable, str(helper), str(os.getpid()), str(value), str(self.root)]
                    options = ({'creationflags': subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP}
                               if sys.platform == 'win32' else {'start_new_session': True})
                    with (self.directory / 'update.log').open('w') as log:
                        subprocess.Popen(command, cwd=self.directory, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=log, **options)
                    self.dismiss()
                    self.app.close()
                    return
                except Exception as error:
                    self.dismiss()
                    messagebox.showerror('Update failed', str(error), parent=self.app.window)
        self.app.window.after(250, self.poll)

    def show_progress(self):
        import tkinter as tk
        from tkinter import ttk
        self.dialog = tk.Toplevel(self.app.window)
        self.dialog.title('Updating DeconFlow')
        self.dialog.transient(self.app.window)
        self.dialog.protocol('WM_DELETE_WINDOW', lambda: None)
        ttk.Label(self.dialog, text='Downloading and verifying the update…\nDeconFlow will restart when ready.',
                  padding=20).pack()
        bar = ttk.Progressbar(self.dialog, mode='indeterminate', length=300)
        bar.pack(padx=20, pady=(0, 20))
        bar.start()
        self.dialog.grab_set()

    def dismiss(self):
        if self.dialog is not None:
            self.dialog.grab_release()
            self.dialog.destroy()
            self.dialog = None
