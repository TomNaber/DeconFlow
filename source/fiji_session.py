"""Owned headless Fiji process and a small, file-based request protocol."""
import atexit
from collections import deque
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import threading
from time import monotonic, sleep
import uuid

if __package__:
    from .platform_support import exclusive_file_lock, process_session_options, stop_process_tree
else:
    from platform_support import exclusive_file_lock, process_session_options, stop_process_tree

SOURCE = Path(__file__).resolve().parent
STARTUP_TIMEOUT = 120
JOB_TIMEOUT = 24 * 3600  # Full-volume batch jobs can run for hours.


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value), encoding='utf-8')
    temporary.replace(path)


def console_log(value):
    print(value, end="", flush=True)


class FijiSession:
    """The owner manages lifetime; clients receive only the directory path."""
    def __init__(self, executable, memory, log=console_log):
        self.executable = Path(executable).expanduser().resolve()
        self.memory = str(memory)
        self.log = log
        self.process = None
        self.closed = False
        self.diagnostics = deque(maxlen=100)
        self._lock = threading.Lock()
        if not self.executable.is_file() or not os.access(self.executable, os.X_OK):
            raise FileNotFoundError(f'No Fiji detected at path: {self.executable}')
        self._temporary = TemporaryDirectory(prefix='fiji-session-')
        self.directory = Path(self._temporary.name)
        atomic_json(self.directory / 'owner.json', {'pid': os.getpid()})
        (self.directory / 'lib').mkdir()
        shutil.copyfile(SOURCE / 'lib/totalvariationterm.cl', self.directory / 'lib/totalvariationterm.cl')
        self.log('Initializing Fiji.\n')
        environment = os.environ.copy()
        for name in ('JAVA_HOME', 'JDK_JAVA_OPTIONS', '_JAVA_OPTIONS', 'CLASSPATH'):
            environment.pop(name, None)
        parameter = 'session_dir=' + json.dumps(str(self.directory))
        try:
            self.process = subprocess.Popen(
                [str(self.executable), '--no-splash', '--headless', f'--mem={self.memory}',
                 '-Dimagej.updater.disableAutocheck=true', '--', '--headless', '--run',
                 str(SOURCE / 'fiji_session_worker.py'), parameter],
                cwd=self.directory, env=environment, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
                **process_session_options(),
            )
        except BaseException:
            self._temporary.cleanup()
            raise
        atexit.register(self.close)
        threading.Thread(target=self._read_log, daemon=True).start()
        threading.Thread(target=self._watch_startup, daemon=True).start()

    def _read_log(self):
        for line in self.process.stdout:
            if not self.closed and line.strip():
                self.diagnostics.append(line.rstrip())
        self.process.stdout.close()
        status = self.process.wait()
        with self._lock:
            if not self.closed:
                atomic_json(self.directory / 'failed.json', {'error': f'Fiji exited with status {status}\n' + '\n'.join(self.diagnostics)})

    def _watch_startup(self):
        started = monotonic()
        try:
            ready = FijiClient(self.directory).wait_ready()
            if not self.closed:
                self.log(f"Fiji ready. {ready['gpu']} ({monotonic() - started:.2f} s)\n")
        except Exception as error:
            if not self.closed:
                self.log(f'Fiji initialization failed: {error}\n')
                self.close()

    def close(self):
        with self._lock:
            if self.closed:
                return
            self.closed = True
            # A native GPU call cannot be safely interrupted in the command loop.
            # Kill only our separately owned process group, then reap off the UI thread.
            if self.process is not None and self.process.poll() is None:
                stop_process_tree(self.process.pid, force=True)
                self.process.wait()
            self._temporary.cleanup()
        atexit.unregister(self.close)


class FijiClient:
    def __init__(self, directory):
        self.directory = Path(directory)

    def _check(self):
        if not self.directory.is_dir():
            raise RuntimeError('Fiji session closed; initialize Fiji again.')
        failed = self.directory / 'failed.json'
        if failed.is_file():
            raise RuntimeError(json.loads(failed.read_text())['error'])

    def wait_ready(self):
        deadline = monotonic() + STARTUP_TIMEOUT
        while True:
            self._check()
            ready = self.directory / 'ready.json'
            if ready.is_file():
                return json.loads(ready.read_text())
            if monotonic() >= deadline:
                raise TimeoutError('Fiji initialization timed out.')
            sleep(0.025)

    @contextmanager
    def _exclusive(self):
        self._check()
        with exclusive_file_lock(self.directory / 'client.lock', JOB_TIMEOUT, self._check):
            yield

    def inspect_geometries(self, geometries, non_circulant):
        """Ask the owning JVM for padding and cache state before any RL calls."""
        with self._exclusive():
            self.wait_ready()
            request_id = uuid.uuid4().hex
            response = self.directory / ('response-' + request_id + '.json')
            atomic_json(self.directory / 'request.json', {
                'id': request_id, 'operation': 'inspect',
                'geometries': geometries, 'non_circulant': non_circulant,
            })
            deadline = monotonic() + JOB_TIMEOUT
            while not response.is_file():
                self._check()
                if monotonic() >= deadline:
                    atomic_json(self.directory / 'failed.json', {'error': 'Fiji job timed out; restart Fiji.'})
                    raise TimeoutError('Fiji job timed out; restart Fiji.')
                sleep(0.025)
            result = json.loads(response.read_text())
            response.unlink()
            if result['id'] != request_id:
                raise RuntimeError('Unexpected Fiji response ID.')
            if 'error' in result:
                raise RuntimeError(result['error'])
            return result['geometries']

    def run(self, input_path, psf_path, output_path, iterations, regularization, non_circulant, progress=None, geometry=None):
        started = monotonic()
        with self._exclusive():
            self.wait_ready()
            queue_ready = monotonic() - started
            request_id = uuid.uuid4().hex
            response = self.directory / ('response-' + request_id + '.json')
            geometry_path = self.directory / ('geometry-' + request_id + '.json')
            geometry_reported = False

            def report_geometry():
                nonlocal geometry_reported
                if not geometry_reported and geometry_path.is_file():
                    if geometry is not None:
                        geometry(json.loads(geometry_path.read_text()))
                    geometry_reported = True

            atomic_json(self.directory / 'request.json', {
                'id': request_id, 'input_path': str(Path(input_path).resolve()),
                'psf_path': str(Path(psf_path).resolve()), 'output_path': str(Path(output_path).resolve()),
                'iterations': iterations, 'regularization': regularization,
                'non_circulant': non_circulant,
            })
            deadline = monotonic() + JOB_TIMEOUT
            while not response.is_file():
                self._check()
                report_geometry()
                if progress is not None:
                    progress()
                if monotonic() >= deadline:
                    # Poison the session so another client cannot submit overlapping work.
                    atomic_json(self.directory / 'failed.json', {'error': 'Fiji job timed out; restart Fiji.'})
                    raise TimeoutError('Fiji job timed out; restart Fiji.')
                sleep(0.025)
            report_geometry()
            geometry_path.unlink(missing_ok=True)
            result = json.loads(response.read_text())
            response.unlink()
            if result['id'] != request_id:
                raise RuntimeError('Unexpected Fiji response ID.')
            if 'error' in result:
                raise RuntimeError(result['error'])
            if not Path(output_path).is_file():
                raise RuntimeError('Fiji did not create an output TIFF.')
            result['timings']['queue_ready'] = queue_ready
            return result['timings']
