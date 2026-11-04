import multiprocessing
import os
from pathlib import Path
import queue
import signal
from tempfile import TemporaryDirectory
import threading
import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import Image, ImageTk

from source.edit_ims_channels import NAMED_COLORS
from source.preview_worker import run


def composite(channels, axis, preferences):
    result = np.zeros((*channels[0][axis].shape, 3), dtype=np.float32)
    for channel in channels:
        selected = preferences.get(str(channel["index"]), {}).get("color", "Original")
        color = NAMED_COLORS.get(selected, channel["color"])
        low, high = channel["range"]
        intensity = np.clip((channel[axis].astype(np.float32) - low) / (high - low), 0, 1)
        result += intensity[..., None] * np.asarray(color, dtype=np.float32)
    return Image.fromarray(np.rint(np.clip(result, 0, 1) * 255).astype(np.uint8))


class Preview(ttk.LabelFrame):
    def __init__(self, parent, preferences, on_select, on_error):
        super().__init__(parent, text="Preview", padding=8)
        self.preferences = preferences
        self.on_select = on_select
        self.on_error = on_error
        self.get_settings = None
        self.on_activity = lambda: None
        self.on_log = lambda value: None
        self.folder = None
        self.filename = tk.StringVar()
        self.process = None
        self.connection = None
        self.receiver = None
        self.generation = 0
        self.events = queue.Queue()
        self.data = None
        self.deconvolved = None
        self.busy = None
        self.revision = 0
        self.pending = None
        self.running = None
        self.sync_job = None
        self.job_directory = None
        self.sync = tk.BooleanVar(value=False)
        self.images = {}
        self.photos = {}
        self.disabled = False
        toolbar = ttk.Frame(self)
        toolbar.pack(fill="x", pady=(0, 8))
        self.files = ttk.Combobox(toolbar, textvariable=self.filename, state="readonly", width=12)
        self.files.pack(side="left", fill="x", expand=True)
        self.files.bind("<<ComboboxSelected>>", lambda event: self.on_select())
        self.load_button = ttk.Button(toolbar, text="Load", command=self.load, state="disabled")
        self.load_button.pack(side="left", padx=(6, 0))
        actions = ttk.Frame(self)
        actions.pack(side="bottom", fill="x", pady=(8, 0))
        self.deconvolve_button = ttk.Button(actions, text="Deconvolve preview", command=self.deconvolve, state="disabled")
        self.deconvolve_button.pack(side="left")
        self.sync_button = ttk.Checkbutton(actions, text="Sync preview", variable=self.sync, command=self.sync_changed)
        self.sync_button.pack(side="left", padx=8)
        self.views = ttk.Frame(self)
        self.views.pack(fill="both", expand=True)
        self.views.columnconfigure(0, weight=3)
        self.views.columnconfigure(1, weight=1)
        self.canvases = {}
        for row, title in enumerate(("Original", "Deconvolved")):
            base = row * 2
            ttk.Label(self.views, text=title).grid(row=base, column=0, columnspan=2, sticky="w", pady=(4, 0))
            self.views.rowconfigure(base + 1, weight=1, uniform="preview")
            for column, axis in enumerate(("xy", "yz")):
                canvas = tk.Canvas(self.views, width=1, height=1, highlightthickness=0, background="#242424")
                canvas.grid(row=base + 1, column=column, sticky="nsew", padx=2, pady=2)
                canvas.bind("<Configure>", lambda event: self.draw())
                self.canvases[row, axis] = canvas
        self.poll_job = self.after(50, self.poll)

    @property
    def selected_path(self):
        return self.folder / self.filename.get() if self.folder and self.filename.get() else None

    def set_folder(self, folder):
        self.folder = Path(folder).expanduser()
        names = [path.name for path in sorted(self.folder.glob("*.ims")) if path.is_file()] if self.folder.is_dir() else []
        self.files.configure(values=names)
        if self.filename.get() not in names:
            self.filename.set(names[0] if names else "")
        self.update_controls()

    def set_disabled(self, disabled):
        self.disabled = disabled
        self.update_controls()

    def update_controls(self):
        disabled = self.disabled
        self.files.configure(state="disabled" if disabled else "readonly")
        self.load_button.configure(state="normal" if self.selected_path and not disabled and self.busy != "load" else "disabled")
        self.sync_button.configure(state="disabled" if disabled else "normal")
        ready = self.matches_loaded() and not disabled and self.busy is None and not self.sync.get()
        self.deconvolve_button.configure(state="normal" if ready else "disabled")
        self.on_activity()

    def matches_loaded(self):
        return (self.data is not None and self.selected_path is not None
                and self.selected_path.resolve() == Path(self.data["path"]).resolve())

    def parameters_changed(self, *args):
        self.revision += 1
        self.sync_changed()

    def sync_changed(self):
        self.pending = None
        if self.sync_job is not None:
            self.after_cancel(self.sync_job)
            self.sync_job = None
        self.update_controls()
        if self.sync.get() and self.matches_loaded() and not self.disabled:
            self.sync_job = self.after(500, lambda: self.deconvolve(automatic=True))

    def deconvolve(self, automatic=False):
        self.sync_job = None
        if self.disabled or not self.matches_loaded() or self.get_settings is None:
            return
        try:
            settings = self.get_settings()
        except (ValueError, OSError, tk.TclError) as error:
            if not automatic:
                self.on_error(str(error))
            return
        if self.sync_job is not None:
            self.after_cancel(self.sync_job)
            self.sync_job = None
        if not self.matches_loaded():
            return
        if automatic:
            processing = {key: value for key, value in settings.items() if key != "TEMP_DIR"}
            if self.deconvolved is not None:
                previous = {key: value for key, value in self.deconvolved["settings"].items() if key != "TEMP_DIR"}
                if processing == previous:
                    return
            if self.running == (self.revision, settings):
                return
        self.pending = (self.revision, settings)
        self.start_pending()

    def start_pending(self):
        if self.pending is None or self.busy is not None or self.disabled or not self.matches_loaded():
            return
        revision, settings = self.pending
        self.pending = None
        try:
            directory = Path(settings["TEMP_DIR"]).expanduser()
            directory.mkdir(parents=True, exist_ok=True)
            self.job_directory = TemporaryDirectory(prefix="preview-", dir=directory)
            self.connection.send((revision, settings, self.job_directory.name))
            self.running = (revision, settings)
            self.busy = "deconvolve"
            self.on_log("\nDeconvolving preview...\n")
        except (OSError, EOFError) as error:
            self.cleanup_job()
            self.on_error(str(error))
        self.update_controls()

    def cleanup_job(self):
        if self.job_directory is not None:
            self.job_directory.cleanup()
            self.job_directory = None

    def load(self):
        path = self.selected_path
        if path is None or self.disabled:
            return
        self.clear()
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=run, args=(child, str(path)), daemon=True)
        try:
            self.process.start()
        except Exception as error:
            self.process = None
            self.connection.close()
            self.connection = None
            self.on_error(str(error))
            return
        finally:
            child.close()
        self.busy = "load"
        self.update_controls()
        self.receiver = threading.Thread(target=self.receive, args=(self.connection, self.generation), daemon=True)
        self.receiver.start()

    def receive(self, connection, generation):
        try:
            while True:
                event = connection.recv()
                self.events.put((generation, event))
        except (EOFError, OSError):
            event = ("error", "Preview worker stopped unexpectedly.")
        self.events.put((generation, event))

    def poll(self):
        try:
            while True:
                generation, (kind, value) = self.events.get_nowait()
                if generation != self.generation:
                    continue
                if kind == "log":
                    self.on_log(value)
                elif kind == "loaded":
                    self.busy = None
                    self.data = value
                    z, y, x = self.data["shape"]
                    dz, dy, dx = self.data["spacing"]
                    self.views.columnconfigure(0, weight=max(1, round(1000 * x * dx / (x * dx + z * dz))))
                    self.views.columnconfigure(1, weight=max(1, round(1000 * z * dz / (x * dx + z * dz))))
                    self.recolor()
                    self.parameters_changed()
                elif kind in ("deconvolved", "failed"):
                    self.busy = None
                    self.running = None
                    self.cleanup_job()
                    revision, result = value
                    if revision == self.revision and self.matches_loaded():
                        if kind == "deconvolved":
                            self.deconvolved = result
                            self.recolor()
                            self.on_log("Preview finished.\n")
                        else:
                            self.on_error(result)
                    self.update_controls()
                    self.start_pending()
                else:
                    self.clear()
                    self.on_error(value)
        except queue.Empty:
            pass
        self.poll_job = self.after(50, self.poll)

    def recolor(self):
        if self.data is not None:
            self.images = {
                (row, axis): composite(data["channels"], axis, self.preferences)
                for row, data in enumerate((self.data, self.deconvolved)) if data is not None
                for axis in ("xy", "yz")
            }
            self.draw()

    def draw(self):
        if self.data is None:
            return
        z, y, x = self.data["shape"]
        dz, dy, dx = self.data["spacing"]
        widths = {"xy": x * dx, "yz": z * dz}
        height = y * dy
        scale = min(
            min((canvas.winfo_width() - 8) / widths[axis], (canvas.winfo_height() - 8) / height)
            for (row, axis), canvas in self.canvases.items()
        )
        if scale <= 0:
            return
        for (row, axis), image in self.images.items():
            canvas = self.canvases[row, axis]
            size = max(1, round(widths[axis] * scale)), max(1, round(height * scale))
            self.photos[row, axis] = ImageTk.PhotoImage(image.resize(size, Image.Resampling.BILINEAR), master=self)
            canvas.delete("all")
            canvas.create_image(canvas.winfo_width() / 2, canvas.winfo_height() / 2, image=self.photos[row, axis])

    def clear(self):
        self.generation += 1
        self.revision += 1
        self.pending = None
        self.running = None
        if self.sync_job is not None:
            self.after_cancel(self.sync_job)
            self.sync_job = None
        if self.process is not None:
            pid = self.process.pid
            if self.process.is_alive():
                try:
                    os.killpg(pid, signal.SIGTERM)
                except ProcessLookupError:
                    self.process.terminate()
            self.process.join(timeout=2)
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if self.process.is_alive():
                self.process.kill()
                self.process.join()
            self.process.close()
            self.process = None
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        if self.receiver is not None:
            self.receiver.join(timeout=2)
            self.receiver = None
        while not self.events.empty():
            try:
                self.events.get_nowait()
            except queue.Empty:
                break
        self.data = None
        self.deconvolved = None
        self.busy = None
        self.cleanup_job()
        self.images.clear()
        self.photos.clear()
        for canvas in self.canvases.values():
            canvas.delete("all")
        self.set_disabled(self.disabled)

    def close(self):
        self.after_cancel(self.poll_job)
        self.clear()
