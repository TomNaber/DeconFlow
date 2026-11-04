import multiprocessing
from pathlib import Path
import queue
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
        self.folder = None
        self.filename = tk.StringVar()
        self.process = None
        self.connection = None
        self.generation = 0
        self.events = queue.Queue()
        self.data = None
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
        self.views = ttk.Frame(self)
        self.views.pack(fill="both", expand=True)
        self.views.columnconfigure(0, weight=3)
        self.views.columnconfigure(1, weight=1)
        self.canvases = {}
        for row, title in enumerate(("Original", "Deconvolved")):
            base = row * 3
            ttk.Label(self.views, text=title).grid(row=base, column=0, columnspan=2, sticky="w", pady=(4, 0))
            self.views.rowconfigure(base + 2, weight=1, uniform="preview")
            for column, axis in enumerate(("xy", "yz")):
                ttk.Label(self.views, text=axis.upper()).grid(row=base + 1, column=column)
                canvas = tk.Canvas(self.views, width=1, height=1, highlightthickness=0, background="#242424")
                canvas.grid(row=base + 2, column=column, sticky="nsew", padx=2, pady=2)
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
        self.load_button.configure(state="normal" if names and not self.disabled else "disabled")

    def set_disabled(self, disabled):
        self.disabled = disabled
        self.files.configure(state="disabled" if disabled else "readonly")
        self.load_button.configure(state="normal" if self.selected_path and not disabled else "disabled")

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
        self.load_button.configure(state="disabled")
        threading.Thread(target=self.receive, args=(self.connection, self.generation), daemon=True).start()

    def receive(self, connection, generation):
        try:
            event = connection.recv()
        except (EOFError, OSError):
            event = ("error", "Preview worker stopped before loading completed.")
        self.events.put((generation, event))

    def poll(self):
        try:
            while True:
                generation, (kind, value) = self.events.get_nowait()
                if generation != self.generation:
                    continue
                self.set_disabled(self.disabled)
                if kind == "loaded":
                    self.data = value
                    z, y, x = self.data["shape"]
                    dz, dy, dx = self.data["spacing"]
                    self.views.columnconfigure(0, weight=max(1, round(1000 * x * dx / (x * dx + z * dz))))
                    self.views.columnconfigure(1, weight=max(1, round(1000 * z * dz / (x * dx + z * dz))))
                    self.recolor()
                else:
                    self.clear()
                    self.on_error(value)
        except queue.Empty:
            pass
        self.poll_job = self.after(50, self.poll)

    def recolor(self):
        if self.data is not None:
            self.images = {axis: composite(self.data["channels"], axis, self.preferences) for axis in ("xy", "yz")}
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
        for axis, image in self.images.items():
            canvas = self.canvases[0, axis]
            size = max(1, round(widths[axis] * scale)), max(1, round(height * scale))
            self.photos[axis] = ImageTk.PhotoImage(image.resize(size, Image.Resampling.BILINEAR), master=self)
            canvas.delete("all")
            canvas.create_image(canvas.winfo_width() / 2, canvas.winfo_height() / 2, image=self.photos[axis])

    def clear(self):
        self.generation += 1
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=2)
            if self.process.is_alive():
                self.process.kill()
                self.process.join()
            self.process.close()
            self.process = None
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        self.data = None
        self.images.clear()
        self.photos.clear()
        for canvas in self.canvases.values():
            canvas.delete("all")
        self.set_disabled(self.disabled)

    def close(self):
        self.after_cancel(self.poll_job)
        self.clear()
