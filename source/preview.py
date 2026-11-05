import multiprocessing
import json
from pathlib import Path
import queue
import sys
from time import perf_counter
from tempfile import TemporaryDirectory
import threading
import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np
from PIL import Image, ImageTk
from PIL.PngImagePlugin import PngInfo

from source.edit_ims_channels import NAMED_COLORS
from source.preview_worker import channel_settings, run
from source.preview_gestures import install_pinch
from source.fiji_session import FijiSession, atomic_json
from source.platform_support import stop_process_tree


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
        self.get_temp_directory = lambda: None
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
        self.channel_variables = {}
        self.busy = None
        self.revision = 0
        self.pending = None
        self.running = None
        self.sync_job = None
        self.job_directory = None
        self.cache_directory = None
        self.prepare_job = None
        self.fiji_session = None
        self.request_started = None
        self.initializing = False
        self.initialization_second = None
        self.sync = tk.BooleanVar(value=False)
        self.images = {}
        self.photos = {}
        self.zoom = tk.DoubleVar(value=100)
        self.zoom_label = tk.StringVar(value="100%")
        self.center = dict.fromkeys("xyz", 0.0)
        self.drag = None
        self.crop = None
        self.crop_drag = None
        self.draw_job = None
        self.disabled = False
        toolbar = ttk.Frame(self)
        toolbar.pack(fill="x", pady=(0, 8))
        self.files = ttk.Combobox(toolbar, textvariable=self.filename, state="readonly", width=12)
        self.files.pack(side="left", fill="x", expand=True)
        self.files.bind("<<ComboboxSelected>>", lambda event: self.on_select())
        self.load_button = ttk.Button(toolbar, text="Load", command=self.load, state="disabled")
        self.load_button.pack(side="left", padx=(6, 0))
        self.channel_bar = ttk.Frame(self)
        self.channel_bar.pack(fill="x", pady=(0, 4))
        actions = ttk.Frame(self)
        actions.pack(side="bottom", fill="x", pady=(8, 0))
        self.deconvolve_button = ttk.Button(actions, text="Deconvolve preview", command=self.deconvolve, state="disabled")
        self.deconvolve_button.pack(side="left")
        self.sync_button = ttk.Checkbutton(actions, text="Sync preview", variable=self.sync, command=self.sync_changed)
        self.sync_button.pack(side="left", padx=8)
        self.export_button = ttk.Button(actions, text="Export screenshot", command=self.export_screenshot, state="disabled")
        self.export_button.pack(side="right")
        zoom_bar = ttk.Frame(self)
        zoom_bar.pack(side="bottom", fill="x", pady=(8, 0))
        ttk.Label(zoom_bar, text="Zoom").pack(side="left")
        self.zoom_slider = ttk.Scale(zoom_bar, from_=10, to=1600, variable=self.zoom, command=self.slider_zoom)
        self.zoom_slider.pack(side="left", fill="x", expand=True, padx=8)
        self.reset_button = ttk.Button(zoom_bar, text="Reset", command=self.reset_zoom, state="disabled")
        self.reset_button.pack(side="right", padx=(8, 0))
        ttk.Label(zoom_bar, textvariable=self.zoom_label, width=6, anchor="e").pack(side="right")
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
                key = row, axis
                canvas.bind("<MouseWheel>", lambda event, key=key: self.wheel_zoom(event, key))
                canvas.bind("<Button-4>", lambda event, key=key: self.wheel_zoom(event, key))
                canvas.bind("<Button-5>", lambda event, key=key: self.wheel_zoom(event, key))
                canvas.bind("<ButtonPress-1>", lambda event, key=key: self.start_drag(event, key))
                canvas.bind("<B1-Motion>", self.drag_motion)
                canvas.bind("<ButtonRelease-1>", self.end_drag)
                canvas.bind("<Motion>", lambda event, key=key: self.crop_cursor(event, key))
                self.canvases[row, axis] = canvas
        self.remove_pinch = install_pinch(self, self.pinch)
        self.update_initialization()
        self.poll_job = self.after(50, self.poll)

    @property
    def selected_path(self):
        return self.folder / self.filename.get() if self.folder and self.filename.get() else None

    @property
    def selected_channels(self):
        return [index for index, variable in self.channel_variables.items() if variable.get()]

    def build_channels(self):
        for widget in self.channel_bar.winfo_children():
            widget.destroy()
        self.channel_variables.clear()
        if self.data is not None:
            for position, channel in enumerate(self.data["channels"]):
                index = channel["index"]
                variable = tk.BooleanVar(value=True)
                self.channel_variables[index] = variable
                ttk.Checkbutton(self.channel_bar, text=f"C{index + 1}: {channel['name']}",
                                variable=variable, command=self.channels_changed).grid(
                                    row=position // 3, column=position % 3, sticky="w", padx=(0, 8))

    def channels_changed(self):
        self.recolor()
        self.parameters_changed()

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
        self.zoom_slider.configure(state="normal" if self.data is not None and not disabled else "disabled")
        self.reset_button.configure(state="normal" if self.data is not None and not disabled else "disabled")
        for widget in self.channel_bar.winfo_children():
            widget.configure(state="disabled" if disabled else "normal")
        ready = (self.matches_loaded() and self.selected_channels and not disabled and self.busy is None
                 and self.crop_drag is None and not self.sync.get())
        self.deconvolve_button.configure(state="normal" if ready else "disabled")
        self.export_button.configure(state="normal" if self.images and not disabled else "disabled")
        self.on_activity()

    def matches_loaded(self):
        return (self.data is not None and self.selected_path is not None
                and self.selected_path.resolve() == Path(self.data["path"]).resolve())

    def configure_fiji(self, executable, memory):
        key = (Path(executable).expanduser().resolve(), str(memory))
        session = self.fiji_session
        if (session is not None and not session.closed and session.process.poll() is None
                and not (session.directory / "failed.json").exists()
                and (session.executable, session.memory) == key):
            return session
        self.stop_fiji()
        self.fiji_session = FijiSession(*key, log=lambda line: self.events.put(
            (None, ("session_log", line))))
        return self.fiji_session

    def stop_fiji(self):
        if self.fiji_session is not None:
            self.fiji_session.close()
            self.fiji_session = None

    def prepare_inputs(self):
        self.prepare_job = None
        if (not self.matches_loaded() or self.disabled or self.busy is not None or self.crop_drag is not None
                or not self.selected_channels or self.get_settings is None):
            return
        try:
            settings = self.get_settings()
            settings["CROP_BOUNDS"] = {a: list(b) for a, b in self.crop.items()}
            self.connection.send(("prepare", self.revision, settings))
        except (ValueError, OSError, EOFError, tk.TclError):
            pass  # Missing PSF/settings are reported on an explicit preview request.

    def parameters_changed(self, *args):
        self.revision += 1
        self.sync_changed()

    def sync_changed(self):
        if self.cache_directory is not None:
            atomic_json(Path(self.cache_directory.name, "revision"), str(self.revision))
        if self.prepare_job is not None:
            self.after_cancel(self.prepare_job)
        self.prepare_job = self.after(250, self.prepare_inputs)
        self.pending = None
        if self.sync_job is not None:
            self.after_cancel(self.sync_job)
            self.sync_job = None
        self.update_controls()
        if self.sync.get() and self.matches_loaded() and self.selected_channels and not self.disabled and self.crop_drag is None:
            self.sync_job = self.after(500, lambda: self.deconvolve(automatic=True))

    def deconvolve(self, automatic=False):
        self.sync_job = None
        if self.disabled or self.crop_drag is not None or not self.matches_loaded() or not self.selected_channels or self.get_settings is None:
            return
        try:
            settings = self.get_settings()
            settings["CROP_BOUNDS"] = {axis: list(bounds) for axis, bounds in self.crop.items()}
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
            processing = {index: channel_settings(settings, index) for index in self.selected_channels}
            if self.deconvolved is not None:
                previous = {channel["index"]: channel["processing"] for channel in self.deconvolved["channels"]}
                if all(previous.get(index) == value for index, value in processing.items()):
                    return
            if self.running is not None and self.running[0] == self.revision:
                running = self.running[1]
                if all(index in running["CHANNEL_INDICES"] and channel_settings(running, index) == value
                       for index, value in processing.items()):
                    return
        settings["FORCE"] = not automatic
        self.pending = (self.revision, settings)
        self.start_pending()

    def start_pending(self):
        if self.pending is None or self.busy is not None or self.disabled or not self.matches_loaded():
            return
        revision, settings = self.pending
        self.pending = None
        self.request_started = perf_counter()
        try:
            directory = Path(settings["TEMP_DIR"]).expanduser()
            directory.mkdir(parents=True, exist_ok=True)
            self.job_directory = TemporaryDirectory(prefix="preview-", dir=directory)
            session = self.configure_fiji(settings["FIJI_EXECUTABLE"], settings["FIJI_MEMORY"])
            settings["FIJI_SESSION"] = str(session.directory)
            self.connection.send((revision, settings, self.job_directory.name))
            self.running = (revision, settings)
            self.busy = "deconvolve"
            self.initializing = True
            self.initialization_second = None
            self.on_log("\n")
            self.update_initialization()
        except (OSError, EOFError) as error:
            self.cleanup_job()
            self.on_error(str(error))
        self.update_controls()

    def update_initialization(self):
        if self.initializing:
            elapsed = int(perf_counter() - self.request_started)
            if elapsed != self.initialization_second:
                self.initialization_second = elapsed
                self.on_log(f"\rInitializing deconvolution preview. {elapsed}s elapsed.")

    def finish_initialization(self, success):
        if self.initializing:
            self.initializing = False
            elapsed = perf_counter() - self.request_started
            message = (f"Deconvolution initialized in {elapsed:.1f}s." if success else
                       f"Deconvolution initialization stopped after {elapsed:.1f}s.")
            self.on_log("\r" + message + "\n")

    def cleanup_job(self):
        self.finish_initialization(False)
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
        try:
            directory = self.get_temp_directory()
            if directory:
                directory = Path(directory).expanduser()
                directory.mkdir(parents=True, exist_ok=True)
            self.cache_directory = TemporaryDirectory(prefix="preview-cache-", dir=directory)
            atomic_json(Path(self.cache_directory.name, "revision"), str(self.revision))
            self.process = context.Process(target=run, args=(child, str(path), self.cache_directory.name), daemon=True)
            self.process.start()
        except Exception as error:
            self.process = None
            self.connection.close()
            self.connection = None
            if self.cache_directory is not None:
                self.cache_directory.cleanup()
                self.cache_directory = None
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
                if kind == "session_log":
                    if not (self.initializing and value.startswith("Fiji ready")):
                        self.on_log(value)
                    continue
                if generation != self.generation:
                    continue
                if kind == "initialized":
                    if value == self.revision:
                        self.finish_initialization(True)
                    continue
                if kind == "log":
                    self.on_log(value)
                elif kind == "loaded":
                    self.busy = None
                    self.data = value
                    self.crop = {axis: [0, size] for axis, size in zip("zyx", value["shape"])}
                    self.reset_view()
                    self.build_channels()
                    z, y, x = self.data["shape"]
                    dz, dy, dx = self.data["spacing"]
                    self.views.columnconfigure(0, weight=max(1, round(1000 * x * dx / (x * dx + z * dz))))
                    self.views.columnconfigure(1, weight=max(1, round(1000 * z * dz / (x * dx + z * dz))))
                    self.recolor()
                    self.parameters_changed()
                elif kind in ("channel", "deconvolved", "failed"):
                    revision, result = value
                    if kind != "channel":
                        self.busy = None
                        self.running = None
                        self.cleanup_job()
                    if revision == self.revision and self.matches_loaded():
                        if kind in ("channel", "deconvolved"):
                            # Each message contains only compatible channels from this request.
                            self.deconvolved = result
                            self.recolor()
                            if kind == "deconvolved":
                                self.on_log(f"Preview finished in {perf_counter() - self.request_started:.0f}s.\n")
                        else:
                            self.stop_fiji()
                            self.on_error(result)
                    self.update_controls()
                    if kind != "channel":
                        self.start_pending()
                        if self.busy is None and revision != self.revision:
                            self.prepare_inputs()
                else:
                    self.clear()
                    self.on_error(value)
        except queue.Empty:
            pass
        self.update_initialization()
        self.poll_job = self.after(50, self.poll)

    def recolor(self):
        if self.data is not None:
            self.images.clear()
            self.photos.clear()
            for canvas in self.canvases.values():
                canvas.delete("all")
            for row, data in enumerate((self.data, self.deconvolved)):
                if data is None:
                    continue
                channels = [channel for channel in data["channels"] if channel["index"] in self.selected_channels]
                if channels:
                    for axis in ("xy", "yz"):
                        self.images[row, axis] = composite(channels, axis, self.preferences)
            self.draw()

    def reset_zoom(self):
        self.end_pan()
        self.reset_view()
        self.draw()

    def reset_view(self):
        self.zoom.set(100)
        self.zoom_label.set("100%")
        self.drag = None
        self.center = dict.fromkeys("xyz", 0.0)
        if self.data is not None:
            self.center = {axis: size * spacing / 2 for axis, size, spacing
                           in zip("zyx", self.data["shape"], self.data["spacing"])}

    def view_scale(self):
        if self.data is None:
            return 0
        z, y, x = self.data["shape"]
        dz, dy, dx = self.data["spacing"]
        widths = {"xy": x * dx, "yz": z * dz}
        height = y * dy
        fit = min(
            min((canvas.winfo_width() - 8) / widths[axis], (canvas.winfo_height() - 8) / height)
            for (row, axis), canvas in self.canvases.items()
        )
        return max(0, fit) * self.zoom.get() / 100

    def slider_zoom(self, value):
        self.zoom_label.set(f"{self.zoom.get():.0f}%")
        self.schedule_draw()

    def set_zoom(self, value, key=None, x=None, y=None):
        if self.data is None or self.disabled:
            return
        old_scale = self.view_scale()
        self.zoom.set(min(1600, max(10, value)))
        new_scale = self.view_scale()
        if key is not None and old_scale > 0 and new_scale > 0:
            canvas = self.canvases[key]
            horizontal = "x" if key[1] == "xy" else "z"
            self.center[horizontal] += (x - canvas.winfo_width() / 2) * (1 / old_scale - 1 / new_scale)
            self.center["y"] += (y - canvas.winfo_height() / 2) * (1 / old_scale - 1 / new_scale)
        self.zoom_label.set(f"{self.zoom.get():.0f}%")
        self.schedule_draw()

    def wheel_zoom(self, event, key):
        if event.num in (4, 5):
            steps = 1 if event.num == 4 else -1
        else:
            steps = event.delta if sys.platform == "darwin" else event.delta / 120
        self.set_zoom(self.zoom.get() * 1.1 ** max(-10, min(10, steps)), key, event.x, event.y)
        return "break"

    def pinch(self, target, x, y, amount):
        if self.data is None or self.disabled:
            return False
        for key, canvas in self.canvases.items():
            if target is canvas:
                self.set_zoom(self.zoom.get() * max(0.01, 1 + amount), key,
                              x - canvas.winfo_rootx(), y - canvas.winfo_rooty())
                return True
        return False

    def start_pan(self, event, key):
        if self.data is not None and not self.disabled:
            self.drag = key, event.x, event.y, dict(self.center)
            self.canvases[key].configure(cursor="fleur")

    def crop_rectangle(self, key):
        scale = self.view_scale()
        canvas = self.canvases[key]
        horizontal = "x" if key[1] == "xy" else "z"
        spacing = dict(zip("zyx", self.data["spacing"]))
        return [canvas.winfo_width() / 2 + (self.crop[horizontal][0] * spacing[horizontal] - self.center[horizontal]) * scale,
                canvas.winfo_height() / 2 + (self.crop["y"][0] * spacing["y"] - self.center["y"]) * scale,
                canvas.winfo_width() / 2 + (self.crop[horizontal][1] * spacing[horizontal] - self.center[horizontal]) * scale,
                canvas.winfo_height() / 2 + (self.crop["y"][1] * spacing["y"] - self.center["y"]) * scale]

    def crop_edges(self, key, x, y):
        if key[0] != 0 or self.crop is None or self.disabled or self.view_scale() <= 0:
            return []
        left, top, right, bottom = self.crop_rectangle(key)
        edges = []
        if top - 7 <= y <= bottom + 7:
            side = 0 if abs(x - left) <= abs(x - right) else 1
            if abs(x - (left, right)[side]) <= 7:
                edges.append(("x" if key[1] == "xy" else "z", side))
        if left - 7 <= x <= right + 7:
            side = 0 if abs(y - top) <= abs(y - bottom) else 1
            if abs(y - (top, bottom)[side]) <= 7:
                edges.append(("y", side))
        return edges

    def crop_cursor(self, event, key):
        if self.drag is not None or self.crop_drag is not None:
            return
        edges = self.crop_edges(key, event.x, event.y)
        cursor = "crosshair" if len(edges) == 2 else "sb_v_double_arrow" if edges and edges[0][0] == "y" else "sb_h_double_arrow" if edges else ""
        self.canvases[key].configure(cursor=cursor)

    def start_drag(self, event, key):
        edges = self.crop_edges(key, event.x, event.y)
        if not edges:
            self.start_pan(event, key)
            return
        self.crop_drag = {"key": key, "edges": edges, "x": event.x, "y": event.y,
                          "bounds": {axis: list(bounds) for axis, bounds in self.crop.items()},
                          "scale": self.view_scale(), "changed": False}
        self.update_controls()

    def drag_motion(self, event):
        if self.crop_drag is None:
            self.pan(event)
            return
        drag = self.crop_drag
        spacing = dict(zip("zyx", self.data["spacing"]))
        sizes = dict(zip("zyx", self.data["shape"]))
        bounds = {axis: list(value) for axis, value in drag["bounds"].items()}
        for axis, side in drag["edges"]:
            delta = event.y - drag["y"] if axis == "y" else event.x - drag["x"]
            value = round(drag["bounds"][axis][side] + delta / (drag["scale"] * spacing[axis]))
            bounds[axis][side] = max(0, min(bounds[axis][1] - 1, value)) if side == 0 else min(sizes[axis], max(bounds[axis][0] + 1, value))
        if bounds != self.crop:
            self.crop = bounds
            if not drag["changed"]:
                drag["changed"] = True
                self.revision += 1
                self.sync_changed()
                self.deconvolved = None
                if self.connection is not None:
                    try:
                        self.connection.send("clear_result")
                    except (OSError, EOFError):
                        pass
                self.recolor()
            self.schedule_draw()

    def end_drag(self, event=None):
        if self.crop_drag is not None:
            key = self.crop_drag["key"]
            self.crop_drag = None
            self.canvases[key].configure(cursor="")
            self.sync_changed()
        self.end_pan()

    def pan(self, event):
        scale = self.view_scale()
        if self.drag is not None and scale > 0:
            key, x, y, center = self.drag
            horizontal = "x" if key[1] == "xy" else "z"
            self.center[horizontal] = center[horizontal] - (event.x - x) / scale
            self.center["y"] = center["y"] - (event.y - y) / scale
            self.schedule_draw()

    def end_pan(self, event=None):
        if self.drag is not None:
            self.canvases[self.drag[0]].configure(cursor="")
            self.drag = None

    def schedule_draw(self):
        if self.draw_job is None:
            self.draw_job = self.after_idle(self.draw)

    def render_views(self):
        scale = self.view_scale()
        if scale <= 0:
            return {}
        dz, dy, dx = self.data["spacing"]
        rendered = {}
        for (row, axis), image in self.images.items():
            canvas = self.canvases[row, axis]
            width, height = canvas.winfo_width(), canvas.winfo_height()
            horizontal = "x" if axis == "xy" else "z"
            spacing = dx if axis == "xy" else dz
            crop = self.deconvolved.get("crop") if row and self.deconvolved else None
            x_offset = crop[horizontal][0] if crop else 0
            y_offset = crop["y"][0] if crop else 0
            transform = (1 / (scale * spacing), 0, (self.center[horizontal] - width / (2 * scale)) / spacing - x_offset,
                         0, 1 / (scale * dy), (self.center["y"] - height / (2 * scale)) / dy - y_offset)
            rendered[row, axis] = image.transform((width, height), Image.Transform.AFFINE, transform,
                                                  Image.Resampling.BILINEAR, fillcolor=canvas.cget("background"))
        return rendered

    def draw(self):
        if self.draw_job is not None:
            self.after_cancel(self.draw_job)
            self.draw_job = None
        for key, image in self.render_views().items():
            canvas = self.canvases[key]
            self.photos[key] = ImageTk.PhotoImage(image, master=self)
            canvas.delete("all")
            canvas.create_image(0, 0, anchor="nw", image=self.photos[key])
        if self.crop is not None and self.view_scale() > 0:
            for axis in ("xy", "yz"):
                key = 0, axis
                canvas = self.canvases[key]
                canvas.delete("crop")
                canvas.create_rectangle(*self.crop_rectangle(key), outline="#ffff00", width=2, tags="crop")

    def screenshot(self):
        rendered = self.render_views()
        row = 1 if (1, "xy") in rendered else 0
        if (row, "xy") not in rendered:
            return None
        canvases = [self.canvases[row, axis] for axis in ("xy", "yz")]
        left = min(canvas.winfo_x() for canvas in canvases)
        top = min(canvas.winfo_y() for canvas in canvases)
        right = max(canvas.winfo_x() + canvas.winfo_width() for canvas in canvases)
        bottom = max(canvas.winfo_y() + canvas.winfo_height() for canvas in canvases)
        background = ttk.Style(self).lookup("TFrame", "background") or "#242424"
        background = tuple(value // 257 for value in self.winfo_rgb(background))
        image = Image.new("RGB", (right - left, bottom - top), background)
        for axis, canvas in zip(("xy", "yz"), canvases):
            image.paste(rendered[row, axis], (canvas.winfo_x() - left, canvas.winfo_y() - top))
        data = self.deconvolved if row else self.data
        channels = [channel for channel in data["channels"] if channel["index"] in self.selected_channels]
        stem = Path(data["path"]).stem
        if row:
            tags = [f"iter{channel['processing']['ITERATIONS']}_reg{format(channel['processing']['RLTV_REGULARIZATION'], '.10g')}"
                    for channel in channels]
            tag = tags[0] if len(set(tags)) == 1 else "_".join(
                f"C{channel['index'] + 1}_{tag}" for channel, tag in zip(channels, tags))
            stem += f"_deconvolved_{tag}"
        metadata = {
            "source": data["path"],
            "view": "Deconvolved" if row else "Original",
            "display_range": [0, 255],
            "zoom_percent": self.zoom.get(),
            "center_xyz": dict(self.center),
            "crop_bounds": data.get("crop"),
            "channels": [{"index": channel["index"], "name": channel["name"],
                          "color": self.preferences.get(str(channel["index"]), {}).get("color", "Original"),
                          "processing": channel.get("processing")} for channel in channels],
        }
        return image, f"{stem}_max_projection.png", metadata

    def export_screenshot(self):
        try:
            snapshot = self.screenshot()
            if snapshot is None:
                return
            image, filename, metadata = snapshot
            path = filedialog.asksaveasfilename(
                parent=self.winfo_toplevel(), title="Export screenshot", initialfile=filename,
                initialdir=str(Path(self.data["path"]).parent), defaultextension=".png",
                filetypes=(("PNG image", "*.png"),))
            if not path:
                return
            info = PngInfo()
            info.add_text("Preview", json.dumps(metadata))
            image.save(path, format="PNG", pnginfo=info)
            self.on_log(f"Saved screenshot: {path}\n")
        except (OSError, ValueError, tk.TclError) as error:
            self.on_error(str(error))

    def clear(self):
        if self.busy == "deconvolve":
            self.stop_fiji()
        if self.prepare_job is not None:
            self.after_cancel(self.prepare_job)
            self.prepare_job = None
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
                stop_process_tree(pid)
            self.process.join(timeout=2)
            if self.process.is_alive():
                stop_process_tree(pid, force=True)
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
        if self.cache_directory is not None:
            self.cache_directory.cleanup()
            self.cache_directory = None
        self.data = None
        self.deconvolved = None
        self.crop = None
        self.crop_drag = None
        self.end_pan()
        self.reset_view()
        self.build_channels()
        self.busy = None
        self.cleanup_job()
        self.images.clear()
        self.photos.clear()
        for canvas in self.canvases.values():
            canvas.delete("all")
        self.set_disabled(self.disabled)

    def close(self):
        self.stop_fiji()
        self.remove_pinch()
        if self.draw_job is not None:
            self.after_cancel(self.draw_job)
            self.draw_job = None
        self.after_cancel(self.poll_job)
        self.clear()
