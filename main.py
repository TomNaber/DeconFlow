import configparser
import importlib
import json
import math
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading


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
    from tkinter import filedialog, messagebox, ttk
    from tkinter.scrolledtext import ScrolledText

    window = tk.Tk()
    window.title("Richardson-Lucy deconvolution and Gibson Lanni point-spread functions")
    window.geometry("1100x950")
    window.minsize(1050, 900)
    mode = tk.StringVar(value="Deconvolution")
    separate = tk.BooleanVar(value=False)
    fields = {}
    process = None
    cancelled = False
    events = queue.Queue()

    toolbar = ttk.Frame(window, padding=10)
    toolbar.pack(fill="x")
    panels = {}
    mode_labels = {
        "Deconvolution": "Richardson-Lucy deconvolution",
        "PSF generation": "Gibson Lanni point-spread function generation",
    }
    for title in mode_labels:
        ttk.Radiobutton(toolbar, text=mode_labels[title], variable=mode, value=title,
                        command=lambda: show_mode()).pack(side="left", padx=5)
        panels[title] = ttk.Frame(window, padding=10)
        fields[title] = {}

    def add_field(panel, group, row, name, label, default, kind=str, browse=None):
        variable = tk.BooleanVar(value=default) if kind is bool else tk.StringVar(value=str(default))
        fields[group][name] = (variable, kind, label)
        caption = ttk.Label(panel, text=label, wraplength=260)
        caption.grid(row=row, column=0, sticky="w", padx=(0, 8), pady=3)
        holder = ttk.Frame(panel)
        holder.grid(row=row, column=1, sticky="ew", pady=3)
        panel.columnconfigure(1, weight=1)
        if kind is bool:
            ttk.Checkbutton(holder, variable=variable).pack(side="left")
        elif name == "SCAN_DIRECTION":
            ttk.Combobox(holder, textvariable=variable, values=("up", "down"),
                         state="readonly", width=12).pack(fill="x")
        else:
            ttk.Entry(holder, textvariable=variable, width=18).pack(side="left", fill="x", expand=True)
        if browse:
            def choose():
                value = filedialog.askdirectory(parent=window) if browse == "directory" else filedialog.askopenfilename(parent=window)
                if value:
                    variable.set(value)
            ttk.Button(holder, text="Browse", command=choose).pack(side="left", padx=(5, 0))
        return caption, holder

    deconvolution = [
        ("INPUT_DIR", "Input folder (Imaris .ims files)", ROOT / "examples", str, "directory"),
        ("PSF_DIR", "Point-spread function folder", ROOT / "psfs", str, "directory"),
        ("OUTPUT_DIR", "Output folder", ROOT / "dcv", str, "directory"),
        ("TEMP_DIR", "Temporary working folder", ROOT / ".deconvolution_temp", str, "directory"),
        ("FIJI_EXECUTABLE", "Fiji executable", "/Applications/Fiji/Fiji.app/Contents/MacOS/fiji-macos-arm64", str, "file"),
        ("FIJI_MEMORY", "Fiji memory limit (e.g. 12g = 12 GB)", "12g", str, None),
        ("ITERATIONS", "Richardson-Lucy iterations", 40, int, None),
        ("RLTV_REGULARIZATION", "Total-variation regularization", 0.0001, float, None),
        ("RLTV_NON_CIRCULANT", "Non-circulant boundary correction", True, bool, None),
        ("OVERWRITE", "Overwrite existing output", False, bool, None),
    ]
    for row, setting in enumerate(deconvolution):
        add_field(panels["Deconvolution"], "Deconvolution", row, *setting)

    psf_panel = panels["PSF generation"]
    ttk.Checkbutton(psf_panel, text="Separate objective design and actual imaging conditions", variable=separate,
                    command=lambda: show_actual()).pack(anchor="w", pady=(0, 5))
    columns = ttk.Frame(psf_panel)
    columns.pack(fill="x")
    left = ttk.Frame(columns)
    right = ttk.Frame(columns)
    left.pack(side="left", fill="both", expand=True, padx=(0, 20))
    right.pack(side="left", fill="both", expand=True)
    psf_fields = [
        ("XY_PIXEL_SIZE_UM", "Lateral pixel size X/Y (µm)", 0.0801669, float),
        ("Z_STEP_UM", "Axial sampling interval Z (µm)", 0.3002, float),
        ("NUMERICAL_APERTURE", "Objective numerical aperture", 1.4, float),
        ("SPECIMEN_RI", "Specimen refractive index", 1.45, float),
        ("OBJECTIVE_WORKING_DISTANCE_UM", "Objective working distance (µm)", 140.0, float),
        ("IMAGING_DEPTH_UM", "Imaging depth (µm)", 400.0, float),
        ("PINHOLE_AIRY_UNITS", "Confocal pinhole (Airy units)", 1.0, float),
        ("PSF_SIZE_Z", "Point-spread function Z size (planes)", 209, int),
        ("PSF_SIZE_XY", "Point-spread function X/Y size (pixels)", 119, int),
        ("OVERSAMPLE_FACTOR", "Calculation oversampling factor", 3, int),
        ("SCAN_DIRECTION", "Axial scan direction (up/down)", "up", str),
    ]
    for row, setting in enumerate(psf_fields):
        add_field(left, "PSF generation", row, *setting)

    pairs = [
        ("IMMERSION_RI_DESIGN", "IMMERSION_RI_ACTUAL", "Immersion refractive index", 1.4811),
        ("COVERSLIP_RI_DESIGN", "COVERSLIP_RI_ACTUAL", "Coverslip refractive index", 1.5255),
        ("COVERSLIP_THICKNESS_DESIGN_UM", "COVERSLIP_THICKNESS_ACTUAL_UM", "Coverslip thickness (µm)", 145.0),
    ]
    pair_widgets = []
    for index, (design, actual, label, default) in enumerate(pairs):
        design_widgets = add_field(right, "PSF generation", index * 2, design, label, default, float)
        actual_widgets = add_field(right, "PSF generation", index * 2 + 1, actual, label + " (actual)", default, float)
        pair_widgets.append((design_widgets[0], actual_widgets, label))
    add_field(right, "PSF generation", 6, "OVERWRITE", "Overwrite existing point-spread functions", True, bool)
    output = ttk.Frame(psf_panel)
    output.pack(fill="x", pady=5)
    add_field(output, "PSF generation", 0, "OUTPUT_DIR", "Output folder", ROOT / "psfs/1000", str, "directory")
    ttk.Label(psf_panel, text="Channels in image channel order: one per line, name, excitation wavelength (nm), emission wavelength (nm)").pack(anchor="w")
    channels = ScrolledText(psf_panel, height=4, wrap="none")
    channels.pack(fill="x", pady=5)
    channels.insert("1.0", "Alexa488, 488, 525\nAlexa568, 552, 603\nAlexa647, 638, 671")

    def show_actual():
        for caption, widgets, label in pair_widgets:
            caption.configure(text=label + (" (design)" if separate.get() else ""))
            for widget in widgets:
                widget.grid() if separate.get() else widget.grid_remove()

    def show_mode():
        for panel in panels.values():
            panel.pack_forget()
        panels[mode.get()].pack(fill="x", after=toolbar)

    if sys.platform == "darwin":
        data_dir = Path.home() / "Library" / "Application Support"
    elif sys.platform == "win32":
        data_dir = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        data_dir = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    settings_path = data_dir / "Batch Deconvolution" / "settings.ini"
    defaults = {
        group: {name: variable.get() for name, (variable, _, _) in entries.items()}
        for group, entries in fields.items()
    }
    default_channels = channels.get("1.0", "end-1c")
    saving_paused = False

    def save_settings(*args):
        if saving_paused:
            return
        settings = configparser.ConfigParser(interpolation=None)
        settings["GUI"] = {"mode": mode.get()}
        for group, entries in fields.items():
            settings[group] = {name: str(variable.get()) for name, (variable, _, _) in entries.items()}
        settings["PSF generation"]["separate_design_actual"] = str(separate.get())
        settings["PSF generation"]["channels"] = json.dumps(channels.get("1.0", "end-1c"))
        try:
            settings_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = settings_path.with_suffix(".tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                settings.write(handle)
            temporary.replace(settings_path)
        except OSError as error:
            events.put(f"Cannot save settings: {error}\n")

    def channels_changed(event):
        if channels.edit_modified():
            channels.edit_modified(False)
            save_settings()

    def restore_defaults():
        nonlocal saving_paused
        saving_paused = True
        group = mode.get()
        for name, value in defaults[group].items():
            fields[group][name][0].set(value)
        if group == "PSF generation":
            separate.set(False)
            channels.delete("1.0", "end")
            channels.insert("1.0", default_channels)
            channels.edit_modified(False)
            show_actual()
        saving_paused = False
        save_settings()

    settings = configparser.ConfigParser(interpolation=None)
    try:
        load_path = settings_path if settings_path.exists() else ROOT / "userdata" / "settings.ini"
        settings.read(load_path, encoding="utf-8")
        for group, entries in fields.items():
            for name, (variable, kind, _) in entries.items():
                if settings.has_option(group, name):
                    value = settings.getboolean(group, name) if kind is bool else settings.get(group, name)
                    variable.set(value)
        separate.set(settings.getboolean("PSF generation", "separate_design_actual", fallback=False))
        saved_channels = json.loads(settings.get("PSF generation", "channels", fallback=json.dumps(default_channels)))
        if not isinstance(saved_channels, str):
            raise ValueError("Invalid saved channel text")
        channels.delete("1.0", "end")
        channels.insert("1.0", saved_channels)
        saved_mode = settings.get("GUI", "mode", fallback="Deconvolution")
        if saved_mode in panels:
            mode.set(saved_mode)
    except (OSError, UnicodeError, ValueError, configparser.Error) as error:
        events.put(f"Could not load all saved settings: {error}\n")

    for entries in fields.values():
        for variable, _, _ in entries.values():
            variable.trace_add("write", save_settings)
    mode.trace_add("write", save_settings)
    separate.trace_add("write", save_settings)
    channels.edit_modified(False)
    channels.bind("<<Modified>>", channels_changed)
    save_settings()

    def start():
        nonlocal process, cancelled
        group = mode.get()
        settings = {"module": "batch_deconvolve_ims" if group == "Deconvolution" else "generate_gibson_lanni_psfs"}
        try:
            hidden = {actual for _, actual, _, _ in pairs} if group == "PSF generation" and not separate.get() else set()
            for name, (variable, kind, label) in fields[group].items():
                if name in hidden:
                    continue
                try:
                    value = kind(variable.get())
                    if kind is str:
                        value = value.strip()
                        if not value:
                            raise ValueError()
                    if kind in (int, float):
                        if not math.isfinite(value) or value < 0 or (value == 0 and name not in ("IMAGING_DEPTH_UM", "RLTV_REGULARIZATION")):
                            raise ValueError()
                except (ValueError, tk.TclError):
                    raise ValueError(f"Enter a valid value for {label}.") from None
                settings[name] = value
            if group == "PSF generation":
                if not separate.get():
                    for design, actual, _, _ in pairs:
                        settings[actual] = settings[design]
                settings["CHANNELS"] = []
                for line in channels.get("1.0", "end").splitlines():
                    if not line.strip():
                        continue
                    parts = [part.strip() for part in line.split(",")]
                    if len(parts) != 3 or not parts[0]:
                        raise ValueError("Each channel needs: name, excitation (nm), emission (nm).")
                    excitation, emission = map(float, parts[1:])
                    if not all(math.isfinite(value) and value > 0 for value in (excitation, emission)):
                        raise ValueError("Channel wavelengths must be positive numbers.")
                    settings["CHANNELS"].append(dict(name=parts[0], excitation_nm=excitation, emission_nm=emission))
                if not settings["CHANNELS"]:
                    raise ValueError("Enter at least one channel.")
            process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--worker"],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace", cwd=ROOT,
                                       start_new_session=True)
            process.stdin.write(json.dumps(settings))
            process.stdin.close()
        except (ValueError, OSError) as error:
            messagebox.showerror("Cannot start", str(error), parent=window)
            return
        run_button.configure(state="disabled")
        cancelled = False
        cancel_button.configure(state="normal")
        events.put(f"\nStarting {mode_labels[group]}...\n")
        threading.Thread(target=read_output, args=(process,), daemon=True).start()

    def cancel():
        nonlocal cancelled
        if process is None or process.poll() is not None:
            return
        cancelled = True
        cancel_button.configure(state="disabled")
        events.put("\nCancelling...\n")
        child = process
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        window.after(2000, lambda: force_stop(child))

    def force_stop(child):
        if process is child:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def read_output(child):
        for line in child.stdout:
            events.put(line)
        child.stdout.close()
        events.put(child.wait())

    def poll():
        nonlocal process
        for _ in range(500):
            try:
                item = events.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, int):
                item = "\nCancelled.\n" if cancelled else "\nFinished successfully.\n" if item == 0 else f"\nFailed (exit code {item}). See output above.\n"
                process = None
                run_button.configure(state="normal")
                cancel_button.configure(state="disabled")
            console.configure(state="normal")
            console.insert("end", item)
            console.see("end")
            console.configure(state="disabled")
        window.after(100, poll)

    def close():
        if process is not None:
            messagebox.showinfo("Job running", "Cancel the current job or wait for it to finish before closing.", parent=window)
        else:
            window.destroy()

    buttons = ttk.Frame(window)
    buttons.pack(fill="x", padx=10, pady=5)
    run_button = ttk.Button(buttons, text="Run", command=start)
    run_button.pack(side="left")
    cancel_button = ttk.Button(buttons, text="Cancel", command=cancel, state="disabled")
    cancel_button.pack(side="left", padx=5)
    ttk.Button(buttons, text="Restore defaults", command=restore_defaults).pack(side="right")
    ttk.Label(window, text="Console").pack(anchor="w", padx=10)
    console = ScrolledText(window, height=12, state="disabled", wrap="word")
    console.pack(fill="both", expand=True, padx=10, pady=(5, 10))
    show_actual()
    show_mode()
    window.protocol("WM_DELETE_WINDOW", close)
    poll()
    window.mainloop()


if __name__ == "__main__":
    run_worker() if "--worker" in sys.argv else main()
