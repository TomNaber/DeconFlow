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

from app_version import __version__

from source.platform_support import default_fiji_executable, process_session_options, stop_process_tree

ROOT = Path(__file__).resolve().parent
WORKSPACE = Path(os.environ.get("BATCH_DECONVOLUTION_WORKSPACE", ROOT))


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
    from source.channel_metadata import read_channels
    from source.edit_ims_channels import NAMED_COLORS
    from source.psf_channels import read_psf_channels
    from source.preview import Preview
    from source.ui_style import workspace_background

    window = tk.Tk()
    window.title(f"Batch Deconvolution {__version__} — Richardson-Lucy and Gibson Lanni")
    width = min(1700, (window.winfo_screenwidth() or 1760) - 60)
    height = min(1000, (window.winfo_screenheight() or 1100) - 100)
    window.geometry(f"{width}x{height}")
    window.minsize(min(1100, width), min(700, height))
    mode = tk.StringVar(value="Deconvolution")
    separate = tk.BooleanVar(value=False)
    fields = {}
    process = None
    running_settings = None
    cancelled = False
    events = queue.Queue()

    workspace = tk.Frame(window, background=workspace_background(window), borderwidth=0)
    workspace.pack(fill="both", expand=True)
    workspace.bind("<Map>", lambda event: workspace.configure(background=workspace_background(window)))
    # Keep controls compact; give additional window width to the preview.
    workspace.columnconfigure(0, weight=0)
    workspace.columnconfigure(1, weight=1)
    workspace.rowconfigure(0, weight=1)
    parameter_panel = ttk.LabelFrame(workspace, padding=8)
    parameter_panel.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
    settings_area = parameter_panel
    settings_area.rowconfigure(0, weight=1)
    settings_area.columnconfigure(0, weight=1)
    settings_canvas = tk.Canvas(settings_area, highlightthickness=0, width=580, height=1)
    settings_canvas.grid(row=0, column=0, sticky="nsew")
    vertical = ttk.Scrollbar(settings_area, orient="vertical", command=settings_canvas.yview)
    vertical.grid(row=0, column=1, sticky="ns")
    settings_canvas.configure(yscrollcommand=vertical.set)
    controls = ttk.Frame(settings_canvas)
    controls_item = settings_canvas.create_window(0, 0, window=controls, anchor="nw")

    def layout_controls(event=None):
        # A notebook otherwise reserves the height of its tallest page.
        notebook.configure(height=panels[mode.get()].winfo_reqheight())
        available_height = settings_canvas.winfo_height()
        content_height = controls.winfo_reqheight()
        settings_canvas.itemconfigure(controls_item, width=settings_canvas.winfo_width(),
                                      height=max(content_height, available_height))
        settings_canvas.configure(scrollregion=settings_canvas.bbox("all"))
        if controls.winfo_reqheight() > settings_canvas.winfo_height():
            vertical.grid()
        else:
            vertical.grid_remove()
            settings_canvas.yview_moveto(0)

    controls.bind("<Configure>", layout_controls)
    settings_canvas.bind("<Configure>", layout_controls)
    notebook = ttk.Notebook(controls, padding=0)
    notebook.pack(fill="x", pady=(10, 0))
    panels = {}
    mode_labels = {
        "Deconvolution": "Richardson-Lucy deconvolution",
        "PSF generation": "Gibson-Lanni PSF",
    }
    for title in mode_labels:
        panels[title] = ttk.Frame(notebook, padding=10)
        notebook.add(panels[title], text=mode_labels[title])
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
            ttk.Combobox(holder, textvariable=variable, values=("Up", "Down"),
                         state="readonly", width=6).pack(side="left")
        else:
            ttk.Entry(holder, textvariable=variable, width=1).pack(side="left", fill="x", expand=True)
        if browse:
            def choose():
                current = Path(variable.get().strip()).expanduser()
                initial = current if browse == "directory" else current.parent
                while not initial.is_dir() and initial != initial.parent:
                    initial = initial.parent
                if browse == "directory":
                    value = filedialog.askdirectory(parent=window, initialdir=str(initial.resolve()))
                else:
                    value = filedialog.askopenfilename(parent=window, initialdir=str(initial.resolve()))
                if value:
                    variable.set(value)
            ttk.Button(holder, text="Browse", command=choose).pack(side="left", padx=(5, 0))
        return caption, holder

    deconvolution = [
        ("INPUT_DIR", "Input folder (Imaris .ims files)", WORKSPACE / "examples", str, "directory"),
        ("PSF_DIR", "Point-spread function folder", WORKSPACE / "psfs", str, "directory"),
        ("OUTPUT_DIR", "Output folder", WORKSPACE / "dcv", str, "directory"),
        ("TEMP_DIR", "Temporary working folder", WORKSPACE / ".deconvolution_temp", str, "directory"),
        ("FIJI_EXECUTABLE", "Fiji executable", default_fiji_executable(), str, "file"),
        ("FIJI_MEMORY", "Fiji memory limit (e.g. 12g = 12 GB)", "12g", str, None),
        ("ITERATIONS", "Richardson-Lucy iterations", 40, int, None),
        ("RLTV_REGULARIZATION", "Total-variation regularization", 0.0001, float, None),
        ("RLTV_NON_CIRCULANT", "Non-circulant boundary correction", True, bool, None),
        ("OVERWRITE", "Overwrite existing output", False, bool, None),
    ]
    for row, setting in enumerate(deconvolution):
        add_field(panels["Deconvolution"], "Deconvolution", row, *setting)

    channel_panel = ttk.LabelFrame(panels["Deconvolution"], text="Select channels", padding=8)
    channel_panel.grid(row=len(deconvolution), column=0, columnspan=2, sticky="ew", pady=(10, 0))
    channel_panel.columnconfigure(1, weight=1)
    channel_preferences = {}
    channel_variables = {}
    channel_row_updates = []
    active_channels = []
    psf_signature = []
    psf_overrides = {}
    psf_selections = {}
    refresh_job = None

    def preview_error(error):
        events.put(f"Preview: {error}\n")
        messagebox.showerror("Preview failed", error.splitlines()[-1], parent=window)

    preview_panel = ttk.LabelFrame(workspace, padding=8)
    preview_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
    preview = Preview(preview_panel, channel_preferences, lambda: refresh_channels(), preview_error)
    preview.pack(fill="both", expand=True)

    def preview_settings():
        if refresh_job is not None:
            window.after_cancel(refresh_job)
            refresh_channels()
        settings = {}
        for name in ("ITERATIONS", "RLTV_REGULARIZATION", "RLTV_NON_CIRCULANT",
                     "FIJI_EXECUTABLE", "FIJI_MEMORY", "PSF_DIR", "TEMP_DIR"):
            variable, kind, label = fields["Deconvolution"][name]
            try:
                value = kind(variable.get())
                if kind is str:
                    value = value.strip()
                    if not value:
                        raise ValueError()
                if kind in (int, float) and (not math.isfinite(value) or value < 0
                        or (name == "ITERATIONS" and value == 0)):
                    raise ValueError()
            except (ValueError, tk.TclError):
                raise ValueError(f"Enter a valid value for {label}.") from None
            settings[name] = value
        settings["PSF_FILES"] = []
        settings["CHANNEL_INDICES"] = []
        settings["OUTPUT_CHANNEL_NAMES"] = {}
        for channel in active_channels:
            if channel["index"] not in preview.selected_channels:
                continue
            filename = psf_selections.get(str(channel["index"]))
            if not filename:
                raise ValueError(f"Select an available PSF for C{channel['index'] + 1}.")
            settings["PSF_FILES"].append(filename)
            settings["CHANNEL_INDICES"].append(channel["index"])
            settings["OUTPUT_CHANNEL_NAMES"][str(channel["index"])] = channel_preferences.get(
                str(channel["index"]), {}).get("name", channel["name"])
        return settings

    preview.get_settings = preview_settings
    preview.get_temp_directory = lambda: fields["Deconvolution"]["TEMP_DIR"][0].get().strip() or None
    preview.on_log = events.put

    def refresh_channels():
        nonlocal active_channels, refresh_job, psf_signature
        window.after_idle(layout_controls)
        refresh_job = None
        for widget in channel_panel.winfo_children():
            widget.destroy()
        channel_variables.clear()
        channel_row_updates.clear()
        psf_selections.clear()
        active_channels = []
        folder = Path(fields["Deconvolution"]["INPUT_DIR"][0].get()).expanduser()
        try:
            preview.set_folder(folder)
            if preview.selected_path is None:
                raise ValueError("No .ims files in the selected input folder.")
            active_channels = read_channels(preview.selected_path)
            if not active_channels:
                raise ValueError("No channels found in the selected input file.")
        except (OSError, KeyError, ValueError) as error:
            ttk.Label(channel_panel, text=str(error), wraplength=520).grid(row=0, column=0, columnspan=4, sticky="w")
            preview.parameters_changed()
            return
        signature = [channel["name"] for channel in active_channels]
        if signature != psf_signature:
            psf_overrides.clear()
            psf_signature = signature
        try:
            catalog = read_psf_channels(Path(fields["Deconvolution"]["PSF_DIR"][0].get()).expanduser())
            psf_error = None
        except (OSError, KeyError, TypeError, ValueError) as error:
            catalog = []
            psf_error = str(error)
        automatic = {str(entry["index"]): entry["psf_file"] for entry in catalog}
        labels = {
            entry["psf_file"]: (entry["name"] if sum(other["name"] == entry["name"] for other in catalog) == 1
                                else f"{entry['name']} ({entry['psf_file']})")
            for entry in catalog
        }
        heading_widths = {}
        for column, label in enumerate(("Input channel", "Output channel name", "PSF channel", "Pseudocolor")):
            heading = ttk.Label(channel_panel, text=label)
            heading.grid(row=0, column=column, sticky="w", padx=5)
            heading_widths[column] = heading.winfo_reqwidth()

        def channel_dropdown(row, column, variable, values):
            holder = ttk.Frame(channel_panel, width=heading_widths[column] + 28)
            holder.grid(row=row, column=column, sticky="ew", padx=5, pady=3)
            dropdown = ttk.Combobox(holder, textvariable=variable, values=values,
                                    state="readonly", width=1)
            holder.configure(height=dropdown.winfo_reqheight())
            holder.pack_propagate(False)
            dropdown.pack(fill="x")
            return dropdown

        for row, channel in enumerate(active_channels, start=1):
            key = str(channel["index"])
            preference = channel_preferences.setdefault(key, {"name": channel["name"], "color": "Original"})
            enabled = tk.BooleanVar(value=preference.get("enabled", True))
            name = tk.StringVar(value=preference["name"])
            color = tk.StringVar(value=preference["color"])
            label = f"C{channel['index'] + 1}: {channel['name']}"
            input_cell = ttk.Frame(channel_panel)
            input_cell.grid(row=row, column=0, sticky="w", padx=5, pady=3)
            selection = ttk.Checkbutton(input_cell, variable=enabled)
            selection.pack(side="left")
            ttk.Label(input_cell, text=label, wraplength=90).pack(side="left")
            name_entry = ttk.Entry(channel_panel, textvariable=name, width=1)
            name_entry.grid(row=row, column=1, sticky="ew", padx=5, pady=3)
            selected_file = psf_overrides.get(key, automatic.get(key))
            if key in psf_overrides and selected_file == automatic.get(key):
                del psf_overrides[key]
            psf = tk.StringVar(value=labels.get(selected_file, f"Unavailable: {selected_file}" if selected_file else "No matching PSF"))
            psf_selections[key] = selected_file if selected_file in labels else None
            psf_dropdown = channel_dropdown(row, 2, psf, tuple(labels.values()))
            color_dropdown = channel_dropdown(row, 3, color, ("Original", *NAMED_COLORS))

            def update_row(*args, key=key, enabled=enabled, name_entry=name_entry,
                           psf_dropdown=psf_dropdown, color_dropdown=color_dropdown,
                           selection=selection):
                selected = enabled.get()
                editable = selected and not preview.disabled
                selection.configure(state="disabled" if preview.disabled else "normal")
                name_entry.configure(state="normal" if editable else "disabled")
                for dropdown in (psf_dropdown, color_dropdown):
                    dropdown.configure(state="readonly" if editable else "disabled")
                channel_preferences[key]["enabled"] = selected

            def selection_changed(*args, update_row=update_row):
                update_row()
                save_settings()
                preview.channels_changed()

            channel_row_updates.append(update_row)
            update_row()
            enabled.trace_add("write", selection_changed)

            def psf_changed(*args, key=key, psf=psf):
                filename = next(filename for filename, label in labels.items() if label == psf.get())
                psf_selections[key] = filename
                if filename == automatic.get(key):
                    psf_overrides.pop(key, None)
                else:
                    psf_overrides[key] = filename
                save_settings()
                preview.parameters_changed()

            psf.trace_add("write", psf_changed)
            # Keep the Tk variable alive for the lifetime of its dropdown.
            channel_variables[key] = (name, color, psf, enabled)

            def changed(*args, key=key, name=name, color=color):
                channel_preferences[key].update(name=name.get(), color=color.get())
                save_settings()
                preview.recolor()

            name.trace_add("write", changed)
            color.trace_add("write", changed)
        if psf_error:
            ttk.Label(channel_panel, text=psf_error, wraplength=520).grid(
                row=len(active_channels) + 1, column=0, columnspan=4, sticky="w")
        save_settings()
        preview.parameters_changed()

    def schedule_channel_refresh(*args):
        nonlocal refresh_job
        preview.parameters_changed()
        if refresh_job is not None:
            window.after_cancel(refresh_job)
        refresh_job = window.after(300, refresh_channels)

    psf_panel = panels["PSF generation"]
    ttk.Checkbutton(psf_panel, text="Separate objective design and actual imaging conditions", variable=separate,
                    command=lambda: show_actual()).pack(anchor="w", pady=(0, 5))
    psf_settings = ttk.Frame(psf_panel)
    psf_settings.pack(fill="x")
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
        ("SCAN_DIRECTION", "Axial scan direction", "Up", str),
    ]
    for row, setting in enumerate(psf_fields):
        add_field(psf_settings, "PSF generation", row, *setting)

    pairs = [
        ("IMMERSION_RI_DESIGN", "IMMERSION_RI_ACTUAL", "Immersion refractive index", 1.4811),
        ("COVERSLIP_RI_DESIGN", "COVERSLIP_RI_ACTUAL", "Coverslip refractive index", 1.5255),
        ("COVERSLIP_THICKNESS_DESIGN_UM", "COVERSLIP_THICKNESS_ACTUAL_UM", "Coverslip thickness (µm)", 145.0),
    ]
    pair_widgets = []
    for index, (design, actual, label, default) in enumerate(pairs):
        design_widgets = add_field(psf_settings, "PSF generation", len(psf_fields) + index * 2, design, label, default, float)
        actual_widgets = add_field(psf_settings, "PSF generation", len(psf_fields) + index * 2 + 1, actual, label + " (actual)", default, float)
        pair_widgets.append((design_widgets[0], actual_widgets, label))
    add_field(psf_settings, "PSF generation", len(psf_fields) + len(pairs) * 2, "OVERWRITE", "Overwrite existing point-spread functions", True, bool)
    add_field(psf_settings, "PSF generation", len(psf_fields) + len(pairs) * 2 + 1, "OUTPUT_DIR", "Output folder", WORKSPACE / "psfs/1000", str, "directory")
    ttk.Label(psf_panel, wraplength=540, text="Channels in image channel order: one per line, name, excitation wavelength (nm), emission wavelength (nm)").pack(anchor="w")
    channels = ScrolledText(psf_panel, width=1, height=4, wrap="none")
    channels.pack(fill="x", pady=5)
    channels.insert("1.0", "Alexa488, 488, 525\nAlexa568, 552, 603\nAlexa647, 638, 671")

    def show_actual():
        window.after_idle(layout_controls)
        for caption, widgets, label in pair_widgets:
            caption.configure(text=label + (" (design)" if separate.get() else ""))
            for widget in widgets:
                widget.grid() if separate.get() else widget.grid_remove()

    def show_mode():
        notebook.select(panels[mode.get()])
        console.configure(height=12 if mode.get() == "Deconvolution" else 8)
        window.after_idle(layout_controls)

    def tab_changed(event=None):
        selected = notebook.select()
        for title, panel in panels.items():
            if str(panel) == selected and mode.get() != title:
                mode.set(title)
                show_mode()
                break

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
        settings["Channel pseudocolor"] = {
            key: json.dumps(value) for key, value in channel_preferences.items()
        }
        if psf_overrides:
            settings["PSF assignment"] = {
                "input_channels": json.dumps(psf_signature),
                "overrides": json.dumps(psf_overrides),
            }
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
        else:
            channel_preferences.clear()
            psf_overrides.clear()
            refresh_channels()
            preview.recolor()
        saving_paused = False
        save_settings()

    settings = configparser.ConfigParser(interpolation=None)
    try:
        load_path = settings_path if settings_path.exists() else ROOT / "userdata" / "settings.ini"
        settings.read(load_path, encoding="utf-8")
        if settings.has_section("PSF assignment"):
            signature = json.loads(settings.get("PSF assignment", "input_channels"))
            overrides = json.loads(settings.get("PSF assignment", "overrides"))
            if (not isinstance(signature, list) or not all(isinstance(name, str) for name in signature)
                    or not isinstance(overrides, dict)
                    or not all(key.isdigit() and isinstance(value, str) for key, value in overrides.items())):
                raise ValueError("Invalid saved PSF assignment")
            psf_signature = signature
            psf_overrides.update(overrides)
        if settings.has_section("Channel pseudocolor"):
            for key, raw in settings.items("Channel pseudocolor"):
                preference = json.loads(raw)
                if (not key.isdigit() or not isinstance(preference, dict)
                        or not isinstance(preference.get("name"), str)
                        or preference.get("color") not in ("Original", *NAMED_COLORS)
                        or not isinstance(preference.get("enabled", True), bool)):
                    raise ValueError("Invalid saved channel settings")
                channel_preferences[key] = preference
        for group, entries in fields.items():
            for name, (variable, kind, _) in entries.items():
                if settings.has_option(group, name):
                    value = settings.getboolean(group, name) if kind is bool else settings.get(group, name)
                    variable.set(value.capitalize() if name == "SCAN_DIRECTION" else value)
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
    fields["Deconvolution"]["INPUT_DIR"][0].trace_add("write", schedule_channel_refresh)
    fields["Deconvolution"]["PSF_DIR"][0].trace_add("write", schedule_channel_refresh)
    for name in ("ITERATIONS", "RLTV_REGULARIZATION", "RLTV_NON_CIRCULANT", "FIJI_EXECUTABLE", "FIJI_MEMORY"):
        fields["Deconvolution"][name][0].trace_add("write", preview.parameters_changed)
    fiji_init_job = None

    def initialize_fiji():
        nonlocal fiji_init_job
        fiji_init_job = None
        if process is not None:
            return
        try:
            preview.configure_fiji(fields["Deconvolution"]["FIJI_EXECUTABLE"][0].get().strip(),
                                   fields["Deconvolution"]["FIJI_MEMORY"][0].get().strip())
        except (OSError, ValueError) as error:
            events.put(f"{error}\n")

    def schedule_fiji_initialization(*args):
        nonlocal fiji_init_job
        if fiji_init_job is not None:
            window.after_cancel(fiji_init_job)
        fiji_init_job = window.after(500, initialize_fiji)

    for name in ("FIJI_EXECUTABLE", "FIJI_MEMORY"):
        fields["Deconvolution"][name][0].trace_add("write", schedule_fiji_initialization)
    schedule_fiji_initialization()
    refresh_channels()
    save_settings()

    def start():
        nonlocal process, cancelled, running_settings
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
                settings[name] = value.lower() if name == "SCAN_DIRECTION" else value
            if group == "Deconvolution":
                if refresh_job is not None:
                    window.after_cancel(refresh_job)
                refresh_channels()
                if not active_channels:
                    raise ValueError("Select an input folder with readable IMS channel metadata.")
                settings["CHANNEL_EDITS"] = {}
                settings["PSF_FILES"] = []
                settings["CHANNEL_INDICES"] = []
                for channel in active_channels:
                    key = str(channel["index"])
                    if not channel_preferences[key].get("enabled", True):
                        continue
                    settings["CHANNEL_INDICES"].append(channel["index"])
                    if not psf_selections.get(key):
                        raise ValueError(f"Select an available PSF for C{int(key) + 1}.")
                    settings["PSF_FILES"].append(psf_selections[key])
                    preference = channel_preferences[key]
                    if not preference["name"].strip():
                        raise ValueError(f"Enter an output name for C{int(key) + 1}.")
                    edit = {"name": preference["name"].strip()}
                    if preference["color"] != "Original":
                        edit["color"] = preference["color"]
                    settings["CHANNEL_EDITS"][key] = edit
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
            if group == "Deconvolution":
                preview.cancel_deconvolution()
                preview.stop_fiji()
            process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--worker"],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace", cwd=ROOT,
                                       **process_session_options())
            process.stdin.write(json.dumps(settings))
            process.stdin.close()
            running_settings = settings
        except (ValueError, OSError) as error:
            messagebox.showerror("Cannot start", str(error), parent=window)
            return
        run_button.configure(state="disabled")
        preview.set_disabled(True)
        cancelled = False
        cancel_button.configure(state="normal")
        events.put(f"\nStarting {mode_labels[group]}...\n")
        threading.Thread(target=read_output, args=(process,), daemon=True).start()

    def cancel():
        nonlocal cancelled
        if process is None or process.poll() is not None:
            if preview.busy == "deconvolve" or preview.pending is not None or preview.sync_job is not None:
                preview.cancel_deconvolution()
                events.put("\nPreview cancelled.\n")
            return
        cancelled = True
        cancel_button.configure(state="disabled")
        events.put("\nCancelling...\n")
        child = process
        stop_process_tree(child.pid)
        window.after(2000, lambda: force_stop(child))

    def force_stop(child):
        if process is child and child.poll() is None:
            stop_process_tree(child.pid, force=True)

    def read_output(child):
        # Keep carriage returns intact for live progress updates.
        import codecs
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        while chunk := child.stdout.buffer.read1(4096):
            events.put(decoder.decode(chunk))
        tail = decoder.decode(b"", final=True)
        if tail:
            events.put(tail)
        child.stdout.close()
        events.put(child.wait())

    def poll():
        nonlocal process, running_settings
        for _ in range(500):
            try:
                item = events.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, int):
                succeeded = item == 0 and not cancelled
                refresh_preview = succeeded
                if succeeded and running_settings["module"] == "generate_gibson_lanni_psfs":
                    # Refresh only when the generated files belong to the current PSF target.
                    output_folder = Path(running_settings["OUTPUT_DIR"]).expanduser().resolve()
                    target_folder = Path(fields["Deconvolution"]["PSF_DIR"][0].get()).expanduser().resolve()
                    refresh_preview = output_folder == target_folder
                item = "\nCancelled.\n" if cancelled else "\nFinished successfully.\n" if item == 0 else f"\nFailed (exit code {item}). See output above.\n"
                process = None
                running_settings = None
                run_button.configure(state="normal")
                cancel_button.configure(state="disabled")
                preview.set_disabled(False)
                schedule_fiji_initialization()
                if refresh_preview:
                    if refresh_job is not None:
                        window.after_cancel(refresh_job)
                    refresh_channels()
            console.configure(state="normal")
            for index, part in enumerate(item.split("\r")):
                if index:
                    console.delete("end-1c linestart", "end-1c")
                if part:
                    console.insert("end", part)
            console.see("end")
            console.configure(state="disabled")
        window.after(100, poll)

    def close():
        if process is not None:
            messagebox.showinfo("Job running", "Cancel the current job or wait for it to finish before closing.", parent=window)
        else:
            preview.close()
            window.destroy()

    buttons = ttk.Frame(controls)
    buttons.pack(fill="x", padx=10, pady=5)
    run_button = ttk.Button(buttons, text="Run", command=start)
    run_button.pack(side="left")
    cancel_button = ttk.Button(buttons, text="Cancel", command=cancel, state="disabled")
    cancel_button.pack(side="left", padx=5)
    def update_activity():
        cancel_button.configure(
            state="normal" if (process is not None and not cancelled) or preview.busy == "deconvolve" else "disabled")
        for update_row in channel_row_updates:
            update_row()

    preview.on_activity = update_activity
    ttk.Button(buttons, text="Restore defaults", command=restore_defaults).pack(side="right")
    console_area = ttk.Frame(controls)
    console_area.pack(side="bottom", fill="both", expand=True, padx=10, pady=(18, 10))
    ttk.Label(console_area, text="Console").pack(anchor="w")
    console = ScrolledText(console_area, width=1, height=8, state="disabled", wrap="word")
    console.pack(fill="both", expand=True, pady=(5, 0))
    notebook.bind("<<NotebookTabChanged>>", tab_changed)
    show_actual()
    show_mode()
    window.protocol("WM_DELETE_WINDOW", close)

    def activate_on_startup():
        window.update_idletasks()
        window.lift()
        if sys.platform == "darwin":
            # Raising a Tk window alone does not activate Python above an IDE.
            from AppKit import NSApplicationActivateIgnoringOtherApps, NSRunningApplication
            NSRunningApplication.currentApplication().activateWithOptions_(
                NSApplicationActivateIgnoringOtherApps)
        window.focus_force()

    # Run once after layout, including when preview setup already mapped the window.
    activate_on_startup()
    poll()
    window.mainloop()


if __name__ == "__main__":
    if "--version" in sys.argv:
        print(__version__)
    else:
        run_worker() if "--worker" in sys.argv else main()
