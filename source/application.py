"""Unified desktop workflow and coordination of PSF and deconvolution jobs."""
import configparser
import copy
import json
import math
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

import h5py

from app_version import __version__
from source.channel_adjustment import ChannelAdjustment, NEUTRAL_CONTRAST, contrast_maximum
from source.channel_metadata import read_channels, text
from source.edit_ims_channels import NAMED_COLORS
from source.platform_support import process_session_options, stop_process_tree
from source.preview import Preview
from source.psf_channels import read_psf_channels
from source.psf_session import PSFSession, memory_argument, memory_gb
from source.psf_view import PSFView, load_projections
from source.ui_style import workspace_background

ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = Path(os.environ.get('BATCH_DECONVOLUTION_WORKSPACE', ROOT))
GENERATOR_FIELDS = [
    ('PSF_SIZE_Z', 'PSF Z size (planes)', 209, int),
    ('PSF_SIZE_XY', 'PSF X/Y size (pixels)', 119, int),
    ('OVERSAMPLE_FACTOR', 'Calculation oversampling factor', 3, int),
    ('SCAN_DIRECTION', 'Axial scan direction', 'Up', str),
    ('XY_PIXEL_SIZE_UM', 'Lateral pixel size X/Y (µm)', .0801669, float),
    ('Z_STEP_UM', 'Axial sampling interval Z (µm)', .3002, float),
    ('NUMERICAL_APERTURE', 'Objective numerical aperture', 1.4, float),
    ('SPECIMEN_RI', 'Specimen refractive index', 1.45, float),
    ('OBJECTIVE_WORKING_DISTANCE_UM', 'Objective working distance (µm)', 140., float),
    ('IMAGING_DEPTH_UM', 'Imaging depth (µm)', 400., float),
    ('PINHOLE_AIRY_UNITS', 'Confocal pinhole (Airy units)', 1., float),
]
PAIRS = [
    ('IMMERSION_RI_DESIGN', 'IMMERSION_RI_ACTUAL', 'Immersion refractive index', 1.4811),
    ('COVERSLIP_RI_DESIGN', 'COVERSLIP_RI_ACTUAL', 'Coverslip refractive index', 1.5255),
    ('COVERSLIP_THICKNESS_DESIGN_UM', 'COVERSLIP_THICKNESS_ACTUAL_UM', 'Coverslip thickness (µm)', 145.),
]
DEFAULT_CHANNELS = [dict(name=n, excitation_nm=x, emission_nm=e) for n,x,e in
                    [('Alexa488',488,525),('Alexa568',552,603),('Alexa647',638,671)]]


class Application:
    def __init__(self, window, settings_path=None):
        self.window = window
        self.loading = True
        self.fields, self.defaults, self.field_widgets = {}, {}, {}
        self.auto_spacing_vars, self.auto_spacing_widgets = {}, {}
        self.input_spacing = {}
        self.preferences, self.definitions, self.assignments = {}, {}, {'internal':{}, 'external':{}}
        self.active_channels, self.imported_files, self.imported_catalog = [], [], []
        self.import_error = None
        self.signature = []
        self.channel_vars, self.definition_vars = {}, {}
        self.enabled_vars, self.external_vars = {}, {}
        self.default_regularization = .0001
        self.session = PSFSession()
        self.events = queue.Queue()
        self.process = self.running = self.pending = None
        self.batch_locked = False
        self.export_task = None
        self.cancelled = False
        self.revision = 0
        self.psf_revision = 0
        self.refresh_job = None
        self.view_token = 0
        self.view_loading = False
        self.psf_sync_job = None
        self.view_threads = []
        self.locked_widgets = {}
        self.source = tk.StringVar(value='internal')
        self.separate = tk.BooleanVar(value=False)
        self.export_enabled = tk.BooleanVar(value=False)
        self.export_folder = tk.StringVar()
        self.import_display = tk.StringVar()
        self.stack_status = tk.StringVar()
        if settings_path is None:
            if sys.platform == 'darwin':
                base = Path.home()/'Library'/'Application Support'
            elif sys.platform == 'win32':
                base = Path(os.environ.get('LOCALAPPDATA') or Path.home()/'AppData'/'Local')
            else:
                base = Path(os.environ.get('XDG_DATA_HOME') or Path.home()/'.local'/'share')
            settings_path = base/'Batch Deconvolution'/'settings.ini'
        self.settings_path = Path(settings_path)
        self.build()
        self.load_settings()
        self.loading = False
        self.show_source()
        self.refresh_input()
        self.preview.get_settings = self.preview_settings
        self.preview.psfs_ready = self.can_resolve
        self.preview.request_psfs = lambda automatic=False:self.request('preview', automatic)
        self.preview.on_cancel = self.cancel
        self.preview.on_activity = self.update_activity
        self.preview.on_log = self.log
        self.window.protocol('WM_DELETE_WINDOW', self.close)
        self.poll_job = self.window.after(100, self.poll)
        self.update_activity()

    def build(self):
        w = self.window
        w.title(f'Batch Deconvolution {__version__} — Richardson-Lucy and Gibson-Lanni')
        w.geometry(f'{min(1700,(w.winfo_screenwidth() or 1760)-60)}x{min(1000,(w.winfo_screenheight() or 1100)-100)}')
        w.minsize(1100,700)
        workspace = tk.Frame(w, background=workspace_background(w))
        workspace.pack(fill='both', expand=True)
        workspace.columnconfigure(1, weight=1)
        workspace.rowconfigure(0, weight=1)
        left = ttk.Frame(workspace, padding=8, width=590)
        left.grid(row=0,column=0,sticky='nsew',padx=(10,5),pady=10)
        left.columnconfigure(0,weight=1)
        left.rowconfigure(0,weight=1)
        outer = ttk.Notebook(left)
        outer.grid(row=0,column=0,sticky='nsew')
        # Native themes can draw the notebook border inside its widget bounds.
        notebook_padding = w.tk.splitlist(ttk.Style(w).lookup('TNotebook','padding') or '0')
        side_padding = (notebook_padding[0], notebook_padding[2] if len(notebook_padding)>2 else notebook_padding[0])
        page = ttk.Frame(outer)
        outer.add(page,text='Richardson-Lucy deconvolution with Gibson-Lanni PSF')
        page.columnconfigure(0,weight=1)
        page.rowconfigure(0,weight=1)
        self.settings_canvas = tk.Canvas(page,width=565,height=1,highlightthickness=0,borderwidth=0)
        self.settings_canvas.grid(row=0,column=0,sticky='nsew')
        scroll = ttk.Scrollbar(page,orient='vertical',command=self.settings_canvas.yview)
        scroll.grid(row=0,column=1,sticky='ns')
        self.settings_canvas.configure(yscrollcommand=scroll.set)
        self.controls = ttk.Frame(self.settings_canvas,padding=8)
        self.settings_item = self.settings_canvas.create_window(0,0,window=self.controls,anchor='nw')
        self.settings_layout_job = None
        self.settings_grid_parents = []
        self.settings_packed_rows = []
        self.controls.bind('<Configure>',self.schedule_settings_layout)
        self.settings_canvas.bind('<Configure>',self.schedule_settings_layout)
        w.bind('<MouseWheel>', self.scroll_settings, add='+')
        w.bind('<Button-4>', self.scroll_settings, add='+')
        w.bind('<Button-5>', self.scroll_settings, add='+')
        # Class bindings run before the window's scroll handler. Replace the
        # native value-changing wheel bindings, including for dynamic rows.
        for widget_class in ('TSpinbox','TCombobox','TScale'):
            for sequence in ('<MouseWheel>','<Button-4>','<Button-5>',
                             '<Shift-MouseWheel>','<Shift-Button-4>','<Shift-Button-5>'):
                w.bind_class(widget_class,sequence,self.scroll_field)
        self.setup = self.group('Setup')
        _,holder = self.field(self.setup,'Deconvolution',0,'INPUT_DIR','Z-stack input folder',WORKSPACE/'examples',str,'directory')
        ttk.Button(holder,text='Refresh',command=self.refresh_input).pack(side='left',padx=(4,0))
        ttk.Label(self.setup,textvariable=self.stack_status).grid(row=1,column=0,columnspan=2,sticky='w')
        self.field(self.setup,'Deconvolution',2,'OUTPUT_DIR','Output folder',WORKSPACE/'dcv',str,'directory')
        self.field(self.setup,'Deconvolution',3,'OVERWRITE','Overwrite existing output',False,bool)
        self.field(self.setup,'Deconvolution',4,'FIJI_MEMORY','ImageJ max memory (GB)',12,float)
        self.psf_setup = self.group('PSF setup')
        radios = ttk.Frame(self.psf_setup)
        radios.pack(fill='x')
        for label,value in [('Internal PSF','internal'),('External PSF','external')]:
            ttk.Radiobutton(radios,text=label,value=value,variable=self.source,command=self.source_changed).pack(side='left',padx=(0,12))
        self.external = ttk.Frame(self.psf_setup)
        self.external_channels = ttk.Frame(self.psf_setup)
        ttk.Label(self.external,text='PSF(s)').pack(side='left',padx=(0,8))
        ttk.Entry(self.external,textvariable=self.import_display,state='readonly',width=1).pack(side='left',fill='x',expand=True)
        ttk.Button(self.external,text='Browse…',command=self.browse_psfs).pack(side='left',padx=4)
        ttk.Button(self.external,text='Refresh',command=self.refresh_external).pack(side='left')
        self.internal = ttk.Frame(self.psf_setup)
        export_check = ttk.Checkbutton(self.internal,text='Export generated PSF',variable=self.export_enabled,command=self.export_changed)
        export_check.pack(anchor='w')
        self.export_row = ttk.Frame(self.internal)
        ttk.Label(self.export_row,text='Export folder').pack(side='left',padx=(0,8))
        ttk.Entry(self.export_row,textvariable=self.export_folder,width=1).pack(side='left',fill='x',expand=True)
        ttk.Button(self.export_row,text='Browse…',command=self.browse_export).pack(side='left',padx=4)
        self.actual_check = ttk.Checkbutton(self.internal,text='Separate objective design and actual imaging conditions',
                                           variable=self.separate,command=self.actual_changed)
        self.actual_check.pack(anchor='w',pady=4)
        scale = ttk.Frame(self.internal)
        scale.pack(fill='x')
        self.settings_packed_rows = [radios,self.external,export_check,self.export_row,self.actual_check]
        for row,setting in enumerate(GENERATOR_FIELDS[:3]):
            self.field(scale,'PSF generation',row,*setting)
        self.optics = self.group('PSF optical settings')
        for row,setting in enumerate(GENERATOR_FIELDS[3:]):
            _,holder = self.field(self.optics,'PSF generation',row,*setting)
            name = setting[0]
            if name in ('XY_PIXEL_SIZE_UM','Z_STEP_UM'):
                var = tk.BooleanVar(value=False)
                check = ttk.Checkbutton(holder,text='Auto',variable=var,state='disabled',
                                        command=self.auto_spacing_changed)
                check.pack(side='left',padx=(4,0))
                self.auto_spacing_vars[name] = var
                self.auto_spacing_widgets[name] = check
        self.pair_widgets = []
        offset = len(GENERATOR_FIELDS)-3
        for index,(design,actual,label,default) in enumerate(PAIRS):
            caption,_ = self.field(self.optics,'PSF generation',offset+index*2,design,label,default,float)
            widgets = self.field(self.optics,'PSF generation',offset+index*2+1,actual,label+' (actual)',default,float)
            self.pair_widgets.append((caption,widgets,label))
        self.definition_panel = ttk.Frame(self.optics)
        self.definition_panel.grid(row=offset+len(PAIRS)*2,column=0,columnspan=2,sticky='ew',pady=(1,8))
        self.deconvolution = self.group('Deconvolution')
        self.field(self.deconvolution,'Deconvolution',1,'RLTV_NON_CIRCULANT','Non-circulant boundary correction',True,bool)
        self.channel_panel = ttk.Frame(self.deconvolution)
        self.channel_panel.grid(row=2,column=0,columnspan=2,sticky='ew',pady=(1,8))
        self.field(self.deconvolution,'Deconvolution',3,'POST_PROCESSING','Post-processing',False,bool)
        self.post_panel = ttk.LabelFrame(self.deconvolution,text='Post-processing',padding=6)
        self.post_panel.grid(row=4,column=0,columnspan=2,sticky='ew')
        self.post_panel.grid_remove()
        self.post_vars = {}
        self.settings_grid_parents = [self.post_panel,self.setup,scale,self.optics,self.deconvolution,
                                      self.definition_panel,self.external_channels,self.channel_panel]
        actions = ttk.Frame(left,padding=(0,2,0,8))
        actions.grid(row=2,column=0,sticky='ew',padx=side_padding)
        self.run_button = ttk.Button(actions,text='Run batch deconvolution',command=lambda:self.request('batch'))
        self.run_button.pack(side='left')
        self.cancel_button = ttk.Button(actions,text='Cancel',command=self.cancel,state='disabled')
        self.cancel_button.pack(side='left',padx=5)
        self.restore_button = ttk.Button(actions,text='Restore defaults',command=self.restore_defaults)
        self.restore_button.pack(side='right')
        self.channel_adjustment = ChannelAdjustment(left, self.adjust_channel)
        self.channel_adjustment.grid(row=1,column=0,sticky='ew',pady=(2,0))
        self.channel_adjustment.grid_remove()
        self.adjustment_open = False
        self.channel_adjustment.table.bind('<Configure>',self.schedule_settings_layout,add='+')
        left.bind('<Configure>',self.schedule_settings_layout,add='+')
        self.console = ScrolledText(left,width=1,height=9,state='disabled',wrap='word')
        self.console.grid(row=3,column=0,sticky='ew',padx=side_padding,pady=(4,0))
        right = ttk.Notebook(workspace)
        right.grid(row=0,column=1,sticky='nsew',padx=(5,10),pady=10)
        self.preview = Preview(right,self.preferences,self.refresh_input,self.error)
        right.add(self.preview,text='Preview deconvolution')
        self.psf_view = PSFView(right,self.preferences,lambda:self.request('view'),self.log,self.error)
        right.add(self.psf_view,text='View PSF')
        self.psf_view.cancel_button.configure(command=self.cancel)
        self.psf_view.sync_button.configure(command=self.schedule_psf_sync)
        for view in (self.preview, self.psf_view):
            view.adjustment_button.configure(command=self.toggle_channel_adjustment)
        self.viewer_layout_job = None
        for view in (self.preview,self.psf_view):
            for section in ('toolbar','actions'):
                getattr(view,section).bind('<Configure>',self.schedule_viewer_layout,add='+')
                getattr(view,section+'_shell').pack_propagate(False)
        right.bind('<<NotebookTabChanged>>',self.schedule_viewer_layout,add='+')
        right.bind('<Configure>',self.schedule_viewer_layout,add='+')
        self.schedule_viewer_layout()
        self.export_folder.trace_add('write',lambda *args:self.save_settings())

    def schedule_viewer_layout(self,event=None):
        if self.viewer_layout_job is None:
            self.viewer_layout_job = self.window.after_idle(self.align_viewers)

    def align_viewers(self):
        self.viewer_layout_job = None
        # Update the hidden preview's wrapping too, before measuring shared rows.
        self.preview.layout_actions()
        self.psf_view.layout_actions()
        for section in ('toolbar','actions'):
            height = (max(view.actions_height for view in (self.preview,self.psf_view)) if section=='actions' else
                      max(getattr(view,section).winfo_reqheight() for view in (self.preview,self.psf_view)))
            for view in (self.preview,self.psf_view):
                shell = getattr(view,section+'_shell')
                if int(str(shell.cget('height'))) != height:
                    shell.configure(height=height)

    def group(self,title):
        group = ttk.LabelFrame(self.controls,text=title,padding=8)
        group.pack(fill='x',pady=(0,8))
        group.bind('<Configure>',self.schedule_settings_layout)
        return group

    def schedule_settings_layout(self,event=None):
        if self.settings_layout_job is None:
            self.settings_layout_job = self.window.after_idle(self.fit_settings)

    def fit_settings(self):
        self.settings_layout_job = None
        self.align_settings_rows()
        if self.adjustment_open:
            left = self.channel_adjustment.master
            actions_height = self.run_button.master.winfo_reqheight()
            # Preserve room for the settings viewport, even when the rows scroll.
            available = left.winfo_height() - actions_height - self.console.winfo_reqheight() - 165
            self.channel_adjustment.fit_height(available)
        width = self.settings_canvas.winfo_width()
        viewport_height = self.settings_canvas.winfo_height()
        content_height = self.controls.winfo_reqheight()
        # Extend the themed frame below its last group instead of exposing canvas.
        height = max(content_height,viewport_height)
        self.settings_canvas.itemconfigure(self.settings_item,width=width,height=height)
        self.settings_canvas.configure(scrollregion=(0,0,width,height))
        if content_height <= viewport_height:
            self.settings_canvas.yview_moveto(0)

    def align_settings_rows(self):
        # Measure native controls so the common height follows the platform/theme.
        # Containers holding several rows retain their own section spacing.
        sections = {self.definition_panel,self.channel_panel,self.post_panel}
        rows = [widget for parent in self.settings_grid_parents
                for widget in parent.grid_slaves() if widget not in sections]
        rows.extend(widget for widget in self.settings_packed_rows if widget.winfo_manager()=='pack')
        controls = []
        def collect(widget):
            if isinstance(widget,(ttk.Entry,ttk.Combobox,ttk.Spinbox,ttk.Button,ttk.Checkbutton,ttk.Radiobutton)):
                controls.append(widget)
            for child in widget.winfo_children():
                collect(child)
        for widget in rows:
            collect(widget)
        row_gap = 2  # One logical pixel above and below the tallest control.
        height = max((widget.winfo_reqheight() for widget in controls),default=0) + row_gap
        for widget in rows:
            # Extra height is padding, leaving native widgets at their natural size.
            # Multiline captions may exceed the baseline without being clipped.
            space = max(row_gap,height-widget.winfo_reqheight())
            padding = (space//2,space-space//2)
            manager = widget.winfo_manager()
            info = widget.grid_info() if manager=='grid' else widget.pack_info()
            raw_padding = info['pady']
            current = ((raw_padding,) if isinstance(raw_padding,int) else
                       tuple(map(int,self.window.tk.splitlist(raw_padding))))
            if len(current)==1:
                current *= 2
            if current != padding:
                if manager=='grid':
                    widget.grid_configure(pady=padding)
                else:
                    widget.pack_configure(pady=padding)

    def field(self,parent,group,row,name,label,default,kind=str,browse=None):
        var = tk.BooleanVar(value=default) if kind is bool else tk.StringVar(value=str(default))
        self.fields.setdefault(group,{})[name] = (var,kind,label)
        self.defaults.setdefault(group,{})[name] = default
        caption = ttk.Label(parent,text=label,wraplength=250)
        caption.grid(row=row,column=0,sticky='w',padx=(0,8),pady=3)
        holder = ttk.Frame(parent)
        holder.grid(row=row,column=1,sticky='ew',pady=3)
        parent.columnconfigure(1,weight=1)
        if kind is bool:
            control = ttk.Checkbutton(holder,variable=var)
        elif name == 'FIJI_MEMORY':
            validate = self.window.register(lambda value: bool(re.fullmatch(r'\d*(?:\.\d*)?',value)))
            control = ttk.Spinbox(holder,textvariable=var,from_=1/1024,to=1048576,increment=.5,width=8,
                                  validate='key',validatecommand=(validate,'%P'))
        elif name == 'SCAN_DIRECTION':
            control = ttk.Combobox(holder,textvariable=var,values=('Up','Down'),state='readonly',width=8)
        elif kind in (int,float):
            increment = 1 if kind is int or name.endswith('_UM') and default>=1 else .01
            if 'RI_' in name or name.endswith('_RI'):
                increment = .001
            elif name in ('XY_PIXEL_SIZE_UM','Z_STEP_UM'):
                increment = .001
            control = ttk.Spinbox(holder,textvariable=var,width=1,
                                  from_=1 if kind is int else (0 if name=='IMAGING_DEPTH_UM' else 1e-12),
                                  to=1e12,increment=increment,format='%.12g')
        else:
            control = ttk.Entry(holder,textvariable=var,width=1)
        control.pack(side='left',fill='x',expand=True)
        self.field_widgets[name] = control
        if browse:
            def choose():
                value = filedialog.askdirectory(parent=self.window) if browse=='directory' else filedialog.askopenfilename(parent=self.window)
                if value:
                    var.set(value)
            ttk.Button(holder,text='Browse…',command=choose).pack(side='left',padx=(4,0))
        var.trace_add('write',lambda *args:self.field_changed(group,name))
        return caption,holder

    def scroll_field(self,event):
        self.scroll_settings(event)
        return 'break'

    def scroll_settings(self,event):
        target = self.window.winfo_containing(event.x_root,event.y_root)
        while target is not None:
            if target is self.channel_adjustment:
                canvas = self.channel_adjustment.canvas
            elif target is self.controls or target is self.settings_canvas:
                canvas = self.settings_canvas
            else:
                canvas = None
            if canvas is not None:
                units = -1 if getattr(event,'num',None)==4 else 1 if getattr(event,'num',None)==5 else -event.delta
                if abs(units)>=120:
                    units/=120
                canvas.yview_scroll(int(units),'units')
                return 'break'
            target = getattr(target,'master',None)

    def show_source(self):
        self.external.pack_forget()
        self.external_channels.pack_forget()
        self.internal.pack_forget()
        self.optics.pack_forget()
        if self.source.get()=='external':
            self.external.pack(fill='x',pady=5)
            self.external_channels.pack(fill='x',pady=(1,5))
        else:
            self.internal.pack(fill='x')
            self.optics.pack(fill='x',pady=(0,8),before=self.deconvolution)
        self.export_row.pack_forget()
        if self.export_enabled.get():
            self.export_row.pack(fill='x',pady=4,before=self.actual_check)
        for caption,widgets,label in self.pair_widgets:
            caption.configure(text=label+(' (design)' if self.separate.get() else ''))
            for widget in widgets:
                widget.grid() if self.separate.get() else widget.grid_remove()

    def source_changed(self):
        self.show_source()
        self.render_external_channels()
        self.render_channels()
        self.invalidate(psf=True)

    def actual_changed(self):
        self.show_source()
        self.invalidate(psf=True)

    def export_changed(self):
        self.show_source()
        self.save_settings()
        self.update_activity()

    def browse_export(self):
        folder = filedialog.askdirectory(parent=self.window,title='Export generated PSFs to folder')
        if folder:
            self.export_folder.set(folder)

    def field_changed(self,group,name):
        if self.loading:
            return
        if name=='POST_PROCESSING':
            self.render_post_processing()
        if name=='INPUT_DIR':
            if self.refresh_job:
                self.window.after_cancel(self.refresh_job)
            self.refresh_job = self.window.after(300,self.refresh_input)
        self.invalidate(psf=group=='PSF generation' or name=='INPUT_DIR',
                        preview=name not in ('OUTPUT_DIR','OVERWRITE'))

    def auto_spacing_changed(self):
        self.update_auto_spacing()
        self.save_settings()

    def update_auto_spacing(self):
        if self.batch_locked:
            return
        values = self.input_spacing
        changed = False
        # Apply both values before invalidating PSFs; field traces otherwise
        # re-enter activity updates with only one spacing value updated.
        self.loading = True
        try:
            for name,auto in self.auto_spacing_vars.items():
                usable = name in values
                self.auto_spacing_widgets[name].configure(state='normal' if usable else 'disabled')
                automatic = usable and auto.get()
                self.field_widgets[name].configure(state='readonly' if automatic else 'normal')
                if automatic:
                    var = self.fields['PSF generation'][name][0]
                    value = str(values[name])
                    if var.get() != value:
                        var.set(value)
                        changed = True
        finally:
            self.loading = False
        if changed:
            self.invalidate(psf=True)

    def invalidate(self,psf=False,preview=True):
        if self.loading:
            return
        if preview:
            self.revision += 1
            if psf or not self.pending or self.pending['kind']!='view':
                self.pending = None
            if self.process and not self.batch_locked and psf:
                self.stop_worker()
            self.preview.cancel_deconvolution()
            self.preview.parameters_changed()
        if psf:
            self.psf_revision += 1
            self.view_token += 1
            self.view_loading = False
            self.psf_view.invalidate()
            self.schedule_psf_sync()
        self.save_settings()
        self.update_activity()

    def cancel_psf_sync(self):
        if self.psf_sync_job is not None:
            self.window.after_cancel(self.psf_sync_job)
            self.psf_sync_job = None

    def schedule_psf_sync(self):
        self.cancel_psf_sync()
        if self.psf_view.sync.get() and not self.batch_locked:
            self.psf_sync_job = self.window.after(500,self.sync_psf_view)
        self.update_activity()

    def sync_psf_view(self):
        self.psf_sync_job = None
        if self.psf_view.sync.get() and self.psf_view.stale:
            self.request('view',automatic=True)
        self.update_activity()

    def refresh_input(self):
        if self.batch_locked:
            return
        if self.refresh_job:
            self.window.after_cancel(self.refresh_job)
            self.refresh_job = None
        folder = Path(self.fields['Deconvolution']['INPUT_DIR'][0].get()).expanduser()
        self.input_spacing = {}
        # set_folder triggers activity callbacks before channel preferences are rebuilt.
        self.active_channels = []
        self.preview.set_folder(folder)
        names = self.preview.files.cget('values')
        self.stack_status.set(f"{len(names)} Z-stack{'s' if len(names)!=1 else ''} found" if names else 'No Z-stacks detected')
        try:
            if self.preview.selected_path:
                self.active_channels = read_channels(self.preview.selected_path)
                with h5py.File(self.preview.selected_path,'r') as handle:
                    for channel in self.active_channels:
                        section = handle.get(f"DataSetInfo/Channel {channel['index']}")
                        try:
                            color = tuple(float(v) for v in text(section.attrs.get('Color','1 1 1')).split())
                            if len(color)!=3 or not all(math.isfinite(v) for v in color):
                                raise ValueError()
                            channel['color'] = tuple(max(0,min(1,v)) for v in color)
                        except (AttributeError,ValueError):
                            channel['color'] = (1,1,1)
                    image = handle.get('DataSetInfo/Image')
                    for name,axis,dimension in [('XY_PIXEL_SIZE_UM',0,'X'),('Z_STEP_UM',2,'Z')]:
                        try:
                            attrs = image.attrs
                            size = int(text(attrs[dimension]))
                            if size <= 0:
                                raise ValueError('Invalid image dimensions.')
                            spacing = (float(text(attrs[f'ExtMax{axis}'])) -
                                       float(text(attrs[f'ExtMin{axis}']))) / size
                            if not math.isfinite(spacing) or spacing <= 0:
                                raise ValueError('Invalid image voxel spacing.')
                            self.input_spacing[name] = spacing
                        except (AttributeError,KeyError,TypeError,ValueError) as error:
                            self.log(f'Cannot read input {dimension} spacing: {error}\n')
        except (OSError,KeyError,ValueError) as error:
            self.log(f'Cannot read input channels: {error}\n')
        signature = [c['name'] for c in self.active_channels]
        if signature and signature != self.signature:
            self.assignments = {'internal':{},'external':{}}
            self.signature = signature
        self.enabled_vars.clear()
        for channel in self.active_channels:
            key = str(channel['index'])
            self.preferences.setdefault(key,dict(name=channel['name'],color='Original',enabled=True,iterations=40))
            self.definitions.setdefault(key,dict(name=channel['name'],excitation_nm='',emission_nm=''))
            self.preferences[key].setdefault('regularization',self.default_regularization)
            var = tk.BooleanVar(value=self.preferences[key].get('enabled',True))
            var.trace_add('write',lambda *args,key=key:self.selection_changed(key))
            self.enabled_vars[key] = var
        indices = [c['index'] for c in self.active_channels]
        if all('input_index' not in self.definitions[str(i)] for i in indices):
            # Migrate old input-to-PSF dropdown assignments when they form a permutation.
            try:
                old_rows = [int(self.assignments['internal'].get(str(i),f'internal:{i}').split(':')[1]) for i in indices]
                if sorted(old_rows)==sorted(indices):
                    for target,row in zip(indices,old_rows):
                        self.definitions[str(row)]['input_index'] = target
            except (ValueError,IndexError):
                pass
        targets = [self.definitions[str(i)].get('input_index',i) for i in indices]
        if set(targets)!=set(indices) or len(set(targets))!=len(targets):
            targets = indices
        for row,target in zip(indices,targets):
            self.definitions[str(row)]['input_index'] = target
        self.render_definitions()
        self.render_external_channels()
        self.render_channels()
        self.invalidate(psf=True)

    def browse_psfs(self):
        paths = filedialog.askopenfilenames(parent=self.window,title='Select PSF TIFF files',
                                           filetypes=[('TIFF files','*.tif *.tiff *.TIF *.TIFF')])
        if paths:
            self.imported_files = list(paths)
            self.assignments['external'].clear()
            self.refresh_external()

    def refresh_external(self):
        self.import_display.set('; '.join(self.imported_files))
        try:
            self.imported_catalog = read_psf_channels(self.imported_files,validate=True)
            self.import_error = None
        except Exception as error:
            self.imported_catalog = []
            self.import_error = str(error)
        if not self.loading:
            self.render_external_channels()
            self.invalidate(psf=True)

    def selection_changed(self,key):
        enabled = self.enabled_vars[key].get()
        self.preferences[key]['enabled'] = enabled
        self.preview.channel_visibility[int(key)] = enabled
        self.render_definitions()
        self.render_external_channels()
        self.render_channels()
        self.preview.recolor()
        self.psf_view.draw()
        self.invalidate(psf=self.source.get()=='internal')

    def input_labels(self):
        return {c['index']:f"C{c['index']+1}: {c['name']}" for c in self.active_channels}

    def map_definition(self,row,target):
        old = self.definitions[row]['input_index']
        # Swap assignments so each input always has exactly one PSF definition.
        for channel in self.active_channels:
            key = str(channel['index'])
            if key!=row and self.definitions[key]['input_index']==target:
                self.definitions[key]['input_index'] = old
                break
        self.definitions[row]['input_index'] = target
        self.render_definitions()
        self.invalidate(psf=True)

    def render_definitions(self):
        for widget in self.definition_panel.winfo_children():
            widget.destroy()
        self.definition_vars.clear()
        labels = self.input_labels()
        for col,label in enumerate(('Input channel','Channel name','Excitation (nm)','Emission (nm)')):
            ttk.Label(self.definition_panel,text=label).grid(row=0,column=col,sticky='w',padx=(0 if col==0 else 3,3))
            self.definition_panel.columnconfigure(col,weight=1)
        for row,channel in enumerate(self.active_channels,1):
            key = str(channel['index'])
            target = self.definitions[key]['input_index']
            enabled = self.enabled_vars[str(target)]
            cell = ttk.Frame(self.definition_panel)
            cell.grid(row=row,column=0,sticky='ew',padx=(0,3))
            ttk.Checkbutton(cell,variable=enabled).pack(side='left')
            selected = tk.StringVar(value=labels[target])
            dropdown = ttk.Combobox(cell,textvariable=selected,values=tuple(labels.values()),width=10,
                                    state='readonly' if enabled.get() else 'disabled')
            dropdown.pack(side='left',fill='x',expand=True)
            selected.trace_add('write',lambda *args,key=key,var=selected:self.map_definition(
                key,next(index for index,label in labels.items() if label==var.get())))
            variables = [enabled,selected]
            for col,name in enumerate(('name','excitation_nm','emission_nm'),1):
                var = tk.StringVar(value=self.definitions[key][name])
                options = dict(textvariable=var,width=1,state='normal' if enabled.get() else 'disabled')
                control = (ttk.Entry(self.definition_panel,**options) if name=='name' else
                           ttk.Spinbox(self.definition_panel,from_=1e-12,to=1e12,increment=1,
                                       format='%.12g',**options))
                control.grid(row=row,column=col,sticky='ew',padx=3,pady=3)
                def changed(*args,key=key,name=name,var=var):
                    self.definitions[key][name] = var.get()
                    self.invalidate(psf=True)
                var.trace_add('write',changed)
                variables.append(var)
            self.definition_vars[key] = variables

    def catalog(self):
        if self.source.get()=='external':
            return self.imported_catalog
        return [dict(index=self.definitions[str(c['index'])]['input_index'],
                     name=self.definitions[str(c['index'])]['name'],
                     psf_file=f"internal:{self.definitions[str(c['index'])]['input_index']}")
                for c in self.active_channels]

    def selected_assignment(self,index):
        if self.source.get()=='internal':
            return f'internal:{index}'
        return self.assignments['external'].get(str(index),next(
            (c['psf_file'] for c in self.imported_catalog if c['index']==index),None))

    def render_external_channels(self):
        for widget in self.external_channels.winfo_children():
            widget.destroy()
        self.external_vars.clear()
        labels = {c['psf_file']:(f"C{c['index']+1}: {c['name']}" if c['index'] is not None
                               else f"{c['name']} ({c['psf_file']})") for c in self.imported_catalog}
        for col,label in enumerate(('Input channel','PSF channel')):
            ttk.Label(self.external_channels,text=label).grid(row=0,column=col,sticky='w',padx=(0 if col==0 else 3,3))
        self.external_channels.columnconfigure(1,weight=1)
        for row,channel in enumerate(self.active_channels,1):
            key = str(channel['index'])
            enabled = self.enabled_vars[key]
            ttk.Checkbutton(self.external_channels,text=self.input_labels()[channel['index']],variable=enabled).grid(
                row=row,column=0,sticky='w',padx=(0,3),pady=3)
            selected = self.assignments['external'].get(key,next(
                (c['psf_file'] for c in self.imported_catalog if c['index']==channel['index']),None))
            var = tk.StringVar(value=labels.get(selected,''))
            ttk.Combobox(self.external_channels,textvariable=var,values=tuple(labels.values()),width=12,
                         state='readonly' if enabled.get() else 'disabled').grid(row=row,column=1,sticky='ew',padx=3)
            def assigned(*args,key=key,var=var):
                self.assignments['external'][key] = next(path for path,label in labels.items() if label==var.get())
                self.invalidate(psf=True)
            var.trace_add('write',assigned)
            self.external_vars[key] = (enabled,var)

    def render_channels(self):
        for widget in self.channel_panel.winfo_children():
            widget.destroy()
        self.channel_vars.clear()
        for col,label in enumerate(('Input channel','New name','Pseudocolor','Regularization','Iterations')):
            ttk.Label(self.channel_panel,text=label).grid(row=0,column=col,sticky='w',padx=(0 if col==0 else 3,3))
        self.channel_panel.columnconfigure(1,weight=1)
        for row,channel in enumerate(self.active_channels,1):
            key = str(channel['index'])
            pref = self.preferences[key]
            enabled = self.enabled_vars[key]
            name = tk.StringVar(value=pref['name'])
            color = tk.StringVar(value=pref['color'])
            regularization = tk.StringVar(value=pref.get('regularization',self.default_regularization))
            iterations = tk.StringVar(value=pref.get('iterations',40))
            input_cell = ttk.Frame(self.channel_panel)
            input_cell.grid(row=row,column=0,sticky='w',padx=(0,3))
            ttk.Checkbutton(input_cell,variable=enabled).pack(side='left')
            ttk.Label(input_cell,text=self.input_labels()[channel['index']],wraplength=80).pack(side='left')
            widgets = [ttk.Entry(self.channel_panel,textvariable=name,width=5),
                       ttk.Combobox(self.channel_panel,textvariable=color,values=('Original',*NAMED_COLORS),state='readonly',width=8),
                       ttk.Spinbox(self.channel_panel,textvariable=regularization,from_=0,to=1e12,
                                   increment=.0001,format='%.12g',width=9),
                       ttk.Spinbox(self.channel_panel,textvariable=iterations,from_=1,to=100000,width=5)]
            for col,widget in enumerate(widgets,1):
                widget.grid(row=row,column=col,sticky='ew',padx=3,pady=3)
                widget.configure(state=('readonly' if col==2 else 'normal') if enabled.get() else 'disabled')
            def changed(*args,key=key,name=name,color=color,iterations=iterations,regularization=regularization):
                old = dict(self.preferences[key])
                self.preferences[key].update(name=name.get(),color=color.get(),iterations=iterations.get(),regularization=regularization.get())
                processing_changed = any(old.get(k)!=self.preferences[key].get(k) for k in ('iterations','name','regularization'))
                self.preview.recolor()
                self.psf_view.draw()
                self.invalidate(preview=processing_changed)
            for var in (name,color,regularization,iterations):
                var.trace_add('write',changed)
            self.channel_vars[key] = (enabled,name,color,regularization,iterations)
        self.render_post_processing()

    def render_post_processing(self):
        for widget in self.post_panel.winfo_children():
            widget.destroy()
        self.post_vars.clear()
        if not self.fields['Deconvolution']['POST_PROCESSING'][0].get():
            self.post_panel.grid_remove()
            return
        self.post_panel.grid()
        for col,label in enumerate(('Channel','New channel name','Filter','Kernel (voxels)')):
            ttk.Label(self.post_panel,text=label).grid(row=0,column=col,sticky='w',padx=3)
        self.post_panel.columnconfigure(1,weight=1)
        for row,channel in enumerate(self.active_channels,1):
            key = str(channel['index'])
            pref = self.preferences[key]
            name = tk.StringVar(value=pref.get('post_name',pref['name']))
            method = tk.StringVar(value=pref.get('post_filter','None'))
            kernel = tk.StringVar(value=pref.get('post_kernel',3))
            ttk.Label(self.post_panel,text=f"C{channel['index']+1}").grid(row=row,column=0,sticky='w')
            entry = ttk.Entry(self.post_panel,textvariable=name,width=8)
            choice = ttk.Combobox(self.post_panel,textvariable=method,values=('None','Median filter'),state='readonly',width=12)
            size = ttk.Combobox(self.post_panel,textvariable=kernel,values=(3,5,7),state='readonly',width=3)
            enabled = pref.get('enabled',True)
            for col,widget in enumerate((entry,choice,size),1):
                widget.grid(row=row,column=col,sticky='ew',padx=3,pady=3)
                widget.configure(state=('normal' if col==1 else 'readonly') if enabled else 'disabled')
            size.configure(state='readonly' if enabled and method.get()=='Median filter' else 'disabled')
            def changed(*args,key=key,name=name,method=method,kernel=kernel,size=size,enabled=enabled):
                self.preferences[key].update(post_name=name.get(),post_filter=method.get(),post_kernel=kernel.get())
                size.configure(state='readonly' if enabled and method.get()=='Median filter' else 'disabled')
                self.invalidate(preview=True)
            for var in (name,method,kernel):
                var.trace_add('write',changed)
            self.post_vars[key] = (name,method,kernel)

    def read_field(self,group,name):
        var,kind,label = self.fields[group][name]
        try:
            value = kind(var.get())
            if kind is str:
                value = value.strip()
                if not value:
                    raise ValueError()
            elif kind in (int,float):
                if not math.isfinite(value) or value<0 or (value==0 and name not in ('IMAGING_DEPTH_UM','RLTV_REGULARIZATION')):
                    raise ValueError()
            return value
        except (ValueError,tk.TclError):
            raise ValueError(f'Enter a valid value for {label}.') from None

    def generator_settings(self):
        if not self.active_channels:
            raise ValueError('Select a Z-stack with readable channel metadata.')
        hidden = {a for _,a,_,_ in PAIRS} if not self.separate.get() else set()
        settings = {name:self.read_field('PSF generation',name) for name in self.fields['PSF generation'] if name not in hidden}
        if not self.separate.get():
            for design,actual,_,_ in PAIRS:
                settings[actual] = settings[design]
        settings['SCAN_DIRECTION'] = settings['SCAN_DIRECTION'].lower()
        if settings['SCAN_DIRECTION'] not in ('up','down'):
            raise ValueError('Select Up or Down for axial scan direction.')
        settings['CHANNELS'] = []
        for c in self.active_channels:
            definition = self.definitions[str(c['index'])]
            input_index = definition['input_index']
            if not self.preferences[str(input_index)].get('enabled',True):
                continue
            name = str(definition['name']).strip()
            try:
                excitation,emission = (float(definition[k]) for k in ('excitation_nm','emission_nm'))
                if not name or not all(math.isfinite(v) and v>0 for v in (excitation,emission)):
                    raise ValueError()
            except (ValueError,TypeError):
                raise ValueError(f"Enter a channel name and positive wavelengths for C{input_index+1}.") from None
            settings['CHANNELS'].append(dict(index=input_index,name=name,excitation_nm=excitation,emission_nm=emission))
        if not settings['CHANNELS']:
            raise ValueError('Select at least one input channel.')
        settings['CHANNELS'].sort(key=lambda channel:channel['index'])
        return settings

    def enabled_indices(self):
        return [c['index'] for c in self.active_channels if self.preferences[str(c['index'])].get('enabled',True)]

    def can_resolve(self,indices):
        try:
            if not indices:
                return False
            if self.source.get()=='internal':
                self.generator_settings()
            available = {c['psf_file'] for c in self.catalog()}
            return all(self.selected_assignment(i) in available and (self.source.get()=='internal' or Path(self.selected_assignment(i)).is_file()) for i in indices)
        except (ValueError,OSError,tk.TclError):
            return False

    def resolved_assignments(self,catalog,indices):
        by_index = {c['index']:c['psf_file'] for c in catalog}
        available = {c['psf_file'] for c in catalog}
        result = {}
        for index in indices:
            selected = self.selected_assignment(index)
            if self.source.get()=='internal':
                selected = by_index.get(index)
            if selected not in available or not Path(selected).is_file():
                raise ValueError(f'Select an available PSF for C{index+1}.')
            result[index] = selected
        return result

    def deconvolution_settings(self,indices,batch=False,catalog=None,require_psfs=True):
        names = list(self.fields['Deconvolution']) if batch else ['FIJI_MEMORY','RLTV_NON_CIRCULANT','POST_PROCESSING']
        settings = {name:self.read_field('Deconvolution',name) for name in names}
        settings['FIJI_MEMORY'] = memory_argument(settings['FIJI_MEMORY'])
        if not indices:
            raise ValueError('Select at least one input channel.')
        settings.update(CHANNEL_INDICES=list(indices),CHANNEL_ITERATIONS={},CHANNEL_REGULARIZATION={},
                        RLTV_REGULARIZATION=self.default_regularization,OUTPUT_CHANNEL_NAMES={},CHANNEL_POST_PROCESSING={})
        if batch:
            settings.pop('OUTPUT_CHANNEL_NAMES')
            settings['CHANNEL_EDITS'] = {}
        for index in indices:
            pref = self.preferences[str(index)]
            try:
                iterations = int(pref.get('iterations',40))
                if iterations<1:
                    raise ValueError()
            except (ValueError,TypeError):
                raise ValueError(f'Enter positive whole-number iterations for C{index+1}.') from None
            try:
                regularization = float(pref.get('regularization',self.default_regularization))
                if not math.isfinite(regularization) or regularization<0:
                    raise ValueError()
            except (ValueError,TypeError):
                raise ValueError(f'Enter non-negative regularization for C{index+1}.') from None
            settings['CHANNEL_REGULARIZATION'][str(index)] = regularization
            name = pref['name'].strip()
            if settings['POST_PROCESSING']:
                name = pref.get('post_name',name).strip()
                method = pref.get('post_filter','None')
                kernel = int(pref.get('post_kernel',3))
                if method not in ('None','Median filter') or kernel not in (3,5,7):
                    raise ValueError(f'Select a valid filter and kernel for C{index+1}.')
                settings['CHANNEL_POST_PROCESSING'][str(index)] = dict(filter=method,kernel=kernel)
            if not name:
                raise ValueError(f'Enter an output name for C{index+1}.')
            settings['CHANNEL_ITERATIONS'][str(index)] = iterations
            if batch:
                edit = {'name':name}
                if pref['color']!='Original':
                    edit['color'] = pref['color']
                settings['CHANNEL_EDITS'][str(index)] = edit
            else:
                settings['OUTPUT_CHANNEL_NAMES'][str(index)] = name
        if require_psfs:
            if catalog is None:
                catalog = self.imported_catalog if self.source.get()=='external' else self.session.get(self.generator_settings())
            if not catalog:
                raise ValueError('Internal PSFs must be resolved before processing.')
            settings['PSF_FILES'] = list(self.resolved_assignments(catalog,indices).values())
        return settings

    def preview_settings(self):
        return self.deconvolution_settings(self.preview.selected_channels)

    def request(self,kind,automatic=False):
        if kind in ('view','batch'):
            self.cancel_psf_sync()
        if self.batch_locked:
            return
        try:
            if self.refresh_job:
                self.refresh_input()
            indices = self.preview.selected_channels if kind=='preview' else self.enabled_indices()
            if not indices:
                raise ValueError('Select at least one input channel.')
            if not self.can_resolve(indices):
                if self.source.get()=='internal':
                    self.generator_settings()
                raise ValueError((self.import_error if self.source.get()=='external' else None) or 'Select a PSF for each enabled channel.')
            generator = self.generator_settings() if self.source.get()=='internal' else None
            settings = None if kind=='view' else self.deconvolution_settings(indices,batch=kind=='batch',require_psfs=False)
            if kind=='batch' and generator and self.export_enabled.get() and not self.export_folder.get().strip():
                raise ValueError('Select a folder for exporting generated PSFs.')
            request = dict(kind=kind,automatic=automatic,revision=self.revision,preview_revision=self.preview.revision,
                           generator=generator,settings=settings,indices=indices,psf_revision=self.psf_revision)
            if kind=='batch':
                settings['PSF_SOURCE'] = 'internal' if generator else 'external'
                settings['PSF_GENERATION_PARAMETERS'] = generator
                self.preview.cancel_deconvolution()
                self.preview.stop_fiji()
                self.set_batch_locked(True)
            if self.process:
                if not self.cancelled and self.running['request']['generator']==generator and self.running['stage']=='generate':
                    if kind=='batch':
                        self.running['request'] = request
                        self.running['waiters'] = {'batch':request}
                    else:
                        self.running['waiters'][kind] = request
                else:
                    self.pending = request
                    self.stop_worker()
                return
            self.resolve(request)
        except (ValueError,OSError,tk.TclError) as error:
            if kind=='batch':
                self.set_batch_locked(False)
            if not automatic:
                self.error(str(error))
        finally:
            self.update_activity()

    def resolve(self,request):
        generator = request['generator']
        catalog = self.session.get(generator) if generator else self.imported_catalog
        if catalog:
            if generator:
                self.log('Using cached internal PSFs.\n')
            self.continue_request(request,catalog)
        elif generator:
            self.launch(self.session.worker_settings(generator),'generate',request)
        else:
            raise ValueError('Select external PSFs.')

    def current(self,request):
        if request['kind']=='view':
            return request['psf_revision']==self.psf_revision
        return request['kind']=='batch' or (request['revision']==self.revision and
               (request['kind']!='preview' or request['preview_revision']==self.preview.revision))

    def continue_request(self,request,catalog):
        if not self.current(request):
            return
        assignments = self.resolved_assignments(catalog,request['indices'])
        if request['kind']=='preview':
            self.preview.deconvolve_resolved(automatic=request['automatic'])
        elif request['kind']=='view':
            self.view_token += 1
            token = self.view_token
            self.view_loading = True
            self.psf_view.status.set('Loading PSF projections…')
            # Cache projections for disabled but assigned channels too, so visibility
            # changes can redraw without reading or generating PSFs again.
            for channel in self.active_channels:
                try:
                    assignments.update(self.resolved_assignments(catalog,[channel['index']]))
                except ValueError:
                    pass
            generator = request['generator']
            spacing = (generator['Z_STEP_UM'],generator['XY_PIXEL_SIZE_UM'],generator['XY_PIXEL_SIZE_UM']) if generator else None
            colors = {c['index']:c['color'] for c in self.active_channels}
            names = {c['psf_file']:c['name'] for c in catalog}
            def load():
                try:
                    self.events.put(('view',token,load_projections(assignments,spacing,names),colors))
                except Exception as error:
                    self.events.put(('view_error',token,str(error)))
            thread = threading.Thread(target=load,daemon=True)
            self.view_threads = [t for t in self.view_threads if t.is_alive()]
            self.view_threads.append(thread)
            thread.start()
        else:
            if request['generator'] and self.export_enabled.get():
                folder = Path(self.export_folder.get().strip()).expanduser()
                existing = [Path(c['psf_file']).name for c in catalog if (folder/Path(c['psf_file']).name).exists()]
                if existing and not messagebox.askyesno('Replace exported PSFs?', 'Replace these files?\n'+'\n'.join(existing),parent=self.window):
                    self.log('Batch stopped: PSF export declined.\n')
                    self.set_batch_locked(False)
                    return
                self.start_export(request,catalog,assignments,folder)
                return
            settings = dict(request['settings'],PSF_FILES=list(assignments.values()),module='batch_deconvolve_ims')
            self.launch(settings,'batch',request)

    def start_export(self,request,catalog,assignments,folder):
        task = dict(request=request,cancel=threading.Event(),folder=folder,count=len(catalog),
                    settings=dict(request['settings'],PSF_FILES=list(assignments.values()),module='batch_deconvolve_ims'))
        self.export_task = task
        def export():
            error = None
            try:
                folder.mkdir(parents=True,exist_ok=True)
                for entry in catalog:
                    if task['cancel'].is_set():
                        break
                    # Stage each copy so cancelling never leaves a partial target TIFF.
                    with tempfile.NamedTemporaryFile(prefix='.psf-export-',dir=folder,delete=False) as handle:
                        temporary = Path(handle.name)
                    try:
                        shutil.copy2(entry['psf_file'],temporary)
                        if task['cancel'].is_set():
                            break
                        temporary.replace(folder/Path(entry['psf_file']).name)
                    finally:
                        temporary.unlink(missing_ok=True)
            except Exception as exception:
                error = str(exception)
            self.events.put(('exported',task,error))
        threading.Thread(target=export,daemon=True).start()
        self.update_activity()

    def launch(self,settings,stage,request):
        child = subprocess.Popen([sys.executable,'-u',str(ROOT/'main.py'),'--worker'],stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',
                                 errors='replace',cwd=ROOT,**process_session_options())
        self.process = child
        self.running = dict(settings=settings,stage=stage,request=request,waiters={request['kind']:request})
        self.cancelled = False
        self.log('\nGenerating Gibson-Lanni PSFs…\n' if stage=='generate' else '\nStarting batch deconvolution…\n')
        def read():
            import codecs
            decoder = codecs.getincrementaldecoder('utf-8')('replace')
            while chunk := child.stdout.buffer.read1(4096):
                self.events.put(decoder.decode(chunk))
            tail = decoder.decode(b'',final=True)
            if tail:
                self.events.put(tail)
            child.stdout.close()
            self.events.put(('finished',child,child.wait()))
        threading.Thread(target=read,daemon=True).start()
        try:
            child.stdin.write(json.dumps(settings))
            child.stdin.close()
        except OSError:
            self.stop_worker()
            raise
        self.update_activity()

    def stop_worker(self):
        if self.process:
            self.cancelled = True
            child = self.process
            if child.poll() is None:
                stop_process_tree(child.pid)
            self.window.after(2000,lambda:stop_process_tree(child.pid,force=True) if self.process is child and child.poll() is None else None)

    def cancel(self):
        self.cancel_psf_sync()
        self.pending = None
        self.revision += 1
        self.psf_revision += 1
        self.view_token += 1
        if self.view_loading or (self.running and 'view' in self.running['waiters']):
            self.psf_view.invalidate()
        self.view_loading = False
        if self.export_task:
            self.export_task['cancel'].set()
        self.preview.cancel_deconvolution()
        self.stop_worker()
        if not self.process and not self.export_task:
            self.set_batch_locked(False)
        self.log('\nCancelled pending work.\n')
        self.update_activity()

    def set_batch_locked(self,locked):
        self.batch_locked = locked
        if locked:
            def visit(widget):
                for child in widget.winfo_children():
                    if isinstance(child,(ttk.Entry,ttk.Button,ttk.Checkbutton,ttk.Radiobutton,ttk.Combobox,ttk.Spinbox)):
                        self.locked_widgets[child] = str(child.cget('state'))
                        child.configure(state='disabled')
                    visit(child)
            visit(self.controls)
        else:
            for widget,state in self.locked_widgets.items():
                if widget.winfo_exists():
                    widget.configure(state=state)
            self.locked_widgets.clear()
        self.preview.set_disabled(locked)
        self.restore_button.configure(state='disabled' if locked else 'normal')

    def toggle_channel_adjustment(self):
        self.adjustment_open = not self.adjustment_open
        self.update_channel_adjustment()
        self.schedule_settings_layout()

    def update_channel_adjustment(self):
        loaded = self.preview.data is not None or bool(self.psf_view.channels)
        self.adjustment_open = loaded and self.adjustment_open
        label = 'Hide channel adjustment' if self.adjustment_open else 'Show channel adjustment'
        for view in (self.preview, self.psf_view):
            view.adjustment_button.configure(text=label, state='normal' if loaded else 'disabled')
        if not self.adjustment_open:
            self.channel_adjustment.grid_remove()
            return

        rows = []
        for row, kind in enumerate(('Original', 'Deconvolved')):
            contrast = self.preview.channel_contrast[row]
            for channel in sorted(self.preview.display_channels(row, visible_only=False), key=lambda c: c['index']):
                index = channel['index']
                rows.append((kind, index, channel['name'], contrast_maximum(channel), self.preview.channel_visible(index),
                             contrast.get(index, NEUTRAL_CONTRAST)))
        for channel in sorted(self.psf_view.selected(visible_only=False), key=lambda c: c['index']):
            index = channel['index']
            rows.append(('PSF', index, channel['name'], contrast_maximum(channel), self.psf_view.channel_visibility.get(index, True),
                         self.psf_view.channel_contrast.get(index, NEUTRAL_CONTRAST)))
        self.channel_adjustment.grid()
        if self.channel_adjustment.display(tuple(rows)):
            self.schedule_settings_layout()

    def adjust_channel(self, kind, index, *, visible=None, contrast=None):
        if kind == 'PSF':
            visibility = self.psf_view.channel_visibility
            contrasts = self.psf_view.channel_contrast
            redraw = self.psf_view.draw
        else:
            visibility = self.preview.channel_visibility
            contrasts = self.preview.channel_contrast[kind == 'Deconvolved']
            redraw = self.preview.recolor
        if visible is not None:
            visibility[index] = visible
        if contrast is not None:
            if contrasts.get(index, NEUTRAL_CONTRAST) == contrast:
                return
            contrasts[index] = contrast
        redraw()
        self.update_channel_adjustment()

    def update_activity(self):
        if self.loading:
            return
        self.update_auto_spacing()
        self.update_channel_adjustment()
        indices = self.enabled_indices()
        ready = self.can_resolve(indices)
        self.run_button.configure(state='normal' if ready and not self.batch_locked else 'disabled')
        self.psf_view.view_button.configure(state='normal' if ready and not self.batch_locked and not self.view_loading else 'disabled')
        self.psf_view.sync_button.configure(state='disabled' if self.batch_locked else 'normal')
        busy = self.psf_sync_job is not None or self.process is not None or self.export_task is not None or self.view_loading or self.preview.busy=='deconvolve' or self.preview.pending is not None or self.preview.sync_job is not None
        self.cancel_button.configure(state='normal' if busy else 'disabled')
        self.preview.cancel_button.configure(state='normal' if busy else 'disabled')
        self.psf_view.cancel_button.configure(state='normal' if busy else 'disabled')

    def poll(self):
        for _ in range(500):
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            if isinstance(event,str):
                self.append_log(event)
                continue
            kind = event[0]
            if kind=='exported':
                task,error = event[1:]
                if task is not self.export_task:
                    continue
                self.export_task = None
                if task['cancel'].is_set():
                    self.log('PSF export cancelled. Batch stopped.\n')
                    self.set_batch_locked(False)
                elif error:
                    self.set_batch_locked(False)
                    self.error(f'PSF export failed: {error}')
                else:
                    self.log(f"Exported {task['count']} PSFs to {task['folder']}\n")
                    try:
                        self.launch(task['settings'],'batch',task['request'])
                    except OSError as error:
                        self.set_batch_locked(False)
                        self.error(str(error))
                self.update_activity()
                continue
            if kind in ('view','view_error'):
                if event[1]==self.view_token:
                    self.view_loading = False
                    if kind=='view':
                        self.psf_view.display(event[2],event[3])
                    else:
                        self.error(event[2])
                self.update_activity()
                continue
            if kind=='finished' and event[1] is self.process:
                running = self.running
                cancelled = self.cancelled
                self.process = self.running = None
                try:
                    if event[2]==0 and not cancelled:
                        if running['stage']=='generate':
                            catalog = self.session.publish(running['request']['generator'],running['settings']['OUTPUT_DIR'])
                            for request in running['waiters'].values():
                                self.continue_request(request,catalog)
                        else:
                            self.log('Batch finished successfully.\n')
                            self.set_batch_locked(False)
                    else:
                        self.log('Job cancelled.\n' if cancelled else f'Job failed (exit code {event[2]}). See output above.\n')
                        if self.pending is None:
                            self.set_batch_locked(False)
                except Exception as error:
                    self.set_batch_locked(False)
                    self.error(str(error))
                pending,self.pending = self.pending,None
                if pending and self.current(pending):
                    try:
                        self.resolve(pending)
                    except Exception as error:
                        self.set_batch_locked(False)
                        self.error(str(error))
                self.update_activity()
        self.poll_job = self.window.after(100,self.poll)

    def log(self,value):
        self.events.put(value)

    def append_log(self,value):
        self.console.configure(state='normal')
        for index,part in enumerate(value.split('\r')):
            if index:
                self.console.delete('end-1c linestart','end-1c')
            if part:
                self.console.insert('end',part)
        self.console.see('end')
        self.console.configure(state='disabled')

    def error(self,value):
        self.log(f'{value}\n')
        messagebox.showerror('Cannot complete request',str(value).splitlines()[-1],parent=self.window)

    def save_settings(self):
        if self.loading:
            return
        config = configparser.ConfigParser(interpolation=None)
        for group,entries in self.fields.items():
            config[group] = {name:str(var.get()) for name,(var,_,_) in entries.items()}
        config['PSF generation']['separate_design_actual'] = str(self.separate.get())
        for name,var in self.auto_spacing_vars.items():
            config['PSF generation'][name+'_auto'] = str(var.get())
        config['PSF generation']['channel_rows'] = json.dumps(self.definitions)
        config['PSF import'] = dict(enabled=str(self.source.get()=='external'),files=json.dumps(self.imported_files))
        config['PSF export'] = dict(enabled=str(self.export_enabled.get()),folder=self.export_folder.get())
        config['Channel pseudocolor'] = {key:json.dumps(value) for key,value in self.preferences.items()}
        config['PSF assignment'] = dict(input_channels=json.dumps(self.signature),overrides=json.dumps(self.assignments['external']),
                                        internal=json.dumps(self.assignments['internal']))
        try:
            self.settings_path.parent.mkdir(parents=True,exist_ok=True)
            temporary = self.settings_path.with_suffix('.tmp')
            with temporary.open('w',encoding='utf-8') as handle:
                config.write(handle)
            temporary.replace(self.settings_path)
        except OSError as error:
            self.log(f'Cannot save settings: {error}\n')

    def load_settings(self):
        config = configparser.ConfigParser(interpolation=None)
        try:
            config.read(self.settings_path if self.settings_path.exists() else ROOT/'userdata'/'settings.ini',encoding='utf-8')
            self.default_regularization = config.getfloat('Deconvolution','rltv_regularization',fallback=.0001)
            for group,entries in self.fields.items():
                for name,(var,kind,_) in entries.items():
                    if config.has_option(group,name):
                        value = config.getboolean(group,name) if kind is bool else config.get(group,name)
                        if name=='FIJI_MEMORY':
                            value = memory_gb(value)
                        if name=='SCAN_DIRECTION':
                            value = value.capitalize()
                        var.set(value)
            self.source.set('external' if config.getboolean('PSF import','enabled',fallback=False) else 'internal')
            self.imported_files = json.loads(config.get('PSF import','files',fallback='[]'))
            if not isinstance(self.imported_files,list) or not all(isinstance(p,str) for p in self.imported_files):
                self.imported_files = []
                raise ValueError('Invalid imported PSF paths.')
            self.refresh_external()
            self.export_enabled.set(config.getboolean('PSF export','enabled',fallback=False))
            self.export_folder.set(config.get('PSF export','folder',fallback=''))
            self.separate.set(config.getboolean('PSF generation','separate_design_actual',fallback=False))
            for name,var in self.auto_spacing_vars.items():
                var.set(config.getboolean('PSF generation',name+'_auto',fallback=False))
            if config.has_option('PSF generation','channel_rows'):
                definitions = json.loads(config.get('PSF generation','channel_rows'))
            else:
                raw = json.loads(config.get('PSF generation','channels',fallback=json.dumps('\n'.join(f"{c['name']}, {c['excitation_nm']}, {c['emission_nm']}" for c in DEFAULT_CHANNELS))))
                definitions = {}
                for index,line in enumerate(raw.splitlines()):
                    if line.strip():
                        name,ex,em = [v.strip() for v in line.split(',')]
                        definitions[str(index)] = dict(name=name,excitation_nm=ex,emission_nm=em)
            if not isinstance(definitions,dict) or not all(k.isdigit() and isinstance(v,dict) and all(f in v for f in ('name','excitation_nm','emission_nm')) for k,v in definitions.items()):
                raise ValueError('Invalid PSF channel settings.')
            self.definitions = definitions
            if config.has_section('Channel pseudocolor'):
                for key,raw in config.items('Channel pseudocolor'):
                    value = json.loads(raw)
                    if not key.isdigit() or not isinstance(value,dict) or not isinstance(value.get('name'),str) or value.get('color') not in ('Original',*NAMED_COLORS):
                        raise ValueError('Invalid saved channel preferences.')
                    self.preferences[key] = value
            self.signature = json.loads(config.get('PSF assignment','input_channels',fallback='[]'))
            for source,option in [('external','overrides'),('internal','internal')]:
                values = json.loads(config.get('PSF assignment',option,fallback='{}'))
                if isinstance(values,dict):
                    self.assignments[source] = {k:(v if source=='internal' else str(Path(v).expanduser().resolve()))
                                                for k,v in values.items() if k.isdigit() and isinstance(v,str)
                                                and (v.startswith('internal:') if source=='internal' else Path(v).is_absolute())}
        except (OSError,ValueError,TypeError,configparser.Error) as error:
            self.log(f'Could not load all saved settings: {error}\n')

    def restore_defaults(self):
        if self.batch_locked:
            return
        self.cancel()
        self.loading = True
        for var in self.auto_spacing_vars.values():
            var.set(False)
        for group,entries in self.defaults.items():
            for name,value in entries.items():
                self.fields[group][name][0].set(value)
        self.preferences.clear()
        self.default_regularization = .0001
        self.definitions = {str(i):copy.deepcopy(c) for i,c in enumerate(DEFAULT_CHANNELS)}
        self.assignments = {'internal':{},'external':{}}
        self.imported_files = []
        self.refresh_external()
        self.source.set('internal')
        self.separate.set(False)
        self.export_enabled.set(False)
        self.export_folder.set('')
        self.loading = False
        self.show_source()
        self.refresh_input()

    def close(self):
        if self.process or self.export_task:
            messagebox.showinfo('Job running','Cancel the current job or wait for it to finish before closing.',parent=self.window)
            return
        self.cancel_psf_sync()
        self.preview.close()
        self.psf_view.close()
        self.view_token += 1
        self.window.after_cancel(self.poll_job)
        def finish():
            if any(thread.is_alive() for thread in self.view_threads):
                self.window.after(50,finish)
                return
            self.session.close()
            self.window.destroy()
        finish()
