"""Colored XY/YZ PSF projections in two portrait viewports on one canvas."""
import json
import math
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog

import numpy as np
import tifffile
from PIL import Image, ImageTk
from PIL.PngImagePlugin import PngInfo

from source.channel_adjustment import NEUTRAL_CONTRAST, apply_contrast
from source.edit_ims_channels import NAMED_COLORS
from source.psf_channels import load_psf_volume
from source.preview_gestures import install_pinch
from source.view_controls import MIN_ZOOM, MAX_ZOOM, ZoomBar, pane_grid
from source.ui_style import workspace_background


def read_spacing(path, fallback=None):
    if fallback is not None:
        return tuple(fallback), False
    try:
        with tifffile.TiffFile(path) as tif:
            metadata = tif.imagej_metadata or {}
            unit = str(metadata.get('unit', '')).lower().replace('μ', 'µ')
            factor = {'um': 1, 'µm': 1, 'micron': 1, 'nm': .001, 'mm': 1000}.get(unit)
            if factor is None:
                raise ValueError()
            page = tif.pages[0]
            x = page.tags['XResolution'].value
            y = page.tags['YResolution'].value
            spacing = (float(metadata['spacing']) * factor, y[1] / y[0] * factor,
                       x[1] / x[0] * factor)
            if not all(math.isfinite(v) and v > 0 for v in spacing):
                raise ValueError()
            return spacing, False
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return (1., 1., 1.), True


def load_projections(assignments, spacing=None, names=None):
    """Read off the UI thread; return display data without retaining 3D volumes."""
    result = []
    loaded = {}
    for index, path in assignments.items():
        if path not in loaded:
            volume = load_psf_volume(path)
            voxel, pixels = read_spacing(path, spacing)
            peak = float(volume.max())
            loaded[path] = dict(xy=volume.max(axis=0) / peak,
                                yz=volume.max(axis=2) / peak,
                                spacing=voxel, pixels=pixels)
        result.append(dict(loaded[path], index=index, path=path,
                           name=(names or {}).get(path, Path(path).stem)))
    return result


class PSFView(ttk.Frame):
    def __init__(self, parent, preferences, request, log, error):
        super().__init__(parent, padding=8)
        self.preferences, self.request, self.log, self.error = preferences, request, log, error
        self.channels = []
        self.channel_visibility = {}
        self.channel_contrast = {}
        self.original_colors = {}
        self.stale = True
        self.zoom = tk.DoubleVar(value=100)
        self.zoom_label = tk.StringVar(value='100%')
        self.status = tk.StringVar(value='Select View PSF to display the current PSFs.')
        self.center = {'x': 0., 'y': 0., 'z': 0.}
        self.drag = None
        self.photo = None
        self.toolbar_shell = ttk.Frame(self)
        self.toolbar_shell.pack(fill='x',pady=(0,8))
        self.toolbar = ttk.Frame(self.toolbar_shell)
        self.toolbar.pack(fill='both',expand=True)
        self.actions_shell = ttk.Frame(self)
        self.actions_shell.pack(side='bottom',fill='x',pady=(8,0))
        self.actions = ttk.Frame(self.actions_shell)
        self.actions.pack(fill='both',expand=True)
        self.sync = tk.BooleanVar(value=False)
        self.view_button = ttk.Button(self.actions,text='View PSF',command=request)
        self.view_button.grid(row=0,column=0,sticky='w')
        self.cancel_button = ttk.Button(self.actions,text='Cancel',state='disabled')
        self.cancel_button.grid(row=0,column=1,sticky='w',padx=(5,0))
        self.sync_button = ttk.Checkbutton(self.actions,text='Sync PSF view',variable=self.sync)
        self.sync_button.grid(row=0,column=2,sticky='w',padx=(8,4))
        self.adjustment_button = ttk.Button(self.actions, width=24, text='Show channel adjustment', state='disabled')
        self.adjustment_button.grid(row=0,column=3,sticky='w',padx=(12,4))
        self.export_button = ttk.Button(self.actions,text='Export screenshot…',command=self.export,state='disabled')
        self.export_button.grid(row=0,column=4,sticky='e')
        self.actions.columnconfigure(3,weight=1)
        self.actions_height = 1
        self.actions.bind('<Configure>',self.layout_actions)
        area = ttk.Frame(self)
        area.pack(fill='both',expand=True)
        self.zoom_bar = ZoomBar(area,self.zoom,self.draw,self.reset)
        self.zoom_bar.pack(side='right',fill='y',padx=(6,0))
        self.slider = self.zoom_bar.slider
        self.zoom_label = self.zoom_bar.label
        self.views = ttk.Frame(area)
        self.views.pack(side='left',fill='both',expand=True)
        self.views.bind('<Configure>',self.layout_views)
        self.canvas = tk.Canvas(self.views,background=workspace_background(self),highlightthickness=0,borderwidth=0,width=1,height=1)
        self.canvas.place(x=0,y=0,width=1,height=1)
        self.canvas.bind('<Configure>', lambda event:self.draw())
        for sequence in ('<MouseWheel>', '<Button-4>', '<Button-5>'):
            self.canvas.bind(sequence, self.wheel)
        self.canvas.bind('<ButtonPress-1>', self.start_pan)
        self.canvas.bind('<B1-Motion>', self.pan)
        self.canvas.bind('<ButtonRelease-1>', lambda event:setattr(self, 'drag', None))
        self.remove_pinch = install_pinch(self, self.pinch)

    def layout_actions(self,event=None):
        buttons = (self.view_button,self.cancel_button,self.sync_button,
                   self.adjustment_button,self.export_button)
        compact = self.master.winfo_width()-24 < sum(b.winfo_reqwidth() for b in buttons)+33
        self.adjustment_button.grid_configure(row=1 if compact else 0,
                                             column=0 if compact else 3,
                                             columnspan=5 if compact else 1,
                                             padx=(0,4) if compact else (12,4),
                                             pady=(5,0) if compact else 0)
        height = max(b.winfo_reqheight() for b in buttons if b is not self.adjustment_button)
        self.actions_height = (height+5+self.adjustment_button.winfo_reqheight() if compact
                               else max(height,self.adjustment_button.winfo_reqheight()))

    def layout_views(self,event=None):
        # Leave unused space to the native themed frame, rather than painting it
        # with the canvas's flat background color.
        left,top,side,gap = pane_grid(self.views.winfo_width(),self.views.winfo_height())
        extent = 2*side+gap
        self.canvas.place_configure(x=left,y=top,width=extent,height=extent)

    def invalidate(self):
        self.stale = True
        self.status.set('PSF settings changed. Select View PSF to update.')
        self.export_button.configure(state='disabled')

    def display(self, channels, colors):
        previous = {c['index']: (c['path'], c['name']) for c in self.channels}
        retained = {c['index'] for c in channels if previous.get(c['index']) == (c['path'], c['name'])}
        for settings in (self.channel_visibility, self.channel_contrast):
            for index in settings.keys() - retained:
                del settings[index]
        self.channels = channels
        self.original_colors = colors
        self.stale = False
        self.status.set('Unit pixel spacing used for PSFs without voxel metadata.'
                        if any(c['pixels'] for c in channels) else '')
        self.reset()

    def selected(self, visible_only=True):
        return [c for c in self.channels
                if self.preferences.get(str(c['index']), {}).get('enabled', True)
                and (not visible_only or self.channel_visibility.get(c['index'], True))]

    def viewports(self, width, height):
        left, top, side, gap = pane_grid(width,height)
        return {'xy': (left,top,side,2*side+gap), 'yz': (left+side+gap,top,side,2*side+gap)}

    def scale(self, rects):
        extents = []
        for channel in self.selected(visible_only=False):
            dz, dy, dx = channel['spacing']
            for axis, pixel_size in (('xy', (dx, dy)), ('yz', (dy, dz))):
                h, w = channel[axis].shape
                _, _, vw, vh = rects[axis]
                extents.append(min(vw / (w * pixel_size[0]), vh / (h * pixel_size[1])))
        return min(extents, default=1.) * self.zoom.get() / 100

    def render(self, width, height):
        rects = self.viewports(width, height)
        scale = self.scale(rects)
        result = Image.new('RGB', (width, height), self.canvas.cget('background'))
        for axis, (left, top, w, h) in rects.items():
            # Match the deconvolution preview's unused image area.
            result.paste('#242424', (left, top, left+w, top+h))
            overlay = np.zeros((h, w, 3), dtype=np.float32)
            coverage = np.zeros((h, w), dtype=bool)
            for channel in self.selected():
                dz, dy, dx = channel['spacing']
                sx, sy = (dx, dy) if axis == 'xy' else (dy, dz)
                horizontal, vertical = ('x', 'y') if axis == 'xy' else ('y', 'z')
                contrast = self.channel_contrast.get(channel['index'], NEUTRAL_CONTRAST)
                array = apply_contrast(channel[axis], contrast)
                ih, iw = array.shape
                transform = (1 / (scale*sx), 0, iw/2 + self.center[horizontal]/sx - w/(2*scale*sx),
                             0, 1 / (scale*sy), ih/2 + self.center[vertical]/sy - h/(2*scale*sy))
                plane = Image.fromarray(array).transform((w, h), Image.Transform.AFFINE, transform,
                                                        Image.Resampling.BILINEAR)
                # Track image bounds independently of intensity so dark PSF
                # pixels stay black while unused viewport space stays dark gray.
                bounds = Image.new('L', (iw, ih), 255).transform(
                    (w, h), Image.Transform.AFFINE, transform, Image.Resampling.NEAREST)
                coverage |= np.asarray(bounds) > 0
                color_name = self.preferences.get(str(channel['index']), {}).get('color', 'Original')
                color = NAMED_COLORS.get(color_name, self.original_colors.get(channel['index'], (1,1,1)))
                overlay += np.asarray(plane)[..., None] * np.asarray(color)
            image = Image.fromarray(np.uint8(np.clip(overlay, 0, 1) * 255))
            result.paste(image, (left, top), Image.fromarray(coverage))
        return result

    def draw(self):
        self.export_button.configure(state='normal' if not self.stale and self.selected() else 'disabled')
        self.zoom_label.set(f'{self.zoom.get():.0f}%')
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        if width <= 1 or height <= 1:
            return
        image = self.render(width, height)
        self.photo = ImageTk.PhotoImage(image, master=self)
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, anchor='nw', image=self.photo)

    def reset(self):
        self.zoom.set(100)
        self.center = dict.fromkeys('xyz', 0.)
        self.draw()

    def set_zoom(self, value, x, y):
        rects = self.viewports(self.canvas.winfo_width(), self.canvas.winfo_height())
        old = self.scale(rects)
        self.zoom.set(min(MAX_ZOOM, max(MIN_ZOOM, value)))
        new = self.scale(rects)
        for axis, (left, top, w, h) in rects.items():
            if left <= x <= left+w and top <= y <= top+h:
                a, b = ('x', 'y') if axis == 'xy' else ('y', 'z')
                self.center[a] += (x-left-w/2) * (1/old-1/new)
                self.center[b] += (y-top-h/2) * (1/old-1/new)
                break
        self.draw()

    def wheel(self, event):
        steps = 1 if getattr(event,'num',None)==4 else -1 if getattr(event,'num',None)==5 else event.delta
        if abs(steps) >= 120:
            steps /= 120
        self.set_zoom(self.zoom.get() * 1.1**max(-10,min(10,steps)), event.x, event.y)
        return 'break'

    def pinch(self, target, x, y, amount):
        if target is not self.canvas:
            return False
        self.set_zoom(self.zoom.get()*max(.01,1+amount), x-self.canvas.winfo_rootx(), y-self.canvas.winfo_rooty())
        return True

    def start_pan(self, event):
        for axis, (left, top, w, h) in self.viewports(self.canvas.winfo_width(), self.canvas.winfo_height()).items():
            if left <= event.x <= left+w and top <= event.y <= top+h:
                self.drag = (axis, event.x, event.y, dict(self.center))
                break

    def pan(self, event):
        if self.drag:
            axis, x, y, center = self.drag
            a, b = ('x','y') if axis == 'xy' else ('y','z')
            scale = self.scale(self.viewports(self.canvas.winfo_width(), self.canvas.winfo_height()))
            self.center[a] = center[a]-(event.x-x)/scale
            self.center[b] = center[b]-(event.y-y)/scale
            self.draw()

    def export(self):
        if self.stale or not self.selected():
            return
        path = filedialog.asksaveasfilename(parent=self.winfo_toplevel(), title='Export PSF screenshot',
                                           initialfile='psf_projections.png', defaultextension='.png',
                                           filetypes=[('PNG image','*.png')])
        if not path:
            return
        try:
            image = self.render(self.canvas.winfo_width(), self.canvas.winfo_height())
            info = PngInfo()
            info.add_text('PSF', json.dumps(dict(zoom=self.zoom.get(), center=self.center,
                          channels=[dict(index=c['index'], color=self.preferences.get(str(c['index']),{}).get('color','Original'),
                                         contrast=self.channel_contrast.get(c['index'], NEUTRAL_CONTRAST),
                                         spacing=c['spacing']) for c in self.selected()])))
            image.save(path, pnginfo=info)
            self.log(f'Saved screenshot: {path}\n')
        except (OSError, ValueError) as error:
            self.error(str(error))

    def close(self):
        self.remove_pinch()
