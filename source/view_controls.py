"""Shared preview geometry and logarithmic zoom controls."""
import math
from pathlib import Path
import tkinter as tk
from tkinter import ttk, font
from source.ui_style import is_dark

MIN_ZOOM = 100
MAX_ZOOM = 6400
ZOOM_STEPS = math.log2(MAX_ZOOM / MIN_ZOOM)
PANE_GAP = 4


def zoom_position(percent):
    return ZOOM_STEPS - math.log2(min(MAX_ZOOM, max(MIN_ZOOM, percent)) / MIN_ZOOM)


def zoom_percent(position):
    return MIN_ZOOM * 2 ** (ZOOM_STEPS - min(ZOOM_STEPS, max(0, float(position))))


def pane_grid(width, height):
    side = max(1, (min(width, height) - PANE_GAP) // 2)
    extent = 2 * side + PANE_GAP
    return (width - extent) // 2, (height - extent) // 2, side, PANE_GAP


def plane_axes(axis):
    return ('x', 'y') if axis == 'xy' else ('y', 'z')


class ZoomBar(ttk.Frame):
    def __init__(self, parent, zoom, changed, reset):
        super().__init__(parent)
        self.zoom = zoom
        self.position = tk.DoubleVar(value=zoom_position(zoom.get()))
        self.label = tk.StringVar()
        assets = Path(__file__).with_name('assets')
        self.icons = {name:tk.PhotoImage(master=self, file=str(assets/f'{name}.png'))
                      for name in ('zoom-in','zoom-out')}
        ttk.Label(self,image=self.icons['zoom-in'],anchor='center').pack()
        self.slider = ttk.Scale(self,from_=0,to=ZOOM_STEPS,orient='vertical',variable=self.position,
                                command=lambda value:self.slide(value,changed))
        self.slider.pack(fill='y',expand=True,pady=4)
        ttk.Label(self,image=self.icons['zoom-out'],anchor='center').pack()
        size = self.icons['zoom-in'].width()
        reset_holder = ttk.Frame(self,width=size,height=size)
        reset_holder.pack(pady=(4,0))
        reset_holder.pack_propagate(False)
        style = ttk.Style(self)
        style.configure('ZoomReset.TButton',padding=0,anchor='center',font=('Arial',14))
        # Overlay the label on the native button instead of nesting it inside
        # Aqua's content insets, which clip the glyph at this compact size.
        style.layout('ZoomReset.TButton',[
            ('Button.button',{'sticky':'nswe'}),
            ('Button.label',{'sticky':'nswe'}),
        ])
        self.reset_button = ttk.Button(reset_holder,text='↺',style='ZoomReset.TButton',
                                       width=1,command=reset,takefocus=True)
        self.reset_button.place(x=0,y=0,width=size,height=size)
        # Reserve the maximum readout width so zooming never moves the panes.
        readout_font = font.nametofont('TkDefaultFont')
        readout = ttk.Frame(self,width=readout_font.measure('6400%')+2,height=readout_font.metrics('linespace')+2)
        readout.pack(pady=(4,0))
        readout.pack_propagate(False)
        ttk.Label(readout,textvariable=self.label,anchor='center').pack(fill='both',expand=True)
        def theme(event=None):
            dark = is_dark(self)
            suffix = '-light' if dark else ''
            for name,icon in self.icons.items():
                icon.configure(file=str(assets/f'{name}{suffix}.png'))
        self.bind('<Map>',theme)
        theme()
        zoom.trace_add('write',self.sync)
        self.sync()

    def sync(self,*args):
        self.position.set(zoom_position(self.zoom.get()))
        self.label.set(f'{self.zoom.get():.0f}%')

    def slide(self,value,changed):
        self.zoom.set(zoom_percent(value))
        changed()
