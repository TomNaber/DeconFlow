"""Shared preview geometry and logarithmic zoom controls."""
import math
from functools import lru_cache
from pathlib import Path
import tkinter as tk
from tkinter import ttk, font
from PIL import ImageDraw, ImageFont
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


def scale_bar_geometry(pixels_per_um, width):
    """Return a rounded physical length and its screen length, or None."""
    if not math.isfinite(pixels_per_um) or pixels_per_um <= 0 or width <= 0:
        return None
    limit = min(120., width * .25)
    maximum = limit / pixels_per_um
    if not math.isfinite(maximum) or maximum <= 0:
        return None
    decade = 10. ** math.floor(math.log10(maximum))
    if decade == 0:
        return None
    # Include the preceding decade to handle rounding at powers of ten.
    candidates = [factor * decade for factor in (.5, 1, 2, 5, 10)]
    value = max(value for value in candidates if value <= maximum)
    return value, value * pixels_per_um


@lru_cache(maxsize=1)
def _scale_bar_font():
    # Pillow's scalable default font lacks the micro sign. Use a system font
    # on macOS/Windows, with a Latin-1 bitmap fallback that includes µ.
    for name in ('Arial.ttf', 'arial.ttf', 'DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(name, 13)
        except OSError:
            pass
    return ImageFont.load_default_imagefont()


def draw_scale_bar(image, pixels_per_um, viewport=None):
    """Annotate a rendered pane in place, so screen and PNG use the same bar."""
    left, top, width, height = viewport or (0, 0, *image.size)
    geometry = scale_bar_geometry(pixels_per_um, width)
    if geometry is None:
        return
    value, length = geometry
    label = f'{value:g} µm'
    draw = ImageDraw.Draw(image)
    label_font = _scale_bar_font()
    bounds = draw.textbbox((0, 0), label, font=label_font, stroke_width=1)
    label_width, label_height = bounds[2] - bounds[0], bounds[3] - bounds[1]
    # Leave room for the label even when it is wider than the bar.
    annotation_width = max(length + 4, label_width)
    if width < annotation_width + 24 or height < label_height + 38:
        return
    center = left + width - 12 - annotation_width / 2
    y = top + height - 16
    x0, x1 = center - length / 2, center + length / 2
    segments = [(x0, y, x1, y), (x0, y - 3, x0, y + 3), (x1, y - 3, x1, y + 3)]
    for color, thickness in (('black', 4), ('white', 2)):
        for segment in segments:
            draw.line(segment, fill=color, width=thickness)
    draw.text((center - label_width / 2 - bounds[0], y - 8 - label_height - bounds[1]),
              label, font=label_font, fill='white', stroke_width=1, stroke_fill='black')


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
