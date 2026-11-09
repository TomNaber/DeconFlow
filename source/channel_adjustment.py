"""Display-only contrast and controls shared by the preview tabs."""
import math
import tkinter as tk
from tkinter import ttk

from source.view_controls import ResetButton

import numpy as np


NEUTRAL_CONTRAST = (0.0, 0.5, 1.0)


def format_intensity(value):
    """Compact half-unit readouts; derived midpoints retain exact gamma internally."""
    return f'{round(value * 2) / 2:.1f}'.removesuffix('.0')


def contrast_range(channel):
    """Return the same range used to normalize the rendered channel."""
    return tuple(channel.get('range', (0.0, 1.0)))


def apply_contrast(intensity, contrast):
    if contrast == NEUTRAL_CONTRAST:
        return intensity
    black, midpoint, white = contrast
    scaled = np.clip((intensity - black) / (white - black), 0, 1)
    gamma = np.log(0.5) / np.log((midpoint - black) / (white - black))
    return np.power(scaled, gamma)


class ContrastSlider(tk.Canvas):
    """Three draggable handles in normalized display coordinates."""
    def __init__(self, parent, command, limits):
        super().__init__(parent, width=120, height=24, highlightthickness=0,
                         takefocus=True, background=ttk.Style(parent).lookup('TFrame', 'background') or '#eeeeee')
        self.command = command
        self.low, self.high = limits
        self.increment = 0.5 / (self.high - self.low)
        self.values = NEUTRAL_CONTRAST
        self.active = 1
        self.bind('<Configure>', self.draw)
        self.bind('<Button-1>', self.press)
        self.bind('<B1-Motion>', self.drag)
        self.bind('<Left>', lambda event: self.step(-self.increment))
        self.bind('<Right>', lambda event: self.step(self.increment))
        self.bind('<Up>', lambda event: self.select(-1))
        self.bind('<Down>', lambda event: self.select(1))

    def draw(self, event=None):
        self.delete('all')
        width = max(1, self.winfo_width() - 20)
        self.create_line(10, 12, 10 + width, 12, fill='#888888', width=3)
        for index, (value, color) in enumerate(zip(self.values, ('#222222', '#999999', '#ffffff'))):
            x = 10 + value * width
            self.create_oval(x - 6, 6, x + 6, 18,
                                fill=color, outline='#3984d6' if index == self.active else '#666666',
                                width=2)

    def set(self, values):
        self.values = values
        self.draw()

    def move(self, value):
        black, midpoint, white = self.values
        fraction = (midpoint - black) / (white - black)
        gap = 1e-6
        lower, upper = ((0.0, white - gap) if self.active == 0 else
                        (black + gap, 1.0) if self.active == 2 else
                        (black + (white - black) * gap, white - (white - black) * gap))
        span = self.high - self.low
        first_bin = math.ceil((self.low + lower * span) * 2)
        last_bin = math.floor((self.low + upper * span) * 2)
        if first_bin > last_bin:
            self.command(self.values)
            return
        native_bin = min(last_bin, max(first_bin, round((self.low + value * span) * 2)))
        value = (native_bin / 2 - self.low) / span
        if self.active == 0:
            black = min(white - gap, max(0.0, value))
            midpoint = black + fraction * (white - black)
        elif self.active == 2:
            white = max(black + gap, min(1.0, value))
            midpoint = black + fraction * (white - black)
        else:
            margin = (white - black) * gap
            midpoint = min(white - margin, max(black + margin, value))
        values = (black, midpoint, white)
        if values != self.values:
            self.command(values)
        else:
            self.draw()

    def press(self, event):
        self.focus_set()
        width = max(1, self.winfo_width() - 20)
        position = (event.x - 10) / width
        self.active = min(range(3), key=lambda i: abs(self.values[i] - position))
        # Grabbing a handle should not shift it before the pointer moves.
        self.drag_offset = self.values[self.active] - position
        if abs(self.drag_offset * width) > 8:
            self.drag_offset = 0.0
            self.move(position)
        self.draw()

    def drag(self, event):
        self.move((event.x - 10) / max(1, self.winfo_width() - 20) + self.drag_offset)

    def step(self, delta):
        self.move(self.values[self.active] + delta)
        return 'break'

    def select(self, delta):
        self.active = (self.active + delta) % 3
        self.draw()
        return 'break'


class ChannelAdjustment(ttk.Notebook):
    def __init__(self, parent, changed):
        super().__init__(parent)
        page = ttk.Frame(self, padding=8)
        self.add(page, text='Channel adjustment')
        self.changed = changed
        self.rows = ()
        self.canvas = tk.Canvas(page, width=1, height=1, highlightthickness=0)
        self.canvas.pack(side='left', fill='both', expand=True)
        self.scrollbar = ttk.Scrollbar(page, orient='vertical', command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.table = ttk.Frame(self.canvas)
        self.item = self.canvas.create_window(0, 0, window=self.table, anchor='nw')
        self.canvas.bind('<Configure>', self.fit_table)
        self.table.bind('<Configure>', self.fit_table)

    def fit_table(self, event=None):
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        content = self.table.winfo_reqheight()
        self.canvas.itemconfigure(self.item, width=width, height=max(content, height))
        self.canvas.configure(scrollregion=(0, 0, width, max(content, height)))

    def fit_height(self, maximum):
        content = self.table.winfo_reqheight()
        if len(self.rows) > 9:
            _, y, _, row_height = self.table.grid_bbox(0, 9, 5, 9)
            content_limit = y + row_height
        else:
            content_limit = content
        height = max(1, min(maximum, content_limit))
        if int(self.canvas.cget('height')) != height:
            self.canvas.configure(height=height)
        if content > height:
            if not self.scrollbar.winfo_manager():
                self.scrollbar.pack(side='right', fill='y', before=self.canvas)
        elif self.scrollbar.winfo_manager():
            self.scrollbar.pack_forget()
            self.canvas.yview_moveto(0)
        self.fit_table()

    def display(self, rows):
        if self.rows == rows:
            return False
        structure_changed = tuple(row[:4] for row in self.rows) != tuple(row[:4] for row in rows)
        self.rows = rows
        if structure_changed:
            for widget in self.table.winfo_children():
                widget.destroy()
            ttk.Label(self.table, text='Channel').grid(row=0, column=0, sticky='w')
            ttk.Label(self.table, text='Contrast').grid(row=0, column=1, columnspan=4)
            self.table.columnconfigure(2, weight=1)
            self.controls = [self.add_row(i, *row[:4]) for i, row in enumerate(rows, 1)]
        for (checkbox, slider, entries), (_, _, _, limits, visible, contrast) in zip(self.controls, rows):
            checkbox.set(visible)
            slider.set(contrast)
            low, high = limits
            for entry, value in zip(entries, contrast):
                entry.set(format_intensity(low + value * (high - low)))
        return structure_changed

    def add_row(self, row, kind, index, name, limits):
        visible = tk.BooleanVar(self)
        ttk.Checkbutton(
            self.table, text=f'C{index + 1} ({kind}): {name}', variable=visible,
            command=lambda: self.changed(kind, index, visible=visible.get()),
        ).grid(row=row, column=0, sticky='w', padx=(0, 8))
        low, high = limits
        entries = [tk.StringVar(self) for _ in range(3)]

        def set_value(contrast):
            slider.set(contrast)
            for entry, value in zip(entries, contrast):
                entry.set(format_intensity(low + value * (high - low)))
            self.changed(kind, index, contrast=contrast)

        slider = ContrastSlider(self.table, set_value, limits)
        slider.grid(row=row, column=2, sticky='ew', padx=4)

        def commit(index):
            current = format_intensity(low + slider.values[index] * (high - low))
            if entries[index].get() == current:
                return
            try:
                value = (float(entries[index].get()) - low) / (high - low)
                if not np.isfinite(value):
                    raise ValueError()
            except ValueError:
                entries[index].set(format_intensity(low + slider.values[index] * (high - low)))
                return
            slider.active = index
            slider.move(value)
            entries[index].set(format_intensity(low + slider.values[index] * (high - low)))

        for point_index, column in ((0, 1), (2, 3), (1, 4)):
            entry = ttk.Entry(self.table, textvariable=entries[point_index], width=6, justify='right')
            entry.grid(row=row, column=column, padx=2, pady=1)
            entry.bind('<Return>', lambda event, i=point_index: commit(i))
            entry.bind('<FocusOut>', lambda event, i=point_index: commit(i))
        ResetButton(self.table, lambda: set_value(NEUTRAL_CONTRAST)).grid(
            row=row, column=5, padx=(4, 0))
        return visible, slider, entries
