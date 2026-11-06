"""Display-only contrast and controls shared by the preview tabs."""
import tkinter as tk
from tkinter import ttk

import numpy as np


NEUTRAL_CONTRAST = 128


def apply_contrast(intensity, contrast):
    if contrast == NEUTRAL_CONTRAST:
        return intensity
    return np.power(intensity, 2 ** ((NEUTRAL_CONTRAST - contrast) / 64))


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
        structure_changed = tuple(row[:3] for row in self.rows) != tuple(row[:3] for row in rows)
        self.rows = rows
        if structure_changed:
            for widget in self.table.winfo_children():
                widget.destroy()
            ttk.Label(self.table, text='Channel').grid(row=0, column=0, sticky='w')
            ttk.Label(self.table, text='Contrast').grid(row=0, column=1, columnspan=4)
            self.table.columnconfigure(2, weight=1)
            self.controls = [self.add_row(i, *row[:3]) for i, row in enumerate(rows, 1)]
        for (checkbox, slider, entry), (_, _, _, visible, contrast) in zip(self.controls, rows):
            checkbox.set(visible)
            if slider.get() != contrast:
                slider.set(contrast)
                entry.set(str(contrast))
        return structure_changed

    def add_row(self, row, kind, index, name):
        visible = tk.BooleanVar(self)
        ttk.Checkbutton(
            self.table, text=f'C{index + 1} ({kind}): {name}', variable=visible,
            command=lambda: self.changed(kind, index, visible=visible.get()),
        ).grid(row=row, column=0, sticky='w', padx=(0, 8))
        ttk.Label(self.table, text='0').grid(row=row, column=1)
        value = tk.DoubleVar(self, value=NEUTRAL_CONTRAST)
        entry_value = tk.StringVar(self, value=str(NEUTRAL_CONTRAST))

        def set_value(contrast):
            value.set(contrast)
            entry_value.set(str(contrast))
            self.changed(kind, index, contrast=contrast)

        ttk.Scale(self.table, from_=0, to=255, variable=value,
                  command=lambda raw: set_value(round(float(raw)))).grid(
                      row=row, column=2, sticky='ew', padx=4)
        ttk.Label(self.table, text='255').grid(row=row, column=3)
        entry = ttk.Spinbox(self.table, textvariable=entry_value, from_=0, to=255, increment=1, width=4)
        entry.grid(row=row, column=4, padx=(6, 3), pady=1)

        def commit(event=None):
            try:
                contrast = int(entry_value.get())
            except ValueError:
                contrast = -1
            if 0 <= contrast <= 255:
                set_value(contrast)
            else:
                entry_value.set(str(round(value.get())))

        entry.bind('<Return>', commit)
        entry.bind('<FocusOut>', commit)
        entry.configure(command=commit)
        ttk.Button(self.table, text='Reset', width=5,
                   command=lambda: set_value(NEUTRAL_CONTRAST)).grid(row=row, column=5, padx=(4, 0))
        return visible, value, entry_value
