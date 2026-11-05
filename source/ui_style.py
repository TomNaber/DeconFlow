"""Small theme helpers; all controls and group borders retain their native style."""

import tkinter as tk
from tkinter import ttk


def is_dark(window):
    if window.tk.call("tk", "windowingsystem") == "aqua":
        try:
            return bool(window.tk.getboolean(window.tk.call(
                "tk::unsupported::MacWindowStyle", "isdark", window.winfo_toplevel()._w)))
        except tk.TclError:
            pass
    background = ttk.Style(window).lookup("TFrame", "background") or "#eeeeee"
    return sum(window.winfo_rgb(background)) / (3 * 257) < 128


def workspace_background(window):
    # A plain Tk frame honors its background on Aqua; a themed frame may not.
    if is_dark(window):
        return "#484848"
    return "#f5f5f5"
