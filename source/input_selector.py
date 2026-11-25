"""Native menu for choosing which discovered Z-stacks to process."""
import tkinter as tk
from tkinter import ttk

from source.scene_io import input_path_label, input_scene_label


class InputSelector(ttk.Menubutton):
    def __init__(self, parent, select):
        super().__init__(parent, text='0/0 selected', direction='below')
        self.select = select
        self.folder = None
        self.scenes = ()
        self.unchecked = frozenset()
        self.scanning = False
        self.menu_dirty = True
        self.items = []
        self.menu = tk.Menu(self, tearoff=False, postcommand=self.prepare_menu)
        self.configure(menu=self.menu)

    def display(self, folder, scenes, unchecked, *, scanning=False):
        folder_changed = folder != self.folder
        if folder_changed or scenes is not self.scenes or scanning != self.scanning:
            self.menu_dirty = True
        self.folder, self.scenes, self.unchecked, self.scanning = folder, scenes, unchecked, scanning
        if not scanning:
            selected = sum(scene.key not in unchecked for scene in scenes)
            self.configure(text=f'{selected}/{len(scenes)} selected')
        elif folder_changed:
            self.configure(text='0/0 selected')

    def prepare_menu(self):
        if self.menu_dirty:
            self.build_menu()
            self.menu_dirty = False
        for keys, variable in self.items:
            variable.set(all(key not in self.unchecked for key in keys))

    def build_menu(self):
        self.menu.delete(0, 'end')
        for submenu in self.menu.winfo_children():
            submenu.destroy()
        self.items.clear()

        keys = () if self.scanning else tuple(scene.key for scene in self.scenes)
        state = 'normal' if keys else 'disabled'
        self.menu.add_command(label='Select all', state=state, command=lambda: self.select(keys, True))
        self.menu.add_command(label='Clear all', state=state, command=lambda: self.select(keys, False))
        if self.scanning:
            return

        self.menu.add_separator()
        if not keys:
            self.menu.add_command(label='No readable Z-stacks found', state='disabled')
            return

        groups = {}
        for scene in self.scenes:
            groups.setdefault(scene.path, []).append(scene)
        for path, group in groups.items():
            if len(group) == 1:
                scene = group[0]
                self.add_check(self.menu, input_scene_label(scene, self.folder), (scene.key,))
                continue
            submenu = tk.Menu(self.menu, tearoff=False)
            self.menu.add_cascade(label=input_path_label(path, self.folder), menu=submenu)
            self.add_check(submenu, 'All stacks', tuple(scene.key for scene in group))
            submenu.add_separator()
            for scene in group:
                self.add_check(submenu, f'[{scene.index + 1}] {scene.name}', (scene.key,))

    def add_check(self, menu, label, keys):
        checked = tk.BooleanVar(master=menu)
        menu.add_checkbutton(label=label, variable=checked, command=lambda: self.select(keys, checked.get()))
        self.items.append((keys, checked))

    def close(self):
        if self.menu.winfo_ismapped():
            self.tk.call('::tk::MenuUnpost', self.menu)
