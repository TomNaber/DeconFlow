"""Package-local loader for the bundled native ImarisWriter."""

import os
from pathlib import Path
import platform

from . import PyImarisWriter

__version__ = "0.1.0a2"
_directory = Path(__file__).resolve().parent
# Keep the handle alive: closing it removes this directory from DLL search.
_dll_directory = os.add_dll_directory(str(_directory)) if os.name == "nt" else None


class _BundledImageConverter(PyImarisWriter.ImageConverter):
    def _get_dll_filename(self):
        names = {"Darwin": "libbpImarisWriter96.dylib", "Windows": "bpImarisWriter96.dll"}
        try:
            name = names[platform.system()]
        except KeyError:
            raise OSError("This ImarisWriter package supports macOS and Windows only.") from None
        return str(_directory / name)


def _exception_init(self, text):
    Exception.__init__(self, text)


PyImarisWriter.ImageConverter = _BundledImageConverter
PyImarisWriter.PyImarisWriterException.__init__ = _exception_init
__all__ = ["PyImarisWriter", "__version__"]
