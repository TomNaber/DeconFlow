Custom ImarisWriter wheel from napari_vipp_imaris_install

Upstream https://github.com/imaris/ImarisWriter
Commit b128e6e7d1a147261e9d5caf24ebc6b5c9c63779
Apache 2 license included in the wheel

The native library was relinked from Homebrew to @rpath/libhdf5.320.dylib and @rpath/liblz4.1.dylib
The added @loader_path/../../.. search path resolves both from the Conda environment lib directory
The library was ad hoc signed and the wheel RECORD regenerated
Python sources and the writer implementation are unchanged

macOS arm64 only
Install through environment.yml

source/lib/totalvariationterm.cl is from https://github.com/clij/clij2-fft/blob/master/lib/totalvariationterm.cl
Its BSD 3 clause license is in source/lib/LICENSE.txt
