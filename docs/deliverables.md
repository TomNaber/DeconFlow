# Native installers

The version lives in `app_version.py` and appears in the window title,
`python main.py --version`, installer filenames, and native metadata.

Build on each target OS and architecture: Apple Silicon macOS 14+, Intel macOS
13+, or Windows x64 (Windows 10+). Cross-compilation is not supported.

```sh
conda env create -f environment.yml
conda activate deconflow
python -m pip install -r requirements-build.txt
```

On macOS:

```sh
bash scripts/build_macos.sh
```

On Windows:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build_windows.ps1
```

Windows builds require Inno Setup 6. Both builders accept `--output PATH` and
produce only a DMG or Setup.exe plus its SHA-256 checksum. They never produce a
portable application ZIP. Existing outputs are not overwritten. GitHub's artifact
ZIP is just the download container for the installer and checksum.

## Bundled runtime

Build from the dedicated `deconflow` environment. The builder verifies Python
dependencies, native imports, and the locked ImageJ and Bio-Formats artifacts.
It compiles the ImageJ Java entry point and uses `jlink` to create a Java 21
runtime without development tools, headers, or manuals. The development JDK is
not included in the installer. Java's desktop APIs remain necessary for ImageIO
and headless AWT even though the ImageJ user interface is not used.

The Python runtime retains application dependencies and native libraries, but
excludes dependency test data, bytecode caches, headers, static libraries,
package installers, and build tools. Licenses and package metadata are retained.
Filtering changes only the packaged files, never the development environment.
The installer includes the pinned headless ImageJ/CLIJ2 and Bio-Formats JARs,
verified against the bundled manifests. No Fiji or Maven installation is needed.
`app/runtime/packaging.json` records the Java modules, exclusions, and build sizes.

## Installing and running

- macOS: open the DMG and drag `DeconFlow.app` to Applications.
- Windows: run Setup.exe and use the DeconFlow Start menu shortcut.

Python, Tk, Java, and image-processing dependencies are included. First launch
extracts and relocates the internal runtime archive into a user-writable cache.
This internal archive is part of the installer, not a separate portable download.
The app can move without moving its runtime, and its installation directory can
be read-only. Runtime extraction verifies Java before marking the cache ready.

For compatibility with existing installations, cache and preferences folders
retain their historical `Batch Deconvolution` name:
`~/Library/Application Support/Batch Deconvolution` on macOS and
`%LOCALAPPDATA%\Batch Deconvolution` on Windows. Default data/output folders
remain under Documents/Batch Deconvolution. Each runtime has a content-addressed
cache; old `runtime-*` directories can be removed when no app uses them. macOS
launcher errors are written to `launcher.log` in the application-support folder.

ImageJ native libraries are extracted into the user cache, and processing uses
temporary directories. Input images and PSFs are not bundled. Missing or corrupt
packaged runtime files require reinstalling the app; source installations can
be repaired with `python scripts/setup_imagej.py`.

These installers are unsigned and not notarized. GPU processing requires a
working OpenCL driver. Build and launcher checks do not replace native-platform
checks of GUI startup, PSF generation, preview, and IMS export. See
[platform verification](platform-verification.md).
