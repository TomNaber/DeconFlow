# Native deliverables

The version lives in `app_version.py`. Update it before building a release; it
appears in the window title, `python main.py --version`, archive names, and the
macOS bundle metadata. Native installer metadata uses numeric versions.

Build on each target OS and architecture: Apple Silicon macOS 14+, Intel macOS
13+, and Windows x64. These minimums reflect the pinned CLIJ2-FFT native binaries
and are also written into the macOS application metadata. Cross-compilation is not supported. Start with a clean environment
from `environment.yml`, activate it, then install the packaging dependency:

```sh
conda env create -f environment.yml
conda activate batch-deconvolution
python -m pip install -r requirements-build.txt
```

On macOS:

```sh
bash scripts/build_macos.sh --installer
```

On Windows, use PowerShell with the environment activated:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build_windows.ps1 --installer
```

Windows installer builds require Inno Setup 6 (available on the Windows CI runner).
With `--installer`, builds also produce a macOS DMG or a per-user Windows Setup.exe
and its SHA-256 checksum. Both accept `--output PATH` and write a versioned ZIP and SHA-256 file into
`dist/` by default. Existing release files are never overwritten. Builds check
installed dependencies, native imports, and Java before packaging. They also run
`scripts/setup_imagej.py` to automatically provision Java 21 in the build
environment and install/verify the pinned ImageJ and Bio-Formats runtimes, then copy them into
the release app folder. Downloads are required only when locked artifacts are
missing or corrupt. Dependency notices and the lock manifest are included. The complete
active Conda environment is included, so build from a dedicated environment
without private files or unrelated packages. See the
[conda-pack documentation](https://conda.github.io/conda-pack/) for runtime
relocation details.

## Running a deliverable

- macOS: open the DMG (or extract the ZIP), move `Batch Deconvolution.app` to Applications, and open it.
- Windows: run Setup.exe and use the Start menu shortcut, or extract the entire
  portable ZIP and double-click `Launch.cmd` inside its folder.

Python, Tk, Java 21, Python dependencies, and the pinned headless ImageJ/CLIJ2
runtime are bundled. Users do not need Conda or Fiji and do not select an
executable. The ImageJ libraries live in `app/runtime/imagej/`; the exact artifact
versions and hashes are in `app/source/imagej/runtime.json`. Startup verifies
these hashes and never falls back to an external Fiji installation.
Bio-Formats 8.5.0 and its dependencies are included as the official upstream
bundle, verified against `app/source/bioformats/runtime.json`. Maven resolution
and Java downloading are disabled during image reading. First launch and image
processing work without internet access; no separate Java or Fiji installation
is needed. The launcher verifies bundled Java before marking initial extraction
complete.
Input images and generated PSFs are not included; generate or select your PSFs.

On first launch the runtime is extracted and relocated into the user's
`Library/Application Support/Batch Deconvolution` folder on macOS or
`%LOCALAPPDATA%\Batch Deconvolution` on Windows. Allow time and disk space for this
one-time extraction. Each runtime archive has a separate cache; the app can be
moved without moving its runtime. Old `runtime-*` cache directories can be
removed when no app is using them. On macOS startup errors are written to
`launcher.log` in that folder. If an interrupted macOS setup leaves a
`runtime-*.lock` directory, remove it after confirming the app is closed and retry.
Default data/output folders are under Documents/Batch Deconvolution, so the
application bundle does not need to be writable. Existing saved settings still apply; obsolete Fiji executable paths are ignored.
ImageJ native-library extraction uses the user cache (`Library/Caches/Batch
Deconvolution` on macOS or `%LOCALAPPDATA%\Batch Deconvolution\cache` on Windows),
and job files use temporary directories. The ImageJ app folder may be read-only.
A missing or corrupt packaged ImageJ runtime requires reinstalling the release.
Source installations can be repaired with `python scripts/setup_imagej.py`.

These setups and portable archives are unsigned and are not notarized. macOS security
policy may block downloaded builds until approved by the user or administrator.
Before distributing a release, verify GUI startup, PSF generation, preview, and
IMS export on a clean machine for each target architecture. A successful build
does not validate the target machine's GPU/OpenCL setup. Java is bundled from
the build environment; ImageJ artifact versions are fixed by the manifest.
Any manifest upgrade must repeat numerical, lifecycle, and native-platform
validation before release.

## Optional post-processing

The **Post-processing** group is always visible within Deconvolution, with per-channel
output names, **None** / **Median filter**, and kernel sizes **3**, **5**, or **7**
voxels (default 3). Median filtering uses a cubic 3D neighborhood with reflected
borders after deconvolution and before uint16 conversion. Preview and IMS export
use the same filter. Disabled channels remain unchanged. Select **None** (the
default) to leave a channel unfiltered; output names default to the deconvolution names.
Settings are saved between sessions and included in batch reports. SciPy is bundled.
