# Native deliverables

The version lives in `app_version.py`. Update it before building a release; it
appears in the window title, `python main.py --version`, archive names, and the
macOS bundle metadata. The initial version is 0.1.0.

Build on each target OS and architecture: Apple Silicon macOS, Intel macOS, and
Windows x64. Cross-compilation is not supported. Start with a clean environment
from `environment.yml`, activate it, then install the packaging dependency:

```sh
conda env create -f environment.yml
conda activate batch-deconvolution
python -m pip install -r requirements-build.txt
```

On macOS:

```sh
bash scripts/build_macos.sh
```

On Windows, use PowerShell with the environment activated:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build_windows.ps1
```

Both accept `--output PATH` and write a versioned ZIP and SHA-256 file into
`dist/` by default. Existing release files are never overwritten. Builds check
installed dependencies, native imports, and Java before packaging. The complete
active Conda environment is included, so build from a dedicated environment
without private files or unrelated packages. See the
[conda-pack documentation](https://conda.github.io/conda-pack/) for runtime
relocation details.

## Running a deliverable

- macOS: extract the ZIP, move `Batch Deconvolution.app` to Applications, and open it.
- Windows: extract the entire ZIP and double-click `Launch.cmd` inside its folder.

Python, Tk, Java, and installed Python dependencies are bundled. Users do not
need Conda. Fiji is **not** bundled: install the matching Fiji distribution,
enable CLIJ, CLIJ2, and clijx-deconvolution, and select its executable in the app.
The first Bio-Formats use may require internet access to fetch its Java artifacts.
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
application bundle does not need to be writable. Existing saved settings still apply.

These are unsigned deliverables, not signed/notarized installers. macOS security
policy may block downloaded builds until approved by the user or administrator.
Before distributing a release, verify GUI startup, PSF generation, preview, and
IMS export on a clean machine for each target architecture. A successful build
does not validate Fiji's GPU/OpenCL setup.
