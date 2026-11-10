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


## Updates and draft releases

Installed copies check the latest published stable GitHub release in the background
on startup. Newer versions prompt before downloading. Downloads must match the
platform, CPU architecture, asset size, and SHA-256 sidecar. Declining leaves the
current installation untouched; network errors are logged without blocking startup.
Source checkouts are not automatically replaced.

After consent, the app waits for active processing/export work, saves settings,
downloads the installer, and exits normally. A detached helper waits for the app
process to exit before installing and relaunching. Windows runs Inno Setup silently
against the current installation directory. macOS mounts the DMG, stages the new
bundle beside the current bundle, and replaces it with rollback on replacement or
launch-command failure. The installation and its parent must be writable; apps
running from the DMG need to be copied to Applications first. OS signing/security
prompts may still require user action. The helper cannot verify successful GUI
startup after the launch command returns.

Update failure details remain in the `deconflow-update-*` temporary directory,
including `update.log` and, on Windows, `installer.log`. Successful downloads are
removed after installation. Preferences, user data, and runtime caches are outside
the replaced application.

Private GitHub repositories require the `DECONFLOW_GITHUB_TOKEN` environment
variable in the app's launch environment, using a token with read-only Contents
access to this repository. Finder launches do not normally inherit terminal shell
environment variables. Tokens are never included in installers. Public repositories
need no token. A missing/inaccessible release does not interrupt startup.

Run the **Build installers** GitHub Actions workflow manually, or push a `vX.Y.Z`
tag matching `app_version.py`. After all three native builds and launcher checks
pass, the workflow uploads the installers and checksums to a **draft** GitHub
release. It refuses to overwrite a published release. Review the draft and publish
it to make it visible to update checks. Increment the version for subsequent
releases; the initial version is `1.0.0`.
