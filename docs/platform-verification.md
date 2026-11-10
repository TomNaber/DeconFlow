# Platform verification

## Managed ImageJ runtime — 8 November 2026

The app now uses `runtime/imagej/`, installed by `scripts/setup_imagej.py` from
`source/imagej/runtime.json`. The runtime contains 60 original upstream JARs,
89,499,588 bytes (85.35 MiB), including ImageJ Common 2.1.1, ImageJ Ops 2.2.0,
CLIJ2 2.5.3.5, CLIJ2-FFT 2.2.0.22, SCIFIO 0.49.0, and Jython 2.7.4. Every artifact
has an exact URL and SHA-256. Initial candidate artifacts were downloaded fresh;
none were copied from the external Fiji installation. Matching installed Fiji
JARs were used only to establish reference versions and expected hashes.

| Check | Apple Silicon | Intel Mac | Windows x64 |
| --- | --- | --- | --- |
| Pinned artifacts downloaded and verified | Passed | Same locked artifacts | Same locked artifacts |
| FFT/JOCL native binaries present | Passed | Inspected | Inspected |
| Fresh native cache and actual GPU deconvolution | Passed | Pending hardware | Pending hardware |
| Numerical comparison against external Fiji | Bit-for-bit identical | Pending hardware | Pending hardware |
| Three-channel preview and IMS export/readback | Passed | Pending hardware | Pending hardware |
| GUI migration and session reuse | Passed | Pending hardware | Pending hardware |
| Startup/active-job cancellation and owner-death cleanup | Passed | Pending hardware | Pending hardware |
| Relocated read-only app directory | Passed | Pending hardware | Pending hardware |

GPU execution used Apple M3 Pro and the existing Conda environment's Azul Java
21.0.10. Deterministic float32 input (ZYX 9 × 15 × 17), a normalized 5 × 5 × 5
Gaussian PSF, two iterations, regularization 0 and 0.0001, and both boundary
modes produced exactly the same voxels as external Fiji. The acceptance threshold
was `rtol=1e-4, atol=1e-6`; maximum observed error was zero. TIFF/BigTIFF input,
float32 output, geometry inspection, and repeated-session FFT cache reuse passed.
The same numerical comparisons passed after switching the app to its installed
runtime. A final fresh-cache job also matched the Fiji baseline bit-for-bit.

Fresh-cache verification exposed an upstream extraction gap: JavaCPP does not
extract all versioned clFFT dylibs. The app now extracts the complete pinned
platform-specific FFT library set (including Windows StatTimer.dll) into its own
manifest-addressed cache and verifies cached bytes before launching Java. This
was rechecked with an empty app cache; no Fiji native cache is needed. macOS
native dependencies reference bundled clFFT plus system libraries. Windows DLL
imports require OpenCL and the VC14/UCRT runtime supplied by the existing Conda
packaging environment; execution still needs validation on Windows hardware.

Binary load commands declare macOS **14.0 on ARM64** and **13.0 on x86_64** for
CLIJ2-FFT. Setup rejects older macOS versions, and release metadata uses these
minimums. The previous macOS 12 target described the Python wheels and did not
establish compatibility of these GPU binaries.

The complete existing small IMS input (three channels, ZYX 9 × 16 × 16) passed
through preview and batch processing with two iterations per channel. The output
IMS reopened with all three finite-valued channels. Its XML batch report recorded
the runtime manifest hash, all artifact hashes, GPU, and actual Java version.
GUI checks used temporary settings: an obsolete executable path was ignored,
the saved memory value survived, and repeated requests reused the session.

Installer checks covered idempotency, corruption detection, repair, simulated
interrupted downloads without replacing the existing installation, and recovery
from an interrupted directory rename. A relocated copy in a path with spaces,
with all app files/directories read-only, completed GPU deconvolution. Startup
cancellation, cancellation during a long native GPU call, normal shutdown,
owner-process death, and actionable startup failure all passed.

CLIJ's optional legacy macro-autocompletion plugin still emits a SciJava plugin
discovery warning because ImageJ Legacy is intentionally absent. Debug inspection
identified that optional plugin; worker services and numerical checks passed.
The app does not suppress other runtime diagnostics.

Verification used temporary commands/data, Python compilation, installer/build
CLI checks, and `git diff --check`. No project tests or test directories were
created or modified. Full release archives were not rebuilt in this verification;
the relocated app check exercised the source/runtime layout used by packaging.
Native Intel and Windows GPU validation remains required before claiming those
platforms are fully verified.

## Existing laptop environment follow-up — 8 November 2026

The initial checks above used `builds/environment`, which already contained Java.
The user's existing `~/miniforge3/envs/batch_deconvolution` environment did not
contain any Java executable, and reproduced the reported “Java 21 is missing”
error. Installed conda-forge OpenJDK 21.0.10 into that existing environment;
Conda also updated OpenSSL and CA certificates. No Python packages were changed.

Repeated validation with that environment's Python executable and no `JAVA_HOME`
override. Java was automatically found at the environment's `lib/jvm/bin/java`.
Runtime setup verification, all four GPU comparisons against the Fiji baseline,
three-channel preview, one-shot deconvolution, complete IMS export/readback,
XML runtime provenance, GUI startup, saved-memory migration, and GUI session
reuse all passed. Outputs and GUI settings stayed in temporary directories.

Setup now checks Java availability even when the ImageJ JARs are already installed.
For an existing Conda environment without Java, the error identifies the active
prefix and gives the exact `conda install --prefix ... -c conda-forge openjdk=21`
repair command. README instructions now cover upgrading an existing environment.

## Complete deliverable and automatic setup — 8 November 2026

Source setup now automatically installs OpenJDK 21 inside the active Conda
prefix when missing; an external Java installation cannot satisfy the release
builder's requirement. Verified using a new Python-only Conda environment with
no Java, no `JAVA_HOME`, and no Conda on PATH. Setup found the Conda installation
through environment package metadata, installed OpenJDK 21.0.10, and succeeded.
A second run reused and verified all runtimes without reinstalling Java.

Setup also downloads the official Bio-Formats 8.5.0 dependency bundle
(53,843,906 bytes), locked by SHA-256 in `source/bioformats/runtime.json`.
The image reader uses that local classpath and the environment's Java, with
Maven resolution and Java fetching disabled. A source-run readback matched the
original image voxels while Maven resolution and outbound Python networking
were explicitly blocked.

Built `dist/complete-runtime-20261108/Batch-Deconvolution-0.1.0-macos-arm64.zip`.
Extracted the actual ZIP into a new directory and invoked its real launcher with
`PATH=/usr/bin:/bin:/usr/sbin:/sbin`, with inherited Java/Conda/Python settings
removed. macOS `sandbox-exec` denied all network operations for the launched app.
The launcher extracted and relocated bundled Python and Java, verified Java, and
completed a three-channel GPU deconvolution and IMS export. The output reopened
successfully; the packaged GUI opened and closed, and Java was confirmed to be
inside the extracted runtime. No system Java, Conda executable, or Maven cache
was required. Local verification details are recorded in
`builds/verification/packaged-runtime-result.json`.

Full-volume output and timing comparisons against external Fiji are documented
in [the runtime comparison](imagej-runtime-comparison.md). All nine float32 output
comparisons were voxel-identical, including their uint16 conversions. Intel Mac
and Windows code paths are implemented but still require native hardware testing.

## Historical external-Fiji verification

Verification performed 5 November 2026. Artifacts and installation changes are
local working-tree changes; they have not been published to GitHub.

| Check | Apple Silicon | Intel Mac | Windows x64 |
| --- | --- | --- | --- |
| Writer wheel built | Native ARM64 | x86_64 cross-build | x64 cross-build |
| Clean Conda + pip installation | Passed | Passed under Rosetta | Dependency resolution passed; native install pending |
| `pip check` | Passed | Passed under Rosetta | Pending native execution |
| Actual IMS export and readback | Passed | Passed under Rosetta | Pending native execution |
| PSF generation | Passed | Passed under Rosetta | Pending native execution |
| Application GUI startup/close | Passed | Pending native execution | Pending native execution |
| Fiji GPU jobs/session reuse | Passed on Apple M3 Pro | Pending native execution | Pending native execution |
| Complete three-channel batch export | Passed | Pending native execution | Pending native execution |
| Fiji process cleanup/startup cancellation | Passed | Pending native execution | Pending native execution |

Rosetta results establish that the x86_64 Python packages and writer execute;
they do not substitute for native Intel GPU/driver checks. No Windows machine was
available. macOS 12 is the wheel deployment target, not a tested OS version.
Linux and Windows ARM are excluded.

## Installation and artifact checks

- Created fresh environments from a source-only snapshot, excluding existing
  environments and build outputs, using the README's `environment.yml` flow.
  Temporary prefixes were used to avoid replacing the user's environment.
- First verified the previous ARM wheel, then repeated installation and export
  with the self-contained 0.1.0a2 wheel and the bundled psfmodels wheel.
- Forced binary-only pip installation. The original install had reused a locally
  cached psfmodels build; the final installation uses versioned vendor artifacts
  and needs no compiler. Python remains 3.13.
- Resolved the Conda dependencies for `osx-64` and `win-64`, and the full Windows
  pip requirements for CPython 3.13/win_amd64, including both vendored wheels.
- Checked that requirement markers select exactly one writer and one PSF wheel
  on each target, and verified wheel `RECORD` entries and `vendor/SHA256SUMS`.
- Inspected both macOS writer slices and the macOS PSF extension: only system
  libraries are linked. The Windows writer imports Windows system/UCRT DLLs;
  the Windows PSF extension also uses the VC14 runtime supplied by Conda Python.
- Parsed changed Python sources and workflow YAML; `git diff --check` passed.

## Runtime checks

The IMS round trip exported deterministic uint16 data with shape
`(T=2, C=2, Z=17, Y=19, X=23)`, exercising a partial final Z block and padded X/Y
blocks. h5py readback matched every voxel, both channel names, channel count, and
all six physical extents. These checks ran on ARM and x86_64 under Rosetta.

A small scalar confocal PSF was generated on both architectures, with the expected
shape, finite values, and nonzero intensity. The complete GUI opened and closed
on ARM using temporary settings, leaving the user's saved settings untouched.

Two small RL-TV jobs completed through the same Fiji session on an Apple M3 Pro.
Their output dimensions and finite values were checked. Closing the session
terminated its owned process and removed its temporary directory; cancelling a
new session during startup also cleaned up successfully.

The complete batch pipeline also processed the existing small three-channel IMS
verification input with two RL-TV iterations per channel, wrote a new IMS file,
and reopened it successfully with all three channels present. Outputs and work
files stayed in temporary directories.

## Remaining native validation

The manually dispatched `Native wheel builds` GitHub Actions workflow builds
writer artifacts on ARM macOS, Intel macOS, and Windows, and separately checks
the documented installation and native imports. It has not been run remotely.
Runner labels follow the [GitHub-hosted runner reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners).

Before describing Intel or Windows as fully verified, run the README installation
on actual target hardware, export and reopen an IMS file, and exercise GUI
preview, a batch export, cancellation during deconvolution, and restart with the
selected native Fiji launcher and OpenCL driver. GitHub runner installation
checks alone cannot establish GPU workflow compatibility.

Verification used temporary commands and temporary data only. No project tests
or test folders were created or modified.


## Minimal installer — 9 November 2026

Built and installed `dist/minimal-macos/DeconFlow-0.1.0-macos-arm64.dmg`
into `/Applications/DeconFlow.app` from the actual mounted DMG. The development
environment was created fresh as `deconflow` from `environment.yml`.

- DMG: **249,577,401 bytes** (249.6 MB); output contains only the installer and
  its SHA-256 checksum, with no portable application ZIP.
- Full build environment file sizes before filtering: 862,721,520 bytes.
- Filtered Python environment file sizes: 291,403,905 bytes.
- Linked Java 21 runtime: 53,002,524 bytes.
- Compressed combined Python/Java archive: 113,066,190 bytes.

Java's compiler is used only during packaging. The installed app uses a compiled
`HeadlessImageJ.class` and a `jlink` runtime without javac or JDK development
files. Dependency tests, Python bytecode caches, headers, static libraries, pip,
conda-pack and wheel code are excluded from the packaged environment. The full
pinned headless ImageJ and Bio-Formats bundles remain intact: their checksums and
native-library extraction behavior are unchanged. Jython and Bio-Formats are
substantial remaining components; this is the reduced configuration verified
with the current processing stack, not a claim of a theoretical minimum size.

Verification used the installed app, an empty temporary HOME/cache, no external
Python/Java/Conda on PATH, `PYTHONNOUSERSITE=1`, and `sandbox-exec` denying network
access. Passed: first-launch runtime extraction, three-channel PSF generation,
all four GPU numerical comparisons (regularization 0 and 0.0001, both boundary
modes), three-channel preview with median filtering, batch IMS export and
Bio-Formats readback, and GUI startup/close. All four numerical comparisons had
zero maximum voxel error against the existing Fiji baseline on Apple M3 Pro.
Normal Launch Services startup also ran the installed bundled Python runtime.

The builder additionally checks that the linked Java runtime starts the compiled
headless ImageJ/Jython entry point before creating an installer. SHA-256, archive
exclusions, installer-only output, Python syntax and workflow YAML were checked.
The Java module list and packaging exclusions are recorded in the installed
`app/runtime/packaging.json`. Local verification output is in
`/private/tmp/deconflow-installed-verification/result.json`.

This smaller runtime has been exercised locally on Apple Silicon only. Windows
and Intel macOS need their own builds and native checks. No release was created
or published during this work.
