# Platform verification

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
