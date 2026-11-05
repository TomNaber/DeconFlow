# Native dependencies

Install through the root `environment.yml`. `requirements.txt` selects wheels by
operating system and processor. `SHA256SUMS` records all distributed artifacts.
There are no Linux or Windows ARM artifacts.

## ImarisWriter

`napari-vipp-imaris` 0.1.0a2 provides the existing
`from napari_vipp_imaris import PyImarisWriter` API. The Python wrapper and native
writer come from [ImarisWriter](https://github.com/imaris/ImarisWriter), commit
`b128e6e7d1a147261e9d5caf24ebc6b5c9c63779`.

Each wheel contains one native library with HDF5 1.14.6, zlib 1.3.1, and LZ4 1.10.0
linked statically. The writer depends only on operating-system libraries at
runtime. macOS libraries use a relative install name and are ad-hoc signed.
Windows loading uses an absolute package-local filename and retains the DLL
search-directory handle. License files for all bundled components are in each
wheel's `.dist-info/licenses` directory.

The build uses a separate CMake configuration, leaving upstream writer sources
unchanged. Source URLs and SHA-256 values are pinned in `imaris/build_wheel.py`.
Python 3.13, CMake 3.20+, and a C++ compiler are needed **only to rebuild**. Use
Apple's Command Line Tools on macOS or Visual Studio 2022 C++ tools on Windows.
From the repository root:

```sh
python vendor/imaris/build_wheel.py --target macos-arm64 --work-dir /tmp/imaris-build
python vendor/imaris/build_wheel.py --target macos-x86_64 --work-dir /tmp/imaris-build
```

On Windows, use a Windows work directory:

```powershell
python vendor/imaris/build_wheel.py --target windows-x64 --work-dir C:\Temp\imaris-build
```

The checked-in Windows writer was cross-compiled on macOS using Zig 0.14.1:

```sh
ZIG_GLOBAL_CACHE_DIR=/tmp/imaris-zig-cache python vendor/imaris/build_wheel.py --target windows-x64 --work-dir /tmp/imaris-build --zig /path/to/zig
```

Zig's macOS ARM archive SHA-256 is
`39f3dc5e79c22088ce878edc821dedb4ca5a1cd9f5ef915e9b3cc3053e8faefa`.
The Windows wheel includes the LLVM C++ runtime and MinGW license notices.
The build script prints each resulting wheel's checksum. Native build jobs are
also available in `.github/workflows/native-wheels.yml`; they do not publish files.
Rebuilds are source-reproducible, not guaranteed byte-identical across toolchains.

## psfmodels

Upstream `psfmodels` 0.3.3 has no published CPython 3.13 wheels. To keep installation
compiler-free, this repository includes a macOS universal2 wheel and a Windows
x64 wheel. Scientific code and API are unchanged. The GPL-3.0-or-later license is
included in each wheel; complete corresponding source is distributed as
`psfmodels/psfmodels-0.3.3.tar.gz` (SHA-256
`4dd4388f26b731d0a39b1e0593648c02aff108a89dea2218632536479e24de57`).

To rebuild the macOS wheel, run these commands in a Python 3.13 build environment
with Apple's Command Line Tools. `LDCXXSHARED` prevents Conda build-directory
library paths from leaking into the extension:

```sh
ARCHFLAGS='-arch arm64 -arch x86_64' MACOSX_DEPLOYMENT_TARGET=12.0 CC=clang CXX=clang++ LDCXXSHARED='clang++ -bundle -undefined dynamic_lookup' python -m pip wheel --no-deps --no-cache-dir vendor/psfmodels/psfmodels-0.3.3.tar.gz -w /tmp/psf-wheels
python vendor/psfmodels/repack.py --macos-wheel /tmp/psf-wheels/psfmodels-0.3.3-cp313-cp313-macosx_12_0_arm64.whl
```

Setuptools may label that fat binary with the host architecture. `repack.py`
verifies both slices and system-only linkage before assigning its universal2 tag.
On an Intel build host, pass the generated x86_64 wheel filename instead.

The Windows wheel repackages conda-forge's unmodified
`psfmodels-0.3.3-py313hdea302a_4.conda`, SHA-256
`08e7584a7a722a865b094d70947a941dff239a45b24724fc6bb571f327102478`.
Its native extension was built with VS2022 and pybind11 3.0.1. Python's Windows
Conda dependencies supply the VC14 and Universal C runtimes it requires.
To reproduce the packaging, use Miniforge's base Python, which includes
`conda-package-handling`:

```sh
python vendor/psfmodels/repack.py --windows
```

The repacker checks the archive digest, removes installation-local metadata,
preserves the library and package code, includes the license, and regenerates
wheel metadata and `RECORD` hashes.

## Other vendored source

`source/lib/totalvariationterm.cl` is from
[clij2-fft](https://github.com/clij/clij2-fft/blob/master/lib/totalvariationterm.cl).
Its BSD 3-clause license is in `source/lib/LICENSE.txt`.
