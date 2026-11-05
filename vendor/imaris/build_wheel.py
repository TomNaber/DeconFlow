"""Build a self-contained ImarisWriter wheel (maintainers only; Python >=3.12)."""

import argparse
import base64
import csv
import hashlib
import io
from pathlib import Path
import platform
import shlex
import subprocess
import tarfile
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent
VERSION = "0.1.0a2"
COMMIT = "b128e6e7d1a147261e9d5caf24ebc6b5c9c63779"
SOURCES = {
    "imaris": (f"https://api.github.com/repos/imaris/ImarisWriter/tarball/{COMMIT}",
               "249f0c477397584d59d0b63312ea8215b83e7535d079c10d5cbd7f674f6d857d"),
    "hdf5": ("https://github.com/HDFGroup/hdf5/archive/refs/tags/hdf5_1.14.6.tar.gz",
             "09ee1c671a87401a5201c06106650f62badeea5a3b3941e9b1e2e1e08317357f"),
    "zlib": ("https://zlib.net/fossils/zlib-1.3.1.tar.gz",
             "9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23"),
    "lz4": ("https://github.com/lz4/lz4/archive/refs/tags/v1.10.0.tar.gz",
            "537512904744b35e232912055ccf8ec66d768639ff3abe5788d90d792ec5f48b"),
}
TAGS = {"macos-arm64": "macosx_12_0_arm64", "macos-x86_64": "macosx_12_0_x86_64",
        "windows-x64": "win_amd64"}


def run(*args):
    subprocess.run([str(arg) for arg in args], check=True)


def fetch_sources(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name, (url, digest) in SOURCES.items():
        archive = directory / f"{name}.tar.gz"
        if not archive.exists():
            urllib.request.urlretrieve(url, archive)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Source checksum mismatch: {archive}")
        destination = directory / name
        if destination.exists():
            continue
        with tarfile.open(archive) as source:
            root = source.getmembers()[0].name.split("/")[0]
            source.extractall(directory / f"unpack-{name}", filter="data")
        (directory / f"unpack-{name}" / root).rename(destination)
        (directory / f"unpack-{name}").rmdir()


def package(sources, library, target, output, runtime_licenses=()):
    tag = "py3-none-" + TAGS[target]
    dist = f"napari_vipp_imaris-{VERSION}.dist-info"
    files = {}
    for name in ("PyImarisWriter.py", "ImarisWriterCtypes.py"):
        files[f"napari_vipp_imaris/{name}"] = (sources / "imaris/python/PyImarisWriter" / name).read_bytes()
    files["napari_vipp_imaris/__init__.py"] = (HERE / "loader.py").read_bytes()
    filename = "bpImarisWriter96.dll" if target == "windows-x64" else "libbpImarisWriter96.dylib"
    files[f"napari_vipp_imaris/{filename}"] = library.read_bytes()
    for name, path in {"ImarisWriter": "imaris/LICENSE.txt", "HDF5": "hdf5/COPYING",
                       "zlib": "zlib/README", "LZ4": "lz4/lib/LICENSE"}.items():
        files[f"{dist}/licenses/{name}.txt"] = (sources / path).read_bytes()
    for name, path in runtime_licenses:
        files[f"{dist}/licenses/{name}.txt"] = path.read_bytes()
    files[f"{dist}/METADATA"] = (
        f"Metadata-Version: 2.4\nName: napari-vipp-imaris\nVersion: {VERSION}\n"
        "Summary: Bundled native ImarisWriter for Batch Deconvolution\n"
        "Requires-Python: >=3.12\nRequires-Dist: numpy>=1.24\n"
        + "".join(f"License-File: {name}.txt\n" for name in
                  ["ImarisWriter", "HDF5", "zlib", "LZ4"] + [name for name, _ in runtime_licenses])
        + f"\nImarisWriter commit {COMMIT}; static HDF5 1.14.6, zlib 1.3.1, LZ4 1.10.0.\n"
    ).encode()
    files[f"{dist}/WHEEL"] = f"Wheel-Version: 1.0\nGenerator: bundled-imaris\nRoot-Is-Purelib: false\nTag: {tag}\n".encode()
    record = io.StringIO(newline="")
    writer = csv.writer(record, lineterminator="\n")
    for name, data in sorted(files.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        writer.writerow((name, "sha256=" + digest, len(data)))
    writer.writerow((f"{dist}/RECORD", "", ""))
    files[f"{dist}/RECORD"] = record.getvalue().encode()
    output.mkdir(parents=True, exist_ok=True)
    wheel = output / f"napari_vipp_imaris-{VERSION}-{tag}.whl"
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    print(f"{hashlib.sha256(wheel.read_bytes()).hexdigest()}  {wheel.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=TAGS, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE.parent)
    parser.add_argument("--zig", type=Path, help="Optional Zig 0.14.1 for Windows cross compilation on macOS")
    args = parser.parse_args()
    work = args.work_dir.resolve()
    sources = work / "sources"
    fetch_sources(sources)
    build = work / args.target
    options = [f"-DSOURCES={sources}", "-DCMAKE_BUILD_TYPE=Release"]
    if args.target.startswith("macos"):
        if platform.system() != "Darwin":
            parser.error("macOS wheels require the Apple SDK on macOS")
        options += [f"-DCMAKE_OSX_ARCHITECTURES={args.target.removeprefix('macos-')}",
                    "-DCMAKE_OSX_DEPLOYMENT_TARGET=12.0"]
    elif args.zig:
        # CMake compiler lists preserve paths containing spaces.
        zig = args.zig.resolve().as_posix()
        for command in ("ar", "ranlib", "rc"):
            wrapper = work / f"zig-{command}"
            wrapper.write_text(f'#!/bin/sh\nexec {shlex.quote(zig)} {command} "$@"\n')
            wrapper.chmod(0o755)
        toolchain = work / "windows-zig.cmake"
        toolchain.write_text(
            'set(CMAKE_SYSTEM_NAME Windows)\nset(CMAKE_SYSTEM_PROCESSOR AMD64)\n'
            f'set(CMAKE_C_COMPILER "{zig};cc;-target;x86_64-windows-gnu")\n'
            f'set(CMAKE_CXX_COMPILER "{zig};c++;-target;x86_64-windows-gnu")\n'
            f'set(CMAKE_AR "{(work / "zig-ar").as_posix()}")\n'
            f'set(CMAKE_RANLIB "{(work / "zig-ranlib").as_posix()}")\n'
            f'set(CMAKE_RC_COMPILER "{(work / "zig-rc").as_posix()}")\n'
        )
        options += [f"-DCMAKE_TOOLCHAIN_FILE={toolchain}"]
    elif platform.system() != "Windows":
        parser.error("Windows wheels require Windows/MSVC or --zig")
    else:
        options += ["-A", "x64", "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded"]
    run("cmake", "-S", HERE, "-B", build, *options)
    run("cmake", "--build", build, "--config", "Release", "--target", "bpImarisWriter96", "--parallel", "4")
    name = "bpImarisWriter96.dll" if args.target == "windows-x64" else "libbpImarisWriter96.dylib"
    libraries = list(build.rglob(name))
    if len(libraries) != 1:
        raise RuntimeError(f"Expected one native library: {libraries}")
    if args.target.startswith("macos"):
        run("codesign", "--force", "--sign", "-", libraries[0])
    runtime_licenses = []
    if args.zig:
        zig_lib = args.zig.resolve().parent / "lib"
        runtime_licenses = [("libcxx", zig_lib / "libcxx/LICENSE.TXT"),
                            ("libcxxabi", zig_lib / "libcxxabi/LICENSE.TXT"),
                            ("libunwind", zig_lib / "libunwind/LICENSE.TXT"),
                            ("Zig", args.zig.resolve().parent / "LICENSE"),
                            ("mingw", zig_lib / "libc/mingw/COPYING")]
    package(sources, libraries[0], args.target, args.output.resolve(), runtime_licenses)


if __name__ == "__main__":
    main()
