"""Package verified Python 3.13 psfmodels binaries without changing its code."""
import argparse
import base64
import csv
import hashlib
import io
from pathlib import Path
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
WINDOWS_URL = "https://conda.anaconda.org/conda-forge/win-64/psfmodels-0.3.3-py313hdea302a_4.conda"
WINDOWS_SHA256 = "08e7584a7a722a865b094d70947a941dff239a45b24724fc6bb571f327102478"
DIST = "psfmodels-0.3.3.dist-info"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--windows", action="store_true")
    group.add_argument("--macos-wheel", type=Path, help="Wheel built with ARCHFLAGS='-arch arm64 -arch x86_64'")
    args = parser.parse_args()
    if args.windows:
        # Maintainer-only dependency, available in Miniforge's base environment.
        from conda_package_handling.api import extract
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            archive = directory / WINDOWS_URL.rsplit("/", 1)[1]
            urllib.request.urlretrieve(WINDOWS_URL, archive)
            if hashlib.sha256(archive.read_bytes()).hexdigest() != WINDOWS_SHA256:
                raise ValueError("Conda psfmodels checksum mismatch")
            extract(str(archive), dest_dir=str(directory / "unpacked"))
            site = directory / "unpacked/Lib/site-packages"
            files = {p.relative_to(site).as_posix(): p.read_bytes() for p in site.rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts}
            files[f"{DIST}/licenses/LICENSE"] = (directory / "unpacked/info/licenses/LICENSE").read_bytes()
        tag = "cp313-cp313-win_amd64"
    else:
        import subprocess
        with zipfile.ZipFile(args.macos_wheel) as archive:
            files = {name: archive.read(name) for name in archive.namelist() if not name.endswith("/")}
        with tempfile.TemporaryDirectory() as directory:
            extension = Path(directory) / "_psfmodels.so"
            extension.write_bytes(next(data for name, data in files.items() if name.endswith(".so")))
            architectures = subprocess.check_output(["lipo", "-archs", str(extension)], text=True).split()
            if set(architectures) != {"arm64", "x86_64"}:
                raise ValueError(f"Expected a universal2 extension; found {architectures}")
            libraries = subprocess.check_output(["otool", "-L", str(extension)], text=True)
            if any(line.startswith("\t") and not line.strip().startswith("/usr/lib/") for line in libraries.splitlines()):
                raise ValueError("Extension must link only to macOS system libraries")
        tag = "cp313-cp313-macosx_12_0_universal2"
    for name in ("RECORD", "INSTALLER", "REQUESTED", "direct_url.json"):
        files.pop(f"{DIST}/{name}", None)
    files[f"{DIST}/WHEEL"] = f"Wheel-Version: 1.0\nGenerator: batch-deconvolution-repack\nRoot-Is-Purelib: false\nTag: {tag}\n".encode()
    record = io.StringIO(newline="")
    writer = csv.writer(record, lineterminator="\n")
    for name, data in sorted(files.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        writer.writerow((name, "sha256=" + digest, len(data)))
    writer.writerow((f"{DIST}/RECORD", "", ""))
    files[f"{DIST}/RECORD"] = record.getvalue().encode()
    wheel = ROOT / f"psfmodels-0.3.3-{tag}.whl"
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    print(hashlib.sha256(wheel.read_bytes()).hexdigest(), wheel.name)


if __name__ == "__main__":
    main()
