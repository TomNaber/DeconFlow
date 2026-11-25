"""Lossless scene writers and validated publication of file/directory outputs."""

from pathlib import Path
from hashlib import sha256
import shutil

import numpy as np
import tifffile

from source.scene_io import Dataset

SUFFIX = {
    "Imaris": ".ims",
    "TIFF": ".tif",
    "BigTIFF": ".btf",
    "OME-TIFF": ".ome.tiff",
    "OME-Zarr": ".ome.zarr",
}


def destination(scene, requested, tag, *, disambiguate=False):
    family = requested
    if family == "Follow input":
        family = scene.family if scene.family in SUFFIX else "OME-TIFF"
    path = Path(scene.path)
    suffix = SUFFIX[family]
    stem = path.stem if scene.family == "Imaris" else path.name
    # Keep historical IMS names when they identify a single input.
    if scene.family != "Imaris" or disambiguate:
        identity = f"{scene.path}\0{scene.index}".encode()
        stem += "-" + sha256(identity).hexdigest()[:10]
    if scene.series_count > 1:
        stem += f"_scene{scene.index + 1}"
    if requested == "Follow input" and scene.family == family:
        suffix = (
            "".join(path.suffixes[-2:])
            if path.stem.lower().endswith(".ome")
            else path.suffix
        )
    return family, f"{stem}_deconvolved_{tag}{suffix}"


def output_dtype(dtype):
    dtype = np.dtype(dtype)
    return dtype if dtype.name in ("uint32", "float32") else np.dtype("uint16")


def encode(staged, output, maximum):
    dtype = output.dtype
    integer = dtype.kind == "u"
    limit = int(np.iinfo(dtype).max) if integer else None
    factor = limit / maximum if integer and maximum > limit else 1.0
    for t, c in np.ndindex(staged.shape[:2]):
        if integer:
            block = staged[t, c].astype(np.float64)
            np.nan_to_num(block, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            np.multiply(block, factor, out=block)
            np.rint(block, out=block)
            np.clip(block, 0, limit, out=block)
            output[t, c] = block
        else:
            output[t, c] = staged[t, c]
    output.flush()
    return factor


def color_hex(ch):
    return "".join(f"{round(255 * v):02X}" for v in ch["color"])


def write_scene(data, path, family, scene, ranges, record, *, follow=False):
    channels = scene.channels
    names = [ch["name"] for ch in channels]
    omissions = record.setdefault("metadata_notes", [])
    if scene.family == "Bio-Formats":
        omissions.append(
            "Vendor-specific acquisition fields outside the exported scene metadata are omitted."
        )
    if scene.family == "Imaris" and family != "Imaris":
        omissions.append(
            "Imaris-specific DataSetInfo sections, scene objects and resolution pyramids are omitted."
        )
    if family == "Imaris":
        from source.ims_writer import write_imaris

        native = scene.family == "Imaris"
        if scene.planes:
            omissions.append(
                "IMS stores time-point timestamps and image extents. Individual plane acquisition records are omitted."
            )
        if not native and any(v is None for v in scene.spacing):
            omissions.append(
                "IMS requires extents. Missing spatial scales use one micrometer per pixel."
            )
        if not native and not scene.times and data.shape[0] > 1:
            omissions.append(
                "Source time intervals unavailable. IMS time points use the same timestamp."
            )
        if not native and not scene.acquired:
            omissions.append(
                "IMS absolute time origin unavailable. Relative times anchored to 1970-01-01."
            )
        write_imaris(
            data,
            path,
            source_ims=scene.path if native else None,
            scene=scene,
            channel_names=names,
            channel_colors=[ch["color"] for ch in channels],
            excitation_wavelengths_nm=[ch.get("excitation_nm") for ch in channels],
            emission_wavelengths_nm=[ch.get("emission_nm") for ch in channels],
            channel_ranges=ranges,
        )
        return
    details = dict(
        channels=channels,
        spacing=scene.spacing,
        times=scene.times,
        acquired=scene.acquired,
        origin=scene.origin,
        planes=scene.planes,
    )
    if family == "OME-Zarr":
        from bioio_ome_zarr.writers import OMEZarrWriter
        from bioio_ome_zarr.writers.metadata import Channel
        import zarr

        version = 3 if follow and scene.ngff == "0.5" else 2
        dt = np.diff(scene.times)
        regular = len(dt) and dt[0] > 0 and np.allclose(dt, dt[0])
        dims = tuple(min(n, b) for n, b in zip(data.shape, (1, 1, 16, 256, 256)))
        colors = [
            Channel(
                label=ch["name"],
                color=color_hex(ch),
                window=dict(start=lo, end=hi, min=lo, max=hi),
            )
            for ch, (lo, hi) in zip(channels, ranges)
        ]
        offset = [scene.times[0] if regular else 0.0, 0.0, *scene.origin]
        with OMEZarrWriter(
            str(path),
            [data.shape],
            data.dtype,
            zarr_format=version,
            chunk_shape=dims,
            channels=colors,
            image_name=scene.name,
            root_transform=dict(type="translation", translation=offset)
            if any(offset)
            else None,
            axes_names=list("tczyx"),
            axes_types=["time", "channel", *(["space"] * 3)],
            axes_units=[
                "second" if regular else None,
                None,
                *["micrometer" if v else None for v in scene.spacing],
            ],
            physical_pixel_size=[
                float(dt[0]) if regular else 1.0,
                1.0,
                *(v or 1.0 for v in scene.spacing),
            ],
        ) as writer:
            for t, c in np.ndindex(data.shape[:2]):
                region = (
                    slice(t, t + 1),
                    slice(c, c + 1),
                    *(slice(0, n) for n in data.shape[2:]),
                )
                writer.write_region(data[region], region)
        zarr.open_group(str(path), mode="r+").attrs["deconflow"] = details
        return
    ome = family == "OME-TIFF"
    meta = {"axes": "TCZYX"}
    if ome:
        rgba = [
            int.from_bytes(bytes.fromhex(color_hex(ch) + "FF"), "big", signed=True)
            for ch in channels
        ]
        entries = []
        for name, color, ch in zip(names, rgba, channels):
            entry = dict(Name=name, Color=color)
            for field, key in (
                ("ExcitationWavelength", "excitation_nm"),
                ("EmissionWavelength", "emission_nm"),
            ):
                if ch.get(key) is not None:
                    entry[field] = ch[key]
            entries.append(entry)
        meta.update(Name=scene.name, Channel=entries)
        if scene.acquired:
            meta["AcquisitionDate"] = scene.acquired
        for a, v in zip("ZYX", scene.spacing):
            if v is not None:
                meta["PhysicalSize" + a] = v
                meta["PhysicalSize" + a + "Unit"] = "µm"
        if scene.planes or scene.times or any(scene.origin):
            recorded = {(t, c, z): fields for t, c, z, fields in scene.planes}
            planes = []
            for t, c, z in np.ndindex(data.shape[:3]):
                plane = dict(recorded.get((t, c, z), {}))
                if scene.times:
                    plane.setdefault("DeltaT", scene.times[t])
                if any(scene.origin):
                    for axis, value in zip("ZYX", scene.origin):
                        key = "Position" + axis
                        if key not in plane:
                            plane[key] = value + (
                                z * (scene.spacing[0] or 0.0) if axis == "Z" else 0.0
                            )
                            plane[key + "Unit"] = "µm"
                planes.append(plane)
            meta["Plane"] = planes
    else:
        meta["deconflow"] = details
    dy, dx = scene.spacing[1:]
    resolution = (
        dict(resolution=(10000.0 / dx, 10000.0 / dy), resolutionunit="CENTIMETER")
        if dx and dy
        else {}
    )
    tifffile.imwrite(
        path,
        data,
        **resolution,
        ome=ome,
        metadata=meta,
        photometric="minisblack",
        compression="deflate",
        bigtiff=(family == "BigTIFF" or data.nbytes >= 2**32 - 2**25),
    )


def validate_output(path, expected, scene):
    with Dataset(path) as ds:
        actual = ds.select(0)
        if actual.shape != expected.shape or np.dtype(actual.dtype) != expected.dtype:
            raise ValueError(
                f"Output dimensions or dtype mismatch: {actual.shape}, {actual.dtype}"
            )
        if [ch["name"] for ch in actual.channels] != [
            ch["name"] for ch in scene.channels
        ]:
            raise ValueError("Output channel names differ")
        for wanted, found in zip(scene.channels, actual.channels):
            if not np.allclose(wanted["color"], found["color"], atol=1 / 255 + 1e-5):
                raise ValueError(f"Output channel color differs for {wanted['name']}")
        if scene.times and (
            len(scene.times) != len(actual.times)
            or not np.allclose(scene.times, actual.times, rtol=0, atol=0.002)
        ):
            raise ValueError("Output frame timestamps differ")
        for t, c in np.ndindex(expected.shape[:2]):
            pixels = ds.volume(t, c)
            if not np.array_equal(pixels, expected[t, c], equal_nan=True):
                raise ValueError(f"Output pixels differ at T{t + 1}, C{c + 1}")


def publish(staged, target, overwrite):
    from source.platform_support import exclusive_file_lock

    with exclusive_file_lock(
        target.parent / ".deconflow-publish.lock", 30, lambda: None
    ):
        _replace(staged, target, overwrite)


def _replace(staged, target, overwrite):
    backup = target.with_name(f".{target.name}.deconflow-previous")
    if backup.is_dir():
        if target.exists():
            shutil.rmtree(backup)
        else:
            backup.rename(target)
    if target.exists() and not overwrite:
        raise FileExistsError(target)
    if not target.is_dir():
        staged.replace(target)
        return
    target.rename(backup)
    try:
        staged.rename(target)
    except BaseException:
        backup.rename(target)
        raise
    shutil.rmtree(backup)
