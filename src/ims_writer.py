#!/usr/bin/env python3
"""Standalone native Imaris IMS writer.

This module writes NumPy-compatible arrays to Imaris 5.5 ``.ims`` files by
using the optional ``napari_vipp_imaris.PyImarisWriter`` extension directly.
It has no dependency on napari or napari-vipp.

The main entry point is :func:`write_imaris`. Arrays may be normal NumPy
arrays, NumPy memmaps, or other array-like objects supporting ``shape``,
``dtype``, and NumPy-style slicing.
"""

from __future__ import annotations

import platform
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from importlib import import_module
from pathlib import Path
from typing import Any

import h5py
import numpy as np

SUPPORTED_IMARIS_DTYPES = frozenset({"uint8", "uint16", "uint32", "float32"})
_CANONICAL_AXES = ("t", "c", "z", "y", "x")
_MAX_BLOCK_SIZE = {"t": 1, "c": 1, "z": 16, "y": 256, "x": 256}
_DEFAULT_COLORS = (
    (0.0, 1.0, 0.0),
    (1.0, 0.25, 0.0),
    (1.0, 0.0, 1.0),
    (0.0, 0.7, 1.0),
    (1.0, 1.0, 0.0),
    (1.0, 1.0, 1.0),
)


class OptionalImarisWriterError(ImportError):
    """Raised when the optional native IMS writer is unavailable."""


def write_imaris(
    data: Any,
    path: str | Path,
    *,
    axes: str = "TCZYX",
    voxel_size_um: tuple[float, float, float],
    channel_names: Sequence[str] | None = None,
    excitation_wavelengths_nm: Sequence[float | None] | None = None,
    emission_wavelengths_nm: Sequence[float | None] | None = None,
    source_ims: str | Path | None = None,
    origin_um: tuple[float, float, float] | None = None,
    objective_na: float | None = None,
    refractive_index: float | None = None,
    acquisition_date: str | datetime | None = None,
    channel_colors: Sequence[Sequence[float] | str | None] | None = None,
    channel_ranges: Sequence[tuple[float, float] | None] | None = None,
    application_name: str = "RedLionfish batch",
    application_version: str = "1.0",
) -> Path:
    """Write one array as an Imaris 5.5 ``.ims`` dataset.

    Parameters
    ----------
    data:
        Array-like image data. It must support ``shape``, ``dtype``, and
        NumPy-style slicing. A disk-backed ``numpy.memmap`` is supported.
    path:
        Destination ``.ims`` path.
    axes:
        Axis order of ``data``. Supported axes are T, C, Z, Y, and X. Missing
        axes are inserted as singleton dimensions. Every non-singleton input
        dimension must have an explicit supported axis label.
    voxel_size_um:
        Spatial sampling as ``(z, y, x)`` in micrometres.
    channel_names, excitation_wavelengths_nm, emission_wavelengths_nm:
        Optional per-channel metadata.
    source_ims:
        Optional original IMS file. Its DataSetInfo metadata, display colors,
        spatial origin/extents, and timestamps are reused where possible.
    origin_um:
        Optional ``(z, y, x)`` origin. Ignored when valid source extents are
        available. Defaults to zero.
    objective_na, refractive_index:
        Optional acquisition metadata. Explicit values override source values.
    acquisition_date:
        ISO timestamp or ``datetime`` used when source time points are absent.
    channel_colors:
        Optional RGB values (0..1 or 0..255), hex strings, or ``None``.
    channel_ranges:
        Optional per-channel display ranges as ``(minimum, maximum)``. These
        affect Imaris visualization only; stored voxel values are unchanged.
    """

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    shape = tuple(int(size) for size in getattr(data, "shape", ()))
    if not shape:
        raise ValueError("IMS export requires a non-scalar image array.")
    if any(size <= 0 for size in shape):
        raise ValueError("IMS export requires a non-empty image array.")

    dtype_value = getattr(data, "dtype", None)
    dtype = np.dtype(dtype_value if dtype_value is not None else np.asarray(data).dtype)
    if dtype.name not in SUPPORTED_IMARIS_DTYPES:
        supported = ", ".join(sorted(SUPPORTED_IMARIS_DTYPES))
        raise ValueError(
            f"IMS export does not silently convert {dtype.name}. Convert the "
            f"image explicitly to one of: {supported}."
        )

    source_axes, sizes = _resolve_axes(shape, axes)
    dz, dy, dx = _validate_voxel_size(voxel_size_um)
    source_path = Path(source_ims) if source_ims is not None else None
    source_parameters = read_imaris_dataset_info(source_path) if source_path else {}

    names = _normalise_metadata_sequence(channel_names, sizes["c"], "channel_names")
    excitation = _normalise_metadata_sequence(
        excitation_wavelengths_nm,
        sizes["c"],
        "excitation_wavelengths_nm",
    )
    emission = _normalise_metadata_sequence(
        emission_wavelengths_nm,
        sizes["c"],
        "emission_wavelengths_nm",
    )
    colors = _normalise_metadata_sequence(channel_colors, sizes["c"], "channel_colors")
    ranges = _normalise_metadata_sequence(channel_ranges, sizes["c"], "channel_ranges")

    writer = _load_writer()
    try:
        _write_with_imariswriter(
            writer=writer,
            data=data,
            path=output_path,
            dtype=dtype,
            source_axes=source_axes,
            sizes=sizes,
            voxel_size_um=(dz, dy, dx),
            origin_um=origin_um,
            source_parameters=source_parameters,
            channel_names=names,
            excitation_wavelengths_nm=excitation,
            emission_wavelengths_nm=emission,
            objective_na=objective_na,
            refractive_index=refractive_index,
            acquisition_date=acquisition_date,
            channel_colors=colors,
            channel_ranges=ranges,
            application_name=application_name,
            application_version=application_version,
        )
    except Exception as exc:
        detail = f" {exc}" if str(exc).strip() else ""
        raise RuntimeError(f"ImarisWriter failed to write {output_path}.{detail}") from exc

    return output_path


def read_imaris_dataset_info(path: str | Path | None) -> dict[str, dict[str, str]]:
    """Read top-level ``DataSetInfo`` parameter sections from an IMS file."""

    if path is None:
        return {}
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Source IMS not found: {source_path}")

    sections: dict[str, dict[str, str]] = {}
    with h5py.File(source_path, "r") as handle:
        group = handle.get("DataSetInfo")
        if group is None:
            return sections
        for section_name, section_group in group.items():
            if not isinstance(section_group, h5py.Group):
                continue
            values: dict[str, str] = {}
            for name, value in section_group.attrs.items():
                values[str(name)] = _decode_hdf5_attribute(value)
            sections[str(section_name)] = values
    return sections


def _decode_hdf5_attribute(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").rstrip("\x00")
    if isinstance(value, str):
        return value.rstrip("\x00")
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"S", "a"}:
            return value.tobytes().decode("utf-8", errors="replace").rstrip("\x00")
        if value.dtype.kind == "U":
            return "".join(value.astype(str).ravel()).rstrip("\x00")
        if value.ndim == 0:
            return _decode_hdf5_attribute(value.item())
        return " ".join(str(item) for item in value.ravel())
    if isinstance(value, np.generic):
        return str(value.item())
    return str(value)


def require_imaris_writer():
    """Return the native writer backend or raise before processing begins."""

    return _load_writer()


def _load_writer():
    errors: list[str] = []

    # Preferred backend from the supplied macOS-arm64 VIPP example.
    try:
        package = import_module("napari_vipp_imaris")
    except Exception as exc:
        errors.append(
            f"napari_vipp_imaris: {type(exc).__name__}: {exc}"
        )
    else:
        writer = getattr(package, "PyImarisWriter", None)
        if writer is not None:
            return writer
        errors.append(
            "napari_vipp_imaris imported, but PyImarisWriter is missing"
        )

    # Compatibility fallback for environments containing the official
    # PyImarisWriter module directly. The public package has historically
    # bundled Windows binaries, but a locally supplied compatible build may
    # expose the same API on another platform.
    try:
        writer = import_module("PyImarisWriter")
    except Exception as exc:
        errors.append(
            f"PyImarisWriter: {type(exc).__name__}: {exc}"
        )
    else:
        required_names = (
            "ImageSize",
            "DimensionSequence",
            "Options",
            "ImageConverter",
        )
        missing = [
            name
            for name in required_names
            if not hasattr(writer, name)
        ]
        if not missing:
            return writer
        errors.append(
            "PyImarisWriter imported, but is missing: "
            + ", ".join(missing)
        )

    system = platform.system()
    machine = platform.machine()
    platform_text = f"{system} {machine}"

    details = "\n  - ".join(errors)
    raise OptionalImarisWriterError(
        "Native IMS export is unavailable in this environment "
        f"({platform_text}). The batch has been stopped before deconvolution "
        "so no processing time is wasted.\n\n"
        "Install or copy the macOS-arm64 'napari_vipp_imaris' wheel/module "
        "that accompanied the supplied VIPP writer example into this exact "
        "Python environment. The similarly named public PyImarisWriter package "
        "may be used only if you have a build compatible with this platform.\n\n"
        "Backend import attempts:\n  - "
        + details
    )


def _resolve_axes(
    shape: tuple[int, ...],
    axes: str,
) -> tuple[tuple[str | None, ...], dict[str, int]]:
    labels = tuple(str(axes).strip().casefold())
    if len(labels) != len(shape):
        raise ValueError(
            f"axes={axes!r} has {len(labels)} labels but data has {len(shape)} dimensions."
        )

    sizes = dict.fromkeys(_CANONICAL_AXES, 1)
    source_axes: list[str | None] = []
    seen: set[str] = set()

    for label, size in zip(labels, shape, strict=True):
        key = None if label in {"_", "?"} else label
        if key is None:
            if size != 1:
                raise ValueError(
                    "Only singleton dimensions may use an ignored '_' or '?' axis."
                )
            source_axes.append(None)
            continue
        if key not in _CANONICAL_AXES:
            raise ValueError(
                f"Unsupported axis {label!r}. Use only T, C, Z, Y, X, or '_' for a singleton."
            )
        if key in seen:
            raise ValueError(f"Duplicate axis {key.upper()} in axes={axes!r}.")
        seen.add(key)
        source_axes.append(key)
        sizes[key] = int(size)

    if "x" not in seen or "y" not in seen:
        raise ValueError("IMS export requires explicit X and Y axes.")

    return tuple(source_axes), {key: int(value) for key, value in sizes.items()}


def _validate_voxel_size(
    voxel_size_um: tuple[float, float, float],
) -> tuple[float, float, float]:
    if len(voxel_size_um) != 3:
        raise ValueError("voxel_size_um must be a (z, y, x) tuple.")
    dz, dy, dx = (float(value) for value in voxel_size_um)
    if not all(np.isfinite(value) and value > 0 for value in (dz, dy, dx)):
        raise ValueError("All voxel dimensions must be finite and greater than zero.")
    return dz, dy, dx


def _normalise_metadata_sequence(
    values: Sequence[Any] | None,
    count: int,
    name: str,
) -> tuple[Any | None, ...]:
    if values is None:
        return tuple(None for _ in range(count))
    result = tuple(values)
    if len(result) != count:
        raise ValueError(f"{name} has {len(result)} entries but the image has {count} channels.")
    return result


def _write_with_imariswriter(
    *,
    writer: Any,
    data: Any,
    path: Path,
    dtype: np.dtype,
    source_axes: tuple[str | None, ...],
    sizes: dict[str, int],
    voxel_size_um: tuple[float, float, float],
    origin_um: tuple[float, float, float] | None,
    source_parameters: dict[str, dict[str, str]],
    channel_names: tuple[Any | None, ...],
    excitation_wavelengths_nm: tuple[Any | None, ...],
    emission_wavelengths_nm: tuple[Any | None, ...],
    objective_na: float | None,
    refractive_index: float | None,
    acquisition_date: str | datetime | None,
    channel_colors: tuple[Any | None, ...],
    channel_ranges: tuple[Any | None, ...],
    application_name: str,
    application_version: str,
) -> None:
    image_size = writer.ImageSize(**sizes)
    block_sizes = {
        key: min(sizes[key], _MAX_BLOCK_SIZE[key]) for key in _CANONICAL_AXES
    }
    block_size = writer.ImageSize(**block_sizes)
    sample_size = writer.ImageSize(**dict.fromkeys(_CANONICAL_AXES, 1))
    sequence = writer.DimensionSequence("x", "y", "z", "c", "t")
    options = writer.Options()

    source_channels = _mapped_channel_parameters(source_parameters, sizes["c"])
    parameters = _merged_parameter_sections(
        sizes=sizes,
        data_type=dtype.name,
        source=source_parameters,
        source_channels=source_channels,
        channel_names=channel_names,
        excitation_wavelengths_nm=excitation_wavelengths_nm,
        emission_wavelengths_nm=emission_wavelengths_nm,
        objective_na=objective_na,
        refractive_index=refractive_index,
        application_name=application_name,
        application_version=application_version,
    )

    class ConsoleCallback(writer.CallbackClass):
        def RecordProgress(self, progress: float, total_bytes_written: int):
            percentage = max(0.0, min(100.0, float(progress) * 100.0))
            print(
                f"\r    IMS export: {percentage:5.1f}% "
                f"({int(total_bytes_written) / (1024 ** 2):.1f} MiB written)",
                end="",
                flush=True,
            )

    converter = None
    try:
        converter = writer.ImageConverter(
            dtype.name,
            image_size,
            sample_size,
            sequence,
            block_size,
            str(path),
            options,
            str(application_name),
            str(application_version),
            ConsoleCallback(),
        )

        for starts in _block_starts(sizes, block_sizes):
            block_index = writer.ImageSize(
                **{key: starts[key] // block_sizes[key] for key in _CANONICAL_AXES}
            )
            if converter.NeedCopyBlock(block_index):
                converter.CopyBlock(
                    _extract_block(
                        data=data,
                        source_axes=source_axes,
                        starts=starts,
                        sizes=sizes,
                        blocks=block_sizes,
                        dtype=dtype,
                    ),
                    block_index,
                )

        converter.Finish(
            _image_extents(
                writer=writer,
                source=source_parameters,
                sizes=sizes,
                voxel_size_um=voxel_size_um,
                origin_um=origin_um,
            ),
            _parameters(writer, parameters),
            _time_infos(
                source=source_parameters,
                count=sizes["t"],
                acquisition_date=acquisition_date,
            ),
            _color_infos(
                writer=writer,
                source_channels=source_channels,
                explicit_colors=channel_colors,
                channel_names=channel_names,
                channel_ranges=channel_ranges,
            ),
            True,
        )
        print()
    finally:
        if converter is not None:
            converter.Destroy()


def _block_starts(sizes: Mapping[str, int], blocks: Mapping[str, int]):
    for t in range(0, sizes["t"], blocks["t"]):
        for c in range(0, sizes["c"], blocks["c"]):
            for z in range(0, sizes["z"], blocks["z"]):
                for y in range(0, sizes["y"], blocks["y"]):
                    for x in range(0, sizes["x"], blocks["x"]):
                        yield {"t": t, "c": c, "z": z, "y": y, "x": x}


def _extract_block(
    *,
    data: Any,
    source_axes: tuple[str | None, ...],
    starts: Mapping[str, int],
    sizes: Mapping[str, int],
    blocks: Mapping[str, int],
    dtype: np.dtype,
) -> np.ndarray:
    slices: list[slice] = []
    for key in source_axes:
        if key is None:
            slices.append(slice(0, 1))
        else:
            stop = min(starts[key] + blocks[key], sizes[key])
            slices.append(slice(starts[key], stop))

    chunk = data[tuple(slices)]
    compute = getattr(chunk, "compute", None)
    if callable(compute):
        chunk = compute()
    chunk = np.asarray(chunk, dtype=dtype)

    ignored = tuple(index for index, key in enumerate(source_axes) if key is None)
    if ignored:
        chunk = np.squeeze(chunk, axis=ignored)

    present = [key for key in source_axes if key is not None]
    ordered = [key for key in _CANONICAL_AXES if key in present]
    if len(present) > 1:
        chunk = np.transpose(chunk, tuple(present.index(key) for key in ordered))

    valid_shape = tuple(
        min(blocks[key], sizes[key] - starts[key]) if key in present else 1
        for key in _CANONICAL_AXES
    )
    chunk = chunk.reshape(valid_shape)

    padded = np.zeros(tuple(blocks[key] for key in _CANONICAL_AXES), dtype=dtype)
    padded[tuple(slice(0, size) for size in valid_shape)] = chunk
    return np.ascontiguousarray(padded)


def _image_extents(
    *,
    writer: Any,
    source: Mapping[str, Mapping[str, str]],
    sizes: Mapping[str, int],
    voxel_size_um: tuple[float, float, float],
    origin_um: tuple[float, float, float] | None,
):
    source_image = source.get("Image", {})
    source_extents = _source_extents(source_image)
    if source_extents is not None:
        min_x, min_y, min_z, max_x, max_y, max_z = source_extents
    else:
        dz, dy, dx = voxel_size_um
        origin_z, origin_y, origin_x = (
            tuple(float(value) for value in origin_um)
            if origin_um is not None
            else (0.0, 0.0, 0.0)
        )
        min_x, min_y, min_z = origin_x, origin_y, origin_z
        max_x = min_x + dx * sizes["x"]
        max_y = min_y + dy * sizes["y"]
        max_z = min_z + dz * sizes["z"]

    return writer.ImageExtents(min_x, min_y, min_z, max_x, max_y, max_z)


def _source_extents(image: Mapping[str, str]) -> tuple[float, ...] | None:
    names = ("ExtMin0", "ExtMin1", "ExtMin2", "ExtMax0", "ExtMax1", "ExtMax2")
    try:
        values = tuple(float(image[name]) for name in names)
    except (KeyError, TypeError, ValueError):
        return None
    if not all(np.isfinite(value) for value in values):
        return None
    if not all(values[index + 3] > values[index] for index in range(3)):
        return None
    return values


def _parameters(writer: Any, sections: Mapping[str, Mapping[str, str]]):
    parameters = writer.Parameters()
    for section, values in sections.items():
        for name, value in values.items():
            parameters.set_value(str(section), str(name), str(value))
    return parameters


_CHANNEL_DISPLAY_PARAMETERS = frozenset(
    {
        "Color",
        "ColorMode",
        "ColorOpacity",
        "ColorRange",
        "ColorTable",
        "ColorTableLength",
        "GammaCorrection",
    }
)


def _merged_parameter_sections(
    *,
    sizes: Mapping[str, int],
    data_type: str,
    source: Mapping[str, Mapping[str, str]],
    source_channels: tuple[dict[str, str], ...],
    channel_names: tuple[Any | None, ...],
    excitation_wavelengths_nm: tuple[Any | None, ...],
    emission_wavelengths_nm: tuple[Any | None, ...],
    objective_na: float | None,
    refractive_index: float | None,
    application_name: str,
    application_version: str,
) -> dict[str, dict[str, str]]:
    sections = {
        str(section): {str(name): str(value) for name, value in values.items()}
        for section, values in source.items()
        if not _is_regenerated_section(section)
    }

    for index, original in enumerate(source_channels):
        values = {
            name: value
            for name, value in original.items()
            if name not in _CHANNEL_DISPLAY_PARAMETERS
        }
        if channel_names[index] is not None:
            values["Name"] = str(channel_names[index])
        if excitation_wavelengths_nm[index] is not None:
            values["LSMExcitationWavelength"] = _metadata_number(
                float(excitation_wavelengths_nm[index])
            )
        if emission_wavelengths_nm[index] is not None:
            values["LSMEmissionWavelength"] = _metadata_number(
                float(emission_wavelengths_nm[index])
            )
        if refractive_index is not None:
            values["RefractionIndexImmersion"] = _metadata_number(refractive_index)
        sections[f"Channel {index}"] = values

    image = sections.setdefault("Image", {})
    image["NumberOfChannels"] = str(int(sizes["c"]))
    image["Unit"] = "um"
    if objective_na is not None:
        image["NumericalAperture"] = _metadata_number(objective_na)

    processing = sections.setdefault("RedLionfish", {})
    processing["ApplicationName"] = str(application_name)
    processing["ApplicationVersion"] = str(application_version)
    processing["DataType"] = str(data_type)
    processing["AxisOrder"] = "TCZYX"
    return sections


def _is_regenerated_section(name: str) -> bool:
    return bool(name == "TimeInfo" or re.fullmatch(r"Channel\s+\d+", name))


def _mapped_channel_parameters(
    source: Mapping[str, Mapping[str, str]],
    count: int,
) -> tuple[dict[str, str], ...]:
    indexed = sorted(
        (
            (int(match.group(1)), dict(values))
            for name, values in source.items()
            if (match := re.fullmatch(r"Channel\s+(\d+)", name)) is not None
        ),
        key=lambda item: item[0],
    )
    by_index = dict(indexed)
    return tuple(dict(by_index.get(index, {})) for index in range(count))


def _metadata_number(value: float) -> str:
    return format(float(value), ".15g")


def _time_infos(
    *,
    source: Mapping[str, Mapping[str, str]],
    count: int,
    acquisition_date: str | datetime | None,
) -> list[datetime]:
    time_info = source.get("TimeInfo", {})
    preserved = [
        _parse_datetime(time_info.get(f"TimePoint{index + 1}", ""))
        for index in range(count)
    ]
    if preserved and all(value is not None for value in preserved):
        return [value.replace(tzinfo=None) for value in preserved if value is not None]

    base = (
        _parse_datetime(acquisition_date)
        or _parse_datetime(source.get("Image", {}).get("RecordingDate", ""))
        or datetime.now(UTC)
    )

    # If no explicit interval is available, preserve ordering with zero spacing.
    interval_seconds = _source_time_interval_seconds(source)
    return [
        (base + timedelta(seconds=interval_seconds * index)).replace(tzinfo=None)
        for index in range(count)
    ]


def _source_time_interval_seconds(source: Mapping[str, Mapping[str, str]]) -> float:
    time_info = source.get("TimeInfo", {})
    first = _parse_datetime(time_info.get("TimePoint1", ""))
    second = _parse_datetime(time_info.get("TimePoint2", ""))
    if first is None or second is None:
        return 0.0
    return max(0.0, (second - first).total_seconds())


def _parse_datetime(value: str | datetime | None) -> datetime | None:
    if isinstance(value, datetime):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _color_infos(
    *,
    writer: Any,
    source_channels: tuple[dict[str, str], ...],
    explicit_colors: tuple[Any | None, ...],
    channel_names: tuple[Any | None, ...],
    channel_ranges: tuple[Any | None, ...],
):
    infos = []
    for index, source in enumerate(source_channels):
        info = writer.ColorInfo()
        rgb = _parse_color(explicit_colors[index])
        table = _color_table(writer, source.get("ColorTable"))
        use_table = (
            rgb is None
            and source.get("ColorMode", "").casefold() != "basecolor"
            and bool(table)
        )
        if use_table:
            info.set_color_table(table)
        else:
            if rgb is None:
                rgb = _base_color(source.get("Color"))
            if rgb is None:
                rgb = _color_from_channel_name(channel_names[index])
            if rgb is None:
                rgb = np.asarray(_DEFAULT_COLORS[index % len(_DEFAULT_COLORS)])
            info.set_base_color(
                writer.Color(float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)
            )

        opacity = _float_parameter(source.get("ColorOpacity"))
        gamma = _float_parameter(source.get("GammaCorrection"))
        if opacity is not None:
            info.mOpacity = opacity
        if gamma is not None and gamma > 0:
            info.mGammaCorrection = gamma

        explicit_range = channel_ranges[index]
        if explicit_range is not None:
            try:
                range_min = float(explicit_range[0])
                range_max = float(explicit_range[1])
            except (TypeError, ValueError, IndexError) as exc:
                raise ValueError(
                    f"Invalid display range for channel {index}: "
                    f"{explicit_range!r}"
                ) from exc

            if (
                not np.isfinite(range_min)
                or not np.isfinite(range_max)
                or range_max <= range_min
            ):
                raise ValueError(
                    f"Channel {index} display range must be finite with "
                    f"maximum > minimum; got {explicit_range!r}."
                )

            info.mRangeMin = range_min
            info.mRangeMax = range_max

        infos.append(info)
    return infos


def _parse_color(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"#[0-9A-Fa-f]{6}", text):
            return np.asarray(
                [int(text[index : index + 2], 16) / 255.0 for index in (1, 3, 5)],
                dtype=np.float32,
            )
        try:
            components = [float(part) for part in re.split(r"[ ,]+", text) if part]
        except ValueError:
            return None
    else:
        try:
            components = [float(part) for part in value]
        except (TypeError, ValueError):
            return None

    if len(components) < 3 or not all(np.isfinite(components[:3])):
        return None
    rgb = np.asarray(components[:3], dtype=np.float32)
    if float(rgb.max()) > 1.0:
        rgb /= 255.0
    return np.clip(rgb, 0.0, 1.0)


def _color_from_channel_name(value: Any | None) -> np.ndarray | None:
    name = str(value or "").casefold()
    if "488" in name or "green" in name:
        return np.asarray((0.0, 1.0, 0.0), dtype=np.float32)
    if "568" in name or "555" in name or "orange" in name:
        return np.asarray((1.0, 0.25, 0.0), dtype=np.float32)
    if "647" in name or "far red" in name or "far-red" in name:
        return np.asarray((1.0, 0.0, 1.0), dtype=np.float32)
    return None


def _base_color(value: str | None) -> np.ndarray | None:
    try:
        components = [float(part) for part in str(value or "").split()[:3]]
    except (TypeError, ValueError):
        return None
    if len(components) != 3 or not all(np.isfinite(components)):
        return None
    return np.clip(np.asarray(components, dtype=np.float32), 0.0, 1.0)


def _color_table(writer: Any, value: str | None) -> list[Any]:
    try:
        components = [float(part) for part in str(value or "").split()]
    except (TypeError, ValueError):
        return []
    if not components or len(components) % 3:
        return []
    colors = []
    for index in range(0, len(components), 3):
        rgb = np.clip(components[index : index + 3], 0.0, 1.0)
        colors.append(writer.Color(float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0))
    return colors


def _float_parameter(value: str | None) -> float | None:
    try:
        result = float(str(value or "").strip())
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


__all__ = [
    "OptionalImarisWriterError",
    "require_imaris_writer",
    "SUPPORTED_IMARIS_DTYPES",
    "read_imaris_dataset_info",
    "write_imaris",
]
