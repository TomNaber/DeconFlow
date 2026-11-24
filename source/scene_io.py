"""Discover acquisitions and read one full-resolution ZYX channel at a time."""

from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import tifffile

from source.channel_metadata import text

AXES = "TCZYX"
OUTPUT_FORMATS = ("Follow input", "Imaris", "TIFF", "BigTIFF", "OME-TIFF", "OME-Zarr")
TIFF_EXTENSIONS = {".tif", ".tiff", ".btf", ".tf2", ".tf8", ".bigtiff"}


@dataclass
class Scene:
    path: str
    index: int
    name: str
    family: str
    shape: tuple
    dtype: str
    channels: list
    spacing: tuple = (None, None, None)
    times: list = field(default_factory=list)  # elapsed seconds
    acquired: str | None = None
    planes: list = field(default_factory=list)
    ngff: str | None = None
    series_count: int = 1
    origin: tuple = (0.0, 0.0, 0.0)  # ZYX, micrometers

    @property
    def label(self):
        name = Path(self.path).name
        return (
            name if self.series_count == 1 else f"{name} [{self.index + 1}] {self.name}"
        )

    @property
    def key(self):
        return self.path, self.index


def channel(index, name=None, color=None, **metadata):
    return dict(
        index=index,
        name=name or f"Channel {index + 1}",
        color=tuple(color) if color is not None else (1.0, 1.0, 1.0),
        **metadata,
    )


def quantity(obj, key, unit):
    value = getattr(obj, key + "_quantity", None)
    if value is None:
        return None
    from pint.errors import DimensionalityError

    try:
        return float(value.to(unit).magnitude)
    except DimensionalityError:
        return None


def imagej_unit_factor(unit, target):
    if not unit:
        return None
    from pint import get_application_registry
    from pint.errors import DimensionalityError, UndefinedUnitError

    try:
        return get_application_registry().Quantity(1, str(unit)).to(target).magnitude
    except (DimensionalityError, UndefinedUnitError, ValueError):
        return None


def ome_info(scene, image):
    pixels = image.pixels
    scene.spacing = tuple(quantity(pixels, f"physical_size_{a}", "um") for a in "zyx")
    for entry, ch in zip(scene.channels, pixels.channels):
        entry["name"] = ch.name or entry["name"]
        if ch.color is not None:
            entry["color"] = tuple(x / 255 for x in ch.color.as_rgb_tuple()[:3])
        for key in ("excitation", "emission"):
            entry[key + "_nm"] = quantity(ch, key + "_wavelength", "nm")
    scene.acquired = (
        image.acquisition_date.isoformat() if image.acquisition_date else None
    )
    step = quantity(pixels, "time_increment", "s")
    times = [i * step if step is not None else None for i in range(scene.shape[0])]
    for plane in pixels.planes:
        fields = {}
        for key, name in (
            ("delta_t", "DeltaT"),
            ("exposure_time", "ExposureTime"),
            ("position_x", "PositionX"),
            ("position_y", "PositionY"),
            ("position_z", "PositionZ"),
        ):
            value = getattr(plane, key)
            if value is not None:
                fields[name] = float(value)
                fields[name + "Unit"] = getattr(plane, key + "_unit").value
        if fields:
            scene.planes.append([plane.the_t, plane.the_c, plane.the_z, fields])
        if plane.the_c == plane.the_z == plane.the_t == 0:
            origin = []
            for axis in "zyx":
                value = (
                    quantity(plane, "position_" + axis, "um")
                    if getattr(plane, "position_" + axis) is not None
                    else None
                )
                origin.append(value or 0.0)
            scene.origin = tuple(origin)
        if plane.the_c == plane.the_z == 0 and plane.delta_t is not None:
            times[plane.the_t] = quantity(plane, "delta_t", "s")
    if all(t is not None for t in times):
        scene.times = times


class Dataset:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.files = {self.path}
        self.resources = ExitStack()

    def __enter__(self):
        try:
            self._open()
        except BaseException:
            self.resources.close()
            raise
        return self

    def __exit__(self, *exc):
        self.resources.close()

    def _open(self):
        path = self.path
        suffix = path.suffix.lower()
        if suffix == ".ims":
            self.handle = self.resources.enter_context(h5py.File(path, "r"))
            self.family = "Imaris"
            self.indices = [0]
        elif suffix in TIFF_EXTENSIONS:
            self.handle = self.resources.enter_context(tifffile.TiffFile(path))
            self.family = (
                "OME-TIFF"
                if self.handle.is_ome
                else "BigTIFF"
                if self.handle.is_bigtiff
                else "TIFF"
            )
            self.indices = list(range(len(self.handle.series)))
            self.files.update(
                Path(f.filehandle.path).resolve() for f in self.handle._files.values()
            )
        elif path.is_dir():
            import zarr

            self.family = "OME-Zarr"
            self.groups = []

            def acquisitions(group):
                attrs = group.attrs.asdict()
                multiscales = attrs.get("ome", attrs).get("multiscales", [])
                if multiscales:
                    self.groups.extend((group.path, i) for i in range(len(multiscales)))
                else:
                    for name, child in sorted(group.groups(), key=lambda item: item[0]):
                        if name != "labels":
                            acquisitions(child)

            acquisitions(zarr.open_group(str(path), mode="r"))
            if not self.groups:
                raise ValueError("No OME-Zarr images found")
            self.indices = list(range(len(self.groups)))
        else:
            from bioio import BioImage
            import bioio_bioformats
            from source.bioformats_runtime import configure_bioformats

            configure_bioformats()
            self.family = "Bio-Formats"
            self.handle = BioImage(str(path), reader=bioio_bioformats.Reader)
            bf = self.handle.reader._bf
            self.resources.callback(bf.destroy)
            with bf.ensure_open():
                self.files.update(Path(p).resolve() for p in bf.used_files())
                self.indices = [
                    i for i in range(len(self.handle.scenes)) if not bf[i].is_thumbnail
                ]

    def select(self, index):
        if index not in self.indices:
            raise ValueError(f"Missing scene {index + 1} in {self.path.name}")
        if self.family == "Imaris":
            scene = self._ims()
        elif self.family in ("TIFF", "BigTIFF", "OME-TIFF"):
            scene = self._tiff(index)
        else:
            scene = self._bioio(index)
        scene.series_count = len(self.indices)
        self.scene = scene
        return scene

    def _ims(self):
        h = self.handle
        self.level = h["DataSet/ResolutionLevel 0"]
        attrs = h["DataSetInfo/Image"].attrs
        spatial = tuple(int(text(attrs[a])) for a in "ZYX")
        origin = tuple(float(text(attrs[f"ExtMin{i}"])) for i in (2, 1, 0))
        spacing = tuple(
            (float(text(attrs[f"ExtMax{i}"])) - low) / size
            for i, low, size in zip((2, 1, 0), origin, spatial)
        )
        channels = []
        for i in range(len(self.level["TimePoint 0"])):
            group = h.get(f"DataSetInfo/Channel {i}")
            meta = group.attrs if group is not None else {}
            rgb = np.fromstring(text(meta.get("Color", "1 1 1")), sep=" ")
            channels.append(
                channel(
                    i,
                    text(meta.get("Name", f"Channel {i + 1}")),
                    rgb if len(rgb) == 3 else None,
                )
            )
            for key, attr in (
                ("excitation_nm", "LSMExcitationWavelength"),
                ("emission_nm", "LSMEmissionWavelength"),
            ):
                if attr in meta:
                    try:
                        channels[-1][key] = float(text(meta[attr]))
                    except ValueError:
                        pass
        dtype = self.level["TimePoint 0/Channel 0/Data"].dtype
        n_t = sum(name.startswith("TimePoint ") for name in self.level)
        scene = Scene(
            str(self.path),
            0,
            self.path.stem,
            self.family,
            (n_t, len(channels), *spatial),
            "uint8" if dtype == np.int8 else dtype.name,
            channels,
            spacing,
            origin=origin,
        )
        scene.acquired = (
            text(attrs["RecordingDate"]) if "RecordingDate" in attrs else None
        )
        timing = h.get("DataSetInfo/TimeInfo")
        if timing is not None:
            stamps = [
                text(timing.attrs.get(f"TimePoint{i + 1}", "")) for i in range(n_t)
            ]
            if all(stamps):
                dates = [
                    datetime.fromisoformat(s.replace("Z", "+00:00")) for s in stamps
                ]
                try:
                    date = (
                        datetime.fromisoformat(scene.acquired)
                        if scene.acquired
                        else dates[0]
                    )
                    # IMS frame stamps can omit the offset supplied by RecordingDate.
                    scene.times = [
                        ((v if v.tzinfo else v.replace(tzinfo=date.tzinfo)) - date).total_seconds()
                        for v in dates
                    ]
                except (ValueError, TypeError):
                    date = dates[0]
                    scene.times = [(v - date).total_seconds() for v in dates]
                scene.acquired = date.isoformat()
        return scene

    def _tiff(self, index):
        tf = self.handle
        series = tf.series[index]
        if tf.is_ome and any(page is None for page in series.pages):
            raise ValueError("OME-TIFF has missing image planes. Check that all companion files are present and readable.")
        axes = series.axes
        if axes in ("QYX", "IYX"):
            axes = "ZYX"
        if any(a not in AXES + "S" for a in axes) or len(set(axes)) != len(axes):
            raise ValueError(f"Unsupported TIFF dimensions {series.axes}")
        self.tiff_axes = axes
        self.tiff_series = series
        self.array = None
        sizes = dict(zip(axes, series.shape))
        self.samples = sizes.get("S", 1)
        sizes["C"] = sizes.get("C", 1) * self.samples
        shape = tuple(sizes.get(a, 1) for a in AXES)
        scene = Scene(
            str(self.path),
            index,
            series.name or self.path.stem,
            self.family,
            shape,
            series.dtype.name,
            [channel(i) for i in range(shape[1])],
        )
        if tf.is_ome:
            from ome_types import from_xml

            ome_info(scene, from_xml(tf.ome_metadata).images[index])
        else:
            shaped = (tf.shaped_metadata[index] or {}) if tf.is_shaped else {}
            meta = shaped.get("deconflow", {})
            for key in ("channels", "spacing", "times", "acquired", "origin", "planes"):
                if key in meta:
                    setattr(scene, key, meta[key])
            if "spacing" not in meta:
                page = series.pages[0]
                unit = page.tags.get("ResolutionUnit")
                um = {2: 25400.0, 3: 10000.0}.get(unit.value if unit else None)
                ij = tf.imagej_metadata or {}
                image_unit = ij.get("unit")
                z_unit = ij.get("zunit", image_unit)
                z_um = imagej_unit_factor(z_unit, "um") if z_unit else um
                values = [abs(ij["spacing"]) * z_um if ij.get("spacing") and z_um else None]
                for axis in "YX":
                    axis_unit = ij.get("yunit", image_unit) if axis == "Y" else image_unit
                    scale = um or imagej_unit_factor(axis_unit, "um")
                    tag = page.tags.get(axis + "Resolution")
                    values.append(
                        scale * tag.value[1] / tag.value[0]
                        if scale and tag and tag.value[0]
                        else None
                    )
                scene.spacing = tuple(values)
                if ij.get("finterval"):
                    seconds = imagej_unit_factor(ij.get("tunit", "s"), "s")
                    if seconds:
                        scene.times = [t * ij["finterval"] * seconds for t in range(shape[0])]
                elif ij.get("fps", 0) > 0:
                    scene.times = [t / ij["fps"] for t in range(shape[0])]
        return scene

    def _bioio(self, index):
        scene_index = index
        group = ""
        if self.family == "OME-Zarr":
            from bioio import BioImage
            import bioio_ome_zarr

            group, scene_index = self.groups[index]
            self.handle = BioImage(str(self.path / group), reader=bioio_ome_zarr.Reader)
        img = self.handle
        img.set_scene(scene_index)
        shape = tuple(getattr(img.dims, a) for a in AXES)
        self.samples = getattr(img.dims, "S", 1)
        shape = (shape[0], shape[1] * self.samples, *shape[2:])
        names = img.channel_names or []
        channels = [
            channel(i, names[i] if i < len(names) else None) for i in range(shape[1])
        ]
        scene = Scene(
            str(self.path),
            index,
            f"{group} / {img.scenes[scene_index]}"
            if group
            else img.scenes[scene_index],
            self.family,
            shape,
            img.dtype.name,
            channels,
        )
        if self.family == "Bio-Formats":
            ome_info(scene, img.reader.ome_metadata.images[index])
        else:
            attrs = img.reader.metadata.attributes
            attrs = attrs.get("ome", attrs)
            ms = attrs["multiscales"][scene_index]
            scene.ngff = attrs.get("version", ms.get("version", "0.4"))
            axes = [dict(name=a) if isinstance(a, str) else a for a in ms["axes"]]
            transforms = [
                *ms["datasets"][0].get("coordinateTransformations", []),
                *ms.get("coordinateTransformations", []),
            ]
            scales = np.ones(len(axes))
            offsets = np.zeros(len(axes))
            for transform in transforms:
                if transform["type"] == "scale":
                    scales *= transform["scale"]
                    offsets *= transform["scale"]
                elif transform["type"] == "translation":
                    offsets += transform["translation"]
            from pint import get_application_registry

            units = get_application_registry()

            def physical(axis, target, values):
                for a, v in zip(axes, values):
                    if a["name"].lower() == axis and a.get("unit"):
                        return float((v * units(a["unit"])).to(target).magnitude)

            scene.spacing = tuple(physical(a, "um", scales) for a in "zyx")
            scene.origin = tuple(physical(a, "um", offsets) or 0.0 for a in "zyx")
            interval = physical("t", "second", scales)
            if interval is not None:
                scene.times = [
                    (physical("t", "second", offsets) or 0.0) + t * interval
                    for t in range(shape[0])
                ]
            for entry, meta in zip(
                channels, attrs.get("omero", {}).get("channels", [])
            ):
                entry["name"] = meta.get("label", entry["name"])
                rgb = meta.get("color", "FFFFFF").lstrip("#")
                entry["color"] = tuple(int(rgb[i : i + 2], 16) / 255 for i in (0, 2, 4))
            extra = img.reader.metadata.attributes.get("deconflow", {})
            for key in ("times", "acquired", "origin", "channels", "spacing", "planes"):
                if key in extra:
                    setattr(scene, key, extra[key])
        return scene

    def volume(self, t, c):
        if self.family == "Imaris":
            dataset = self.level[f"TimePoint {t}/Channel {c}/Data"]
            data = dataset[tuple(slice(0, n) for n in self.scene.shape[2:])]
            if data.dtype == np.int8:
                data = data.view(np.uint8)
        elif self.family in ("TIFF", "BigTIFF", "OME-TIFF"):
            import zarr

            if self.array is None:
                store = self.tiff_series.aszarr(level=0)
                self.resources.callback(store.close)
                self.array = zarr.open_array(store, mode="r")
            selection = {"T": t, "C": c // self.samples, "S": c % self.samples}
            data = self.array[
                tuple(selection.get(a, slice(None)) for a in self.tiff_axes)
            ]
            remaining = [a for a in self.tiff_axes if a not in selection]
            data = data.transpose([remaining.index(a) for a in "ZYX" if a in remaining])
        else:
            args = dict(T=t, C=c // self.samples)
            if "S" in self.handle.dims.order:
                args["S"] = c % self.samples
            data = self.handle.get_image_dask_data("ZYX", **args).compute()
        return np.asarray(data).reshape(self.scene.shape[2:])


def discover(folder, on_error=print):
    from bioio_bioformats.reader_metadata import ReaderMetadata

    supported = {e.strip().lower() for e in ReaderMetadata.get_supported_extensions()}
    seen, groups = set(), []
    for path in sorted(Path(folder).iterdir()):
        if path.name.startswith(".") or path.resolve() in seen:
            continue
        suffix = path.suffix.lower()
        if path.is_dir():
            if suffix != ".zarr" and not any(
                (path / p).is_file() for p in (".zattrs", "zarr.json")
            ):
                continue
        elif suffix not in supported | TIFF_EXTENSIONS | {".ims"}:
            continue
        try:
            with Dataset(path) as ds:
                scenes = []
                for index in ds.indices:
                    try:
                        scenes.append(ds.select(index))
                    except Exception as error:
                        on_error(f"{path.name} scene {index + 1}: {error}")
                if scenes:
                    # Some secondary files also open alone. Prefer the reader
                    # which declares the complete set of constituent files.
                    groups = [
                        (files, items)
                        for files, items in groups
                        if not files < ds.files
                    ]
                    groups.append((ds.files, scenes))
                    seen.update(ds.files)
        except Exception as error:
            on_error(f"{path.name}: {error}")
    return [scene for _, scenes in groups for scene in scenes]
