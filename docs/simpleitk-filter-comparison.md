# SimpleITK filter comparison — 2026-11-11

**SimpleITK is now the default for preview and batch processing**, following user
review and acceptance of this comparison. The evaluated native settings were
adopted without the diagnostic adjustments described below. The full example has
identical median output; Gaussian and bilateral outputs differ. These measurements
do not establish equivalence of downstream scientific measurements.

## Reproduction and artifacts

- Input: `examples/Organ-of-Corti-488CtBP2-568Epsin-647GluR23.ims`.
- SHA-256: `5eca6f6ef0fd301c920eb4efbcd05b5f50c64971d5ced68c8056361226cc0c90`.
- One timepoint; each channel is 131 × 182 × 994 voxels (ZYX), or 23,698,948 voxels.
- Spacing ZYX: 0.30020164896946355, 0.08018543956043957, 0.08016851106639838 µm.
- Apple M3 Pro; macOS 26.7.1; Python 3.13.18; NumPy 2.5.1; SciPy 1.18.1;
  SimpleITK 2.5.7 / ITK 5.4; SimpleITK default 11 threads.
- Complete arrays, TIFFs, figures, saved settings, PSFs, run log, JSON metrics,
  and temporary diagnostic scripts are in `builds/filter-comparison-2026-11-11/`
  (ignored by Git). Original exports and application settings were not overwritten.

Run from the repository with the application's Python environment:

```bash
python scripts/compare_filter_backends.py \
  --settings builds/filter-comparison-2026-11-11/settings.ini \
  --input examples/Organ-of-Corti-488CtBP2-568Epsin-647GluR23.ims \
  --output /tmp/new-filter-comparison
```

The output directory must be new. This utility targets the saved internal-PSF
setup and the first three channels. It retains saved deconvolution settings,
including the GUI's reversed scan-direction convention and PSF input-channel
assignments. Both backends receive the **same saved float32 deconvolution result**
for each channel. Filtering comparisons are therefore isolated from deconvolution
variability. Exact equality refers to image arrays, not TIFF file bytes.

| Channel | Filter | Filter parameters | Deconvolution |
|---|---|---|---|
| C1 | Gaussian | σ = 0.3187783510997201 µm; XY ≈ 3.976 pixels | 60 iterations; regularization 0.0001; Alexa488 |
| C2 | Median | 3×3×3 | 20 iterations; regularization 0.001; Alexa647 |
| C3 | Bilateral | σ = 0.098408 µm approximately; XY ≈ 1.228 pixels; intensity σ = 184.746267254681 | 60 iterations; regularization 0.001; Alexa568 |

Full-precision parameters are in `comparison.json`. Widths are physical standard
deviations, accounting for the slightly unequal X/Y spacing. Gaussian and bilateral
operate independently on XY planes. Median operates in XYZ. No intensity
normalization is performed. Final conversion uses the unchanged `store_result`.

## Results

### Final uint16 pixels

| Filter | Changed voxels | Changed % | Maximum absolute difference | Mean absolute difference | RMSE |
|---|---:|---:|---:|---:|---:|
| Gaussian | 244,256 | 1.030662 | 11 | 0.010525 | 0.105643 |
| Median | 0 | 0 | 0 | 0 | 0 |
| Bilateral | 39,687 | 0.167463 | 9 | 0.001723 | 0.043447 |

### Floating-point filter results

| Filter | Changed voxels | Changed % | Maximum absolute difference | Mean absolute difference | RMSE |
|---|---:|---:|---:|---:|---:|
| Gaussian | 23,698,627 | 99.998646 | 11.328930 | 0.010996 | 0.050257 |
| Median | 0 | 0 | 0 | 0 | 0 |
| Bilateral | 23,614,493 | 99.643634 | 8.849899 | 0.001720 | 0.025609 |

Many floating-point differences are smaller than one output intensity unit;
rounding eliminates most, but not all. These averages include background voxels
and must not be interpreted as errors specifically within biological structures.

### Filter wall time

| Filter | Legacy | SimpleITK | Observed ratio |
|---|---:|---:|---:|
| Gaussian | 0.261 s | 0.232 s | 1.13× |
| Median | 6.031 s | 0.582 s | 10.37× |
| Bilateral | 5.134 s | 0.725 s | 7.08× |

These are single-run observations, including array conversion and per-plane
dispatch but excluding file writes. They are not repeated benchmark estimates;
the small Gaussian difference should not be treated as a reliable speedup.
Deconvolution ran once per channel (17.77, 5.93, 12.47 seconds respectively).
SimpleITK filtering here is CPU-based; ImageJ deconvolution used the GPU.

## Why the images differ

**Gaussian:** SciPy samples a continuous Gaussian with a 4σ truncation and reflected
boundaries. The candidate uses ITK's discrete Gaussian with maximum error 0.01,
maximum kernel width 32, image spacing enabled, and edge-replicating boundaries.
An impulse measurement found radius 10 for the candidate versus radius 16 for
SciPy at the C1 setting. On representative plane Z=65, changing only SciPy's
boundary mode explains differences up to 3.750 units and leaves the interior
unchanged. Tightening the discrete kernel error to 1e-8 (width limit 128) reduces
interior RMSE from 0.019854 to 0.005400, but does not eliminate the difference.
Float64 versus float32 differences are only about 1.24e-6 maximum on this plane.
Thus the differences are principally algorithm/support/boundary choices, not
intensity scaling or floating-point rounding. The candidate keeps the explicit
standard settings; diagnostic changes were not adopted.

**Median:** The selected 3×3×3 neighbourhood matches exactly everywhere. For radius
one, reflection and edge replication provide the same exterior samples. This
does **not** establish equality for larger kernels: a separate random-volume
check with 5×5×5 found boundary differences while the interior matched.

**Bilateral:** ITK uses a range-weight lookup table (100 entries here), ignores
intensity distances ≥4 times intensity σ, and replicates edges. The legacy
implementation evaluates exponentials directly and reflects edges. ITK's spatial
radius is ceil(2.5σ/spacing), versus ceil(3σ/spacing) in the legacy code. Both happen
to be radius 4 for C3, so spatial support is not responsible for this comparison.

On representative plane Z=18, explicitly matching reflection reduces maximum
difference from 7.871 to 1.417 units. Increasing the table to 10,000 entries as
well reduces it to 0.015991 (RMSE 0.000431). A temporary float64 reconstruction
of ITK's documented weighting agrees with the candidate within 2.88e-5 maximum;
the legacy formula agrees with a float64 direct calculation within 0.000254.
The range cutoff has zero measured effect on that representative plane. This
supports boundary treatment and lookup quantization as the main causes here.
Neither diagnostic adjustment is enabled in the candidate.

### Boundary versus interior, final uint16

Conservative boundary bands are 18 XY pixels for Gaussian, one voxel on every
axis for median, and seven XY pixels for bilateral. Z edges are not excluded for
the plane-wise filters.

| Filter | Interior changed % | Interior maximum difference | Boundary changed % | Boundary maximum difference |
|---|---:|---:|---:|---:|
| Gaussian | 0.525015 | 5 | 2.753951 | 11 |
| Median | 0 | 0 | 0 | 0 |
| Bilateral | 0.052977 | 2 | 1.326118 | 9 |

The largest discrepancies cluster near boundaries, but both smoothing filters
also change interior pixels. The JSON contains counts, MAE, RMSE, and signed means
for each region in both output representations.

## Validation and implementation status

Temporary checks covered constants, impulses, edges, unequal XY spacing (transpose
consistency), and absence of cross-Z smoothing for Gaussian/bilateral. The actual
example comparison completed successfully. No repository tests or README changes
were added. SimpleITK is the default in `post_process(...)`. The previous backend
is retained through explicit `backend='legacy'` for reproducible comparisons;
the comparison utility explicitly selects each backend. Unknown backends raise an error.

SimpleITK 2.5.7 is pinned in requirements and installed in the local `deconflow`
environment. Published CPython 3.11+ ABI-compatible wheels cover Python 3.13 on
macOS ARM64, macOS x86-64, and Windows x86-64. Only macOS ARM64 was executed here;
packaged Windows/Intel application builds were not performed.

The slice figures use matching image scales and separately labelled absolute
difference scales. Slice selection is based on signal, not largest discrepancy.
They are useful for inspection, not proof of scientific equivalence. Native
SimpleITK behaviour was selected after review; closer boundary and approximation
matching was not adopted.

## References

- [SimpleITK discrete Gaussian API](https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1DiscreteGaussianImageFilter.html)
- [ITK 5.4 bilateral implementation](https://github.com/InsightSoftwareConsortium/ITK/blob/v5.4.0/Modules/Filtering/ImageFeature/include/itkBilateralImageFilter.hxx)
- [SimpleITK 2.5.7 wheels and citation guidance](https://pypi.org/project/simpleitk/2.5.7/)
- Lowekamp et al., [The Design of SimpleITK](https://doi.org/10.3389/fninf.2013.00045), 2013.
- Tomasi and Manduchi, [Bilateral Filtering for Gray and Color Images](https://doi.org/10.1109/ICCV.1998.710815), 1998.
