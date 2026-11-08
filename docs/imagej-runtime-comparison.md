# Fiji versus managed ImageJ — 8 November 2026

Compared on this laptop (Apple M3 Pro), using the user’s `batch_deconvolution`
Python environment. Both runtimes used the same worker, OpenCL GPU, custom TV
kernel, float32 input TIFFs, and normalized PSFs. External Fiji was untouched.

Input: the repository example IMS, all three full channels, XYZ **994 × 182 × 131**
(23,698,948 voxels per channel). PSFs: ZYX **80 × 40 × 40**, generated once from
the saved optical settings and reused by both runtimes. Non-circulant correction
was disabled, matching the saved preference. JVM memory was 12 GiB.

## Output equality

Every managed-runtime result matched the corresponding Fiji result **bit-for-bit**:
three channels × three repetitions = nine comparisons. There were zero differing
float32 voxels, maximum absolute error 0, and RMSE 0. Rounded/clipped uint16 export
arrays were also identical. Fiji’s repeated outputs were identical to each other.
This establishes unchanged output for this dataset and these settings; it is not
a claim about every possible image or GPU.

## Timing

One persistent session per runtime; Fiji ran first, then managed ImageJ. Each
channel ran three times. Warm figures below are the median of repetitions 2–3.
Job time includes TIFF reading/writing, GPU transfer, and deconvolution; it excludes
session startup, PSF generation, IMS input conversion, and final IMS export.

| Channel | Iterations / regularization | Fiji deconvolution | ImageJ deconvolution | Fiji full job | ImageJ full job |
| --- | --- | --- | --- | --- | --- |
| CtBP2 | 60 / 0.001 | 9.72 s | 9.73 s | 11.70 s | 11.63 s |
| Myo7A | 15 / 0.0001 | 2.82 s | 2.71 s | 4.50 s | 4.51 s |
| GluR23 | 20 / 0.001 | 3.54 s | 3.50 s | 5.39 s | 5.30 s |

Time from session construction to GPU readiness: Fiji **6.94 s**,
managed ImageJ **2.45 s** (one measurement each).

The first channel’s first GPU call took 11.65 s in Fiji versus 11.23 s in ImageJ;
its full job took 14.28 s versus 13.50 s. Startup and initial FFT compilation are
therefore distinct from steady-state deconvolution cost.

Steady-state deconvolution is effectively unchanged. Differences of roughly
0–4% in these short repeated measurements are not sufficient to claim a consistent
algorithm speedup. This was a sequential, warm-system benchmark, not a randomized
cold-boot experiment; other laptop activity and thermal state can affect timings.

Raw timings, settings, dependency hashes, and comparisons are in the local
`builds/verification/runtime-comparison/results.json`. Baseline and managed
first-run TIFFs and generated PSFs are retained alongside it. No project tests
or test directories were created.
