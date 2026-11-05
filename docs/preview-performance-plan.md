# Preview performance investigation and implementation plan

Investigated 4 November 2026. This is a plan, not a change to the application or its dependencies. No tests were created or modified.

## Recommendation

Keep one headless Fiji process alive across preview channels and requests, initialize it asynchronously when a valid Fiji executable is selected, and receive a result-complete message without waiting for JVM exit. Then cache prepared crop TIFFs and normalized PSFs. Keep the current CLIJ2-FFT invocation, parameters, full PSFs, input preprocessing, and output conversion unchanged.

Do not make whole-volume TIFF export or Fiji-side cropping the first optimization: the source image is already retained in Python RAM and the measured small-crop TIFF write cost is negligible compared with process lifecycle overhead.

## What the code currently does

- `source/preview.py:load` creates a Python worker for the selected IMS file. `source/preview_worker.py:load_volume` reads every channel at resolution 0 / time point 0 and computes original projections. The worker retains these source volumes between requests.
- `deconvolve_volume` reloads and normalizes PSFs on every request, extracts each requested crop, makes a contiguous float32 copy, sanitizes non-finite values, and clamps negatives.
- `source/batch_deconvolve_ims.py:run_fiji` writes an input TIFF and PSF TIFF, copies the TV kernel, launches Fiji, drains stdout, waits for process exit, reads the output TIFF, and removes the temporary directory. This happens separately for every channel in both preview and batch processing.
- `source/fiji_rltv_worker.py` loads TIFFs through ImageJ services, initializes CLIJ2, pushes buffers, calls `DeconvolveRichardsonLucyFFT.deconvolveRichardsonLucyFFT`, pulls and saves the result, releases buffers, cleans up FFT state, and calls `System.exit(0)`.
- The Python worker rounds/clips to uint16, computes XY/YZ maximum projections, and sends projections to the GUI only after all selected channels finish. Full outputs stay in the worker; they are not transferred to Tk.
- The GUI polls every 50 ms and recolors/renders projections. Automatic preview has a deliberate 500 ms debounce. Changes during computation invalidate the displayed result but do not stop the remaining channels in the obsolete request.
- Existing `Elapsed seconds` starts before the CLIJ2 call and ends after GPU pull and TIFF save. It is not a pure deconvolution measurement. The Python total also excludes GUI rendering and some preprocessing.

## Baseline measurement

A temporary profiling harness copied the existing Fiji script and inserted timestamps around stages; the published packages and numerical call were not edited. Benchmark files are under `/tmp/preview-fiji-profile`; logs are `/tmp/preview-fiji-profile.log` and `/tmp/preview-fiji-profile-repeat.log`. These temporary artifacts are not application dependencies.

Environment: Apple M3 Pro GPU; installed Fiji executable from the application defaults; 12 GB heap; 40 iterations; regularization 0.0001; non-circulant enabled. Source: `examples/3-1-3_Part-1_8-5-25_a_0027-A1-23__Crop_Stack-crop_stack_1.ims`, channel 0. Crop Z `[40,104)`, Y `[64,182)`, X `[400,528)` (64 × 118 × 128). Full Alexa488 PSF: 209 × 119 × 119. All three source channels were loaded, as in the current preview worker. The initial profiling slice requested Y up to 192, but NumPy clipped it to the metadata-defined image height of 182; the actual crop dimensions above correct that earlier report.

First successful launch in this investigation:

| Stage | Seconds |
|---|---:|
| Source load and original projections, once per file | 0.214 |
| Crop preparation plus PSF read/normalization | 0.008 |
| Input and PSF TIFF writes | 0.019 |
| Fiji launch to script entry | 5.762 |
| Script imports | 0.117 |
| Input and PSF open/float conversion | 0.280 |
| CLIJ2 initialization/clear | 0.099 |
| GPU pushes and output allocation | 0.057 |
| Full CLIJ2 deconvolution call | 6.180 |
| GPU result pull | 0.002 |
| TIFF result save | 0.434 |
| Buffer/FFT cleanup | 0.001 |
| Completion marker to process exit | 5.470 |
| Python result read, uint16 conversion, projections | 0.009 |

Total from TIFF preparation through result projections: **18.43 s**, approximately **34% in the CLIJ2 call and 66% elsewhere**. This excludes the once-per-file load, prep, Python module import, IPC, and GUI drawing. Launch plus exit wait alone was **11.23 s (61%)**. These are observed stage durations, not a promised persistent-worker speedup.

Two corrected subsequent launches completed successfully with filesystem/OpenCL caches already exercised. Each still launched a separate JVM, exactly as the current application does:

| Stage | Repeat 1 | Repeat 2 |
|---|---:|---:|
| Fiji launch to script entry | 3.759 s | 3.396 s |
| Script entry through GPU push/allocation | 0.703 s | 0.435 s |
| Full CLIJ2 deconvolution call | 2.231 s | 2.210 s |
| Pull, save, and cleanup | 0.485 s | 0.373 s |
| Completion marker to process exit | 5.446 s | 5.445 s |
| TIFF writes + Python output read/conversion/projections | 0.012 s | 0.017 s |
| Total TIFF preparation through result projections | **12.634 s** | **11.875 s** |
| Full CLIJ2 call / remaining work | **17.7% / 82.3%** | **18.6% / 81.4%** |

The repeated-launch startup-plus-exit budget was **8.84–9.21 s**. The persistent-worker implementation has not been benchmarked, so these are opportunities rather than measured savings. GUI rendering remains unmeasured.

The full plugin call includes its internal normalization, padding, FFT setup, and computation. Its internal “Richardson Lucy Started/Finished” messages bracketed about 1.9 s on this first run; that is not a GPU-profiler measurement and should not be substituted for the full 6.18 s algorithm call. Preserve all internal algorithm work.

Limitations: a small single-channel crop, few runs, warm filesystem/OpenCL caches possible, stdout receipt timing for launch/exit boundaries. Java internal durations use a monotonic clock. GUI click-to-visible latency was not measured in this headless harness. A preliminary sandboxed launch failed; one subsequent repeat collided with the preceding output file and was discarded and stopped. The corrected repeat removes its own prior output before each run.

## Implementation sequence

### 1. Add stage timings and a stable baseline

Add request ID, file generation, channel ID, shape, and monotonic durations to structured worker messages. Record:

- User action / debounce / queue time; file load and projections separately.
- Crop preparation, PSF load, TIFF writes, launch-to-ready.
- Fiji TIFF opens, GPU initialization, pushes/allocation, full algorithm call, pull, save, cleanup.
- Python output read/conversion/projection, IPC receive, and GUI draw completion (plus a next-idle timestamp as a display proxy).

Use local clock deltas inside each process, not subtraction of unrelated Java/Python absolute clocks. Include residual/unattributed time and avoid double counting overlapped preparation. Report time to first channel and time to all selected channels. Capture the same representative small, medium, and large crops, first use and repeated use, before and after each phase. Use at least five repeats for medians/ranges. No timing thresholds or projected speedups should be treated as measured until this is done.

### 2. Persistent Fiji session: highest priority

Introduce a session manager (e.g. `source/fiji_session.py`) owned by the GUI/application lifecycle, independent of the per-file Python preview process. Provide the same abstraction to the batch worker so a batch uses one process across its channels/time points. Initially serialize all GPU jobs; never run preview and batch workers on the same GPU concurrently.

Run the existing Jython worker as a command loop, keeping ImageJ services and the CLIJ2 context alive. A private session directory with atomic JSON request/response files is a simple dependency-free transport; keep logs separate from protocol messages. Responses must include request ID, channel, status, timings, and validated output path. Use unique output names because ImageJ refuses to overwrite an existing TIFF. A result is ready only after save and per-job cleanup succeed, not merely when a file appears.

Keep the exact numerical call and per-job buffer ownership. Initially retain existing per-job FFT cleanup; measure its cost before considering changes. Remove per-job `System.exit`; exit only on session shutdown. Keep `clij2.clear()` only where compatible with ownership: phase 2 can retain it between jobs, but a later GPU buffer cache cannot survive it. Do not claim that persistence removes the plugin's internal FFT planning or padding.

On executable selection (including the initially populated default), debounce path edits and validate existence/executability in the background. Log `Initializing Fiji…`, then `Fiji ready — <GPU>` only after service imports and GPU initialization succeed. Distinguish `No Fiji detected at path`, `Fiji launch failed`, and `CLIJ2/GPU initialization failed`. Path changes and heap-limit changes restart the session; crop, iterations, regularization, and PSF changes do not.

Use bounded startup/job timeouts, health checks, and process ownership. On cancellation during a native call, terminate only the owned session process group and reinitialize; cancellation between channels can simply discard remaining work. Reject stale responses by file generation and revision. File changes should release file caches without unnecessarily restarting Fiji. App close must stop the session and remove its private directory. Preserve an explicit one-shot fallback on session failure, with a logged reason and no silent duplicate jobs.

Readiness preloads the JVM/services/context; first native-call initialization may remain. Add an optional tiny warm-up only if measurements justify it, using the same plugin and reliable cleanup. Do not report that warming a tiny shape preplans all other FFT shapes.

### 3. Prepare and cache crops, then avoid redundant channel work

After file loading, prepare selected channels in the background without delaying original-image display. When crop dragging ends, debounce and prepare the latest crop; discard obsolete preparations. Existing in-RAM source arrays remain the source of truth. Write input TIFFs once per file-generation/channel/crop/preprocessing key. Iterations and regularization changes reuse these files.

Cache normalized PSFs and their staged TIFFs by file identity/content fingerprint. Preserve current float32 sanitization and float64-sum normalization exactly. Do not change PSF generation. Copy the TV kernel once per session. Put reusable files in session/file cache directories rather than the current per-request directory, which is deleted immediately after each job. Bound disk/RAM usage; never evict a file still referenced by a job.

Cache successful results per channel using source identity, crop bounds, PSF fingerprint, iterations, regularization, non-circulant flag, and algorithm/runtime identity. Current automatic-preview skipping is all-or-nothing across selected channels: make it per-channel so changing one PSF or adding one channel does not rerun unchanged channels. Keep an explicit manual recompute path. Coloring, contrast, and display zoom remain render-only changes.

### 4. Return each channel as it completes

Send each channel's uint16 projections immediately and update the relevant composite while remaining channels run. Keep status explicit (`1/3 channels ready`) and never mix channels from different crops or incompatible revisions. Existing merging of channel projections needs an explicit compatibility check. Check for a superseding request between channels so obsolete work does not consume the rest of the queue.

This improves time to first visible result without changing the algorithm or total required work. Preserve screenshot metadata and behavior for partially completed/current results.

### 5. Consider Fiji-side RAM caching only after the preceding measurements

If TIFF open/transfer time becomes material, keep prepared crop images and normalized PSFs in Fiji CPU RAM with bounded eviction. This removes repeated TIFF reads while retaining TIFF as the validated transport. GPU caching is a later option because it needs strict ownership and changes to global `clear()` behavior; reserve room for the plugin's padded working buffers, which can substantially exceed crop size.

Whole-file TIFF export plus Fiji-native crop is optional for repeated large crops only. It duplicates the Python source in JVM RAM and writes float32 full volumes even when the selected ROI is tiny. Compare its one-time staging cost and memory footprint against cumulative crop staging savings before enabling it. Preserve ZYX↔XYZ mapping, exclusive crop endpoints, signed-int8 reinterpretation, sanitization, and contiguous zero-origin crop behavior. Do not deconvolve the full image then crop the result: that changes boundary behavior.

Raw shared memory/memory mapping or a new Python–Java bridge is not justified by the measured millisecond TIFF write cost and is outside the initial implementation.

## Preserve correctness and published packages

Leave requirements, Fiji plugins, native FFT libraries, PSF generation, full PSF support, iteration count, regularization, boundary handling, float32 inputs, and uint16 output rounding/clipping unchanged. Do not replace CLIJ2, change precision, downsample, truncate PSFs, tile differently, or alter the plugin implementation.

Installed package fingerprints captured during investigation:

| Artifact | SHA-256 |
|---|---|
| `clij2-fft_-2.2.0.22.jar` | `ffababec503e6f5a8ae209156e115dff3e41604ea7eb0a991a5c0abff7ee1f01` |
| `clij2_-2.5.3.5.jar` | `e62351f3be262abfd77f68b7629b553b5d184006c7a16aaa2f66b3954cca7b0a` |
| `clij_-1.9.0.1.jar` | `8212a1bbbc4f199fd17615f351b7baa92fa54a841283d6f8b3c67743fbab4cb4` |
| `source/lib/totalvariationterm.cl` | `52e870d5cc2acfdee03cd4c0647684d66cabe3087b2266bdca17a4c25e83abab` |

Before implementation, also inventory the Fiji launcher/JVM, relevant transitive jars, and native libraries; these four hashes are not a complete installation lock.

Validate through temporary/manual comparison runs, not new or modified project tests: compare staged arrays, full float outputs, uint16 outputs, projections, and crop positioning against the current one-shot path. Compare repeated baseline runs to characterize GPU nondeterminism; investigate numerical differences rather than silently weakening equivalence. Cover multiple channels, repeated requests, changed crop/PSF/parameters, stale work, cancellation, invalid paths, worker crash, and app close. Confirm bounded memory over many requests and that no Fiji processes remain after shutdown. Recheck batch output behavior.

The upstream [CLIJ2-FFT source](https://github.com/clij/clij2-fft/blob/master/src/main/java/net/haesleinhuepf/clijx/plugins/DeconvolveRichardsonLucyFFT.java) supports treating the plugin as an opaque numerical call with internal normalization/padding/FFT work. Upstream master is context, not proof of the installed version; implementation must retain and inspect the installed artifacts above.

## Acceptance and expected benefit

Accept phase 2 when warm previews launch no new JVM, do not wait for JVM exit, preserve outputs, and keep cancellation/cleanup functional. Accept phase 3 when repeated settings-only requests reuse prepared inputs and changing one channel does not recompute unchanged ones. Confirm improvements in end-to-end GUI timing, not just the algorithm log.

For the first measured run, eliminating launch and exit wait alone removes an observed 11.23 s from an 18.43 s path in principle. This is an overhead budget, not a forecast: persistent-session timing, first-call effects, transport, and GUI behavior still require measurement. Prioritize the persistent process over whole-file TIFF conversion.
