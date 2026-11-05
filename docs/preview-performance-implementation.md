# Persistent Fiji implementation

Implemented 4 November 2026. The preceding investigation is in `preview-performance-plan.md`.

## Changes

- `source/fiji_session.py` owns a headless Fiji process, starts it without waiting on the Tk thread, reports readiness/GPU identity, drains logs, and exposes serialized requests through a private temporary directory. Request IDs and atomic JSON responses distinguish completion from process exit. Startup timeout: 120 seconds; conservative full-volume job/queue timeout: 24 hours.
- `source/fiji_session_worker.py` keeps ImageJ services and CLIJ2 alive. It runs the same `DeconvolveRichardsonLucyFFT.deconvolveRichardsonLucyFFT` call with the same float32 inputs, arguments, buffer release, and FFT cleanup. It returns after saving and cleaning up each job instead of exiting the JVM. An owner watchdog prevents an abandoned session surviving its application.
- Fiji initializes after a 500 ms debounce when the executable or memory setting changes, including the saved/default executable at app startup. Invalid paths and startup failures appear in the console. File reloads retain an idle session. Cancellation stops the owned Fiji process group; later requests restart it. Batch processing closes the preview session before starting its own persistent session, preventing simultaneous preview/batch GPU jobs.
- The preview worker retains the existing in-RAM IMS volumes and prepares cropped TIFFs and normalized PSFs after a 250 ms debounce when settings are valid. Caches live in the system temporary folder and hold at most one crop/PSF per selected channel. File reload/close removes them. PSF file identity, size, mtime, and ctime invalidate cached PSFs/results. Preparation does not block Tk or require Fiji readiness.
- Automatic preview caches projections by channel and processing settings; an explicit **Deconvolve preview** always recomputes. Parameters-only changes reuse staged inputs. Results are sent and drawn one channel at a time. Each response contains compatible projections from one request; stale generations/revisions are ignored. An atomic revision marker allows obsolete work to stop between channels.
- Timing logs cover source load/projections, crop/PSF preparation, readiness/queue wait, TIFF staging, Fiji TIFF opening, GPU push, the full deconvolution call, GPU pull, TIFF save, cleanup, Python read/conversion/projection, and request-to-channel-draw time. Drawing time is a Tk rendering proxy, not a monitor presentation measurement.
- Batch `run_fiji` reuses a session across channels/time points. The original implementation remains as `run_fiji_one_shot` for comparisons. There is no silent fallback that could duplicate GPU work.

The published packages, dependencies, original one-shot script, TV kernel, PSF generator, numerical parameters, crop preprocessing, and uint16 rounding/clipping were not changed. No tests or test folders were created or modified.

## Verification and measured improvement

Temporary, non-repository verification scripts ran against the installed Apple M3 Pro/Fiji stack. The actual sample crop was 64 × 118 × 128 voxels with the full 209 × 119 × 119 Alexa488 PSF, 40 iterations, regularization 0.0001, and non-circulant boundaries enabled.

| Comparison | Result |
|---|---|
| One-shot output vs first persistent result | Float32 bit-for-bit equal; max absolute difference 0; uint16 bit-for-bit equal |
| One-shot output vs repeated persistent result | Float32 bit-for-bit equal; max absolute difference 0; uint16 bit-for-bit equal |
| One-shot wall time in comparison run | Approximately 14 s |
| First job after session readiness | 4.357 s |
| Repeated persistent job | 2.619 s, including 1.984 s inside the algorithm call |
| Earlier repeated one-shot baseline | 11.875–12.634 s |

The repeated persistent path is about **78–79% faster** than the earlier repeated one-shot baseline for this sample, excluding one-time session startup. Session startup in the comparison took 6.93 s and can overlap file loading and crop selection. These are a few local runs, not broad performance guarantees; larger volumes can remain dominated by CLIJ2.

Additional completed checks:

- Two-channel processing emitted channel counts `[1, 2]`; repeating unchanged work reused both cached results with identical projections.
- Changing regularization recomputed both channels while reusing staged inputs.
- A real Tk preview window rendered all four original/deconvolved views and showed the first channel before completion of the second.
- Changing only one selected PSF reused the other channel's result in automatic preview.
- Reloading a file while idle retained the same Fiji PID.
- Cancellation released the preview worker and owned Fiji session; invalid executable paths were rejected.
- A missing-input request returned its error without hanging. Killing the owned Fiji process was detected, and a new session reached readiness successfully.
- Python syntax compilation and `git diff --check` passed.

Full-volume batch file writing was not repeated: the writer is unchanged. Verification exercises the shared persistent numerical path, but does not establish bitwise equivalence for every image/GPU/settings combination. Full-volume Fiji RAM caching, shared memory, GPU buffer caching, and algorithm-internal FFT changes remain deliberately unimplemented; measured TIFF staging cost did not justify them.

## Follow-up: crop-dependent stalls and logging (5 November 2026)

Removed all routine `Preview input preparation` messages, including cache-hit messages, and suppressed blank Fiji log lines.

The reported 35.720 s was inside the plugin call, versus 2.613 s for the earlier crop. Input preparation, opening, transfers, and output saving did not account for the spike. The previous wrapper called `clij2fftWrapper.cleanup()` after every channel. Inspection of the installed 2.2.0.22 macOS arm64 native library confirms that this function calls `clfftTeardown()`. The native low-pointer convolution/deconvolution paths destroy their own plans; their compiled-program resources need not be torn down after every successful channel.

The worker now retains clFFT resources across successful calls, with a limit of eight FFT geometry/mode keys. Before admitting a ninth key it clears this cache; it also clears on a failed job and normal worker-loop termination. Forced process termination releases the process's resources. Image buffers are still closed and CLIJ2 cleared on every job. No library, algorithm, numerical argument, crop extent, or padding rule was changed.

When crop/PSF dimensions change, a single diagnostic reports input XYZ, PSF XYZ, and the padded FFT XYZ dimensions, using the installed ImageJ FFT-size operation. It distinguishes a previously used FFT geometry from a first use. JVM garbage-collection time is reported only if it exceeds 0.1 s during the algorithm call. This helps investigate future spikes without printing repeated preparation/cache-hit timings.

Verification used temporary scripts outside the repository, not added project tests:

- Four runs at XYZ 128×118×64 and 129×118×64 were bit-for-bit identical to the previous worker's float32 outputs. Both crops pad to FFT XYZ 250×240×280.
- A temporary one-key cache exercised eviction by switching input width 128 → 141 → 128, which changes padded FFT width 250 → 270 → 250.
- On first use of width 270, about 4.3 s elapsed before the native `Richardson Lucy Started` message, followed by about 2.0 s to its finish message. These log boundaries demonstrate substantial setup outside the iteration loop; they do not separately measure compilation versus other internal setup.
- The supplied 35 s stall was not reproduced. First-use FFT/OpenCL setup is the leading explanation, not a proven attribution of that exact run. Identical dimensions with different crop position should not need different FFT kernels. A crop resize can cross a padded-size boundary even if the visual change is small.

Retaining resources prevents avoidable teardown between channels, but is not a claim that a never-before-used FFT size becomes free. The plugin still creates/bakes plans internally. Background warm-up would shift first-use work earlier, not remove it; changing crop/padding/PSF sizes to avoid new kernels could change results and was not done.

Reference: the [clFFT documentation](https://clmathlibraries.github.io/clFFT/) explains plan baking, OpenCL kernel compilation, and teardown.
