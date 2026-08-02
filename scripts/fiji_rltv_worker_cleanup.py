#@ String input_path
#@ String psf_path
#@ String output_path
#@ Integer iterations
#@ Double regularization
#@ Boolean non_circulant
#@ String device_name
#@ IOService io
#@ OpService ops
#@ DatasetService ds

"""Headless Fiji Jython worker for one 3D CLIJ2-FFT RL-TV deconvolution."""

from java.lang import System
from net.haesleinhuepf.clij2 import CLIJ2
from net.haesleinhuepf.clij.coremem.enums import NativeTypeEnum
from net.haesleinhuepf.clijx.plugins import DeconvolveRichardsonLucyFFT
from net.haesleinhuepf.clijx.plugins import clij2fftWrapper


def open_float32(path):
    dataset = io.open(path)
    return ops.convert().float32(dataset.getImgPlus())


print("Opening input: " + input_path)
print("Opening PSF: " + psf_path)

image = open_float32(input_path)
psf = open_float32(psf_path)

if image.numDimensions() != 3:
    raise ValueError("Input must be 3D; dimensions=" + str(image.numDimensions()))
if psf.numDimensions() != 3:
    raise ValueError("PSF must be 3D; dimensions=" + str(psf.numDimensions()))

if device_name is not None and len(device_name.strip()) > 0:
    clij2 = CLIJ2.getInstance(device_name)
else:
    clij2 = CLIJ2.getInstance()

clij2.clear()
print("Using GPU: " + clij2.getGPUName())
print("Fiji working directory: " + System.getProperty("user.dir"))

gpu_image = None
gpu_psf = None
gpu_output = None

try:
    gpu_image = clij2.push(image)
    gpu_psf = clij2.push(psf)
    gpu_output = clij2.create(gpu_image.getDimensions(), NativeTypeEnum.Float)

    print(
        "Running RL-TV: iterations=" + str(iterations)
        + ", regularization=" + str(regularization)
        + ", non_circulant=" + str(non_circulant)
    )

    started = System.currentTimeMillis()

    success = DeconvolveRichardsonLucyFFT.deconvolveRichardsonLucyFFT(
        clij2,
        gpu_image,
        gpu_psf,
        gpu_output,
        int(iterations),
        float(regularization),
        bool(non_circulant),
    )

    if success is False:
        raise RuntimeError("CLIJ2-FFT reported deconvolution failure.")

    deconvolved = clij2.pullRAI(gpu_output)
    output_dataset = ds.create(deconvolved)
    io.save(output_dataset, output_path)

    elapsed = (System.currentTimeMillis() - started) / 1000.0
    print("Saved float32 result: " + output_path)
    print("Elapsed seconds: " + str(elapsed))

finally:
    if gpu_output is not None:
        gpu_output.close()
    if gpu_psf is not None:
        gpu_psf.close()
    if gpu_image is not None:
        gpu_image.close()
    clij2.clear()

# Release clFFT's process-global plans/resources before forcing the headless
# Fiji JVM to exit. Each worker handles exactly one channel, and workers run
# serially, so this cleanup is not concurrent with another native FFT call.
cleanup_status = clij2fftWrapper.cleanup()
print("CLIJ2-FFT cleanup status: " + str(cleanup_status))

# Fiji can leave non-daemon ImageJ/SciJava threads alive after a successful
# headless --run invocation. The result has already been completely written,
# so explicitly terminate this one-job Fiji subprocess after native cleanup.
print("FIJI_RLTV_WORKER_DONE")
System.out.flush()
System.err.flush()
System.exit(0)
