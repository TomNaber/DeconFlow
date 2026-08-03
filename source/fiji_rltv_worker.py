#@ String input_path
#@ String psf_path
#@ String output_path
#@ Integer iterations
#@ Double regularization
#@ Boolean non_circulant
#@ IOService io
#@ OpService ops
#@ DatasetService ds

from java.lang import System
from net.haesleinhuepf.clij2 import CLIJ2
from net.haesleinhuepf.clij.coremem.enums import NativeTypeEnum
from net.haesleinhuepf.clijx.plugins import DeconvolveRichardsonLucyFFT
from net.haesleinhuepf.clijx.plugins import clij2fftWrapper


def open_float32(path):
    return ops.convert().float32(io.open(path).getImgPlus())


print("Opening input: " + input_path)
print("Opening PSF: " + psf_path)
image = open_float32(input_path)
psf = open_float32(psf_path)
clij2 = CLIJ2.getInstance()
clij2.clear()
print("Using GPU: " + clij2.getGPUName())

gpu_image = gpu_psf = gpu_output = None
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
    DeconvolveRichardsonLucyFFT.deconvolveRichardsonLucyFFT(
        clij2,
        gpu_image,
        gpu_psf,
        gpu_output,
        int(iterations),
        float(regularization),
        bool(non_circulant),
    )
    io.save(ds.create(clij2.pullRAI(gpu_output)), output_path)
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

print("CLIJ2-FFT cleanup status: " + str(clij2fftWrapper.cleanup()))
print("FIJI_RLTV_WORKER_DONE")
System.out.flush()
System.err.flush()
System.exit(0)
