#@ String session_dir
#@ IOService io
#@ OpService ops
#@ DatasetService ds

# Jython 2.7: keep syntax compatible with the Fiji installation.
import json
import os
import time
import traceback
import threading
from java.lang import System, ProcessHandle


def publish(name, value):
    path = os.path.join(session_dir, name)
    with open(path + '.tmp', 'w') as handle:
        json.dump(value, handle)
    os.rename(path + '.tmp', path)


def clock():
    return System.nanoTime() / 1.e9


try:
    from net.haesleinhuepf.clij2 import CLIJ2
    from net.haesleinhuepf.clij.coremem.enums import NativeTypeEnum
    from net.haesleinhuepf.clijx.plugins import DeconvolveRichardsonLucyFFT
    from net.haesleinhuepf.clijx.plugins import clij2fftWrapper
    clij2 = CLIJ2.getInstance()
    clij2.clear()
    publish('ready.json', {'gpu': str(clij2.getGPUName())})
except:
    publish('failed.json', {'error': traceback.format_exc()})
    System.exit(1)


def deconvolve(request):
    timings = {}
    start = clock()
    image = ops.convert().float32(io.open(request['input_path']).getImgPlus())
    psf = ops.convert().float32(io.open(request['psf_path']).getImgPlus())
    timings['open'] = clock() - start
    gpu_image = gpu_psf = gpu_output = None
    try:
        start = clock()
        clij2.clear()
        gpu_image = clij2.push(image)
        gpu_psf = clij2.push(psf)
        gpu_output = clij2.create(gpu_image.getDimensions(), NativeTypeEnum.Float)
        timings['push'] = clock() - start
        start = clock()
        DeconvolveRichardsonLucyFFT.deconvolveRichardsonLucyFFT(
            clij2, gpu_image, gpu_psf, gpu_output,
            int(request['iterations']), float(request['regularization']), bool(request['non_circulant']),
        )
        timings['deconvolution'] = clock() - start
        start = clock()
        result = clij2.pullRAI(gpu_output)
        timings['pull'] = clock() - start
        start = clock()
        io.save(ds.create(result), request['output_path'])
        timings['save'] = clock() - start
    finally:
        start = clock()
        if gpu_output is not None:
            gpu_output.close()
        if gpu_psf is not None:
            gpu_psf.close()
        if gpu_image is not None:
            gpu_image.close()
        clij2.clear()
        status = clij2fftWrapper.cleanup()
        if status != 0:
            raise RuntimeError('CLIJ2-FFT cleanup failed: ' + str(status))
        timings['cleanup'] = clock() - start
    return timings


def watch_owner():
    with open(os.path.join(session_dir, 'owner.json')) as handle:
        owner = json.load(handle)['pid']
    while True:
        process = ProcessHandle.of(long(owner))
        if not os.path.isdir(session_dir) or not process.isPresent() or not process.get().isAlive():
            System.exit(0)
        time.sleep(1)


watcher = threading.Thread(target=watch_owner)
watcher.daemon = True
watcher.start()
request_path = os.path.join(session_dir, 'request.json')
while os.path.isdir(session_dir):
    if not os.path.isfile(request_path):
        time.sleep(0.025)
        continue
    try:
        with open(request_path) as handle:
            request = json.load(handle)
        os.remove(request_path)
        try:
            response = {'id': request['id'], 'timings': deconvolve(request)}
        except:
            response = {'id': request['id'], 'error': traceback.format_exc()}
        publish('response-' + request['id'] + '.json', response)
    except:
        if os.path.isdir(session_dir):
            publish('failed.json', {'error': traceback.format_exc()})
        break
System.exit(0)
