#@ String session_dir
#@ IOService io
#@ OpService ops
#@ DatasetService ds

# Jython 2.7: keep syntax compatible with the managed ImageJ runtime.
import json
import os
import time
import traceback
import threading
from java.lang import System, ProcessHandle
from java.lang.management import ManagementFactory
from net.imglib2 import FinalDimensions
from jarray import array


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
    publish('ready.json', {'gpu': str(clij2.getGPUName()), 'java_version': str(System.getProperty('java.version')),
                           'java_vendor': str(System.getProperty('java.vendor'))})
except:
    publish('failed.json', {'error': 'ImageJ GPU initialization failed. Check the OpenCL driver and app runtime.\n' + traceback.format_exc()})
    System.exit(1)


# Keep compiled clFFT programs between jobs, while bounding the number of
# geometries retained. Per-call FFT plans and image buffers are still released.
fft_geometries = set()
MAX_FFT_GEOMETRIES = 8


def reset_fft_cache():
    status = clij2fftWrapper.cleanup()
    if status != 0:
        raise RuntimeError('CLIJ2-FFT cleanup failed: ' + str(status))
    fft_geometries.clear()


def fft_dimensions(image_shape, psf_shape):
    extended = FinalDimensions(array([i + p for i, p in zip(image_shape, psf_shape)], 'l'))
    return tuple(long(n) for n in ops.filter().fftSize(extended, False)[0])


def inspect_geometries(request):
    # Predict the same cache resets as the upcoming channel sequence without
    # changing the actual cache or allocating image/GPU buffers.
    cached = set(fft_geometries)
    geometries = []
    for item in request['geometries']:
        fft_shape = fft_dimensions(item['crop'], item['psf'])
        key = (fft_shape, bool(request['non_circulant']))
        if key not in cached and len(cached) >= MAX_FFT_GEOMETRIES:
            cached.clear()
        geometries.append(dict(item, fft=fft_shape,
                               cache='reused' if key in cached else 'new'))
        cached.add(key)
    return geometries


def deconvolve(request):
    timings = {}
    start = clock()
    image = ops.convert().float32(io.open(request['input_path']).getImgPlus())
    psf = ops.convert().float32(io.open(request['psf_path']).getImgPlus())
    timings['open'] = clock() - start
    image_shape = tuple(long(image.dimension(i)) for i in range(image.numDimensions()))
    psf_shape = tuple(long(psf.dimension(i)) for i in range(psf.numDimensions()))
    # Use the same dimension calculation as the installed OpenCLFFTUtility.
    fft_shape = fft_dimensions(image_shape, psf_shape)
    fft_key = (fft_shape, bool(request['non_circulant']))
    if fft_key not in fft_geometries and len(fft_geometries) >= MAX_FFT_GEOMETRIES:
        reset_fft_cache()
    publish('geometry-' + request['id'] + '.json', {
        'crop': image_shape, 'psf': psf_shape, 'fft': fft_shape,
        'cache': 'reused' if fft_key in fft_geometries else 'new',
    })
    gpu_image = gpu_psf = gpu_output = None
    succeeded = False
    try:
        start = clock()
        clij2.clear()
        gpu_image = clij2.push(image)
        gpu_psf = clij2.push(psf)
        gpu_output = clij2.create(gpu_image.getDimensions(), NativeTypeEnum.Float)
        timings['push'] = clock() - start
        collectors = ManagementFactory.getGarbageCollectorMXBeans()
        gc_before = sum(max(0, c.getCollectionTime()) for c in collectors)
        start = clock()
        DeconvolveRichardsonLucyFFT.deconvolveRichardsonLucyFFT(
            clij2, gpu_image, gpu_psf, gpu_output,
            int(request['iterations']), float(request['regularization']), bool(request['non_circulant']),
        )
        timings['deconvolution'] = clock() - start
        gc_elapsed = sum(max(0, c.getCollectionTime()) for c in collectors) - gc_before
        if gc_elapsed > 100:
            print('JVM collection time during deconvolution: ' + str(gc_elapsed / 1000.0) + ' s')
            System.out.flush()
        start = clock()
        result = clij2.pullRAI(gpu_output)
        timings['pull'] = clock() - start
        start = clock()
        io.save(ds.create(result), request['output_path'])
        timings['save'] = clock() - start
        fft_geometries.add(fft_key)
        succeeded = True
    finally:
        start = clock()
        if gpu_output is not None:
            gpu_output.close()
        if gpu_psf is not None:
            gpu_psf.close()
        if gpu_image is not None:
            gpu_image.close()
        clij2.clear()
        if not succeeded:
            reset_fft_cache()
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
            if request.get('operation') == 'inspect':
                response = {'id': request['id'], 'geometries': inspect_geometries(request)}
            else:
                response = {'id': request['id'], 'timings': deconvolve(request)}
        except:
            response = {'id': request['id'], 'error': traceback.format_exc()}
        publish('response-' + request['id'] + '.json', response)
    except:
        if os.path.isdir(session_dir):
            publish('failed.json', {'error': traceback.format_exc()})
        break
reset_fft_cache()
System.exit(0)
