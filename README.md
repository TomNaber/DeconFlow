# Batch Deconvolution

Supports macOS 14 or newer on Apple Silicon, macOS 13 or newer on Intel,
and Windows x64. These macOS minimums come from the pinned CLIJ2-FFT binaries. GPU
processing requires a working OpenCL driver; Windows users should install their
GPU vendor's driver. Native GPU validation currently covers Apple Silicon;
see [platform verification](docs/platform-verification.md) for remaining checks.

Install [Miniforge](https://github.com/conda-forge/miniforge), then run these
commands from a terminal with Conda available:

```sh
git clone https://github.com/TomNaber/Imaris-Batch-Deconvolution.git
cd Imaris-Batch-Deconvolution
conda env create -f environment.yml
conda activate batch-deconvolution
python scripts/setup_imagej.py
python main.py
```