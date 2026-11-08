# Batch Deconvolution

## Install

Download the setup for your platform from [Releases](https://github.com/TomNaber/Imaris-Batch-Deconvolution/releases).

- **macOS:** open the DMG and drag the app to Applications (Apple Silicon: macOS 14+; Intel: macOS 13+).
- **Windows x64:** run `Setup.exe` (Windows 10+).

Python, Java, and ImageJ are included. GPU processing needs an OpenCL driver
(on Windows, install your GPU vendor's driver).

## Dev install

Install [Miniforge](https://github.com/conda-forge/miniforge), then run:

```sh
git clone https://github.com/TomNaber/Imaris-Batch-Deconvolution.git
cd Imaris-Batch-Deconvolution
conda env create -f environment.yml
conda activate batch-deconvolution
python scripts/setup_imagej.py
python main.py
```
