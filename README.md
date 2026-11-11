# DeconFlow

<p align="center">
  <img src="docs/images/deconvolution-preview.png" width="49%" alt="Deconvolution preview">
  <img src="docs/images/psf-preview.png" width="49%" alt="Gibson-Lanni PSF">
</p>

## About

Why do we need another deconvolution app? Because we want all of the following at the same time:
- Gibson-Lanni point-spread functions to correct for all media the light passes through.
- Richardson-Lucy deconvolution with non-circulant edge correction and total variation regularization.
- GPU acceleration of deconvolution.
- Interactive preview showing XY and YZ dimensions of Z-stack with linked zoom and positioning between original and deconvolved preview and supporting running deconvolution on a crop, allowing you to see the result of a parameter change (usually) in seconds.
- Visualization of point-spread functions, allowing you to see the effect of changing a parameter on the point-spread function.
- Optional post-processing filters to smoothen out residual noise.
- Batch processing. 
- Intuitive GUI with all dependencies bundled.
- Open source.

This app uses the [psfmodels](https://github.com/tlambert03/PSFmodels) plugin to generate Gibson-Lanni point-spread functions. A minimal headless ImageJ installation with [CLIJ2](https://github.com/clij/clij2/) is bundled and integrated for GPU-accelerated Richardson-Lucy deconvolution with non-circulant boundary correction and total variation regularization. Gaussian, median, and bilateral filtering are supported for both raw and deconvolved images, with [SimpleITK](https://github.com/SimpleITK/SimpleITK) serving as the filtering backend.

## Usage

1. Select an input folder with .ims files and an output folder.
2. Fill the point-spread function values.
3. Use the preview tab to load a file from your selected folder and drag the crop area to a representative section.
4. Tune the deconvolution parameters by running deconvolution on a small crop (use sync preview to rerun deconvolution on the crop).
5. Optionally apply a post-processing filter (e.g., a bilateral filter to smoothen noisy surfaces while preserving edges).
6. Once the desired result is achieved on the preview, run batch deconvolution on your Z-stacks. 

## Install

Download the setup for your platform from [Releases](https://github.com/TomNaber/Imaris-Batch-Deconvolution/releases).

- **macOS:** open the DMG and drag the app to Applications (Apple Silicon: macOS 14+, Intel: macOS 13+).
- **Windows x64:** run `Setup.exe` (Windows 10+).

Python, Java, and ImageJ are included.

## Citation

If you use this app, please cite it as follows (use the version you used):

<!-- citation:start -->
> Naber, T. (2026). *DeconFlow: Batch Richardson–Lucy deconvolution with Gibson–Lanni point-spread functions* (Version 1.1.0) [Computer software]. https://github.com/TomNaber/Imaris-Batch-Deconvolution/releases/tag/v1.1.0
<!-- citation:end -->

## Key references

- Lambert, T. (2019). PSFmodels: Python bindings for scalar and vectorial models of the point spread function [Computer software]. GitHub. https://github.com/tlambert03/PSFmodels
- Haase, R., Royer, L. A., Steinbach, P., Schmidt, D., Dibrov, A., Schmidt, U., Weigert, M., Maghelli, N., Tomancak, P., Jug, F., & Myers, E. W. (2020). CLIJ: GPU-accelerated image processing for everyone. Nature Methods, 17(1), 5–6. https://doi.org/10.1038/s41592-019-0650-1
- Lowekamp, B. C., Chen, D. T., Ibáñez, L., & Blezek, D. (2013). The design of simpleitk. Frontiers in Neuroinformatics, 7. https://doi.org/10.3389/fninf.2013.00045
- Gibson, S. F., & Lanni, F. (1992). Experimental test of an analytical model of aberration in an oil-immersion objective lens used in three-dimensional light microscopy. Journal of the Optical Society of America A, 9(1), 154. https://doi.org/10.1364/JOSAA.9.000154
- Dey, N., Blanc‐Feraud, L., Zimmer, C., Roux, P., Kam, Z., Olivo‐Marin, J., & Zerubia, J. (2006). Richardson–Lucy algorithm with total variation regularization for 3D confocal microscope deconvolution. Microscopy Research and Technique, 69(4), 260–266. https://doi.org/10.1002/jemt.20294
- Bertero, M., & Boccacci, P. (2005). A simple method for the reduction of boundary effects in the Richardson-Lucy approach to image deconvolution. Astronomy & Astrophysics, 437(1), 369–374. https://doi.org/10.1051/0004-6361:20052717

## Dev install

Install [Miniforge](https://github.com/conda-forge/miniforge), then run:

```sh
git clone https://github.com/TomNaber/DeconFlow.git
cd DeconFlow
conda env create -f environment.yml
conda activate batch-deconvolution
python scripts/setup_imagej.py
python main.py
```
