macOS 12 or newer on Apple Silicon

Install [Miniforge](https://github.com/conda-forge/miniforge) and [Fiji](https://fiji.sc)

In Fiji enable CLIJ CLIJ2 and clijx-deconvolution under Help > Update > Manage update sites then apply changes and restart

```sh
git clone https://github.com/TomNaber/Imaris-Batch-Deconvolution.git
cd Imaris-Batch-Deconvolution
conda env create -f environment.yml
conda activate batch-deconvolution
python main.py
```

Select the Fiji executable in the GUI
