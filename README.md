# Automatic segmentation of STEM images

Automatic segmentation of Scanning Transmission Electron Microscope (STEM) images with unsupervised machine learning

### Learning more
If you want to learn more about this project, you may read [our manuscript](https://github.com/NingWang1990/pySTEM/blob/master/paper/paper_segmentation.pdf?raw=true) that is not yet completely finished, or [our presentation](https://github.com/NingWang1990/pySTEM/blob/master/slides/BiGmax2020.pdf?raw=true) for BiGmax workshop 2020.

**Try it now**:
 [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/NingWang1990/pySTEM/master?filepath=examples)

## Getting Started

### Prerequisites
python 3, numpy, scipy, scikit-learn, fftw3 
### Installing
via conda, the simplest way, highly recommended
```
conda install -c conda-forge pystem
```
via pip
```
1. First make sure that FFTW3 library is installed 
2. pip install pystem
```
via source code
```
1. First make sure that FFTW3 library is installed 
then type commands:
git clone https://github.com/NingWang1990/pySTEM.git
cd pySTEM
python setup.py build
python setup.py install --user
```

### How to use
Examples can be found in the examples folder 
Give it a try right now by simply clicking the 'launch binder' button


