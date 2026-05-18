# MorphoWithMultinest

A short library that performs single Sérsic profile fits in python using Multinest as fitter.

# Installation

## Installation of the environment


First clone this repository with

```
git clone https://github.com/WilfriedMercier/MorphoWithMultinest.git
```

To install and run the code, first create a new conda environment using the project YAML file:

```
conda create --file environment.yaml
```

Activate the environment with 

```
conda activate MorphoWithMultinest
```

Then use pip to install additional packages:

```
pip install corner pymultinest
```

## Installation of MultiNest

To install [MultiNest](https://johannesbuchner.github.io/PyMultiNest/install.html), one must first clone the Git repository, then compile the code as follows

```
git clone https://github.com/JohannesBuchner/MultiNest
cd Multinest/build
cmake ..
make
```

It the compilation fails, make sure

1. that all the required dependencies are installed on the machine (cmake, git, gcc, gfortran, BLAS, LAPACK, ATLAS) - see the [documentation](https://johannesbuchner.github.io/PyMultiNest/install.html#prerequisites-for-building-the-libraries) for more information
2. to add the following line to your `.bashrc` file `export CMAKE_POLICY_VERSION_MINIMUM=3.5` if you have a modern version of cmake

Once cmake and make have successfully finished, the shared library is available in the directory `Multinest/lib`. The last step is to add the following line to your `.bashrc` file:

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:pathToMultinest/MultiNest/lib
```

where you replace `pathToMultinest` with your path.

## Installation of the package

Once all the dependencies are installed, you can install the package with pip using the command

```
pip install -e .
```

# Morphological fitting with MultiNest

An example of a Bayesian morphological fit is in `examples/JADES-GS-z14-0` and can be run with

```
python run_fit.py configuration.yaml
```

All the information necessary for the fit (location of the input files, magnitude zero point, MultiNest sampling parameters, etc.) are given in the configuration file. The `run_fit.py` python script simply loads the main function from `MorphoWithMultinest` that starts the fit given the configuration file. Alternatively, one can do

```
python -c 'from MorphoWithMultinest import run; run("configuration.yaml")'
```

**Note:** The task taking most of the inference time is the psf convolution of each generated model with the psf. To speedup this process, by default and if possible, the code will try to use PyTorch in GPU mode. If there is no compatible GPU on the system, it will fallback on CPU mode and, if PyTorch is not installed, it will use Astropy instead. However, note that using PyTorch in CPU mode or Astropy significantly increased the inference time compared to the GPU mode.

# Outputs of the fit

Once the fit is finished, the fitter returns in the terminal the best fit results and its uncertainty as follows:

```
2026-05-18 15:50:59,880 - INFO      xc_1        yc_1       mag_1        re_1        n_1         q_1         pa_1       mag_2        re_2        n_2     
2026-05-18 15:50:59,880 - INFO ------------------------------------------------------------------------------------------------------------------------
2026-05-18 15:50:59,888 - INFO     48.336      49.107      26.134     5.35458     1.20678     0.579909    84.5067     26.3009     5.34319     0.954768  
2026-05-18 15:50:59,888 - INFO   0.0264706   0.0160002   0.0221266   0.0946627   0.0401668   0.00780975   0.676399   0.0280832    0.13475    0.0732638
```

Then, it generates a PDF recap file showing the original images, best-fit models, and residuals:


Finally, it produces a plot showing the median profile in each band with the 1 and 2 sigma uncertainties around:

