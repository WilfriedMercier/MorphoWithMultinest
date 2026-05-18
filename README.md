# MorphoWithMultinest

A short library that performs single Sérsic profile fits in python using Multinest as fitter.

# Installation

## Installation of the environment


To install and run the code, first create a new conda environment using the project YAML file:

`conda create --file environment.yaml`

Activate the environment with 

`conda activate MorphoWithMultinest`

Then use pip to install additional packages:

`pip install corner pymultinest`

## Installation of MultiNest

To install [MultiNest](https://johannesbuchner.github.io/PyMultiNest/install.html), one must first clone the Git repository, then compile the code as follows

```
git clone https://github.com/JohannesBuchner/MultiNest
cd Multinest/build
cmake ..
make
```

It the compilation fails, make sure that

1. all the required dependencies are installed on the machine (cmake, git, gcc, gfortran, BLAS, LAPACK, ATLAS) - see the [documentation](https://johannesbuchner.github.io/PyMultiNest/install.html#prerequisites-for-building-the-libraries) for more information
2. to add the following line to your `.bashrc` file `export CMAKE_POLICY_VERSION_MINIMUM=3.5` if you have a modern version of cmake

Once cmake and make have successfully finished, the shared library is available in the directory `Multinest/lib`. The last step is to add the following line to your `.bashrc` file:

`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:pathToMultinest/MultiNest/lib`

where you replace `pathToMultinest` with your path.

# Running a morphological fit

An example of a Bayesian morphological fit is in `examples/JADES-GS-z14-0` and be run with

`python run_fit.py configuration.yaml`

All the information necesasry for the fit (location of the input files, magnitude zero point, MultiNest sampling parameters, etc.) are given in the configuration file. The `run_fit.py` python script simply loads the main function from MorphoWithMultinest that starts the fit given the configuration file. Alternatively, one can do

`python -c 'from MorphoWithMultinest import run; run("configuration.yaml")'`
