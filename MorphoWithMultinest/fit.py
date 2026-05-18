# -*- coding: utf-8 -*-

r'''
.. codeauthor:: Wilfried Mercier - LAM <wilfried.mercier@lam.fr>

Run a single Sérsic model fit on multiple bands with shared center position, axis ratio, and PA.
'''

import yaml
import time
import logging
import pathlib
import colorlog
import pymultinest
import matplotlib.pyplot as     plt
from   astropy.io        import fits
from   numpy.typing      import NDArray

from .utils  import load_files, CustomError, CONV_MODES
from .models import ModelSersic2D, MultipleModelSersic2D
from .output import MultinestOutputHandler

# Define logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# Define the log format with colors
formatter = colorlog.ColoredFormatter(
    "%(asctime)s - %(log_color)s%(levelname)-4s%(reset)s %(message)s",
    log_colors={
        'INFO':    'green',
        'WARNING': 'yellow',
        'ERROR':   'red',
        'CRITICAL':'red,bg_white',
        'DEBUG':   'cyan',
    }
)

handler.setFormatter(formatter)
logger.addHandler(handler)

try:

    import torch

    logger.info('Using Pytorch for PSF convolution.')
    
    CONV_MODE = CONV_MODES.TORCH

    if not torch.cuda.is_available():
        logger.info('Pytorch: no GPU found to accelerate PSF convolution... Falling back on the CPU.')
    else:
        logger.info('Using Pytorch in GPU mode to accelerate PSF convolution.')

except:

    logger.info('Could not find Pytorch on system... Falling back on astropy for fast PSF convolution.')
    
    CONV_MODE = CONV_MODES.ASTROPY

def run_multinest(
        config   : dict,
        files    : dict[str, list[NDArray] | NDArray],
        skip_fit : bool = False
    ) -> None:
    r'''
    Run a fit using multinest.

    :param config: configuration dictionary
    :param files: dictionary containing the different files
    :param skip_fit: whether to skip the fitting process and load the results or not
    '''

    def prior(cube: NDArray, ndim: int, nparams: int) -> None:
        r"""
        Define the limits of the parameter space where multinest is allowed to explore

        :param ndarray cube: data with n_params dimension
        :param ndim: number of dimension if different of the number of parameters
        :param nparams: number of parameters
        """

        def apply_limit(element: float, limits: tuple[float, float]) -> float:
            return element * (limits[1] - limits[0]) + limits[0]

        # Set bounds on RA and Dec coordinates
        for pos in range(nparams):
            cube[pos] = apply_limit(cube[pos], model.limits[pos])
        
        return
    
    # Generate the model
    models = []
    for band, im, err, psf, mag_offset in zip(files['bands'], files['data'], files['err'], files['psf'], files['mag_offset']):

        models.append(ModelSersic2D(
            im,
            psf,
            band,
            err        = err,
            mask       = files['mask'], # type: ignore
            mag_offset = mag_offset,
        ))

    model = MultipleModelSersic2D(*models)
    p0    = model.parameters

    out_path = pathlib.Path(config['out_path'])

    # Call PyMultiNest
    if not skip_fit:

        t1   = time.time()
        
        pymultinest.run(
            model.log_likelihood, prior, len(p0), 
            outputfiles_basename = str(out_path / '_'),
            **config['multinest']
        )

        t2 = time.time()
        logger.info(f' fit done in: {t2-t1:6.2f} s \n')

    else: logger.info('Skipping fitting and directly loading the results.')

    # Print the results on screen
    output = MultinestOutputHandler(out_path, model.parameter_names, model.bands, model.mag_offsets)
    output.print_results()

    # Create a recap file with the best-fit model
    logger.info(f'Generating a PDF recap file at {str(out_path / "recap.pdf")}')
    model.parameters = output.best_fit['parameters']
    
    model.generate_model()
    model.plot()

    plt.savefig(str(out_path / "recap.pdf"), bbox_inches='tight', transparent=True)

    # Save best-fit model
    hdul : list[fits.PrimaryHDU | fits.ImageHDU] = [fits.PrimaryHDU()]
    for model in model.models:

        # Add the original image
        hdu                    = fits.ImageHDU(data=model.im)
        hdu.header['XTENSION'] = 'IMAGE'
        hdu.header['BAND']     = model.band
        hdul.append(hdu)

        # Add the model
        hdu                    = fits.ImageHDU(data=model.model)
        hdu.header['XTENSION'] = 'MODEL'
        hdu.header['BAND']     = model.band
        hdul.append(hdu)

        # Add the residuals
        hdu                    = fits.ImageHDU(data=model.im - model.model) # type: ignore
        hdu.header['XTENSION'] = 'RESIDUAL'
        hdu.header['BAND']     = model.band
        hdul.append(hdu)

    hdul = fits.HDUList(hdul)
    hdul.writeto(out_path / 'model.fits', overwrite=True)

    # Create a profile plot with uncertainties
    logger.info(f'Generating a profile plot at {str(out_path / "profile.pdf")}')
    output.plot_profile(config['plot_info']['unit'], config['plot_info']['pscale']) # type: ignore
    plt.savefig(str(out_path / 'profile.pdf'), bbox_inches='tight', transparent=True)

    # Create a corner plot for the parameters
    logger.info(f'Generating a corner plot at {str(out_path / "corner.pdf")}')
    output.corner()
    plt.savefig(str(out_path / 'corner.pdf'), bbox_inches='tight', transparent=True)

    return

def run(config_file: str | pathlib.Path, skip_fit: bool = False) -> None:
    r'''
    Load the files from the configuration file and then run the fit with Multinest.
    '''

    try:

        # Load the configuration file and verify its integrity
        with open(config_file, 'r') as f: 
            config = yaml.load(f, Loader=yaml.SafeLoader)

        # Check that files are consistent with each other
        files      = load_files(config)

        # Run multinest
        logger.info('Running multinest for fast Bayesian parameter estimation.')
        run_multinest(config, files, skip_fit=skip_fit)

    except CustomError: pass

    return