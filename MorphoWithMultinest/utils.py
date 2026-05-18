# -*- coding: utf-8 -*-

r'''
.. codeauthor:: Wilfried Mercier - LAM <wilfried.mercier@lam.fr>

Utility functions and classes.
'''

import enum
import textwrap
import logging
import functools
import pathlib
import matplotlib        as     mpl
import matplotlib.pyplot as     plt
import astropy.io.fits   as     fits
from   numpy.typing      import NDArray
from   scipy.special     import gammaincinv

logger = logging.getLogger()

class CONV_MODES(enum.Enum):
    TORCH   = enum.auto()
    ASTROPY = enum.auto()

class CustomError(Exception): pass

@functools.cache
def compute_bn(n: float | int) -> float:
    r"""
    Compute the value of bn used in the definition of a Sersic profile
    
    .. math::
            
        2\gamma(2n, b_n) = \Gamma(2n)

    :param n: Sersic index of the profile
    """
    
    return gammaincinv(2*n, 0.5)

def load_files(config: dict) -> dict[str, list[NDArray] | NDArray]:
    r'''
    Load all the files and perform some checks.
    
    :param config: configuration dictionary containing all the information regarding the files that must be loaded to perform the fit and other parameters
    '''

    def load_file(file: pathlib.Path) -> NDArray:
        with fits.open(file) as hdul:
            return hdul[0].data # type: ignore
    
    config = check_config(config)

    # Output dictionary containing all the files
    files = {'bands' : [], 'data' : [], 'err' : [], 'psf' : [], 'mask' : None, 'mag_offset' : []}

    # Load the mask and use it as reference to check the shape of the other files
    files['mask'] = load_file(config['mask']).astype(bool)
    ref_shape     = files['mask'].shape

    if len(ref_shape) != 2:
        logger.error(f'Mask does not have 2 dimensions but {len(ref_shape)}.')
        raise CustomError

    for band, params in config['bands'].items():

        files['bands'].append(band)
        files['mag_offset'].append(params['mag_offset'])
    
        # Load the images
        files['data'].append(load_file(params['image']))
        check_shape(files['data'][-1].shape, ref_shape, 'image', band)
        
        # Load the error maps
        files['err'].append(load_file(params['error']))
        check_shape(files['err'][-1].shape, ref_shape, 'error', band)
        
        # Load the psfs
        files['psf'].append(load_file(params['psf']))
        check_shape(files['psf'][-1].shape, ref_shape, 'psf', band)
    
    logger.info('All files successfully loaded.')

    return files

def check_shape(shape1: tuple[int, int], shape2: tuple[int, int], which: str, band: str) -> None:
    r'''
    Check that two shapes are the same.
    
    :param shape1: shape to be checked
    :param shape2: reference shape
    :param which: kind of file being checked
    :param band: band being checked
    '''

    if len(shape1) != 2:
        logger.error(f'File {which} for band {band} does not have 2 dimensions but {len(shape1)}.')
        raise CustomError

    if shape1 != shape2:
        logger.error(f'Not all files have the same shape. Mask has shape {shape2} but file {which} for band {band} has shape {shape1}.')
        raise CustomError

    return

def check_config(config: dict) -> dict:
    r'''Check that the configuration file is properly setup.'''

    if 'mask'  not in config: logger.error('No mask provided in the configuration file.');  raise CustomError
    if 'bands' not in config: logger.error('No bands provided in the configuration file.'); raise CustomError
    
    for band, params in config['bands'].items():

        for which, wtype in zip(('image', 'error', 'psf', 'mag_offset'), (str, str, str, float)):
            
            if which not in params: 
                logger.error(f'Missing {which} in band {band} in the configuration file.'); raise CustomError

            if not isinstance(params[which], wtype):
                logger.error(f'{which} in band {band} must be a string.'); raise CustomError
            
            if which != 'mag_offset' and not pathlib.Path(params[which]).exists():
                logger.error(f'{which} in band {band} not found on disk.'); raise CustomError

    if 'out_path' not in config: logger.error('Missing an output path in the configuration file.'); raise CustomError
    if not (out_path := pathlib.Path(config['out_path'])).exists(): out_path.mkdir()

    if 'multinest' not in config:

        logger.warning(textwrap.dedent('''\
        Multinest parameters missing in the configuration file. Using the following parameters instead:
            \tresume               = False, 
            \tverbose              = False, 
            \tmax_iter             = 0,
            \tn_live_points        = 600, 
            \tsampling_efficiency  = 0.8, 
            \tevidence_tolerance   = 0.5,
            \tn_iter_before_update = 100, 
            \tnull_log_evidence    = -1.0e+90, 
            \tmax_modes            = 100,
            \tmode_tolerance       = -1.0e+60\
        '''))

        config['multinest'] = {
            'resume'               : False, 
            'verbose'              : False, 
            'max_iter'             : 0,
            'n_live_points'        : 600, 
            'sampling_efficiency'  : 0.8, 
            'evidence_tolerance'   : 0.5,
            'n_iter_before_update' : 100, 
            'null_log_evidence'    : -1.0e+90, 
            'max_modes'            : 100,
            'mode_tolerance'       : -1.0e+60
        }

    if 'plot_info' not in config:

        logger.warning('No plot information provided. Assuming a default flux unit of MJy/sr and a default pixel scale of 30 mas.')

        config['plot_info'] = {'unit' : 'MJy/sr', 'pscale' : 0.03}

    elif not isinstance(config['plot_info']['unit'], str):
        logger.error('Flux unit for the recap plots must be a string.'); raise CustomError
    elif not isinstance(config['plot_info']['pscale'], float):
        logger.error('Pixel scale for the recap plots must be a float.'); raise CustomError
    elif config['plot_info']['pscale'] <= 0:
        logger.error('Pixel scale for the recap plots must be > 0.'); raise CustomError

    logger.info('Configuration file ok.')

    return config

def set_rc_params() -> None:

    params = {
        'legend.fancybox': False,
        'legend.frameon': False,
        'legend.edgecolor': 'none',

        'figure.figsize': (5, 5),
        'figure.dpi': 300,

        'axes.labelsize': 14,

        'xtick.top': True,
        'xtick.bottom': True,
        'xtick.labelbottom': True,
        'xtick.labeltop': False,
        'xtick.direction': 'in',
        'xtick.labelsize': 14,

        'ytick.left': True,
        'ytick.right': True,
        'ytick.labelleft': True,
        'ytick.labelright': False,
        'ytick.direction': 'in',
        'ytick.labelsize': 14,

        'font.family': 'serif',
        'font.serif': 'Times',
        'font.size': 12,
        'text.usetex': True,

        'mathtext.fontset': 'custom',
        'mathtext.rm': 'Times'
    }

    for key, value in params.items():
        mpl.rcParams[key] = value

    return