# -*- coding: utf-8 -*-

r'''
.. codeauthor:: Wilfried Mercier - LAM <wilfried.mercier@lam.fr>

Define classes that handle the modeling of one or multiple Sérsic models.
'''

import copy
import logging
import numpy                               as     np
from   numpy.typing                        import NDArray
from   matplotlib.gridspec                 import GridSpec
from   scipy.special                       import gamma
from   astropy.modeling.models             import Sersic2D
from   astropy.visualization.mpl_normalize import simple_norm
from   astropy.convolution                 import convolve_fft

# Custom imports
from .utils import compute_bn, CONV_MODES, set_rc_params

logger = logging.getLogger()

try:

    import torch
    import torch.nn.functional as F

    CONV_MODE = CONV_MODES.TORCH

    if not torch.cuda.is_available():
        logger.info('Pytorch: no GPU found to accelerate PSF convolution... Falling back on the CPU.')
    else:
        logger.info('Using Pytorch in GPU mode to accelerate PSF convolution.')

except:

    logger.info('Could not find pytorch installed... Falling back on astropy for fast PSF convolution.')
    CONV_MODE = CONV_MODES.ASTROPY

class ModelSersic2D:
    r"""
    .. codeauthor:: Wilfried Mercier.

    Class that allows to generate a 2D Sérsic model and returns a log-likelihood with potentially fixed values.

    :param im: image file of the galaxy in the given band
    :param psf: psf file in the given band
    :param band: band name
    :param err: error file of the galaxy in the given band
    :param mask: mask file of the galaxy in the given band. Values must be 0/1 or True/False with 1 or True for the regions to mask.
    :param mag_offset: magnitude offset used to generate the model
    :param values: initial values for the parameters given as a dict. Allowed keys are: xc, yc, mag, re, n, q, pa
    :param limits: min-max limits on the parameters given as a dict. Allowed keys are: xc, yc, mag, re, n, q, pa
    """

    def __init__(
            self, 
            im         : NDArray,
            psf        : NDArray,
            band       : str,
            err        : NDArray | None = None,
            mask       : NDArray | None = None,
            mag_offset : float      = 25.0,
            values     : dict[str, float]               = {},
            limits     : dict[str, tuple[float, float]] = {}
        ) -> None:
        
        self.band       = band
        self.mag_offset = mag_offset

        # Load the image
        self.im         = im.astype(np.float32)

        # Create a 2D grid
        self.y, self.x  = np.indices(self.im.shape)
        self.npix       = self.im.size

        # Load the PSF and make it odd-sized
        self.psf    = psf.astype(np.float32)
        self.psf   /= self.psf.sum()

        if self.psf.shape[0] % 2 == 0: self.psf = self.psf[1:]
        if self.psf.shape[1] % 2 == 0: self.psf = self.psf[:, 1:]

        # If the user uses Pytorch, we load the psf on the GPU
        if CONV_MODE == CONV_MODES.TORCH:
            self.kernel = torch.from_numpy(self.psf).reshape(1, 1, *self.psf.shape).to('cuda')

        # If no error, we assume a constant uncertainty everywhere
        if err is None: self.err  = np.ones_like(self.im)
        else:           self.err  = err.astype(np.float32)

        # If no mask, we assume all pixels must be fitted
        if mask is None: self.mask = np.ones_like(self.im).astype(bool)
        else:            self.mask = ~mask.astype(bool)

        # Initialize the model parameters
        self.model_mag : float = self.estimate_mag    if 'mag' not in values else values['mag']
        self.model_re  : float = 5.0                  if 're'  not in values else values['re']
        self.model_n   : float = 2.0                  if 'n'   not in values else values['n']
        self.model_pa  : float = 0.0                  if 'pa'  not in values else values['pa']
        self.model_q   : float = 0.5                  if 'q'   not in values else values['q']
        self.model_xc  : float = self.im.shape[0] / 2 if 'xc'  not in values else values['xc']
        self.model_yc  : float = self.im.shape[1] / 2 if 'yc'  not in values else values['yc']

        # Initialize the limits on the model parameters
        self.model_mag_lim : tuple[float, float] = (10.0, 35.0)            if 'mag' not in limits else limits['mag']
        self.model_re_lim  : tuple[float, float] = (0.5, 100.0)            if 're'  not in limits else limits['re']
        self.model_n_lim   : tuple[float, float] = (0.5, 8.0)              if 'n'   not in limits else limits['n']
        self.model_pa_lim  : tuple[float, float] = (-90.0, 90.0)           if 'pa'  not in limits else limits['pa']
        self.model_q_lim   : tuple[float, float] = (0.0, 1.0)              if 'q'   not in limits else limits['q']
        self.model_xc_lim  : tuple[float, float] = (0.0, self.im.shape[0]) if 'xc'  not in limits else limits['xc']
        self.model_yc_lim  : tuple[float, float] = (0.0, self.im.shape[1]) if 'yc'  not in limits else limits['yc']

        # Initialize the model
        self.model : NDArray | None  = None

        return
    
    @property
    def estimate_mag(self) -> float:
        return -2.5 * np.log10(self.im[self.mask].sum()) + self.mag_offset

    def set_parameters(
            self,
            xc  : float, 
            yc  : float,
            mag : float,
            re  : float,
            n   : float,
            q   : float,
            pa  : float, 
        ) -> None:
        """
        Sets the value of the model parameters.

        :param xc: x-coordinate of the center
        :param yc: y-coordinate of the center
        :param mag: total magnitude of the model
        :param re: half-light radius in pixels
        :param n: Sersic index
        :param q: axis ratio
        :param pa: position angle of the major axis in degree
        """

        self.model_xc  = xc
        self.model_yc  = yc
        self.model_pa  = pa
        self.model_q   = q
        self.model_mag = mag
        self.model_re  = re
        self.model_n   = n

        return
    
    def convolve(self, image: NDArray) -> NDArray:

        if CONV_MODE == CONV_MODES.ASTROPY:
            return convolve_fft(image, self.psf, boundary='fill', fill_value=0.0) # type: ignore
        elif CONV_MODE == CONV_MODES.TORCH:
            
            image  = torch.from_numpy(image.astype(np.float32)).reshape(1, 1, *image.shape).to('cuda') # type: ignore
    
            return (
                F.conv2d(image, self.kernel, padding=(self.psf.shape[0]-1)//2) # type: ignore
                 .to('cpu')
                 .numpy()
                 .squeeze()
             )

        else: raise ValueError('Convolution mode must be ASTROPY or TORCH.')

    def generate_model(self) -> None:
        '''Use Astropy to generate a PSF-convolved 2D model.'''

        if self.model_xc  is None: raise TypeError('Cannot generate the model if no xc is provided')
        if self.model_yc  is None: raise TypeError('Cannot generate the model if no yc is provided')
        if self.model_n   is None: raise TypeError('Cannot generate the model if no n is provided')
        if self.model_mag is None: raise TypeError('Cannot generate the model if no magnitude is provided')
        if self.model_re  is None: raise TypeError('Cannot generate the model if no re is provided')
        if self.model_q   is None: raise TypeError('Cannot generate the model if no q is provided')
        if self.model_pa  is None: raise TypeError('Cannot generate the model if no pa is provided')

        x         = self.x - self.model_xc
        y         = self.y - self.model_yc

        n         = self.model_n
        twon      = 2*n
        bn        = compute_bn(self.model_n)

        ftot      = 10**((self.mag_offset - self.model_mag) / 2.5)
        amplitude = (ftot * bn**(twon)) / (np.pi*twon*self.model_re*self.model_re*gamma(twon)*np.exp(bn))

        # Generate Sérsic 2D model
        galaxy = Sersic2D(
            amplitude = amplitude,
            r_eff     = self.model_re,
            n         = self.model_n,
            x_0       = 0,
            y_0       = 0,
            ellip     = 1 - self.model_q,
            theta     = np.deg2rad(90 + self.model_pa)
        )

        # Convolve galaxy with PSF
        self.model = self.convolve(galaxy(x, y).astype(np.float32))

        return

    def log_likelihood(self, cube: NDArray, *args, **kwargs) -> float | np.float64 | np.floating:
        r"""
        Log likelihood function which is maximized by multinest.

       -sum[(data-model)^2 / (2*err^2)]

       .. note::
        
            cube has its parameters in the following order:
                - x, y, mag, re, n, q, pa

        :param cube: data cube
        """
        
        self.set_parameters(*cube)
        self.generate_model()

        if self.model is None or self.err is None: 
            raise TypeError('Cannot compute log likelihood because the model or the error map is None')
        
        chi2 = -(self.model[self.mask] - self.im[self.mask])**2 / (2*self.err[self.mask]**2)

        return np.sum(chi2)
    
    @property
    def limits(self) -> list[tuple[float, float]]:
        r'''
        Return the min-max limits of the parameters in the following order:
            - x, y, mag, re, n, q, pa
        '''

        return [
            self.model_xc_lim, 
            self.model_yc_lim, 
            self.model_mag_lim, 
            self.model_re_lim, 
            self.model_n_lim, 
            self.model_q_lim,
            self.model_pa_lim
        ]
    
    @property
    def parameters(self) -> list[float]:
        r'''
        Return the current parameter values in the following order:
            - x, y, mag, re, n, q, pa
        '''

        if self.model_xc  is None: raise TypeError('Cannot generate the model if no xc is provided')
        if self.model_yc  is None: raise TypeError('Cannot generate the model if no yc is provided')
        if self.model_n   is None: raise TypeError('Cannot generate the model if no n is provided')
        if self.model_mag is None: raise TypeError('Cannot generate the model if no magnitude is provided')
        if self.model_re  is None: raise TypeError('Cannot generate the model if no re is provided')
        if self.model_q   is None: raise TypeError('Cannot generate the model if no q is provided')
        if self.model_pa  is None: raise TypeError('Cannot generate the model if no pa is provided')

        return [
            self.model_xc, 
            self.model_yc, 
            self.model_mag, 
            self.model_re, 
            self.model_n, 
            self.model_q,
            self.model_pa
        ]
    
    @property
    def parameter_names(self) -> list[str]:
        r'''
        Return the parameter names in the following order:
            - x, y, mag, re, n, q, pa
        '''

        return ['xc', 'yc', 'mag', 're', 'n', 'q', 'pa']
    
class MultipleModelSersic2D:
    r'''
    .. codeauthor:: Wilfried Mercier

    Combine multiple Sérsic models with the following constraints:
        - same x, y center coordinates
        - same axis ratio q
        - same position angle PA
    '''

    def __init__(self, *models: ModelSersic2D) -> None:

        self.models      = models
        self.mag_offsets = [model.mag_offset for model in self.models]

        return

    def log_likelihood(self, cube: NDArray, *args, **kwargs) -> float | np.float64 | np.floating:
        r'''Global log-likelihood for all the models.'''

        lk    = 0.0
        index = 0

        for pos, model in enumerate(self.models):

            if pos == 0:
                lk   += model.log_likelihood(cube[:7])
                index = 7
            else:
                parameters = [cube[0], cube[1], cube[index], cube[index+1], cube[index+2], cube[5], cube[6]]
                lk        += model.log_likelihood(np.array(parameters))
                index     += 3

        return lk
    
    @property
    def bands(self) -> list[str]: return [model.band for model in self.models]
    
    @property
    def limits(self) -> list[tuple[float, float]]:
        r'''
        Return the limits on the parameter in the following order:
            - x, y, mag, re, n, q, pa for the first model followed by
            - mag, re, n for the other ones
        '''

        for pos, model in enumerate(self.models):

            if pos == 0: limits = model.limits
            else:        limits.extend(model.limits[2:5])
                
        return limits
    
    @property
    def parameters(self) -> list[float]:
        r'''
        Return the current parameter values in the following order:
            - x, y, mag, re, n, q, pa for the first model followed by
            - mag, re, n for the other ones
        '''

        for pos, model in enumerate(self.models):

            if pos == 0: parameters = model.parameters
            else:        parameters.extend(model.parameters[2:5]) # type: ignore

        return parameters # type: ignore
    
    @parameters.setter
    def parameters(self, parameters: list[float]) -> None:

        index = 0

        for pos, model in enumerate(self.models):       

            if pos == 0:
                model.set_parameters(*parameters[:7])
                index = 7
            else:
                model.set_parameters(*[
                    parameters[0], parameters[1], 
                    parameters[index], parameters[index+1], parameters[index+2],
                    parameters[5], parameters[6]
                ])

                index     += 3

        return

    @property
    def parameter_names(self) -> list[str]:
        r'''
        Return the parameter names in the following order:
            - x, y, mag, re, n, q, pa for the first model followed by
            - mag, re, n for the other ones
        '''

        for pos, model in enumerate(self.models):

            if pos == 0: parameter_names = [f'{par}_1' for par in model.parameter_names]
            else:        parameter_names.extend([f'{par}_{pos+1}' for par in model.parameter_names[2:5]]) # type: ignore

        return parameter_names # type: ignore
    
    def generate_model(self) -> None:
        r'''Generate a new model with the current parameter values.'''

        for model in self.models: model.generate_model()
        return

    def plot(self) -> None:
        r'''Generate a recap plot of multi-band fitting.'''

        import matplotlib.pyplot as plt

        f  = plt.figure(figsize=(3*6, 6*len(self.models)))
        gs = GridSpec(len(self.models)+1, 3, height_ratios=[1]*len(self.models) + [0.05])
        set_rc_params()

        for pos, model in enumerate(self.models):

            ax       = f.add_subplot(gs[pos, 0])
            data_ok  = copy.deepcopy(model.im)
            data_nok = copy.deepcopy(model.im)
            mask     = ~model.mask

            data_ok[mask]   = np.nan
            data_nok[~mask] = np.nan

            norm     = simple_norm(data_ok, percent=99)

            ret_im = plt.imshow(data_ok,  origin='lower', cmap='plasma', norm=norm) # type: ignore
            plt.imshow(data_nok, origin='lower', cmap='plasma', alpha=0.5, norm=norm) # type: ignore
            plt.contour(mask, origin='lower', colors='k')

            if pos == 0: plt.title('Image')

            ax       = f.add_subplot(gs[pos, 1])
            data_ok  = copy.deepcopy(model.model)
            data_nok = copy.deepcopy(model.model)

            data_ok[mask]   = np.nan # type: ignore
            data_nok[~mask] = np.nan # type: ignore

            plt.imshow(data_ok,  origin='lower', cmap='plasma', norm=norm) # type: ignore
            plt.imshow(data_nok, origin='lower', cmap='plasma', alpha=0.5, norm=norm) # type: ignore
            plt.contour(mask, origin='lower', colors='k')

            if pos == 0: plt.title('Model')

            ax       = f.add_subplot(gs[pos, 2])
            data_ok  = copy.deepcopy(1 - model.model/model.im) # type: ignore
            data_nok = copy.deepcopy(1 - model.model/model.im) # type: ignore

            data_ok[mask]   = np.nan
            data_nok[~mask] = np.nan

            ret_res = plt.imshow(data_ok * 100,  origin='lower', cmap='bwr', vmin=-100, vmax=100)
            plt.imshow(data_nok * 100, origin='lower', cmap='bwr', vmin=-100, vmax=100, alpha=0.5)
            plt.contour(mask, origin='lower', colors='k')

            if pos == 0: plt.title('Residuals')

        ax = f.add_subplot(gs[len(self.models), 2])
        cbar = plt.colorbar(ret_res, cax=ax, orientation='horizontal') # type: ignore
        cbar.ax.set_xlabel(r'Relative error (\%)')

        ax = f.add_subplot(gs[len(self.models), :2])
        plt.colorbar(ret_im, cax=ax, orientation='horizontal', label='Flux [arbitrary unit]') # type: ignore