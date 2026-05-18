# -*- coding: utf-8 -*-

r'''
.. codeauthor:: Wilfried Mercier - LAM <wilfried.mercier@lam.fr>

Script that fits a Sérsic model with Galfit and generates an output map
'''

import glob
import corner
import pymultinest
import logging
import functools
import pathlib
import numpy                               as     np
import matplotlib.pyplot                   as     plt
from   numpy.typing                        import NDArray
from   astropy.modeling.models             import Sersic1D

# Custom imports
from .utils import CustomError

logger = logging.getLogger()

class MultinestOutputHandler:
    r'''
    Class that handles automatically the generation of Bayesian plots from the output of multinest.

    :param path: output directory where the multinest files are located
    :param parameter_names: names of the parameters
    :param bands: names of the bands
    :mag_offsets: magnitude zeropoint for each band
    '''

    def __init__(self, 
        path            : pathlib.Path, 
        parameter_names : list[str],
        bands           : list[str],
        mag_offsets     : list[float]
    ) -> None:

        self.bands           = bands
        self.mag_offsets     = mag_offsets
        self.parameter_names = parameter_names
        self.path            = path
        weight_file          = glob.glob(str(self.path / '*post_equal_weights.dat'))
        
        if len(weight_file) == 0: 
            logger.error(f'No posterior file found in directory {self.path}')
            raise CustomError
        
        if len(weight_file) > 1: 
            logger.error(f'Too many posterior files found in directory {self.path}')
            raise CustomError

        self.weight_file = weight_file[0]
        self.data        = np.loadtxt(self.weight_file)

        if self.data.ndim != 2:
            logger.error('Bayesian parameter plots unavailable because the posterior file is not 2D.')
            raise CustomError
        
        self.analyzer = pymultinest.Analyzer(
            n_params = len(self.parameter_names), 
            outputfiles_basename = str(self.path / '_')
        )

        return
    
    def corner(self) -> None:

        _ = plt.figure()
        corner.corner(self.data[:, :-1], labels=self.parameter_names, bins=10, show_titles=True, quantiles=[0.16, 0.5, 0.84])
        return
    
    @functools.cached_property
    def stats(self) -> dict: return self.analyzer.get_mode_stats()
    
    @functools.cached_property
    def best_fit(self) -> dict: return self.analyzer.get_best_fit()

    def print_results(self) -> None:

        message = ' '
        for par in self.parameter_names: 
            message += '{0:^{width}}'.format(par, width=12)

        logger.info(message)
        logger.info('-' * (len(self.parameter_names)*12))

        message = ' '
        for par in self.best_fit['parameters']:
            message += '{0:^{width}.{prec}}'.format(par, width=12, prec=6)

        logger.info(message)
        
        message = ' '
        for par in self.stats['modes'][0]['sigma']:
            message += '{0:^{width}.{prec}}'.format(par, width=12, prec=6)

        logger.info(message)

        return
    
    def _sample_profile_from_posterior(
            self, 
            r: NDArray, 
            zeropoint: float,
            index : int = 0
    ) -> dict[str, NDArray]:
        
        out    = {}
        models = []

        for parameters in self.data[:, :-1]:

            models.append(
                Sersic1D(
                    amplitude = 10**((zeropoint - parameters[index + 2]) / 2.5),
                    r_eff     = parameters[index + 3],
                    n         = parameters[index + 4]
                )(r)
            )

        # Compute median profile and 16th and 84th percentiles
        out['median'] = np.nanmedian(  models,       axis=0)
        out['q05']    = np.nanquantile(models, 0.05, axis=0)
        out['q95']    = np.nanquantile(models, 0.95, axis=0)
        out['q16']    = np.nanquantile(models, 0.16, axis=0)
        out['q84']    = np.nanquantile(models, 0.84, axis=0)

        return out
    
    def plot_profile(self, unit_flux: str, pscale: float) -> None:

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        parameters = self.best_fit['parameters']

        reff   = []
        index  = 0

        for _ in self.bands: reff.append(parameters[index + 3])

        max_x      = np.nanmax(reff)
        
        r = np.linspace(0, 3*parameters[3], 1000)
        _ = plt.figure(figsize=(6, 6))

        index  = 0

        for pos, (band, zeropoint) in enumerate(zip(self.bands, self.mag_offsets)):

            profiles = self._sample_profile_from_posterior(r, zeropoint, index=index)

            plt.plot(r*pscale, profiles['median'], ls='-', color=colors[pos], label=f'Median {band}')
            plt.fill_between(r*pscale, profiles['q16'], profiles['q84'], color=colors[pos], alpha=0.3, label=fr'$1\sigma$/$2\sigma$ [{band}]')
            plt.fill_between(r*pscale, profiles['q05'], profiles['q95'], color=colors[pos], alpha=0.3)

            plt.axvline(reff[-1]*pscale, ls='--', color=colors[pos], label=fr'$R_e$ [{band}]')

            if pos == 0 : index = 5
            else        : index += 3

        plt.xlim([0, max_x * pscale * 3])
        plt.xlabel(r'Distance to center [arcsec]')
        plt.ylabel(fr'Flux [{unit_flux}]')
        plt.yscale('log')
        plt.legend(frameon=False)

        return