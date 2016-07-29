from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from matplotlib import pyplot as plt

from astropy.io import fits
from photutils import CircularAperture, daofind
from astropy.stats import mad_std

__all__ = ['init_centroids']

def init_centroids(first_image_path, master_flat, master_dark, target_centroid,
                   max_number_stars=10, min_flux=0.2, plots=False):

    first_image = (fits.getdata(first_image_path) - master_dark)/master_flat

    bkg_sigma = mad_std(first_image)

    sources = daofind(first_image, fwhm=5., threshold=4.0 * bkg_sigma)
    positions = np.vstack([sources['xcentroid'], sources['ycentroid']])
    target_index = np.argmin(np.abs(target_centroid - positions), axis=1)[0]
    flux_threshold = sources['flux'] > min_flux * sources['flux'].data[target_index]

    fluxes = sources['flux'][flux_threshold]
    positions = positions[:, flux_threshold]

    brightest_positions = positions[:, np.argsort(fluxes)[::-1][:max_number_stars]]
    target_index = np.argmin(np.abs(target_centroid - brightest_positions),
                             axis=1)[0]

    apertures = CircularAperture(positions, r=12.)
    brightest_apertures = CircularAperture(brightest_positions, r=12.)
    apertures.plot(color='b', lw=1, alpha=0.2)
    brightest_apertures.plot(color='r', lw=2, alpha=0.8)

    if plots:
        plt.imshow(first_image, vmin=np.percentile(first_image, 0.01),
                   vmax=np.percentile(first_image, 99.9), cmap=plt.cm.viridis,
                   origin='lower')
        plt.plot(target_centroid[0, 0], target_centroid[1, 0], 's')

        plt.show()

    # Reorder brightest positions array so that the target comes first
    indices = list(range(brightest_positions.shape[1]))
    indices.pop(target_index)
    indices = [target_index] + indices
    brightest_positions = brightest_positions[:, indices]

    return brightest_positions
