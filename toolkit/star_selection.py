from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from matplotlib import pyplot as plt


from astropy.io import fits
from photutils import CircularAperture, daofind, irafstarfind
from astropy.stats import mad_std

from astropy.convolution import convolve_fft, Tophat2DKernel
from pyfftw.interfaces.scipy_fftpack import fft2, ifft2

__all__ = ['init_centroids']


def init_centroids(first_image_path, master_flat, master_dark, target_centroid,
                   max_number_stars=10, min_flux=0.2, plots=False):
    from .reduction import rebin_image

    first_image = (rebin_image(fits.getdata(first_image_path), 2) - master_dark[:-1, :-1])/master_flat[:-1, :-1]

    tophat_kernel = Tophat2DKernel(27)
    convolution = convolve_fft(first_image, tophat_kernel, fftn=fft2, ifftn=ifft2)

    convolution -= np.median(convolution)

    from astropy.stats import mad_std
    mad = mad_std(convolution)

    convolution[convolution < -5*mad] = 0.0

    from skimage.filters import threshold_otsu, threshold_yen
    from skimage.measure import label, regionprops

    thresh = threshold_yen(convolution)

    masked = np.ones_like(convolution)
    masked[convolution <= thresh] = 0

    label_image = label(masked)

    regions = regionprops(label_image, convolution)
    centroids = [region.weighted_centroid for region in regions]


    #positions = np.vstack([sources['xcentroid'], sources['ycentroid']])
    # positions = np.array(centroids).T

    positions = np.vstack([[y for x, y in centroids], [x for x, y in centroids]])

    if plots:
        apertures = CircularAperture(positions, r=12.)
        brightest_apertures = CircularAperture(positions, r=12.)
        apertures.plot(color='b', lw=1, alpha=0.2)
        plt.imshow(first_image, vmin=np.percentile(first_image, 0.01),
                   vmax=np.percentile(first_image, 99.9), cmap=plt.cm.viridis,
                   origin='lower')

    return positions

    # target_index = np.argmin(np.abs(target_centroid - positions), axis=1)[0]
    # flux_threshold = sources['flux'] > min_flux * sources['flux'].data[target_index]
    #
    # fluxes = sources['flux'][flux_threshold]
    # positions = positions[:, flux_threshold]
    #
    # brightest_positions = positions[:, np.argsort(fluxes)[::-1][:max_number_stars]]
    # target_index = np.argmin(np.abs(target_centroid - brightest_positions),
    #                          axis=1)[0]
    #
    # apertures = CircularAperture(positions, r=12.)
    # brightest_apertures = CircularAperture(brightest_positions, r=12.)
    # apertures.plot(color='b', lw=1, alpha=0.2)
    # brightest_apertures.plot(color='r', lw=2, alpha=0.8)
    #
    # if plots:
    #     plt.imshow(first_image, vmin=np.percentile(first_image, 0.01),
    #                vmax=np.percentile(first_image, 99.9), cmap=plt.cm.viridis,
    #                origin='lower')
    #     plt.plot(target_centroid[0, 0], target_centroid[1, 0], 's')
    #
    #     plt.show()
    #
    # # Reorder brightest positions array so that the target comes first
    # indices = list(range(brightest_positions.shape[1]))
    # indices.pop(target_index)
    # indices = [target_index] + indices
    # brightest_positions = brightest_positions[:, indices]
    #
    # return brightest_positions
