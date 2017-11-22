from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from matplotlib import pyplot as plt

from astropy.stats import mad_std
from astropy.io import fits
from photutils import CircularAperture

from astropy.convolution import convolve_fft, Tophat2DKernel
from pyfftw.interfaces.scipy_fftpack import fft2, ifft2

__all__ = ['init_centroids']


def init_centroids(first_image_path, master_flat, master_dark, target_centroid,
                   max_number_stars=10, min_flux=0.2, plots=False):

    first_image = np.median([(fits.getdata(path) - master_dark)/master_flat
                             for path in first_image_path], axis=0)

    tophat_kernel = Tophat2DKernel(5)
    convolution = convolve_fft(first_image, tophat_kernel, fftn=fft2, ifftn=ifft2)

    convolution -= np.median(convolution)

    mad = mad_std(convolution)

    convolution[convolution < -5*mad] = 0.0

    from skimage.filters import threshold_yen
    from skimage.measure import label, regionprops

    thresh = threshold_yen(convolution)/4 # Use /4 for planet c, /2 for planet b
    #thresh = threshold_otsu(convolution)/15

    masked = np.ones_like(convolution)
    masked[convolution <= thresh] = 0

    label_image = label(masked)

    plt.figure()
    plt.imshow(label_image, origin='lower', cmap=plt.cm.viridis)
    plt.show()

    # regions = regionprops(label_image, convolution)
    regions = regionprops(label_image, first_image)

    # reject regions near to edge of detector
    buffer_pixels = 50
    regions = [region for region in regions
               if ((region.weighted_centroid[0] > buffer_pixels and
                   region.weighted_centroid[0] < label_image.shape[0] - buffer_pixels)
               and (region.weighted_centroid[1] > buffer_pixels and
                    region.weighted_centroid[1] < label_image.shape[1] - buffer_pixels))]

    #centroids = [region.weighted_centroid for region in regions]
    #intensities = [region.mean_intensity for region in regions]

    target_intensity = regions[0].mean_intensity
    target_diameter = regions[0].equivalent_diameter
    #  and region.equivalent_diameter > 0.8 * target_diameter
    centroids = [region.weighted_centroid for region in regions
                 if min_flux * target_intensity < region.mean_intensity]
    # intensities = [region.mean_intensity for region in regions
    #                if min_flux * target_intensity < region.mean_intensity]
#    centroids = np.array(centroids)[np.argsort(intensities)[::-1]]

    distances = [np.sqrt((target_centroid[0] - d[0])**2 +
                         (target_centroid[1] - d[1])**2) for d in centroids]

    centroids = np.array(centroids)[np.argsort(distances)]

    positions = np.vstack([[y for x, y in centroids], [x for x, y in centroids]])

    if plots:
        apertures = CircularAperture(positions, r=12.)
        apertures.plot(color='r', lw=2, alpha=1)
        plt.imshow(first_image, vmin=np.percentile(first_image, 0.01),
                   vmax=np.percentile(first_image, 99.9), cmap=plt.cm.viridis,
                   origin='lower')
        plt.scatter(positions[0, 0], positions[1, 0], s=150, marker='x')

        plt.show()
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
