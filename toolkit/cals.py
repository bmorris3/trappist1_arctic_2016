from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from astropy.utils.console import ProgressBar
from astropy.io import fits

from .regression import regression_model, regression_coeffs

__all__ = ['generate_master_flat_and_dark']


def generate_master_flat_and_dark(flat_paths, dark_paths, master_flat_path,
                                  master_dark_path):
    """
    Create a master flat from night-sky flats, and a master dark.

    Parameters
    ----------
    flat_paths : list
        List of paths to flat fields
    dark_paths : list
        List of paths to dark frames
    master_flat_path : str
        Path to master flat that will be created
    master_dark_path : str
        Path to master dark frame that will be created
    """

    # Make master dark frame:
    testdata = fits.getdata(dark_paths[0])
    allflatdarks = np.zeros((testdata.shape[0], testdata.shape[1],
                             len(dark_paths)))
    for i, darkpath in enumerate(dark_paths):
        allflatdarks[:, :, i] = fits.getdata(darkpath)
    masterflatdark = np.median(allflatdarks, axis=2)

    fits.writeto(master_dark_path, masterflatdark, clobber=True)

    # Make master flat field:
    testdata = fits.getdata(flat_paths[0])
    allflats = np.zeros((testdata.shape[0], testdata.shape[1], len(flat_paths)))

    for i, flatpath in enumerate(flat_paths):
        flat_dark_subtracted = fits.getdata(flatpath) - masterflatdark # * 2
        allflats[:, :, i] = flat_dark_subtracted

    coefficients = np.ones((allflats.shape[0], allflats.shape[1]), dtype=float)

    median_pixel_flux = np.atleast_2d(np.median(allflats, axis=(0, 1))).T

    margin = 5

    with ProgressBar(allflats.size) as bar:
        for i in range(margin, allflats.shape[0]-margin):
            for j in range(margin, allflats.shape[1]-margin):
                bar.update()

                pixel_fluxes = allflats[i, j, :]
                pixel_errors = np.sqrt(pixel_fluxes)

                mask = np.ones_like(pixel_fluxes).astype(bool)
                indices = np.arange(len(pixel_fluxes))

                while True:
                    # If pixel fluxes are negative, set that pixel to one
                    if pixel_fluxes.mean() < 100:
                        c = [1.0]
                        break

                    inds = indices[mask]
                    c = regression_coeffs(median_pixel_flux[mask],
                                          pixel_fluxes[mask],
                                          pixel_errors[mask])
                    m = regression_model(c, median_pixel_flux[mask])
                    sigmas = np.abs(pixel_fluxes[mask] - m)/pixel_errors[mask]
                    max_sigma_index = np.argmax(sigmas)

                    # If max outlier is >3 sigma from model, mask it and refit
                    if sigmas[max_sigma_index] > 3 and np.count_nonzero(mask) > 3:
                        mask[inds[max_sigma_index]] = False

                    # If max outlier is <3 sigma from model or there are 3 points
                    # left unmasked in the pixel flux series, use that coefficient
                    else:
                        break
                coefficients[i, j] = c[0] if not np.isnan(c[0]) else 1.0

    #np.save('coefficients.npy', coefficients)
    master_flat = coefficients/np.median(coefficients[coefficients != 1])
    fits.writeto(master_flat_path, master_flat, clobber=True)


def test_flat(image_path, master_flat_path, master_dark_path):
    import matplotlib.pyplot as plt

    image_no_flat = fits.getdata(image_path) - fits.getdata(master_dark_path)
    image = image_no_flat / fits.getdata(master_flat_path)

    fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    ax[0].imshow(image, origin='lower', interpolation='nearest',
                 cmap=plt.cm.viridis, vmin=np.percentile(image, 0.1),
                 vmax=np.percentile(image, 99.9))
    ax[1].hist(image_no_flat.ravel(), 200, label='No flat', alpha=0.4, log=True,
               histtype='stepfilled')
    ax[1].hist(image.ravel(), 200, label='Flat', alpha=0.4, log=True,
               histtype='stepfilled')
    ax[1].set_title(master_flat_path)
    ax[1].legend()

# import matplotlib.pyplot as plt
# full_m = model(c, median_pixel_flux)
# plt.errorbar(indices, pixel_fluxes, yerr=pixel_errors, fmt='o')
# plt.plot(indices, full_m)
# plt.plot(indices[~mask], pixel_fluxes[~mask],'ro')
# plt.title(c)
# plt.show()