from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from glob import glob
import os

import numpy as np
from astropy.utils.console import ProgressBar
from astropy.io import fits

from toolkit import regression_model, regression_coeffs

# Paths to data files:
allflats_path = 'outputs/allflats.npy'
flat_dark_30s_paths = glob('/Users/bmmorris/data/UT160711/dark_quad_30.*.fits')
flat_paths = glob('/Users/bmmorris/data/UT160711/nightskyflatz.*.fits')
master_flat_path = 'outputs/masterflat_linreg.fits'

# Construct cube of night sky flats

if not os.path.exists(allflats_path):
    testdata = fits.getdata(flat_dark_30s_paths[0])
    alldarks = np.zeros((testdata.shape[0], testdata.shape[1],
                         len(flat_dark_30s_paths)))
    allflatdarks = np.zeros((testdata.shape[0], testdata.shape[1],
                             len(flat_dark_30s_paths)))
    for i, darkpath in enumerate(flat_dark_30s_paths):
        allflatdarks[:, :, i] = fits.getdata(darkpath)
    masterflatdark = np.median(allflatdarks, axis=2)

    testdata = fits.getdata(flat_paths[0])
    allflats = np.zeros((testdata.shape[0], testdata.shape[1], len(flat_paths)))

    for i, flatpath in enumerate(flat_paths):
        flat_dark_subtracted = fits.getdata(flatpath) - masterflatdark # * 2

        flat_dark_subtracted_cleaned = flat_dark_subtracted.copy()

        allflats[:, :, i] = flat_dark_subtracted_cleaned

    np.save(allflats_path, allflats)
else:
    allflats = np.load(allflats_path)

coefficients = np.ones((allflats.shape[0], allflats.shape[1]), dtype=float)

median_pixel_flux = np.atleast_2d(np.median(allflats, axis=(0, 1))).T

margin = 50

with ProgressBar(allflats.shape[0]*allflats.shape[1]) as bar:
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
                c = regression_coeffs(median_pixel_flux[mask], pixel_fluxes[mask],
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

np.save('coefficients.npy', coefficients)
master_flat = coefficients/np.median(coefficients[coefficients != 1])
fits.writeto(master_flat_path, master_flat, clobber=True)


            # import matplotlib.pyplot as plt
            # full_m = model(c, median_pixel_flux)
            # plt.errorbar(indices, pixel_fluxes, yerr=pixel_errors, fmt='o')
            # plt.plot(indices, full_m)
            # plt.plot(indices[~mask], pixel_fluxes[~mask],'ro')
            # plt.title(c)
            # plt.show()