from glob import glob
import os

import numpy as np
from astropy.utils.console import ProgressBar
from astropy.stats import sigma_clipped_stats
from astropy.io import fits
from photutils import daofind
from sklearn import linear_model

# Paths to data files:
allflats_path = 'allflats.npy'
flat_dark_30s_paths = glob('/Users/bmmorris/data/UT160711/dark_quad_30.*.fits')
flat_paths = glob('/Users/bmmorris/data/UT160711/nightskyflatz.*.fits')
master_flat_path = 'masterflat.fits'

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
        flat_dark_subtracted = fits.getdata(flatpath) - masterflatdark * 2

        mean, median, std = sigma_clipped_stats(flat_dark_subtracted,
                                                sigma=3.0, iters=5)
        sources = daofind(flat_dark_subtracted - median, fwhm=4.0,
                          threshold=5.0*std)

        replace_radius = 25
        masked_image = np.ones_like(flat_dark_subtracted).astype(bool)
        x, y = np.meshgrid(range(masked_image.shape[1]),
                           range(masked_image.shape[0]))
        for x_centroid, y_centroid in zip(sources['xcentroid'],
                                          sources['ycentroid']):
            not_near_star = ((x - x_centroid)**2 + (y - y_centroid)**2 >
                             replace_radius)
            masked_image &= not_near_star

        flat_dark_subtracted_cleaned = flat_dark_subtracted.copy()
        flat_dark_subtracted_cleaned[~masked_image] = np.median(flat_dark_subtracted)

        allflats[:, :, i] = flat_dark_subtracted_cleaned

    np.save(allflats_path, allflats)
else:
    allflats = np.load(allflats_path)

intercepts = np.zeros((allflats.shape[0], allflats.shape[1]), dtype=float)
order = 1
delta_std = 1.15
    
X = np.arange(allflats.shape[2])[:, np.newaxis]

margin = 50
with ProgressBar(allflats.shape[0]*allflats.shape[1]) as bar:
    for i in range(margin, allflats.shape[0]-margin):
        for j in range(margin, allflats.shape[1]-margin):
            bar.update()

            y = allflats[i, j, :]

            try:
                linreg = linear_model.LinearRegression()
                model_ransac = linear_model.RANSACRegressor(linreg)
                model_ransac.fit(X, y)

                intercepts[i, j] = model_ransac.estimator_.intercept_
            except ValueError:
                intercepts[i, j] = np.median(y)

master_flat = intercepts/np.median(intercepts[intercepts != 0])
master_flat[master_flat == 0] = np.median(master_flat[margin:master_flat.shape[0] - margin,
                                                      margin:master_flat.shape[1] - margin])
fits.writeto(master_flat_path, master_flat, clobber=True)