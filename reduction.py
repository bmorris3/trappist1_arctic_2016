from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from glob import glob
import os

import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt

from astropy.io import fits
from astropy.time import Time
from astropy.utils.console import ProgressBar
from photutils.morphology import centroid_2dg
from photutils import CircularAperture, CircularAnnulus, aperture_photometry

########################################################################
## Settings:
make_master_dark = True
master_flat = fits.getdata('outputs/masterflat_morning_linreg.fits')

# A DS9 regions file defines the initial guesses for stellar centroids,
# the first star in the regions file is TRAPPIST-1
regions_file_path = 'outputs/trappist_5.reg'
master_dark_path = 'masterdark.fits'

# Collect images for photometry
images = sorted(glob('/Users/bmmorris/data/UT160711/TRAPPIST1.*.fits'))
# Ignore two images with obvious cosmic rays:
images.pop(images.index('/Users/bmmorris/data/UT160711/TRAPPIST1.0143.fits'))
images.pop(images.index('/Users/bmmorris/data/UT160711/TRAPPIST1.0295.fits'))
# Ignore last images near sunrise
images = images[:-130]

dark_paths = glob('/Users/bmmorris/data/UT160711/dark_quad_30.*.fits')

aperture_radii = np.arange(7, 18)
centroid_stamp_half_width = 5

########################################################################

# Use the regions file to find initial stellar centroid guesses
star_positions = []
counter = 0
for line in open(regions_file_path, 'r').read().splitlines():
    if line.startswith('circle'):
        y_center, x_center = map(float,
                                 line.split('(')[1].split(')')[0].split(',')[:2])
        star_positions.append([y_center, x_center])
        counter += 1

# Make the calibration frames
if make_master_dark or not os.path.exists(master_dark_path):
    testdata = fits.getdata(dark_paths[0])
    alldarks = np.zeros((testdata.shape[0], testdata.shape[1], len(dark_paths)))
    for i, dark_path in enumerate(dark_paths):
        alldarks[:, :, i] = fits.getdata(dark_path)

    # darks=30 s, target exposures=10 s
    master_dark = np.median(alldarks, axis=2)

    fits.writeto(master_dark_path, master_dark, clobber=True)
else:
    master_dark = fits.getdata(master_dark_path)

# Initialize some empty arrays to fill with data:
times = np.zeros(len(images))
fluxes = np.zeros((len(images), len(star_positions), len(aperture_radii)))
errors = np.zeros((len(images), len(star_positions), len(aperture_radii)))
xcentroids = np.zeros((len(images),len(star_positions)))
ycentroids = np.zeros((len(images),len(star_positions)))
airmass = np.zeros(len(images))
airpress = np.zeros(len(images))
humidity = np.zeros(len(images))
telfocus = np.zeros(len(images))

means = np.zeros((len(images), len(aperture_radii)))
medians = np.zeros(len(images))

with ProgressBar(len(images)) as bar:
    for i in range(len(images)):
        bar.update()

        # Subtract image by the dark frame, normalize by flat field
        imagedata = (fits.getdata(images[i]) - master_dark) / master_flat

        # Collect information from the header
        imageheader = fits.getheader(images[i])
        times[i] = Time(imageheader['DATE-OBS'], format='isot', scale='utc').jd
        medians[i] = np.median(imagedata)
        airmass[i] = imageheader['AIRMASS']
        airpress[i] = imageheader['AIRPRESS']
        humidity[i] = imageheader['HUMIDITY']
        telfocus[i] = imageheader['TELFOCUS']

        # Initial guess for each stellar centroid informed by previous centroid
        for j in range(len(star_positions)):
            if i == 0:
                init_x = star_positions[j][0]
                init_y = star_positions[j][1]
            else:
                init_x = ycentroids[i-1][j]
                init_y = xcentroids[i-1][j]

            # Cut out a stamp of the full image centered on the star
            image_stamp = imagedata[init_y - centroid_stamp_half_width:
                                    init_y + centroid_stamp_half_width,
                                    init_x - centroid_stamp_half_width:
                                    init_x + centroid_stamp_half_width]

            # Measure stellar centroid with 2D gaussian fit
            x_stamp_centroid, y_stamp_centroid = centroid_2dg(image_stamp)
            y_centroid = x_stamp_centroid + init_x - centroid_stamp_half_width
            x_centroid = y_stamp_centroid + init_y - centroid_stamp_half_width

            xcentroids[i, j] = x_centroid
            ycentroids[i, j] = y_centroid

        positions = np.vstack([ycentroids[i, :], xcentroids[i, :]])

        for k, aperture_radius in enumerate(aperture_radii):
            target_apertures = CircularAperture(positions, aperture_radius)
            background_annuli = CircularAnnulus(positions,
                                                r_in=aperture_radius,
                                                r_out=aperture_radius + 10)
            flux_in_annuli = aperture_photometry(imagedata,
                                                  background_annuli)['aperture_sum'].data
            background = flux_in_annuli/background_annuli.area()
            flux = aperture_photometry(imagedata,
                                       target_apertures)['aperture_sum'].data
            background_subtracted_flux = (flux - background *
                                          target_apertures.area())

            fluxes[i, :, k] = background_subtracted_flux
            errors[i, :, k] = np.sqrt(flux)

        # for ap, aperture_radius in enumerate(aperture_radii):
        #     position = [y_centroid, x_centroid]
        #     target_aperture = CircularAperture(position, aperture_radius)
        #     annulus_aperture = CircularAnnulus(position,
        #                                        r_in=aperture_radius,
        #                                        r_out=aperture_radius + 10)
        #     flux_in_annulus = aperture_photometry(imagedata,
        #                                           annulus_aperture)['aperture_sum'].data
        #     background = flux_in_annulus/annulus_aperture.area()
        #     flux = aperture_photometry(imagedata - background,
        #                                target_aperture)['aperture_sum'].data
        #
        #     fluxes[i, j, ap] = flux
        #     errors[i, j, ap] = np.sqrt(flux + background*flux.size)


## Save some values
np.save('outputs/xcentroids.npy', xcentroids)
np.save('outputs/ycentroids.npy', ycentroids)
np.save('outputs/fluxes.npy', fluxes)
np.save('outputs/errors.npy', errors)
np.save('outputs/airmass.npy', airmass)
np.save('outputs/airpress.npy', airpress)
np.save('outputs/humidity.npy', humidity)
np.save('outputs/telfocus.npy', telfocus)
np.save('outputs/medians.npy', medians)
np.save('outputs/times.npy', times)

## Taken from git/research/koi351/variableaperture.ipynb
stds = np.zeros_like(aperture_radii, dtype=float)
correctedlcs = []
correctedlc_errs = []

for j, ap in enumerate(aperture_radii):
    target = fluxes[:, 0, j]
    compStars = fluxes[:,1:, j]

    numCompStars = np.shape(fluxes[:, 1:, j])[1]
    initP = np.zeros([numCompStars]) + 1./numCompStars

    ########################################################

    def errfunc(p,target):
        if all(p >=0.0): 
            #return np.dot(p,target.T) - compStars ## Find only positive coefficients
            return np.dot(p,compStars[:,:].T) - target ## Find only positive coefficients

    #return np.dot(p,compStarsOOT.T) - target
    bestFitP = optimize.leastsq(errfunc, initP[:] ,args=(target.astype(np.float64)),
                                maxfev=10000000, epsfcn=np.finfo(np.float32).eps)[0]
    print('\nDefault weight:',1./numCompStars)
    print('Best fit regression coefficients:',bestFitP)

    #self.comparisonStarWeights = np.vstack([compStarKeys,bestFitP])
    meanComparisonStar = np.dot(bestFitP, compStars.T)

    meanCompError = np.zeros_like(meanComparisonStar)
    for i in range(1,len(fluxes[0, :, j])):
        meanCompError += ((bestFitP[i-1]*fluxes[:, i, j]/np.sum(bestFitP[i-1]*fluxes[:, i, j]))**2 *
                          (errors[:, i, j]/fluxes[:, i, j])**2)
    meanCompError = meanComparisonStar*np.sqrt(meanCompError)

    lc = fluxes[:, 0, j]/meanComparisonStar
    lc /= np.median(lc)
    lc_err = lc*np.sqrt((errors[:, 0, j]/fluxes[:, 0, j])**2 + (meanCompError/meanComparisonStar)**2)

    # Fit airmass to the light curve
    def amtrend(p, t, airmass=airmass):
        '''p = [OOT flux, airmass coeff]'''
        return p[0]*(1 + p[1]*(airmass-1))

    def errfunc(p, t, y, airmass=airmass):
        '''
        p = [OOT flux, airmass coeff]
        y = light curve
        '''
        return amtrend(p, t) - y

    bestp = optimize.leastsq(errfunc, [1.0, 0.0], args=(times, lc))[0]
    correctedlc = lc/amtrend(bestp, times)
    correctedlc_err = lc_err/amtrend(bestp, times)
    correctedcomplc = correctedlc
    if False:
        # Make lc out of just comp stars
        comp0 = fluxes[:, 1, j]
        comp1 = fluxes[:, 2, j]
        bestp = optimize.leastsq(errfunc, [1.0, 0.0], args=(times, comp0/comp1))[0]
        correctedcomplc = comp0/comp1/amtrend(bestp, times)
        if False:
            fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
            ax[0].plot(times, comp0/comp1, '.')
            ax[0].plot(times, amtrend(bestp, times),'r')
            ax[1].plot(times, correctedcomplc,'.')
            plt.show()

    stds[j] = np.std(correctedcomplc)
    correctedlcs.append(correctedlc)
    correctedlc_errs.append(correctedlc_err)

best_aperture_index = np.argmin(stds)
print("Best aperture: {0}".format(aperture_radii[best_aperture_index]))
correctedlc = correctedlcs[best_aperture_index]
correctedlc_errs = correctedlc_errs[best_aperture_index]


fig, ax = plt.subplots(2, 2, figsize=(10, 10))
mininttime = int(np.min(times))
legendprops = {'numpoints':1, 'loc':'lower center'}

ax[0, 0].plot(times - mininttime, target, '.', label='KOI-0377')

for i in range(compStars.shape[1]):
    ax[0, 0].plot(times - mininttime, compStars[:,i], '.', label='Comp {0}'.format(i))
ax[0, 0].set_xlabel('JD - {0:d}'.format(mininttime))
ax[0, 0].set_ylabel('Counts')
ax[0, 0].set_title('Raw fluxes')
#ax[0, 0].legend(**legendprops)

ax[0, 1].plot(times - mininttime, target, '.', label='Target')
ax[0, 1].plot(times - mininttime, meanComparisonStar, '.', label='Mean Comp')
ax[0, 1].set_xlabel('JD - {0:d}'.format(mininttime))
ax[0, 1].set_ylabel('Counts')
ax[0, 1].legend(**legendprops)
ax[0, 1].set_title('Mean comparison star')

ax[1, 0].errorbar(times - mininttime, correctedlc, yerr=correctedlc_err, 
                  fmt='.', color='k', ecolor='gray')
ax[1, 0].set_title('AM-corrected light curve')
ax[1, 0].set_xlabel('JD - {0:d}'.format(mininttime))
ax[1, 0].set_ylabel('Normalized flux')

plt.show()
