import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

from toolkit import (generate_master_dark, photometry, transit_model_c,
                     PhotometryResults, PCA_light_curve, params_c)

# Image paths
image_paths = sorted(glob('/Users/bmmorris/data/UT160619/trappist-1.*.fits'))[:-10]
dark_paths = glob('/Users/bmmorris/data/UT160619/dark-45s.*.fits')
night_flat_paths = glob('/Users/bmmorris/data/UT160711/nightskyflatz.*.fits')
master_flat_path = 'outputs/masterflat_binning3x3.fits' #'outputs/masterflat_placebo.fits'  #
master_dark_path = 'outputs/masterdark_binning3x3.fits'

# Photometry settings
target_centroid = [633, 579]#np.array([[633], [579]])
comparison_flux_threshold = 0.2
aperture_radii = np.arange(7, 30)
centroid_stamp_half_width = 5
psf_stddev_init = 2
aperture_annulus_radius = 10
transit_parameters = params_c


output_path = 'outputs/trappist1c_20160619.npz'
force_recompute_photometry = False #True

# Calculate master dark/flat:
if not os.path.exists(master_dark_path):
    generate_master_dark(dark_paths, master_dark_path)

# Do photometry:
if not os.path.exists(output_path) or force_recompute_photometry:
    print('Calculating photometry:')
    phot_results = photometry(image_paths, master_dark_path, master_flat_path,
                              target_centroid, comparison_flux_threshold,
                              aperture_radii, centroid_stamp_half_width,
                              psf_stddev_init, aperture_annulus_radius,
                              output_path)

else:
    phot_results = PhotometryResults.load(output_path)

print('Calculating PCA...')
light_curve = PCA_light_curve(phot_results, transit_parameters,
                              validation_duration_fraction=0.5, plots=True,
                              plot_validation=True, validation_time=0.9,
                              outlier_rejection=False)


plt.figure()
plt.plot(phot_results.times, light_curve, 'k.')
plt.plot(phot_results.times, transit_model_c(phot_results.times), 'r')
plt.xlabel('Time [JD]')
plt.ylabel('Flux')
plt.show()