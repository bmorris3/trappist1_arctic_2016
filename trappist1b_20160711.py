import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

from toolkit import (generate_master_flat_and_dark, photometry,
                     PhotometryResults, PCA_light_curve, params_b)

# Image paths
image_paths = sorted(glob('/Users/bmmorris/data/UT160711/TRAPPIST1.*.fits'))[:-130]
dark_30s_paths = glob('/Users/bmmorris/data/UT160711/dark_quad_30.*.fits')
night_flat_paths = glob('/Users/bmmorris/data/UT160711/nightskyflatz.*.fits')
master_flat_path = 'outputs/masterflat.fits'
master_dark_path = 'outputs/masterdark.fits'

# Photometry settings
target_centroid = [465, 456]
comparison_flux_threshold = 0.05
aperture_radii = np.arange(7, 25)
centroid_stamp_half_width = 3
psf_stddev_init = 2
aperture_annulus_radius = 10
transit_parameters = params_b

output_path = 'outputs/trappist1b_20160711.npz'
force_recompute_photometry = False # True

# Calculate master dark/flat:
if not os.path.exists(master_dark_path) or not os.path.exists(master_flat_path):
    print('Calculating master flat:')
    generate_master_flat_and_dark(night_flat_paths, dark_30s_paths,
                                  master_flat_path, master_dark_path)

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
import astropy.units as u
light_curve = PCA_light_curve(phot_results, transit_parameters, plots=True,
                              plot_validation=False, buffer_time=1*u.min,
                              validation_duration_fraction=0.7,
                              validation_time=-1, outlier_rejection=True)

plt.figure()
plt.plot(phot_results.times, light_curve, 'k.')
plt.xlabel('Time [JD]')
plt.ylabel('Flux')
plt.show()