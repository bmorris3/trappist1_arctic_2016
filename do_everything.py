import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

from toolkit import reduce, PhotometryResults, PCA_light_curve, params_b

image_paths = sorted(glob('/Users/bmmorris/data/UT160711/TRAPPIST1.*.fits'))[:-130]
master_flat_path = 'outputs/masterflat_morning_linreg.fits'
master_dark_path = 'outputs/masterdark.fits'

# Photometry settings
target_centroid = np.array([[456], [465]])
comparison_flux_threshold = 0.15
aperture_radii = np.arange(7, 18)
centroid_stamp_half_width = 5
psf_stddev_init = 2
aperture_annulus_radius = 10
transit_parameters = params_b

output_path = 'outputs/trappist1b_20160711.npz'
force_recompute_photometry = False  # True


if not os.path.exists(output_path) or force_recompute_photometry:
    print('Calculating photometry:')
    phot_results = reduce(image_paths, master_dark_path, master_flat_path,
                          target_centroid, comparison_flux_threshold,
                          aperture_radii, centroid_stamp_half_width,
                          psf_stddev_init, aperture_annulus_radius, output_path)

else:
    phot_results = PhotometryResults.load(output_path)

print('Calculating PCA...')
light_curve = PCA_light_curve(phot_results, transit_parameters)



plt.figure()
plt.plot(phot_results.times, light_curve, 'k.')
plt.xlabel('Time [JD]')
plt.ylabel('Flux')
plt.show()