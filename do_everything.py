import os
from glob import glob
import numpy as np

from toolkit import reduce, PhotometryResults

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

output_path = 'outputs/trappist1b.npz'
force_recompute_photometry = True


if not os.path.exists(output_path) or force_recompute_photometry:
    print('Photometry:')
    phot_results = reduce(image_paths, master_dark_path, master_flat_path,
                          target_centroid, comparison_flux_threshold,
                          aperture_radii, centroid_stamp_half_width,
                          psf_stddev_init, aperture_annulus_radius, output_path)

else:
    phot_results = PhotometryResults.load(output_path)


