# trappist1_arctic_2016

APO ARC 3.5 m/ARCTIC photometry of TRAPPIST-1

The ``toolkit`` directory contains a light Python package for analyzing TRAPPIST-1 light curves with ARCTIC. It depends on the following Python packages: 

* photutils
* astropy
* numpy
* matplotlib
* scikit-learn
* scikit-image
* pyfftw
* batman


#### Workflow

1. Run ``trappist1b_20160711.py`` to compute photometry for planet b
2. Run ``fit_lc_acf_b.py`` to run the MCMC+GP sampler for planet b
3. Run ``vis_mcmc_b.py`` to make plots showing the results for planet b
4. Run ``trappist1c_20160619.py`` to compute photometry for planet c
5. Run ``fit_lc_acf_c.py`` to run the MCMC+GP sampler for planet c
6. Run ``vis_mcmc_c.py`` to make plots showing the results for planet c

#### Results

The posterior samples for the mid-transit times of planets b and c are stored in the ``outputs`` directory as HDF5 archives. 