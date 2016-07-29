import george
import matplotlib.pyplot as plt
import numpy as np
from george import kernels
from toolkit.transit_model import params_b, transit_model_b_depth_t0
from astropy.stats import mad_std

samples = np.load('outputs/samples.npy')#np.loadtxt('outputs/samples_converged.txt')
times = np.load('outputs/times.npy')
light_curve = np.load('outputs/bestlc.npy')
light_curve_errors = np.load('outputs/bestlc_errors.npy')

init_transit_model = transit_model_b_depth_t0(times, params_b.rp**2,
                                              params_b.t0)

residuals = light_curve - init_transit_model

n_iterations = 5
inliers = np.zeros_like(residuals).astype(bool)
while np.count_nonzero(~inliers) > 0:
    inliers = np.abs(residuals) < 2.5*mad_std(residuals)
    print('remove {0}'.format(np.count_nonzero(~inliers)))
    times = times[inliers]
    light_curve = light_curve[inliers]
    light_curve_errors = light_curve_errors[inliers]
    residuals = residuals[inliers]

depth_mcmc, t0_mcmc, lna_mcmc, lntau_mcmc, lnw_mcmc = map(lambda v: (v[1], v[2]-v[1],
# depth_mcmc, t0_mcmc, tau_mcmc = map(lambda v: (v[1], v[2]-v[1],
                                                       v[1]-v[0]),
                                              zip(*np.percentile(samples, [16, 50, 84], axis=0)))

from fit_lc_acf import model2, transit_model_b_depth_t0
model1 = model2

bestp = [i[0] for i in
         [depth_mcmc, t0_mcmc, lna_mcmc, lntau_mcmc, lnw_mcmc]]

model_nogp = transit_model_b_depth_t0(times, depth_mcmc[0], t0_mcmc[0])

from toolkit.transit_model import params_b, transit_model_b_depth_t0
from interpacf import interpolated_acf, dominant_period

initp = np.array([params_b.rp**2, params_b.t0])

init_transit_model = transit_model_b_depth_t0(times, *initp)

lags, acf = interpolated_acf(times, light_curve - init_transit_model)
residual_period = dominant_period(lags, acf, fwhm=20,
                                  min=0.005, max=0.02)

gp = george.GP(np.exp(lna_mcmc[0]) * kernels.ExpKernel(np.exp(lntau_mcmc[0])))
scaled_errors = np.sqrt(light_curve_errors**2 + np.exp(2*lnw_mcmc[0]))
gp.compute(times, scaled_errors)

mu, cov = gp.predict(light_curve - model_nogp, times)

fig, ax = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

ax[0].errorbar(times, light_curve, yerr=scaled_errors, color='k', fmt='.',
               ecolor='silver')
ax[0].plot(times, model_nogp + mu, 'r', lw=2)

ax[1].errorbar(times, light_curve - mu, yerr=scaled_errors,
               color='k', fmt='.', ecolor='silver')
ax[1].plot(times, model_nogp, 'r', lw=2)

ax[2].errorbar(times, light_curve - model_nogp - mu, yerr=scaled_errors,
               color='k', fmt='.', ecolor='silver')
ax[2].axhline(0, ls='--')

ax[0].set_xlim([times.min(), times.max()])
ax[0].set_title('GP + Transit')
ax[1].set_title('Transit')
ax[2].set_title('Residuals')
plt.show()

# examine correlated noise:
residuals = light_curve - model_nogp - mu



#n_bins_range = np.unique(np.logspace(np.log(2), np.log10(len(residuals)), n_trials,
#                                     dtype=int))
#n_trials = len(n_bins_range)
#bin_widths = np.zeros(n_trials)
#noise = np.zeros(n_trials)
# for i, n_bins in enumerate(n_bins_range):
#     stop_number = (len(residuals) // n_bins) * n_bins
#     split_residuals = np.split(residuals[:stop_number], n_bins)
#     noise[i] = np.std(np.mean(split_residuals, axis=1))
#     bin_widths[i] = len(split_residuals[0])

# def rebin(array, bin_width):
#     new_shape = array.shape[0]//bin_width, bin_width
#     return np.resize(array, new_shape).mean(-1)
#
# n_trials = 200
# bin_widths = np.unique(np.rint(np.logspace(0, np.log10(len(residuals)), n_trials)))
# n_trials = len(bin_widths)
# noise = np.zeros(n_trials)
#
# for i, bin_width in enumerate(bin_widths):
#     noise[i] = np.std(rebin(residuals, bin_width))
#
# plt.loglog(bin_widths, noise)
# plt.loglog(bin_widths, np.std(residuals)*bin_widths**-0.5)
# plt.loglog(bin_widths, np.median(light_curve_errors)*bin_widths**-0.5)
# plt.show()
#
#
#
