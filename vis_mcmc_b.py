import george
import matplotlib.pyplot as plt
import numpy as np
from george import kernels
from toolkit import PCA_light_curve, PhotometryResults
from toolkit.transit_model import params_b, transit_model_b_t0
import astropy.units as u
from astropy.stats import mad_std

samples = np.load('outputs/samples.npy')[1000:, :]#np.loadtxt('outputs/samples_converged.txt')
path = 'outputs/trappist1b_20160711.npz'
phot_results = PhotometryResults.load(path)
times = phot_results.times

transit_parameters = params_b
light_curve = PCA_light_curve(phot_results, transit_parameters, plots=False,
                              plot_validation=False, buffer_time=1*u.min,
                              validation_duration_fraction=0.7,
                              validation_time=-1)
light_curve_errors = np.ones_like(light_curve) * mad_std(light_curve)

init_transit_model = transit_model_b_t0(times, params_b.t0)

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

t0_mcmc, lna_mcmc, lntau_mcmc, lnw_mcmc = map(lambda v: (v[1], v[2]-v[1],
# depth_mcmc, t0_mcmc, tau_mcmc = map(lambda v: (v[1], v[2]-v[1],
                                                       v[1]-v[0]),
                                              zip(*np.percentile(samples, [16, 50, 84], axis=0)))

np.savetxt('outputs/t0_b.txt', t0_mcmc)

from fit_lc_acf_b import model2
model1 = model2

bestp = [i[0] for i in
         [t0_mcmc, lna_mcmc, lntau_mcmc, lnw_mcmc]]

model_nogp = transit_model_b_t0(times, t0_mcmc[0])

# from interpacf import interpolated_acf, dominant_period

initp = np.array([params_b.t0])

init_transit_model = transit_model_b_t0(times, *initp)

# lags, acf = interpolated_acf(times, light_curve - init_transit_model)
# residual_period = dominant_period(lags, acf, fwhm=20,
#                                   min=0.005, max=0.02)

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
fig.savefig('plots/transit_b.png', bbox_inches='tight', dpi=200)

plt.figure()
plt.hist(samples[0, :], 30, range=[times.min(), times.max()])
plt.xlabel('JD')
plt.savefig('plots/t0_b.png', bbox_inches='tight', dpi=200)
plt.show()
# examine correlated noise:
#residuals = light_curve - model_nogp - mu

