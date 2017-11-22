import emcee
import george
import matplotlib.pyplot as plt
import numpy as np
from george import kernels
import astropy.units as u
from astropy.stats import mad_std
from toolkit import PhotometryResults, PCA_light_curve
from toolkit.transit_model import params_b, transit_model_b_t0

# Load data
path = 'outputs/trappist1b_20160711.npz'
phot_results = PhotometryResults.load(path)
times = phot_results.times
transit_parameters = params_b

expected_mid_transit_jd = ((np.max(np.abs(times - transit_parameters.t0) //
                                       transit_parameters.per) ) *
                               transit_parameters.per + transit_parameters.t0)
transit_parameters = params_b
light_curve = PCA_light_curve(phot_results, transit_parameters, plots=False,
                              plot_validation=False, buffer_time=1*u.min,
                              validation_duration_fraction=0.7,
                              validation_time=-1)
light_curve_errors = np.ones_like(light_curve) * mad_std(light_curve)

init_transit_model = transit_model_b_t0(times, params_b.t0)

residuals = light_curve - init_transit_model

original_time_series = dict(times=times, light_curve=light_curve,
                            light_curve_errors=light_curve_errors,
                            residuals=residuals)

n_iterations = 5
inliers = np.zeros_like(residuals).astype(bool)
while np.count_nonzero(~inliers) > 0:
    inliers = np.abs(residuals) < 2.5*mad_std(residuals)
    print('remove {0}'.format(np.count_nonzero(~inliers)))
    times = times[inliers]
    light_curve = light_curve[inliers]
    light_curve_errors = light_curve_errors[inliers]
    residuals = residuals[inliers]

times = times[inliers]
light_curve = light_curve[inliers]
light_curve_errors = light_curve_errors[inliers]
residuals = residuals[inliers]


def model2(theta, times):
    t0, lna, lntau, lnw = theta
    return transit_model_b_t0(times, t0)


def lnlike2(theta, t, y, yerr):
    t0, lna, lntau, lnw = theta
    gp = george.GP(np.exp(lna) * kernels.Matern32Kernel(np.exp(lntau)))
    gp.compute(t, np.sqrt(yerr**2 + np.exp(2*lnw)))
    return gp.lnlikelihood(y - model2(theta, t))


def lnprior2(theta):
    t0, lna, lntau, lnw = theta

    if (expected_mid_transit_jd - 0.01 < t0 < expected_mid_transit_jd + 0.01 and
        -15 < lntau < 5 and -20 < lna < 10.0 and -11 < lnw < 10.0):
        # Prior on depth from the discovery paper:
        #return -0.5 * (depth - params_b.rp**2)**2 / params_b.depth_error**2
        return 0
    return -np.inf


def lnprob2(theta, x, y, yerr):
    lp = lnprior2(theta)
    return lp + lnlike2(theta, x, y, yerr) if np.isfinite(lp) else -np.inf

if __name__ == '__main__':
    plt.plot(original_time_series['times'], original_time_series['residuals'], 'r.')
    plt.plot(times, residuals, 'b.')
    plt.show()

    initp = np.array([expected_mid_transit_jd])
    init_transit_model = transit_model_b_t0(times, initp[0])

    # Initial guesses based on iterative guessing
    lna_init = -0.1
    lntau_init = np.log((2/24)**2)
    lnw_init = np.log(np.median(light_curve_errors))

    initial = np.concatenate([initp, [lna_init, lntau_init, lnw_init]])

    ndim, nwalkers = len(initial), 2*len(initial)
    n_steps = 10000

    # Set initial walker positions
    load_init_walker_positions = False
    if not load_init_walker_positions:
        pos = []
        while len(pos) < nwalkers:
            proposed_pos = initial.copy()
            proposed_pos += np.array([0.03*p*np.random.randn()
                                      if i != 0 else 3e-8*p*np.random.randn()
                                      for i, p in enumerate(proposed_pos)])

            if np.isfinite(lnprior2(proposed_pos)):
                pos.append(proposed_pos)
    else:
        pos = np.load('outputs/samples_near_convergence.npy')[-nwalkers:, :]

    print('begin mcmc')
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob2, threads=4,
                                    args=(times, light_curve,
                                          light_curve_errors))
    sampler.run_mcmc(pos, n_steps)
    burn_in = 1000
    samples = sampler.chain[:, burn_in:, :].reshape((-1, ndim))

    import corner

    np.save('outputs/samples_b.npy', samples)

    corner.corner(samples[::10, :], labels=['depth', 't0', 'lna', 'lntau', 'lnw'])

    plt.show()
