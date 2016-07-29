import emcee
import george
import matplotlib.pyplot as plt
import numpy as np
from george import kernels
from astropy.stats import mad_std

from toolkit.transit_model import params_b, transit_model_b_depth_t0

# Load data
times = np.load('outputs/times.npy')
light_curve = np.load('outputs/bestlc.npy')
light_curve_errors = np.load('outputs/bestlc_errors.npy')

init_transit_model = transit_model_b_depth_t0(times, params_b.rp**2,
                                              params_b.t0)

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

# plt.plot(original_time_series['times'], original_time_series['residuals'], 'r.')
# plt.plot(times, residuals, 'b.')
# #plt.plot(times[inliers], residuals[inliers], '.')
# plt.show()

times = times[inliers]
light_curve = light_curve[inliers]
light_curve_errors = light_curve_errors[inliers]
residuals = residuals[inliers]


def model2(theta, times):
    depth, t0, lna, lntau, lnw = theta
    # depth, t0, tau = theta

    return transit_model_b_depth_t0(times, depth, t0)


def lnlike2(theta, t, y, yerr):
    depth, t0, lna, lntau, lnw = theta
    # depth, t0, tau = theta

    gp = george.GP(np.exp(lna) * kernels.Matern32Kernel(np.exp(lntau)),
                   solver=george.HODLRSolver)
    gp.compute(t, np.sqrt(yerr**2 + np.exp(2*lnw)))
    return gp.lnlikelihood(y - model2(theta, t))


def lnprior2(theta):
    depth, t0, lna, lntau, lnw = theta
    # depth, t0, tau = theta

    if (0 < depth < 1 and params_b.t0 - 0.01 < t0 < params_b.t0 + 0.01 and
        #-15 < lntau < 1 and -20 < lna < 0.0 and -13 < lnw < 0.0):
        -15 < lntau < 1 and -20 < lna < 0.0 and -11 < lnw < 0.0):
        # Prior on depth from the discovery paper:
        return -0.5 * (depth - params_b.rp**2)**2 / params_b.depth_error**2
    return -np.inf


def lnprob2(theta, x, y, yerr):
    lp = lnprior2(theta)
    return lp + lnlike2(theta, x, y, yerr) if np.isfinite(lp) else -np.inf

if __name__ == '__main__':

    # Set init transit parameters to those from discovery paper
    initp = np.array([params_b.rp**2, params_b.t0])
    init_transit_model = transit_model_b_depth_t0(times, initp[0], initp[1])

    # Compute acf, find autoregressive period
    # lags, acf = interpolated_acf(times, light_curve - init_transit_model)
    # residual_period = dominant_period(lags, acf, fwhm=20, plot=True,
    #                                   min=0.005, max=0.02)
    # print(residual_period)
    # plt.show()

    # Initial guesses based on iterative guessing
    lna_init = -0.1
    lntau_init = np.log((2/24)**2)
    lnw_init = np.log(np.median(light_curve_errors))

    initial = np.concatenate([initp, [lna_init, lntau_init, lnw_init]])

    ndim, nwalkers = len(initial), 4*len(initial)
    n_steps = 2000

    # Set initial walker positions
    load_init_walker_positions = False
    if not load_init_walker_positions:
        pos = []
        while len(pos) < nwalkers:
            proposed_pos = initial.copy()
            proposed_pos += np.array([0.03*p*np.random.randn()
                                      if i != 1 else 3e-8*p*np.random.randn()
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

    np.save('outputs/samples.npy', samples)

    corner.corner(samples[::10, :], labels=['depth', 't0', 'lna', 'lntau', 'lnw'])

    plt.show()
