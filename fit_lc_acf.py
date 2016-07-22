import emcee
import george
import matplotlib.pyplot as plt
import numpy as np
from george import kernels
from interpacf import interpolated_acf, dominant_period
from astropy.stats import mad_std

from toolkit.transit_model import params_b, transit_model_b_depth_t0

# Load data
times = np.load('outputs/times.npy')
light_curve = np.load('outputs/bestlc.npy')
light_curve_errors = np.load('outputs/bestlc_errors.npy')

init_transit_model = transit_model_b_depth_t0(times, params_b.rp**2,
                                              params_b.t0)

residuals = light_curve - init_transit_model

inliers = np.abs(residuals) < 3*mad_std(residuals)

times = times[inliers]
light_curve = light_curve[inliers]
light_curve_errors = light_curve_errors[inliers]
residuals = residuals[inliers]

def model2(theta, times):
    depth, t0, lna = theta
    return transit_model_b_depth_t0(times, depth, t0)


def lnlike2(theta, t, y, yerr, tau):
    depth, t0, lna = theta
    gp = george.GP(np.exp(lna) * kernels.Matern32Kernel(tau),
                   solver=george.HODLRSolver)
    gp.compute(t, yerr)
    return gp.lnlikelihood(y - model2(theta, t))


def lnprior2(theta):
    depth, t0, lna = theta

    if (0 < depth < 1 and params_b.t0 - 0.01 < t0 < params_b.t0 + 0.01 and
        -20 < lna < 1):
        # Prior on depth from the discovery paper:
        return -0.5 * (depth - params_b.rp**2)**2/0.00025**2
    return -np.inf


def lnprob2(theta, x, y, yerr, tau):
    lp = lnprior2(theta)
    return lp + lnlike2(theta, x, y, yerr, tau) if np.isfinite(lp) else -np.inf

if __name__ == '__main__':

    # Set init transit parameters to those from discovery paper
    initp = np.array([params_b.rp**2, params_b.t0])
    init_transit_model = transit_model_b_depth_t0(times, initp[0], initp[1])

    # Compute acf, find autoregressive period
    lags, acf = interpolated_acf(times, light_curve - init_transit_model)
    residual_period = dominant_period(lags, acf, fwhm=20, plot=True,
                                      min=0.005, max=0.02)
    print(residual_period)
    plt.show()


    # Initial guesses based on iterative guessing
    lna_init = -2  #-5.16520654, -5.33 #1.04027941e-05
    initial = np.concatenate([initp, [lna_init]])

    ndim, nwalkers = len(initial), 8*len(initial)
    n_steps = 3000

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
                                          light_curve_errors, residual_period))
    sampler.run_mcmc(pos, n_steps)
    burn_in = 2000
    samples = sampler.chain[:, burn_in:, :].reshape((-1, ndim))

    import corner

    np.save('outputs/samples.npy', samples)

    corner.corner(samples[::10, :], labels=['depth', 't0', 'lna'])

    plt.show()
