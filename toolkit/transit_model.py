from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import batman
from copy import deepcopy
import astropy.units as u

__all__ = ['transit_model_b', 'transit_model_c', 'transit_model_b_depth_t0',
           'transit_model_c_depth_t0', 'params_b', 'params_c']

# Initialize transit parameter objects with the properties of TRAPPIST-1 b, c
# Planet b:
params_b = batman.TransitParams()
params_b.per = 1.510848
params_b.t0 = 2450000 + 7322.51765
params_b.inc = 89.41
params_b.rp = np.sqrt(0.00754)
params_b.a = 20.45
params_b.ecc = 0
params_b.w = 90
params_b.u = [0.65, 0.28]
params_b.limb_dark = 'quadratic'

params_b.depth_error = 0.00025
params_b.duration = 36.12 * u.min

# Planet c:
params_c = batman.TransitParams()
params_c.per = 2.421848
params_c.t0 = 2450000 + 7362.72618
params_c.inc = 89.50
params_c.rp = np.sqrt(0.00672)
params_c.a = 28.0
params_c.ecc = 0
params_c.w = 90
params_c.u = [0.65, 0.28]
params_c.limb_dark = 'quadratic'

params_c.depth_error = 0.042
params_c.duration = 41.78 * u.min


# Planet d:
params_d = batman.TransitParams()
params_d.per = 18.20175
params_d.t0 = 2450000 + 7294.7741
params_d.inc = 89.87
params_d.rp = np.sqrt(0.00826)

# Estimate a/R* for planet d:
p = np.array([params_b.per, params_c.per])
a = np.array([params_b.a, params_c.a])
mean_ratio = np.mean(p**2 / a**3)
a_estimate = (params_d.per**2 / mean_ratio)**(1/3)

params_d.a = a_estimate
params_d.ecc = 0
params_d.w = 90
params_d.u = [0.65, 0.28]
params_d.limb_dark = 'quadratic'

params_d.depth_error = 0.073
params_d.duration = 83.3 * u.min


def transit_model_b(times, params=params_b):
    """
    Get a transit model for TRAPPIST-1b at ``times``.

    Parameters
    ----------
    times : `~numpy.ndarray`
        Times in Julian date
    params : `~batman.TransitParams`
        Transiting planet parameters

    Returns
    -------
    flux : `numpy.ndarray`
        Fluxes at each time
    """
    m = batman.TransitModel(params, times)
    model = m.light_curve(params)
    return model


def transit_model_c(times, params=params_c):
    """
    Get a transit model for TRAPPIST-1c at ``times``.

    Parameters
    ----------
    times : `~numpy.ndarray`
        Times in Julian date
    params : `~batman.TransitParams`
        Transiting planet parameters

    Returns
    -------
    flux : `numpy.ndarray`
        Fluxes at each time
    """
    m = batman.TransitModel(params, times)
    model = m.light_curve(params)
    return model


def transit_model_d(times, params=params_d):
    """
    Get a transit model for TRAPPIST-1c at ``times``.

    Parameters
    ----------
    times : `~numpy.ndarray`
        Times in Julian date
    params : `~batman.TransitParams`
        Transiting planet parameters

    Returns
    -------
    flux : `numpy.ndarray`
        Fluxes at each time
    """
    m = batman.TransitModel(params, times)
    model = m.light_curve(params)
    return model


def transit_model_b_depth_t0(times, depth, t0, f0=1.0):
    """
    Get a transit model for TRAPPIST-1b at ``times``.

    Parameters
    ----------
    times : `~numpy.ndarray`
        Times in Julian date
    depth : float
        Depth of transit
    t0 : float
        Mid-transit time [JD]
    Returns
    -------
    flux : `numpy.ndarray`
        Fluxes at each time
    """
    params = deepcopy(params_b)
    params.t0 = t0
    params.rp = np.sqrt(depth)
    m = batman.TransitModel(params, times)
    model = f0*m.light_curve(params)
    return model


def transit_model_c_depth_t0(times, depth, t0):
    """
    Get a transit model for TRAPPIST-1c at ``times``.

    Parameters
    ----------
    times : `~numpy.ndarray`
        Times in Julian date
    depth : float
        Depth of transit
    t0 : float
        Mid-transit time [JD]
    Returns
    -------
    flux : `numpy.ndarray`
        Fluxes at each time
    """
    params = deepcopy(params_c)
    params.t0 = t0
    params.rp = np.sqrt(depth)
    m = batman.TransitModel(params, times)
    model = m.light_curve(params)
    return model
