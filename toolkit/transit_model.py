from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import batman
from copy import deepcopy
import astropy.units as u

__all__ = ['transit_model_b', 'transit_model_c', 'transit_model_b_depth_t0',
           'transit_model_c_depth_t0', 'params_b', 'params_c', 'params_d',
           'params_e', 'params_f', 'params_g', 'params_h']

# Initialize transit parameter objects with the properties of TRAPPIST-1 b, c
# Planet b:
params_b = batman.TransitParams()
params_b.per = 1.51087081
params_b.t0 = 2450000 + 7322.51736
params_b.inc = 89.65
params_b.rp = np.sqrt(0.7266/100)
params_b.a = 20.50
params_b.ecc = 0
params_b.w = 90
params_b.u = [0.65, 0.28]
params_b.limb_dark = 'quadratic'

params_b.depth_error = 0.00025
params_b.duration = 36.4 * u.min

# Planet c:
params_c = batman.TransitParams()
params_c.per = 2.4218233
params_c.t0 = 2450000 + 7282.80728
params_c.inc = 89.67
params_c.rp = np.sqrt(0.687/100)
params_c.a = 28.08
params_c.ecc = 0
params_c.w = 90
params_c.u = [0.65, 0.28]
params_c.limb_dark = 'quadratic'

params_c.depth_error = 0.00025
params_c.duration = 42.37 * u.min


# Planet d:
params_d = batman.TransitParams()
params_d.per = 4.049610
params_d.t0 = 2450000 + 7670.14165
params_d.inc = 89.75
params_d.rp = np.sqrt(0.367/100)
params_d.a = 39.55
params_d.ecc = 0
params_d.w = 90
params_d.u = [0.65, 0.28]
params_d.limb_dark = 'quadratic'

params_d.depth_error = 0.00025
params_d.duration = 49.13 * u.min

# Planet e:
params_e = batman.TransitParams()
params_e.per = 6.099615
params_e.t0 = 2450000 + 7660.37859
params_e.inc = 89.86
params_e.rp = np.sqrt(0.519/100)
params_e.a = 51.97
params_e.ecc = 0
params_e.w = 90
params_e.u = [0.65, 0.28]
params_e.limb_dark = 'quadratic'

params_e.depth_error = 1e-5
params_e.duration = 57.21 * u.min

# Planet f:
params_f = batman.TransitParams()
params_f.per = 9.206690
params_f.t0 = 2450000 + 7671.39767
params_f.inc = 89.680
params_f.rp = np.sqrt(0.673/100)
params_f.a = 68.4
params_f.ecc = 0
params_f.w = 90
params_f.u = [0.65, 0.28]
params_f.limb_dark = 'quadratic'

params_f.depth_error = 1e-5
params_f.duration = 62.60 * u.min


# Planet g:
params_g = batman.TransitParams()
params_g.per = 12.35294
params_g.t0 = 2450000 + 7665.34937
params_g.inc = 89.710
params_g.rp = np.sqrt(0.782/100)
params_g.a = 83.2
params_g.ecc = 0
params_g.w = 90
params_g.u = [0.65, 0.28]
params_g.limb_dark = 'quadratic'

params_g.depth_error = 0.073
params_g.duration = 68.40 * u.min


# Planet h:
params_h = batman.TransitParams()
params_h.per = 18.764
params_h.t0 = 2457700-37.44518
params_h.inc = 89.87
params_h.rp = 0.059
params_h.a = 114
params_h.ecc = 0
params_h.w = 90
params_h.u = [0.65, 0.28]
params_h.limb_dark = 'quadratic'

params_h.depth_error = 0.073

# Estimate duration of h:
b = 0.26
t_tot = np.arcsin(np.sqrt((1 + params_h.rp)**2 - b**2) / params_h.a /
                  np.sin(np.radians(params_h.inc))) * params_h.per*u.day/np.pi
params_h.duration = t_tot.to(u.min)

def transit_model(times, params):
    m = batman.TransitModel(params, times)
    model = m.light_curve(params)
    return model


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


def transit_model_b_t0(times, t0, f0=1.0):
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
    m = batman.TransitModel(params, times)
    model = f0*m.light_curve(params)
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
