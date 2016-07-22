"""
These regression methods are modified from Rodrigo Luger's regression methods
in Everest: https://github.com/rodluger/everest
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np

__all__ = ['regression_coeffs', 'regression_model']


def regression_coeffs(X, Y, errors):
    """
    Compute linear regression coefficients for basis vectors in the columns
    of matrix ``X``.

    Parameters
    ----------
    X : `~numpy.ndarray`
        Design matrix of regressors
    Y : `~numpy.ndarray`
        Data to fit ``X`` to
    errors : `~numpy.ndarray`
        Uncertainties on each element of ``Y``

    Returns
    -------
    C : `~numpy.ndarray`
        Regression coefficients for each column of ``X``
    """
    K_inv = np.diag(1. / errors ** 2)
    A = np.dot(np.dot(X.T, K_inv), X)
    B = np.dot(np.dot(X.T, K_inv), Y)
    C = np.linalg.solve(A, B)
    return C


def regression_model(C, X):
    """
    Compute linear regression model given the regression coefficients.

    Parameters
    ----------
    C : `~numpy.ndarray`
        Regression coefficients for each column of ``X``
    X : `~numpy.ndarray`
        Design matrix of regressors

    Returns
    -------
    M : `~numpy.ndarray`

    """
    return np.dot(C, X.T)