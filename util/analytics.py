#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Small functions for analyzing the adversarial examples
"""

import numpy as np
import numpy.linalg as la

def average_norm(x, orig=None, filter='default'):
    """ Average norm
    
    Args:
        x: The perturbations if orig is None, otherwise
           the adversarial examples
        orig: The original images
        filter: Function that filters invalid perturbations,
                or string 'default' to use the function valid_index
    
    Returns:
        Mean and std of the norms of x or x-orig, if orig is not None
    """
    if orig is not None:
        N = min(len(orig), len(x))
        x = x[:N] - orig[:N]

    if filter is None:
        idx = np.ones(N, dtype=np.bool)
    elif filter == 'default':
        idx = valid_index(x)
    elif callable(filter):
        idx = filter(x)
    else:
        raise ValueError('Invalid filter')
    
    norms = np.array([la.norm(xi) for xi in x[valid_index(x)] ])
    return norms.mean(), norms.std()


def valid_index(x):
    """ Boolean mask of images without NaN of inf """
    return np.isfinite(x).all(axis=tuple(range(1,x.ndim)))


#def certanties(x):
#    x = x.argsort()
