#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Boundary attack
"""

import numpy as np
import numpy.linalg as la


def boundary_attack(original, is_adv, eps_min, max_steps, dtype=np.float32):
    """
    Computes an adversarial example using the boundary attack
    Args:
        original: original image as numpy array
        is_adv: objective function. is_adv(x) == True iff x is adversarial
        eps_min: tolerance for termination of algorithm
        max_steps: maximum number of iterations
        dtype: dtype of adversarial example

    Returns: valid adversarial example as numpy array

    """
    epsilon = 0.5
    gamma = 1
    min_orth = 0.4
    max_orth = 0.6
    min_perp = 0.17
    max_perp = 0.32
    mov_avg_rate = 1/10
    
    orth_rate = (max_orth - min_orth) / 2
    perp_rate = (max_perp - min_perp) / 2
    
    shape = original.shape
    perp_success = False
    orth_success = False
    pert_norm_old = 1
   
    adv = np.random.uniform(size=shape)
    while not is_adv(adv):
        adv = np.random.uniform(size=shape)
    
    steps = 0
    while epsilon > eps_min and steps < max_steps and pert_norm_old > eps_min:
        # Orthogonal step
        pert_norm_old = la.norm(original-adv)
        eta = np.random.normal(size=shape)
        eta *= gamma * pert_norm_old / la.norm(eta)

        pert = adv + eta - original
        pert *= pert_norm_old / la.norm(pert)
        adv_orth = np.clip(original+pert, 0, 1)
        
        orth_success = is_adv(adv_orth)
        steps += 1
        if orth_success:
            # Perpendicular
            adv_perp = np.clip((1-epsilon)*adv_orth + epsilon*original, 0, 1)
            perp_success = is_adv(adv_perp)
            steps += 1
            if perp_success:
                adv = adv_perp
            else:
                adv = adv_orth

        orth_rate = mov_avg_rate*orth_success + (1-mov_avg_rate)*orth_rate
        perp_rate = mov_avg_rate*perp_success + (1-mov_avg_rate)*perp_rate
        
        # Update epsilon, gamma
        if orth_rate < min_orth:
            gamma *= 0.7
        if orth_rate > max_orth:
            gamma *= 1.3
        if perp_rate < min_perp:
            epsilon *= 0.7
        if perp_rate > max_perp:
            epsilon = 0.3 + 0.7*epsilon

    return adv.astype(dtype)
