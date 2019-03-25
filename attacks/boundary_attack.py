#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 20:36:17 2019

@author: felix
"""

## Eager mode

import tensorflow as tf
import numpy as np
import numpy.linalg as la

#def random_perturbation(adv, original, delta, epsilon):
#    pert_norm_old = la.norm(original-adv)
#    eta = np.random.normal(size=adv.shape)
#    eta *=  delta * pert_norm_old / la.norm(eta)
#    adv_new = np.clip(adv+eta, min=0, max=1)
#    
#    #Orthogonal projection
#    pert = adv_new - original
#    pert = (1-epsilon) * pert_norm_old / la.norm(pert)
#    adv_new = np.clip(original+pert, min=0, max=1)
#    
#    return adv_new

def boundary_attack(original, is_adv, eps_min, max_steps):
    epsilon = 0.5
    delta = 1
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
        eta *= delta * pert_norm_old / la.norm(eta)
        #adv_new = np.clip(adv+eta, 0,1)
        pert = adv + eta - original
        pert *= pert_norm_old / la.norm(pert)
        adv_orth = np.clip(original+pert,0,1)        
        
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
        
        # Update epsilon, delta
        if orth_rate < min_orth:
            delta *= 0.7
        if orth_rate > max_orth:
            delta *= 1.3
        if perp_rate < min_perp:
            epsilon *= 0.7
        if perp_rate > max_perp:
            epsilon = 0.3 + 0.7*epsilon

            
    return adv