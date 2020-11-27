#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 16:26:04 2019
Neural Network functions for the diff programming nn training of the wall
profiles. 
@author: lcoopmans
"""

import jax
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as np
from jax import grad, jacrev
from jax.experimental import stax, optimizers
from jax.experimental.stax import Conv, Dense, MaxPool, Relu, Flatten, LogSoftmax, Sigmoid
from jax import random

##TODO: promote Vmax and sigma into neural network
def get_xL(net_params, inputs, xL_0, xL_f):
     predictions = net_apply(net_params, inputs)
     ratio = predictions[:, 0]
     ratio = ratio / np.sum(ratio)
     ratio_sum = np.cumsum(ratio)
     ratio_sum = np.concatenate((np.array([0]),ratio_sum))
     xL = xL_0 + (xL_f - xL_0) * ratio_sum
     #plt.plot(predictions)
     return xL
 
# define loss
def loss(net_params, inputs, xL_0, xL_f, N,Vmax,sigma,xR,mu_phys,Delta,t,dt):
    predictions2 = net_apply(net_params, inputs)
    ratio2 = predictions2[:, 0]
    ratio2 = ratio2 / np.sum(ratio2)
    ratio_sum2 = np.cumsum(ratio2)
    ratio_sum2 = np.concatenate((np.array([0]),ratio_sum2))
    xL = xL_0 + (xL_f - xL_0) * ratio_sum2
    return -Transport_Majorana(N,Vmax,sigma,xL,xR,mu_phys,Delta,t,dt);

#learning step
def step(i, opt_state, inputs, xL_0, xL_f, N,Vmax,sigma,xR,mu_phys,Delta,t,dt):
    p = get_params(opt_state)
    g = grad(loss)(p, inputs, xL_0, xL_f, N,Vmax,sigma,xR,mu_phys,Delta,t,dt)
    return opt_update(i, g, opt_state)
