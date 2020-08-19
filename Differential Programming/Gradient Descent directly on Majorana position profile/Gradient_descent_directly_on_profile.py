#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Main code for training a n.n to find the optimal wall profile using 
    differential programming. 
"""

import jax
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as np
from jax import grad, jacrev
from jax import random
from Majorana_transport import *
import matplotlib.pyplot as plt
import time

start_time = time.time()
#Define the parameters of the wire setup (Kitaev Chain)
N = 55 
mu_phys = 1.0
Delta = 0.3/2
t = 1.0

#Define the duration and timestep
T_total = 8.0
dt = 0.1
tlist = np.linspace(0,T_total,int(T_total/dt))
    
#Define the initial external potential profile parameters 
Vmax = 30.1*np.ones(len(tlist)+5)
sigma = 1*np.ones(len(tlist)+5)
xL_0 = 5.0
xR = float(N-5)

#Define the parameters for motion of the left wall
xL_f = 5.75

alpha0=2#e-2 #learning rate for the direct
alpha_decay = 0.99 # decreasing slowly the size of the learning rate
episodes = 10  

#initialize a wall profile
xL_tmp = np.linspace(xL_0,xL_f,int(np.round(T_total/dt)))

#uncomment to load an earlier profile 
#np.load('profile.npy')

Overlap1 = Transport_Majorana(N,Vmax,sigma,xL_tmp,xR,mu_phys,Delta,t,dt)
print("Starting Infidelity:", 1-Overlap1)
learning_curve = []

def loss(N,Vmax,sigma,xL,xR,mu_phys,Delta,t,dt):
	return 1-Transport_Majorana(N,Vmax,sigma,xL,xR,mu_phys,Delta,t,dt)

for i in range(episodes):
    then = time.time()
    
    alpha= alpha0*(alpha_decay)**i
    xL_grad = jacrev(loss,3)
    gradient = xL_grad(N,Vmax,sigma,xL_tmp,xR,mu_phys,Delta,t,dt)
    xL_tmp = np.concatenate((np.concatenate((np.array([xL_0]),xL_tmp[1:(len(xL_tmp)-1)]-alpha*gradient[1:(len(xL_tmp)-1)])),np.array(([xL_f]))))
    Overlap_tmp = Transport_Majorana(N,Vmax,sigma,xL_tmp,xR,mu_phys,Delta,t,dt)
    learning_curve.append(Overlap_tmp)
    print("Current Overlap:",1-Overlap_tmp)
    print("Improvement from last step:", Overlap_tmp-Overlap1)
    print('Gradient descent step {} took: '.format(i), time.time()-then, 'seconds')
    
    



