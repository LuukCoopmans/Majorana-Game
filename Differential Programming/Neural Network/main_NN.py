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
from jax.experimental import stax, optimizers
from jax.experimental.stax import Conv, Dense, MaxPool, Relu, Flatten, LogSoftmax, Sigmoid
from jax import random
from Majorana_transport import *
#from NNfunctions import *
import matplotlib.pyplot as plt
import time

start_time = time.time()
#Define the parameters of the wire setup (Kitaev Chain)
N = 55
mu_phys = 1.0
Delta = 0.3/2
t = 1.0

#Define the duration and timestep
T_total = 16.0
dt = 0.1
tlist = np.linspace(0,T_total,int(T_total/dt))
    
#Define the initial external potential profile parameters 
Vmax = 30.1*np.ones(len(tlist))
sigma = 1*np.ones(len(tlist))
xL_0 = 5.0
xR = float(N-5)

#Define the parameters for motion of the left wall
xL_f = [8.4]

neurons1 = 100
neurons2 = 100
neurons3 = 100
#layer2_sizes=[5,7,10,15,20]
alpha=1e-3 #learning rate for the optimizer
episodes = 3

final_fidelity_neurons2 = [] 
for i in range(len(xL_f)):
    

    net_init, net_apply = stax.serial(
            Dense(neurons1), Relu,
            Dense(neurons2), Relu,
            Dense(neurons3), Relu,
            Dense(1), Sigmoid
            )

    # Initialize parameters, not committing to a batch shape

    rng = random.PRNGKey(1)
    in_shape = (-1, 1,)
    out_shape, net_params = net_init(rng, in_shape)
    net_params = [[a.astype(np.float64) for a in d] for d in net_params] # convert data type
    inputs = np.array([[k] for k in tlist[1:len(tlist)]/T_total])

    def get_xL(net_params, inputs, xL_0, xL_f):
        predictions = net_apply(net_params, inputs)
        ratio = predictions[:, 0]
        ratio = np.concatenate((np.array([0]),ratio))
        ratio = np.concatenate((ratio,np.array([1])))
        xL = xL_0 + (xL_f - xL_0) * ratio
        #plt.plot(predictions)
        return xL
 
    # define loss
    def loss(net_params, inputs, xL_0, xL_f, N,Vmax,sigma,xR,mu_phys,Delta,t,dt):
        predictions2 = net_apply(net_params, inputs)
        ratio2 = predictions2[:, 0]
        ratio2 = np.concatenate((np.array([0]),ratio2))
        ratio2 = np.concatenate((ratio2,np.array([1])))
        xL = xL_0 + (xL_f - xL_0) * ratio2
        return -Transport_Majorana(N,Vmax,sigma,xL,xR,mu_phys,Delta,t,dt);

    # learning step
    def step(i, opt_state, inputs, xL_0, xL_f, N,Vmax,sigma,xR,mu_phys,Delta,t,dt):
        p = get_params(opt_state)
        g = grad(loss)(p, inputs, xL_0, xL_f, N,Vmax,sigma,xR,mu_phys,Delta,t,dt)
        return opt_update(i, g, opt_state)

    # Apply network to get xL
    xL = get_xL(net_params, inputs, xL_0, xL_f[i])
    xL2 = np.linspace(xL_0, xL_f[i], len(tlist), endpoint=True)


    Overlap1 = Transport_Majorana(N,Vmax,sigma,xL,xR,mu_phys,Delta,t,dt);
    Overlap2 = Transport_Majorana(N,Vmax,sigma,xL2,xR,mu_phys,Delta,t,dt);

    # define optimizer
    opt_init, opt_update, get_params = optimizers.adam(step_size=alpha)
    opt_state = opt_init(net_params)

    learning_curve = []
    for j in range(episodes):
        then = time.time()
        opt_state = step(i, opt_state, inputs,xL_0, xL_f[i], N,Vmax,sigma,xR,mu_phys,Delta,t,dt)
        net_params_tmp = get_params(opt_state)
        xL_tmp = get_xL(net_params_tmp, inputs, xL_0, xL_f[i])
        plt.figure(4)
        plt.plot(xL_tmp)
        Overlap_tmp = Transport_Majorana(N,Vmax,sigma,xL_tmp,xR,mu_phys,Delta,t,dt)
        learning_curve.append(Overlap_tmp)
        print(1-Overlap_tmp,'The learning episode {} took: '.format(j), time.time()-then, 'seconds')

    net_params = get_params(opt_state)
    xL3 = get_xL(net_params, inputs, xL_0, xL_f[i])
    Overlap3 = Transport_Majorana(N,Vmax,sigma,xL3,xR,mu_phys,Delta,t,dt);
    print('compare different overlaps')
    print(neurons1, Overlap1)
    print(neurons1, Overlap2)
    print(neurons1, Overlap3)

    #Postprocessing saving and plotting
    final_fidelity_neurons2.append(Overlap3)
    Qloss_inbetween_final_wall = Transport_Majorana_all_fidelities(N,Vmax,sigma,xL3,xR,mu_phys,Delta,t,dt)

 
    tlist = np.linspace(0,T_total,len(xL3))
    plt.figure(4)
    plt.xlabel('time t/dt')
    plt.ylabel('Left wall position x_L(t)')
  

    #plot the final wall profile
    plt.figure(1)
    plt.clf()
    plt.xlabel('time t')
    plt.ylabel('Left wall position x_L(t)')
    plt.plot(tlist,xL3)


    #fidelity as a function of time up to T 
    
    plt.figure(2)
    plt.clf()
    plt.plot(tlist, Qloss_inbetween_final_wall)
    plt.xlabel('time t')
    plt.ylabel('Fidelity F(t)')

    #plot the learning curve
    plt.figure(3)
    plt.clf()
    plt.plot(learning_curve)
    plt.ylabel('Final Fidelity at T_total')
    plt.xlabel('Learning episode number')


    print('Total learning took:', time.time()-start_time,'for neurons',neurons2)
    plt.show()
    



