"""
	A general module for natural evolutionary strageties RL (NES)
  This script can also be used to check finer time discretization effect
	params -> params_to_xL -> NES_profile: (params, params_to_xL) -> params_new
"""
import jax
import numpy as onp
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as np
from jax import grad, jacrev, jit, vmap
from jax.experimental import stax, optimizers
from jax.experimental.stax import Conv, Dense, MaxPool, Relu, Flatten, LogSoftmax
from jax import random
from functools import partial
from Majorana_transport_np import *
from NNfunctions import *
from Protocol import *
import matplotlib   
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import time
import sys
import os

start_time = time.time()

################ standard setup ################

# adjustable parameters
delta_xL = float(sys.argv[1]) #3.15
T_total = int(sys.argv[2]) #14
v_max = float(sys.argv[3]) #0.444571428571 
num_bang = int(sys.argv[4]) #6
omega = float(sys.argv[5]) #8.0
epoch = int(sys.argv[6]) #3.15
mode = sys.argv[7] 
num = sys.argv[8] 

### Define the parameters of the wire setup (Kitaev Chain)
N = 110 
mu_phys = 1.0
Delta = 0.3/2
t = 1.0    # hopping
dt = 0.01  # dt=0.01 for accurate
tlist = np.linspace(0,T_total,int(T_total/dt))
dtxforward = 0.01
dxforward = 0.28 #1.6145 #1.1 #0.28
dtxbackward = 0.8
dxbackward = 0.17 #0.96 #0.65 #0.17

    
### Define the initial external potential profile parameters 
Vmax = 30.1*np.ones(len(tlist))  # wall potential height
sigma = 1*np.ones(len(tlist)) # wall spreading
v_critical = Delta
xL_0 = 5.0
xR = float(N-5)
xL_f = xL_0 + delta_xL # final position of left wall

### protocal initialization
if mode=='bang':
    xL, vL = x_wall_position_bang_bang2(T_total, dt, v_max, num_bang, xL_0, xL_f)
elif mode=='ramp':
    xL, vL = x_wall_smooth_protocol1(omega, T_total, dt, xL_0, xL_f)
elif mode=='jump':
    scan_p = jump_back_profile
    xL = scan_p(T_total, dt, dxforward, dtxforward, dxbackward, dtxbackward, xL_0, xL_f)
elif mode=='jump2':
    scan_p = jump_back_profile2
    xL = scan_p(T_total, dt, dxforward, dtxforward, dxbackward, dtxbackward, xL_0, xL_f)
elif mode=='bang_load' or 'ramp_load':
    date="06-06-2020"
    parent = "/home/path/data"
    path = os.path.join(parent, mode[:-5], "L{}".format(delta_xL)+"+T{}".format(T_total,2), date+"+"+str(num), "data/")
    xL_old = onp.loadtxt(path+"xL_collection.txt")[-1]
    Overlap_old = onp.loadtxt(path+"learning_curve.txt")[-1]
    xL = x_wall_interpolate(xL_old, T_total, dt, mode=1)
else:
    xL = np.linspace(xL_0, xL_f, len(tlist), endpoint=True)  # linear protocol


################ read file test  ################
xL_tmp = xL
start = time.time()
Overlap1 = Transport_Majorana(N,Vmax,sigma,xL,xR,mu_phys,Delta,t,dt)
print('initial 1-Fidelity:', 1.0-Overlap1)
print('time:', time.time()- start)
if mode=='bang_load' or mode=='ramp_load':
    #print(Overlap1-Overlap_old, Overlap_old, Overlap1)
    print(1-Overlap1-Overlap_old, Overlap_old, 1-Overlap1)


################ define loss function and evolutionary RL  ################

def loss(N,Vmax,sigma,xL,xR,mu_phys,Delta,t,dt):
		return 1-Transport_Majorana(N,Vmax,sigma,xL,xR,mu_phys,Delta,t,dt)


def loss2(N,Vmax,sigma,xL,xR,mu_phys,Delta,t,dt):
		"""penelty on intermediate position that exceeds the two ends
		"""
		if (xL[1:-1] < xL[0]).any() or (xL[1:-1] > xL[-1]).any():
				cost = 1000
		else: 
				cost = 1-Transport_Majorana(N,Vmax,sigma,xL,xR,mu_phys,Delta,t,dt)
		return cost


def NES_profile_xL(params, score_function, npop=50, sigma_noise=0.1, alpha=0.05):
		"""Natural Evolutionary strategy

    Args:
				npop: population size
				sigma: standard deviation
				alpha: learning rate
		"""
		num_params = xL.shape[0]
		N = onp.random.randn(npop, num_params-2)
		R = onp.zeros(npop)
		for j in range(npop):
		  xL_try = xL.copy()
		  xL_try[1:-1] = xL[1:-1] + sigma_noise*N[j]
		  R[j] = score_function(xL=xL_try)
		A = (R - onp.mean(R)) / (onp.std(R)+1e-6)
		xL_update = xL.copy()
		xL_update[1:-1] = xL[1:-1] - alpha/(npop*sigma_noise) * onp.dot(N.T, A)
		return xL_update


def NES_profile(params, params_to_xL, score_function, npop=50, sigma_noise=0.1, alpha=0.05):
  """Natural Evolutionary strategy
  
  Args:
  		npop: population size
  		sigma_noise: standard deviation
  		alpha: learning rate
  """
  num_params = params.shape[0]
  N = onp.random.randn(npop, num_params)
  R = onp.zeros(npop)
  for j in range(npop):
    #start = time.time()
    params_try =params.copy()
    params_try = params_try + sigma_noise*N[j]
    xL_try = params_to_xL(params_try)
    R[j] = score_function(xL=xL_try)
    #print(j, time.time()-start)
    #R = jax.ops.index_update(R, jax.ops.index[j], score_function(xL=xL_try))
  A = (R - onp.mean(R)) / (onp.std(R)+1e-6)
  params_update = params - alpha/(npop*sigma_noise) * onp.dot(N.T, A)
  return params_update


def list_to_array(params_flat):
  """convert list of neural network parameters into 1d array
  """

  params_arr = params_flat[0].reshape(-1)
  for i in range(len(params_flat)-1):
    params_arr = np.concatenate([params_arr, params_flat[i+1].reshape(-1)])

  return params_arr


def array_to_list(params_arr, params_size, params_shape):
  """convert 1d array of neural network parameters back to list
  """

  start = 0
  end = 0
  params_list = []
  for i in range(len(params_size)):
    end += params_size[i]
    params_list.append(params_arr[start:end].reshape(params_shape[i]))
    start += params_size[i]

  return params_list


def NES_profile_nn(params, params_to_xL, score_function, npop=50, sigma_noise=0.1, alpha=0.05):
  """Natural Evolutionary strategy
  
  Args:
  		params: in orignal pytree form
  		npop: population size
  		sigma: standard deviation
  		alpha: learning rate
  """
  
  params_flat, treedef = jax.tree_flatten(params)
  params_shape = [l.shape for l in params_flat]
  params_size = [l.size for l in params_flat]
  params_arr = list_to_array(params_flat)
  
  num_params = np.sum(np.array(params_size))
  N = onp.random.randn(npop, num_params)
  R = onp.zeros(npop)
  for j in range(npop):
    params_try =params_arr.copy() # 1d array
    params_try = params_try + sigma_noise*N[j] 
    params_try = array_to_list(params_try, params_size, params_shape) # list 
    params_try = jax.tree_unflatten(treedef, params_try) # pytree
    xL_try = params_to_xL(params_try)
    R[j] = score_function(xL=xL_try)
  A = (R - np.mean(R)) / (np.std(R)+1e-6)
  params_update = params_arr - alpha/(npop*sigma_noise) * np.dot(N.T, A)
  params_update = array_to_list(params_update, params_size, params_shape)
  params_update = jax.tree_unflatten(treedef, params_update)

  return params_update


def SGES_profile_nn(params, params_to_xL, score_function, npop=50, sigma_noise=0.1, alpha=0.05):
  """Simple Gaussian Evolutionary strategy
  
  Args:
  		params: in orignal pytree form
  		npop: population size
  		sigma: standard deviation
  		alpha: learning rate
  """
  
  params_flat, treedef = jax.tree_flatten(params)
  params_shape = [l.shape for l in params_flat]
  params_size = [l.size for l in params_flat]
  params_arr = list_to_array(params_flat)
  top = int(0.2*npop)
  
  num_params = np.sum(np.array(params_size))
  N = onp.random.randn(npop, num_params)
  R = onp.zeros(npop)
  params_collect = onp.zeros((npop, params_arr.size))
  for j in range(npop):
    params_try =params_arr.copy() # 1d array
    params_try = params_try + sigma_noise*N[j] 
    params_collect[j] = params_try
    params_try = array_to_list(params_try, params_size, params_shape) # list 
    params_try = jax.tree_unflatten(treedef, params_try) # pytree
    xL_try = params_to_xL(params_try)
    R[j] = score_function(xL=xL_try)
 
  idx = onp.argpartition(R, top)
  params_update = onp.mean(params_collect[idx[:top]], axis=0)
  var = onp.linalg.norm(params_collect[idx[:top]] - params_update) / onp.sqrt(top*num_params)
  params_update = array_to_list(params_update, params_size, params_shape)
  params_update = jax.tree_unflatten(treedef, params_update)

  return params_update, var


################ params_to_xL function setup  ################

# direct position profile
def position_profile(params, xL_0, xL_f):
  """create position profile given bulk profile params and two end points xL_0 and xL_f

  Args:
    params: bulk position profile
    xL_0: left point of xL
    xL_f: right point of xL

  Returns:
    xL: position profile
  """

  xL = onp.zeros(params.shape[0]+2)
  xL[1:-1] = params
  xL[0] = xL_0
  xL[-1] = xL_f
  ### below for jax version
  #xL = np.zeros(params.shape[0]+2)
  #xL = jax.ops.index_update(xL, jax.ops.index[1:-1], params)
  #xL = jax.ops.index_update(xL, jax.ops.index[0], xL_0)
  #xL = jax.ops.index_update(xL, jax.ops.index[-1], xL_f)
  return xL

f_position_xL = partial(position_profile, xL_0=xL_0, xL_f=xL_f)


# direct coarse position profile
def coarse_to_fine(f, g):
	"""apply interpolate function to any fine scale profile function
	Args:
		f: interpolate function
		g: fine scale profile function
	"""
	return lambda x: f(g(x))

dt_fine = dt
interp_mode = 0  # mode: 0 is repeat, 1 is linear interpolation
f_interpolate = partial(x_wall_interpolate, T=T_total, dt=dt_fine, mode=interp_mode)
f_coarse_position_xL = coarse_to_fine(f_interpolate, f_position_xL)


# define velocity profile
def velocity_to_position(params, xL_0, xL_f, dt):
  """convert velocity profile to position profile

  Args:
    params: bulk & right edge velocity profile
    xL_0: left point of xL
    xL_f: right point of xL
    dt: time discretization step

  Returns:
    xL: position profile
  """

  xL = onp.zeros(params.shape[0]+1)
  xL[0] = xL_0
  xL[-1] = xL_f
  vL = (xL_f- xL_0) / (onp.sum(params) * dt) * params
  for i in range(len(vL)-1):
    xL[i+1] = xL[i] + vL[i]*dt

  return xL

f_vL = partial(velocity_to_position, xL_0=xL_0, xL_f=xL_f, dt=dt)
f_coarse_velocity_xL = coarse_to_fine(f_interpolate, f_vL)

# define neural network
neurons1 = 100 
#neurons2 = 20 
net_init, net_apply = stax.serial(
        Dense(neurons1), Relu,
        #Dense(neurons2), Relu,
        Dense(1), Sigmoid
        )
rng = random.PRNGKey(1)
in_shape = (-1, 1,)
out_shape, net_params = net_init(rng, in_shape)
inputs = np.array([[k] for k in tlist[1:len(tlist)]/T_total])


# neural network profile 
def nn_profile(params, inputs, xL_0, xL_f):
    predictions = net_apply(params, inputs)
    ratio = predictions[:, 0]
    ratio = ratio / np.sum(ratio)
    ratio_sum = np.cumsum(ratio)
    ratio_sum = np.concatenate((np.array([0]),ratio_sum))
    xL = xL_0 + (xL_f - xL_0) * ratio_sum
    #plt.plot(predictions)
    return xL

f_nn = partial(nn_profile, inputs=inputs, xL_0=xL_0, xL_f=xL_f)


################ training session  ################

### training setup
alpha0 = 0.1
alpha_decay = 0.99
sigma_noise = 0.1
npop = 100
dt_coarse = 5*dt
xL_coarse, vL_coarse = x_wall_position_bang_bang2(T_total, dt_coarse, v_max, num_bang, xL_0, xL_f)
params_v = onp.zeros(len(vL)+1)
params_v[:-1] = vL
params_v[-1] = (xL_f - xL_0 - onp.sum(vL)*dt) /dt
params_v_coarse = onp.zeros(len(vL_coarse)+1)
params_v_coarse[:-1] = vL_coarse
params_v_coarse[-1] = (xL_f - xL_0 - onp.sum(vL_coarse)*dt_coarse) /dt_coarse

### cost function setup
score_function = partial(loss, N=N,Vmax=Vmax,sigma=sigma,xR=xR,mu_phys=mu_phys,Delta=Delta,t=t,dt=dt)
#score_function = partial(loss2, N=N,Vmax=Vmax,sigma=sigma,xR=xR,mu_phys=mu_phys,Delta=Delta,t=t,dt=dt)


# initialize xL
#params = xL.copy()[1:-1]  # direct xL
params = xL_coarse[1:-1]  # coarse grain xL
#params = net_params.copy()  # neural network params
#params = params_v.copy()  # velocity induced xL
#params = params_v_coarse.copy()  # coarse grain velocity induced xL
params_to_xL = f_coarse_position_xL #f_vL  #f_nn  #f_coarse_position_xL  #f_coarse_velocity_xL #f_position_xL
xL_tmp = params_to_xL(params)
#xL_tmp = xL.copy()
#Overlap1 = Transport_Majorana(N,Vmax,sigma,xL_tmp,xR,mu_phys,Delta,t,dt)
Overlap1 = 1-score_function(xL=xL_tmp)
xL_collection = [xL_tmp]
learning_curve = [1-Overlap1]
print('initial 1-Fidelity:', 1.0-Overlap1)
if mode=='bang_load' or mode=='ramp_load':
    print(Overlap1-Overlap_old, Overlap_old, Overlap1)


### training starts
for i in range(epoch):
  then = time.time()
  alpha = alpha0 * (alpha_decay**i)
  #xL_tmp = NES_profile_xL(xL_tmp, score_function, npop, sigma_noise, alpha)
  params = NES_profile(params, params_to_xL, score_function, npop, sigma_noise, alpha)
  #params = NES_profile_nn(params, params_to_xL, score_function, npop, sigma_noise, alpha)
  #params, sigma_noise = SGES_profile_nn(params, params_to_xL, score_function, npop, sigma_noise, alpha)
  print('sigma', sigma_noise)
  xL_tmp = params_to_xL(params)
  #Overlap_tmp = Transport_Majorana(N,Vmax,sigma,xL_tmp,xR,mu_phys,Delta,t,dt)
  Overlap_tmp = 1-score_function(xL=xL_tmp)
  xL_collection.append(xL_tmp)
  learning_curve.append(1.0-Overlap_tmp)
  print('1-Fidelity:', 1.0-Overlap_tmp,'Gradient descent step {} took: '.format(i), time.time()-then, 'seconds')
  print('alpha value: ', alpha, 'Overlap diff: ', Overlap_tmp-Overlap1)


### plot setup
learning_curve = np.array(learning_curve)
xL_collection = np.array(xL_collection)
best_index = onp.where(learning_curve == onp.amin(learning_curve))
valid_index = onp.where(learning_curve>0)
learning_curve = np.append(learning_curve, learning_curve[best_index[0][0]])
onp.savetxt('./data/learning_curve.txt', learning_curve)
onp.savetxt('./data/xL_collection.txt', xL_collection)


f, ax = plt.subplots()
ax.plot(valid_index[0], learning_curve[valid_index])
ax.set_xlabel('number of steps')
ax.set_ylabel('Fidelity')
plt.tight_layout()
f.savefig("./plot/Fidelity.png")
f.savefig("./plot/Fidelity.pdf")
plt.close('all')

f, ax = plt.subplots()
for i in range(epoch):
    ax.plot(xL_collection[i])
plt.tight_layout()
f.savefig("./plot/xL_collection.png")
f.savefig("./plot/xL_collection.pdf")
plt.close('all')

f, ax = plt.subplots()
for i in range(10):
    ax.plot(xL_collection[i])
plt.tight_layout()
f.savefig("./plot/xL_collection_first10.png")
f.savefig("./plot/xL_collection_first10.pdf")
plt.close('all')

f, ax = plt.subplots()
ax.plot(xL_collection[best_index[0][0]])
plt.tight_layout()
f.savefig("./plot/xL_collection_best.png")
f.savefig("./plot/xL_collection_best.pdf")
plt.close('all')
