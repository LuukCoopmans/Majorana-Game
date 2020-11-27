import numpy as onp

def x_wall_smooth_protocol1(omega,T_total,dt,x_a,x_b):
    
    l = x_b - x_a
    tau = onp.pi/omega
    T_vmax = T_total - 2*tau
    
    if T_vmax < 0:
        raise Exception("Build up time too long at this frequecy omega")
        
    vmax = l/(T_total-tau)
    
    v_wall_1 = vmax * (1-onp.cos(omega*onp.arange(0,onp.pi/omega,dt)))/2
    v_wall_2 = vmax * onp.ones(onp.int(T_total/dt)-2*len(v_wall_1))
    v_wall_3 = vmax * (1-onp.cos((omega*onp.arange((onp.pi/omega+T_vmax),(2*onp.pi/omega+T_vmax),dt)-omega*T_vmax)))/2
    
    v_wall = onp.concatenate((v_wall_1,onp.concatenate((v_wall_2,v_wall_3))))
    
    x_wall = x_a + onp.cumsum(v_wall*dt)
    return x_wall, v_wall


def x_wall_position_bang_bang(T,dt,vmax,n,x_a,x_b):
    """
        TO DO: ROUNDING OF THE NUMBER OF PULSES SUCH THAT NOT OVER/UNDERSHOOTS
        Returns an array of the wall positions for the bang-bang protocols
        of total duration T with max velocity vmax. The wall moves in this 
        time between x_a and x_b.
    """
    w_pulse = n/T #pulse frequency
    
    t_pulse = (x_b-x_a)/(vmax*n) #how long each pulse should last
    t_0 = (T-n*t_pulse)/(n-1) #time we keep the wall still between pulses
    
    if t_0 <0:
        raise Exception("T is to small to get to the other wall at this speed vmax")
    
    N_pulse = onp.rint(t_pulse/dt)
    N_0 = onp.rint(t_0/dt)
    
    x = onp.array([])
    
    for i in range(n):
        x0 = x_a + i*N_pulse*vmax*dt
        
        x_pulse = x0+onp.arange(1,(N_pulse+1))*vmax*dt
        if i != (n-1):
            x = onp.concatenate((x,onp.concatenate((x_pulse,(x0+N_pulse*vmax*dt)*onp.ones(onp.int(N_0))))))
        else:
            x = onp.concatenate((x,x_pulse))           
            
    return x,w_pulse


def x_wall_position_bang_bang2(T,dt,vmax,n,x_a,x_b):
    """
        Returns an array of the wall positions for the bang-bang protocols
        of total duration T with max velocity vmax. The wall moves in this 
        time between x_a and x_b.
    """
    T = T-2*dt
    w_pulse = n/T #pulse frequency
    
    t_pulse = (x_b-x_a)/(vmax) #how long each pulse should last
    t_0 = (T-t_pulse) #total time we keep the wall still between pulses
    
    if t_0 <0:
        raise Exception("T is to small to get to the other wall at this speed vmax")
    
    N_pulse_1 = onp.floor(t_pulse/dt/n)
    N_pulse_extra = onp.floor((t_pulse/dt/n-N_pulse_1)*n)
    if N_pulse_extra != 0:
        alpha = onp.floor(n/N_pulse_extra)
    else:
        alpha = 0
        
    N_0 = onp.floor((T/dt-n*N_pulse_1-N_pulse_extra)/(n-1))
    N_0_extra = onp.floor(T/dt-n*N_pulse_1-N_pulse_extra-N_0*(n-1))
    if N_0_extra !=0:
        beta = onp.floor((n-1)/N_0_extra)
    else:
        beta = 0
    
    v_wall = onp.array([])
    
    for i in range(n):
        if alpha == 0:
            v_wall = onp.concatenate((v_wall,vmax*onp.ones(onp.int(N_pulse_1)))) 
        elif i%alpha == 0 and i < alpha*N_pulse_extra:
            v_wall = onp.concatenate((v_wall,vmax*onp.ones(onp.int(N_pulse_1+1))))
        else:
            v_wall = onp.concatenate((v_wall,vmax*onp.ones(onp.int(N_pulse_1))))
        
        if beta == 0 and i < (n-1):
            v_wall = onp.concatenate((v_wall,onp.zeros(onp.int(N_0))))
        elif beta == 0 and i == (n-1):
            v_wall = onp.concatenate((v_wall,onp.zeros(onp.int(T/dt-len(v_wall)))))
        elif i%beta == 0 and i < beta*N_0_extra:
            v_wall = onp.concatenate((v_wall,onp.zeros(onp.int(N_0+1))))
        elif i < (n-1):
            v_wall = onp.concatenate((v_wall,onp.zeros(onp.int(N_0))))
        elif i == (n-1):
            v_wall = onp.concatenate((v_wall,onp.zeros(onp.int(T/dt-len(v_wall)))))
                    
        
    x_wall = onp.concatenate((onp.concatenate((onp.array([x_a]),x_a + onp.cumsum(v_wall*dt))),onp.array([x_b])))
    return x_wall, v_wall


#repeat or interpolate an existing profile from x_wall profile
#mode = 0 for repeat, mode = 1 for interpolate
#num_pts and num_new_pts are integer and they can divide each other
def x_wall_interpolate(x_wall, T, dt, mode=1):
  num_pts = len(x_wall)    	
  num_new_pts = int(T/dt)
  num_interp = int(num_new_pts / num_pts)
  if mode == 0: 
    x_wall_new = onp.repeat(x_wall, num_interp)	
  elif mode ==1: 
    #x_range = onp.linspace(0, num_pts-1, num_interp*(num_pts-1)+1)
    x_range = onp.linspace(0, num_pts-1, num_new_pts)
    x_wall_new = onp.interp(x_range, onp.arange(num_pts), x_wall)
  return x_wall_new


# repeat or interpolate an existing profile from velocity profile
def get_xL_from_velocity(v,dt,T,xL_a,xL_b):
    n = T/dt/(len(v))
    v_new = onp.repeat(v,n)
#    x = onp.arange(0,n*len(v),n)    #if want to interpolate
#    x_new = onp.arange(n*len(v)-1)
#    v_new = onp.interp(x_new,x,v)
    xL = xL_a + onp.cumsum(v_new[1:len(v_new)-1])*dt
    xL = onp.concatenate((onp.array([xL_a]),xL,onp.array([xL_b])))
    return xL


def v_to_xL(v_profile, dt, xL_0):
  """convert v_profile to xL profile
  """
  xL = onp.zeros(len(v_profile)+1)
  xL[0] = xL_0
  for i in range(1,len(v_profile)+1):
  	xL[i] = xL[i-1] + v_profile[i-1]*dt
  
  return xL


def jump_back_profile(T, dt, Dxjump, Dtxjump, Dxback, Dtxback, x0, xF):
    """ Returns the profile in which first one jump is done forward, then one
        jump backward followed by linear motion. In the end then one jump 
        backward and one final jump forward to the final wall position.
        For stability every time should be chosen in multiples of dt.
    """
    jump_forward = Dxjump/Dtxjump*onp.linspace(dt,Dtxjump,onp.int(onp.round(Dtxjump/dt)))
    jump_backward = -Dxback/Dtxback*onp.linspace(dt,Dtxback,onp.int(onp.round(Dtxback/dt)))
    linear_motion = ((xF-x0-2*Dxjump+2*Dxback)/(T-2*Dtxjump-2*Dtxback))*onp.linspace(dt,T-2*Dtxjump-2*Dtxback,onp.int(onp.round((T-2*Dtxjump-2*Dtxback)/dt)))
    
    xwall = onp.concatenate((onp.array([x0]),
                            x0+jump_forward,
                            x0+Dxjump+jump_backward,
                            x0+Dxjump-Dxback+linear_motion,
                            xF - Dxjump + Dxback + jump_backward,
                            xF - Dxjump + jump_forward))
    return xwall

	
def jump_back_profile2(T, dt, Dxjump, Dtxjump, Dxback, Dtxback, x0, xF):
  """This is the jump-wait-backward at the beginning and back-wait-jump at the 
      end protocal

  Args:
    T: total time
    dt: discretization time steps
    Dxjump: forward jump distance
    Dtxjump: forward jump time, default to be instanenous, equal to dt
    Dxback: backward jump distance
    Dtxback: wait time between forward and backward jump
    x0: starting location
    xF: end location
  
  Returns:

  """

  jump_forward = Dxjump/Dtxjump*onp.linspace(dt,Dtxjump,onp.int(onp.round(Dtxjump/dt)))
  forward_wait = (x0+Dxjump) * onp.ones(int(Dtxback//dt))
  linear_motion = ((xF-x0-2*Dxjump+2*Dxback)/(T-2*Dtxjump-2*Dtxback))*onp.linspace(dt,T-2*Dtxjump-2*Dtxback,onp.int(onp.round((T-2*Dtxjump-2*Dtxback)/dt)))
  backward_wait = (xF-2*Dxjump+Dxback) * onp.ones(int(Dtxback//dt))

  xwall = onp.concatenate((onp.array([x0]),
                          x0+jump_forward,
                          forward_wait,
                          x0+Dxjump-Dxback+linear_motion,
                          backward_wait,
                          xF - Dxjump + jump_forward))
  return xwall



