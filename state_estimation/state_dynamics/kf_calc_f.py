########################################################################
## Calculates the system dynamics equation f(x,u,t)
########################################################################
import numpy as np

def kf_calc_f(t, x, u):

    n = x.size
    xdot    = np.zeros([n,1])
    
    # system dynamics x[k]=u[k] and upwash derivative is zero  
    xdot[0] = u[0]
    xdot[1] = u[1]
    xdot[2] = u[2]
    xdot[3] = 0
        
    return xdot
        
        