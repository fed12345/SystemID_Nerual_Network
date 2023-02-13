########################################################################
## Calculates the system dynamics equation h(x,u,t)
########################################################################
import numpy as np

def kf_calc_h(t, x, u):
    
    n       =  u.size
    zpred    = np.zeros([n,1])
    
    # system dynamics 

    zpred[0] = np.arctan(x[2]/x[0])*(1 + x[3])
    zpred[1] = np.arctan(x[1]/np.sqrt(x[0]**2+x[2]**2))
    zpred[2] = np.sqrt(x[0]**2+x[1]**2+x[2]**2)
        
    return zpred
        
        