#############################################################
# This file contains functions that calculate viscosity.    #
# Only theoretical mathematical functions are included,     #
# for functions related to the simulation or other          #
# computational aspects, see the files ./muller_plathe.py   #
# or ./regression.py.                                       #
# For more theoretical functions, # see ./eos.py.           #
#############################################################

import numpy as np
import eos


def enskog(pf, sigma, T, m, k=1.0):
    """ Returns the theoretical value of the 
        viscosity for a given packing fraction.
    """
    eta_0 =  5 * np.sqrt((m*k*T)/np.pi) / (16*sigma**2)
    V_excl = 2*np.pi*(sigma**3)/3
    rho = 6*pf/np.pi
    g = eos.rdf_PY(pf)
    eta = eta_0 * (
        1/g 
        + 0.8 * V_excl * rho
        + 0.776 * V_excl**2 * rho**2 * g
    )
    return eta


def get_viscosity(Ptot, A, t, dv_dz):
    """ Computes the measured value of the viscosity,
        from the Müller-Plathe experiment.
        Dangerous behaviour: 
            The first values of the array will be undefined,
            due to for example division by zero.
            This is not a problem since these values 
            are not used in the actual computation of 
            the viscosity, but a check would be wise.
        Input:
            Ptot:   np.array. The total momentum which has 
                    passed through A at a time index.
            A:      float. Cross section of area through 
                    which the momentum flux passes.
            t:      np.array of time values.
            dv_dz:  float. Slope of velocity profile.
        Output:
            eta:        Computed viscosity.
            
    """
    #print("In visc:\t", Ptot.shape, t.shape, dv_dz.shape)
    j = Ptot/(2*A*t)
    eta = -j/dv_dz
    return eta

def get_area(Lx, Ly):
    """ Return the cross-section of the box, 
        through which the momentum flux Jz(vx) passes.
        Input:
            Lx: float. Length of box in x-direction.
            Ly: float. Length of box in y-direction.
        Output:
            A:  float. Area.
    """
    return 4*Lx*Ly
