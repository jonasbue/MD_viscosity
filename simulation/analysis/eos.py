#########################################################
# This script takes a log file from LAMMPS and plots    #
# its compressibility factor vs. the Carnahan-Starling  #
# equation of state.                                    #
#########################################################

import numpy as np
import convert_LAMMPS_output as convert


def Z(p, V, N, T, k=1):
    """ Computes compressibility factor of a system.
        Inputs:
            p:      pressure, float.
            V:      volume, float.
            N:      number of particles, float.
            T:      temperature, float.
            k:      boltzmann constant, float.
                    Default value is 1, which is the
                    value in lj reduced units.
        Returns:
            Z:      Compressibility factor.
            
    """
    return p*V/(N*k*T)


def Z_Carnahan_Starling(n):
    """ Returns expected compressibility factor
        based on the Carnahan-Starling EoS,
        for a given packing fraction n.
        Inputs:
            n:  packing franction of system, float.
        Returns:
            Z:  compressibility factor, float.
    """
    return (1 + n + n**2 - n**3) / (1 - n)**3


N = 1000
eta_list = np.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.5])
variable_list = ["p", "V", "T"]


