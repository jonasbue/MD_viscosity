#########################################################
# Outdated description:                                 #
# This script takes a log file from LAMMPS and plots    #
# its compressibility factor vs. the Carnahan-Starling  #
# equation of state.                                    #
#########################################################

import numpy as np
import convert_LAMMPS_output as convert
import viscosity


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


def partial_pf(sigma, x, rho, i):
    """ Returns a list of the cumulative 
        packing fraction of gas component l.
    """
    # I have no idea what this equation should be.
    x_sigma = 0
    for l in range(i):
        x_sigma += x*np.diag(sigma)**l
    #if i >= len(x_sigma):
    #    i = -1
    return (np.pi/6 * rho * np.sum(x_sigma))


def Z_SPT(sigma, x, rho):
    def xi(l):
        return partial_pf(sigma, x, rho, l)
    Z = ( 6/(np.pi*rho)
        * ( xi(0)/(1-xi(3))
        + 3*(xi(1)*xi(2))/(1-xi(3))**2
        + 3*xi(2)**3/(1-xi(3))**3
        )
    )
    return Z

def mix_rdf(sigma, x, rho, i, j):
    """ The radial distribution function in a mixture of
        hard sphere gases.
        Inputs:
            sigma:  np.array of size n*n, where n is the 
                    number of components in the fluid.
            xi:     np.array of length n. One-dimensional.
    """
    i -= 1
    j -= 1
    def xi(l):
        return partial_pf(sigma, x, rho, l)
    total_pf = xi(3)
    print("Computed pf =", total_pf)
    g_ij = ( 
        1/(1+xi(3))
        + 3*xi(2)/(1-xi(3))**2
        * sigma[i,i]*sigma[j,j]/(sigma[i,i] + sigma[j,j])
        + 3*xi(2)**2/(1-xi(3))**3
        * (sigma[i,i]*sigma[j,j]/(sigma[i,i] + sigma[j,j]))**2
    )
    return g_ij

def test_mix():
    sigma = np.array([[1.0,1.0],[1.0,1.0]])
    x = np.array([0.5, 0.5])
    pf = 0.5
    rho = 6*pf/np.pi
    print("pf = ", pf)
    g = mix_rdf(sigma, rho, x, 1,1)
    print("g_mix = ", g)
    print("g_one = ", viscosity.radial_distribution(pf))
    g = mix_rdf(sigma, rho, x, 1,2)
    print("g_mix = ", g)
    g = mix_rdf(sigma, rho, x, 2,2)
    print("g_mix = ", g)
    Z = Z_SPT(sigma, x, rho)
    print("Z_mix = ", Z)
    print("Z_CS = ", Z_Carnahan_Starling(pf))
