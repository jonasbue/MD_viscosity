#########################################################
# This file contains various functions that calculate   #
# different equations of state for a HS fluid.          #
# Function names are a bit unclear, so here is a table  #
# of contents:                                          #
# Z: Packing fraction, or the value of the EoS.         #
# rdf: Radial distribution function.                    #
# CS: Carnahan-Starling.                                #
# PY: Percus-Yervick.                                   #
# SPT: Scaled Particle Theory.                          #
#########################################################

import numpy as np
import convert_LAMMPS_output as convert
import theory 
import utils


def Z_measured(p, V, N, T, k=1):
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


def Z_measured_mix(p, rho, T, k=1):
    """ rho is the number density """
    return p/(rho*T)


def get_Z_from_C(C, eos):
    pf = C["PF"]
    T = C["TEMP"]
    N_list = utils.get_component_lists(C, "N")
    sigma_list = utils.get_component_lists(C, "SIGMA")
    mass_list = utils.get_component_lists(C, "MASS")
    x = N_list/np.sum(N_list)
    sigma = viscosity.get_sigma(sigma_list)
    rho = pf_to_rho(sigma, x, pf)
    Z = eos(sigma, x, rho)
    return Z


def partial_pf(sigma, x, rho):
    # xi(3) is the packing fraction of the system.
    # rho is the number density, N/V.
    def xi(l):
        pfi = np.pi/6 * rho * np.sum(x*np.diagonal(sigma**l))
        return pfi
    return xi


def pf_to_rho(sigma, x, pf):
    rho = 6*pf/np.pi/np.sum(x*np.diag(sigma)**3)
    return rho

def rho_to_pf(sigma, x, rho):
    pf = rho / (6/np.pi/np.sum(x*np.diag(sigma)**3))
    return pf

def Z_CS(sigma, x, rho):
    """ Returns expected compressibility factor
        based on the Carnahan-Starling EoS,
        for a given packing fraction n.
        Inputs:
            pf:  packing franction of system, float.
        Returns:
            Z:  compressibility factor, float.
    """
    pf = rho_to_pf(sigma, x, rho)
    return (1 + pf + pf**2 - pf**3) / (1 - pf)**3


def Z_SPT(sigma, x, rho):
    """ Returns expected compressibility factor
        based on the SPT EoS,
        for a given packing fraction n.
        Inputs:
        Returns:
            Z:  compressibility factor, float.
    """
    xi = partial_pf(sigma, x, rho)
    Z = ( 6/(np.pi*rho)
        * ( xi(0)/(1-xi(3))
        + 3*(xi(1)*xi(2))/(1-xi(3))**2
        + 3*xi(2)**3/(1-xi(3))**3
        )
    )
    return Z


def Z_PY(sigma, x, rho):
    """ Returns expected compressibility factor
        based on the Persus-Yervick EoS,
        for a given packing fraction n.
        Inputs:
        Returns:
            Z:  compressibility factor, float.
    """
    xi = partial_pf(sigma, x, rho)
    Z = (
        Z_SPT(sigma, x, rho) 
        - 18/(np.pi*rho)
        * xi(3)*xi(2)**3 / (1-xi(3))**3
    )
    return Z


def Z_BMCSL(sigma, x, rho):
    xi = partial_pf(sigma, x, rho)
    Z = (
        1/xi(0) * (
            xi(0)/(1-xi(3)) 
            + 3*xi(1)*xi(2) / (1-xi(3))**2
            + (3-xi(3))*xi(2)**3 / (1-xi(3))**3
        )
    )
    return Z


def rdf_CS(pf):
    xi = pf
    #return (1-xi/2)/(1-xi)**2
    return 1/(1-xi) + 3/2*xi/(1-xi)**2 + 1/2*xi**2/(1-xi)**3


def rdf_PY(pf):
    """ Returns the thoretical radial distribution 
        function, as given in Pousaneh and de Wijn's paper.

    """
    xi = pf
    return (1+xi/2)/(1-xi)**2


def rdf_SPT_one(pf):
    xi = pf
    return 1/(1-xi) + 3/2*xi/(1-xi)**2 + 3/4*xi**2/(1-xi)**3



def rdf_SPT(sigma, x, rho, i, j):
    """ The radial distribution function in a mixture of
        hard sphere gases.
        Inputs:
            sigma:  np.array of size n*n, where n is the 
                    number of components in the fluid.
            xi:     np.array of length n. One-dimensional.
    """
    xi = partial_pf(sigma, x, rho)
    g_ij = ( 
        1/(1-xi(3))
        + 3*xi(2)/(1-xi(3))**2
        * sigma[i,i]*sigma[j,j]/(sigma[i,i] + sigma[j,j])
        + 3*xi(2)**2/(1-xi(3))**3
        * (sigma[i,i]*sigma[j,j]/(sigma[i,i] + sigma[j,j]))**2
    )
    return g_ij


def rdf_PY_mix(sigma, x, rho, i, j):
    xi = partial_pf(sigma, x, rho)
    g_ij = (
        1/(1-xi(3))
        + 3*xi(2)/(1-xi(3))**2 
        * (sigma[i,i] * sigma[j,j]) / (sigma[i,i] + sigma[j,j])
    )
    return g_ij


def rdf_BMCSL(sigma, x, rho, i, j):
    xi = partial_pf(sigma, x, rho)
    g_ij = (
        1/(1-xi(3))
        + 3*xi(2)/(1-xi(3))**2 
        * (sigma[i,i] * sigma[j,j]) / (sigma[i,i] + sigma[j,j])
        + 2*xi(2)**2 / (1-xi(3))**3
        * ((sigma[i,i] * sigma[j,j]) / (sigma[i,i] + sigma[j,j]))**2
    )
    return g_ij
