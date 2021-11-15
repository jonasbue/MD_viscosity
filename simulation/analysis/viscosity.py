#############################################################
# This file contains functions that calculate viscosity.    #
# Only theoretical mathematical functions are included,     #
# for functions related to the simulation or other          #
# computational aspects, see the files ./muller_plathe.py   #
# or ./regression.py.                                       #
# For more theoretical functions, # see ./eos.py.           #
#############################################################

import numpy as np
import sys
import eos


def enskog(pf, sigma, T, m, k=1.0, rdf=eos.rdf_PY):
    """ Returns the theoretical value of the 
        viscosity for a given packing fraction.
    """
    V_excl = 2*np.pi*(sigma**3)/3
    eta_0 = zero_density_viscosity(m, sigma, T, k)
    rho = 6*pf/np.pi
    g = rdf(pf)
    eta = eta_0 * (
        1/g 
        + 0.8 * V_excl * rho
        + 0.776 * V_excl**2 * rho**2 * g
    )
    return eta


def zero_density_viscosity(m, sigma, T, k):
    return 5 * np.sqrt((m*k*T)/np.pi) / (16*sigma**2)

def get_enskog_from_C(C):
    pf = C["PF"]
    T = C["TEMP"]
    N_list = np.array([C["N_L"], C["N_H"]])
    sigma_list = np.array([C["SIGMA_L"], C["SIGMA_H"]])
    mass_list = np.array([C["MASS_L"], C["MASS_H"]])
    x = N_list/np.sum(N_list)
    return enskog(pf, sigma_list[0], T, mass_list[0])

def get_thorne_from_C(C, rdf):
    pf = C["PF"]
    T = C["TEMP"]
    N_list = np.array([C["N_L"], C["N_H"]])
    sigma_list = np.array([C["SIGMA_L"], C["SIGMA_H"]])
    mass_list = np.array([C["MASS_L"], C["MASS_H"]])
    x = N_list/np.sum(N_list)
    return thorne(pf, x, mass_list, sigma_list, T, rdf)

def thorne(pf, x, m, sigma_list, T, rdf=eos.rdf_SPT):
    N       = len(x)

    sigma   = get_sigma(sigma_list)
    rho     = 6*pf/np.pi/np.sum(x*np.diag(sigma)**3)
    b       = get_b(sigma)
    alpha   = get_alpha(b)
    eta_0   = get_eta_0(N, m, T, sigma)

    sigma_eff = get_effective_sigma(pf, sigma, x, rho, rdf)
    Xi      = get_Xi(x, sigma_eff, N, rho)
    y       = get_y(x, m, sigma, alpha, Xi, N, rho)
    H       = get_H(x, Xi, eta_0, m, N, rho)
    omega   = get_omega_mix(N, rho, x, eta_0, Xi, alpha)

    eta     = eta_mix(H,  y, omega)
    return eta

def eta_mix(H, y, omega):
    # Stack (H,y) so that we have Hy = [[H ... H y] ... [ y... y 0]]
    Hy = np.insert(H, len(H[0,:]), y, axis=1)
    y_zero = np.insert(y, len(y), 0)
    Hy = np.insert(Hy, len(Hy[:,0]), y_zero, axis=0)

    eta = -np.linalg.det(Hy)/np.linalg.det(H) + 3*omega/5
    return eta

def get_sigma(sigma_list):
    N = len(sigma_list)
    sigma_ij = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            sigma_ij[i,j] = (sigma_list[i] + sigma_list[j])/2
    return sigma_ij


def get_eta_0(N, m, T, sigma, k=1):
    eta_0 = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            m_reduced = 2*m[i]*m[j] / (m[i]+m[j])
            eta_0[i,j] = zero_density_viscosity(
                    m_reduced, sigma[i,j], T, k
                )
    return eta_0


def get_y(x, m, sigma, alpha, Xi, N, rho):
    """ Returns a list of all y_i in the Thorne 
        equation for multicomponent fluids.
    """
    y = np.ones(N)
    for i in range(N):
        for j in range(N):
            y[i] += m[j]/(m[i]+m[j]) * x[j] * alpha[i,j] * Xi[i,j] * rho
    y = y * x
    return y


def get_effective_sigma(pf, sigma, x, rho, rdf):
    N = len(sigma)
    Xi = np.zeros(N)
    for i in range(N):
        Xi[i] = rdf(sigma, x, rho, i, i)

    sigma_eff_list = (12/(5*np.pi*rho) * (Xi - 1))**(1/3)
    sigma_eff = get_sigma(sigma_eff_list)
    return sigma_eff

def get_Xi(x, sigma, N, rho):
    Xi = np.ones((N,N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                ij = sigma[i,j]
                ik = sigma[i,k]
                jk = sigma[j,k]
                Xi[i,j] += rho*np.pi/36 * (
                    x[k]*(ij**3     
                        + ik**3 * (ik / ij)**3
                        + jk**3 * (jk / ij)**3
                        + 18 * (ik**2 * jk**2 / ij)
                        + 16 * (ik**3 + jk**3)    
                        + 16 * (ik * jk / ij)**3
                        -  9 * (ik**2 + jk**2) 
                             * (ij + (ik**2 * jk**2)/ij**3)
                        -  9 * (ik**4 + jk**4) / ij
                    )
                )
    return Xi


def get_H(x, Xi, eta_0, m, N, rho):
    A = np.ones((N,N))
    H = np.zeros((N,N))
    for i in range(N):
        H[i,i] += x[i]**2*Xi[i,i]/eta_0[i,i]
        for j in range(N):
            if i!=j:
                H[i,i] += (
                    2*x[i]*x[j]*Xi[i,j] / eta_0[i,j]
                    * m[i]*m[j]/(m[i] + m[j])**2
                    * (5/(3*A[i,j]) + m[j]/m[i])
                )
                H[i,j] = (
                    -2*x[i]*x[j]*Xi[i,j] / eta_0[i,j]
                    * m[i]*m[j]/(m[i] + m[j])**2
                    * (5/(3*A[i,j]) - 1)
                )
    return H
                

def get_alpha(b):
    """ Returns the matrix alpha_ij, which can be
        found either by modified Enskog theory,
        or less precicely from sigma.
        Input:
            sigma:  np.array of shape (n,n).
                    Contains the center-to-center
                    distance between to particles
                    upon contact.
    """
    alpha = 4*b/5
    return alpha


def get_b(sigma):
    b_ij = 2*np.pi/3 * sigma**3
    return b_ij


def get_omega_mix(N, rho, x, eta_0, Xi, alpha):
    omega = 0
    for i in range(N):
        for j in range(N):
            omega += x[i] * x[j] * eta_0[i,j] * alpha[i,j]**2 * Xi[i,j]
    omega *= 25*rho**2/16
    return omega



def get_viscosity(Ptot, A, t, dv_dz):
    """ Computes the measured value of the viscosity,
        from the Müller-Plathe experiment.

        This function may give warning about divisions 
        by zero. This is not a problem, since these 
        values are first in the list, and are not used 
        in the actual computation of the viscosity.
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
