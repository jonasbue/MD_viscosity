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


def enskog(pf, sigma, T, m, k=1.0):
    """ Returns the theoretical value of the 
        viscosity for a given packing fraction.
    """
    # 
    V_excl = 2*np.pi*(sigma**3)/3
    eta_0 = get_eta_0_single_component(m, sigma, T, k)
    rho = 6*pf/np.pi
    g = eos.rdf_PY(pf)
    eta = eta_0 * (
        1/g 
        + 0.8 * V_excl * rho
        + 0.776 * V_excl**2 * rho**2 * g
    )
    return eta

def get_eta_0_single_component(m, sigma, T, k):
    return 5 * np.sqrt((m*k*T)/np.pi) / (16*sigma**2)

def thorne(pf, x, m, sigma_list, T):
    N       = len(x)
    rho     = 6*pf/np.pi

    sigma   = get_sigma(sigma_list, N)
    b       = get_b(sigma)
    alpha   = get_alpha(b)
    eta_0   = get_eta_0(N, m, T, sigma)

    Xi      = get_Xi(x, sigma, N, rho)
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

def get_sigma(sigma_list, N):
    sigma_ij = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            sigma_ij[i,j] = (sigma_list[i] + sigma_list[j])/2
    return sigma_ij


def get_eta_0(N, m, T, sigma, k=1):
    m_reduced=np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            m_reduced[i,j] = 2*m[i]*m[j] / (m[i]+m[j])
    # TODO: Check if this definition of eta_ij is correct
    eta_0 =  5 * np.sqrt((m*k*T)/np.pi) / (16*sigma**2)
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


def get_Xi(x, sigma, N, rho):
    Xi = np.ones((N,N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                # This is a mess, but I think this solutions is 
                # simpler -- and more readable -- than any other
                ij = sigma[i,j]
                ik = sigma[i,k]
                jk = sigma[j,k]
                Xi[i,j] += rho*np.pi/36 * (
                    x[k]*(ij**3     
                        + ik**3 * (ik / ij)**3
                        + jk**3 * (jk / ij)**3
                        + 18*(ik**2 * jk**2 / ij)
                        + 16*(ik**3 + jk**3)    
                        + 16*(ik * jk / ij)**3
                        - 9*(ik**2 + jk**2) * (ij + (ik**2 * jk**2) / ij**3)
                        - 9*(ik**4 + jk**4) / ij
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

"""
def get_Xi(x, sigma, n):
    b_ij = get_b(sigma)
    Xi = 1
    s0 = np.einsum("i,j->ij", sigma[:,0], sigma[:,0].T)
    s1 = np.einsum("i,j->ij", sigma[:,1], sigma[:,1].T)
    s2 = np.einsum("i->i", sigma[:,0] + sigma[:,0].T)/2
    s3 = np.einsum("i->i", sigma[:,1] + sigma[:,1].T)/2
    print("sigma sigma = ", (s0 + s1)/sigma)
    #print("sigma + sigma = ", np.tile(s2+s3, (2,1)) + np.fliplr(np.diag(s2+s3)))
    for i in range(n):
        for j in range(n):
            for k in range(n):
        Xi += n*np.pi/36* (
            x[k] * (
                sigma**3 
                + np.einsum("i,j->ij", sigma[:,k]**3, sigma[:,k]**3) / sigma**3
                + np.einsum("i,j->ij", sigma[:,k].T**3, sigma[:,k].T) / sigma**3
                + 18*(np.einsum("i,j->ij", sigma[:,k]**2, sigma[:,k].T**2) / sigma)
                + 16*(sigma[:,k]**3 + sigma[:,k].T**3) # ???
                + 16*(np.einsum("i,j->ij", sigma[:,k], sigma[:,k].T) / sigma)**3
                - 9*(sigma[:,k]**2 + sigma[:,k].T**2) # ???
                * (sigma + np.einsum("i,j->ij", sigma[:,k]**2, sigma[:,k].T**2) / sigma**3)
                - 9*(sigma[:,k]**4 + sigma[:,k].T**4) / sigma # ???
            )
        )
    return Xi
"""

