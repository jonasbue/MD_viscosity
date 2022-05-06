#############################################################
# This file contains functions that calculate viscosity.    #
# Only theoretical mathematical functions are included,     #
# for functions related to the simulation or other          #
# computational aspects, see the files ./muller_plathe.py   #
# or ./regression.py.                                       #
#############################################################

import numpy as np
import sys
import utils


def enskog_2sigma(pf, sigma_1, T, m, rdf, k=1.0, sigma_2=1.0):
    """
        To allow the same framework for this as the other visc functions,
        sigma_2=1.0 => sigma_1 == sigma_2.

    """
    V_excl = 2*np.pi*(sigma_1**3)/3
    eta_0 = zero_density_viscosity(m, sigma_2, T, k)
    rho = 6*pf/np.pi
    g = rdf(pf) # should use sigma_2
    eta = eta_0 * (
        1/g 
        + 0.8 * V_excl * rho
        + 0.776 * V_excl**2 * rho**2 * g
    )
    return eta


def enskog(pf, sigma, T, m, rdf, k=1.0):
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

def get_viscosity_from_C(C, viscosity, rdf):
    pf = C["PF"]
    T = C["TEMP"]
    N_list = utils.get_component_lists(C, "N")
    sigma_list = utils.get_component_lists(C, "SIGMA")
    mass_list = utils.get_component_lists(C, "MASS")
    x = N_list/np.sum(N_list)
    visc = np.zeros(C["ATOM_TYPES"])
    for i in range(len(visc)):
        visc[i] = viscosity(pf, sigma_list[i], T, mass_list[i], rdf)
    return visc

def get_thorne_from_C(C, rdf):
    """ Takes a system configuration and an RDF function,
        and computes the Thorne viscosity of the system.
        Input:
            C:      Configuration of the system. Dict.
                    Can be obtained from a LAMMPS log file.
                    C must contain the following:
                        PF:         Packing fraction 
                        TEMP:       Temperature
                        N_L:        Number of particle 1
                        N_H:        Number of particle 2
                        SIGMA_L:    Diameter of particle 1
                        SIGMA_H:    Diameter of particle 2
                        MASS_L:     Mass of particle 1
                        MASS_H:     Mass of particle
            rdf:    Either of the rdf functions from this script.
        Output:
            thorne: Viscosity of system. Float.
    """
    pf = C["PF"]
    T = C["TEMP"]
    N_list = np.array([C["N_L"], C["N_H"]])
    sigma_list = np.array([C["SIGMA_L"], C["SIGMA_H"]])
    mass_list = np.array([C["MASS_L"], C["MASS_H"]])
    x = N_list/np.sum(N_list)
    return thorne(pf, x, mass_list, sigma_list, T, rdf)

def thorne(pf, x, m, sigma_list, T, rdf):
    N       = len(x)

    sigma   = get_sigma(sigma_list)
    rho     = 6*pf/np.pi/np.sum(x*np.diag(sigma)**3)
    b       = get_b(sigma)
    alpha   = get_alpha(b)
    eta_0   = get_eta_0(N, m, T, sigma)

    #sigma_eff = get_effective_sigma(pf, sigma, x, rho, rdf)
    #Xi      = get_Xi(x, sigma_eff, N, rho)
    Xi      = get_rdf(x, sigma, N, rho, rdf)
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


def get_rdf(x, sigma, N, rho, rdf):
    Xi = np.ones((N,N))
    for i in range(N):
        for j in range(N):
            Xi[i,j] = rdf(sigma, x, rho, i, j)
    return Xi


def get_Xi(x, sigma, N, rho):
    """ 
        This obtains an RDF using the formula in di Pippo et al, which is a
        semi-empirical method. Results are qualitatively the same, but this is
        not tested.
        IMPORTANT: This method is now obsolete, and should not be used.
    """
    print("\
        WARNING: Using an obsolete method to compute RDFs. \
        Check source code that computes viscosity. \
        ")
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


##############################################################
## Equation of state functions                              ##
##############################################################

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
    sigma = get_sigma(sigma_list)
    rho = pf_to_rho(sigma, x, pf)
    Z = eos(sigma, x, rho, temp=T)
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

def Z_CS(sigma, x, rho, **kwargs):
    """ Returns expected compressibility factor
        based on the Carnahan-Starling EoS,
        for a given packing fraction.
        Inputs:
            pf:  packing franction of system, float.
        Returns:
            Z:  compressibility factor, float.
    """
    pf = rho_to_pf(sigma, x, rho)
    return (1 + pf + pf**2 - pf**3) / (1 - pf)**3


def Z_BN(sigma, x, rho, **kwargs):
    """ Returns expected compressibility factor
        based on the Boublik-Nezbeda EoS,
        for a given packing fraction pf.
        Inputs:
            pf:  packing franction of system, float.
        Returns:
            Z:  compressibility factor, float.
    """
    pf = rho_to_pf(sigma, x, rho)
    return (1 + pf + pf**2 - 2/3*pf**3*(1+pf)) / (1 - pf)**3


def Z_SPT(sigma, x, rho, **kwargs):
    """ Returns expected compressibility factor
        based on the SPT EoS,
        for a given packing fraction.
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


def Z_PY(sigma, x, rho, **kwargs):
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


def Z_BMCSL(sigma, x, rho, **kwargs):
    xi = partial_pf(sigma, x, rho)
    Z = (
        1/xi(0) * (
            xi(0)/(1-xi(3)) 
            + 3*xi(1)*xi(2) / (1-xi(3))**2
            + (3-xi(3))*xi(2)**3 / (1-xi(3))**3
        )
    )
    return Z


def Z_kolafa(sigma, x, rho, temp=1.0, Z_HS=Z_BN, **kwargs):
    """ Computes the EOS of a Lennard-Jones fluid, 
        using the EOS of Kolafa et al. 
        Input:
            sigma:  Diameter of the particles.
            x:      Mole fraction of the particles. Should be 1, 
                    because this EOS does not apply to mixtures.
                    Required in Z_HS().
            rho:    Density of the fluid.
            temp:   Temperature.
            Z_HS:   An EOS function for a HS system, of the same
                    (sigma, x rho) configuration.
        Output:
            Z:      Compressibility factor of the system.
    """
        
    # Coefficients of the EOS, from table 3.
    # The indices are defined as follows in the original paper:
    #   i in {0,-1,-2,-4}; j in [2..6].
    # These indices are set to zero, giving no contribution to the EOS.
    C_ij = np.array([
        #0  j=1 j=2         j=3         j=4         j=5         j=6
        [0, 0,  2.015,      -28.1788,   28.283,     -10.424,    0],         # i=0
        [0, 0,  -19.5837,   75.623,     -120.7059,  93.927,     -27.3774],  # i=-1
        [0, 0,  29.347,     -112.3536,  170.649,    -123.06669, 34.4229],   # i=-2
        [0, 0,  0,          0,          0,          0,          0],         # i=-3
        [0, 0,  -13.37,     65.3806,    -115.0923,  88.9197,    -25.6210],  # i=-4
    ])
    C_d_hBH = np.array([
        1.08014,    # 0
        0.00069,    # 1
        -0.06392,   # ln
        0.01112,    # -2
        -0.07638,   # -1
    ])
    C_delta_B2 = np.array([
        0.0246,     # 0
        -0.5854,    # -7
        0.4310,     # -6
        0.8736,     # -5
        -4.137,     # -4
        2.9062,     # -3
        -7.0218,    # -2
        0,          # -1
    ])
    gamma = 1.92907278  # Damping parameter. Adjustable.
                        # Value from table 3.

    #delta_B cmes from eq. 29 in the paper, with coefficients from above
    def f(T):
        a = 0
        for i in range(-7,1): # 1 is not included.
            a += C_delta_B2[i]*T**(i/2)
        return a + C_d_hBH[2]*np.log(T)

    delta_B = f(temp)
    a = Z_HS(sigma, x, rho) 
    b = rho*(1-2*gamma*rho**2)*np.exp(-gamma*rho**2)*delta_B

    c = 0
    # FIX this
    for i in range(-4,1):
        for j in range(0,7):
            c += j*C_ij[i,j]*temp**(i/2-1)*rho**j 
    Z = a + b + c
    return Z

# LJ EOSes to implement:
# Kolafa (done)
# Gottschalk
# Thol
# Mecke
# Hess


##############################################################
## Radial distribution functions                            ##
##############################################################

def get_rdf_from_C(C, g, i=1, j=1):
    pf = C["PF"]
    T = C["TEMP"]
    N_list = utils.get_component_lists(C, "N")
    sigma_list = utils.get_component_lists(C, "SIGMA")
    mass_list = utils.get_component_lists(C, "MASS")
    x = N_list/np.sum(N_list)
    rdf = np.zeros(C["ATOM_TYPES"])
    for k in range(len(rdf)):
        rdf[k] = g(pf, sigma_list[k], T, mass_list[k], i, j)
    return rdf



def rdf_CS(pf, *args):
    xi = pf
    #return (1-xi/2)/(1-xi)**2
    return 1/(1-xi) + 3/2*xi/(1-xi)**2 + 1/2*xi**2/(1-xi)**3


def rdf_PY(pf, *args):
    """ Returns the thoretical radial distribution 
        function, as given in Pousaneh and de Wijn's paper.

    """
    xi = pf
    return (1+xi/2)/(1-xi)**2


def rdf_SPT_one(pf, *args):
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


def rdf_LJ(pf, T=1.0, *args):
    """ 
        Gives the RDF (at contact) for a one-component Lennard-Jones fluid,
        as given by Morsali et al. r = 1 gives the RDF at contact.
    """
    rho = 6*pf/np.pi
    # Coefficients for the parameters. First index (horizontal) is the 
    # coefficient number, 
    #   q1, q2, q3, q4, q5, q6, q7, q8, q9, rmsd,
    # and the second index (vertical) is the parameter
    #   0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    #   a, b, c, d, g, h, k, l, m, n, s
    q = np.array([
        [9.24792, -2.64281, 0.133386, -1.35932, 1.25338, 0.45602, -0.326422, 0.045708, -0.0287681, 0.02],   # a
        [-8.33289, 2.1714, 1.00063, 0, 0, 0, 0, 0, 0, 0.002],                                               # b
        [-0.0677912, -1.39505, 0.512625, 36.9323, -36.8061, 21.7353, -7.76671, 1.36342, 0, 0.0345],         # c
        [-26.1615, 27.4846, 1.68124, 6.74296, 0, 0, 0, 0, 0, 0.0106],                                       # d
        [0.663161, -0.243089, 1.24749, -2.059, 0.04261, 1.65041, -0.343652, -0.037698, 0.008899, 0.0048],   # g
        [0.0325039, -1.28792, 2.5487, 0, 0, 0, 0, 0, 0, 0.0015],                                            # h
        [16.4821, -0.300612, 0.0937844, -61.744, 145.285, -168.087, 98.2181, -23.0583, 0, 0.0074],          # k
        [-6.7293, -59.5002, 10.2466, -0.43596, 0, 0, 0, 0, 0, 0.008],                                       # l
        [1.25225, -1.0179, 0.358564, -0.18533, 0.0482119, 1.27592, -1.78785, 0.634741, 0, 0.0392],          # s
        [-5.668, -3.62671, 0.680654, 0.294481, 0.186395, -0.286954, 0, 0, 0, 0.0096],                       # m
        [6.01325, 3.84098, 0.60793, 0, 0, 0, 0, 0, 0, 0.002],                                               # n
    ])
    # There is one unique(ish) equation P_ji for every coefficient a, b, c ...
    # While the paper gives an expression for g(r), only the equations
    # that give g(sigma) are included here.
    # a
    #P = ( q[1,i] 
    #    + q[2,i]*exp(-q[3,i]*T) 
    #    + q[4,i]*np.exp(-q[5,i]) 
    #    + q[6,i]/rho 
    #    + q[7,i]/rho**2 
    #    + q[8,i*T]/rho**3
    #)
    s = ( 
            ( q[8,0] 
            + q[8,1]*rho
            + q[8,2]/T
            + q[8,3]/T**2
            + q[8,4]/T**3
        ) / (
            q[8,5]
            + q[8,6]*rho
            + q[8,7]*rho**2
        )
    )
    m = ( q[9,0] 
        + q[9,1]*np.exp(-q[9,3]*T) 
        + q[9,2]/T
        + q[9,4]*rho 
        + q[9,5]*rho**2 
    )
    n = ( q[10,0] 
        + q[10,1]*np.exp(-q[10,2]*T) 
    )
    g = s*np.exp(-(m+n)**4) # r == 1 == sigma
    #g = ( 1 
    #    + r**(-2) * np.exp(-(a*r+b)) * sin(c*r+d)
    #    + r**(-2) * np.exp(-(g*r+h)) * cos(k*r+l)
    #)
    return g
