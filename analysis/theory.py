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
#import tests


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

def Z_gottschalk(sigma, x, rho, temp=1.0, Z_HS=Z_BN, **kwargs):
    """ Computes the EOS of a Lennard-Jones fluid, 
        using the EOS of Gottschalk. 
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
        
    # Parameters for the thermal virial coefficients of the EOS, from table V.
    B_i = np.array([
        [1.221844737e-1,	-1.832133004e-2,	-5.737837739e-2,    -1.107146794e-1],
        [-2.533814785,		-2.221029066e-1,	2.384059560e-1,		3.639967813e-1],
        [2.321052047,		-2.290140445,		-3.175043752e-1,	-1.722555372e-1],
        [-2.221116991e1,	2.497587053,		1.411210874e-1,		5.355823913e-2],
        [6.037723605e1,		-1.491751608,		-4.065269634e-2,	-9.119290154e-3],
        [-8.614627023e1,	5.194910488e-1,		7.132450669e-3,		6.312327708e-4],
        [7.947702893e1,		-7.580241786e-2,	-7.501879316e-4,	-6.471729317e-6],
        [-5.013039389e1,	-9.570910251e-3,	5.000252419e-5,		-6.635662426e-7],
        [2.179355452e1,		6.444596963e-3,		-2.224242683e-6,    1.145665574e-8],
        [-6.423839356,		-1.323484892e-3,	6.334525666e-8,		-5.093701999e-10],
        [1.222200983,		1.400743960e-4,		-1.124571857e-9,	0],
        [-1.351435025e-1,	-7.861096502e-6,	1.120406875e-11,	0],
        [6.519707093e-3,	1.749011555e-7,		-4.806632984e-14,	0]
    ])
    # Coefficients of the fitted correction parameters of the EOS, from table VI.
    C_i = np.array([
        [-3.848657712e3,	1.214533953e4,		5.841998321e4,		-4.717257385e5,		1.411244301e6,		-2.385034755e6,		2.465995272e6,		-1.550792557e6,		5.466032853e5,		-8.300129372e4],
        [1.940790808e3,		-9.125315944e3,		5.336019753e3,		7.338796875e4,		-2.875424926e5,		5.411774951e5,		-5.990022139e5,		3.971186192e5,		-1.464740110e5,		2.318027845e4],
        [-6.786775725e2,	3.397150517e3,		-6.39357702e3,		1.526191655e3,		2.109845310e4,		-5.532941146e4,		7.174967760e4,		-5.317159494e4,		2.148725004e4,		-3.684215524e3],
        [1.592726729e2,		-7.355400823e2,		1.37142001e3,		-1.479809349e3,		9.648092327e2,		2.842688756e2,		-1.763990307e3,		2.096231490e3,		-1.141346457e3,		2.443351261e2],
        [-2.733389532e1,	1.207984183e2,		-1.843597578e2,		1.033775279e2,		-1.118483146e1,		5.21058649e1,		-1.121805527e2,		7.436426163e1,		-1.333984225e1,		-2.185586771],
        [3.305728801,		-1.595650007e1,		2.756765411e1,		-1.762232710e1,		-5.953997923,		1.569383441e1,		-7.327702248,		-1.053356075,		1.604683226,		-2.645512917e-1],
        [-2.396300005e-1,	1.301737514,		-2.765120060,		3.029682621,		-1.736288154,		2.519375463e-1,		3.057812577e-1,		-1.621886602e-1,	1.067758340e-2,		3.982533293e-3],
        [8.107532579e-3,    -4.828021321e-2,	1.107783436e-1,		-1.382464464e-1,	1.102848617e-1,		-5.923068772e-2,	1.906868401e-2,		-2.542729177e-3,	0,		            0],
        [-5.209209916e-5,	4.779918832e-4,		-1.100471432e-3,	1.049186458e-3,	    -4.233596547e-4,	5.305829584e-5,		0,		            0,		            0,		            0],
        [-1.863883724e-6,	4.808860997e-6,		-4.538508711e-6,	1.455012606e-6,	    0,                  0,                  0,                  0,                  0,                  0,],
        [6.787957968e-9,	-3.433240822e-9,	0,		            0,		            0,                  0,                  0,                  0,                  0,                  0,]
])

    B_SS = np.array(
        [3.79107, 3.52751, 2.11494, 0.76953] # B^SS_i
    )
    ci = np.array(
        [1.529031885, 2.795121498, 4.903830267, 5.539252062] # c_i
    )
    C_SS = np.array(
        [2.356773117e3, -3.264039611e3, -7.804186018e4, 4.734725795e6, -1.317864191e6, 2.146863058e6, -2.165267779e6, 1.335386749e6, -4.628739042e5, 6.922915835e4] # C^SS_i
    )
    di = np.array(
        [4.85, 4.85, 4.85, 4.85, 4.85, 4.85, 4.85, 4.85, 4.85, 4.85] # d_i
    )

    T = temp
    # B_2 is known excactly:
    # TODO: B_2
    # def I(alpha, x):

    # Compute the virial coefficients:
    # B (thermal virial coefficients) and C (correction virial coefficients)
    # are almost identically defined. Therefore, call virial_coefficient()
    # with the corresponding parameter arrays to compute them.
    def virial_coefficient(i, T, n, A_i, A_SS, ai):
        # i starts at n, so subtract n in every list index
        I = i-n
        # ki is just the number of indices. Affects the precision
        ki = A_i.shape[0]
        value = A_SS[I]
        for k in range(1,ki+1):
            #   k starts at 1, so subtract 1 in every list
            K = k-1
            value += A_i[K,I] * (np.exp(ai[I]/np.sqrt(T)-1)**((2*k-1)/4))
        return (T/4)**(-(i-1)/4) * value


    b = 0
    n = 2
    def B(i, T):
        return virial_coefficient(i, T, n+1, B_i, B_SS, ci)
    # Test-plotting the virial coefficients. Due to the definition
    # of the virial_coefficient() function, this must be called here.
    # (2,7) comes only from the fact that B_3-B_6 are given in the paper
    for i in range(n,7):
        b += rho**(i-2) * B(i, T)

    c = 0
    n = 7
    def C(i, T):
        return virial_coefficient(i, T, n, C_i, C_SS, di)
    #tests.test_virial_coefficients(1, C)
    # C_7,...,C_17 are given in the paper
    for i in range(n,17):
        c += rho**(i-2) * C(i, T)
    Ar_01 = rho * (b + 0*c)
    Z = 1 + Ar_01
    return Z


def Z_thol(sigma, x, rho, temp=1.0, Z_HS=Z_BN, **kwargs):
    """ Computes the EOS of a Lennard-Jones fluid, 
        using the EOS of Thol et al. 
        Validity range: T in (0.661,9), p<65.
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

    # Fitted parameters: n, t, d, l, eta, beta, gamma and epsilon.
    # Given in Table 2 of the paper.
    n = np.array([
        0.52080730e-2,	0.21862520e+1,	-0.21610160e+1,	0.14527000e+1,
        -0.20417920e+1,	0.18695286e0,	-0.90988445e-1,	-0.49745610e0,
        0.10901431e0,	-0.80055922e0,	-0.56883900e0,	-0.62086250e0,
        -0.14667177e+1,	0.18914690e+1,	-0.13837010e0,	-0.38696450e0,
        0.12657020e0,	0.60578100e0,	0.11791890e+1,	-0.47732679e0,
        -0.99218575e+1,	-0.57479320e0,	0.37729230e-2
    ])
    t = np.array([
        1.000, 0.320, 0.505, 0.672, 0.843, 0.898, 1.294, 2.590, 
        1.786, 2.770, 1.786, 1.205, 2.830, 2.548, 4.650, 1.385, 
        1.460, 1.351, 0.660, 1.496, 1.830, 1.616, 4.970
    ])
    d = np.array([
        4, 1, 1, 2, 2, 3, 5, 2, 2, 3, 1, 1, 1, 1, 2, 3, 3, 2, 1, 2, 3, 1, 1
    ])
    l = np.array([
        0, 0, 0, 0, 0, 0, 1, 2, 1, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ])
    eta = np.array([
		0,		0,		0,		0,		0,		0,		
        0,		0,		0,		0,		0,		0,		
		2.067,	1.522,	8.820,	1.722,	0.679,	1.883,	
		3.925,	2.461,	28.20,	0.753,	0.820
    ])
    beta = np.array([
		0,		0,		0,		0,		0,		0,		
        0,		0,		0,		0,		0,		0,		
		0.625,	0.638,	3.910,	0.156,	0.157,	0.153,
		1.160,	1.730,	383.0,	0.112,	0.119
    ])
    gamma = np.array([
		0,		0,		0,		0,		0,		0,		
        0,		0,		0,		0,		0,		0,		
		0.710,	0.860,	1.940,	1.480,	1.490,	1.945,
		3.020,	1.110,	1.170,	1.330,	0.240
    ])
    epsilon = np.array([
		0,		0,		0,		0,		0,		0,		
        0,		0,		0,		0,		0,		0,		
		0.2053,	0.4090,	0.6000,	1.2030,	1.8290,	1.3970,
		1.3900,	0.5390,	0.9340,	2.3690,	2.4300
    ])

    # Should it be 1 or Z_HS? Discuss with supervisors.
    Z = Z_HS(sigma, x, rho)
    tau = 1/temp
    for i in range(6):
        Z += n[i]*rho**(d[i]-1)*d[i]*tau**t[i]
    for i in range(7,12):
        Z += n[i]*tau**t[i]*np.exp(-rho**l[i]) * (
                d[i]*rho**(d[i]-1)
                + rho**d[i]*(-l[i]*rho**(l[i]-1))
            )
    for i in range(13,23):
        Z += n[i]*tau**t[i]*np.exp(
                -beta[i]*(tau-gamma[i])**2 - eta[i]*(rho-epsilon[i])**2
            ) * (
                d[i]*rho**(d[i]-1) - rho**d[i]*eta[i]*2*(rho-epsilon[i])
        )
    return Z

def Z_mecke(sigma, x, rho, temp=1.0, Z_HS=Z_BN, **kwargs):
    """ Computes the EOS of a Lennard-Jones fluid, 
        using the EOS of Mecke et al. (1996).
        Validity range: 
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
    c = np.array([
        0.33619760720e-05,	 -0.14707220591e+01,	-0.11972121043e+00,	
        -0.11350363539e-04,	 -0.26778688896e-04,    0.12755936511e-05,	
        0.40088615477e-02,	 0.52305580273e-05,	    -0.10214454556e-07,	
        -0.14526799362e-01,	 0.64975356409e-01,	    -0.60304755494e-01,	
        -0.14925537332e+00,	 -0.31664355868e-03,    0.28312781935e-01,	
        0.13039603845e-03,	 0.10121435381e-01,	    -0.15425936014e-04,	
        -0.61568007279e-01,	 0.76001994423e-02,	    -0.18906040708e+00,	
        0.33141311846e+00,	 -0.25229604842e+00,    0.13145401812e+00,	
        -0.48672350917e-01,	 0.14756043863e-02,	    -0.85996667747e-02,	
        0.33880247915e-01,	 0.69427495094e-02,	    -0.22271531045e-07,	
        -0.22656880018e-03,	 0.24056013779e-02
    ])
    m = np.array([
        -2.0,	-1.0,	-1.0,	-1.0,	-0.5,	-0.5,	0.5,	0.5,	
        1.0,	-5.0,	-4.0,	-2.0,	-2.0,	-2.0,	-1.0,	-1.0,	
        0.0,	0.0,	-5.0,	-4.0,	-3.0,	-2.0,	-2.0,	-2.0,	
        -1.0,	-10.0,	-6.0,	-4.0,	0.0,	-24.0,	-10.0,	-2.0
    ])
    n = np.array([
        9,	1,	2,	9,	8,	10,	1,	7,	
        10,	1,	1,	1,	2,	8,	1,	10,	
        4,	9,	 2,	5,	1,	2,	3,	4,	
        2,	3,	4,	2,	2,	5,	2,	10
    ])
    p = np.array([
        0,	0,	0,	0,	0,	0,	0,	0,	
        0,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	
        -1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	
        -1,	-1,	-1,	-1,	-1,	-1,	-1,	-1
    ])
    q = np.array([
        0, 0, 0, 0, 0, 0, 0, 0, 
        0, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 2, 2, 2, 2, 2, 
        2, 3, 3, 3, 3, 4, 4, 4
    ])

    Z = Z_HS(sigma, x, rho)
    rho_c = 0.3107
    T_c = 1.328
    rho = rho/rho_c     # This EOS uses rho and T normalized by
    T = temp/T_c        # the critical temperature and density.
    for i in range(2):
        Z += rho*c[i]*T**m[i]*n[i] * np.exp(p[i]*rho**q[i]) * (
                rho**(n[i]-1) + p[i]*q[i]*rho**(q[i]-1)
            )
    return Z


def Z_hess(sigma, x, rho, temp=1.0, Z_HS=Z_BN, **kwargs):
    """ Computes the EOS of a Lennard-Jones fluid, 
        using the EOS of Hess (1998).
        Validity range: 
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



#tests.test_eos()



# LJ EOSes to implement:
# N     Name                    Implemented?    Working?
# 1.    Kolafa and Nezbeda      Yes             Yes
# 2.    Gottschalk              Yes             No
# 3.    Thol                    Yes             No
# 4.    Mecke                   No              No
# 5.    Hess                    No              No

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
