import numpy as np

def make_unit_dict(r, t, T, p, pf):
    rho = pf*6/np.pi
    units = {
        "r"     : r,
        "t"     : t,
        "T"     : T,
        "p"     : p,
        "rho"   : rho,
    }
    return units


def lj_to_real_units(sigma, m, epsilon, lj):
    Kb  = 1.38e-23
    r   = lj["r"]*sigma
    t   = lj["t"]/np.sqrt(epsilon/(m*sigma**2))
    T   = lj["T"]*epsilon/Kb
    p   = lj["p"]*epsilon/sigma**3
    rho = lj["rho"]*m/sigma**3
    real = make_unit_dict(r, t, T, p, rho)
    return real
    

def real_to_lj_units(sigma, m, epsilon, real):
    Kb  = 1.38e-23
    r   = real["r"]/sigma
    t   = real["t"]*np.sqrt(epsilon/(m*sigma**2))
    T   = real["T"]*Kb/epsilon
    p   = real["p"]*sigma**3/epsilon
    rho = real["rho"]*sigma**3
    lj = make_unit_dict(r, t, T, p, rho)
    return lj
