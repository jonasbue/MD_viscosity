import pandas as pd
import numpy as np 
from scipy.optimize import curve_fit

import save
import theory

def simplified_enskog(pf, T, omega, sigma):
    sigma_c = np.array([1.0])
    sigma = np.array([sigma])
    x = 1.0
    rho = theory.pf_to_rho(sigma_c, x, pf)
    F = theory.F_kolafa
    g = theory.get_rdf_from_F(theory.F_thol, sigma_c, x, rho, T)
    #g = theory.get_rdf_from_F(F, sigma_c, x, rho, T)
    # Use sigma=1.0 and omega=omega in eta_0. 
    # All variation in effective particle size in the 
    # zero-density limit is accounted for by omega.
    V_excl = 2*np.pi*(sigma.flatten()**3)/3
    eta_0 = theory.zero_density_viscosity(1.0, 1.0, T, 1.0, omega)
    eta = eta_0 * (
        1/g 
        + 0.8 * V_excl * rho
        + 0.776 * V_excl**2 * rho**2 * g
    )
    return eta


def fit_viscosity(filename, eq, resolution, fit_sigma=True, pf_max = 0.2, xmin=0.01, xmax=0.51):
    data = pd.read_csv(filename, sep=", ", engine="python")
    pf_original = data["pf"]

    data = data.loc[data["pf"] <= pf_max]
    x = data["pf"]
    pf = np.unique(x)
    T = np.unique(data["T"])
    visc = np.array(data["viscosity"])
    # First index gives temperature, second gives PF.
    visc = np.reshape(visc, (len(T), len(pf)), order="F")

    # Arrays to store output
    # One value for every (pf,T)-point.
    pf_save = np.linspace(xmin, xmax, resolution)
    data_shape = (len(T), len(pf_save))
    eta = np.zeros(data_shape)  
    omega = np.zeros(data_shape)
    sigma = np.zeros(data_shape)
    for i in range(len(T)):
        def f(pf, omega, sigma):
            return simplified_enskog(pf, T[i], omega, sigma)
        def g(pf, omega):
            return simplified_enskog(pf, T[i], omega, 1.0)
        if fit_sigma:
            values, covariance = curve_fit(f, pf, visc[i])
            collision_integral, sigma_eff = values[0], values[1]
            std_dev = np.sqrt(np.diag(covariance))
            #print(f"T={T[i]:.2f}\tomega={collision_integral:.2f}+/-{std_dev[0]:.2f}\tsigma={sigma_eff:.2f}+/-{std_dev[1]:.2f}")
            #print(f"eta={eta[i]}")
            eta[i,:] = simplified_enskog(pf_save, T[i], collision_integral, sigma_eff)
            omega[i,:] = collision_integral
            sigma[i,:] = sigma_eff
        else:
            values, covariance = curve_fit(g, pf, visc[i])
            collision_integral = values[0]
            std_dev = np.sqrt(np.diag(covariance))[0]
            #print(f"T={T[i]:.1f}\tomega={collision_integral:.2f}+/-{std_dev:.2f}")
            omega[i,:] = collision_integral
    return omega, eta, sigma


