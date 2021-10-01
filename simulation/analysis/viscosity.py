import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import convert_LAMMPS_output as convert
import logfiles

# Increase font size in plots
font = {
    "size"  : "22",
}
plt.rc("font", **font)

def radial_distribution(pf):
    """ Returns the thoretical radial distribution 
        function, as given in Faezeh and de Wijn's paper.
    """
    xi = pf
    return (1-xi/2)/(1-xi)**3

def enskog(pf, sigma, T, m, k=1.0):
    """ Returns the theoretical value of the 
        viscosity for a given packing fraction.
    """
    eta_0 =  5 * np.sqrt((m*k*T)/np.pi) / (16*sigma**2)
    V_excl = 2*np.pi*(sigma**3)/3
    rho = 6*pf/np.pi
    g = radial_distribution(pf)
    eta = eta_0 * (
        1/g 
        + 0.8 * V_excl * rho
        + 0.776 * V_excl**2 * rho**2 * g
    )
    return eta


def viscosity(Ptot, L, t, z, dv_dz):
    """ Computes the measured value of the viscosity,
        from the Müller-Plathe experiment.
    """
    A = (2*L)**2
    j = Ptot/(2*A*t)
    eta = -j/dv_dz
    return eta


def get_velocity_profile(filename):
    log_table = logfiles.load_system(filename)
    zvx = logfiles.unpack_variables(
        log_table, filename, ["z", "vx"]
    )
    z = zvx[0]
    vx = zvx[1]
    return vx, z


def velocity_profile_regression(vx, z):
    n = int(len(vx)/2)
    lower_half = vx[:n]
    upper_half = vx[n:]
    z_lower = z[:n]
    z_upper = z[n:]
    lower_reg = stats.linregress(
        z_lower, 
        lower_half, 
        alternative="greater"
    )
    upper_reg = stats.linregress(
        z_upper, 
        upper_half, 
        alternative="less"
    )
    return lower_reg, upper_reg

def plot_velocity_regression(lower_reg, upper_reg, z_lower, z_upper):
    plt.plot(
        lower_reg.slope*z_lower + 1*lower_reg.intercept, 
        z_lower
    )
    plt.plot(
        upper_reg.slope*z_upper + 1*upper_reg.intercept, 
        z_upper
    )

def get_slope(lower_reg, upper_reg):
    return (lower_reg.slope + np.abs(upper_reg.slope))/2


def plot_velocity_profile(vx, z, packing):
    fig_title = f"Velocity profile, $\\xi = {packing}$"
    plt.plot(
        vx, 
        z,
        "o", 
        label=fig_title,
    )
    plt.xlabel("$v_x$")
    plt.ylabel("$z$")
    plt.legend(loc="upper right")


def find_viscosity(log_filename, fix_filename):
    vx, z = get_velocity_profile(fix_filename)

    variable_list = ["t", "Px"]
    log_table = logfiles.load_system(log_filename)
    log_vals = logfiles.unpack_variables(log_table, log_filename, variable_list)

    constants = convert.extract_constants_from_log(log_filename)
    L = constants["L"]

    t = log_vals[variable_list.index("t")]*constants["DT"]
    Ptot = log_vals[variable_list.index("Px")]
    
    lower_reg, upper_reg = velocity_profile_regression(vx, z)
    dv = get_slope(lower_reg, upper_reg)
    print("dv = ", dv)

    eta = viscosity(Ptot, L, t, z, dv)
    return eta, constants


def plot_viscosity(packing, eta):
    plt.plot(
        packing,
        np.mean(eta[200:]),
        "o",
        color="k",
        label=f"{packing}"
    )
    plt.xlabel("Packing fraction")
    plt.ylabel("Viscosity")

