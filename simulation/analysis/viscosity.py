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


def viscosity(Ptot, A, t, z, dv_dz):
    """ Computes the measured value of the viscosity,
        from the Müller-Plathe experiment.
    """
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
    #plt.plot(lower_half, z_lower, "o")
    #plt.show()
    #plt.plot(upper_half, z_upper, "x")
    #plt.show()
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
    #print(lower_reg.slope)
    #print(upper_reg.slope)
    return lower_reg, upper_reg, z_lower, z_upper

def plot_velocity_regression(lower_reg, upper_reg, z_lower, z_upper):
    print("PLOT:\t", upper_reg.slope)
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

def get_avg(lower, upper):
    return (lower + np.abs(upper))/2

def find_uncertainty(reg):
    #std_err_lower, std_int_err_lower = find_uncertainty(lower_reg)
    #std_err_upper, std_int_err_upper = find_uncertainty(upper_reg)
    max_slope = reg.slope + reg.stderr
    min_slope = reg.slope - reg.stderr
    return max_slope, min_slope

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
    Lx = constants["LX"]
    Ly = constants["LY"]

    t = log_vals[variable_list.index("t")]*constants["DT"]
    t = t - t[0]
    Ptot = log_vals[variable_list.index("Px")]
    
    lower_reg, upper_reg, zl, zu  = velocity_profile_regression(vx, z)
    dv = get_slope(lower_reg, upper_reg)
    print("FIND:\t", upper_reg.slope)


    dev_low_max, dev_low_min = find_uncertainty(lower_reg)
    dev_upp_max, dev_upp_min = find_uncertainty(upper_reg)
    dev_slope_max = get_avg(dev_low_max, dev_upp_max)
    dev_slope_min = get_avg(dev_low_min, dev_upp_min)

    eta = viscosity(Ptot, 2*Lx*2*Ly, t, z, dv)
    eta_max = viscosity(Ptot, 2*Lx*2*Ly, t, z, dev_slope_max)
    eta_min = viscosity(Ptot, 2*Lx*2*Ly, t, z, dev_slope_min)
    return eta, constants, eta_max, eta_min


def plot_viscosity(packing, eta, std_err=None):
    plt.plot(
        packing,
        np.mean(eta[500:]),
        "o",
        color="k",
        label=f"{packing}"
    )
    if std_err.any() != 0:
        #yerr = np.mean(std_err[:,500:],axis=1).T
        yerr = np.mean(std_err[500:])
        print(yerr)
        plt.errorbar(
            packing,
            np.mean(eta[500:]),
            yerr = yerr
        )
    plt.xlabel("Packing fraction")
    plt.ylabel("Viscosity")

