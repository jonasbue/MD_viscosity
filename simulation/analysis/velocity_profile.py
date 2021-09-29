import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import convert_LAMMPS_output as convert
import logfiles

# Increase font size in plots
font = {
    "size"  : "22",
}
plt.rc("font", **font)

def radial_distribution(pf):
    xi = pf
    return (1-xi/2)/(1-xi)**3

def enskog(pf, sigma, T, m, k=1.0):
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


def viscosity(Ptot, L, t, z, dv):
    A = (2*L)**2
    j = Ptot/(2*A*t)
    dv_dz = dv
    eta = -j/dv_dz
    return eta


def get_velocity_profile(filename):
    # Figure out how to handle this if file already exists.
    convert.convert_fix_to_csv(filename)

    A = pd.read_csv(filename+".csv")
    A = np.array(A).transpose()
    zvx = logfiles.unpack_variables(
        A, filename, ["z", "vx"]
    )
    vx = zvx[1]
    z = zvx[0]
    return vx, z


def velocity_profile_regression(vx, z):
    n = int(len(vx)/2)
    print(n)
    lower_half = vx[:n]
    upper_half = vx[n:]
    plt.plot(lower_half, z[:n])
    plt.plot(upper_half, z[n:])
    plt.show()


def plot_velocity_profile(vx, z, packing):
    vx, z = get_velocity_profile(fix_filename)
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
    plt.show()


def find_viscosity(log_filename, fix_filename, packing, dv):
    vx, z = get_velocity_profile(fix_filename)

    variable_list = ["t", "Px"]
    log_table = logfiles.load_system(log_filename)
    log_vals = logfiles.unpack_variables(log_table, log_filename, variable_list)
    constants = convert.extract_constants_from_log(log_filename)

    t = log_vals[variable_list.index("t")]*0.001
    Ptot = log_vals[variable_list.index("Px")]
    packing = constants["PF"]
    L = constants["L"]

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


packing_list = np.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.5])
#packing_list = np.array([0.5])
#dv_list = np.array([4, 4.8, 4.6, 4, 2, 0.8])
dv_list = np.array([2, 4.2, 4.4, 4.2, 3.2, 1.2])
C = {}
for packing in packing_list:
    fix_name = f"data/MP_viscosity_eta_{packing}.profile"
    vx, z = get_velocity_profile(fix_name)
    velocity_profile_regression(vx, z)
    plot_velocity_profile(vx, z, packing)

for (i, packing) in enumerate(packing_list):
    log_name = f"data/log.eta_{packing}.lammps"
    fix_name = f"data/MP_viscosity_eta_{packing}.profile"
    eta, C = find_viscosity(log_name, fix_name, packing, dv_list[i])
    plot_viscosity(packing, eta)

m, sigma, T, N = C["MASS"], C["SIGMA"], C["TEMP"], C["N"]
pf = np.linspace(0,0.5)
plt.plot(pf, enskog(pf, sigma, T, m, k=1.0))
#plt.plot(pf, radial_distribution(pf))
plt.show()
