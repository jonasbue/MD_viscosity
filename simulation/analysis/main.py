import numpy as np
import matplotlib.pyplot as plt
import sys

import files
import viscosity
import plotting
import eos

sysargs = sys.argv

# Convert all files in data to csv format.
if "convert" in sysargs:
    files.all_files_to_csv("data")
packing_list = files.find_all_packing_fractions("data")
#filenames = files.find_all_filenames("data")
cut_fraction = 0.9
per_time=True

def main_viscosity():
    C = {}
    if "verbose" in sysargs:
        print("Number of packings:", len(packing_list)) 
    PF_list = np.zeros(len(packing_list))
    eta_list = np.zeros(len(packing_list))
    std_err_list = np.zeros((2, len(packing_list)))

    for (i, packing) in enumerate(packing_list):
        if "verbose" in sysargs:
            print(packing)
        fix_name = f"data/fix.viscosity_eta_{packing}.lammps"
        log_name = f"data/log.eta_{packing}.lammps"

        # Plot velocity profiles and regressions of them
        if "plot-profiles" in sysargs:
            plotting.plot_velocity_profile_from_file(fix_name)


        # Compute and plot viscosity for all packing fractions
        eta, C, eta_max, eta_min = viscosity.find_viscosity_from_files(
            log_name, fix_name, per_time
        )
        cut = int(cut_fraction*len(eta))
        print(f"Cutting the first {cut} values.")

        PF_list[i] = C["PF"]
        eta_list[i] = np.mean(eta[cut:])
        eta_error = np.array([eta_min, eta_max])
        std_err_list[:,i] = np.mean(eta_error[:,cut:], axis=1)
        if "time-to-eq" in sysargs:
            plt.plot(np.linspace(0,100,num=len(eta)), eta)
            plt.show()
        if "verbose" in sysargs:
            print(eta_list) 

    plotting.plot_viscosity(
        6*packing_list/np.pi,
        eta_list,
        std_err_list,
    )

    # Plot theoretical Enskog equation
    m, sigma, T, N = C["MASS"], C["SIGMA"], C["TEMP"], C["N"]
    pf = np.linspace(0,0.6)
    rho = 6*pf/np.pi
    plt.plot(rho, viscosity.enskog(pf, sigma, T, m, k=1.0))
    plt.show()


def main_equation_of_state():
    # Calculate values of Z from measured p, V and T.
    for eta in eta_list:
        filename = f"log.eta_{eta}.lammps"
        # constants = convert.extract_constants_from_log(filename)
        log_table = files.load_system(filename)
        pvt = files.unpack_varables(log_table, filename, variable_list)
        plot_Z(
            np.mean(pvt[variable_list.index("p")]), 
            np.mean(pvt[variable_list.index("V")]), 
            np.mean(pvt[variable_list.index("T")]), 
            eta
        )

    # Plot theoretical values, from CS-EoS
    eta_range = np.linspace(0, 0.5)
    plt.plot(
        eta_range, 
        Z_Carnahan_Starling(eta_range), 
        "-", 
        label="Carnahan-Starling EoS",
        linewidth=3
    )

    # Show figure
    plt.xlabel("Packing fraction")
    plt.ylabel("Compressibility factor")
    plt.legend()
    plt.show()


if "eos" in sysargs:
    main_equation_of_state()

if "viscosity" in sysargs:
    main_viscosity()
