import numpy as np
import matplotlib.pyplot as plt
import sys

import files
import viscosity
import plotting
import tests
import muller_plathe

sysargs = sys.argv

# Convert all files in data to csv format.
if "convert" in sysargs:
    files.all_files_to_csv("data")
packing_list = files.find_all_packing_fractions("data")
#filenames = files.find_all_filenames("data")
cut_fraction = 0.95
per_time=True

def main_viscosity():
    C = {}
    if "verbose" in sysargs:
        print("Number of packings:", len(packing_list)) 
    PF_list = np.zeros(len(packing_list))
    eta_list = np.zeros(len(packing_list))
    error_list = np.zeros(len(packing_list))

    for (i, packing) in enumerate(packing_list):
        if "verbose" in sysargs:
            print(packing)
        fix_name = f"data/fix.viscosity_eta_{packing}.lammps"
        log_name = f"data/log.eta_{packing}.lammps"

        # Plot velocity profiles and regressions of them
        if "plot-profiles" in sysargs:
            plotting.plot_velocity_profile_from_file(fix_name)

        # Compute and plot viscosity for all packing fractions
        eta, C, eta_err = muller_plathe.find_viscosity_from_files(
            log_name, fix_name, cut_fraction, per_time
        )

        PF_list[i] = C["PF"]
        eta_list[i] = np.mean(eta)
        error_list[i] = np.mean(eta_err)
        if "time-to-eq" in sysargs:
            plt.plot(np.linspace(0,100,num=len(eta)), eta)
            plt.show()
        if "verbose" in sysargs:
            print("Viscosities:", eta_list) 
            print("Errors:\n", error_list) 

    plotting.plot_viscosity(
        6*packing_list/np.pi,
        eta_list,
        error_list,
    )

    m, sigma, T, N = C["MASS"], C["SIGMA"], C["TEMP"], C["N"]
    # Plot theoretical Enskog equation
    pf = np.linspace(0,0.6)
    rho = 6*pf/np.pi
    plt.plot(rho, viscosity.enskog(pf, sigma, T, m, k=1.0))
    plt.show()

    #renormalize to relative viscosity:
    plotting.plot_viscosity(
        6*packing_list/np.pi,
        eta_list/viscosity.enskog(packing_list, sigma, T, m),
        error_list/viscosity.enskog(packing_list, sigma, T, m),
    )
    plt.plot(rho, np.ones_like(rho))
    plt.show()


def main_equation_of_state():
    N = 1000
    eta_list = np.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.5])
    variable_list = ["p", "V", "T"]

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


def main_mix():
    tests.test_mix()

if "eos" in sysargs:
    main_equation_of_state()

if "viscosity" in sysargs:
    main_viscosity()

if "mix" in sysargs:
    main_mix()
