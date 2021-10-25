import numpy as np
import matplotlib.pyplot as plt
import logging
import sys

import files
import viscosity
import plotting
import tests
import muller_plathe
import eos

sysargs = sys.argv
log = logging.getLogger()
log.addHandler(logging.StreamHandler(sys.stdout))
if "debug" in sysargs:
    # Debug is more correct to use, but info is cleaner.
    log.setLevel(logging.INFO)

# Convert all files in data to csv format.
if "convert" in sysargs:
    files.all_files_to_csv("data/one_component")
    files.all_files_to_csv("data/two_component")
    files.all_files_to_csv("data/equillibrium")


def main_viscosity(mix=True):
    log.debug("Number of packings:", len(packing_list)) 
    C = {}
    PF_list = np.zeros(len(packing_list))
    eta_list = np.zeros(len(packing_list))
    error_list = np.zeros(len(packing_list))

    for (i, packing) in enumerate(packing_list):
        log.debug(f"Now computing for packing fraction = {packing}")
        if mix:
            path = "data/two_component/"
            fix_name = f"{path}fix.viscosity_mix_eta_{packing}.lammps"
            log_name = f"{path}log.mix_eta_{packing}.lammps"
        else:
            path = "data/one_component/"
            fix_name = f"{path}fix.viscosity_eta_{packing}.lammps"
            log_name = f"{path}log.eta_{packing}.lammps"
        if "plot-profiles" in sysargs:
            # Plot velocity profiles and regressions of them
            plotting.plot_velocity_profile_from_file(fix_name)

        # Compute and plot viscosity for all packing fractions
        eta, C, eta_err = muller_plathe.find_viscosity_from_files(
            log_name, fix_name, cut_fraction, per_time
        )

        PF_list[i] = C["PF"]
        eta_list[i] = np.mean(eta)
        error_list[i] = np.mean(eta_err)
        log.debug("Viscosity = {eta_list[i]} +/- {error_list[i]}")
    if mix:
        plotting.plot_result_vs_thorne(
            eta_list, 
            packing_list, 
            error_list, 
            C
        )
    else:
        plotting.plot_result_vs_enskog(
            eta_list, 
            packing_list, 
            error_list, 
            C
        )


def mix_eos():
    PF_list = np.zeros(len(packing_list))
    eta_list = np.zeros(len(packing_list))
    error_list = np.zeros(len(packing_list))
    C = {}
    Z_list = np.zeros(len(packing_list))

    for (i, packing) in enumerate(packing_list):
        if "verbose" in sysargs:
            print(packing)
        dir = "data/equillibrium/"
        fix_name = f"{dir}fix.viscosity_mix_eta_{packing}.lammps"
        log_name = f"{dir}log.mix_eta_{packing}.lammps"
        variable_list = ["p", "V", "T"]
        
        # constants = convert.extract_constants_from_log(filename)
        log_table = files.load_system(log_name)
        pvt = files.unpack_variables(log_table, log_name, variable_list)
        Z_list[i] = eos.Z_measured(
            np.mean(pvt[variable_list.index("p")]), 
            np.mean(pvt[variable_list.index("V")]), 
            1000,
            np.mean(pvt[variable_list.index("T")])
        )

    # Plot theoretical values, from CS-EoS
    pf = np.linspace(0, 0.5)
    rho_list = 6*pf/np.pi
    SPT = np.zeros_like(pf)
    PY = np.zeros_like(pf)
    for i, rho in enumerate(rho_list):
        sigma = np.array([1,1])
        sigma = viscosity.get_sigma(sigma)
        x = np.array([0.5,0.5])
        SPT[i] = eos.Z_SPT(sigma, x, rho)
        PY[i] = eos.Z_PY(sigma, x, rho)
    CS = eos.Z_CS(pf)
    pf_vals = np.array([0.1,0.2,0.4])
    plt.plot(
        pf_vals, 
        Z_list,
        "o", 
        label="Measured",
        linewidth=3
    )
    plt.plot(
        pf, 
        SPT,
        "-", 
        label="SPT",
        linewidth=3
    )
    plt.plot(
        pf, 
        CS,
        "-", 
        label="CS",
        linewidth=3
    )
    plt.plot(
        pf, 
        PY,
        "-", 
        label="PY",
        linewidth=3
    )
    # Show figure
    plt.xlabel("Packing fraction")
    plt.ylabel("Compressibility factor")
    plt.legend()
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


if "one-eos" in sysargs:
    main_equation_of_state()
if "mix-eos" in sysargs:
    packing_list = files.find_all_packing_fractions("data/equillibrium")
    mix_eos()
if "test" in sysargs:
    tests.test_thorne()
    tests.test_rdf()
if "viscosity" in sysargs:
    #packing_list = packing_list[:-2]
    cut_fraction = 0.9
    per_time=False
    if "one" in sysargs:
        filenames = files.get_all_filenames("data/one_component")
        packing_list = files.find_all_packing_fractions("data/one_component")
        main_viscosity(mix=False)
    if "mix" in sysargs:
        filenames = files.get_all_filenames("data/two_component")
        log.info(filenames)
        packing_list = files.find_all_packing_fractions("data/two_component")
        main_viscosity(mix=True)

