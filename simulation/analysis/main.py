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
import convert_LAMMPS_output as convert

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


def main_eos(path, number_of_components, packing_list):
    PF_list = np.zeros(len(packing_list))
    Z_list = np.zeros(len(packing_list))
    sigma_list = np.zeros(number_of_components)
    N_list = np.zeros(number_of_components)
    V_list = np.zeros(len(packing_list))
    C = {}

    for (i, packing) in enumerate(packing_list):
        if number_of_components > 1:
            fix_name = f"{path}/fix.viscosity_mix_eta_{packing}.lammps"
            log_name = f"{path}/log.mix_eta_{packing}.lammps"
        else:
            fix_name = f"{path}/fix.viscosity_eta_{packing}.lammps"
            log_name = f"{path}/log.eta_{packing}.lammps"
        variable_list = ["p", "V", "T"]
        
        constants = convert.extract_constants_from_log(log_name)
        log_table = files.load_system(log_name)
        pvt = files.unpack_variables(log_table, log_name, variable_list)

        p = np.mean(pvt[variable_list.index("p")])
        V = constants["LX"]*constants["LY"]*constants["LZ"]
        T = np.mean(pvt[variable_list.index("T")])
        PF_list[i] = constants["PF"]
        V_list[i] = V

        if number_of_components == 2:
            N_list = np.array([constants["N_L"], constants["N_H"]])
            sigma_list = np.array([constants["SIGMA_L"], constants["SIGMA_H"]])
            Z_list[i] = (
                    eos.Z_measured(p, V, N_list[0], T)
                    + eos.Z_measured(p, V, N_list[1], T)
                )
        else:
            N_list = constants["N"]
            sigma_list = constants["SIGMA"]
            Z_list[i] = eos.Z_measured(p, V, N_list, T)

    # Plot measured packing fraction
    plt.plot(PF_list, Z_list, "o", label="Measured Z")

    # Pot carnahan Starling
    pf = np.linspace(0, 0.5)
    CS = eos.Z_CS(pf)
    plt.plot(pf, CS, "-", label="CS (one component)", linewidth=3)

    # Plot theoretical values, from CS-EoS
    if number_of_components > 1:
        #rho_list = np.sum(N_list)/V_list
        #print(rho_list)
        x = N_list/np.sum(N_list)
        SPT = np.zeros_like(pf)
        PY = np.zeros_like(pf)
        sigma = viscosity.get_sigma(sigma_list)
        # TODO: Write a rho/pf conversion function.
        rho_list = 6*pf/np.pi/np.sum(x*np.diag(sigma)**3)
        for i, rho in enumerate(rho_list):
            SPT[i] = eos.Z_SPT(sigma, x, rho)
            PY[i] = eos.Z_PY(sigma, x, rho)
        plt.plot(pf, SPT, "-", label="SPT", linewidth=3)
        plt.plot(pf, PY, "-", label="PY", linewidth=3)
            
    # Show figure
    plt.xlabel("Packing fraction")
    plt.ylabel("Compressibility factor")
    plt.legend()
    plt.show()


path = None
mix = False
N = 1
if "mix" in sysargs:
    path = "data/two_component"
    mix = True
    N = 2
if "eos" in sysargs:
    path = "data/equillibrium"
if "one" in sysargs:
    path = "data/one_component"

filenames = files.get_all_filenames(path)
packing_list = files.find_all_packing_fractions(path)

if "eos" in sysargs:
    packing_list = files.find_all_packing_fractions(path)
    main_eos(path, N, packing_list)
if "test" in sysargs:
    tests.test_eos()
    #tests.test_thorne()
    #tests.test_rdf()
if "viscosity" in sysargs:
    cut_fraction = 0.9
    per_time=False
    main_viscosity(mix)
