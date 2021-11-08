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
import save
import utils
import convert_LAMMPS_output as convert

sysargs = sys.argv
log = logging.getLogger()
log.addHandler(logging.StreamHandler(sys.stdout))
if "debug" in sysargs:
    # logging.Debug is more correct to use, 
    # but info is cleaner.
    log.setLevel(logging.INFO)


def save_viscosity(
        cut_fraction, 
        path,
        filenames,
        savename
    ):
    C = {}
    PF_list = np.zeros(len(filenames))
    eta_list = np.zeros(len(filenames))
    error_list = np.zeros(len(filenames))
    data = np.zeros((len(filenames),9))

    for (i, f) in enumerate(filenames):
        utils.status_bar(i, len(filenames), fmt="train")
        fix_name = f"{path}/" + f[0]
        log_name = f"{path}/" + f[1]
        log.info(f"Loading file\t{fix_name}")
        log.info(f"Loading file\t{log_name}")

        # Compute and plot viscosity for all packing fractions
        eta, C, eta_err = muller_plathe.find_viscosity_from_file(
            log_name, fix_name, cut_fraction, per_time
        )
        save.insert_results_in_array(data, np.mean(eta), np.mean(eta_err), C, i)
    save.save_simulation_data("savetest.csv", data)

def save_eos(path, filenames, cut_fraction, number_of_components):
    packing_list = np.zeros(len(filenames))
    PF_list = np.zeros(len(filenames))
    Z_list = np.zeros(len(filenames))
    V_list = np.zeros(len(filenames))
    sigma_list = np.zeros(number_of_components)
    N_list = np.zeros(number_of_components)
    C = {}
    data = np.zeros((len(filenames), 9))

    for (i, f) in enumerate(filenames):
        fix_name = f"{path}/" + f[0]
        log_name = f"{path}/" + f[1]
        log.info(f"Loading file\t{fix_name}")
        log.info(f"Loading file\t{log_name}")

        variable_list = ["p", "V", "T"]
        
        constants = convert.extract_constants_from_log(log_name)
        log_table = files.load_system(log_name)
        pvt = files.unpack_variables(log_table, log_name, variable_list)

        eq_steps = int(
                constants["EQUILL_TIME"]
                /constants["DT"]
                /constants["THERMO_OUTPUT"]
            )
        cut = int(eq_steps*cut_fraction)
        p = np.mean(pvt[variable_list.index("p")][cut:eq_steps])
        T = np.mean(pvt[variable_list.index("T")][cut:eq_steps])

        N_list = np.array([constants["N_L"], constants["N_H"]])
        sigma_list = np.array([constants["SIGMA_L"], constants["SIGMA_H"]])
        x = N_list/np.sum(N_list)
        rho = 6*constants["PF"]/np.pi/np.sum(x*np.diag(sigma_list)**3)
        Z = (
                eos.Z_measured_mix(p, rho, T)
            )
        save.insert_results_in_array(data, Z, 0, constants, i)
    save.save_simulation_data("../report/data/eos.csv", data, data_name="Z")


def plot_viscosity(
        cut_fraction, 
        path,
        filenames,
        per_time=False,
        plot_vs_time=False
    ):
    C = {}
    PF_list = np.zeros(len(filenames))
    eta_list = np.zeros(len(filenames))
    error_list = np.zeros(len(filenames))
    data = np.zeros((len(filenames),9))

    for (i, f) in enumerate(filenames):
        utils.status_bar(i, len(filenames), fmt="train")
        fix_name = f"{path}/" + f[0]
        log_name = f"{path}/" + f[1]
        log.info(f"Loading file\t{fix_name}")
        log.info(f"Loading file\t{log_name}")

        # Compute and plot viscosity for all packing fractions
        eta, C, eta_err = muller_plathe.find_viscosity_from_file(
            log_name, fix_name, cut_fraction, per_time
        )

        PF_list[i] = C["PF"]
        eta_list[i] = np.mean(eta)
        error_list[i] = np.mean(eta_err)

        # Note: Cut fraction must be low for this plot to be useful
        if "plot-profiles" in sysargs:
            # Plot velocity profiles and regressions of them
            plotting.plot_velocity_profile_from_file(fix_name)
        if plot_vs_time:
            t = np.linspace(C["RUN_TIME"]*cut_fraction, C["RUN_TIME"], len(eta))
            plt.plot(t, eta, label="Viscosity")
            markers, caps, bars = plt.errorbar(
                t[::5],
                eta[::5],
                yerr = eta_err[::5],
                fmt="yo",
                ecolor="b"
            )
            plt.title(f"Measured viscosity vs time, packing={C['PF']}")
            plt.legend()
            plt.show()
    if mix:
        plotting.plot_result_vs_thorne(eta_list, PF_list, error_list, C, relative=True)
    else:
        plotting.plot_result_vs_enskog(eta_list, PF_list, error_list, C)


def plot_eos(path, filenames, cut_fraction, number_of_components):
    packing_list = np.zeros(len(filenames))
    PF_list = np.zeros(len(filenames))
    Z_list = np.zeros(len(filenames))
    V_list = np.zeros(len(filenames))
    sigma_list = np.zeros(number_of_components)
    N_list = np.zeros(number_of_components)
    C = {}
    data = np.zeros((len(filenames), 9))

    for (i, f) in enumerate(filenames):
        fix_name = f"{path}/" + f[0]
        log_name = f"{path}/" + f[1]
        log.info(f"Loading file\t{fix_name}")
        log.info(f"Loading file\t{log_name}")

        variable_list = ["p", "V", "T"]
        
        constants = convert.extract_constants_from_log(log_name)
        log_table = files.load_system(log_name)
        pvt = files.unpack_variables(log_table, log_name, variable_list)

        eq_steps = int(
                constants["EQUILL_TIME"]
                /constants["DT"]
                /constants["THERMO_OUTPUT"]
            )
        cut = int(eq_steps*cut_fraction)
        #print(cut, eqsteps)
        p = np.mean(pvt[variable_list.index("p")][cut:eq_steps])
        T = np.mean(pvt[variable_list.index("T")][cut:eq_steps])
        V = 8*constants["LX"]*constants["LY"]*constants["LZ"]
        PF_list[i] = constants["PF"]
        V_list[i] = V

        if number_of_components == 2:
            N_list = np.array([constants["N_L"], constants["N_H"]])
            sigma_list = np.array([constants["SIGMA_L"], constants["SIGMA_H"]])
            x = N_list/np.sum(N_list)
            rho = 6*PF_list[i]/np.pi/np.sum(x*np.diag(sigma_list)**3)
            Z_list[i] = (
                    eos.Z_measured_mix(p, rho, T)
                )
            save.insert_results_in_array(data, Z_list[i], 0, constants, i)
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
        save.save_simulation_data("../report/data/eos.csv", data, data_name="Z")
            
    # Show figure
    plt.xlabel("Packing fraction")
    plt.ylabel("Compressibility factor")
    plt.legend()
    plt.show()


path = None
per_time=False
savedata=False
mix=True
N = 1
cut_fraction = 0.9
if "mix" in sysargs:
    path = "new_data/varying_mass"
    N = 2
if "equal-mix" in sysargs:
    path = "data/equal_two_component"
    N = 2
if "one" in sysargs:
    path = "data/one_component"
    mix = False
if "per-time" in sysargs:
    per_time=True
if "nosave" in sysargs:
    savedata=False

# Convert all files in data to csv format.
if "convert" in sysargs:
    files.all_files_to_csv(path)

filenames = files.get_all_filenames(path)
packing_list = files.find_all_packing_fractions(path)
filenames = files.sort_files(filenames, packing_list)

if "test" in sysargs:
    #tests.test_eos()
    tests.test_thorne()
    tests.test_rdf()
if "eos" in sysargs:
    plot_eos(path, filenames, cut_fraction, N)
if "viscosity" in sysargs:
    plot_viscosity(cut_fraction, path, filenames)
if "plot-vs-time" in sysargs:
    cut_fraction = 0.01
    plot_viscosity(cut_fraction, path, filenames, plot=True, plot_vs_time=True)
