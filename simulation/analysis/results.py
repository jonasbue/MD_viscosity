#############################################################
# This script converts a number of LAMMPS files to .csv     #
# files containing viscosity and equation of state data.    #
# Files are ready to be plotted in report with pgfplots.    #
#############################################################

import numpy as np
import logging
import sys

import files
import viscosity
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


## Rename to convert_something_something
def main():
    N = 2
    cut_fraction = 0.9
    data_path_list = ["data/varying_mass", "data/varying_sigma", "data/varying_fraction"]
    save_path_list = ["varying_mass", "varying_sigma", "varying_fraction"]

    for path, savename in zip(data_path_list, save_path_list):
        #files.all_files_to_csv(path)
        save_dir = "../report/data/"
        filenames = files.get_all_filenames(path)
        packing_list = files.find_all_packing_fractions(path)
        filenames = files.sort_files(filenames, packing_list)

        save_viscosity(cut_fraction, path, filenames, savename=f"{save_dir}visc_{savename}.csv")
        save_viscosity(cut_fraction, path, filenames, savename=f"{save_dir}visc_norm_{savename}.csv", normalize=True)

        #save_eos(path, filenames, cut_fraction, N, savename=f"{save_dir}eos_{savename}.csv")
        save_theory(path, filenames, savename=f"{save_dir}theory_{savename}.csv")


def save_viscosity(cut_fraction, path, filenames, savename, per_time=False, normalize=False):
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
        if normalize:
            thorne_value = viscosity.get_thorne_from_C(C)
            save.insert_results_in_array(data, np.mean(eta)/thorne_value, np.mean(eta_err)/thorne_value, C, i)
        else:
            save.insert_results_in_array(data, np.mean(eta), np.mean(eta_err), C, i)
    print("")
    save.save_simulation_data(savename, data)


def save_eos(path, filenames, cut_fraction, number_of_components, savename, normalize=False):
    data = np.zeros((len(filenames), 9))
    for (i, f) in enumerate(filenames):
        fix_name = f"{path}/" + f[0]
        log_name = f"{path}/" + f[1]
        log.info(f"Loading file\t{fix_name}")
        log.info(f"Loading file\t{log_name}")

        variable_list = ["p", "V", "T"]
        C = convert.extract_C_from_log(log_name)
        log_table = files.load_system(log_name)
        pvt = files.unpack_variables(log_table, log_name, variable_list)

        eq_steps = int(
                C["EQUILL_TIME"]
                /C["DT"]
                /C["THERMO_OUTPUT"]
            )
        cut = int(eq_steps*cut_fraction)
        p = np.mean(pvt[variable_list.index("p")][cut:eq_steps])
        T = np.mean(pvt[variable_list.index("T")][cut:eq_steps])

        N_list = np.array([C["N_L"], C["N_H"]])
        sigma_list = np.array([C["SIGMA_L"], C["SIGMA_H"]])
        x = N_list/np.sum(N_list)
        rho = 6*C["PF"]/np.pi/np.sum(x*np.diag(sigma_list)**3)
        Z = eos.Z_measured_mix(p, rho, T)

        save.insert_results_in_array(data, Z, 0, constants, i)
    save.save_simulation_data(savename, data, data_name="Z")


def save_theory(path, filenames, savename, N=50):
    print(len(filenames))
    pf_experiment = files.find_all_packing_fractions(path)
    print(len(pf_experiment))
    data_shape = (int(len(filenames)/len(pf_experiment))*N, 13)
    data = np.zeros(data_shape)
    pf = np.linspace(0,0.5,N)
    included_masses = np.empty(0)
    included_sigmas = np.empty(0)
    included_numbers = np.empty(0)
    for (i, f) in enumerate(filenames):
        fix_name = f"{path}/" + f[0]
        log_name = f"{path}/" + f[1]
        log.info(f"Loading file\t{fix_name}")
        log.info(f"Loading file\t{log_name}")

        C = convert.extract_constants_from_log(log_name)

        T = C["TEMP"]
        N_list = np.array([C["N_L"], C["N_H"]])
        sigma_list = np.array([C["SIGMA_L"], C["SIGMA_H"]])
        mass_list = np.array([C["MASS_L"], C["MASS_H"]])
        x = N_list/np.sum(N_list)

        # Check if the configuration has already been included,
        # since filenames contains multiple packing fractions.
        if (
            N_list[1] not in included_numbers
            or mass_list[1] not in included_masses
            or sigma_list[1] not in included_sigmas
        ):
            included_numbers = np.append(included_numbers, N_list[1])
            included_masses = np.append(included_masses, mass_list[1])
            included_sigmas = np.append(included_sigmas, sigma_list[1])
            print("I'm", i)

            thorne_vals = np.zeros(50)
            SPT_vals = np.zeros(50)
            PY_vals = np.zeros(50)
            for j in range(N):
                thorne_vals[j] = viscosity.thorne(
                    pf[j], x, mass_list, sigma_list, T)
                rho = 6*pf[j]/np.pi/np.sum(x*np.diag(sigma_list)**3)
                sigma = viscosity.get_sigma(sigma_list)
                SPT_vals[j] = eos.Z_SPT(sigma, x, rho)
                PY_vals[j] = eos.Z_PY(sigma, x, rho)
                # rdf_SPT = ...
                # rdf_PY = ...
                vals = np.array([thorne_vals[j], SPT_vals[j], PY_vals[j]])

                # TODO: Find out how to estimate 
                errors = np.array([30, -10, 20])
                save.insert_results_in_array(data, vals, errors, C, i*N+j, pf=pf[j])

    save.save_simulation_data(savename, data, data_name="viscosity, SPT_EoS, PY_EoS", error_name="viscosity_error, SPT_error, PY_error")

main()
