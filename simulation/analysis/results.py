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
        files.all_files_to_csv(path)
        save_dir = "../report/data/"
        filenames = files.get_all_filenames(path)
        packing_list = files.find_all_packing_fractions(path)
        filenames = files.sort_files(filenames, packing_list)

        save_viscosity(cut_fraction, path, filenames, savename=f"{save_dir}visc_{savename}.csv")
        save_eos(path, filenames, cut_fraction, N, savename=f"{save_dir}eos_{savename}.csv")


def save_viscosity(cut_fraction, path, filenames, savename, per_time=False):
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
    save.save_simulation_data(savename, data)


def save_eos(path, filenames, cut_fraction, number_of_components, savename):
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
        Z = eos.Z_measured_mix(p, rho, T)

        save.insert_results_in_array(data, Z, 0, constants, i)
    save.save_simulation_data(savename, data, data_name="Z")

main()
