import numpy as np
import logging
import sys

import files
import theory
import save
import convert
import utils
import block_average

sysargs = sys.argv
log = logging.getLogger()
log.addHandler(logging.StreamHandler(sys.stdout))
if "debug" in sysargs:
    # logging.Debug is more correct to use, 
    # but info is cleaner.
    log.setLevel(logging.INFO)



def compute_all_eoss(
        directory, 
        theory_functions, 
        computation_params
    ):
    path = directory
    filenames = files.get_all_filenames(directory)
    eos_list = theory_functions
    N = computation_params["particle_types"]
    data = save.create_data_array(filenames, eos_list, N)
    for (i, f) in enumerate(filenames):
        utils.status_bar(i, len(filenames), fmt="arrow")
        fix_name = f"{path}/" + f[0]
        log_name = f"{path}/" + f[1]
        log.info(f"Loading file\t{fix_name}")
        log.info(f"Loading file\t{log_name}")

        Z, C, error = get_eos_from_file(log_name, fix_name, computation_params["cut_fraction"])

        theoretical_values = [theory.get_Z_from_C(C, Z) for Z in theory_functions]
        values = np.array([Z, error])
        values = np.append(values, theoretical_values)
        save.insert_results_in_array(data, values, C, i)
    print("")
    return data


def get_eos_from_file(log_name, fix_name, cut_fraction):
    variable_list = ["p", "V", "T"]
    C = convert.extract_constants_from_log(log_name)
    log_table = files.load_system(log_name)
    pvt = files.unpack_variables(log_table, log_name, variable_list)

    eq_steps = int(
            C["EQUILL_STEPS"]
            /C["THERMO_OUTPUT"]
        )
    cut = int(eq_steps*cut_fraction)
    p = np.array(pvt[variable_list.index("p")][cut:eq_steps])
    T = np.array(pvt[variable_list.index("T")][cut:eq_steps])

    N_list = utils.get_component_lists(C, "N")
    sigma_list = utils.get_component_lists(C, "SIGMA")
    x = N_list/np.sum(N_list)
    rho = 6*C["PF"]/np.pi/np.sum(x*np.diag(sigma_list)**3)
    Z = theory.Z_measured_mix(p, rho, T)
    std = block_average.get_block_average(Z)
    Z = np.mean(Z)
    return Z, C, std
