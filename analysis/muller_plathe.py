#########################################################
# This file contains two large functions that compute   #
# the viscosity from a Müller-Plathe experiment.        #
#########################################################

import numpy as np
import convert
import theory
import files
import tests
import utils
import regression
import save

import logging
import sys

sysargs = sys.argv
log = logging.getLogger()
log.addHandler(logging.StreamHandler(sys.stdout))
if "debug" in sysargs:
    # logging.Debug is more correct to use, 
    # but info is cleaner.
    log.setLevel(logging.INFO)

def compute_all_viscosities(
        directory, 
        computation_params, 
        theory_functions, 
        theoretical_viscosity
    ):
    """
        Performs a viscosity computation on all simulation
        data in one directory.
    """
    cut_fraction = computation_params["cut_fraction"]
    N = computation_params["particle_types"]
    per_time = computation_params["per_time"]
    path = directory
    filenames = files.get_all_filenames(directory)
    rdf_list = theory_functions
    data = save.create_data_array(filenames, rdf_list, N)

    for (i, f) in enumerate(filenames):
        utils.status_bar(i, len(filenames), fmt="train")
        fix_name = f"{path}/" + f[0]
        log_name = f"{path}/" + f[1]
        log.info(f"Loading file\t{fix_name}")
        log.info(f"Loading file\t{log_name}")

        # Compute and plot viscosity for all packing fractions
        eta, C, error = find_viscosity_from_file(
            log_name, 
            fix_name, 
            computation_params, 
        )

        theoretical_value = [theory.get_viscosity_from_C(C, theoretical_viscosity, rdf) for rdf in rdf_list]
        values = np.array([eta, error])
        values = np.append(values, theoretical_value)
        save.insert_results_in_array(data, values, C, i)
    print("")
    return data


def compute_viscosity(
        vx, z, t, A, Ptot, 
        number_of_chunks, 
        cut_fraction, 
        per_time,
        step,
    ):
    """ Computes the viscosity of a fluid, given arrays of 
        values extracted from a Müller-Plathe experiment.
        Input:
            vx:     np.array of velocity values.
            z:      np.array of positions in z-direction.
            t:      np.array of time values.
            A:      float. Cross section of area through 
                    which the momentum flux passes.
            Ptot:   np.array. The total momentum which has 
                    passed through A at a time index.
        Output:
            eta:        Computed viscosity.
            eta_max:    Estimated maximum value of eta:
                        eta+standard error.
            eta_min:    Estimated minimum value of eta:
                        eta-standard error.
    """
    dv, v_err, t, z, vx = regression.get_velocity_regression(vx, z, t, number_of_chunks, cut_fraction, step, per_time)
    Ptot = utils.cut_time(cut_fraction, Ptot)[::step]
    # Only unique time steps:
    t = np.unique(t)
    assert Ptot.shape == np.unique(t).shape, f"Ptot: {Ptot.shape}, t: {t.shape}"

    eta = np.mean(theory.get_viscosity(Ptot, A, t, dv))
    err_abs = eta*v_err/np.mean(dv)      # This is the absolute error
    err_rel = v_err/np.mean(dv)          # Relative error
    return eta, err_abs, err_rel


def find_viscosity_from_file(
        log_filename, 
        fix_filename, 
        computation_params
    ):
    """ Given a log and fix file from a Müller-Plathe simulation, 
        compute the viscosity of the simulated fluid.
        Inputs:
            log_filename:   string. Name of log file.
            fix_filename:   string. Name of fix file.
        Outputs:
            eta:        Computed viscosity.
            constants:  Constants defined in input script.
            eta_max:    Estimated maximum value of eta:
                        eta+standard error.
            eta_min:    Estimated minimum value of eta:
                        eta-standard error.
    """
    # Extract vx and z from fix file.
    vx, z = regression.get_velocity_profile(fix_filename)
    C, Lz, t, A, Ptot, N_chunks = files.extract_simulation_variables(log_filename, fix_filename)

    per_time = computation_params["per_time"]
    step = computation_params["step"]
    cut_fraction = computation_params["cut_fraction"]

    # Compute viscosity.
    eta, eta_abs, eta_rel = compute_viscosity(vx, z*2*Lz, t, A, Ptot, N_chunks, cut_fraction, per_time, step=step)
    return eta, C, eta_abs
