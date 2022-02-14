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
    if per_time:
        t, vx = utils.make_time_dependent(vx, t, number_of_chunks)
        t, z = utils.make_time_dependent(z, t, number_of_chunks)

        # Remove early values. They are not useful.
        t = utils.cut_time(cut_fraction, t)
        z = utils.cut_time(cut_fraction, z)
        vx = utils.cut_time(cut_fraction, vx)

        vx_lower, vx_upper, z_lower, z_upper = regression.isolate_slabs(vx, z)
        dv, v_err = regression.regression_for_each_time(
                vx_lower, vx_upper, z_lower, z_upper, t)

        t = t[::step]
        dv = dv[::step]
        v_err = v_err[::step]
        v_err = np.sqrt(np.mean(v_err**2))
        Ptot = utils.cut_time(cut_fraction, Ptot)[::step]
    else:
        # Remove early values. They are not useful.
        N = number_of_chunks
        T = len(t)
        z = np.reshape(z, (T,N))
        vx = np.reshape(z, (T,N))

        t = utils.cut_time(cut_fraction, t)     # These arrays contain the
        z = utils.cut_time(cut_fraction, z)     # same values many times, 
        vx = utils.cut_time(cut_fraction, vx)   # corresponding to 
                                                # different chunks.
        # Remove correlated time steps.
        # This will skip every [step] time steps,
        # to remove time correlation.
        t = t[::step]
        z = z[::step].flatten()
        vx = vx[::step].flatten()

        vx_lower, vx_upper, z_lower, z_upper = regression.isolate_slabs(vx, z)
        dv, v_err = regression.regression_for_single_time(
                vx_lower, vx_upper, z_lower, z_upper)

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
    C, Lz, t, A, Ptot, N_chunks = extract_simulation_variables(log_filename, fix_filename)

    per_time = computation_params["per_time"]
    step = computation_params["step"]
    cut_fraction = computation_params["cut_fraction"]

    # Compute viscosity.
    eta, eta_abs, eta_rel = compute_viscosity(vx, z*2*Lz, t, A, Ptot, N_chunks, cut_fraction, per_time, step=step)
    return eta, C, eta_abs


def extract_simulation_variables(log_filename, fix_filename):
    """
        From a log and a fix file from one simulation,
        extract the variables (time, box dimensions, 
        total transferred momentum, number of chunks)
        that are needed to compute the viscosity.
    """
    # Extract time and momentum transferred from log file.
    variable_list = ["t", "Px"]
    # Two steps are used to minimize initial system 
    # energy. Skip these two. They are NOT skipped 
    # in the corresponding dump/fix files.
    log_table = files.load_system(log_filename, skiprows=2)
    log_vals = files.unpack_variables(
        log_table, 
        log_filename, 
        variable_list
    )

    # Extract all constants from log file.
    constants = convert.extract_constants_from_log(log_filename)
    t0 = constants["EQUILL_TIME"]
    dt = constants["DT"]
    eq_steps = constants["EQUILL_STEPS"]
    sim_steps = constants["RUN_STEPS"]
    N_measure_eq = int(eq_steps/constants["THERMO_OUTPUT"])
    N_measure_sim = int(sim_steps/constants["THERMO_OUTPUT"])

    # Have t contain time values instead of 
    # time steps, and make it start at t=0.
    t = log_vals[variable_list.index("t")][N_measure_eq+4:]
    t = t*dt
    t = t - t0

    # Extract total transferred momentum
    # The +4 is to remove some extra steps
    Ptot = log_vals[variable_list.index("Px")][N_measure_eq+4:] 

    # Get cross-section area.
    Lx = constants["LX"]
    Ly = constants["LY"]
    Lz = constants["LZ"]
    A = theory.get_area(Lx, Ly)

    fix_variable_list = ["t_fix", "Nchunks"]
    fix_table = files.load_system(fix_filename)
    fix_vals = files.unpack_variables(
        fix_table, 
        fix_filename, 
        fix_variable_list
    )
    N_chunks = int(fix_vals[fix_variable_list.index("Nchunks")][0])

    # Check that chunk number is correct.
    tests.assert_chunk_number(N_chunks, constants)
    return constants, Lz, t, A, Ptot, N_chunks 
