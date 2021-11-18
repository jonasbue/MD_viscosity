#########################################################
# This file contains two large functions that compute   #
# the viscosity from a Müller-Plathe experiment.        #
#########################################################

import numpy as np
import convert_LAMMPS_output as convert
import viscosity
import files
import tests
import utils
import regression

def compute_viscosity(
        vx, z, t, A, Ptot, 
        number_of_chunks, 
        cut_fraction, 
        per_time
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
        dv, std_err = regression.regression_for_each_time(
                vx_lower, vx_upper, z_lower, z_upper, t)
    else:
        # Remove early values. They are not useful.
        t = utils.cut_time(cut_fraction, t)     # These arrays contain the
        z = utils.cut_time(cut_fraction, z)     # same values many times, 
        vx = utils.cut_time(cut_fraction, vx)   # corresponding to 
                                                # different chunks.

        vx_lower, vx_upper, z_lower, z_upper = regression.isolate_slabs(vx, z)
        dv, std_err = regression.regression_for_single_time(
                vx_lower, vx_upper, z_lower, z_upper)

    Ptot = utils.cut_time(cut_fraction, Ptot)
    t = np.unique(t)
    assert Ptot.shape == np.unique(t).shape, f"Ptot: {Ptot.shape}, t: {t.shape}"
    # Only unique times:
    eta = viscosity.get_viscosity(Ptot, A, t, dv)

    # eta_max = - eta_min, so only one value is needed.
    # For generality, both are computed here.
    eta_max = -eta*std_err
    eta_min = eta*std_err
    return eta, eta_max, eta_min


def find_viscosity_from_file(
        log_filename, 
        fix_filename, 
        cut_fraction, 
        per_time=True
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


    # Extract time and momentum transferred from log file.
    variable_list = ["t", "Px"]
    log_table = files.load_system(log_filename)
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
    A = viscosity.get_area(Lx, Ly)

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

    # Compute viscosity.
    eta, eta_max, eta_min = compute_viscosity(vx, z*2*Lz, t, A, Ptot, N_chunks, cut_fraction, per_time)
    return eta, constants, eta_max
