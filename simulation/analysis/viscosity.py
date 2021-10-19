import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import convert_LAMMPS_output as convert
import files
import tests
import eos


def enskog(pf, sigma, T, m, k=1.0):
    """ Returns the theoretical value of the 
        viscosity for a given packing fraction.
    """
    eta_0 =  5 * np.sqrt((m*k*T)/np.pi) / (16*sigma**2)
    V_excl = 2*np.pi*(sigma**3)/3
    rho = 6*pf/np.pi
    g = eos.rdf_PY(pf)
    eta = eta_0 * (
        1/g 
        + 0.8 * V_excl * rho
        + 0.776 * V_excl**2 * rho**2 * g
    )
    return eta


def viscosity(Ptot, A, t, dv_dz):
    """ Computes the measured value of the viscosity,
        from the Müller-Plathe experiment.
        Dangerous behaviour: 
            The first values of the array will be undefined,
            due to for example division by zero.
            This is not a problem since these values 
            are not used in the actual computation of 
            the viscosity, but a check would be wise.
        Input:
            Ptot:   np.array. The total momentum which has 
                    passed through A at a time index.
            A:      float. Cross section of area through 
                    which the momentum flux passes.
            t:      np.array of time values.
            dv_dz:  float. Slope of velocity profile.
        Output:
            eta:        Computed viscosity.
            
    """
    #print("In visc:\t", Ptot.shape, t.shape, dv_dz.shape)
    j = Ptot/(2*A*t)
    eta = -j/dv_dz
    return eta


def get_velocity_profile(fix_filename):
    """ From a LAMMPS fix file, extracts the x-velocities and z-coordinates
        and returns them as separate np.arrays.
        Input:
            fix_filename:   string. Name of a LAMMPS fixfile.
        Output:
            vx: np.array of velocity components for different time steps.
            z:  np.array of z-coordinates in the same time steps.
    """
    log_table = files.load_system(fix_filename)
    zvx = files.unpack_variables(
        log_table, fix_filename, ["z", "vx"]
    )
    z = zvx[0]
    vx = zvx[1]
    return vx, z


def remove_nans(arr):
    """ Removes all instances of np.nan in arr. """
    return arr[~np.isnan(arr)]


def isolate_slabs(vx, z):
    """ Given vx and z, returns arrays of velocity
        and z coordinates separate for the lower 
        and upper slab in the Müller-Plathe experiment.
    """
    # Check the dimension of vx. 
    # If it is 2D, then it contains 
    # only one velocity value per time.
    if vx.ndim == 2:
        # If all vx and z correspond to one single 
        # time, we can use a simple solution.
        n = len(z[0,:])//2
        z_lower = z[:,:n]
        z_upper = z[:,n:]
        lower_half = vx[:,:n]
        upper_half = vx[:,n:]
    elif vx.ndim == 1:
        # If vx and z contain values from multiple times,
        # we need to read the arrays.
        # This method does not work for 1D-arrays.
        z_max = np.nanmax(z)
        z_mid = z_max/2

        lower_half = np.where(z<=z_mid, vx, np.nan).reshape(vx.shape)
        upper_half = np.where(z>=z_mid, vx, np.nan).reshape(vx.shape)
        z_lower = np.where(z<=z_mid, z, np.nan).reshape(z.shape)
        z_upper = np.where(z>=z_mid, z, np.nan).reshape(z.shape)
        
        lower_half = remove_nans(lower_half)
        upper_half = remove_nans(upper_half)
        z_lower = remove_nans(z_lower)
        z_upper = remove_nans(z_upper)
    return lower_half, upper_half, z_lower, z_upper


def velocity_profile_regression(vx, z):
    """ Performs a linear regression on vx and z.
        Separate regressions are performed on the 
        lower half (first values) and the upper 
        half (last values) of the slabs from 
        the Müller-Plathe experiment.
        Input:
            vx: np.array of velocity values.
            z:  np.array of positions in z-direction.
        Output:
            lower_reg:  Linear regression from the lower half of the slab.
            upper_reg:  Linear regression from the upper half of the slab.
            z_lower:    z-coordinates from the lower half of the slab.
            z_upper:    z-coordinates from the upper half of the slab.
    """
    reg = stats.linregress(
        z, 
        vx, 
    )
    return reg


def regression_for_single_time(vx_lower, vx_upper, z_lower, z_upper):
    """ Performs a linear regression of the velocity profile,
        using values from all times. This approach is faster, 
        but slightly rough compared to computing a different
        slope for all times, and then obtaining a mean.
    """
    lower_reg = velocity_profile_regression(vx_lower, z_lower)
    upper_reg = velocity_profile_regression(vx_upper, z_upper)
    dv = get_avg(lower_reg.slope, upper_reg.slope)
    std_err = get_avg(lower_reg.stderr, upper_reg.slope)
    return dv, std_err


def regression_for_each_time(vx_lower, vx_upper, z_lower, z_upper, t):
    """ Performs a linear regression of the velocity profile,
        using values from one single time step. This approach 
        is slower, but a priori slightly more robust compared 
        to computing the slope from all time values at once.
    """
    dv = np.zeros(len(t))
    std_err = np.zeros(len(t))
    for i in range(len(t)):
        print(f"\r{100*i/len(t):.0f} %", end="")
        lower_reg = velocity_profile_regression(vx_lower[i], z_lower[i])
        upper_reg = velocity_profile_regression(vx_upper[i], z_upper[i])
        dv[i] = get_avg(lower_reg.slope, upper_reg.slope)
        std_err[i] = get_avg(lower_reg.stderr, upper_reg.stderr)
    print("")
    return dv, std_err

def get_avg(lower, upper):
    """ Returns an average (typically of slopes).
        Inputs:
            lower:  float.
            upper:  float.
        Outputs:
            avg:    float. The average of lower and upper.
    """
    return (np.abs(lower) + np.abs(upper))/2


def find_uncertainty(reg):
    """ Returns the uncertainty in the slope estimation,
        in the form of the maximum and minimum slope which
        the standard error from sc.linregress() gives.
        Input:
            reg:    sc.LinregressInstance. Linear regression object.
        Output:
            max_slope:  maximum slope based on linear regression.
            min_slope:  minimum slope based on linear regression.
    """
    return reg.stderr, -reg.stderr


def get_area(Lx, Ly):
    """ Return the cross-section of the box, 
        through which the momentum flux Jz(vx) passes.
        Input:
            Lx: float. Length of box in x-direction.
            Ly: float. Length of box in y-direction.
        Output:
            A:  float. Area.
    """
    return 4*Lx*Ly

def make_time_dependent(arr, t, number_of_chunks):
    """ Takes an array with multiple values for each timestep,
        and converts it to a 2D array in which first dimension
        is the time and second dimension contains the values
        at that time.
    """
    t = np.unique(t)

    # Ad hoc: With current data, two values are missing from first timestep.
    arr = np.insert(arr, 0, np.nan)
    arr = np.insert(arr, 0, np.nan)
    arr = np.reshape(arr, (len(t), number_of_chunks))
    return t, arr


def cut_time(cut_fraction, arr):
    """ Removes the first values of arr,
        to remove early values from the analysis.
    """
    cut = int(cut_fraction*len(arr))
    print(f"Cutting the first {cut} values.")
    arr = arr[cut:]
    print(arr.shape)
    return arr


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
        t, vx = make_time_dependent(vx, t, number_of_chunks)
        t, z = make_time_dependent(z, t, number_of_chunks)

        # Remove early values. They are not useful.
        t = cut_time(cut_fraction, t)
        z = cut_time(cut_fraction, z)
        vx = cut_time(cut_fraction, vx)

        vx_lower, vx_upper, z_lower, z_upper = isolate_slabs(vx, z)
        dv, std_err = regression_for_each_time(vx_lower, vx_upper, z_lower, z_upper, t)
    else:
        # Remove early values. They are not useful.
        t = cut_time(cut_fraction, t)
        z = cut_time(cut_fraction, z)
        vx = cut_time(cut_fraction, vx)
        vx_lower, vx_upper, z_lower, z_upper = isolate_slabs(vx, z)
        dv, std_err = regression_for_single_time(
                vx_lower, vx_upper, z_lower, z_upper)

    Ptot = Ptot[-1]

    eta = viscosity(Ptot, A, t, dv)

    # eta_max = - eta_min, so only one value is needed.
    # For generality, both are computed here.
    eta_max = viscosity(Ptot, A, t, dv**2)*std_err
    eta_min = viscosity(Ptot, A, t, dv**2)*(-std_err)
    return eta, eta_max, eta_min


def find_viscosity_from_files(
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
    vx, z = get_velocity_profile(fix_filename)


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

    # Extract total transferred momentum
    Ptot = log_vals[variable_list.index("Px")]

    # Get cross-section area.
    Lx = constants["LX"]
    Ly = constants["LY"]
    Lz = constants["LZ"]
    A = get_area(Lx, Ly)

    fix_variable_list = ["t_fix", "Nchunks"]
    fix_table = files.load_system(fix_filename)
    fix_vals = files.unpack_variables(
        fix_table, 
        fix_filename, 
        fix_variable_list
    )
    N_chunks = int(fix_vals[fix_variable_list.index("Nchunks")][0])

    # Have t contain time values instead of 
    # time steps, and make it start at t=0.
    t = fix_vals[fix_variable_list.index("t_fix")]*constants["DT"]
    t = t - constants["EQUILL_TIME"]

    # Check that chunk number is correct.
    tests.assert_chunk_number(N_chunks, constants)

    # Compute viscosity.
    # Should z be multiplied by 2*Lz to get the correct height?
    eta, eta_max, eta_min = compute_viscosity(vx, z*2*Lz, t, A, Ptot, N_chunks, cut_fraction, per_time)
    return eta, constants, eta_max
