import numpy as np
import pandas as pd
from scipy import stats
import convert_LAMMPS_output as convert
import files


def radial_distribution(pf):
    """ Returns the thoretical radial distribution 
        function, as given in Pousaneh and de Wijn's paper.
    """
    xi = pf
    return (1-xi/2)/(1-xi)**3


def enskog(pf, sigma, T, m, k=1.0):
    """ Returns the theoretical value of the 
        viscosity for a given packing fraction.
    """
    eta_0 =  5 * np.sqrt((m*k*T)/np.pi) / (16*sigma**2)
    V_excl = 2*np.pi*(sigma**3)/3
    rho = 6*pf/np.pi
    g = radial_distribution(pf)
    eta = eta_0 * (
        1/g 
        + 0.8 * V_excl * rho
        + 0.776 * V_excl**2 * rho**2 * g
    )
    return eta


def viscosity(Ptot, A, t, z, dv_dz):
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
            z:      np.array of positions in z-direction.
            dv_dz:  float. Slope of velocity profile.
        Output:
            eta:        Computed viscosity.
            
    """
    #j = np.divide(Ptot, (2*A*t)) 
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
    return arr[~np.isnan(arr)]


def isolate_slabs(vx, z):
    """ Given vx and z, returns arrays of velocity
        and z coordinates separate for the lower 
        and upper slab in the Müller-Plathe experiment.
    """
    z_max = np.amax(z)
    z_mid = z_max/2
    lower_half = np.where(z<=z_mid, vx, np.nan)
    upper_half = np.where(z>=z_mid, vx, np.nan)
    z_lower = np.where(z<=z_mid, z, np.nan)
    z_upper = np.where(z>=z_mid, z, np.nan)

    z_lower = remove_nans(z_lower)
    z_upper = remove_nans(z_upper)
    lower_half = remove_nans(lower_half)
    upper_half = remove_nans(upper_half)
    #n = len(z)//2
    #z_lower = z[:n]
    #z_upper = z[n:]
    #lower_half = vx[:n]
    #upper_half = vx[n:]

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
    max_slope = reg.slope + reg.stderr
    min_slope = reg.slope - reg.stderr
    return max_slope, min_slope


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


def compute_viscosity(vx, z, t, A, Ptot, number_of_chunks):
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
    ## TODO:    The viscosity is computed for every time step,
    ##          but the slope is not. Change that to make
    ##          dv.shape == t.shape

    vx_lower, vx_upper, z_lower, z_upper = isolate_slabs(vx, z)
    lower_reg = velocity_profile_regression(vx_lower, z_lower)
    upper_reg = velocity_profile_regression(vx_upper, z_upper)
    dv = get_avg(lower_reg.slope, upper_reg.slope)

    dev_low_max, dev_low_min = find_uncertainty(lower_reg)
    dev_upp_max, dev_upp_min = find_uncertainty(upper_reg)
    dev_slope_max = get_avg(dev_low_max, dev_upp_max)
    dev_slope_min = get_avg(dev_low_min, dev_upp_min)

    eta = viscosity(Ptot, A, t, z, dv)
    eta_max = viscosity(Ptot, A, t, z, dev_slope_max)
    eta_min = viscosity(Ptot, A, t, z, dev_slope_min)
    return eta, eta_max, eta_min


def find_viscosity_from_files(log_filename, fix_filename):
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
    log_vals = files.unpack_variables(log_table, log_filename, variable_list)

    # Extract all constants from log file.
    constants = convert.extract_constants_from_log(log_filename)

    # Extract total transferred momentum
    Ptot = log_vals[variable_list.index("Px")]

    # Have t contain time values instead of time steps, 
    # and make it start at t=0.
    t = log_vals[variable_list.index("t")]*constants["DT"]
    t = t - t[0] 

    # Get cross-section area.
    Lx = constants["LX"]
    Ly = constants["LY"]
    A = get_area(Lx, Ly)

    # Check that chunk number is correct.
    assert_chunk_number(fix_filename, constants):

    # Compute viscosity.
    eta, eta_max, eta_min = compute_viscosity(vx, z*2*Lz, t, A, Ptot, N_chunks)
    return eta, constants, eta_max, eta_min


def assert_chunk_number(fix_filename, constants):
    """ Checks that the number of chunks given in
        fix viscosity and fix ave/chunk are the same.
        If they are not the same, computation is assumed 
        to be prone to error, and the program is halted.
    """
    fix_variable_list = ["Nchunks"]
    fix_table = files.load_system(fix_filename)
    N_chunks = files.unpack_variables(fix_table, fix_filename, fix_variable_list)
    N_chunks_given = constants["CHUNK_NUMBER"]
    chunk_thickness = constants["CHUNK_THICKNESS"]
    N_chunks = int(N_chunks[0][0])

    assert(2*constants["LZ"] == chunk_thickness*N_chunks)
    assert(N_chunks_given == N_chunks)
