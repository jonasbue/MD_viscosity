#############################################################
# This file contains all functions related to regression    #
# of the velocity profile. This includes functions that     #
# manipulate arrays to fit the analysis scheme.             #
#############################################################

import numpy as np
from scipy import stats
import files
import utils

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
        
        lower_half = utils.remove_nans(lower_half)
        upper_half = utils.remove_nans(upper_half)
        z_lower = utils.remove_nans(z_lower)
        z_upper = utils.remove_nans(z_upper)
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
    dv = utils.get_avg(lower_reg.slope, upper_reg.slope)
    v_err_lower = np.abs(find_uncertainty(lower_reg.stderr, vx_lower))
    v_err_upper = np.abs(find_uncertainty(upper_reg.stderr, vx_upper))
    error = np.sqrt(v_err_lower**2 + v_err_upper**2)/2
    return dv, error


def regression_for_each_time(vx_lower, vx_upper, z_lower, z_upper, t):
    """ Performs a linear regression of the velocity profile,
        using values from one single time step. This approach 
        is slower, but a priori slightly more robust compared 
        to computing the slope from all time values at once.
    """
    dv = np.zeros(len(t))
    std_err = np.zeros(len(t))
    for i in range(len(t)):
        utils.status_bar(i, len(t))
        lower_reg = velocity_profile_regression(vx_lower[i], z_lower[i])
        upper_reg = velocity_profile_regression(vx_upper[i], z_upper[i])
        dv[i] = utils.get_avg(lower_reg.slope, upper_reg.slope)
        std_err[i] = utils.get_avg(lower_reg.stderr, upper_reg.stderr)
    return dv, std_err


def find_uncertainty(std_err, value, conf=95):
    """ Returns the uncertainty in the slope estimation,
        as a confidence interval of percentage conf.
        Input:
            reg:    sc.LinregressInstance. Linear regression object.
            value:  some numerical quantity. Velocity in MP experiment.
            conf:   desired confidence.
        Output:
            err:    error bound of the slope.
    """
    t = stats.t
    tinv = lambda p, df: abs(t.ppf(p/2, df))

    ts = tinv(1-conf/100, len(value))
    err = ts * std_err
    return err
