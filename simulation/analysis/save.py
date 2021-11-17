#############################################################
# This file contains code which takes LAMMPS output         #
# and saves the results of the viscosity measurement        #
# as a csv file, so the results can be used by pgfplots.    #
#############################################################
import numpy as np

def save_simulation_data(filename, data, data_name="viscosity", fmt="%.3e"):
    """ Appends resuls of a simulation, and the
        configuration of the system, to a given
        .csv file.
        The file saves data as follows:
            pf  N1  N2  m1  m2  sigma1  sigma2  viscosity   error
            0.1 500 500 1.0 1.0 1.0     2.0     1.2         0.1
            0.2 500 500 1.0 1.0 1.0     2.0     1.5         0.12
            0.3 500 500 1.0 1.0 1.0     2.0     2.5         0.15
        Any combination of data can be extracted
        from such a file using the function 
        thiscolumn from pgfplots, but it will be
        much simpler to handle the data if the
        file contains only one varying quantity.
        This quantity should be specified in the 
        name of the data file.
    """
    header = f"pf, N1, N2, m1, m2, sigma1, sigma2, {data_name}"
    np.savetxt(filename, data, header=header, fmt=fmt, delimiter=", ", comments="")


def insert_results_in_array(data, value, C, i, err=None, pf=None):
    parameters = get_system_config(C, pf)
    if np.isscalar(value):
        vals = np.append(parameters, value)
        vals = np.append(vals, err)
        data[i] = vals
    else:
        l = len(parameters)
        data[i,:l] = parameters
        data[i,l:l+len(value)] = value
    return data

def get_system_config(C=None, pf=None):
    if C != None:
        N1 = C["N_L"]
        N2 = C["N_H"]
        m1 = C["MASS_L"]
        m2 = C["MASS_H"]
        sigma1 = C["SIGMA_L"]
        sigma2 = C["SIGMA_H"]
        if pf == None:
            pf = C["PF"]
        parameters = np.array([pf, N1, N2, m1, m2, sigma1, sigma2])
    else:
        parameters = np.zeros(7)
    return parameters
