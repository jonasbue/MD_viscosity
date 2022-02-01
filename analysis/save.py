#############################################################
# This file contains code which takes LAMMPS output         #
# and saves the results of the viscosity measurement        #
# as a csv file, so the results can be used by pgfplots.    #
#############################################################
import numpy as np

import eos
import viscosity

def get_data_name(theory_functions, viscosity_function=None):
    """
        Given a list of theoretical functions, and
        optionally a viscosity_function, this returns a
        string with the unique name of the theoretical
        functions, to be used as a file header.
    """
    #if viscosity_function != None:
    #    viscosity_function_names = [f.__name__ for f in viscosity_function_names]

    # TODO: Start here.
    function_names = {
        eos.Z_SPT   : "EOS_SPT",
        eos.Z_PY    : "EOS_PY",
        eos.Z_BMCSL : "EOS_BMCSL",
        eos.Z_CS    : "EOS_CS",
        eos.rdf_SPT   : "RDF_SPT",
        eos.rdf_PY    : "RDF_PY",
        eos.rdf_BMCSL : "RDF_BMCSL",
        eos.rdf_CS    : "RDF_CS",
        viscosity.enskog    : "enskog_",
        viscosity.thorne    : "thorne_",
        None                : "",
    }
    data_name = "".join(
        [f",\t{function_names[viscosity_function]}{function_names[t]}" for t in theory_functions])
    return data_name


def add_column_to_file(filename, new_column_data, new_column_name, fmt="%.3e"):
    """ Takes a (csv) file and a 1D np.array, and appends 
        the contents of the array as a new column to the file.
    """
    # TODO: Test this function
    df = pd.read_csv(filename)
    df[new_column_name] = new_column_data
    # TODO: Add some assertions
    df.to_csv(filename, float_format=fmt)

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
        if err:
            vals = np.append(vals, err)
        data[i] = vals
    else:
        l = len(parameters)
        data[i,:l] = parameters
        data[i,l:l+len(value)] = value
    return data

def get_system_config(C=None, pf=None):
    if C != None:
        number_of_components = C["ATOM_TYPES"]
        if number_of_components == 1:
            N = C["N"]
            m = C["MASS"]
            sigma = C["SIGMA"]
            if pf == None:
                pf = C["PF"]
            parameters = np.array([pf, N, m, sigma])
        elif number_of_components == 2:
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


def create_data_array(filenames, theory_functions):
    rows = len(filenames)
    columns = (
        len(get_system_config()) 
        + 2 + len(theory_functions)
    )
    data = np.zeros((rows,columns))
    return data

