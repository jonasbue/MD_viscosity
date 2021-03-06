#############################################################
# This file contains code which takes LAMMPS output         #
# and saves the results of the viscosity measurement        #
# as a csv file, so the results can be used by pgfplots.    #
#############################################################
import numpy as np
import pandas as pd

import theory
import curve_fit

def get_data_name(theory_functions, viscosity_function=None):
    """
        Given a list of theoretical functions, and
        optionally a viscosity_function, this returns a
        string with the unique name of the theoretical
        functions, to be used as a file header.
    """
    function_names = {
        theory.Z_SPT        : "EOS_SPT",
        theory.Z_PY         : "EOS_PY",
        theory.Z_BMCSL      : "EOS_BMCSL",
        theory.Z_CS         : "EOS_CS",
        theory.Z_kolafa     : "EOS_kolafa",
        theory.Z_gottschalk : "EOS_gottschalk",
        theory.Z_mecke      : "EOS_mecke",
        theory.Z_thol       : "EOS_thol",
        theory.Z_hess       : "EOS_hess",
        theory.rdf_SPT      : "RDF_SPT",
        theory.rdf_PY       : "RDF_PY",
        theory.rdf_BMCSL    : "RDF_BMCSL",
        theory.rdf_CS       : "RDF_CS",
        theory.rdf_LJ       : "RDF_LJ",
        theory.enskog       : "enskog_",
        theory.thorne       : "thorne_",
        theory.F_CS         : "F-CS",
        theory.F_kolafa     : "F-kolafa",
        theory.F_thol       : "F-thol",
        theory.F_gottschalk : "F-gottschalk",
        theory.F_mecke      : "F-mecke",
        theory.F_hess       : "F-hess",
        theory.get_viscosity_from_F : "enskog_",
        theory.get_rdf_from_F       : "", # This causes a bit of trouble if named.
        theory.get_Z_from_F         : "EOS_",
        theory.get_internal_energy  : "U_",
        curve_fit.fit_viscosity     : "enskog_fit_",
    }
    if viscosity_function:
        data_name = "".join([f", {function_names[viscosity_function]}{function_names[t]}" for t in theory_functions])
    else:
        data_name = "".join(
            [f", {function_names[t]}" for t in theory_functions])
    return data_name


def add_column_to_file(filename, new_column_data, new_column_name, fmt="%.3e"):
    """ Takes a (csv) file and a 1D np.array, and appends 
        the contents of the array as a new column to the file.
    """
    df = pd.read_csv(filename)
    if new_column_name in df:
        # Overwrite
        df[new_column_name] = new_column_data
    else:
        df.insert(len(df.columns), new_column_name, new_column_data)
    # Consider np savetxt, for the same strucure as in all other files.
    df.to_csv(filename, float_format=fmt, sep=",", index=False)


def save_simulation_data(filename, data, number_of_components=1, data_name="viscosity", fmt="%.3e"):
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
    header = ""
    if number_of_components == 2:
        header = f"pf, N1, N2, m1, m2, T, sigma1, sigma2, cut, {data_name}"
    if number_of_components == 1:
        header = f"pf, N, m, T, sigma, cut, {data_name}"
    # This does not work as intended
    data = np.sort(data.view('float,'*len(data[0])), order=['f0','f3'], axis=0).view(np.float)
    np.savetxt(filename, data, header=header, fmt=fmt, delimiter=", ", comments="")


def insert_results_in_array(data, value, C, i, err=None, pf=None):
    """
        Given a 1 or 2D array of data, inserts 
        a 0 or 1D value array at index i.
        Returns the new array.
    """
    parameters = get_system_config(C, pf)
    if np.isscalar(value):
        vals = np.append(parameters, value)
        if err:
            vals = np.append(vals, err)
        data[i] = vals
    else:
        l = len(parameters)
        assert len(value) == len(data[i,l:l+len(value)]), (
            f"ERROR: value has shape {value.shape} and does not " \
            f"fit into the remaining columns of data, which has " \
            f"shape {data[i,l:l+len(value)].shape}"
        )
        data[i,:l] = parameters
        data[i,l:l+len(value)] = value
    return data


def get_system_config(C=None, pf=None, number_of_components=None):
    """
        Given a many particle system, this function creates 
        an array of its characterizing quantities, such as
        packing fraction, particle number and mass etc.
        To be used in datafiles.
    """
    #parameters = np.zeros(1)
    if C != None:
        number_of_components = C["ATOM_TYPES"]
        if number_of_components == 1:
            N = C["N"]
            m = C["MASS"]
            T = C["TEMP"]
            sigma = C["SIGMA"]
            cut = C["CUTOFF"]
            if pf == None:
                pf = C["PF"]
            parameters = np.array([pf, N, m, T, sigma, cut])
        elif number_of_components == 2:
            N1 = C["N_L"]
            N2 = C["N_H"]
            m1 = C["MASS_L"]
            m2 = C["MASS_H"]
            T  = C["TEMP"]
            sigma1 = C["SIGMA_L"]
            sigma2 = C["SIGMA_H"]
            if pf == None:
                pf = C["PF"]
            parameters = np.array([pf, N1, N2, m1, m2, T, sigma1, sigma2])
    # If no arguments, return empty array with corect size:
    else: 
        if number_of_components == 1:
            # NOTE: This should be 4 normally.
            # TODO: Make versatile functionality here, based on required config.
            parameters = np.zeros(6)
        elif number_of_components == 2:
            parameters = np.zeros(8)
    return parameters


def create_data_array(filenames, theory_functions, number_of_components, extra_values=0):
    """
        Given a list of filenames, and a list of theoretical
        functions of interest, this returns an empty array 
        containing enough space for all quantities.
    """
    rows = len(filenames)
    columns = (
        len(get_system_config(number_of_components=number_of_components))
        + 2 + len(theory_functions)*number_of_components + extra_values
    )
    data = np.zeros((rows,columns))
    return data
