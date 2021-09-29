#####################################################
# This file contains functions to load a log file   #
# from LAMMPS, and convert it into useful data.     #
# The conversion from log.lammps to .csv uses       #
# functions provided by Christopher, while the      #
# functions below extract names from the header     #
# and provide a simple wrapping for the             #
# .csv-conversion functions.                        #
#####################################################

import numpy as np
import pandas as pd
import convert_LAMMPS_output as convert


def load_system(filename):
    """ Loads a LAMMPS logfile, and returns
        thermodynamical variables from it,
        as a numpy array.
        Inputs:
            filename:   name of logfile, string.
        Returns:
            log_table:  Array of logged values from LAMMPS.
                        First index selects the variable,
                        second index selects the timestep.
    """
    convert.convert_log_to_csv(filename)
    log_table = pd.read_csv(
        filename + ".csv", 
        skiprows=2, 
        delimiter=","
    )
    log_table = np.array(log_table)
    return log_table.transpose()


def get_variable_indices(header, variables):
    """ Searches the header of a log file for 
        indices of different variables.
        Inputs:
            header:     headline of thermo output table, np.array.
            variables:  list of variable names as strings/chars.
                        Can include the following:
                            pressure:       "p"
                            volume:         "V"
                            temperature:    "T"
        Returns:
            indices:    np array containing indices of the
                        variables in the same order as they
                        were given in the input argument.
    """
    var_dict = {
        "Step"  : "t",
        "Press"     : "p",
        "Volume"    : "V",
        "Temp"      : "T",
        "Coord1"    : "z",
        "vx"        : "vx",
        "f_MULLER_PLATHE": "Px",
    }
    indices = np.zeros_like(variables, dtype=int)
    for (i, var) in enumerate(header):
        if var in var_dict.keys():
            if var_dict[var] in variables:
                n = variables.index(var_dict[var])
                indices[n] = i
    return indices


def unpack_variables(log_table, filename, variables):
    """ Unpacks specified variables from log table 
        with a header from a LAMMPS log file.

        This assumes that load_system(filename)
        has already been run on this filename,
        so that there exists a .csv file and
        a log_table for the correct simulation run.

        Inputs:
            log_table:  Array of logged values from LAMMPS.
                        First index selects the variable,
                        second index selects the timestep.
            filename:   name of logfile, string.
            variables:  list of variable names as strings/chars.
                        Can include the following:
                            pressure:       "p"
                            volume:         "V"
                            temperature:    "T"
            
    """
    headers = np.array(
        pd.read_csv(filename+".csv", nrows=0).columns
    )
    indices = get_variable_indices(headers, variables)
    return log_table[indices]
