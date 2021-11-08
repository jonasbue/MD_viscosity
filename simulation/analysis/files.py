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
from os import listdir
import logging
import sys

log = logging.getLogger("__main__." + __name__)
log.addHandler(logging.StreamHandler(sys.stdout))
if "debug" in sys.argv:
    log.setLevel(logging.INFO)


def get_all_filenames(directory):
    files = [f for f in listdir(directory)]
    return files


def sort_files(filenames, packing_fractions):
    fix = []
    log = []
    for pf in packing_fractions:
        for f in filenames:
            extension = get_file_extension(f)
            if extension[-3:] != ".sh" and f[:3] != "in.":
                if (
                        get_packing_from_filename(f) == pf 
                        and extension != ".csv"
                    ):
                    filetype = get_filetype(f)
                    if filetype == "fix":
                        fix.append(f)
                    elif filetype == "log":
                        log.append(f)
    files = np.array([fix, log], dtype=str)
    return files.transpose()

def file_to_csv(filename, filetype):
    """ Converts a file to a csv, depending on its type.
        This is simply a wrapper of the functions in 
        convert_LAMMPS_output.
    """
    if filetype == "fix":
        print("Converting", filename, "of type", filetype)
        convert.convert_fix_to_csv(filename)
    elif filetype == "log":
        print("Converting", filename, "of type", filetype)
        convert.convert_log_to_csv(filename)


def get_filetype(filename):
    """ Returns the filetype of a file.
        Filetype should be "fix" or "log".
        For more types, this function requires an extension.
        Input:
            filename:   string. Name of file.
        Output:
            filetype:   string.
    """
    filetype = filename[:3]
    return filetype

def get_file_extension(filename):
    """ Returns the file extension of a file.
        Only works for three-letter extensions,
        so more robust code might be required here.
        This function is expected to be used only with
        ".csv" format.
        Input:
            filename:   string. Name of file.
        Output:
            extension:   string. For example ".csv".
    """
    extension = filename[-4:]
    return extension

def all_files_to_csv(directory):
    """ Converts all files in a directory to csv,
        provided that there is a method for the
        correct filetype defined in file_to_csv().
    """
    files = [f for f in listdir(directory)]
    for filename in files:
        filetype = get_filetype(filename)
        if get_file_extension(filename) != ".csv":
            file_to_csv(f"{directory}/{filename}", filetype)


def read_filename(filename):
    vals = {}
    value_identifiers = ["N", "sigma", "mass", "pf", ".lammps"]
    for name in value_identifiers:
        get_value_from_filename(filename, name, name, next_name)
    
def get_value_from_filename(filename, value_name, value_key, next_name):
    value_index = filename.find(value_name) + len(value_name)
    next_index = filename.find(next_name)
    sub_index = filename.find("_", start=value_index)
    if sub_index+1 == next_index:
        value = float(filename[value_index:next_index])
    else:
        value = (
                    float(filename[value_index:sub_index]), 
                    float(filename[value_index+sub_index:next_index])
                )
        value_name = (value_name+"L", value_name+"H")
    return value, value_name

def get_packing_from_filename(filename):
    pf_key = "pf_"
    end_key = ".lammps"
    pf_index = filename.find(pf_key) + len(pf_key)
    end_index = filename.find(end_key)
    pf_val = float(filename[pf_index:end_index])
    return pf_val


def find_all_packing_fractions(directory):
    """ Searches a directory for .csv files,
        and returns a list of all packing fractions
        associated with the data in the file.
        This does not guarantee that the files
        correspond to the same types of experiments.
    """
    packing_list = np.array([])
    files = [f for f in listdir(directory)]
    for filename in files:
        filetype = get_filetype(filename)
        if get_file_extension(filename) == ".csv":
            pf_val = get_packing_from_filename(filename)
            packing_list = np.append(packing_list, pf_val)
    packing_list = np.unique(packing_list)
    return np.sort(packing_list)


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
    log.info(f"loading file: {filename}")
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
        "Step"              : "t",
        "Timestep"          : "t_fix",
        "Press"             : "p",
        "Volume"            : "V",
        "Temp"              : "T",
        "Coord1"            : "z",
        "vx"                : "vx",
        "f_MULLER_PLATHE"   : "Px",
        "Number_of_chunks"  : "Nchunks",
    }
    indices = np.zeros_like(variables, dtype=int)
    for (i, var) in enumerate(header):
        if var in var_dict.keys():
            if var_dict[var] in variables:
                n = variables.index(var_dict[var])
                indices[n] = i
    return indices


def get_header(filename):
    """ Returns the header of a csv file.
        Input:
            filename:   string. Name of file, without extension.
        Output:
            header:     np.array. Every element is a string 
                        from the first line of the file.
    """
    header = np.array(
        pd.read_csv(filename+".csv", nrows=0).columns
    )
    return header


def unpack_variables(log_table, log_filename, variables):
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
    header = get_header(log_filename)
    indices = get_variable_indices(header, variables)
    return log_table[indices]
