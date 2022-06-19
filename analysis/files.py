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
import convert
from os import listdir
import logging
import sys
import theory
import tests
import save

log = logging.getLogger("__main__." + __name__)
log.addHandler(logging.StreamHandler(sys.stdout))
if "debug" in sys.argv:
    log.setLevel(logging.INFO)


def get_all_filenames(directory):
    """ 
        Returns a list of all files in a directory, 
        sorted by the packing fraction of the corresponding
        LAMMPS simulation. 
        Files that are not of type fix, log or dump are ignored.
    """
    files = [f for f in listdir(directory)]
    packing_list = find_all_packing_fractions(directory)
    files = sort_files(files, packing_list)
    return files


def sort_files(filenames, packing_fractions, ignore_rdf=True, ignore_dump=False):
    """
        Given a list of filenames, returns a list containing
        only the LAMMPS output files (fix, log, dump), sorted
        so that files from one simulation are grouped together.
        Input:
            filenames:          1D list of filenames
            packing fractions:  packing fractions of the simulated systems
        Output:
            files:              2D list of filenames. Shape is (N, 3), 
                                so that the three files from simulation 
                                i are at index [i,:].
    """
    fix = []
    log = []
    dump = []
    rdf = []
    vel = []
    for pf in packing_fractions:
        for f in filenames:
            extension = get_file_extension(f)
            filetype = get_filetype(f)
            if filetype in ["fix", "log", "dump", "rdf", "vel"]:
                if get_packing_from_filename(f) == pf:
                    if extension != ".csv":
                        if filetype == "fix":
                            fix.append(f)
                        elif filetype == "log":
                            log.append(f)
                        elif filetype == "dump":
                            dump.append(f)
                    # RDF and vel files will always have a .csv extension. 
                    # They are used for plotting.
                    elif filetype == "rdf":
                        rdf.append(f)
                    elif filetype == "vel":
                        vel.append(f)
    files = np.array([fix, log, rdf, vel], dtype=str)
    files = np.sort(files)
    return files.transpose()
    ## This is a mess. Needs fixing.
    #if not ignore_dump and len(dump):
    #    if not ignore_dump and rdf:
    #        assert len(rdf) == len(fix), f"ERROR. There are {len(rdf)} rdf files, while there are {len(fix)} fix files."
    #        files = np.array([fix, log, dump, rdf], dtype=str)
    #        if vel:
    #            files = np.array([fix, log, dump, rdf, vel], dtype=str)
    #    else:
    #        files = np.array([fix, log, dump], dtype=str)
    #else:
    #    assert len(fix) == len(log), "A file is missing!"
    #    files = np.array([fix, log], dtype=str)
    #files = np.sort(files)
    #return files.transpose()


def file_to_csv(filename, filetype):
    """ Converts a file to a csv, depending on its type.
        This is simply a wrapper of the functions in 
        convert_LAMMPS_output.
    """
    if filetype == "fix" or filetype == "log" or filetype == "dump":
        print("Converting", filename, "of type", filetype)
    if filetype == "fix":
        convert.convert_fix_to_csv(filename)
    elif filetype == "log":
        convert.convert_log_to_csv(filename)
    elif filetype == "dump":
        convert.convert_dump_to_csv(filename)


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
    # Exceptions:
    if filetype == "dum":
        filetype = "dump"
    if filetype == "in.":
        filetype = "script"
    if filetype == "see":
        filetype = "seeds"
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
    # Dangerous syntax: These values may change between use-cases:
    value_identifiers = ["N", "sigma", "temp", "pf", ".lammps"]
    for (i, name)  in enumerate(value_identifiers[:-1]):
        next_name = value_identifiers[i+1]
        val, val_name = get_value_from_filename(filename, name, next_name, value_identifiers[-1])
        vals[name] = val
    return vals
    

def get_value_from_filename(filename, value_name, next_name, end_name):
    name_index = filename.find(value_name)
    value_index = name_index + len(value_name)
    next_index = filename.find(next_name)
    sub_index = filename.find("_", value_index + 1)
    if sub_index == next_index:
        value = float(filename[value_index:next_index].strip("_"))
    else:
        value = float(filename[value_index:next_index].strip("_"))
        #value_name = (value_name+"L", value_name+"H")
        value_name = filename[name_index:value_index].strip("_"), 
    return (value, value_name)


def get_packing_from_filename(filename):
    start = "pf_"
    end = ".lammps"
    pf_val = get_parameter_from_filename(filename, start, end)
    return pf_val


def get_mass_from_filename(filename):
    start = "mass_1_"
    end = "_pf"
    m = get_parameter_from_filename(filename, start, end)
    return m


def get_fraction_from_filename(filename):
    start = "N_"
    end = "_"
    N1 = get_parameter_from_filename(filename, start, end)
    start = "N_"+str(int(N1))
    end = "_"
    N2 = get_parameter_from_filename(filename, start, end)
    return N1, N2


def get_fraction_from_filename(filename):
    start = "sigma_"
    end = "_mass"
    sigma = get_parameter_from_filename(filename, start, end)
    return sigma 


def get_parameter_from_filename(filename, start_key, end_key):
    start_index = filename.find(start_key) + len(start_key)
    end_index = filename.find(end_key, start_index)
    val = float(filename[start_index:end_index])
    return val


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
    packing_list = np.sort(packing_list)
    return packing_list


def load_system(filename, skiprows=0):
    """ Loads a LAMMPS logfile, and returns
        thermodynamical variables from it,
        as a numpy array.
        Inputs:
            filename:   name of logfile, string.
        Returns:
            data:       Array of logged values from LAMMPS.
                        First index selects the variable,
                        second index selects the timestep.
    """
    log.info(f"loading file: {filename}")
    data = pd.read_csv(
        filename + ".csv", 
        skiprows=skiprows, 
        delimiter=","
    )
    data = np.array(data)
    return data.transpose()


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
                            ...
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
        "PotEng"            : "U",
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
    log_table = load_system(log_filename, skiprows=2)
    log_vals = unpack_variables(
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
    fix_table = load_system(fix_filename)
    fix_vals = unpack_variables(
        fix_table, 
        fix_filename, 
        fix_variable_list
    )
    N_chunks = int(fix_vals[fix_variable_list.index("Nchunks")][0])

    # Check that chunk number is correct.
    tests.assert_chunk_number(N_chunks, constants)
    return constants, Lz, t, A, Ptot, N_chunks 


def search_for_configs(filename):
    filename = f"data/processed/{filename}"
    configs = pd.read_csv(filename, sep=", ", engine="python")
    cols = configs.columns[len(system_config):]
    configs = configs.drop(columns=cols)
    configs = configs.drop_duplicates(subset=configs.columns[1:len(system_config)-1])
    return configs


def get_all_configs(directory):
    log_files = get_all_filenames(directory)[:,1]
    config_array = np.zeros((len(log_files), len(save.get_system_config(number_of_components=1))))
    for (i, f) in enumerate(log_files):
        log = f"{directory}/{f}"
        C = convert.extract_constants_from_log(log)
        c = save.get_system_config(C)
        config_array[i,:] = c
    return config_array

