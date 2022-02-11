#############################################################
# This script converts a number of LAMMPS files to .csv     #
# files containing viscosity and equation of state data.    #
# Files are ready to be plotted in report with pgfplots.    #
#############################################################

import numpy as np
import matplotlib.pyplot as plt
import logging
import sys

import files
import theory 
import muller_plathe
import save
import utils
import rdf
import convert
import block_average
import eos

sysargs = sys.argv
log = logging.getLogger()
log.addHandler(logging.StreamHandler(sys.stdout))
if "debug" in sysargs:
    # logging.Debug is more correct to use, 
    # but info is cleaner.
    log.setLevel(logging.INFO)

data_path_list = ["./data/run_test"]
save_path_list = ["test"]
#data_path_list = ["./data/heavy_low_exchange_rate"]
#save_path_list = ["heavy"]

# 1. convert all files in directory
# 2. compute viscosity and save data, with theoretical values for the same system
# 3. compute the EOS at equilibrium
# 4. compute the RDF at equilibrium
# 5. plot interesting quantities from the data files

# def compute_all_viscosities(directory):
# def compute_eos_at_equilibrium(directory):
# def compute_rdf_at_equilibrium(directory):
# def set_computation_parameters():
# 
# convert_all_to_csv(directory):
# params = set_computation_parameters():
# data = compute_all_viscosities(directory):
# save(data)
# data = compute_eos_at_equilibrium(directory):
# save(data)
# data = compute_rdf_at_equilibrium(directory):
# save(data)

## Rename to convert_something_something
def main():
    computation_params = {
        "particle_types": 1,
        "cut_fraction"  : 0.3,
        "step" 		    : 4,
        "per_time"		: False,
    }

    theoretical_viscosity = theory.enskog

    for path, savepath in zip(data_path_list, save_path_list):
        if "convert" in sysargs:
            files.all_files_to_csv(path)
        save_dir = "./data/processed/"
        filenames = files.get_all_filenames(path)
        # Brief function to wrap the name of the saved files.
        def get_savename(body):
            return f"{save_dir}{body}_{savepath}.csv"

        #compute_viscosity_from_directory(
        #    path, get_savename("visc"), get_rdf_list(), computation_params, theoretical_viscosity)
        compute_eos_from_directory(
            path, get_savename("eos"), get_eos_list(), computation_params)
        compute_rdf_from_directory(
            path, get_savename("rdf"), get_rdf_list(), computation_params)
        #save_theory(path, filenames, get_savename("theory"))
        #save_rdf(path, filenames, get_savename=("rdf"))


def get_rdf_list():
    #return [theory.rdf_SPT, theory.rdf_PY_mix, theory.rdf_BMCSL]
    return [theory.rdf_PY, theory.rdf_CS]


def get_eos_list():
    #return [theory.rdf_SPT, theory.rdf_PY_mix, theory.rdf_BMCSL]
    return [theory.Z_PY, theory.Z_CS]


def compute_viscosity_from_directory(
        directory, 
        savename, 
        theory_functions, 
        computation_params,
        theoretical_viscosity
    ):
    # Compute all the viscosities in directory
    data = muller_plathe.compute_all_viscosities(
        directory, 
        computation_params, 
        theory_functions,
        theoretical_viscosity
    )

    data_name = "viscosity, error"
    data_name += save.get_data_name(theory_functions, theoretical_viscosity) 
    save.save_simulation_data(savename, data, data_name=data_name)


def compute_eos_from_directory(
        directory, 
        savename, 
        theory_functions, 
        computation_params
    ):
    # Compute all the viscosities in directory
    data = eos.compute_all_eoss(
        directory, 
        theory_functions,
        computation_params
    )
    data_name = "Z, error"
    data_name += save.get_data_name(theory_functions) # TODO: Start in this function
    save.save_simulation_data(savename, data, data_name=data_name)


def compute_rdf_from_directory(
        directory, 
        savename, 
        theory_functions, 
        computation_params
    ):
    # Compute all the viscosities in directory
    data = rdf.compute_all_rdfs(
        directory, 
        theory_functions,
        computation_params
    )
    data_name = "g_sigma, error"
    data_name += save.get_data_name(theory_functions) # TODO: Start in this function
    save.save_simulation_data(savename, data, data_name=data_name)

main()
