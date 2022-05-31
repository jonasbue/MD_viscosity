#############################################################
# This script converts a number of LAMMPS files to .csv     #
# files containing viscosity and equation of state data.    #
# Files are ready to be plotted in report with pgfplots.    #
#############################################################

import numpy as np
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
import regression

sysargs = sys.argv
log = logging.getLogger()
log.addHandler(logging.StreamHandler(sys.stdout))
if "debug" in sysargs:
    # logging.Debug is more correct to use, 
    # but info is cleaner.
    log.setLevel(logging.INFO)

data_path_list = ["./data/lj"]
save_path_list = ["lj"]

computation_params = {
    "particle_types": 1,
    "cut_fraction"  : 0.3,
    "step" 		    : 6,
    "per_time"		: False,
}

# 1. convert all files in directory
# 2. compute viscosity and save data, along with theoretical values 
# 3. compute the EOS at equilibrium
# 4. compute the RDF at equilibrium
# 5. plot interesting quantities from the data files

## Rename to convert_something_something
def main():
    theoretical_viscosity = theory.enskog

    for path, savepath in zip(data_path_list, save_path_list):
        if "convert" in sysargs:
            files.all_files_to_csv(path)
        save_dir = "./data/processed/"
        filenames = files.get_all_filenames(path)
        # Brief function to wrap the name of the saved files.
        def get_savename(body):
            return f"{save_dir}{body}_{savepath}.csv"

        if "visc" in sysargs:
            compute_viscosity_from_directory(
                path, get_savename("visc"), get_rdf_list(), computation_params, theoretical_viscosity)
        if "eos" in sysargs:
            compute_eos_from_directory(
                path, get_savename("eos"), get_eos_list(), computation_params)
        if "rdf" in sysargs:
            compute_rdf_from_directory(
                path, get_savename("rdf"), get_rdf_list(), computation_params)
        if "vel" in sysargs:
            compute_velcity_profile_from_directory(
                path, get_savename("vel"), computation_params)
        # To make nice plots, it is convenient to save a separate 
        # file of theoretical values, with denser data points than 
        # the numerical data. TODO: Cleanup.
        #save_theory(path, filenames, get_savename("theory"))
        #save_rdf(path, filenames, get_savename=("rdf"))


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
    save.save_simulation_data(savename, data, data_name=data_name, number_of_components=computation_params["particle_types"])


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
    data_name += save.get_data_name(theory_functions) 
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
        computation_params,
        cut=0.9,
        dr=0.04
    )
    data_name = "g_sigma, error"
    data_name += save.get_data_name(theory_functions) 
    save.save_simulation_data(savename, data, data_name=data_name)


def compute_velcity_profile_from_directory(
        directory, 
        savename, 
        computation_params
    ):
    # Compute all the viscosities in directory
    data = regression.compute_all_velocity_profiles(directory, computation_params)
    data_name = "z, vx"
    save.save_simulation_data(savename, data, data_name=data_name)

def get_rdf_list():
    return [theory.rdf_PY, theory.rdf_CS, theory.rdf_LJ]


def get_eos_list():
    return [
        theory.Z_CS,
        theory.Z_kolafa,
        theory.Z_gottschalk,
        theory.Z_thol,
        theory.Z_mecke,
        theory.Z_hess
    ]

main()
