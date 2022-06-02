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
    "step" 		    : 4,
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
                path, get_savename("visc"), get_helmholtz_list(), computation_params, theoretical_viscosity)
        if "eos" in sysargs:
            compute_eos_from_directory(
                path, get_savename("eos"), get_eos_list(), computation_params)
        if "rdf" in sysargs:
            compute_rdf_from_directory(
                path, get_savename("rdf"), get_rdf_list(), computation_params)
        if "vel" in sysargs:
            compute_velcity_profile_from_directory(
                path, get_savename("vel"), computation_params)

        if "theory" in sysargs:
            compute_all_theoretical_values(
                    path,
                    get_savename("theory_eos_of_pf"),
                    get_eos_list(),
                    #get_rdf_list(),
                    #get_helmholtz_list(),
                    "pf",
                    theory_function=None,
                    xmin=0.01,
                    xmax=0.51,
            )
            compute_all_theoretical_values(
                    path,
                    get_savename("theory_eos_of_T"),
                    get_eos_list(),
                    #get_rdf_list(),
                    #get_helmholtz_list(),
                    "T",
                    theory_function=None,
                    xmin=1.3,
                    xmax=4.0,
            )
            compute_all_theoretical_values(
                    path,
                    get_savename("theory_visc_of_pf"),
                    #get_eos_list(),
                    #get_rdf_list(),
                    get_helmholtz_list(),
                    "pf",
                    theory_function=theory.get_viscosity_from_F,
                    xmin=0.01,
                    xmax=0.51,
            )
            compute_all_theoretical_values(
                    path,
                    get_savename("theory_visc_of_T"),
                    #get_eos_list(),
                    #get_rdf_list(),
                    get_helmholtz_list(),
                    "T",
                    theory_function=theory.get_viscosity_from_F,
                    xmin=1.3,
                    xmax=4.0,
            )
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
    # Saving is done within this function.
    regression.compute_all_velocity_profiles(directory, computation_params)


def compute_all_theoretical_values(
        directory,
        savename,
        equation_list,
        ordinate_variable,
        theory_function=None,
        xmin=0.01,
        xmax=0.51,
    ):
    """
        Given a directory of lammps output files,
        compute the theoretical value of EOSs and 
        RDFs for all the simulated systems. Store
        the results in one single file.
    """
    ordinate_index = 0
    header = f"pf,N,m,T,sigma,cut"
    if ordinate_variable == "T":
        ordinate_index = 3
        header = f"T,pf,N,m,sigma,cut"

    system_configs = files.get_all_configs(directory)
    ordinate_variable
    # Drop packing fraction and remove duplicates
    # We now have an array of all (N, m, T, sigma, cut) 
    # that were used to generate the data.
    system_configs = np.delete(system_configs, ordinate_index, axis=1)
    system_configs = np.unique(system_configs, axis=0)

    # Now, we can compute every theoretical function in eos_list,
    # rdf_list, and helmholtz_list with these configurations.
    x = np.linspace(xmin, xmax)
    # Join C and pf into one large array of configurations 
    # Shape: (len(pf)*len(C), 6)
    C = np.tile(system_configs, (len(x),1))
    C = np.insert(C, ordinate_index, np.array([np.repeat(x, len(C)//len(x))]), axis=1)

    # For compatibility with pandas, use slightly different conventions to save.
    np.savetxt(savename, C, header=header, fmt="%.3e", delimiter=",", comments="")
    data = np.zeros(len(C[:,0]))
    for eq in equation_list:
        for i in range(len(C)):
            c = C[i] 
            utils.status_bar(i, len(C))
            # comp_fraction (usually "x") equals one for one-component fluids.
            # Code needs generalization to work with mulit-component systems.
            sigma, comp_fraction, pf, T = np.array([c[3]]), np.array([1]), c[0], c[2]
            rho = theory.pf_to_rho(sigma, comp_fraction, pf)
            if theory_function:
                data[i] = theory_function(eq, sigma, comp_fraction, rho, T)
            else:
                data[i] = eq(sigma, comp_fraction, rho, temp=T)
        name = save.get_data_name([eq], viscosity_function=theory_function).replace(",", "").strip()
        save.add_column_to_file(savename, data, name)


def old(
        directory,
        savename,
        eos_list,
        rdf_list,
        helmholtz_list,
        function=None
    ):
    for Z in eos_list:
        for i in range(len(C)):
            c = C[i] 
            utils.status_bar(i, len(C))
            sigma, x, pf = np.array([c[3]]), np.array([1]), c[0]
            rho = theory.pf_to_rho(sigma, x, pf)
            data[i] = Z(sigma, x, rho, temp=T)
        name = save.get_data_name([Z]).replace(",", "").strip()
        save.add_column_to_file(savename, data, name)
    for g in rdf_list:
        for i in range(len(C)):
            c = C[i] 
            utils.status_bar(i, len(C))
            sigma, x, pf = np.array([c[3]]), np.array([1]), c[0]
            rho = theory.pf_to_rho(sigma, x, pf, temp=T)
            data[i] = g(sigma, x, rho)
        name = save.get_data_name([g]).replace(",", "").strip()
        save.add_column_to_file(savename, data, name)
    for F in helmholtz_list:
        for i in range(len(C)):
            c = C[i] 
            utils.status_bar(i, len(C))
            sigma, x, pf, T = np.array([c[3]]), np.array([1]), c[0], c[2]
            rho = theory.pf_to_rho(sigma, x, pf)
            data[i] = theory.get_rdf_from_F(F, sigma, x, rho, T)
        name = save.get_data_name([F]).replace(",", "").strip()
        save.add_column_to_file(savename, data, name)
    for F in helmholtz_list:
        for i in range(len(C)):
            c = C[i] 
            utils.status_bar(i, len(C))
            sigma, x, pf, T = np.array([c[3]]), np.array([1]), c[0], c[2]
            rho = theory.pf_to_rho(sigma, x, pf)
            data[i] = theory.get_viscosity_from_F(F, sigma, x, rho, T)
        name = "enskog_"+save.get_data_name([F]).replace(",", "").strip()
        save.add_column_to_file(savename, data, name)



def get_rdf_list():
    return [theory.rdf_PY, theory.rdf_CS, theory.rdf_LJ]


def get_helmholtz_list():
    #return [theory.F_kolafa, theory.F_thol, theory.F_mecke, theory.F_gottschalk, theory.F_hess, theory.rdf_LJ]
    return [theory.F_kolafa, theory.F_thol, theory.F_mecke, theory.F_gottschalk]


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
