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
import eos
import regression
import names
import curve_fit

sysargs = sys.argv
log = logging.getLogger()
log.addHandler(logging.StreamHandler(sys.stdout))
if "debug" in sysargs:
    # logging.Debug is more correct to use, 
    # but info is cleaner.
    log.setLevel(logging.INFO)

data_path_list = ["./data/lj"]
save_path_list = ["lj"]
resolution = 25

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

        if "visc" in sysargs:
            compute_viscosity_from_directory(
                path, names.get_savename("visc", save_dir, savepath), names.get_helmholtz_list(), computation_params, theoretical_viscosity)
        if "eos" in sysargs:
            compute_eos_from_directory(
                path, names.get_savename("eos", save_dir, savepath), names.get_eos_list(), computation_params)
        if "rdf" in sysargs:
            compute_rdf_from_directory(
                path, names.get_savename("rdf", save_dir, savepath), names.get_rdf_list(), computation_params)
        if "vel" in sysargs:
            compute_velcity_profile_from_directory(
                path, names.get_savename("vel", save_dir, savepath), computation_params)
        if "theory" in sysargs:
            # This is a collection of function calls to make 
            # different theory data files for plotting.
            theory_plotting(path, save_dir, savepath, resolution=resolution) 
        if "fit" in sysargs:
            fit_parameters(save_dir, savepath, resolution=resolution)


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
    #data = eos.compute_all_eoss(
    #    directory, 
    #    theory_functions,
    #    computation_params
    #)
    #data_name = "Z, error"
    #data_name += save.get_data_name(theory_functions) 
    #save.save_simulation_data(savename, data, data_name=data_name)

    # Compute the internal energy of all simulations as well.
    data = eos.compute_all_internal_energies(
        directory, 
        theory_functions,
        computation_params
    )
    data_name = "U, error"
    data_name += save.get_data_name(theory_functions) 
    savename = savename.replace("eos", "U")
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
        dr=0.04,
        recompute=False
    )
    data_name = "g_max, g_one, g_eff, error"
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


def theory_plotting(path, save_dir, savepath, resolution=20):
    compute_all_theoretical_values(
            path, names.get_savename("theory_eos_of_pf", save_dir, savepath),
            names.get_eos_list(), "pf", resolution=resolution,
            #theory_function=theory.get_Z_from_F,
    )
    compute_all_theoretical_values(
            path, names.get_savename("theory_eos_of_T", save_dir, savepath),
            names.get_eos_list(), "T", resolution=resolution,
            #method_list=get_method_list(),
    )
    compute_all_theoretical_values(
            path, names.get_savename("theory_rdf_of_pf", save_dir, savepath),
            names.get_helmholtz_list(), "pf", resolution=resolution,
            #method_list=get_method_list(),
            theory_function=theory.get_rdf_from_F,
    )
    compute_all_theoretical_values(
            path, names.get_savename("theory_rdf_of_T", save_dir, savepath),
            names.get_helmholtz_list(), "T", resolution=resolution,
            theory_function=theory.get_rdf_from_F,
    )
    compute_all_theoretical_values(
            path, names.get_savename("theory_visc_of_pf", save_dir, savepath),
            names.get_helmholtz_list(), "pf", resolution=resolution,
            #method_list=get_method_list(),
            theory_function=theory.get_viscosity_from_F,
            #collision_integrals=names.get_fitted_collision_integrals(),
    )
    compute_all_theoretical_values(
            path, names.get_savename("theory_visc_of_T", save_dir, savepath),
            names.get_helmholtz_list(), "T", resolution=resolution,
            #method_list=get_method_list(),
            theory_function=theory.get_viscosity_from_F,
            #collision_integrals=get_fitted_collision_integrals(),
    )
    #compute_all_theoretical_values(
    #        path, names.get_savename("theory_visc_of_pf_fudged", save_dir, savepath),
    #        names.get_helmholtz_list(), "pf",
    #        #method_list=get_method_list(),
    #        theory_function=theory.get_viscosity_from_F,
    #        collision_integrals=get_fitted_collision_integrals(),
    #)
    #compute_all_theoretical_values(
    #        path, names.get_savename("theory_visc_of_T_fudged", save_dir, savepath),
    #        names.get_helmholtz_list(), "T",
    #        #method_list=get_method_list(),
    #        theory_function=theory.get_viscosity_from_F,
    #        collision_integrals=get_fitted_collision_integrals(),
    #)
    # To make nice plots, it is convenient to save a separate 
    # file of theoretical values, with denser data points than 
    # the numerical data. TODO: Clean up function calls.
    #save_theory(path, filenames, names.get_savename("theory"))
    #save_rdf(path, filenames, names.get_savename=("rdf"))



def compute_all_theoretical_values(
        directory,
        savename,
        equation_list,
        ordinate_variable,
        method_list=[],
        theory_function=None,
        collision_integrals=[],
        xmin=0.02,
        xmax=0.50,
        resolution=25,
    ):
    """
        Given a directory of lammps output files,
        compute the theoretical value of EOSs and 
        RDFs for all the simulated systems. Store
        the results in one single file.
    """
    ordinate_index = 0
    N_index = 1
    header = f"pf,N,m,T,sigma,cut"
    if ordinate_variable == "T":
        ordinate_index = 3
        #N_index = 2
        #header = f"T,pf,N,m,sigma,cut"
        xmin, xmax = 1.3, 4.0
        resolution=28
    system_configs = files.get_all_configs(directory)
    ordinate_variable
    # If there are multiple values for N, we can get rid of all but one.
    # Produces smaller files.
    N = system_configs[0,N_index]
    # Set all values of N to the first one.
    # This does not affect the function values, but makes for smaller files.
    system_configs[:,N_index] = N #system_configs[:,N_index]==N, N, np.nan axis=0)

    # Drop packing fraction and remove duplicates
    # We now have an array of all (N, m, T, sigma, cut) 
    # that were used to generate the data.
    system_configs = np.delete(system_configs, ordinate_index, axis=1)
    system_configs = np.unique(system_configs, axis=0)

    # Now, we can compute every theoretical function in eos_list,
    # rdf_list, and helmholtz_list with these configurations.
    x = np.linspace(xmin, xmax, resolution)
    # Join C and pf into one large array of configurations 
    # Shape: (len(pf)*len(C), 6)
    C = np.tile(system_configs, (len(x),1))
    C = np.insert(C, ordinate_index, np.array([np.repeat(x, len(C)//len(x))]), axis=1)

    # For compatibility with pandas, use slightly different conventions to save.
    np.savetxt(savename, C, header=header, fmt="%.3e", delimiter=",", comments="")
    data = np.zeros(len(C[:,0]))

    for (j, eq) in enumerate(equation_list):
        for i in range(len(C)):
            utils.status_bar(i, len(C))
            c = C[i] 
            coll=1.0
            if len(collision_integrals):
                coll=collision_integrals[j]
            # comp_fraction (usually "x") equals one for one-component fluids.
            # Code needs generalization to work with mulit-component systems.
            sigma, comp_fraction, pf, T = np.array([c[4]]), np.array([1]), c[0], c[3]
            rho = theory.pf_to_rho(sigma, comp_fraction, pf)
            if theory_function:
                if eq.__name__[:3] == "rdf":
                    if theory_function.__name__[:14] == "get_viscosity":
                        data[i] = theory_function(eq, sigma, comp_fraction, rho, T, collision_integral=coll, no_F=True)
                    elif theory_function.__name__[:7] == "get_rdf":
                        data[i] = eq(pf, T=T)
                else:
                    data[i] = theory_function(eq, sigma, comp_fraction, rho, T, collision_integral=coll)
            else:
                if eq.__name__[:3] == "rdf":
                    data[i] = eq(pf, T=T)
                else:
                    data[i] = eq(sigma, comp_fraction, rho, temp=T, collision_integral=coll)
        name = save.get_data_name([eq], viscosity_function=theory_function).replace(",", "").strip()
        save.add_column_to_file(savename, data, name)


def fit_parameters(save_dir, savepath, resolution=20):
    savename = names.get_savename("theory_visc_of_pf", save_dir, savepath)
    filename = names.get_savename("visc", save_dir, savepath)
    omega, eta, sigma = curve_fit.fit_viscosity(
        filename,
        curve_fit.simplified_enskog, 
        resolution,
        fit_sigma=False
    )
    name = "omega"
    omega = omega.flatten(order="F")
    save.add_column_to_file(savename, omega, name)

    omega, eta, sigma = curve_fit.fit_viscosity(
        filename,
        curve_fit.simplified_enskog,
        resolution,
        fit_sigma=True
    )
    name = "enskog_F-thol-fitted"
    eta = eta.flatten(order="F")
    sigma = sigma.flatten(order="F")
    save.add_column_to_file(savename, eta, name)
    save.add_column_to_file(savename, sigma, name)
    #save.add_column_to_file(savename, omega, name)

main()
