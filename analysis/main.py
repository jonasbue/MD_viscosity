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

sysargs = sys.argv
log = logging.getLogger()
log.addHandler(logging.StreamHandler(sys.stdout))
if "debug" in sysargs:
    # logging.Debug is more correct to use, 
    # but info is cleaner.
    log.setLevel(logging.INFO)


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
        "particle_types": 2,
        "cut_fraction"  : 0.3,
        "step" 		    : 20,
        "per_time"		: False,
    }

    data_path_list = ["./data/run_test"]
    save_path_list = ["savename"]
    theoretical_viscosity = theory.enskog

    for path, savepath in zip(data_path_list, save_path_list):
        if "convert" in sysargs:
            files.all_files_to_csv(path)
        save_dir = "./data/processed/"
        filenames = files.get_all_filenames(path)
        # Brief function to wrap the name of the saved files.
        def get_savename(body):
            return f"{save_dir}{body}_{savepath}.csv"

        compute_viscosity_from_directory(
            path, get_savename("visc"), get_rdf_list(), computation_params, theoretical_viscosity)
        compute_eos_from_directory(
            path, get_savename("eos"), get_eos_list(), computation_params)
        #save_theory(path, filenames, get_savename("theory"))
        #save_rdf(path, filenames, get_savename=("rdf"))

def get_rdf_list():
    #return [theory.rdf_SPT, theory.rdf_PY_mix, theory.rdf_BMCSL]
    return [theory.rdf_PY, theory.rdf_CS]
def get_eos_list():
    #return [theory.rdf_SPT, theory.rdf_PY_mix, theory.rdf_BMCSL]
    return [theory.Z_PY, theory.Z_CS]


# This function should not be.
# Have one compute_all, and then one save_data which takes the array.
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

    data_name = save.get_data_name(theory_functions, theoretical_viscosity) # TODO: Start in this function
    save.save_simulation_data(savename, data, data_name=data_name)

def compute_eos_from_directory(
        directory, 
        savename, 
        theory_functions, 
        computation_params
    ):
    # Compute all the viscosities in directory
    data = compute_all_eoss(
        directory, 
        theory_functions,
        computation_params
    )
    data_name = save.get_data_name(theory_functions) # TODO: Start in this function
    save.save_simulation_data(savename, data, data_name=data_name)

def compute_all_eoss(
        directory, 
        theory_functions, 
        computation_params
    ):
    path = directory
    filenames = files.get_all_filenames(directory)
    eos_list = theory_functions
    data = save.create_data_array(filenames, eos_list)
    for (i, f) in enumerate(filenames):
        #utils.status_bar(i, len(filenames), fmt="arrow")
        fix_name = f"{path}/" + f[0]
        log_name = f"{path}/" + f[1]
        log.info(f"Loading file\t{fix_name}")
        log.info(f"Loading file\t{log_name}")

        Z, C, error = get_eos_from_file(log_name, fix_name, computation_params["cut_fraction"])

        theoretical_values = [theory.get_Z_from_C(C, Z) for Z in theory_functions]
        values = np.array([Z, error])
        values = np.append(values, theoretical_values)
        save.insert_results_in_array(data, values, C, i)
    print("")
    return data


def get_eos_from_file(log_name, fix_name, cut_fraction):
    variable_list = ["p", "V", "T"]
    C = convert.extract_constants_from_log(log_name)
    log_table = files.load_system(log_name)
    pvt = files.unpack_variables(log_table, log_name, variable_list)

    eq_steps = int(
            C["EQUILL_STEPS"]
            /C["THERMO_OUTPUT"]
        )
    cut = int(eq_steps*cut_fraction)
    p = np.array(pvt[variable_list.index("p")][cut:eq_steps])
    T = np.array(pvt[variable_list.index("T")][cut:eq_steps])

    N_list = utils.get_component_lists(C, "N")
    sigma_list = utils.get_component_lists(C, "SIGMA")
    x = N_list/np.sum(N_list)
    rho = 6*C["PF"]/np.pi/np.sum(x*np.diag(sigma_list)**3)
    Z = theory.Z_measured_mix(p, rho, T)
    std = block_average.get_block_average(Z)
    Z = np.mean(Z)
    return Z, C, std

def save_eos(path, filenames, cut_fraction, number_of_components, savename):
    mix_eos_list = [theory.Z_SPT, theory.Z_PY, theory.Z_BMCSL]
    one_eos_list = [theory.Z_CS]
    columns = len(save.get_system_config()) + 2 + len(mix_eos_list) + len(one_eos_list)
    data = np.zeros((len(filenames), columns))
    data_name = "Z, error"
    for (i, f) in enumerate(filenames):
        utils.status_bar(i, len(filenames), fmt="arrow")
        fix_name = f"{path}/" + f[0]
        log_name = f"{path}/" + f[1]
        log.info(f"Loading file\t{fix_name}")
        log.info(f"Loading file\t{log_name}")

        x, rho, Z, std = get_eos_from_file(log_name, fix_name, number_of_components, cut_fraction)

        mix_eos_vals = np.zeros(len(mix_eos_list))
        one_eos_vals = np.zeros(len(one_eos_list))
        for (j, eq) in enumerate(mix_eos_list):
            mix_eos_vals[j] = eq(theory.get_sigma(sigma_list), x, rho)
        for (j, eq) in enumerate(one_eos_list):
            one_eos_vals[j] = eq(C["PF"])
            
        err = std 
        values = np.array([Z, err])
        values = np.append(values, mix_eos_vals)
        values = np.append(values, one_eos_vals)

        save.insert_results_in_array(data, values, C, i)
    data_name += "".join(
        [f", Z_{r.__name__[2:]}" for r in mix_eos_list])
    data_name += "".join(
        [f", Z_{r.__name__[2:]}" for r in one_eos_list])
    save.save_simulation_data(savename, data, data_name=data_name)


def save_theory(path, filenames, savename, N=50):
    """ This function is now obsolete. """
    pf_experiment = files.find_all_packing_fractions(path)

    skipped = 0
    included = 0
    # First, because I can't figure it out: 
    # Count the number of entries to save.
    for (i, f) in enumerate(filenames):
        if files.get_packing_from_filename(f[0]) != pf_experiment[0]:
            skipped += 1
            continue
        included += 1

    #print("Included:", included)
    #print("Skipped:", skipped)
    data_shape = (N*included, 19)
    data = np.zeros(data_shape)

    pf = np.linspace(0,0.5,N)
    i = -1
    for f in filenames:
        if files.get_packing_from_filename(f[0]) != pf_experiment[0]:
            continue
        else:
            i += 1
        fix_name = f"{path}/" + f[0]
        log_name = f"{path}/" + f[1]
        log.info(f"Loading file\t{fix_name}")
        log.info(f"Loading file\t{log_name}")

        C = convert.extract_constants_from_log(log_name)

        T = C["TEMP"]
        N_list = np.array([C["N_L"], C["N_H"]])
        sigma_list = np.array([C["SIGMA_L"], C["SIGMA_H"]])
        mass_list = np.array([C["MASS_L"], C["MASS_H"]])
        x = N_list/np.sum(N_list)

        thorne_vals_SPT = np.zeros(N)
        thorne_vals_PY = np.zeros(N)
        thorne_vals_BMCSL = np.zeros(N)
        enskog_vals_1 = np.zeros(N)
        enskog_vals_2 = np.zeros(N)
        SPT_vals = np.zeros(N)
        PY_vals = np.zeros(N)
        CS_vals = np.zeros(N)
        BMCSL_vals = np.zeros(N)
        rdf_SPT = np.zeros(N)
        rdf_PY = np.zeros(N)
        rdf_BMCSL = np.zeros(N)

        for j in range(N):
            thorne_vals_SPT[j] = theory.thorne(
                pf[j], x, mass_list, sigma_list, T, rdf=theory.rdf_SPT)
            thorne_vals_PY[j] = theory.thorne(
                pf[j], x, mass_list, sigma_list, T, rdf=theory.rdf_PY_mix)
            thorne_vals_BMCSL[j] = theory.thorne(
                pf[j], x, mass_list, sigma_list, T, rdf=theory.rdf_BMCSL)
            enskog_vals_1[j] = theory.enskog(
                pf[j], sigma_list[0], T, mass_list[0])
            enskog_vals_2[j] = theory.enskog(
                pf[j], sigma_list[1], T, mass_list[1])
            rho = 6*pf[j]/np.pi/np.sum(x*np.diag(sigma_list)**3)
            sigma = theory.get_sigma(sigma_list)
            SPT_vals[j] = theory.Z_SPT(sigma, x, rho)
            PY_vals[j] = theory.Z_PY(sigma, x, rho)
            BMCSL_vals[j] = theory.Z_BMCSL(sigma, x, rho)
            CS_vals[j] = theory.Z_CS(pf[j])

            # TODO: Fix indices. Probably separate file?
            rdf_SPT[j] = theory.rdf_SPT(sigma, x, rho, 0, 0)
            rdf_PY[j] = theory.rdf_PY_mix(sigma, x, rho, 0, 0)
            rdf_BMCSL[j] = theory.rdf_BMCSL(sigma, x, rho, 0, 0)
            vals = np.array([
                thorne_vals_SPT[j], 
                thorne_vals_PY[j], 
                thorne_vals_BMCSL[j], 
                enskog_vals_1[j], 
                enskog_vals_2[j], 
                SPT_vals[j], 
                PY_vals[j], 
                BMCSL_vals[j],
                CS_vals[j],
                rdf_SPT[j],
                rdf_PY[j],
                rdf_BMCSL[j],
            ])
            data = save.insert_results_in_array(data, vals, C, i*N+j, pf=pf[j])

    save.save_simulation_data(savename, data, 
            data_name="thorne_SPT, thorne_PY_mix, thorne_BMCSL, enskog_1, enskog_2, SPT_EoS, PY_EoS, BMCSL_EoS, CS_EoS, SPT_RDF_ii, PY_RDF_ii, BMCSL_RDF_ii"
    )


def save_rdf(path, filenames, savename):
    rdf_at_contact = np.zeros(len(filenames))
    rdf_list = [theory.rdf_PY_mix, theory.rdf_SPT, theory.rdf_BMCSL]
    theoretical = np.zeros((len(filenames), 3))
    pf = np.zeros(len(filenames))
    columns = len(save.get_system_config())+2+len(rdf_list)
    data = np.zeros((len(filenames), columns))
    for i, filename in enumerate(filenames):
        dump_name = f"{path}/" + filename[2]
        print(f"Loading file\t{dump_name}")
        log_name = f"{path}/" + filename[1]
        print(f"Loading file\t{log_name}")

        # Compute rdf for all dump files.
        dump_to_rdf.calcRDF(dump_name, 1, 100, 5, 30, 0.05, only_component_2=True)

        # Export a g_sigma for every simulation run.  g_r can be plotted as is,
        # but that should be done only for one or two system configurations in
        # the report.
        rdf_name = dump_name.replace('dump', 'RDF')+'.csv' 
        g_sigma, g_r, r, std = rdf.export_RDF_data(rdf_name, savename)
        # TODO: Estimate error
        err = std
        rdf_at_contact[i] = g_sigma

        # Compute theoretical RDF at contact:
        C           = convert.extract_constants_from_log(log_name)
        pf          = C["PF"]
        T           = C["TEMP"]
        N_list      = np.array([C["N_L"], C["N_H"]])
        sigma_list  = np.array([C["SIGMA_L"], C["SIGMA_H"]])
        mass_list   = np.array([C["MASS_L"], C["MASS_H"]])
        sigma       = theory.get_sigma(sigma_list)
        x           = N_list/np.sum(N_list)
        rho         = 6*pf/np.pi/np.sum(x*np.diag(sigma)**3)

        rdf_values = np.zeros(2+len(rdf_list))
        rdf_values[0] = g_sigma
        rdf_values[-1] = err
        for j, Xi in enumerate(rdf_list):
            g = Xi(sigma, x, rho, 0, 0)
            rdf_values[1+j] = g
            theoretical[i, j] = g
        
        save.insert_results_in_array(data, rdf_values, C, i)
    save.save_simulation_data(savename, data, data_name="rdf_measured, rdf_PY_mix, rdf_SPT, rdf_BMCSL, error")
    print(f"Saved to file\t{savename}")
    if "debug" in sysargs:
        dump_to_rdf.plotRDF_fromCSV(path=path + "/")

main()

