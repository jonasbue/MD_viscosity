#############################################################
# This script converts a number of LAMMPS files to .csv     #
# files containing viscosity and equation of state data.    #
# Files are ready to be plotted in report with pgfplots.    #
#############################################################

import numpy as np
import logging
import sys

import files
import viscosity
import muller_plathe
import eos
import save
import utils
import convert_LAMMPS_output as convert

sysargs = sys.argv
log = logging.getLogger()
log.addHandler(logging.StreamHandler(sys.stdout))
if "debug" in sysargs:
    # logging.Debug is more correct to use, 
    # but info is cleaner.
    log.setLevel(logging.INFO)


## Rename to convert_something_something
def main():
    N = 2
    cut_fraction = 0.7
    per_time=False

    #data_path_list = ["data/varying_mass", "data/varying_sigma", "data/varying_fraction"]
    #save_path_list = ["varying_mass", "varying_sigma", "varying_fraction"]
    data_path_list = ["data/varying_fraction", "data/varying_mass", "data/varying_sigma"]
    save_path_list = ["varying_fraction", "varying_mass", "varying_sigma"]
    #data_path_list = ["data/varying_fraction_equals/varying_fraction_equals"]
    #save_path_list = ["varying_fraction_equals"]
    #data_path_list = ["large_data/data/varying_mass_large", "large_data/data/varying_fraction_large"]
    #save_path_list = ["varying_mass", "varying_fraction"]
    #data_path_list = ["data/const_temp"]
    #save_path_list = ["const_temp"]

    for path, savename in zip(data_path_list, save_path_list):
        if "convert" in sysargs:
            files.all_files_to_csv(path)
        save_dir = "../report/data/"
        filenames = files.get_all_filenames(path)
        packing_list = files.find_all_packing_fractions(path)
        filenames = files.sort_files(filenames, packing_list)
        for i in range(len(filenames)):
            print("------------------------------------------------------------")
            print(filenames[i,0])
            print(filenames[i,1])

        save_viscosity(cut_fraction, path, filenames, savename=f"{save_dir}visc_{savename}.csv", per_time=per_time)
        save_eos(path, filenames, cut_fraction, N, savename=f"{save_dir}eos_{savename}.csv")
        save_theory(path, filenames, savename=f"{save_dir}theory_{savename}.csv")


def save_viscosity(cut_fraction, path, filenames, savename, per_time=False):
    rdf_list = [eos.rdf_SPT, eos.rdf_PY_mix, eos.rdf_BMCSL]
    one_comp_rdf_list = [eos.rdf_CS, eos.rdf_PY]
    columns = len(save.get_system_config()) + 2 + len(rdf_list) + 2*len(one_comp_rdf_list)
    data = np.zeros((len(filenames),columns))
    data_name = "viscosity, error"
    for (i, f) in enumerate(filenames):
        utils.status_bar(i, len(filenames), fmt="percent")
        fix_name = f"{path}/" + f[0]
        log_name = f"{path}/" + f[1]
        log.info(f"Loading file\t{fix_name}")
        log.info(f"Loading file\t{log_name}")

        # Compute and plot viscosity for all packing fractions
        eta, C, eta_err = muller_plathe.find_viscosity_from_file(
            log_name, fix_name, cut_fraction, per_time
        )
        eta = np.mean(eta)
        print(eta_err)
        error = np.mean(eta_err)
        print(eta_err)
        if per_time:
            error = np.amax(np.abs(eta_err))

        thorne_values = np.zeros(len(rdf_list))
        enskog_values = np.zeros((len(one_comp_rdf_list),2))
        for (j, rdf) in enumerate(rdf_list):
            thorne_values[j] = viscosity.get_thorne_from_C(C, rdf)
        for (j, rdf) in enumerate(one_comp_rdf_list):
            enskog_values[j] = viscosity.get_enskog_from_C(C, rdf)

        values = np.array([eta])
        values = np.append(values, error)
        values = np.append(values, thorne_values)
        values = np.append(values, enskog_values[:,0])
        values = np.append(values, enskog_values[:,1])

        save.insert_results_in_array(data, values, C, i)

    print("")
    data_name += "".join(
        [f", thorne_{r.__name__[4:]}" for r in rdf_list])
    data_name += "".join(
        [f", enskog1_{r.__name__[4:]}" for r in one_comp_rdf_list])
    data_name += "".join(
        [f", enskog2_{r.__name__[4:]}" for r in one_comp_rdf_list])
    save.save_simulation_data(savename, data, data_name=data_name)


def save_eos(path, filenames, cut_fraction, number_of_components, savename):
    mix_eos_list = [eos.Z_SPT, eos.Z_PY, eos.Z_BMCSL]
    one_eos_list = [eos.Z_CS]
    columns = len(save.get_system_config()) + 2 + len(mix_eos_list) + len(one_eos_list)
    data = np.zeros((len(filenames), columns))
    data_name = "Z, error"
    for (i, f) in enumerate(filenames):
        fix_name = f"{path}/" + f[0]
        log_name = f"{path}/" + f[1]
        log.info(f"Loading file\t{fix_name}")
        log.info(f"Loading file\t{log_name}")

        variable_list = ["p", "V", "T"]
        C = convert.extract_constants_from_log(log_name)
        log_table = files.load_system(log_name)
        pvt = files.unpack_variables(log_table, log_name, variable_list)

        eq_steps = int(
                C["EQUILL_TIME"]
                /C["DT"]
                /C["THERMO_OUTPUT"]
            )
        cut = int(eq_steps*cut_fraction)
        p = np.mean(pvt[variable_list.index("p")][cut:eq_steps])
        T = np.mean(pvt[variable_list.index("T")][cut:eq_steps])

        N_list = np.array([C["N_L"], C["N_H"]])
        sigma_list = np.array([C["SIGMA_L"], C["SIGMA_H"]])
        x = N_list/np.sum(N_list)
        rho = 6*C["PF"]/np.pi/np.sum(x*np.diag(sigma_list)**3)
        Z = eos.Z_measured_mix(p, rho, T)

        mix_eos_vals = np.zeros(len(mix_eos_list))
        one_eos_vals = np.zeros(len(one_eos_list))
        for (j, eq) in enumerate(mix_eos_list):
            mix_eos_vals[j] = eq(viscosity.get_sigma(sigma_list), x, rho)
        for (j, eq) in enumerate(one_eos_list):
            one_eos_vals[j] = eq(C["PF"])
            
        err = 0.0 # TODO: Compute this
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
    data_shape = (len(filenames)*N, 16)
    data = np.zeros(data_shape)
    pf = np.linspace(0,0.5,N)
    included_masses = np.empty(0)
    included_sigmas = np.empty(0)
    included_numbers = np.empty(0)
    for (i, f) in enumerate(filenames):
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

        # Check if the configuration has already been included,
        # since filenames contains multiple packing fractions.
        if (
            N_list[1] not in included_numbers
            or mass_list[1] not in included_masses
            or sigma_list[1] not in included_sigmas
        ):
            included_numbers = np.append(included_numbers, N_list[1])
            included_masses = np.append(included_masses, mass_list[1])
            included_sigmas = np.append(included_sigmas, sigma_list[1])

            thorne_vals_SPT = np.zeros(N)
            thorne_vals_PY = np.zeros(N)
            thorne_vals_BMCSL = np.zeros(N)
            enskog_vals_1 = np.zeros(N)
            enskog_vals_2 = np.zeros(N)
            SPT_vals = np.zeros(N)
            PY_vals = np.zeros(N)
            CS_vals = np.zeros(N)
            BMCSL_vals = np.zeros(N)
            for j in range(N):
                thorne_vals_SPT[j] = viscosity.thorne(
                    pf[j], x, mass_list, sigma_list, T, rdf=eos.rdf_SPT)
                thorne_vals_PY[j] = viscosity.thorne(
                    pf[j], x, mass_list, sigma_list, T, rdf=eos.rdf_PY_mix)
                thorne_vals_BMCSL[j] = viscosity.thorne(
                    pf[j], x, mass_list, sigma_list, T, rdf=eos.rdf_BMCSL)
                enskog_vals_1[j] = viscosity.enskog(
                    pf[j], sigma_list[0], T, mass_list[0])
                enskog_vals_2[j] = viscosity.enskog(
                    pf[j], sigma_list[1], T, mass_list[1])
                rho = 6*pf[j]/np.pi/np.sum(x*np.diag(sigma_list)**3)
                sigma = viscosity.get_sigma(sigma_list)
                SPT_vals[j] = eos.Z_SPT(sigma, x, rho)
                PY_vals[j] = eos.Z_PY(sigma, x, rho)
                CS_vals[j] = eos.Z_CS(pf[j])
                BMCSL_vals[j] = eos.Z_CS(pf[j])
                # rdf_SPT = ...
                # rdf_PY = ...
                vals = np.array([
                    thorne_vals_SPT[j], 
                    thorne_vals_PY[j], 
                    thorne_vals_BMCSL[j], 
                    enskog_vals_1[j], 
                    enskog_vals_2[j], 
                    SPT_vals[j], 
                    PY_vals[j], 
                    CS_vals[j],
                    BMCSL_vals[j]
                ])

                save.insert_results_in_array(data, vals, C, i*N+j, pf=pf[j])

    save.save_simulation_data(savename, data, 
        data_name="thorne_SPT, thorne_PY, thorne_BMCSL, enskog_1, enskog_2, SPT_EoS, PY_EoS, BMCSL_EoS, CS_EoS"
    )

main()
