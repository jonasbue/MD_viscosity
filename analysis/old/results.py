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
import viscosity
import muller_plathe
import eos
import save
import utils
import rdf
import convert_LAMMPS_output as convert
import RDF_from_DUMP as dump_to_rdf 
import block_average

sysargs = sys.argv
log = logging.getLogger()
log.addHandler(logging.StreamHandler(sys.stdout))
if "debug" in sysargs:
    # logging.Debug is more correct to use, 
    # but info is cleaner.
    log.setLevel(logging.INFO)


## Rename to convert_something_something
def main():
    N = 1
    cut_fraction = 0.3
    step = 20
    per_time=False

    data_path_list = ["./data/run_test"]
    save_path_list = ["savename"]

    for path, savename in zip(data_path_list, save_path_list):
        if "convert" in sysargs:
            files.all_files_to_csv(path)
        save_dir = "./data/processed/"
        filenames = files.get_all_filenames(path)
        print(path)
        print("Number of files: ", len(filenames))
        packing_list = files.find_all_packing_fractions(path)
        #print("Number of packing fractions:", len(packing_list))
        #print("Number of configurations:", len(filenames)//len(packing_list))
        filenames = files.sort_files(filenames, packing_list)

        save_viscosity(cut_fraction, path, filenames, savename=f"{save_dir}visc_{savename}.csv", per_time=per_time, step=step)
        save_eos(path, filenames, cut_fraction, N, savename=f"{save_dir}eos_{savename}.csv")
        save_theory(path, filenames, savename=f"{save_dir}theory_{savename}.csv")
        #save_rdf(path, filenames, savename=f"{save_dir}rdf_{savename}.csv")


def save_viscosity(cut_fraction, path, filenames, savename, per_time=False, step=20):
    rdf_list = [eos.rdf_SPT, eos.rdf_PY_mix, eos.rdf_BMCSL]
    one_comp_rdf_list = [eos.rdf_CS, eos.rdf_PY]
    columns = len(save.get_system_config()) + 2 + len(rdf_list) + 2*len(one_comp_rdf_list)
    data = np.zeros((len(filenames),columns))
    data_name = "viscosity, error"
    for (i, f) in enumerate(filenames):
        utils.status_bar(i, len(filenames), fmt="train")
        fix_name = f"{path}/" + f[0]
        log_name = f"{path}/" + f[1]
        log.info(f"Loading file\t{fix_name}")
        log.info(f"Loading file\t{log_name}")

        # Compute and plot viscosity for all packing fractions
        eta, C, error = muller_plathe.find_viscosity_from_file(
            log_name, fix_name, cut_fraction, per_time, step
        )

        thorne_values = np.zeros(len(rdf_list))
        enskog_values = np.zeros((len(one_comp_rdf_list),2))
        #C["SIGMA_H"] = 15.4
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
        utils.status_bar(i, len(filenames), fmt="arrow")
        fix_name = f"{path}/" + f[0]
        log_name = f"{path}/" + f[1]
        log.info(f"Loading file\t{fix_name}")
        log.info(f"Loading file\t{log_name}")

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

        N_list = np.array([C["N_L"], C["N_H"]])
        sigma_list = np.array([C["SIGMA_L"], C["SIGMA_H"]])
        x = N_list/np.sum(N_list)
        rho = 6*C["PF"]/np.pi/np.sum(x*np.diag(sigma_list)**3)
        Z = eos.Z_measured_mix(p, rho, T)
        std = block_average.get_block_average(Z)
        Z = np.mean(Z)

        mix_eos_vals = np.zeros(len(mix_eos_list))
        one_eos_vals = np.zeros(len(one_eos_list))
        for (j, eq) in enumerate(mix_eos_list):
            mix_eos_vals[j] = eq(viscosity.get_sigma(sigma_list), x, rho)
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
            BMCSL_vals[j] = eos.Z_BMCSL(sigma, x, rho)
            CS_vals[j] = eos.Z_CS(pf[j])

            # TODO: Fix indices. Probably separate file?
            rdf_SPT[j] = eos.rdf_SPT(sigma, x, rho, 0, 0)
            rdf_PY[j] = eos.rdf_PY_mix(sigma, x, rho, 0, 0)
            rdf_BMCSL[j] = eos.rdf_BMCSL(sigma, x, rho, 0, 0)
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
    rdf_list = [eos.rdf_PY_mix, eos.rdf_SPT, eos.rdf_BMCSL]
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
        sigma       = viscosity.get_sigma(sigma_list)
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
