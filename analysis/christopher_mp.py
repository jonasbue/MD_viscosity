#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import MP_from_dump_and_plotting as mp
import viscosity
import files
import eos
import save
import utils
import convert_LAMMPS_output as convert

data_path_list = ["small_box/"]
save_path_list = ["christopher"]


for path, savename in zip(data_path_list, save_path_list):
    filenames = files.get_all_filenames(path)
    packing_list = files.find_all_packing_fractions(path)
    filenames = files.sort_files(filenames, packing_list)

    rdf_list = [eos.rdf_SPT, eos.rdf_PY_mix, eos.rdf_BMCSL]
    one_comp_rdf_list = [eos.rdf_CS, eos.rdf_PY]
    columns = len(save.get_system_config()) + 2 + len(rdf_list) + 2*len(one_comp_rdf_list)
    data = np.zeros((len(filenames),columns))
    data_name = "viscosity, error"
    for (i, f) in enumerate(filenames):
        utils.status_bar(i, len(filenames), fmt="train")

        eta, error, shear, shear_err, P = mp.MullerPlathe(path, f[1], measurment_time="RUN_TIME", box_base="LX", box_height="LZ")

        C = convert.extract_constants_from_log(path + f[1])
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


"""
for path, savename in zip(data_path_list, save_path_list):
    filenames = files.get_all_filenames(path)
    packing_list = files.find_all_packing_fractions(path)
    filenames = files.sort_files(filenames, packing_list)
    N = len(filenames)
    eta_list = np.zeros(N)
    err_list = np.zeros(N)
    for (i, f) in enumerate(filenames):
        eta, error, shear, shear_err, P = mp.MullerPlathe(path, f[1], measurment_time="RUN_TIME", box_base="LX", box_height="LZ")
        eta_list[i] = eta
        err_list[i] = error
    plt.errorbar(packing_list, eta_list, err_list)
    plt.show()

"""
