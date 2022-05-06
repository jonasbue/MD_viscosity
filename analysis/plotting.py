#########################################################
# This file contains all relevant plotting functions,   #
# wrapped to reduce repetitions and manual writing.     #
#########################################################

#import matplotlib
import matplotlib.pyplot as plt
import termplotlib as tpl
import numpy as np
import pandas as pd
#import regression
#import viscosity
#import eos
import files
import save
import sys
import units
import theory

# Increase font size in plots
font = {
    "size"  : "18",
}
plt.rc("font", **font)

filenames = []
path = ""
for arg in sys.argv:
    if ".csv" in arg:
        filenames.append(arg)
    if "path" in arg:
        path = arg[arg.index("-")+1:]

path = "data/processed/"
data_name = "lj"

def plot_result(path, filename, x_name, y_name, theory_name, system_config, *args, pltstr="-", norm=True):
    #files.get_all_filenames(path)
    data = pd.read_csv(path+filename, delimiter=", ", engine="python")
    x = np.zeros_like(data[x_name])
    y, t, err = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
    variable_names = get_var_names()

    # Iterate through all simulations, and save those 
    # that match the criteria provided in system_config.
    x_index = variable_names.index(x_name)
    #y_index = variable_names.index(y_name)
    indices = [variable_names.index(el) for el in args]
    indices.append(x_index)
    var_indices = np.delete(np.arange(len(system_config)), indices)
    N, m, T, s, c = 0, 0, 0, 0, 0
    for i in range(len(data)):
        row = data.iloc[i]
        C = row[:len(system_config)].to_numpy()
        conf = C[var_indices] == system_config[var_indices]
        if conf.all():
            x[i] = row[x_name]
            y[i] = row[y_name]
            err[i] = row["error"]
            if theory_name:
                t[i] = row[theory_name]
            else:
                t[i] = np.ones_like(x[i])
            N, m, T, s, c = int(C[1]), C[2], C[3], C[4], C[5]
    # Remove unused space in the arrays.
    x, y, t, err = np.trim_zeros(x), np.trim_zeros(y), np.trim_zeros(t), np.trim_zeros(err)
    n = len(np.unique(x))
    if len(y) != n or len(t) != n:
        print("WARNING: Some simulations overlap in parameter space.")
    if norm:
        y = y/t
        t = t/t
        err = err/t
    plt.title(f"N = {N}, m = {m}, T = {T}, $\sigma$ = {s}")
    plt.errorbar(x, y, yerr=err, fmt=pltstr, label=f"{y_name}, numerical")
    if theory_name:
        plt.plot(x, t, "kx", label=theory_name+", theoretical")
    #plt.legend()
    #plt.plot(x, t, "x")
    #plt.plot(x, np.zeros_like(x), "0"), 
    savename = f"figures/lj_{y_name}({x_name})_N_{N}_m_{m}_T_{T}_sigma_{s}.png"
    plt.savefig(savename)
    #plt.show()
    

def plot_vs_literature(system_config, fluid_name, data_path):
    # Non-general code. Under construction.
    variable_names = get_var_names()
    fluid_data = pd.read_csv(data_path, sep=", ", engine="python")
    u = 1.66e-27

    sigma = system_config[4]
    m = system_config[2]
    #epsilon = 1.0
    T = system_config[3]
    #N = system_config[1]
    lj_units = units.make_unit_dict(T=T, m=system_config[2], pf=system_config[0])

    row = fluid_data[fluid_data["gas"] == fluid_name]
    sigma_fluid     = row["rw"].values[0]*2
    m_fluid         = row["m"].values[0]*u
    epsilon_fluid   = row["epsilon"].values[0]
    real_conf = units.lj_to_real_units(sigma_fluid, m_fluid, epsilon_fluid, lj_units)
    lj_conf = units.real_to_lj_units(sigma_fluid, m_fluid, epsilon_fluid, real_conf)

    # GOAL: 
    # Take a data file for some fluid, with Z, p, T etc. and convert between
    # LJ and SI units.


def plot_literature_results(filename, system_config):
    data = pd.read_csv(filename, sep=", ", engine="python")
    data = data.sort_values(by="rho")
    data = data[data["T"]==system_config[get_var_names().index("T")]]
    plt.plot(data["rho"]*np.pi/6, data["Z"], "x-", label="Thol, 2016")


def get_var_names():
    variable_names = ["pf", "N", "m", "T", "sigma", "cutoff"]
    return variable_names


def search_for_configs(filename):
    filename = f"data/processed/{filename}"
    configs = pd.read_csv(filename, sep=", ", engine="python")
    cols = configs.columns[len(system_config):]
    configs = configs.drop(columns=cols)
    configs = configs.drop_duplicates(subset=configs.columns[1:len(system_config)-1])
    return configs


#variable_names = [         "pf",   "N",    "m",    "T", "sigma", "cutoff"]
system_config = np.array([4.0e-1, 3.0e+3, 1.0e+0, 1.5e+00, 1.0e+00, 6.75e+0])
lit_filename = "data/literature/thol_2016.csv"
# These can all be done by one single function
if "eos" in sys.argv:
    filenames = [f"eos_{data_name}.csv"] 
    for filename in filenames:
        system_configs = search_for_configs(filename)
        for i in range(len(system_configs.index)):
            system_config = np.array(system_configs.iloc[i])
            plot_result(path, filename, "pf", "Z", "EOS_LJ", system_config, "cutoff", pltstr="o-", norm=False)
            plot_literature_results(lit_filename, system_config)
            plt.legend()
            plt.show()
if "rdf" in sys.argv:
    filenames = [f"rdf_{data_name}.csv"] 
    for filename in filenames:
        system_configs = search_for_configs(filename)
        for i in range(len(system_configs.index)):
            system_config = np.array(system_configs.iloc[i])
            plot_result(path, filename, "pf", "g_sigma", "RDF_LJ", system_config, "cutoff", pltstr="o-", norm=False)
            #plot_literature_results(lit_filename, system_config)
            plt.legend()
            plt.show()
if "visc" in sys.argv:
    filenames = [f"visc_{data_name}.csv"] 
    for filename in filenames:
        system_configs = search_for_configs(filename)
        for i in range(len(system_configs.index)):
            system_config = np.array(system_configs.iloc[i])
            plot_result(path, filename, "pf", "viscosity", "enskog_RDF_PY", system_config, "cutoff", pltstr="o-", norm=False)
            #plot_literature_results(lit_filename, system_config)
            plt.legend()
            plt.show()
if "lit" in sys.argv:
    plot_literature_results(lit_filename, system_config)
    plt.legend()
    plt.show()
if "data" in sys.argv:
    filenames = [f"eos_{data_name}.csv"] 
    for filename in filenames:
        #plot_result(path, filename, "pf", "Z", "EOS_LJ", system_config, "sigma", pltstr="o", norm=False)
        plot_vs_literature(system_config, "Ar", "real_fluids.csv")
        plt.legend()
        plt.show()
if "all" in sys.argv:
    filenames = [f"visc_{data_name}.csv"] 
    for filename in filenames:
        system_configs = search_for_configs(filename)
        for i in range(len(system_configs.index)):
            system_config = np.array(system_configs.iloc[i])
            plot_result(path, filename, "pf", "viscosity", "enskog_RDF_LJ", system_config, "cutoff", pltstr="o-", norm=False)
            #plot_literature_results(lit_filename, system_config)
            plt.legend()
            plt.show()
if "rdf-of-r" in sys.argv:
    path = f"data/{data_name}/"
    filenames = files.get_all_filenames(path)[:,3]
    for filename in filenames:
        f = path+filename
        system_info = files.read_filename(path+f)
        data = pd.read_csv(f, delimiter=", ", engine="python")
        r = np.array(data["r"])
        rdf = np.array(data["g"])
        #err = np.array(data["err1"])
        plt.plot(r, rdf)
        plt.legend("$g(r)$")
        plt.title(f"N = {system_info['N']}, T = {system_info['temp']}, pf = {system_info['pf']}")
        savename = f"figures/rdf/{filename[:-4]}.png"
        plt.savefig(savename)
        plt.close()
if "vel" in sys.argv:
    path = f"data/{data_name}/"
    filenames = files.get_all_filenames(path)[:,4]
    for f in filenames:
        system_info = files.read_filename(path+f)
        data = pd.read_csv(path+f, delimiter=", ", engine="python")
        vx = np.array(data["vx"])
        z = np.array(data["z"])
        reg_l = np.array(data["reg_lower"])
        reg_u = np.array(data["reg_upper"])
        plt.plot(vx, z, "o")
        plt.plot(reg_l, z, "-")
        plt.plot(reg_u, z, "-")
        plt.legend("$v_x$")
        plt.xlabel("$v_x$")
        plt.ylabel("$z$")
        plt.title(f"N = {system_info['N']}, T = {system_info['temp']}, pf = {system_info['pf']}")
        savename = f"figures/vel/{f[:-4]}.png"
        plt.savefig(savename)
        plt.close()
