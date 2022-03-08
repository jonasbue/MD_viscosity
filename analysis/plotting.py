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

# Increase font size in plots
font = {
    "size"  : "22",
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

def plot_result(path, filename, x_name, y_name, theory_name, *args, pltstr="-"):
    #files.get_all_filenames(path)
    data = pd.read_csv(path+filename, delimiter=", ")
    x = np.zeros_like(data[x_name])
    y = np.zeros_like(x)
    t = np.zeros_like(x)
    variable_names = [          "pf",   "N",    "m",    "T",    "sigma", "cutoff"]
    system_config = np.array([2.0e-1, 3.0e+3, 1.0e+0, 1.5e+0, 1.0e+00, 6.75e+0])
    # Iterate through all simulations, and save those 
    # that match the criteria provided in system_config.
    x_index = variable_names.index(x_name)
    #y_index = variable_names.index(y_name)
    indices = [variable_names.index(el) for el in args]
    indices.append(x_index)
    var_indices = np.delete(np.arange(len(system_config)), indices)
    print(var_indices)
    for i in range(len(data)):
        row = data.iloc[i]
        C = row[:len(system_config)].to_numpy()
        conf = C[var_indices] == system_config[var_indices]
        if conf.all():
            print(i)
            x[i] = row[x_name]
            y[i] = row[y_name]
            t[i] = row[theory_name]
    # Remove unused space in the arrays.
    x = np.trim_zeros(x)
    y = np.trim_zeros(y)
    t = np.trim_zeros(t)
    n = len(np.unique(x))
    if len(y) != n or len(t) != n:
        "WARNING: Some simulations overlap in parameter space."
    plt.title(f"N = {system_config[1]}, T = {system_config[3]}, $\sigma$ = {system_config[4]}")
    plt.plot(x, y, pltstr, label=f"{y_name}")
    plt.plot(x, t, "kx", label=theory_name)
    plt.legend()
    #plt.plot(x, t, "x")
    #plt.plot(x, np.zeros_like(x), ":"), 
    plt.show()
    
if "eos" in sys.argv:
    filenames = ["eos_lj.csv"] 
    for filename in filenames:
        plot_result(path, filename, "pf", "Z", "EOS_LJ", "T", "sigma", pltstr="o")
if "visc" in sys.argv:
    filenames = ["visc_lj.csv"]
    for filename in filenames:
        plot_result(path, filename, "pf", "viscosity", "enskog_RDF_PY", pltstr="k-")
    #plot_result(path, filename, "pf", "T", pltstr="o", rowsarg="T", rowsval=1.0)
