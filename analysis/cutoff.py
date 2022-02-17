import matplotlib.pyplot as plt
import termplotlib as tpl
import numpy as np
import pandas as pd
import files
import sys

# Increase font size in plots
font = {
    "size"  : "22",
}
plt.rc("font", **font)

path = "data/processed/"
filenames = ["eos_cut.csv"]


def cutoff_error(path, filename):
    data = pd.read_csv(path+filename, delimiter=", ")
    pf = np.array(data["pf"])
    cut = np.array(data["cut"])
    Z = np.array(data["Z"])

    # Rehape into 2D arrays. The first dimension has different 
    # cut lengths, the second has different packing fractions.
    cut = np.reshape(cut, (len(np.unique(cut)),-1))
    pf = np.reshape(pf, cut.shape)
    Z = np.reshape(Z, cut.shape)

    # The best Z estimate is the one corresponding to the highest cut.
    # This should be the last value, but a more robust approach is possible.
    assert cut[-1,-1] == np.amax(cut), f"{cut[-1,-1]} is not {np.amax(cut)}"
    best_Z = Z[-1,:]
    # Compute the difference (norm) between best_Z and the other values.
    diff = (Z**2 - best_Z[np.newaxis,:]**2)/Z**2
    diff = np.sum(np.sqrt(np.abs(diff)),axis=1)

    # Plot results, using matplotlib or the more esoteric termplotlib.
    fig = tpl.figure()
    fig.plot(cut[:,0], diff)
    fig.show()
    plt.plot(cut[:,0], diff, "o")
    plt.show()




def plot_result(path, filename, x_name, y_name, pltstr="-"):
    #files.get_all_filenames(path)
    data = pd.read_csv(path+filename, delimiter=", ")
    plt.plot(data[x_name], data[y_name], pltstr)
    plt.show()
    
#for filename in filenames:
#    plot_result(path, filename, "pf", "Z", pltstr="o")
cutoff_error(path, filenames[0])
