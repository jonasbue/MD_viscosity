import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import convert_LAMMPS_output as convert

# Increase font size in plots
font = {
    "size"  : "22",
}
plt.rc("font", **font)

def Z(p, V, N, T, k=1):
    """ Computes compressibility factor of a system.
        Inputs:
            p:      pressure, float.
            V:      volume, float.
            N:      number of particles, float.
            T:      temperature, float.
            k:      boltzmann constant, float.
                    Default value is 1, which is the
                    value in lj reduced units.
        Returns:
            Z:      Compressibility factor.
            
    """
    return p*V/(N*k*T)

def Z_Carnahan_Starling(n):
    """ Returns expected compressibility factor
        based on the Carnahan-Starling EoS,
        for a given packing fraction n.
        Inputs:
            n:  packing franction of system, float.
        Returns:
            Z:  compressibility factor, float.
    """
    return (1 + n + n**2 - n**3) / (1 - n)**3

def load_system(filename):
    """ Loads a LAMMPS logfile, and returns
        thermodynamical variables from it,
        as a numpy array.
        Inputs:
            filename:   name of logfile, string.
        Returns:
            log:        Array of logged values from LAMMPS.
                        First index selects the variable,
                        second index selects the timestep.
    """
    convert.convert_log_to_csv(filename)
    log = pd.read_csv(
        filename + ".csv", 
        skiprows=2, 
        delimiter=","
    )
    log = np.array(log)
    return log.transpose()

def plot_Z(p, V, T, eta):
    """ Plots compressibility factor of the system,
        from measurred values of p, V and T,
        for a given packing fraction.
        Inputs:
            p:      pressure, one-dimensional array.
            V:      volume, one-dimensional array.
            T:      temperature, one-dimensional array.
            eta:    packing fraction, float. 
    """
    plt.plot(
        np.full_like(p, eta),
        Z(p, V, N, T),
        "o", label=f"$\eta$ = {eta}"
    )                          

def unpack_global_varables(log_table):
    """ Unpacks temperature, pressure and volume 
        from log table generated woth load_system.
        NOTE: Super unsafe. 
        Note also the order of the variables.
    """
    # TODO: Use header to make this safer.
    p = log_table[6]
    V = log_table[7]
    T = log_table[5]
    return p, V, T

N = 1000
eta_list = np.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.5])

# Calculate values of Z from measured p, V and T.
for eta in eta_list:
    log_table = load_system(f"log.eta_{eta}.lammps")
    p, V, T = unpack_global_varables(log_table)
    plot_Z(p, V, T, eta)

# Plot theoretical values, from CS-EoS
eta_range = np.linspace(0, 0.5)
#plt.plot(
#    eta_range, 
#    Z_Carnahan_Starling(eta_range), 
#    "-", 
#    label="Carnahan-Starling EoS",
#    linewidth=3
#)

# Show figure
plt.xlabel("Packing fraction")
plt.ylabel("Compressibility factor")
plt.legend()
plt.show()
