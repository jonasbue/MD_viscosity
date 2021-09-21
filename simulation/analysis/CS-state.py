#########################################################
# This script takes a log file from LAMMPS and plots    #
# its compressibility factor vs. the Carnahan-Starling  #
# equation of state.                                    #
#########################################################

import numpy as np
import matplotlib.pyplot as plt
import logfiles
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



N = 1000
eta_list = np.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.5])
variable_list = ["p", "V", "T"]


# Calculate values of Z from measured p, V and T.
for eta in eta_list:
    filename = f"log.eta_{eta}.lammps"
    constants = convert.extract_constants_from_log(filename)
    print(constants)
    log_table = logfiles.load_system(filename)
    pvt = logfiles.unpack_varables(log_table, filename, variable_list)
    plot_Z(
        np.mean(pvt[variable_list.index("p")]), 
        np.mean(pvt[variable_list.index("V")]), 
        np.mean(pvt[variable_list.index("T")]), 
        eta
    )

# Plot theoretical values, from CS-EoS
eta_range = np.linspace(0, 0.5)
plt.plot(
    eta_range, 
    Z_Carnahan_Starling(eta_range), 
    "-", 
    label="Carnahan-Starling EoS",
    linewidth=3
)

# Show figure
plt.xlabel("Packing fraction")
plt.ylabel("Compressibility factor")
plt.legend()
plt.show()
