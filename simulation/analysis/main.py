import numpy as np
import matplotlib.pyplot as plt
import files
import viscosity
import plotting
import eos

# Convert all files in data to csv format.
#files.all_files_to_csv("data")
packing_list = files.find_all_packing_fractions("data")

def main_viscosity():
    C = {}
    PF_list = np.zeros(len(packing_list))
    eta_list = np.zeros(len(packing_list))
    std_err_list = np.zeros((2, len(packing_list)))

    for (i, packing) in enumerate(packing_list):
        fix_name = f"data/fix.viscosity_eta_{packing}.lammps"
        log_name = f"data/log.eta_{packing}.lammps"

        # Plot velocity profiles and regressions of them
        #vx, z = viscosity.get_velocity_profile(fix_name)
        #lreg, ureg, zl, zu = viscosity.velocity_profile_regression(vx, z)
        #ploting.plot_velocity_profile(vx, z, packing)
        #ploting.plot_velocity_regression(lreg, ureg, zl, zu)
        #plt.show()

        # Compute and plot viscosity for all packing fractions
        eta, C, eta_max, eta_min = viscosity.find_viscosity_from_files(
            log_name, fix_name
        )
        PF_list[i] = C["PF"]
        eta_list[i] = np.mean(eta[500:])
        eta_error = np.array([eta_min, eta_max])
        std_err_list[:,i] = np.mean(eta_error[:,500:], axis=1)


    plotting.plot_viscosity(
        packing_list,
        eta_list,
        std_err_list,
    )


    # Plot theoretical Enskog equation
    m, sigma, T, N = C["MASS"], C["SIGMA"], C["TEMP"], C["N"]
    pf = np.linspace(0,0.6)
    plt.plot(pf, viscosity.enskog(pf, sigma, T, m, k=1.0))
    plt.show()


main_viscosity()

def main_equation_of_state():
    # Calculate values of Z from measured p, V and T.
    for eta in eta_list:
        filename = f"log.eta_{eta}.lammps"
        # constants = convert.extract_constants_from_log(filename)
        log_table = files.load_system(filename)
        pvt = files.unpack_varables(log_table, filename, variable_list)
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
