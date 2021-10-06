import numpy as np
import matplotlib.pyplot as plt

import logfiles
import viscosity

# Convert all files in data to csv format.
#logfiles.all_files_to_csv("data")
packing_list = logfiles.find_all_packing_fractions("data")
print(packing_list)


#----------------------------------------------------------- 
# From viscosity.py
#----------------------------------------------------------- 


#packing_list = np.array([0.5])
C = {}
PF_list = np.zeros(len(packing_list))
eta_list = np.zeros(len(packing_list))
std_err_list = np.zeros((2, len(packing_list)))
for (i, packing) in enumerate(packing_list):
    fix_name = f"data/fix.viscosity_eta_{packing}.lammps"
    log_name = f"data/log.eta_{packing}.lammps"

    vx, z = viscosity.get_velocity_profile(fix_name)
    lreg, ureg, zl, zu = viscosity.velocity_profile_regression(vx, z)
    #viscosity.plot_velocity_profile(vx, z, packing)
    #viscosity.plot_velocity_regression(lreg, ureg, zl, zu)
    #plt.show()

    eta, C, eta_max, eta_min = viscosity.find_viscosity(log_name, fix_name)
    PF_list[i] = C["PF"]
    eta_list[i] = np.mean(eta[500:])
    eta_error = np.array([eta_min, eta_max])
    std_err_list[:,i] = np.mean(eta_error[:,500:], axis=1)


viscosity.plot_viscosity(
    packing_list,
    eta_list,
    std_err_list,
)


# Plot theoretical Enskog equation
m, sigma, T, N = C["MASS"], C["SIGMA"], C["TEMP"], C["N"]
pf = np.linspace(0,0.6)
plt.plot(pf, viscosity.enskog(pf, sigma, T, m, k=1.0))
plt.show()


#plt.plot(pf, radial_distribution(pf))
