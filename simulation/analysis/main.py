import numpy as np
import matplotlib.pyplot as plt

import logfiles
import viscosity

# Convert all files in data to csv format.
#logfiles.all_files_to_csv("data")


#----------------------------------------------------------- 
# From viscosity.py
#----------------------------------------------------------- 


packing_list = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
#packing_list = np.array([0.5])
C = {}
for (i, packing) in enumerate(packing_list):
    fix_name = f"data/fix.viscosity_eta_{packing}.lammps"
    log_name = f"data/log.eta_{packing}.lammps"

    vx, z = viscosity.get_velocity_profile(fix_name)
    lreg, ureg, zl, zu = viscosity.velocity_profile_regression(vx, z)
    #viscosity.plot_velocity_profile(vx, z, packing)
    #viscosity.plot_velocity_regression(lreg, ureg, zl, zu)
    #plt.show()

    eta, C, eta_max, eta_min = viscosity.find_viscosity(log_name, fix_name)
    #print(eta_max + eta_min)
    viscosity.plot_viscosity(
        C["PF"], 
        eta, 
        eta_max
    )


# Plot theoretical Enskog equation
m, sigma, T, N = C["MASS"], C["SIGMA"], C["TEMP"], C["N"]
pf = np.linspace(0,0.6)
plt.plot(pf, viscosity.enskog(pf, sigma, T, m, k=1.0))
plt.show()


#plt.plot(pf, radial_distribution(pf))
