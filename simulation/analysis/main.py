import numpy as np
import matplotlib.pyplot as plt

import logfiles
import viscosity

# Convert all files in data to csv format.
logfiles.all_files_to_csv("data")


#----------------------------------------------------------- 
# From viscosity.py
#----------------------------------------------------------- 


packing_list = np.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.5])
#packing_list = np.array([0.5])
#dv_list = np.array([4, 4.8, 4.6, 4, 2, 0.8])
C = {}
for (i, packing) in enumerate(packing_list):
    fix_name = f"data/MP_viscosity_eta_{packing}.profile"
    log_name = f"data/log.eta_{packing}.lammps"

    vx, z = viscosity.get_velocity_profile(fix_name)
    lreg, ureg = viscosity.velocity_profile_regression(vx, z)
    #viscosity.plot_velocity_profile(vx, z, packing)

    #fix_name = f"data/MP_viscosity_eta_{packing}.profile"
    eta, C = viscosity.find_viscosity(log_name, fix_name)
    print(eta)
    viscosity.plot_viscosity(packing, eta)


# Plot theoretical Enskog equation
m, sigma, T, N = C["MASS"], C["SIGMA"], C["TEMP"], C["N"]
pf = np.linspace(0,0.5)
plt.plot(pf, viscosity.enskog(pf, sigma, T, m, k=1.0))
plt.show()


#plt.plot(pf, radial_distribution(pf))
