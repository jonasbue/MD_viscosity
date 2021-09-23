import numpy as np
import matplotlib.pyplot as plt

# Increase font size in plots
font = {
    "size"  : "22",
}
plt.rc("font", **font)

def get_velocity_profile(filename):
    A = np.loadtxt(filename, delimiter = " ")
    vx = A[:,-1]
    z = A[:,1]
    return vx, z

vx, z = get_velocity_profile("MP_viscosity.profile")
plt.plot(vx, z, "o", label="Velocity (average over several time steps)")
plt.xlabel("$v_x$")
plt.ylabel("$z$")
plt.legend(loc="upper right")
plt.show()
