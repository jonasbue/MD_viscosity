import numpy as np
import matplotlib.pyplot as plt
font = {
    "family": "normal",
    "size"  : "22",
}
plt.rc("font", **font)

def Z(p, V, N, T, k=1):
    return p*V/(N*k*T)

def Z_theoretical(n):
    # n is the packing fraction of the system.
    # Using n instead of eta, for readability.
    return (1 + n + n**2 - n**3) / (1 - n)**3

def load_pressure(eta):
    # Assumes filename on the form
    #   eta.csv
    # with eta being a decimal number.
    filename = "data/{:.0e}.csv".format(eta)
    pressures = np.loadtxt(filename)
    return pressures

def plot_Z(P, eta):
    plt.plot(
        np.full(len(P[:,1]), eta), 
        Z(P[:,1], (2*P[:,3])**3, N, P[:,2]),
        "o", label=f"$\eta$ = {eta}"
    )                          

N = 1000
eta_0 = 0.01
eta_1 = 0.1
eta_2 = 0.2
eta_3 = 0.3
eta_4 = 0.4
eta_5 = 0.5
eta_range = np.linspace(0, 0.5)

p0 = load_pressure(eta_0)
p1 = load_pressure(eta_1)
p2 = load_pressure(eta_2)
p3 = load_pressure(eta_3)
p4 = load_pressure(eta_4)
p5 = load_pressure(eta_5)

plot_Z(p0, eta_0)
plot_Z(p1, eta_1)
plot_Z(p2, eta_2)
plot_Z(p3, eta_3)
plot_Z(p4, eta_4)
plot_Z(p5, eta_5)

plt.plot(
    eta_range, 
    Z_theoretical(eta_range), 
    "-", 
    label="Carnahan-Starling EoS",
    linewidth=3
)
plt.xlabel("Packing fraction")
plt.ylabel("Compressibility factor")
plt.legend()
plt.show()
