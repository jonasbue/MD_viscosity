#########################################################
# This file contains all relevant plotting functions,   #
# wrapped to reduce repetitions and manual writing.     #
#########################################################

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import regression
import viscosity
import eos

# Increase font size in plots
font = {
    "size"  : "22",
}
plt.rc("font", **font)


def plot_result_vs_thorne(
        eta_list, 
        packing_list, 
        error_list,
        C, 
        mix=True, 
        relative=False
    ):
    m = np.array([C["MASS_L"], C["MASS_H"]])
    sigma = np.array([C["SIGMA_L"], C["SIGMA_H"]])
    T = C["TEMP"]
    N = np.array([C["N_L"], C["N_H"]])
    x = N/np.sum(N)
    pf = np.linspace(0,0.5)

    # Plot the measured viscosity
    plot_viscosity(
        packing_list,
        eta_list,
        error_list,
    )

    # Plot the theoretical viscosity
    thorne_eta_list = np.zeros_like(pf)
    for i, pfi in enumerate(pf):
        thorne_eta_list[i] = viscosity.thorne(pfi, x, m, sigma, T)
    #plt.plot(pf, thorne_eta_list, 
    #    label="Thorne equation, two components")

    # Plot the Enskog equation as well, for comparison
    enskog_eta_list = viscosity.enskog(
            pf,
            np.mean(sigma),
            T,
            np.mean(m)
        )
    #plt.plot(pf, enskog_eta_list, 
    #    label="Enskog equation, one component")
    plt.legend()
    plt.show()

def plot_result_vs_enskog(
        eta_list, 
        packing_list, 
        error_list,
        C, 
        mix=True, 
        relative=False
    ):
    # Extract constants from C
    m, sigma, T, N = C["MASS"], C["SIGMA"], C["TEMP"], C["N"]
    pf = np.linspace(0,0.5)

    if relative==False:
        # Plot the measured viscosity.
        plot_viscosity(
            packing_list,
            eta_list,
            error_list,
        )
        # Plot the Enskog equation
        plt.plot(pf, viscosity.enskog(pf, sigma, T, m, k=1.0), 
                label="Enskog viscosity, one component"
        )
    else:
        # Divide by Enskog to get more readable plots
        plot_viscosity(
            packing_list,
            eta_list/viscosity.enskog(packing_list, sigma, T, m),
            error_list/viscosity.enskog(packing_list, sigma, T, m),
            label="Measured viscosity, two components"
        )
        # Plot the Enskog equation, which is one in this case.
        plt.plot(pf, np.ones_like(pf))
    plt.legend()
    plt.show()


def plot_velocity_regression(reg, z, slab=""):
    plt.plot(
        reg.slope*z + reg.intercept,
        z,
        label=f"{slab} slab regression"
    )


def plot_velocity_regression_error(reg, z):
    plt.plot(
        (reg.slope+reg.stderr)*z 
        + (reg.intercept+reg.intercept_stderr),
        z,
        label="Upper error bound",
        linestyle="--"
    )
    plt.plot(
        (reg.slope-reg.stderr)*z 
        + (reg.intercept-reg.intercept_stderr),
        z,
        label="Lower error bound",
        linestyle=":"
    )


def plot_velocity_profile(vx, z, packing=None):
    """ Plots velocity of particles in chunks,
        as computed by lammps script.
        Output will be a scatter plot showing the 
        average velocity vx(z) at many different times.
        Input:
            vx:         np.array of velocities. One-dimensional.
            z:          np.array of velocities. One-dimensional.
            packing:    Packing fraction, float. Only used for labeling figure.
    """
    fig_title = f"Velocity profile, $\\xi = {packing}$"
    plt.plot(
        vx, 
        z,
        "o", 
        label=fig_title,
    )
    plt.xlabel("$v_x$")
    plt.ylabel("$z$")
    plt.legend(loc="upper right")


def plot_viscosity(packing, eta, std_err=None):
    """ Plots viscosity for a list of packing fractions,
        including error bars if provided.
        Input:
            packing:    np.array of packing fractions. One-dimensional.
            eta:        np.array of computed viscosities. One-dimensional.
            std_err:    np.array of uncertainty estimate. Two-dimensional.
                        First row is uncertainty below eta,
                        and second row is uncertainty above eta.
    """
    plt.errorbar(
        packing,
        eta,
        yerr = std_err,
        fmt="o",
    )
    plt.xlabel("Packing fraction")
    plt.ylabel("Viscosity")


def plot_Z(p, V, N, T, pf):
    """ Plots compressibility factor of the system,
        from measurred values of p, V and T,
        for a given packing fraction.
        Inputs:
            p:      pressure, one-dimensional array.
            V:      volume, one-dimensional array.
            T:      temperature, one-dimensional array.
            pf:    packing fraction, float. 
    """
    plt.plot(
        np.full_like(p, pf),
        eos.Z_measured(p, V, N, T),
        "o", label=f"$\\xi$ = {pf}"
    )                          

def plot_velocity_profile_from_file(fix_name):
    vx, z = regression.get_velocity_profile(fix_name)
    vxl, vxu, zl, zu = regression.isolate_slabs(vx, z)
    lreg = regression.velocity_profile_regression(vxl, zl)
    ureg = regression.velocity_profile_regression(vxu, zu)

    plot_velocity_profile(vxl[::6], zl[::6])
    plot_velocity_profile(vxu[::6], zu[::6])
    plot_velocity_regression(lreg, zl, slab="Lower")
    plot_velocity_regression_error(lreg, zl)
    plot_velocity_regression(ureg, zu, slab="Upper")
    plot_velocity_regression_error(ureg, zu)
    plt.legend(loc="center left")
    plt.show()

