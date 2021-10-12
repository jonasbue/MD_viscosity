import matplotlib
import matplotlib.pyplot as plt

# Increase font size in plots
font = {
    "size"  : "22",
}
plt.rc("font", **font)
matplotlib.rcParams['agg.path.chunksize'] = 1000000


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
    #plt.fill_between(
    #    (lower_reg.slope+lower_reg.stderr)*z_lower 
    #    + (lower_reg.intercept+lower_reg.intercept_stderr),
    #    lower_reg.slope*z_lower + lower_reg.intercept,
    #    #(lower_reg.slope-lower_reg.stderr)*z_lower 
    #    #+ (lower_reg.intercept-lower_reg.intercept_stderr),
    #    z_lower
    #)


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