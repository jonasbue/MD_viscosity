import theory
import numpy as np

def get_eos_list():
    return [
        theory.Z_CS,
        theory.Z_kolafa,
        theory.Z_gottschalk,
        theory.Z_thol,
        theory.Z_mecke,
        theory.Z_hess
    ]


def get_rdf_list():
    return [theory.rdf_CS, theory.rdf_LJ]


def get_helmholtz_list():
    #return [theory.F_kolafa, theory.F_thol, theory.F_mecke, theory.F_gottschalk, theory.F_hess, theory.rdf_LJ]
    return [theory.F_CS, theory.F_kolafa, theory.F_thol, theory.F_mecke, theory.F_gottschalk, theory.F_hess, theory.rdf_LJ]

def get_fitted_collision_integrals():
    """ These are guesses. """
    #return [1.4, 0.8, 0.8, 1.4, 0.8, 0.9, 0.9]
    HS = 1 # Collision integral is excactly one.
    # Names are:    CS Kolafa Thol Meck Gott Hess Morsali
    #return np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])*HS
    return np.array([1.0, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2])*HS

def get_method_list():
    """ Out of date """
    return ["kolafa", "kolafa", "thol", "mecke", "gottschalk", "kolafa"]

def get_savename(body, save_dir, savepath):
    return f"{save_dir}{body}_{savepath}.csv"
