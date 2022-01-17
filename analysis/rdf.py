import numpy as np
import pandas as pd
import RDF_from_DUMP as dump_to_rdf 

# Given a directory, compute the RDF in the system, 
# as a function of time, during equillibration. 


def export_RDF_data(rdf_name, savename):
    """
        Takes a directory with RDF files and creates a csv for use in pgfplots.
        The output data gile contains g(sigma) for all rdfs in the directory.
        The original RDF file must be created with calcRDF from RDF_from_DUMP.
    """

    data = pd.read_csv(rdf_name)
    g_r = np.array(data["g1"])      # Array of rdf measured at different r
    std = np.array(data["err1"])
    r = np.array(data["r"])         # Array of r
    g_sigma = np.amax(g_r)          # Value of rdf measured at sigma
    g_sigma_index = np.argmax(g_r)  # Index of g_sigma
    std = std[g_sigma_index]
    return g_sigma, g_r, r, std

def show_an_rdf(rdf_name):
    data = pd.read_csv(rdf_name)
    g_r = np.array(data["g1"])      # Array of rdf measured at different r
    std = np.array(data["err1"])
    r = np.array(data["r"])         # Array of r
