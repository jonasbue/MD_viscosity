import numpy as np
import pandas as pd
import RDF_from_DUMP as dump_to_rdf 

# Given a directory, compute the RDF in the system, during
# equillibration. 


def export_RDF_data(rdf_name, savename):
    """
        Takes a directory with RDF files and creates a csv for use in pgfplots.
        The output data gile contains g(sigma) for all rdfs in the directory.
        The original RDF file must be created with calcRDF from RDF_from_DUMP.
    """

    data = pd.read_csv(rdf_name)
    g_r = np.array(data["g1_ii"])             # Array of rdf measured at different r
    r = np.array(data["r"])             # Array of rdf measured at different r
    g_sigma = np.amax(g_r)              # Value of rdf measured at sigma
    return g_sigma, g_r, r
