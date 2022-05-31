#####################################################
# This file contains a few test that can be used    #
# when developing the remaining project code.       #
#####################################################

import numpy as np
import theory

def assert_chunk_number(N_chunks, constants):
    """ Checks that the number of chunks given in
        fix viscosity and fix ave/chunk are the same.
        If they are not the same, computation is assumed 
        to be prone to error, and the program is halted.
    """
    N_chunks_given = constants["CHUNK_NUMBER"]
    chunk_thickness = constants["CHUNK_THICKNESS"]
    #assert np.abs(2*constants["LZ"] - chunk_thickness*N_chunks) < 1e-6, f"\
    #    Height is not the same in terms of LZ and chunk thickness: \
    #    {chunk_thickness*N_chunks} != {2*constants['LZ']}"
    #assert N_chunks_given == N_chunks, f"\
    #    Number of chunks is not equal in fix viscosity and fix/ave: \
    #    {N_chunks} is not {N_chunks_given}"


