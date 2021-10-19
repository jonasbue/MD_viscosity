#####################################################
# This file contains a few test that can be used    #
# when developing the remaining project code.       #
#####################################################

import eos
import numpy as np

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
    assert N_chunks_given == N_chunks, f"\
        Number of chunks is not equal in fix viscosity and fix/ave: \
        {N_chunks} is not {N_chunks_given}"

def test_mix():
    sigma = np.array([[1.0,1.0],[1.0,1.0]])
    x = np.array([1.0, 0.0])
    pf = 0.4
    rho = 6*pf/np.pi
    print(eos.rdf_PY(pf))
    print(eos.Z_CS(pf))
    g = eos.rdf_SPT(sigma, x, rho, 1,1)
    print(g)
    g = eos.rdf_SPT(sigma, x, rho, 1,2)
    print(g)
    g = eos.rdf_SPT(sigma, x, rho, 2,2)
    print(g)
    g = eos.rdf_PY_mix(sigma, x, rho, 1,1)
    print(g)
    g = eos.rdf_PY_mix(sigma, x, rho, 1,2)
    print(g)
    g = eos.rdf_PY_mix(sigma, x, rho, 2,2)
    print(g)
    Z = eos.Z_SPT(sigma, x, rho)
    print(Z)
    Z = eos.Z_PY(sigma, x, rho)
    print(Z)

