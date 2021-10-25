#####################################################
# This file contains a few test that can be used    #
# when developing the remaining project code.       #
#####################################################

import numpy as np
import matplotlib.pyplot as plt
import eos
import viscosity

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


def test_thorne():
    # xi is the mole fraction, so np.sum(x) should equal 1.
    # The packing fraction, however, is 
    # a different, independent quantity.

    pf_list = np.linspace(0,0.5)
    thorne_eta_list = np.zeros_like(pf_list)

    x1 = 0.5
    sigma_list = np.array([1,1])
    x = np.array([1-x1,x1])
    m = np.array([1,1])
    T = 1.5
    for i, pf in enumerate(pf_list):
        thorne_eta_list[i] = viscosity.thorne(pf, x, m, sigma_list, T)
    plt.plot(pf_list, thorne_eta_list, label=f"Thorne", linestyle="-")

    enskog_eta_list = viscosity.enskog(
            pf_list,
            np.mean(sigma_list),
            T,
            np.mean(m)
        )
    plt.plot(pf_list, enskog_eta_list, label="Enskog", linestyle="-")
    plt.title("Enskog vs. Thorne with one component")
    plt.xlabel("Packing fraction")
    plt.ylabel("Viscosity")
    plt.legend()
    plt.show()


def test_rdf():
    pf = np.linspace(0.001,0.5)
    sigma_list = np.array([1,2])
    sigma = viscosity.get_sigma(sigma_list, len(sigma_list))
    print(sigma)
    x1 = 0.5
    x = np.array([1-x1,x1])
    m = np.array([1,1])
    rho = 6*pf/np.pi
    plt.plot(pf, eos.Z_CS(pf), label="Carnahan-Starling")
    plt.plot(pf, eos.Z_PY(sigma, x, rho), label="Percus-Yervick", linestyle="-.")
    plt.plot(pf, eos.Z_SPT(sigma, x, rho), label="SPT", linestyle="--")
    plt.title("Comparing equations of state")
    plt.xlabel("Packing fraction")
    plt.ylabel("Compressibility factor")
    plt.legend()
    plt.show()
