import numpy as np
import matplotlib.pyplot as plt
import theory

def test_Z():
    pf = np.linspace(0.01, 0.5)
    T = 2.0
    #T = np.linspace(0.7,1.5)
    #pf = 0.3
    sigma = np.array([1.0])
    sigma = theory.get_sigma(sigma)
    x = np.array([1.0])
    rho = 6*pf/np.pi/np.sum(x*np.diag(sigma)**3)
    #plt.title("CS EoS")
    #plt.legend()
    #plt.show()

    #plt.plot(pf, theory.Z_SPT(sigma, x, rho), label="SPT EoS")
    #plt.plot(pf, theory.Z_PY(sigma, x, rho), label="PY EoS")
    #plt.plot(pf, theory.Z_BMCSL(sigma, x, rho), label="BMCSL EoS", linestyle="--")

    #t = theory.Z_kolafa(sigma, x, rho, temp=T)
    t = np.ones_like(pf)
    #plt.plot(pf, theory.Z_CS(sigma, x, rho)/t, label="CS EOS", linestyle=":")
    plt.plot(pf, theory.Z_BN(sigma, x, rho, temp=T)/t, label="BN EOS", linestyle=":")
    #plt.plot(pf, theory.Z_kolafa(sigma, x, rho, temp=T)/t, 
    #    "k:", label="Kolafa Z")
    #plt.plot(pf, theory.Z_gottschalk(sigma, x, rho, temp=T)/t,
    #    "g:", label="Gottschalk Z")
    #plt.plot(pf, theory.Z_thol(sigma, x, rho, temp=T)/t, 
    #    "c:", label="Thol Z")
    #plt.plot(pf, theory.Z_mecke(sigma, x, rho, temp=T)/t, 
    #    "m:", label="Mecke Z")
    plt.plot(pf, theory.Z_hess(sigma, x, rho, temp=T)/t, 
        "b:", label="Hess Z")
    #plt.ylim((-2.0, 5.0))
    #plt.title("Lennard-Jones Zs")
    #plt.legend()
    #plt.show()



def test_helmholtz():
    pf = np.linspace(0.01, 0.5)
    T = 2.0
    #T = np.linspace(0.7,1.5)
    #pf = 0.3
    sigma = np.array([1.0])
    sigma = theory.get_sigma(sigma)
    x = np.array([1.0])
    rho = 6*pf/np.pi/np.sum(x*np.diag(sigma)**3)

    plt.plot(pf, theory.F_BN(sigma, x, rho, temp=T), label="BN F (HS)", linestyle=":")
    #plt.plot(pf, theory.Z_SPT(sigma, x, rho), label="SPT F")
    #plt.plot(pf, theory.Z_PY(sigma, x, rho), label="PY F")
    #plt.plot(pf, theory.Z_BMCSL(sigma, x, rho), label="BMCSL F", linestyle="--")
    # Kolafa and Gottschalk (in this implementation) agree (somewhat) around T=4.0.
    #t = theory.F_kolafa(sigma, x, rho, temp=T)

    #plt.plot(pf, theory.F_kolafa(sigma, x, rho, temp=T), 
    #        label="Kolafa F", linestyle="-.")
    #plt.plot(pf, theory.F_gottschalk(sigma, x, rho, temp=T), 
    #        label="Gottschalk F", linestyle=":")
    #plt.plot(pf, theory.F_thol(sigma, x, rho, temp=T), 
    #        label="Thol F", linestyle="--")
    #plt.plot(pf, theory.F_mecke(sigma, x, rho, temp=T), 
    #        label="Mecke F", linestyle="--")
    plt.plot(pf, theory.F_hess(sigma, x, rho, temp=T), 
           label="Hess F", linestyle="dotted")
    plt.ylim((-2.0, 5.0))
    plt.title("Helmholtz free energy of Lennard-Jones fluid")
    plt.legend()
    #plt.show()


def test_eos_from_helmholtz():
    pf = np.linspace(0.01, 0.5)
    T = 2.0
    #T = np.linspace(0.7,1.5)
    #pf = 0.3
    sigma = np.array([1.0])
    sigma = theory.get_sigma(sigma)
    x = np.array([1.0])
    rho = 6*pf/np.pi/np.sum(x*np.diag(sigma)**3)

    t = np.ones_like(pf)
    #plt.plot(pf, theory.get_Z_from_F(theory.F_BN, sigma, x, rho, T)/t, "m-",
    #        label="BN EOS Helmholtz derived")
    #t = theory.Z_kolafa(sigma, x, rho, temp=T)
    #plt.plot(pf, theory.get_Z_from_F(theory.F_kolafa, sigma, x, rho, T, method="kolafa")/t, "k-",
    #        label="Kolafa EOS (Helmholtz derived)")
    #plt.plot(pf, theory.get_Z_from_F(theory.F_gottschalk, sigma, x, rho, T)/t, "g-",
    #        label="Gottschalk EOS Helmholtz derived")
    #plt.plot(pf, theory.get_Z_from_F(theory.F_mecke, sigma, x, rho, T, method="kolafa")/t, "m-",
    #        label="mecke EOS Helmholtz derived")
    #plt.plot(pf, theory.get_Z_from_F(theory.F_thol, sigma, x, rho, T, method="thol")/t, "c-",
    #        label="thol EOS Helmholtz derived")
    plt.plot(pf, theory.get_Z_from_F(theory.F_hess, sigma, x, rho, T, method="kolafa")/t, "b-",
            label="Hess EOS Helmholtz derived")

    plt.ylim((-2.0, 5.0))
    plt.title("Lennard-Jones EOSs derived from Helmholtz free energy")
    plt.legend()
    plt.show()


def test_rdf_from_helmholtz():
    pf = np.linspace(0.01, 0.5)
    T = 3.0
    #T = np.linspace(0.7,1.5)
    #pf = 0.3
    sigma = np.array([1.0])
    sigma = theory.get_sigma(sigma)
    x = np.array([1.0])
    rho = 6*pf/np.pi/np.sum(x*np.diag(sigma)**3)

    plt.plot(pf, theory.get_rdf_from_F(theory.F_kolafa, sigma, x, rho, T, method="kolafa"), 
            "k-", label="kolafa RDF, Helmholtz derived")
    plt.plot(pf, theory.get_rdf_from_F(theory.F_gottschalk, sigma, x, rho, T, method=""), 
            "g-", label="gottschalk RDF, Helmholtz derived")
    plt.plot(pf, theory.get_rdf_from_F(theory.F_mecke, sigma, x, rho, T, method=""), 
            "m-", label="mecke RDF, Helmholtz derived")
    plt.plot(pf, theory.get_rdf_from_F(theory.F_thol, sigma, x, rho, T, method="thol"), 
            "c-", label="thol RDF, Helmholtz derived")
    plt.plot(pf, theory.get_rdf_from_F(theory.F_hess, sigma, x, rho, T, method="thol"), 
            "k:", label="Hess RDF, Helmholtz derived")
    plt.plot(pf, theory.rdf_LJ(pf, T=T), 
            "k-.", label="Morsali RDF")

    plt.ylim((-5.0, 5.0))
    plt.title("Lennard-Jones RDF at contact")
    plt.legend()
    plt.show()


def test_viscosity_from_helmholtz():
    pf = np.linspace(0.01, 0.5)
    T = 2.0
    #T = np.linspace(0.7,1.5)
    #pf = 0.3
    sigma = np.array([1.0])
    sigma = theory.get_sigma(sigma)
    x = np.array([1.0])
    rho = 6*pf/np.pi/np.sum(x*np.diag(sigma)**3)
    m = 1.0

    t = theory.enskog(pf, sigma.flatten(), T, m, theory.rdf_CS, k=1.0)
    plt.plot(pf, t, "--")
    plt.plot(pf, theory.get_viscosity_from_F(theory.F_kolafa, sigma, x, rho, T, method="kolafa"), 
            label="kolafa RDF, Helmholtz derived", linestyle="-")
    #plt.plot(pf, theory.get_viscosity_from_F(theory.F_gottschalk, sigma, x, rho, T), 
    #        label="gottschalk RDF, Helmholtz derived", linestyle=":")
    plt.plot(pf, theory.get_viscosity_from_F(theory.F_mecke, sigma, x, rho, T), 
            label="mecke RDF, Helmholtz derived", linestyle=":")
    #plt.plot(pf, theory.get_viscosity_from_F(theory.F_thol, sigma, x, rho, T), 
    #        label="thol RDF, Helmholtz derived", linestyle="-")
    plt.plot(pf, theory.get_viscosity_from_F(theory.F_BN, sigma, x, rho, T), 
            label="BN RDF, Helmholtz derived", linestyle="-.")
    plt.plot(pf, theory.rdf_LJ(pf, T=T), 
            label="Morsali RDF", linestyle="--")

    #plt.ylim((-2.0, 5.0))
    plt.title("Lennard-Jones vicosity with RDF from Helmholtz free energy")
    plt.legend()
    plt.show()



#test_Z()
#test_helmholtz()
#test_eos_from_helmholtz()
test_rdf_from_helmholtz()
#test_viscosity_from_helmholtz()






### Old stuff:
#def test_thorne():
#    # xi is the mole fraction, so np.sum(x) should equal 1.
#    # The packing fraction, however, is 
#    # a different, independent quantity.
#
#    pf_list = np.linspace(0,0.5)
#    thorne_eta_list = np.zeros_like(pf_list)
#
#    x1 = 0.5
#    x2 = 0.5
#    sigma_list = np.array([1,1])
#    x = np.array([1-x1,x1])
#    m = np.array([1,2])
#    T = 1.5
#    for i, pf in enumerate(pf_list):
#        thorne_eta_list[i] = viscosity.thorne(pf, x, m, sigma_list, T)
#    #plt.plot(pf_list, thorne_eta_list, 
#    #    label=f"Thorne, sigma={sigma_list}", linestyle="-")
#
#    enskog_sigma = np.array([1])
#    enskog_eta_list = viscosity.enskog(
#            pf_list,
#            enskog_sigma,
#            T,
#            m[0]
#        )
#    #plt.plot(pf_list, enskog_eta_list, 
#    #    label=f"Enskog, sigma={enskog_sigma}", linestyle="--")
#    #plt.title("Enskog vs. Thorne with one component")
#    #plt.xlabel("Packing fraction")
#    #plt.ylabel("Viscosity")
#    #plt.legend()
#    #plt.show()
#
#
#def test_rdf():
#    pf = np.linspace(0.001,0.5)
#    sigma_list = np.array([1,2])
#    sigma = viscosity.get_sigma(sigma_list)
#    x1 = 0.5
#    x = np.array([1-x1,x1])
#    m = np.array([1,1])
#    rho = 6*pf/np.pi/np.sum(x*np.diag(sigma)**3)
#    #plt.plot(pf, eos.rdf_PY(pf), 
#    #        label="PY (one component)", color="y")
#
#    #plt.plot(pf, eos.rdf_PY_mix(sigma, x, rho, 0, 0), 
#    #        label="Percus-Yervick(0,0)", color="k", linestyle="-.")
#    #plt.plot(pf, eos.rdf_SPT(sigma, x, rho, 0, 0), 
#    #        label="SPT(0,0)", color="k", linestyle="--")
#    #plt.plot(pf, eos.rdf_PY_mix(sigma, x, rho, 1, 1), 
#    #        label="Percus-Yervick(1,1)", color="b", linestyle="-.")
#    #plt.plot(pf, eos.rdf_SPT(sigma, x, rho, 1, 1), 
#    #        label="SPT(1,1)", color="b", linestyle="--")
#    #plt.title("Comparing radial distribution functions")
#    #plt.xlabel("Packing fraction")
#    #plt.ylabel("Compressibility factor")
#    #plt.legend()
#    #plt.show()
#
#def test_thorne_for_different_rdfs():
#    # xi is the mole fraction, so np.sum(x) should equal 1.
#    # The packing fraction, however, is 
#    # a different, independent quantity.
#
#    pf_list = np.linspace(0,0.5)
#    rdf_list = [eos.rdf_SPT, eos.rdf_PY_mix]
#    thorne_eta_list = np.zeros((len(pf_list), len(rdf_list)))
#
#    x1 = 0.5
#    x2 = 0.5
#    sigma_list = np.array([1,1])
#    x = np.array([1-x1,x1])
#    m = np.array([1,1])
#    T = 1.5
#    for i, pf in enumerate(pf_list):
#        thorne_eta_list[i,0] = viscosity.thorne(pf, x, m, sigma_list, T, rdf=rdf_list[0])
#        thorne_eta_list[i,1] = viscosity.thorne(pf, x, m, sigma_list, T, rdf=rdf_list[1])
#    #plt.plot(pf_list, thorne_eta_list, 
#    #    label=f"Thorne, sigma={sigma_list}", linestyle="-")
#
#    enskog_sigma = np.array([1])
#    enskog_eta_list = viscosity.enskog(
#            pf_list,
#            enskog_sigma,
#            T,
#            m[0]
#        )
#    #plt.plot(pf_list, enskog_eta_list, 
#    #    label=f"Enskog, sigma={enskog_sigma}", linestyle="--")
#    #plt.title("Enskog vs. Thorne with one component")
#    #plt.xlabel("Packing fraction")
#    #plt.ylabel("Viscosity")
#    #plt.legend()
#    #plt.show()
#
