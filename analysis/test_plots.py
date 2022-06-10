import numpy as np
import matplotlib.pyplot as plt
import theory

pf = np.linspace(0.01, 0.5)
T = 2.0
#T = np.linspace(0.7,1.5)
#pf = 0.3

def test_Z(pf, T):
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
    #plt.plot(pf, theory.Z_BN(sigma, x, rho, temp=T)/t, label="BN EOS", linestyle=":")
    plt.plot(pf, theory.Z_kolafa(sigma, x, rho, temp=T)/t, 
        "k:", label="Kolafa Z")
    plt.plot(pf, theory.Z_gottschalk(sigma, x, rho, temp=T)/t,
        "g:", label="Gottschalk Z")
    plt.plot(pf, theory.Z_thol(sigma, x, rho, temp=T)/t, 
        "r:", label="Thol Z")
    plt.plot(pf, theory.Z_mecke(sigma, x, rho, temp=T)/t, 
        "m:", label="Mecke Z", linewidth=3)
    plt.plot(pf, theory.Z_hess(sigma, x, rho, temp=T)/t, 
        "b:", label="Hess Z")
    #plt.ylim((-2.0, 5.0))
    #plt.title("Lennard-Jones Zs")
    #plt.legend()
    #plt.show()



def test_helmholtz(pf, T, temp = False):
    sigma = np.array([1.0])
    sigma = theory.get_sigma(sigma)
    x = np.array([1.0])
    rho = 6*pf/np.pi/np.sum(x*np.diag(sigma)**3)

    #plt.plot(pf, theory.F_BN(sigma, x, rho, temp=T), label="BN F (HS)", linestyle=":")
    #plt.plot(pf, theory.Z_SPT(sigma, x, rho), label="SPT F")
    #plt.plot(pf, theory.Z_PY(sigma, x, rho), label="PY F")
    #plt.plot(pf, theory.Z_BMCSL(sigma, x, rho), label="BMCSL F", linestyle="--")
    # Kolafa and Gottschalk (in this implementation) agree (somewhat) around T=4.0.
    #t = theory.F_kolafa(sigma, x, rho, temp=T)
    if temp:
        plt.plot(pf, theory.F_CS(sigma, x, rho, temp=T), 
                label="CS F", linestyle="-")
        plt.plot(pf, theory.F_kolafa(sigma, x, rho, temp=T), 
                label="Kolafa F", linestyle="-")
        plt.plot(pf, theory.F_gottschalk(sigma, x, rho, temp=T), 
                label="Gottschalk F", linestyle="-")
        plt.plot(pf, theory.F_thol(sigma, x, rho, temp=T), 
                label="Thol F", linestyle="-")
        plt.plot(pf, theory.F_mecke(sigma, x, rho, temp=T), 
                label="Mecke F", linestyle="-")
        plt.plot(pf, theory.F_hess(sigma, x, rho, temp=T), 
               label="Hess F", linestyle="-")
    else:
        T = np.linspace(1.25,4.0)
        pf = 0.4
        rho = 6*pf/np.pi/np.sum(x*np.diag(sigma)**3)
        j = 0
        for F in [theory.F_kolafa, theory.F_thol, theory.F_mecke, theory.F_gottschalk, theory.F_hess]:
            A = np.zeros_like(T)
            for i in range(len(T)):
                A[i] = F(sigma, x, rho, temp=T[i])
            plt.plot(T, A, label=F.__name__, linestyle=["-", "-", ":", "-.", "--"][j])
            j += 1
    #plt.ylim((-2.0, 5.0))
    plt.title("Helmholtz free energy of Lennard-Jones fluid")
    plt.legend()
    plt.show()


def test_eos_from_helmholtz(pf, T):
    sigma = np.array([1.0])
    sigma = theory.get_sigma(sigma)
    x = np.array([1.0])
    rho = 6*pf/np.pi/np.sum(x*np.diag(sigma)**3)

    t = np.ones_like(pf)
    #plt.plot(pf, theory.get_Z_from_F(theory.F_BN, sigma, x, rho, T)/t, "m-",
    #        label="BN EOS Helmholtz derived")
    #t = theory.Z_kolafa(sigma, x, rho, temp=T)
    plt.plot(pf, theory.get_Z_from_F(theory.F_kolafa, sigma, x, rho, T)/t, "k-",
            label="Kolafa EOS (Helmholtz derived)")
    plt.plot(pf, theory.get_Z_from_F(theory.F_gottschalk, sigma, x, rho, T, method="thol")/t, "g-",
            label="Gottschalk EOS Helmholtz derived")
    plt.plot(pf, theory.get_Z_from_F(theory.F_mecke, sigma, x, rho, T)/t, "m-",
            label="mecke EOS Helmholtz derived", linewidth=3)
    plt.plot(pf, theory.get_Z_from_F(theory.F_thol, sigma, x, rho, T, method="thol")/t, "r-",
            label="thol EOS Helmholtz derived")
    plt.plot(pf, theory.get_Z_from_F(theory.F_hess, sigma, x, rho, T)/t, "b-",
            label="Hess EOS Helmholtz derived")
    plt.ylim((-2.0, 5.0))
    plt.title("Lennard-Jones EOSs derived from Helmholtz free energy")
    plt.legend()
    #plt.show()


def test_rdf_from_helmholtz(pf, T, of_temp=False):
    sigma = np.array([1.0])
    sigma = theory.get_sigma(sigma)
    x = np.array([1.0])
    rho = 6*pf/np.pi/np.sum(x*np.diag(sigma)**3)

    if not of_temp:
        plt.plot(pf, theory.rdf_CS(pf), 
                "k--", label="CS RDF")
        plt.plot(pf, theory.get_rdf_from_F(theory.F_CS, sigma, x, rho, T, method="kolafa"), 
                "k--", label="CS RDF, Helmholtz derived")
        plt.plot(pf, theory.get_rdf_from_F(theory.F_BN, sigma, x, rho, T, method="kolafa"), 
                "k--", label="BN RDF, Helmholtz derived")
        plt.plot(pf, theory.get_rdf_from_F(theory.F_kolafa, sigma, x, rho, T, method="kolafa"), 
                "k-", label="kolafa RDF, Helmholtz derived")
        plt.plot(pf, theory.get_rdf_from_F(theory.F_gottschalk, sigma, x, rho, T, method="thol"), 
                "g-", label="gottschalk RDF, Helmholtz derived")
        #plt.plot(pf, theory.get_rdf_from_F(theory.F_mecke, sigma, x, rho, T, method=""), 
        #        "m-", label="mecke RDF, Helmholtz derived")
        plt.plot(pf, theory.get_rdf_from_F(theory.F_thol, sigma, x, rho, T, method="thol"), 
                "c-", label="thol RDF, Helmholtz derived")
        plt.plot(pf, theory.get_rdf_from_F(theory.F_hess, sigma, x, rho, T, method=""), 
                "k:", label="Hess RDF, Helmholtz derived")
        plt.plot(pf, theory.rdf_LJ(pf, T=T), 
                "k-.", label="Morsali RDF")
        plt.plot(pf, np.ones_like(pf), "--", label="one")
        pf = 0.05
        rho = 6*pf/np.pi/np.sum(x*np.diag(sigma)**3)
        j = 0
    else:
        for F in [theory.F_kolafa, theory.F_thol, theory.F_mecke, theory.F_gottschalk, theory.F_hess]:
            A = np.zeros_like(T)
            for i in range(len(T)):
                A[i] = theory.get_rdf_from_F(F, sigma, x, rho, temp=T[i])
            plt.plot(T, A, label=F.__name__, linestyle=["-", "-", ":", "-.", "--"][j])
            j += 1
    #plt.ylim((-5.0, 5.0))
    plt.title("Lennard-Jones RDF at contact")
    plt.legend()
    #plt.show()


def test_internal_energy_from_helmholtz(pf, T):
    sigma = np.array([1.0])
    sigma = theory.get_sigma(sigma)
    x = np.array([1.0])
    rho = 6*pf/np.pi/np.sum(x*np.diag(sigma)**3)

    plt.plot(pf, theory.get_internal_energy(theory.F_kolafa, sigma, x, rho, T, method="kolafa"), 
            "k-", label="kolafa internal_energy, Helmholtz derived")
    plt.plot(pf, theory.get_internal_energy(theory.F_gottschalk, sigma, x, rho, T, method="thol"), 
            "g-", label="gottschalk internal_energy, Helmholtz derived")
    plt.plot(pf, theory.get_internal_energy(theory.F_mecke, sigma, x, rho, T, method=""), 
            "m-", label="mecke internal_energy, Helmholtz derived")
    plt.plot(pf, theory.get_internal_energy(theory.F_thol, sigma, x, rho, T, method="thol"), 
            "c-", label="thol internal_energy, Helmholtz derived")
    plt.plot(pf, theory.get_internal_energy(theory.F_hess, sigma, x, rho, T, method="thol"), 
            "k:", label="Hess internal_energy, Helmholtz derived")

    plt.ylim((-5.0, 5.0))
    plt.title("Lennard-Jones internal energy")
    plt.legend()
    #plt.show()



def test_viscosity_from_helmholtz(pf, T):
    sigma = np.array([1.0])
    sigma = theory.get_sigma(sigma)
    x = np.array([1.0])
    rho = 6*pf/np.pi/np.sum(x*np.diag(sigma)**3)
    m = 1.0

    #t = theory.get_viscosity_from_F(theory.F_CS, sigma, x, rho, T)
    t = np.ones_like(pf)
    #plt.plot(pf, t, "--", label="one")
    plt.plot(pf, np.full_like(pf, theory.zero_density_viscosity(m, sigma, T, 1.0, 1.0)), "--", label="$\eta_O$")
    plt.plot(pf, theory.get_viscosity_from_F(theory.F_kolafa, sigma, x, rho, T)/t, 
            label="kolafa RDF, Helmholtz derived", linestyle="-")
    plt.plot(pf, theory.get_viscosity_from_F(theory.F_gottschalk, sigma, x, rho, T, method="thol")/t, 
            label="gottschalk RDF, Helmholtz derived", linestyle="--", color="green")
    plt.plot(pf, theory.get_viscosity_from_F(theory.F_mecke, sigma, x, rho, T)/t, 
            label="mecke RDF, Helmholtz derived", linestyle=":")
    plt.plot(pf, theory.get_viscosity_from_F(theory.F_thol, sigma, x, rho, T, method="thol")/t, 
            label="thol RDF, Helmholtz derived", linestyle="--", color="lime")
    plt.plot(pf, theory.get_viscosity_from_F(theory.F_hess, sigma, x, rho, T, method="hess")/t, 
            label="hess RDF, Helmholtz derived", linestyle="--", color="lime")
    plt.plot(pf, theory.get_viscosity_from_F(theory.F_CS, sigma, x, rho, t)/t, 
            label="CS RDF, Helmholtz derived", linestyle="-.")
    def g(pf, T=1.0):
        rho = 6*pf/np.pi
        return theory.rdf_LJ(sigma, x, rho)
    plt.plot(pf, theory.enskog(pf, sigma, T, m, g).flatten()/t, 
            label="Morsali RDF", linestyle="--")

    plt.ylim((-0.5, 4.0))
    plt.title("Lennard-Jones vicosity with RDF from Helmholtz free energy")
    plt.legend()
    #plt.show()


def test_enskog():
    pf = np.linspace(0.025,0.5)
    sigma = np.array([1.0])
    x = 1.0
    T = 1.5
    rho = theory.pf_to_rho(sigma, x, pf)
    def g(pf, T=1.0):
        rho = theory.pf_to_rho(sigma, x, pf)
        return theory.get_rdf_from_F(theory.F_thol, sigma, x, rho, T=T)
    visc = theory.enskog(pf, sigma, T, 1.0, g)
    plt.plot(pf, visc)



#test_Z(pf, T)
#test_eos_from_helmholtz(pf, T)
#plt.show()
#test_helmholtz(pf, T)
#plt.show()
#test_helmholtz(pf, T, temp=True)
#plt.show()
#test_internal_energy_from_helmholtz(pf, T)
#plt.show()
test_rdf_from_helmholtz(pf, T)
plt.show()
#test_viscosity_from_helmholtz(pf, T)
#plt.show()
#test_enskog()
#plt.show()
