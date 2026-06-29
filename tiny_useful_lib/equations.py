import numpy as np
import numdifftools as nd
from scipy.optimize import root

# physical constants
e=1.6*1e-19
hbar=1.05*1e-34
k_b = 1.380649*1e-23


def Ec_of_C(C): 
    """
       Parameters: C : fF
       Returns: Ec : GHz
    """
    return e**2/(4*np.pi*hbar)/ C *1e6


def C_of_Ec(Ec): 
    """
       Parameters: Ec : GHz
       Returns: C : fF
    """
    return e**2/(4*np.pi*hbar)/ Ec *1e6


def El_of_L(L): 
    """
       Parameters: L : nH
       Returns: El : GHz
    """
    return hbar/(8*np.pi*e**2)/ L


def L_of_El(El): 
    """
       Parameters: El : GHz
       Returns: L : nH
    """
    return hbar/(8*np.pi*e**2)/ El


def g_of_C(C, C_1, C_2):
    """
       Compute energy of capacitive coupling of two islands

       Parameters:

       C: fF, capacitance between the islnads
       C_1: fF, capacitance of the first island to the ground
       C_2: fF, capacitance of the second island to the ground

       Returns:
       g: GHz, capacitive coupling
       
    """
    return C/(C_1*C_2 + C*C_1 + C*C_2) * 4 * e**2 / (2*np.pi*hbar)*1e6


def Z_of_resonator_mode(Z_0=50, n=1, regime='half'):
    """
       Compute efective electrical impedance of the given resonator mode
        
       Parameters:
           
       Z_0 : Ω, default: 50
           physical line impedance defined as sqrt(L/C)
       n : natural, default: 1
           number defining the mode
       regime : str, 'half' or 'quarter', default: 'half'
           to define the resonator regime

       Returns:
       Z : Ω
    """

    if(regime=='half'): return 2/np.pi/n*Z_0
    if(regime=='quarter'): return 4/np.pi/(2*n - 1)*Z_0
    raise ValueError('Invalid resonator regime')


def El_of_resonator_mode(f_0, Z_0=50, n=1, regime='half'):
    """
       Compute efective inductive energy of the given resonator mode
        
       Parameters:
           
       f_0 : GHz
           original fundamental frequency of the resonator
       Z_0 : Ω; default: 50
           physical line impedance defined as sqrt(L/C)
       n : natural, default: 1
           number defining the mode
       regime : str, 'half' or 'quarter', default: 'half'
           to define the resonator regime

       Returns:
       El : GHz
    """

    if(regime=='half'): return El_of_L(1/np.pi**2/n**2 * Z_0/f_0)
    if(regime=='quarter'): return El_of_L(2/np.pi**2/(2*n - 1)**2 * Z_0/f_0)
    raise ValueError('Invalid resonator regime')


def C_of_resonator_mode(f_0, Z_0=50, regime='half'):
    """
       Compute efective capacitance for all resonator modes
        
       Parameters:
           
       f_0 : GHz
           original fundamental frequency of the resonator
       Z_0 : Ω, default: 50
           physical line impedance defined as sqrt(L/C)
       regime : str, 'half' or 'quarter', default: 'half'
           to define the resonator regime

       Returns:
       C : fF
    """

    if(regime=='half'): return 1/(4*Z_0*f_0) * 1e6
    if(regime=='quarter'): return 1/(8*Z_0*f_0) * 1e6
    raise ValueError('Invalid resonator regime')
    

def Z_of_osc(El, Ec):
    """
       compute impedance of oscillator with Hamiltonian:
       H/h = 4*Ec*n^2 + El/2*phi^2
       Parameters: 
       El : GHz
       Ec : GHz
       
       Returns: Z : Ω
    """
    return hbar/e**2*np.sqrt(Ec/El/2)


def f_of_osc(El, Ec):
    """
       compute frequency of oscillator with Hamiltonian:
       H/h = 4*Ec*n^2 + El/2*phi^2
       Parameters: 
       El : GHz
       Ec : GHz
       
       Returns: f : GHz
    """
    return np.sqrt(8*Ec*El)


def Z_of_cpw(f, L, c, Z_l, Z_0=50):
    """
       compute effective impedance of a loaded cpw:
          __________________________          ___
         |                                   |
        Z_l         Z_0, c            --->   Z
         |__________________________         |___
                       L

       Parameters:

       f : GHz, wave frequency
       L : mm, cpw length
       c : fF/mm, cpw differential capacitance
       Z_l : Ω/'open'/'closed'/function(f in GHz),
           loading impedance (can be set as open or closed boundary condition as well)
       Z_0 : Ω, characteristic cpw impedance

       Returns:

       Z : Ω
    """  
    omega = 2*np.pi*f*1e9

    if(callable(Z_l)):
        return Z_0*(Z_l(f) + 1j*Z_0*np.tan(omega*Z_0*c*1e-15*L))/(Z_0 + 1j*Z_l(f)*np.tan(omega*Z_0*c*1e-15*L))
    elif(type(Z_l)!=str):
        return Z_0*(Z_l + 1j*Z_0*np.tan(omega*Z_0*c*1e-15*L))/(Z_0 + 1j*Z_l*np.tan(omega*Z_0*c*1e-15*L))
    elif(Z_l=='short'):
        return 1j*Z_0*np.tan(omega*Z_0*c*1e-15*L)
    elif(Z_l=='open'):
        return -1j*Z_0/np.tan(omega*Z_0*c*1e-15*L)
    else:
        raise ValueError('Invalid Z_l matrix!')


def f_of_resonator_cpw(L, c, Z_in, Z_out, Z_0=50, f_bounds=[0.1, 10]):
    """
       compute frequencies of lossless cpw rersonator modes:
          __________________________
         |                          |
       Z_in         Z_0, c         Z_out
         |__________________________|
                       L
       compute effective Im{Y} at 1/pi of L and then solve Y(omega) = 0

       Parameters:

       L : mm, cpw length
       c : fF/mm, cpw differential capacitance
       Z_in, Z_out : Ω/'open'/'closed'/function(f in GHz),
       Z_0 : Ω, characteristic cpw impedance
       f_bounds : [f_0, f_1], GHz
           define range of search for resonances

       Returns:

       f : 1-D np.array, GHz
           ordered array of resonances founded with intitial points in f_bounds
           
    """ 
    def Y_im(f): return np.imag(1/Z_of_cpw(f, L/np.pi, c, Z_in, Z_0=Z_0) + 1/Z_of_cpw(f, L*(np.pi - 1)/np.pi, c, Z_out, Z_0=Z_0))

    f_sol = np.zeros(100)
    f0_list = np.linspace(f_bounds[0], f_bounds[1], 100)
    
    for n in range(100):
        f = root(Y_im, x0=f0_list[n]).x
        if(np.min(np.abs(f_sol - f)) > 1e-2): f_sol[n] = f

    return f_sol[f_sol.nonzero()]

    
def n_zpf(Z): 
    """
       Compute zero-point fluctuation of Cooper-pair operator in harmonic oscillator,
       needed in n = 1j*n_zpf*(at - a) or n = n_zpf*(at + a)
       Parameters: Z : Ω
       Returns: n_zpf : float
    """
    return np.sqrt(hbar/(8*e**2)/ Z)


def phi_zpf(Z):
    """
       Compute zero-point fluctuation of Cooper-pair operator in harmonic oscillator,
       needed in phi = phi_zpf*(at + a) or phi = -1j*phi_zpf*(at - a)
       Parameters: Z : Ω
       Returns: phi_zpf : float
    """
    return np.sqrt(2*e**2/hbar * Z)


def Ej_of_R(R, delta=0.204):
    """
       Compute Ej with simplified Ambegoakar-Baratoﬀ relation
       Parameters:
       R : kΩ
           resistance of the Josephson junction
       delta : meV, default: 0.204
           effective superconducting energy gap
       Returns: Ej : GHz
    """
    return delta/8/e/R *1e-15


def R_of_Ej(Ej, delta=0.204):
    """
       Compute Josephson junction R with simplified Ambegoakar-Baratoﬀ relation
       Parameters:
       Ej : GHz
       delta : meV, default: 0.204
           effective superconducting energy gap
       Returns: R : kΩ
    """
    return delta/8/e/Ej *1e-15


def Ej_of_F_SQUID(Ej_1, Ej_2, F):
    """
       compute effective SQUID Ej under external flux F
       used equation is described in DOI: 10.1063/1.5089550
        
       Parameters:
       Ej_1 : GHz
       Ej_2 : GHz
       F : flux quanta
           external constant flux

       Returns:
       Ej : GHz
    """
    gamma = Ej_2/Ej_1
    d = (gamma - 1)/(gamma + 1)
    Ej = (Ej_1 + Ej_2)*np.sqrt(np.cos(np.pi*F)**2 + d**2 * np.sin(np.pi*F)**2)
    return Ej

    
def F_of_Ej_SQUID(Ej_1, Ej_2, Ej):
    """
       compute external flux F needed to achieve particular effective SQUID Ej
       used equation is based on DOI: 10.1063/1.5089550
        
       Parameters:
       Ej_1 : GHz
       Ej_2 : GHz
       Ej : GHz

       Returns:
       F : flux quanta
           external constant flux
    """
    
    if(Ej_1 + Ej_2 < Ej or abs(Ej_1 - Ej_2) > Ej): raise ValueError('Unreachable target Ej value!')
    
    gamma = Ej_2/Ej_1
    d = (gamma - 1)/(gamma + 1)
    Phi = np.arcsin(np.sqrt((Ej**2 - (Ej_1 + Ej_2)**2)/(Ej_1 + Ej_2)**2/(d**2 - 1)))

    return Phi/np.pi


def kappa_0(f, pin_q, Z=50):
    """
       compute kappa factor for dumping via capacitive pin, giving dissipative part of
       Lindblad equation for system operator n in the form:
       
       kappa_0*(N_T + 1)*D[n^-]ro + kappa_0*N_T*D[n^+]ro
       
       where n^-/+ are upper/lower triangle parts of n, and N_T is number of termal photons

       equation: kappa = 4*pi*f*Z*pin_q/hbar
       *generilized form for ~ Eq.68 in DOI: 10.1103/RevModPhys.93.025005

       Parameters:
       f : GHz
           frequency poind
       pin_q : GHz/mV
           effective system charge at pin capcitor
       Z : Ω
           pin line impedance

       Returns:
       kappa_0 : 2pi*GHz
           must be multiplied by (n_ij)^2 to get kappa for the certain i <-> j transition
           notably T_1 = 1/kappa_0
    """

    kappa_0 = 4*np.pi*f*Z/hbar * (pin_q*2*np.pi*hbar*1e12)**2
    return kappa_0


def N_termal(f, T):
    """
       compute number of termal photons in oscillator according to plank distribution

       N = 1/(e^(2*pi*f*hbar/k/T) - 1)

       Parameters:
       f : GHz
       T : mK

       Returns:
       N : float
    """
    N = 1/(np.exp(hbar*2*np.pi*f*1e9/(k_b*T*1e-3)) - 1)
    return N


def pin_q_of_C(C_pin, C):
    """
       compute rescaled effective system charge at pin capcitor

       pin_q = 2*e*C_pin/(C + C_pin)

       Parameters:
       C_pin : fF
       C : fF
       
       Returns:
       pin_q : GHz/mV
    """
    
    return 2*e*C_pin/(C + C_pin) / (2*np.pi*hbar) * 1e-12


def kappa_osc_to_line(f, C_0, C, C_in=None, C_out=None, Z_in=50, Z_out=50, mode_params=False):
    """
       compute classical kappa for an oscillator connected to the line via capacitance C

       Z_in --- C_in --- --- C_out --- Z_out
                        |                              
                        C                       -->    Z
                        |                             
                      L&C_0

       compute effective impedance Z, and admittance Y=1/Z,
       and then solution oif Y(x +iy) = 0 gives
       kappa = 2*y and omega_new = x since 
       there is only one oscillator mode here
       DOI: 10.1103/RevModPhys.93.025005
       DOI: 10.1103/PhysRevLett.108.240502

       Parameters: 
       f : GHz, oscillator frequency
       C_0, C, C_in, C_out : fF
       Z_in, Z_out : Ω

       Retutns:
       kappa : 2pi*GHz
       f_new : GHz (if mode_params=True)
       oscillator mode characteristic impedance sqrt(L/C): Ω (if mode_params=True)
    """
    omega = 2*np.pi*f*1e9

    def Y(x):
        Z_1 = Z_in
        Z_2 = Z_out
        
        if(C_in!=None):  Z_1 = Z_1 + 1/(1j*x*C_in*1e-15)
        if(C_out!=None): Z_2 = Z_2 + 1/(1j*x*C_out*1e-15)
        
        Z_e = 1/(1j*x*C*1e-15) + 1/(1/Z_1 + 1/Z_2)
        
        return 1/Z_e + 1j*x*C_0*1e-15 + (omega**2 * C_0*1e-15)/(1j*x)

    def Y_R2(x): return [np.real(Y(x[0] + 1j*x[1])), np.imag(Y(x[0] + 1j*x[1]))]
    omega_new, kappa = root(Y_R2, x0=[omega, 1e9]).x
    kappa = 2*kappa

    if(mode_params):
          
        def Y_im(x): return np.imag(Y(x))       
        Y_d = nd.Derivative(Y_im)
        C_mod = Y_d(omega_new)/2
        Z_mod = 1/omega_new/C_mod

        return kappa*1e-9, omega_new/(2*np.pi)*1e-9, Z_mod

    return kappa*1e-9
    

def kappa_osc_via_bandpass_eq(f_r, f_f, g, kappa_f, method='Heinsoo', C_in=0, Z_0=50):
    """
       compute kappa of a readout resonator seen by feedline via bandpass purcell filter.
       Two equations based on input-output theory are available:
       1) derived by J.Heinsoo from DOI: https://doi.org/10.1103/PhysRevApplied.10.034040
       2) derived by E.A.Sete from DOI: https://doi.org/10.1103/PhysRevApplied.10.034040
       first model gives results closer to straighforward microwave approach (via admittance),
       espetially for small filter kappas < 100 MHz

                       Heinsoo model                         Sete model

       feedline IN --- C_in --- --- feedline OUT              feedline
                               |                                  |
                         purcell filter                     purcell filter
                         (f_f, kappa_f)                     (f_f, kappa_f)
                               |                                  |       
                               g                                  g
                               |                                  |
                        readout reso (f_r)                 readout reso (f_r)
                        
       Parameters:
       f_r : resonator frequency, GHz
       f_f : filter frequency, GHz - MUST INCLUDE SHIFT BY FEEDLINE!!!
       g : filter-resonator coupling in the form g*(a + at)*(b + bt), GHz
       kappa_f : filter bandwith, 2pi*GHz
       method : "Heinsoo" (default) or "Sete"
       C_in : input capacitor, fF
       Z_0 : feadline impedance, Ω

       Returns:
       kappa_r : reqdot resonator bandwith, 2pi*GHz              

    """ 
    w_r = 2*np.pi*f_r
    w_f = 2*np.pi*f_f
    J = 2*np.pi*g

    if(method=='Heinsoo'):
        # input capacitor impact
        gamma = 1/(1 + 2j*w_f*Z_0*C_in*1e-6)
        w_f = w_f + kappa_f*np.imag(gamma)/4
        kappa_f_mod = kappa_f*(1 + np.real(gamma))/2
    
        kappa_r = (kappa_f_mod - np.real(np.sqrt(-16 * J**2 + (kappa_f_mod - 2j*(w_f - w_r))**2)))/2
    elif(method=='Sete'):
        kappa_r = 4 * J**2 / kappa_f / (1 + (2*(w_r - w_f)/kappa_f)**2)
    else:
        raise ValueError('Invalid method name')
        
    return kappa_r


def purcell_bandpass_filter_factor_eq(f_q, f_r, f_f, kappa_f):
    """
       compute factor F=kappa_q/kappa_r for resonator filtered by pandpass purcell filter, 
       where kappa_q - kappa of the readout reso seen by qubit, and
       kappa_r - kappa of the same reso seen by line (readout kappa)
       equation can be found in DOI: https://doi.org/10.1103/PhysRevA.92.012325
        
       Parameters:
       f_q : qubit dressed frequency, GHz
       f_r : resonator dressed frequency, GHz
       f_f : filter dressed frequency, GHz
       kappa_f : filter bandwith, 2pi*GHz

       Returns:
       F : kappa_q/kappa_r factor, to get purcell limit of the qubit one
       needs to use additional factor K between readout reso kappa and
       qubit kappa. The result is QUBIT KAPPA = K*F*KAPPA_R
       
    """
    F = (1 + (4*np.pi*(f_r - f_f)/kappa_f)**2)/(1 + (4*np.pi*(f_q - f_f)/kappa_f)**2)
    return F

        
def C_of_Ej(Ej, Ej_to_S=400, C_to_S=45):
    """
       Parameters:
       Ej_to_S : GHz/um^2
       C_to_S : fF/um^2

       Returns:
       C_of_Ej : fF
    """

    return Ej/Ej_to_S*C_to_S


def purcell_kappa_factor_eq(f_q, f_r, g, method='Yen'):
    """
       compute purcell kappa factor with oscillator equation (works for transmon):
       1) derived by Yen: K = (g/(f_q - f_r))**2 * (f_q/f_r)**3 * (2*f_q/(f_q + f_r))**2
          from DOI:https://doi.org/10.1103/PhysRevApplied.23.024068
       2) classical one: K = (g/(f_q - f_r))**2
          from DOI: https://doi.org/10.1103/RevModPhys.93.025005

       note that qubit purcell kappa = K*kappa_reso

       Parameters:
       f_q : qubit frequency, GHz
       f_r : resonator frequency, GHz
       g : qubit-resonator coupling in the form g*(a + at)*(b + bt) ~ g*(a + at)*σx, GHz
       method : 'Yen' (default) or 'Classic'

       Returns:
       K : kappa factor, float
    """

    if(method=='Yen'):
        K = (g/(f_q - f_r))**2 * (f_q/f_r)**3 * (2*f_q/(f_q + f_r))**2
    elif(method=='Classic'):
        K = (g/(f_q - f_r))**2
    else:
        raise ValueError('Invalid method name')      
    
    return K


def bbq_resonances_of_Y(f, Y, L, C, jump_factor=8):
    """
       Part of black box quantization, find resonances based on
       admittance Y(omega) + 1/(i*omega*L) + i*omega*C
       
       Parameters:
       f : 1D np.array, GHz
           frequencies for Y(f)
       Y : 1D np.array, Ω
           admittance seen by JJ
       L : JJ inductance, nH
       C : JJ capacitance, fF
       jump_factor : float,
           defines maximal slope near valid roots 
           (increase for narrow peaks)

       Returns:
       roots : 1D np.array, GHz
           resonant frequencies
    """
    Y_im = np.imag(Y) - 1/(2*np.pi*L*f) + 2*np.pi*C*1e-6*f
    roots = []
    avd = 0
    
    for n in range(Y_im.shape[0] - 1): avd += np.abs((Y_im[n+1]-Y_im[n]))
    
    for n in range(Y_im.shape[0] - 1):

        if(Y_im[n+1]*Y_im[n] < 0 and np.abs((Y_im[n+1]-Y_im[n])) < avd/Y_im.shape[0]*jump_factor):
            roots.append((f[n+1]+f[n])/2)

    return np.asarray(roots)
    
