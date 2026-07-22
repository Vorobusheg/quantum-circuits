import numpy as np
import warnings
import numdifftools as nd
from scipy.optimize import root, least_squares
try:
    from scipy.interpolate import AAA
    _HAS_AAA = True
except ImportError:
    _HAS_AAA = False
    warnings.warn("scipy>=1.15 not found: AAA interpolator disabled, ", stacklevel=2)

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


def capacitive_coupling(C, C_1, C_2):
    """
       Compute capacitive elements of energy matrix 
       for two capacitivly coupled islands

       Parameters:

       C: fF, capacitance between the islnads
       C_1: fF, capacitance of the first island to the ground
       C_2: fF, capacitance of the second island to the ground

       Returns:
       g: GHz, capacitive coupling
       Ec_1: GHz, effective capacitive energy
       Ec_2: GHz, effective capacitive energy
       
    """
    g = C/(C_1*C_2 + C*C_1 + C*C_2) * 4*e**2/(2*np.pi*hbar)*1e6
    Ec_1 = (C + C_2)/(C_1*C_2 + C*C_1 + C*C_2) * 4*e**2/(2*np.pi*hbar)*1e6/8
    Ec_2 = (C + C_1)/(C_1*C_2 + C*C_1 + C*C_2) * 4*e**2/(2*np.pi*hbar)*1e6/8
    
    return g, Ec_1, Ec_2


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

        
def C_of_Ej(Ej, Ej_to_S=400, C_to_S=45):
    """
       Parameters:
       Ej_to_S : GHz/um^2
       C_to_S : fF/um^2

       Returns:
       C_of_Ej : fF
    """

    return Ej/Ej_to_S*C_to_S


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


def lambda_of_f(f, Z_0=50, c_cpw=169): 
    """
       compute wavelength of a wave with frequency 
       f in cpw under given Z_0 and c_cpw
        
       Parameters:
       f : GHz
       Z_0 : Ω
       c_cpw : fF/mm

       Returns:
       lambda : mm
    """
    return 1/(f*1e9*Z_0*c_cpw*1e-15)


############################################################################
########################## BIG MICROWAVE BLOCK #############################
############################################################################


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
       Z_l : Ω/'open'/'short'/function(f in GHz),
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


def find_Y_zeros(Y, band, dw_frac=0.1, n_scan=6001, n_aaa=4001, tolY=1e-8):
    """
       (AI written)
       find all upper-half-plane (j -> +i notation!!!) zeros z_k = x_k + i*y_k 
       of an admittance Y(x) inside a frequency band: x_k -> mode frequency, 
       kappa_k = 2*y_k. Seeds are collected from three complementary sources:
       (1) rising zero-crossings of Im Y with analytic y0 = ReY/(dImY/dw)
           -> narrow modes of ANY linewidth,
       (2) local minima of |Y| -> intermediate modes,
       (3) poles of an AAA rational fit of Z = 1/Y -> broad modes deep in
           the complex plane (needs scipy >= 1.15, else skipped with warning).
       Every seed is polished by a least-squares solve BOUNDED to a box
       around the seed, so a polish can never escape to a different root.
       
       Parameters:
       Y : callable
           vectorized admittance in S vs frequency (GHz)
       band : GHz
           search window for the real part of the zeros
       n_scan : int
           real-axis scan points; must resolve the Im Y features of the
           narrowest reactance variation (NOT the mode linewidths)
       n_aaa : int
           max sample count for the AAA fit (scan grid is subsampled to it),
           increas in case of width modes failure
       tolY : float
           acceptance threshold |Y(x_k)| < tolY for a polished root
       dw_frac : float
           polish-box half-width as a fraction of the seed frequency;
           reduce if distinct roots are closer than dw_frac*w in frequency
           
       Returns:
       zeros : complex ndarray, GHz
           z_k = x_k + i*y_k sorted by x_k; f_k = x_k, kappa_k = 2*y_k,
           mode quality factor Q_k = x_k/(2*y_k)
    """
    wlo, whi = band[0], band[1]
    ws = np.linspace(wlo, whi, n_scan)
    Yv = Y(ws)
    mag, imY = np.abs(Yv), np.imag(Yv)
 
    seeds = []
    for i in range(n_scan - 1):
        if imY[i] < 0 <= imY[i+1]:
            w0 = ws[i] - imY[i]*(ws[i+1]-ws[i])/(imY[i+1]-imY[i])
            dIm = (imY[i+1]-imY[i])/(ws[i+1]-ws[i])
            y0 = np.real(Y(w0))/dIm
            if y0 > 0:
                seeds.append(w0 + 1j*y0)
    for i in range(1, n_scan - 1):
        if mag[i] < mag[i-1] and mag[i] < mag[i+1]:
            seeds += [ws[i]*(1 + 1j*yf) for yf in (1e-4, 1e-3, 1e-2)]
    if _HAS_AAA:
        try:
            sub = ws[::max(1, n_scan//n_aaa)]
            pol = AAA(sub.astype(complex), 1/Y(sub), rtol=1e-13).poles()
            seeds += list(pol[(pol.imag > 0) &
                              (pol.real > wlo) & (pol.real < whi)])
        except Exception as e:
            warnings.warn(f"AAA seeding failed ({type(e).__name__}: {e}); "
                          "broad modes may be missed", stacklevel=2)

    def _polish_bounded(Y, z0, dw_frac=dw_frac):
        """Bounded least-squares polish of Y(z)=0 in a box around the seed."""
        s = abs(z0)
        dw = dw_frac*z0.real
        lo = [(z0.real - dw)/s, 1e-12]
        hi = [(z0.real + dw)/s,
              min(10*max(z0.imag, 1e-6*z0.real), 0.5*z0.real)/s]
        x0 = [z0.real/s, np.clip(z0.imag/s, lo[1], hi[1])]
        fun = lambda v: [np.real(Y((v[0] + 1j*v[1])*s)),
                         np.imag(Y((v[0] + 1j*v[1])*s))]
        sol = least_squares(fun, x0=x0, bounds=(lo, hi), xtol=3e-16, ftol=3e-16, gtol=None)
        z = (sol.x[0] + 1j*sol.x[1])*s
        return z if abs(Y(z)) < abs(Y(z0)) else z0
    
    roots = []
    for z0 in seeds:
        z = _polish_bounded(Y, z0)
        if (np.isfinite(z) and abs(Y(np.around(z, 15))) < tolY and z.imag >= 0
                and wlo < z.real < whi):
            roots.append(z)
            
    uniq = []
    for z in sorted(roots, key=lambda q: abs(Y(q))):
        if not any(abs(z - u) < 1e-10*abs(u) for u in uniq):
            uniq.append(z)
    return np.asarray(sorted(uniq, key=lambda q: q.real))


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
       Z_in, Z_out : Ω/'open'/'short'/function(f in GHz),
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


def kappa_n(f, pin_q, Z=50):
    """
       compute kappa coefficient for dumping via capacitive pin:

       Mode (f) - pin_q - Z  <~~~  L&C - Cg - Z
       
       assume dissipative part of Lindblad equation for 
       system operator n in the form:
       
       kappa_n*(N_T + 1)*D[n^-]ro + kappa_0*N_T*D[n^+]ro
       
       where n^-/+ are upper/lower triangle parts of n, and N_T is number of termal photons

       equation: 
           kappa_n = 4*pi*f*Z*pin_q/hbar
           kappa_ij = kappa_n*|n_ij|^2  (for specific transition i -> j)
       *generilized form for ~ Eq.68 in DOI: 10.1103/RevModPhys.93.025005

       gives the same results as kappa_osc_mw, especially for small Cg/C

       Parameters:
       f : GHz
           bare transition frequency (NOT SHIFTED BY THE FEEDLINE)
       pin_q : GHz/mV
           effective system charge at pin capacitor
       Z : Ω
           pin impedance

       Returns:
       kappa_n : GHz
           must be multiplied by (n_ij)^2 to get kappa for the certain i <-> j transition
           notably T_1 = 1/kappa_0/2/pi
    """

    kappa_n = 2*f*Z/hbar * (pin_q*2*np.pi*hbar*1e12)**2
    return kappa_n

 
def kappa_osc_mw(f, C, C_g, C_in=None, Z_in=50, Z_out=50, Z_0=50, 
                 dist=0, c_cpw=169, mode_info=False, method='zeros', band=None, 
                 dw_frac=0.1, n_scan=6001, n_aaa=4001, tolY=1e-8):
    """
       compute kappas, frequencies, and b-participation for a system 
       of two oscillators consequencially coupled to feedline as
       shown in the following:
       
       Z_in(w) --- C_in (optional) -------- --- Z_out(w)
                                     <-->  |
                                     dist C_g
                                           |
                                          L&C

       there are to methods available:
       1) 'zeros' – computes equivalent Y(w) function and searches for
           its zeros x_k = w_k + i*y_k -> mode frequency w_k, kappa_k = 2*y_k
           ADVANTAGE – allow feedline and frequency dependent Z_in/out; 
           DROWBACK – can miss roots.
           DOI: 10.1103/RevModPhys.93.025005
           DOI: 10.1103/PhysRevLett.108.240502
           
       2) 'MNA' – use magic of linear lumped elements and combine effective
           linear equation (MNA equations in the Laplace domain) to find
           resonance modes. 
           ADVANTAGE – vectorized and stable;
           DROWBACK – can't handle feedline or complex Z_in/out.
           ISBN:978-0-442-28108-3
           
       Parameters:
       f : GHz, bare frequencies of oscillators
       C_g/in : fF
       Z_in/out : Ω
           can be value, or None (for termination), or a function of frequency (complex) in GHz
       Z_0 : CPW impedance, Ohm
       dist : distance between C_in and oscillator contact, mm
           (works only for C_in != None and method='zeros'!)
       c_cpw : CPW capacitance per length, fF/mm
       mode_info : if True – returns modes frequencies and effective impedances
       method : can be 'zeros' or 'MNA'
       band/dw_frac/n_scan/n_aaa=4001/tolY=1e-8 – params of find_Y_zeros 
           which are described in find_Y_zeros documentation
       
       Returns:
       kappa_modes : 1-D np.array, GHz,
       
       f_modes : 1-D np.array, GHz, (IF mode_info=True)
       Z_modes : 1-D np.array, Ω, (IF mode_info=True)
       
    """
    w = 2*np.pi*f*1e9
    L = 1/(w**2 * C*1e-15)

    # for terminated lines
    if Z_in is None: Z_in=float('inf')
    if Z_out is None: Z_out=float('inf')
        
    if(method=='zeros'):
        def Z_l(x):
            if(callable(Z_in)): 
                Z_1 = Z_in(x/2/np.pi/1e9)
            else:
                Z_1 = Z_in
                
            if(callable(Z_out)): 
                Z_2 = Z_out(x/2/np.pi/1e9)
            else:
                Z_2 = Z_out
                
            if C_in is not None:
                Z_1 = Z_1 + 1/(1j*x*C_in*1e-15)
                Z_1 = Z_of_cpw(x/2/np.pi/1e9, dist, c_cpw, Z_1, Z_0=Z_0)
                
            return 1/(1j*x*C_g*1e-15) + 1/(1/Z_1 + 1/Z_2)   
        
        def Y(x):
            x = 2*np.pi*1e9*x
            return 1/Z_l(x) + 1j*x*C*1e-15 + 1/(1j*x*L)
    
        if band is None:
            band = (0.5*f, 1.5*f)
        uniq = find_Y_zeros(Y, band, dw_frac, n_scan, n_aaa, tolY)
    
        f_modes = np.real(uniq)
        kappas = 2*np.imag(uniq)

        if(mode_info):
            # effective impedances according to Nigg
            def Y_im_by_2(f): return np.imag(Y(f))/2
            C_eff = nd.Derivative(Y_im_by_2)
            Z_modes = []
            for w0 in uniq: Z_modes.append(1/(np.real(w0)*C_eff(w0)))
            Z_modes = np.asarray(Z_modes)
            
    elif(method=='MNA'):
        
        if(callable(Z_in) or callable(Z_out)): 
            raise ValueError('MNA doesn\'t support f-dependent Z_in/out!')
 
        # ---- node map ----
        if C_in is None:
            n1, n2 = 0, 1
            N = 2
        else:
            ns, n1, n2 = 0, 1, 2
            N = 3
        iL = N
        n = N + 1
     
        G = np.zeros((n, n)); Cm = np.zeros((n, n))
     
        def g_stamp(a, b, val, M):
            if a >= 0: M[a, a] += val
            if b >= 0: M[b, b] += val
            if a >= 0 and b >= 0: M[a, b] -= val; M[b, a] -= val
     
        # source / input branch
        if C_in is None:
            g_stamp(n1, -1, 1.0/Z_in, G)
        else:
            g_stamp(ns, -1, 1.0/Z_in, G)
            g_stamp(ns, n1, C_in*1e-15, Cm)
        # load
        g_stamp(n1, -1, 1.0/Z_out, G)
        # coupling and oscillator
        g_stamp(n1, n2, C_g*1e-15, Cm)
        g_stamp(n2, -1, C*1e-15, Cm)
        # inductor (current unknown)
        G[n2, iL] += 1.0; G[iL, n2] -= 1.0
        Cm[iL, iL] += L
     
        # ---- pencil (G + sC)x = 0 ----
        mu = np.linalg.eigvals(-np.linalg.solve(G, Cm))
        s = 1.0/mu[np.abs(mu) > 1e-25]
        s = s[np.imag(s) > 0]
     
        f_modes = np.imag(s)/(2e9*np.pi)
        kappas = -2*np.real(s)/(2e9*np.pi)
        Z_modes = [np.abs(s[0])*L]
    
    else:
        raise ValueError('Invalid method')

    if(mode_info):
        return kappas, f_modes, Z_modes
    else:
        return kappas


def kappa_osc_via_osc_mw(f_a, f_b, C_a, C_b, C, C_g, C_in=None, Z_in=50, Z_out=50,
                         Z_0=50, dist=0, c_cpw=169, method='zeros', band=None, 
                         dw_frac=0.1, n_scan=6001, n_aaa=4001, tolY=1e-8):
    """
       compute kappas, frequencies, and b-participation for a system 
       of two oscillators consequencially coupled to feedline as
       shown in the following:
       
       Z_in(w) --- C_in (optional) -------- --- Z_out(w)
                                     <-->  |
                                     dist  C
                                           |
                                        L_a&C_a
                                           |
                                          C_g
                                           |
                                        L_b&C_b

       there are to methods available:
       1) 'zeros' – computes equivalent Y(w) function and searches for
           its zeros x_k = w_k + i*y_k -> mode frequency w_k, kappa_k = 2*y_k
           ADVANTAGE – allow feedline and frequency dependent Z_in/out; 
           DROWBACK – can miss roots.
           DOI: 10.1103/RevModPhys.93.025005
           DOI: 10.1103/PhysRevLett.108.240502
           
       2) 'MNA' – use magic of linear lumped elements and combine effective
           linear equation (MNA equations in the Laplace domain) to find
           resonance modes. 
           ADVANTAGE – vectorized and stable;
           DROWBACK – can't handle feedline or complex Z_in/out.
           ISBN:978-0-442-28108-3
           
       Parameters:
       f_a/b : GHz, bare frequencies of oscillators
       C_a/b/g/in : fF
       Z_in/out : Ω
           can be value, or None (for termination), or a function of frequency (complex) in GHz
       Z_0 : CPW impedance, Ohm
       dist : distance between C_in and oscillator contact, mm
           (works only for C_in != None and method='zeros'!)
       c_cpw : CPW capacitance per length, fF/mm
       method : can be 'zeros' or 'MNA'
       band/dw_frac/n_scan/n_aaa=4001/tolY=1e-8 – params of find_Y_zeros 
           which are described in find_Y_zeros documentation
       
       Returns: 
       kappa_modes : 1-D np.array, GHz, 
       f_modes : 1-D np.array, GHz,
       p_b : 1-D np.array, participation of osc b
           (p_b ~ 1: b-like mode, p_b ~ 0: a-like mode)
       
    """
    w_a = 2*np.pi*f_a*1e9
    w_b = 2*np.pi*f_b*1e9
    L_a = 1/(w_a**2 * C_a*1e-15)
    L_b = 1/(w_b**2 * C_b*1e-15)

    # for terminated lines
    if Z_in is None: Z_in=float('inf')
    if Z_out is None: Z_out=float('inf')
        
    if(method=='zeros'):
        def Z_l(x):
            if(callable(Z_in)): 
                Z_1 = Z_in(x/2/np.pi/1e9)
            else:
                Z_1 = Z_in
                
            if(callable(Z_out)): 
                Z_2 = Z_out(x/2/np.pi/1e9)
            else:
                Z_2 = Z_out
                
            if C_in is not None:
                Z_1 = Z_1 + 1/(1j*x*C_in*1e-15)
                Z_1 = Z_of_cpw(x/2/np.pi/1e9, dist, c_cpw, Z_1, Z_0=Z_0)
                
            return 1/(1j*x*C*1e-15) + 1/(1/Z_1 + 1/Z_2)   
        
        def Y(x):
            x = 2*np.pi*1e9*x
            Z_a = 1/(1/Z_l(x) + 1j*x*C_a*1e-15 + 1/(1j*x*L_a)) + 1/(1j*x*C_g*1e-15)
            return 1/Z_a + 1j*x*C_b*1e-15 + 1/(1j*x*L_b)
    
        def v_a_over_v_b(x):
            """Mode voltage ratio from the C_g divider: v_a = v_b * Z_na/(Z_na+Z_Cg)."""
            x = 2*np.pi*1e9*x
            Z_na = 1/(1/Z_l(x) + 1j*x*C_a*1e-15 + 1/(1j*x*L_a))
            return Z_na/(Z_na + 1/(1j*x*C_g*1e-15))
    
        if band is None:
            band = (0.5*min(f_a, f_b), 1.5*max(f_a, f_b))
        uniq = find_Y_zeros(Y, band, dw_frac, n_scan, n_aaa, tolY)
    
        # capacitive-energy participation of oscillator b in each mode (v_b = 1)
        p_b = np.empty(uniq.size)
        for i, z in enumerate(uniq):
            E_a = C_a*1e-15*np.abs(v_a_over_v_b(z))**2
            E_b = C_b*1e-15
            p_b[i] = E_b/(E_a + E_b)
    
        f_modes = np.real(uniq)
        kappas = 2*np.imag(uniq)
            
    elif(method=='MNA'):
        
        if(callable(Z_in) or callable(Z_out)): 
            raise ValueError('MNA doesn\'t support f-dependent Z_in/out!')
            
        # ---- node map ----
        if C_in is None:
            n1, n2, n3 = 0, 1, 2
            N = 3
        else:
            ns, n1, n2, n3 = 0, 1, 2, 3
            N = 4
        iLa, iLb = N, N + 1
        n = N + 2
      
        G = np.zeros((n, n)); Cm = np.zeros((n, n))
     
        def g_stamp(a, b, val, M):
            if a >= 0: M[a, a] += val
            if b >= 0: M[b, b] += val
            if a >= 0 and b >= 0: M[a, b] -= val; M[b, a] -= val
     
        # source / input branch
        if C_in is None:
            g_stamp(n1, -1, 1.0/Z_in, G)
        else:
            g_stamp(ns, -1, 1.0/Z_in, G)
            g_stamp(ns, n1, C_in*1e-15, Cm)
        # load
        g_stamp(n1, -1, 1.0/Z_out, G)
        # chain
        g_stamp(n1, n2, C*1e-15, Cm)
        g_stamp(n2, -1, C_a*1e-15, Cm)
        g_stamp(n2, n3, C_g*1e-15, Cm)
        g_stamp(n3, -1, C_b*1e-15, Cm)
        # inductors (current unknowns)
        for node, ik, L in ((n2, iLa, L_a), (n3, iLb, L_b)):
            G[node, ik] += 1.0; G[ik, node] -= 1.0
            Cm[ik, ik] += L
     
        # ---- pencil (G + sC)x = 0 : nonzero eigenvalues of -G^{-1}C are 1/s ----
        mu, vec = np.linalg.eig(-np.linalg.solve(G, Cm))
        keep = np.abs(mu) > 1e-25
        s = 1.0/mu[keep]; vec = vec[:, keep]
        osc = np.imag(s) > 0                       # one of each conjugate pair
        s = s[osc]; vec = vec[:, osc]
     
        # ---- participation of resonator b in each mode (capacitive energy) ----
        Ea = C_a*1e-15*np.abs(vec[n2, :])**2
        Eb = C_b*1e-15*np.abs(vec[n3, :])**2
        p_b = Eb/(Ea + Eb)
     
        order = np.argsort(np.imag(s))
        f_modes = np.imag(s[order])/(2e9*np.pi)
        kappas = -2*np.real(s[order])/(2e9*np.pi)
        p_b = p_b[order]
    
    else:
        raise ValueError('Invalid method')
    
    return kappas, f_modes, p_b


def kappa_osc_via_osc_eq(f_a, f_b, g, kappa_a, regime='capacitor', C_in=0, dist_by_lambda=0, Z_0=50, mode_info=False):
    """
       compute kappas, frequencies, and b-participation for a system 
       of two oscillators consequencially coupled to feedline as
       shown in the following:

       feedline IN --- C_in (optional) -------- --- feedline OUT
                                         <-->  |
                                         dist  C
                                               |
                                          oscillator A
                                         (f_a, kappa_a)
                                               |
                                               g
                                               |
                                          oscillator B
                                             (f_b)

       Use input-output theory similarly to J.Heinsoo
       (DOI: https://doi.org/10.1103/PhysRevApplied.10.034040), but accaunting
       for phase accumulation between C and C_in. The equation also includes
       additional frequency terms to match Yen's equation in the area of strongly
       detuned oscillators (DOI:https://doi.org/10.1103/PhysRevApplied.23.024068).
       Specifically, the kappa_a is modified to give proper frequency dependence.
       The resulting equation for B oscillator kappa_b:

       kappa_b = kappa_a_m/2 - |Im{root(4*J^2 + (w_a_m - w_b - i*k_a_m/2)^2)}|,
       
       where:
       
       w_a_m = w_a + k_a/4*Im{gamma*exp(2*i*phi)}
       k_a_m = k_a/2*(1 + Re{gamma*exp(2*i*phi)}) * (w_b/w_a)**3*(2*w_b/(w_b+w_a))**2,
       
       and
       
       gamma = 1/(1 - 2j*w_b*Z_0*C_in)
       phi = 2*np.pi*dist_by_lambda*w_b/w_a
    
       Parameters:
       f_a,b : frequency, GHz (dressed by "single Z_0 port" and a-b capacitor shift)
       g : coupling, GHz (dressed by "single Z_0 port" and a-b capacitor shift)
       kappa_a : A-oscillator bandwith, GHz (for "single Z_0 port" load)
       regime : str
           defines state of port 1 input:
           'pass' – port 1 connected directly without capacitor (default)
           'capacitor' – port 1 connected via capacitor C_in
           'cut' – port 1 is not connected at all
       C_in : input capacitor, fF
       Z_0 : feadline impedance, Ω
       mode_info : bool
           return kappas, frequencies, and b-osc participation for both
           new modes representing [a, b] – IN THAT ORDER
       
       Returns:
       IF mode_info=False
       kappa_b : GHz
       IF mode_info=True
       [kappa_a, kappa_b] : np.array, GHz
       [freq_a, freq_b] : np.array, GHz
       [p_b_a, p_b_b] : np.array, float in [0, 1],
           fraction of B-oscillator in new "a" and "b" modes
       

    """ 
 
    w_a = 2*np.pi*f_a
    w_b = 2*np.pi*f_b
    J = 2*np.pi*g
    k_a = kappa_a*2*np.pi

    if(regime=='capacitor'):
        # phase accumulating between C_in and the reso connector
        phi = 2*np.pi*dist_by_lambda*f_b/f_a
        # input capacitor reflection
        gamma = 1/(1 - 2j*w_b*Z_0*C_in*1e-6)
        w_a_m = w_a + k_a/4*np.imag(gamma*np.exp(2j*phi))
        k_a_m = k_a/2*(1 + np.real(gamma*np.exp(2j*phi)))
    
    elif(regime=='pass'):
        phi = 0*e
        gamma = 0*e
        w_a_m = w_a
        k_a_m = k_a/2
        
    elif(regime=='cut'):
        phi=0*e
        gamma=1*e
        w_a_m = w_a
        k_a_m = k_a
    else:
        raise ValueError('Invalid regime, choose between: pass, cut, capacitor')

    # Yen equation inspired modification
    k_a_m = (w_b/w_a)**3 * (2*w_b/(w_b+w_a))**2 * k_a_m

    # diagonalisation of input-output equation system for a&b
    sigma = (w_a_m - w_b)/2 + k_a_m/4j
    sign = np.sign(np.imag(np.sqrt(J**2 + sigma**2)))

    w_a_new = (w_a_m + w_b)/2 + np.real(np.sqrt(J**2 + sigma**2))*sign
    w_b_new = (w_a_m + w_b)/2 - np.real(np.sqrt(J**2 + sigma**2))*sign
    k_a_new = k_a_m/2 + 2*np.imag(np.sqrt(J**2 + sigma**2))*sign
    k_b_new = k_a_m/2 - 2*np.imag(np.sqrt(J**2 + sigma**2))*sign
    # from eigenvectors
    eps = J/sigma
    norm_a_new = np.abs(eps)**2 + np.abs(1 + np.sqrt(eps**2+1))**2
    norm_b_new = np.abs(eps)**2 + np.abs(1 - np.sqrt(eps**2+1))**2
    p_b_a_new = np.abs(eps)**2/norm_a_new
    p_b_b_new = np.abs(eps)**2/norm_b_new

    k_new = np.asarray([k_a_new, k_b_new])
    w_new = np.asarray([w_a_new, w_b_new])
    p_b = np.asarray([p_b_a_new, p_b_b_new])
    
    if(mode_info):
        return k_new/2/np.pi, w_new/2/np.pi, p_b
    else:
        return k_new[1]/2/np.pi

    
def bbq_resonances_of_Y(f, Y, L, C, tolY=1e-12, jump_factor=8, method='complex', kappas=False):
    """
       Part of black box quantization, find resonances based on
       admittance Y(omega) + 1/(i*omega*L) + i*omega*C, there
       are to methods available:
       1) 'complex' – finds zeros of Y doing interpolation
       2) 'imag' – finds zeros of Im(Y) by searching for 
           sign changes
       
       Parameters:
       f : 1D np.array, GHz
           frequencies for Y(f)
       Y : 1D np.array, Ω
           admittance seen by JJ
       L : JJ inductance, nH
       C : JJ capacitance, fF
       jump_factor : float, (for 'imag' method)
           defines maximal slope near valid roots 
           (increase for narrow peaks)
       tolY : float, (for 'complex' method)
           defines boarder of abs(Y) in potential
           root to take it as root
       method : 'complex' or 'imag'
       kappas : if True – returns kappas for modes

       Returns:
       roots : 1D np.array, GHz
           resonant frequencies
       kappas : 1D np.array, GHz (IF kappas=True)
           resonant kappas
    """
    
    Y = Y + 1/(2j*np.pi*L*f) + 2j*np.pi*C*1e-6*f
    
    if(method=='complex'):
        if(_HAS_AAA):
            Y_inter = AAA(f.astype(complex), Y)
            z = find_Y_zeros(Y_inter, band=[f[0], f[-1]], tolY=tolY)
            roots = np.real(z)
            kappas = 2*np.imag(z)
        else:
            raise ValueError('No AAA interpolatr, update scipy (>1.15)!')
            
    elif(method=='imag'):  
        Y_im = np.imag(Y)
        roots = []
        avd = 0
        
        for n in range(Y_im.shape[0] - 1): avd += np.abs((Y_im[n+1]-Y_im[n]))
        
        for n in range(Y_im.shape[0] - 1):   
            if(Y_im[n+1]*Y_im[n] < 0 and np.abs((Y_im[n+1]-Y_im[n])) < avd/Y_im.shape[0]*jump_factor):
                roots.append((f[n+1]+f[n])/2)
    else:
        raise ValueError('Invalid method')

    if(kappas):
        return np.asarray(roots), np.asarray(kappas)
    else:
        return np.asarray(roots)


def scattering_osc_eq(f, f_a, kappa_a_0, kappa_a_1, gamma_a, regime='pass', 
                      C_in=None, dist_by_lambda=None, Z_0=50, mode_info=False):
    """
       compute scattering matrix for 3-port network with 
       readout resonator (a) coupled to a feedline.
       The network scheme:

       port 1 --- C_in (optional) --- --- port 0
                                     |
                                readout reso 
                    (f_a, kappa_a_0, kappa_a_1, gamma_a)
                                     |
                                   port 2
                                   
       WARNING
       1) The model based on input-output theory and assumes linear 
       response of oscillator => works for f close to f_a
       2) Since input-output theory assums exp(-i*omega*t)
       evolution, proper mw notation is j -> -i !!!
       
       Parameters:
       
       f : signal frequency, GHz
       f_a : reso frequency, GHz (dressed by "single Z_0 port")
       kappa_a_0 : kappa of reso to the main feedline, GHz (to "single Z_0 port")
       kappa_a_1 : kappa of reso to port 2 feedline, GHz (to "single Z_0 port")
       gamma_a : inner loss of read reso, GHz
       regime : str
           defines state of port 1 input:
           'pass' – port 1 connected directly without capacitor (default)
           'capacitor' – port 1 connected via capacitor C_in
           'cut' – port 1 is not connected at all
       mode_info : bool
           return vector of "a" steady state in basis of input signals
       
       Parameters for regime='capacitor':
       
       C_in : capacitance of input capacitor, fF
       dist_by_lambda : fraction of reso wavelength 
           distance bettwen input capacitor and filter to feedline connector
       Z_0 : feedline impedance, Ω

       Returns:
       S : scattering matrix, 2-D np.array
       a_v : oscillator steady state vector, 1-D np.array
       
    """

    # shape modifier in case of f is an numpy array
    try: e = np.ones(f.shape)
    except: e = 1
    
    w = 2*np.pi*f
    w_a = 2*np.pi*(f_a - f)
    k_a_0 = kappa_a_0*2*np.pi
    k_a_1 = kappa_a_1*2*np.pi

    # port 1 line configurating
    if(regime=='capacitor'):
        # phase accumulating between C_in and the reso connector
        phi = 2*np.pi*dist_by_lambda*f/f_a
        # input capacitor reflection
        gamma = 1/(1 - 2j*w*Z_0*C_in*1e-6)
        w_a_m = w_a + k_a_0/4*np.imag(gamma*np.exp(2j*phi))
        k_a_m = k_a_0/2*(1 + np.real(gamma*np.exp(2j*phi)))
    
    elif(regime=='pass'):
        phi = 0*e
        gamma = 0*e
        w_a_m = w_a
        k_a_m = k_a_0/2
        
    elif(regime=='cut'):
        phi=0*e
        gamma=1*e
        w_a_m = w_a
        k_a_m = k_a_0
    else:
        raise ValueError('Invalid regime, choose between: pass, cut, capacitor')

    # supportive variable
    d = 2j*w_a_m + k_a_m + k_a_1 + gamma_a*2*np.pi
        
    # building S matrix in basis (1, 2, 3)
    a_v = np.asarray([np.sqrt(k_a_0)*(1 + gamma*np.exp(2j*phi)),
                      np.sqrt(k_a_0)*(1 - gamma)*np.exp(1j*phi),
                      2*np.sqrt(k_a_1)*e])/d

    S = np.asarray([[gamma*np.exp(2j*phi), (1 - gamma)*np.exp(1j*phi), 0*e],
                    [(1 - gamma)*np.exp(1j*phi), gamma, 0*e],
                    [0*e, 0*e, 1*e]])

    S = S - np.asarray([np.sqrt(k_a_0)/2*(1 + gamma*np.exp(2j*phi))*a_v,
                        np.sqrt(k_a_0)/2*(1 - gamma)*np.exp(1j*phi)*a_v,
                        np.sqrt(k_a_1)*a_v])

    if(mode_info): return S, a_v
    else: return S


def scattering_osc_via_osc_eq(f, f_a, f_b, g, kappa_a, kappa_b, gamma_a, gamma_b, regime='pass', 
                              C_in=None, dist_by_lambda=None, Z_0=50, mode_info=False):
    """
       compute scattering matrix for 3-port network with 
       readout resonator (a) coupled to a feedline via  
       bandpass purcell filter (b). The network scheme:

       port 1 --- C_in (optional) --- --- port 0
                                     |
                              purcell filter
                         (f_a, kappa_a, gamma_a)
                                     |       
                                     g
                                     |
                                readout reso 
                          (f_b, kappa_b, gamma_b)
                                     |
                                   port 2

       This model is inspired by one decribed by J. Heinsoo in 
       DOI: https://doi.org/10.1103/PhysRevApplied.10.034040
       but includes MISSING phase asquired by wave between
       the input capavitor and filter connector!

       WARNING
       1) Works only for close f_a and f_b, because assumes 
       frequency-fixed kappa_a/b
       2) The model based on input-output theory and assumes linear 
       response of oscillator => works for f close to f_a
       3) Since input-output theory assums exp(-i*omega*t)
       evolution, proper mw notation is j -> -i !!!
       
       Parameters:
       
       f : signal frequency, GHz
       f_a : filter frequency, GHz (dressed by "single Z_0 port" and a-b capacitor)
       f_b : readout resonator frequency, GHz (dressed by "single Z_0 port" and a-b capacitor)
       g : readout reso to filter coupling, GHz
       kappa_a : kappa of filter to feedline, GHz (to "single Z_0 port")
       kappa_b : KAPPA OF READ RESO TO QUBIT, GHz (to "single Z_0 port")
       gamma_a/b : inner loss of filter/read reso, GHz
       regime : str
           defines state of port 1 input:
           'pass' – port 1 connected directly without capacitor (default)
           'capacitor' – port 1 connected via capacitor C_in
           'cut' – port 1 is not connected at all
       mode_info : bool
           return vectors of "a" and "b" steady states in basis of input signals
       
       Parameters for regime='capacitor':
       
       C_in : capacitance of input capacitor, fF
       dist_by_lambda : fraction of FILTER wavelength 
           distance bettwen input capacitor and filter to feedline connector
       Z_0 : feedline impedance, Ω

       Returns:
       S : scattering matrix, 2-D np.array
       a_v : oscillator steady state vector, 1-D np.array
       b_v : oscillator steady state vector, 1-D np.array
       
    """

    # shape modifier in case of f is an numpy array
    try: e = np.ones(f.shape)
    except: e = 1
    
    w = 2*np.pi*f
    w_a = 2*np.pi*(f_a - f)
    w_b = 2*np.pi*(f_b - f)
    J = 2*np.pi*g
    k_a = kappa_a*2*np.pi
    k_b = kappa_b*2*np.pi

    # port 1 line configurating
    if(regime=='capacitor'):
        # phase accumulating between C_in and the reso connector
        phi = 2*np.pi*dist_by_lambda*f/f_a
        # input capacitor reflection
        gamma = 1/(1 - 2j*w*Z_0*C_in*1e-6)
        w_a_m = w_a + k_a/4*np.imag(gamma*np.exp(2j*phi))
        k_a_m = k_a/2*(1 + np.real(gamma*np.exp(2j*phi)))
    
    elif(regime=='pass'):
        phi = 0*e
        gamma = 0*e
        w_a_m = w_a
        k_a_m = k_a/2
        
    elif(regime=='cut'):
        phi=0*e
        gamma=1*e
        w_a_m = w_a
        k_a_m = k_a
    else:
        raise ValueError('Invalid regime, choose between: pass, cut, capacitor')

    # supportive variables
    d_a = 2j*w_a_m + k_a_m + gamma_a*2*np.pi
    d_b = 2j*w_b + k_b + gamma_b*2*np.pi
        
    # building S matrix in basis (1, 2, 3)
    a_v = np.asarray([d_b*np.sqrt(k_a)*(1 + gamma*np.exp(2j*phi)),
                      d_b*np.sqrt(k_a)*(1 - gamma)*np.exp(1j*phi),
                      -4j*J*np.sqrt(k_b)*e])/(4*J**2 + d_a*d_b)

    b_v = np.asarray([-2j*J*np.sqrt(k_a)*(1 + gamma*np.exp(2j*phi))*e,
                      -2j*J*np.sqrt(k_a)*(1 - gamma)*np.exp(1j*phi)*e,
                      2*np.sqrt(k_b)*d_a])/(4*J**2 + d_a*d_b)

    S = np.asarray([[gamma*np.exp(2j*phi), (1 - gamma)*np.exp(1j*phi), 0*e],
                    [(1 - gamma)*np.exp(1j*phi), gamma, 0*e],
                    [0*e, 0*e, 1*e]])

    S = S - np.asarray([np.sqrt(k_a)/2*(1 + gamma*np.exp(2j*phi))*a_v,
                        np.sqrt(k_a)/2*(1 - gamma)*np.exp(1j*phi)*a_v,
                        np.sqrt(k_b)*b_v])

    if(mode_info): return S, a_v, b_v
    else: return S
    

def transmission_line_dressing_eq(f, kappa, regime='pass', C_in=None, dist_by_lambda=None, Z_0=50, f_s=None):
    """
       compute dressed f and kappa (at f_s frequency point) 
       for an oscillator coupled to transmission line:

       port 1 --- C_in (optional) --- --- port 0
                                     |
                                   reso 
                                (f, kappa)
                                   
       WARNING
       1) The model based on input-output theory and assumes linear 
       response of oscillator => works for close f and f_s
       2) Since input-output theory assums exp(-i*omega*t)
       evolution, proper mw notation is j -> -i !!!
       
       Parameters:
       
       f : reso frequency, GHz (dressed by "single Z_0 port")
       kappa : kappa of reso to the feedline, GHz (to "single Z_0 port")
       regime : str
           defines state of port 1 input:
           'pass' – port 1 connected directly without capacitor (default)
           'capacitor' – port 1 connected via capacitor C_in
       
       Parameters for regime='capacitor':
       
       C_in : capacitance of input capacitor, fF
       dist_by_lambda : fraction of signal wavelength (f_s)
           distance bettwen input capacitor and filter to feedline connector
       Z_0 : feedline impedance, Ω
       f_s : frequency of kappa onservation, equal to f by default, GHz

       Returns:
       f_dressed : GHz
       kappa_dressed : GHz
       
    """
    
    if f_s is None: f_s=f
        
    w = 2*np.pi*f
    w_s = 2*np.pi*f_s
    k = kappa*2*np.pi
    
    # port 1 line configurating
    if(regime=='capacitor'):
        # phase accumulating between C_in and the reso connector
        phi = 2*np.pi*dist_by_lambda
        # input capacitor reflection
        gamma = 1/(1 - 2j*w_s*Z_0*C_in*1e-6)
        w_m = w + k/4*np.imag(gamma*np.exp(2j*phi))
        k_m = k/2*(1 + np.real(gamma*np.exp(2j*phi)))
    
    elif(regime=='pass'):
        w_m = w
        k_m = k/2
    
    else:
        raise ValueError('Invalid regime, choose between: pass, capacitor')

    return w_m/2/np.pi, k_m/2/np.pi


def transmission_line_baring_eq(f, kappa, regime='pass', C_in=None, dist_by_lambda=None, Z_0=50, f_s=None):
    """
       compute bare (dressed by "single Z_0 port") 
       f and kappa (at f_s frequency point) for an oscillator 
       coupled to transmission line:

       port 1 --- C_in (optional) --- --- port 0
                                     |
                                   reso 
                                (f, kappa)
                                   
       WARNING
       1) The model based on input-output theory and assumes linear 
       response of oscillator => works for close f and f_s
       2) Since input-output theory assums exp(-i*omega*t)
       evolution, proper mw notation is j -> -i !!!
       
       Parameters:
       
       f : reso frequency, GHz (dressed by transmission line)
       kappa : kappa of reso to the feedline, GHz (dressed by transmission line)
       regime : str
           defines state of port 1 input:
           'pass' – port 1 connected directly without capacitor (default)
           'capacitor' – port 1 connected via capacitor C_in
       
       Parameters for regime='capacitor':
       
       C_in : capacitance of input capacitor, fF
       dist_by_lambda : fraction of signal wavelength (f_s)
           distance bettwen input capacitor and filter to feedline connector
       Z_0 : feedline impedance, Ω
       f_s : frequency of kappa onservation, equal to f by default, GHz

       Returns:
       f_bare : GHz (dressed by "single Z_0 port")
       kappa_bare : GHz (dressed by "single Z_0 port")
       
    """
    
    if f_s is None: f_s=f
        
    w = 2*np.pi*f
    w_s = 2*np.pi*f_s
    k = kappa*2*np.pi
    
    # port 1 line configurating
    if(regime=='capacitor'):
        # phase accumulating between C_in and the reso connector
        phi = 2*np.pi*dist_by_lambda
        # input capacitor reflection
        gamma = 1/(1 - 2j*w_s*Z_0*C_in*1e-6)
        k_bare = 2*k/np.real(1 + gamma*np.exp(2j*phi))
        w_bare = w - k_bare/4*np.imag(gamma*np.exp(2j*phi))
    elif(regime=='pass'):
        w_bare = w
        k_bare = 2*k
    
    else:
        raise ValueError('Invalid regime, choose between: pass, capacitor')

    return w_bare/2/np.pi, k_bare/2/np.pi

    