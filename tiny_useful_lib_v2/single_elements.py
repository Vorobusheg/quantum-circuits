import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.linalg import cosm
from scipy.optimize import minimize
from .math import dagger
from .equations import phi_zpf, n_zpf, Z_of_osc, f_of_osc


def oscillator(f, Z=None, truncation=20):
    """
       defines oscillator spectrum and operators
       Hamiltonian: H/h = f*at*a
        
       Parameters:

       f : GHz, oscillator frequency
       Z : Ω, default: None
           oscillator impedance
       truncation : int, default: 20
           number of levels in the final spectrum

       Returns:
       spectrum : 1-D np.array
       at : 2-D np.array (in case of Z=None)
       a : 2-D np.array (in case of Z=None)
       phi : 2-D np.array (in case of Z!=None)
       n : 2-D np.array (in case of Z!=None)
    """
    spectrum = np.linspace(0, f * (truncation - 1), truncation)

    # annihilation operator
    a = np.zeros((truncation, truncation), dtype=complex)
    for n in range(truncation - 1):
        a[n, n + 1] = np.sqrt(n + 1)

    # creation operator
    at = np.zeros((truncation, truncation), dtype=complex)
    for n in range(truncation - 1):
        at[n + 1, n] = np.sqrt(n + 1)

    if(Z==None):
        return (spectrum, at, a)
    else:
        phi = phi_zpf(Z)*(at + a)
        n = n_zpf(Z)*1j*(at - a)
        return (spectrum, phi, n)

        
def fluxonium(Ej, El, Ec, truncation=10, osc_truncation=None, F=0, Q=0, rounding=True):
    """
       compute fluxonium spectrum and operators in diagonalized basis
       Hamiltonian: H/h = 4*Ec*n^2 + El/2*phi^2 - Ej*cos(phi)
       Diagonalization via Ej*cos(phi) perturbation of oscillator(El, Ec)
        
       Parameters:
       
       Ej : GHz
       El : GHz
       Ec : GHz
       truncation : int, default: 10
           number of levels in the final fluxonium spectrum
       osc_truncation : int, default: int(1.5*truncation) + 20
           number of levels in the initial oscillator
       F : flux quanta
           external constant flux
       Q : couper pair number
           external constant charge
       rounding : bool
           round phi and n to -13 order to exclude numerical errors

       Returns:
       
       spectrum : 1-D np.array
       phi : 2-D np.array
       n : 2-D np.array
    """

    if(osc_truncation == None):
        osc_truncation = int(1.5*truncation) + 20

    # definition of the initial oscillator
    f = f_of_osc(El, Ec)
    _, at, a = oscillator(f, truncation=osc_truncation)
    _, phi, n = oscillator(f, Z=Z_of_osc(El, Ec), truncation=osc_truncation)

    # perturbation of the oscillator
    H = f*at@a - Ej*cosm(phi) + 8*Ec*n*Q + El*phi*(2*np.pi*F)
    (e, v) = eigsh(H, k=truncation, which='SA', maxiter=5000)
    sorted_indices = np.argsort(e)
    spectrum = e[sorted_indices]
    eig_states = v[:, sorted_indices]

    # fix phase gauge
    for k in range(eig_states.shape[1]):
        psi_0 = np.abs(eig_states[:, k]).argmax()
        rot = np.conjugate(eig_states[psi_0, k]/abs(eig_states[psi_0, k]))
        eig_states[:, k]=eig_states[:, k]*rot
        
    # transfer of phi and n to the fluxonium eigen basis
    phi = dagger(eig_states)@phi@eig_states
    n = dagger(eig_states)@n@eig_states

    if(rounding):
        phi = np.around(phi, 13)
        n = np.around(n, 13)

    # spectrum alignment
    spectrum = spectrum - spectrum[0]
    
    return (spectrum, phi, n)


def conj_variables_in_q_repres(grid=0, h=0, left_border=0):
    """
       build matrices of conjugated q and p in q-grid basis

       Parameters:
       
       grid : int
           size of q-grid
       h : float
           step of q-grid
       left_border : float
           left border of the q-grid
       
       Returns:
       
       q : 2-D np.array
       p : 2-D np.array
       
    """
    q = np.zeros((grid, grid), dtype=complex)
    p = np.zeros((grid, grid), dtype=complex)

    for n in range(grid):
        
        q[n, n] = h * n + left_border

        if (n == 0):
            p[n, n + 1] = -1j / (2 * h)
        elif (n == grid - 1):
            p[n, n - 1] = 1j / (2 * h)
        else:
            p[n, n + 1] = -1j / (2 * h)
            p[n, n - 1] = 1j / (2 * h)

    return (q, p)

    
def transmon(Ej, Ec, truncation=10, grid=None, Q=0, rounding=True):
    """
       compute transmon spectrum and operators in diagonalized basis
       Hamiltonian: H/h = 4*Ec*n^2 - Ej*cos(phi)
       Diagonalization via Ej*cos(phi) perturbation of oscillator(El, Ec)
        
       Parameters:
       
       Ej : GHz
       Ec : GHz
       truncation : int, default: 10
           number of levels in the final transmon spectrum
       grid : int, default: int(1.5*truncation) + 10
           2*grid + 1 = size of cooper-pair grid used for the diagonalization
       Q : couper pair number
           external constant charge
       rounding : bool
           round phi and n to -13 order to exclude numerical errors

       Returns:
       
       spectrum : 1-D np.array
       phi : 2-D np.array
       n : 2-D np.array
    """
    if(grid == None): grid=int(1.5*truncation) + 20

    # transmon Hamilonian on the n-grid (grid step = 1 for cooper-pairs)
    H = np.zeros((2 * grid + 1, 2 * grid + 1), dtype=complex)
    for k in range(2 * grid + 1):

        n = k - grid

        if (k == 0):
            H[k, k] = 4*Ec * (n + Q) ** 2
            H[k, k + 1] = -Ej/2
        elif (k == 2 * grid):
            H[k, k] = 4*Ec * (n + Q) ** 2
            H[k, k - 1] = -Ej/2
        else:
            H[k, k] = 4*Ec * (n + Q) ** 2
            H[k, k - 1] = -Ej/2
            H[k, k + 1] = -Ej/2

    # diagonalization
    (e, v) = eigsh(H, k=truncation, which='SA', maxiter=5000)
    sorted_indices = np.argsort(e)
    spectrum = e[sorted_indices]
    eig_states = v[:, sorted_indices]

    # fix phase gauge
    for k in range(eig_states.shape[1]):
        for el in np.flip(eig_states[:, k]): 
            if(abs(el) > 1e-1): 
                rot = -np.conjugate(el/abs(el))
                break
        eig_states[:, k]=eig_states[:, k]*rot
        
    # getting phi and n matrices on the grid
    (n, phi) = conj_variables_in_q_repres(grid=2 * grid + 1, h=1, left_border=-grid)
    
    # transfer of phi and n to the fluxonium eigen basis
    phi = dagger(eig_states)@phi@eig_states
    n = dagger(eig_states)@n@eig_states

    if(rounding):
        phi = np.around(phi, 11)
        n = np.around(n, 11)
            
    # spectrum alignment
    spectrum = spectrum - spectrum[0]
    
    return (spectrum, phi, n)


def transmon_search(f01, alpha, bounds=[(2, 100), (0.01, 3)], x0=[40, 1.5], truncation=5):
    """
       find energies of transmon, which has given f01 and anharmonicity,
       using scipy.optimize.minimize
        
       Parameters:
       
       f01 : GHz
           frequency of 0-1 transition
       alpha : GHz
       bounds : [(Ej_min, Ej_max), (Ec_min, Ec_max)], default: [(2, 100), (0.01, 3)]
           defines area of energy search
       x0 : 1-D np.array or list, [Ej_0, Ec_0], default: [40, 1.5]
           initial point for the optimizer
       truncation : int
           defines truncation of transmon in optimizator
           
       Returns:
       
       E_opt : 1-D np.array, GHz (Ej, Ec)
    """
    def loss(x):
        eigval, _, q = transmon(x[0], x[1], truncation=truncation)
        return (eigval[1] - f01)**2 + (eigval[2] - 2*eigval[1] - alpha)**2
    
    E_opt = minimize(loss, x0=x0, bounds=bounds).x
    
    return E_opt


def fluxonium_search(f01, f12=None, f03=None, alpha=None, F=0,
                     bounds=[(2, 100), (0.1, 4), (0.1, 2)], 
                     x0_list=[[4, 1, 1]],
                     weights=[1, 1, 1], truncation=5):
    """
       finds energies of fluxonium, which has given spectrum parameters:
       1) f01, f12, f03 - for this regime don't define alpha
       2) f01, alpha - for this regime don't define f12 and f03
       taking arbitrary number of initial points (x0_list), applies
       scipy.optimize.minimize to each and returns the best result
       
       Parameters:
       
       f01 : GHz
       f12 : GHz, default: None
       f03 : GHz, default: None
       alpha : GHz, default: None
       F : flux quanta
           external constant flux
       bounds : [(Ej_min, Ej_max), (El_min, El_max), (Ec_min, Ec_max)], default: [(2, 100), (0.2, 1.5), (0.1, 2)]
           defines area of energy search
       x0_list : 2-D np.array or list, [[Ej_0, Ec_0], [Ej_0, Ec_0], ...], default: [[4, 1, 1]]
           list of initial points for the optimizer, each point corresponds to distinct optimization try
       weights : 1-D np.array or list, default: [1, 1, 1]
           defines weights of energies in the optimization (Ej, El, Ec)
       truncation : int, default: 5
           defines truncation of fluxonium in optimizator
        
       Returns:
       
       E_opt : 1-D np.array, GHz (Ej, El, Ec)
    """

    # choose search regime
    if(alpha == None):
        def loss(x):
            eigval, _, _ = fluxonium(x[0], x[1], x[2], truncation=truncation, F=F)
            loss=weights[0]*(eigval[1] - f01)**2 +\
                 weights[1]*(eigval[2] - eigval[1] - f12)**2 +\
                 weights[2]*(eigval[3] - f03)**2
            return loss
    elif(f12==None and f03==None):
        def loss(x):
            eigval, _, q = fluxonium(x[0], x[1], x[2], truncation=truncation, F=F)
            return (eigval[1] - f01)**2 + (eigval[2] - 2*eigval[1] - alpha)**2
        
    # optimization with different initial points
    loss_vals = []
    x_vals = []
    
    for x0 in x0_list:
        opt = minimize(loss, x0=x0, bounds=bounds)
        loss_vals.append(opt.fun)
        x_vals.append(opt.x)

    loss_vals = np.asarray(loss_vals)
    x_vals = np.asarray(x_vals)
    E_opt = x_vals[np.argmin(loss_vals)]
    
    return E_opt

