import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import eigsh
from scipy.linalg import cosm, eigh, eigh_tridiagonal
from scipy.optimize import minimize
from scipy.special import eval_genlaguerre, gammaln
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


def conj_variables_in_q_repres(grid=0, h=0, left_border=0, method='sinc'):
    """
       build matrices of conjugated q and p = -i d/dq in q-grid basis, [q, p] = i

       Parameters:

       grid : int
           size of q-grid
       h : float
           step of q-grid
       left_border : float
           left border of the q-grid
       method : 'sinc' (default) or 'fd'
           'sinc' : spectral (sinc) derivative, p_jk = -i (-1)^(j-k) / ((j - k) h), p_jj = 0.
                    Exact for wavefunctions that are band-limited on the grid, so the error
                    falls exponentially with 1/h; p @ p is a correct kinetic term.
                    With h = 1 on an integer charge grid it is exactly the compact phase:
                    phi = -p, <n|phi|m> = i (-1)^(n-m) / (n - m), phi in (-pi, pi).
           'fd'   : 3-point central difference (previous behavior), error ~ h^2.
                    On an integer charge grid it gives -sin(phi) instead of -phi, and
                    p @ p couples only every second grid point (doubled spectrum).

       Returns:

       q : 2-D np.array
       p : 2-D np.array

    """
    k = np.arange(grid)
    q = np.diag(left_border + h*k).astype(complex)

    if method == 'sinc':
        d = np.subtract.outer(k, k)
        with np.errstate(divide='ignore', invalid='ignore'):
            p = np.where(d != 0, -1j*(-1.0)**d/(d*h), 0)
    else:
        p = (-1j*np.eye(grid, k=1) + 1j*np.eye(grid, k=-1))/(2*h)

    return (q, p)

    
def transmon(Ej, Ec, truncation=10, grid=None, Q=0, rounding=True, exp_phi=False):
    """
       compute transmon spectrum and operators in diagonalized basis
       Hamiltonian: H/h = 4*Ec*(n + Q)^2 - Ej*cos(phi)
       Diagonalization in the charge basis, where H is real and tridiagonal

       Parameters:

       Ej : GHz
       Ec : GHz
       truncation : int, default: 10
           number of levels in the final transmon spectrum
       grid : int, default: int(1.5*truncation) + 20, larger for Ej/Ec > ~300
           2*grid + 1 = size of cooper-pair grid used for the diagonalization,
           centred at n = -round(Q)
       Q : cooper pair number
           external constant charge, any real number
       rounding : bool
           round phi, n (and exp_iphi) to -11 order to exclude numerical errors
       exp_phi : bool, default: False
           also return exp(i*phi) in the diagonalized basis

       Returns:

       spectrum : 1-D np.array
       phi : 2-D np.array
           junction phase in (-pi, pi]; exact for states localized near phi = 0 (transmon regime),
           cut-dependent for delocalized states (small Ej/Ec, levels near the top of the cosine)
       n : 2-D np.array
           integer part of the charge, the total charge is n + Q
       exp_iphi : 2-D np.array, only if exp_phi=True
           (no branch cut, valid in every regime)
    """
    if(grid is None):
        # previous default, enlarged for large Ej/Ec: the charge reached by the highest requested level
        # (transmon ladder or free rotor) plus a margin for its tail, and the charge tail
        # up to ~sqrt(Ej/Ec) that the charge dispersion needs
        K = max(truncation, 4)
        n_top = max(np.sqrt(K*np.sqrt(8*Ej*Ec)/(4*Ec)), K/2)
        grid = max(int(1.5*truncation) + 20,
                   int(np.ceil(n_top + 3 + 0.3*np.sqrt(Ej/Ec))),
                   int(np.ceil(np.sqrt(Ej/Ec) + 3)))

    # cooper-pair grid n0 - grid ... n0 + grid, centred at n0 = -round(Q), so that n + Q stays near 0 for any Q;
    # q = n and p = -i d/dn = -phi (sinc derivative: exact phase in (-pi, pi])
    (q, p) = conj_variables_in_q_repres(grid=2*grid + 1, h=1, left_border=-round(Q) - grid)
    n_grid = q.diagonal().real

    # diagonalization: H is real and tridiagonal, diagonal 4Ec(n + Q)^2, off-diagonals -Ej/2.
    # Bisection with tol = 2*tiny finds every level to full precision of its own size,
    # not of the largest diagonal element 4Ec*grid^2, so small charge dispersions stay exact
    (spectrum, eig_states) = eigh_tridiagonal(4*Ec*(n_grid + Q)**2, np.full(2*grid, -Ej/2),
                                              select='i', select_range=(0, truncation - 1),
                                              tol=2*np.finfo(float).tiny)

    # fix phase gauge: the last component with |v| > 0.1 is made real negative;
    # if no component is that large, the largest one is used
    big = np.abs(eig_states) > 1e-1
    pivot = np.where(big.any(axis=0), 2*grid - np.argmax(big[::-1], axis=0),
                     np.argmax(np.abs(eig_states), axis=0))
    eig_states = eig_states*np.sign(-eig_states[pivot, np.arange(truncation)])

    # transfer of phi and n to the transmon eigen basis (eig_states are real);
    # all operators are returned as complex matrices (n and exp(i*phi) are real in this gauge)
    phi = eig_states.T @ (-p) @ eig_states
    n = (eig_states.T @ (n_grid[:, None]*eig_states)).astype(complex)

    if(rounding):
        phi = np.around(phi, 11)
        n = np.around(n, 11)

    # spectrum alignment
    spectrum = spectrum - spectrum[0]

    if(exp_phi):
        exp_iphi = (eig_states[1:].T @ eig_states[:-1]).astype(complex)
        if(rounding):
            exp_iphi = np.around(exp_iphi, 11)
        return (spectrum, phi, n, exp_iphi)

    return (spectrum, phi, n)


def displacement_in_Fock_basis(alpha, dim):
    """
       build a matrix of displacement operator in Fock basis 
       truncated with the given dimention
       
       Parameters:

       alpha : shift amplitude
       dim : Fock space dimention

       Returns:

       D : 2-D np.array
    """

    def D_elem(alpha, n, m):
    # for n > m
        el = np.exp((gammaln(m+1) - gammaln(n+1))/2) * np.exp(-np.abs(alpha)**2/2) *\
             alpha**(n-m) * eval_genlaguerre(m, n-m, np.abs(alpha)**2)
        return el
    
    D = np.zeros((dim, dim), dtype=complex)
    for n in range(dim):
        D[n, n] = D_elem(alpha, n, n)
        for m in range(n):
            D[n, m] = D_elem(alpha, n, m)
            D[m, n] = (-1)**(n-m) * np.conjugate(D[n, m])

    return D

def transmon_L(Ej, Ec, Ecj, El, truncation=10, grid=None, N=None, Q=0, rounding=True, exp_phi=False):
    """
       diagonalize transmon with sequential inductance, return spectrum
       and operators in diagonalized basis. Circuit:
 
        0_ L (El) _1________
        |          |        |
       C (Ec)     X (Ej)  Cj (Ecj)
        |__________|________|
 
       Variables definitions:
       (φ, n) and (θ, m)
       where φ = φ_1 (junction) and θ = φ_0 - φ_1 (across L),
       In case of external coupling between capacitor and inductance (big pads),
       all coupling goes to "m" (usually)
 
       Hamiltonian in circuit frame:
       H = 4*Ec*m^2 + 4*Ecj*(n + Q - m)^2 + El/2*θ^2 - Ej*cos(φ)
 
       Hamiltonian in polaron frame:
       H = sqrt(8*El(Ec + Ecj)) * at*a + 4*Ec/(1 + Ec/Ecj)*(n + Q)^2 - Ej/2*(e^i*phi * D(-alpha) + e^-i*phi * D(+alpha))
       where alpha = (El/2/Ecj)^-1/4 * (1 + Ec/Ecj)^-3/4
 
       Parameters:
 
       Ej, Ec, Ecj, El : GHz  
       truncation : int, default: 10
           number of levels in the final transmon spectrum
       grid : int, default: cutoff rule (requested levels within ~1 kHz)
           the rule: grid=ceil(max(grid, sqrt(min(Ejs/Ecs, 140)) + 1.5))
           2*grid + 1 = size of cooper-pair grid used for the diagonalization,
           centred at n = -round(Q)
       N : int, default: cutoff rule
           number of Fock states of the internal (L - Cj) mode
       Q : cooper pair number
           external constant charge, any real number
       rounding : bool
           round the operators to -11 order to exclude numerical errors
       exp_phi : bool, default: False
           also return exp(i*φ) of the junction in diagonalized basis
 
       Returns:
 
       spectrum : 1-D np.array
       φ, n : 2-D np.arrays
       θ, m : 2-D np.arrays
       exp_iφ : 2-D np.array, only if exp_phi=True
           (no branch cut, valid in every regime)
 
       φ = φ_cm - λθ: the branch cut (-pi, pi] is on the slow island phase φ_cm (polaron frame),
       θ is exact. Reliable for states localized in φ_cm (transmon regime, low levels);
       the diagonal ones of the total charge and of the charge on C are n + Q and m + Q/(1 + Ec/Ecj).
    """
 
    # define supportive variables
    gamma = El/Ecj/2
    eps = Ec/Ecj
    alpha = 1/(gamma*(1 + eps)**3)**0.25
    Ecs = Ec/(1 + eps)                                   # e^2/2(C + C_J): charging energy in the polaron frame
    beta = Ej/(El*(1 + eps)**2)                          # series-inductance parameter
    Ejs = Ej*np.exp(-alpha**2/2)                            # zero-point-renormalized Ej
    wp = np.sqrt(8*Ejs*Ecs)                                 # transmon plasma frequency
    w = np.sqrt(8*El*(Ec + Ecj))                            # internal (L - C_J) mode frequency
 
    if grid is None:
        # charge cutoff: charge reached by the highest requested level (transmon ladder or free rotor,
        # at least 4 levels counted) plus a margin for the tail beyond the turning point,
        # which decays more slowly for large Ej/Ec
        K = max(truncation, 4)
        n_top = max(np.sqrt(K*wp/(4*Ecs)), K/2)
        grid = n_top + 2 + 0.2*np.sqrt(Ejs/Ecs)
        # floor for the 0-1 charge dispersion: it needs the charge tail up to about sqrt(Ejs/Ecs);
        # above Ejs/Ecs ~ 140 the dispersion is below ~1 Hz and the floor stops growing
        grid = int(np.ceil(max(grid, np.sqrt(min(Ejs/Ecs, 140)) + 1.5)))
 
    if N is None:
        # Fock cutoff: smallest N for which the 2nd-order error of the dropped Fock states k >= N,
        # (Ej^2/4w) e^(-alpha**2) sum_{k>=N} alpha**2^k/(k k!), is below tol.
        # t is the first dropped term; the rest of the tail is bounded by a geometric series
        # with ratio alpha**2/(N + 1), so the whole tail is at most t/(1 - alpha**2/(N + 1)).
        tol = 1e-7
        t, N = Ej**2/(4*w)*np.exp(-alpha**2)*alpha**2, 1
        while t >= tol*(1 - alpha**2/(N + 1)):
            t *= alpha**2*N/(N + 1)**2                # next term: alpha**2^(N+1)/((N+1)(N+1)!)
            N += 1
        # margins for beyond-2nd-order dressing (beta ~ 1)
        # and for internal-mode quanta that fall inside the requested spectrum
        n_osc = int((truncation - 1)*np.sqrt(8*Ejs*Ecs/(1 + beta))/w)
        N += int(np.ceil(2 + 3*beta)) + 2*n_osc
 
    # cooper-pair grid n0 - grid ... n0 + grid, centred at n0 = -round(Q), so that n + Q stays near 0 for any Q;
    # q = n and p = -i d/dn = -φ (sinc derivative: exact phase in (-pi, pi])
    nc = 2*grid + 1
    (q, p) = conj_variables_in_q_repres(grid=nc, h=1, left_border=-round(Q) - grid)
    n = q.diagonal().real
 
    # Hamiltonian in the product basis |n>|k> (charge-major order): real and block-tridiagonal,
    # diagonal 4Ecs(n + Q)^2 + w k, blocks <n+1|H|n> = -Ej/2 D(-alpha) = -Ej/2 D(alpha)^T, plus h.c.
    D = displacement_in_Fock_basis(alpha, N).real                       # D(+alpha), real for real alpha
    dim = nc*N
    diag = (4*Ecs*(n[:, None] + Q)**2 + w*np.arange(N)).ravel()
 
    # diagonalization: lowest levels
    if dim <= 400:
        # small matrix: dense array filled block by block (no kron), LAPACK dsyevr for the lowest levels.
        # Only the lower triangle is filled: eigh(lower=True) reads only that triangle.
        H = np.zeros((dim, dim))
        H.flat[::dim + 1] = diag
        b = np.arange(nc - 1)
        H.reshape(nc, N, nc, N)[b + 1, :, b, :] = -Ej/2*D.T
        e, v = eigh(H, lower=True, subset_by_index=[0, truncation - 1], overwrite_a=True, check_finite=False)
    else:
        # larger matrix: sparse storage, shift-invert Lanczos (ARPACK on (H - sigma)^-1 via sparse LU)
        S = sps.diags(np.ones(nc - 1), -1)                              # e^{iφ}: |n> -> |n+1>
        Ds = sps.csr_matrix(D)
        H = (sps.diags(diag) - Ej/2*(sps.kron(S, Ds.T) + sps.kron(S.T, Ds))).tocsc()
        # sigma below the whole spectrum: the Josephson term -Ej/2 (e^{iφ} D(-alpha) + h.c.) has norm <= Ej,
        # the other terms are >= min 4Ecs(n + Q)^2. Then 'LM' of (H - sigma)^-1 = the lowest levels of H.
        sigma = 4*Ecs*np.min((n + Q)**2) - Ej - 1
        e, v = eigsh(H, k=truncation, sigma=sigma, which='LM')
    order = np.argsort(e)
    spectrum, v = e[order], v[:, order]
 
    # fix phase gauge: the last component with |v| > 0.1 is made real negative;
    # if no component is that large, the largest one is used
    nl = v.shape[1]
    big = np.abs(v) > 1e-1
    pivot = np.where(big.any(axis=0), dim - 1 - np.argmax(big[::-1], axis=0), np.argmax(np.abs(v), axis=0))
    v = v*np.sign(-v[pivot, np.arange(nl)])                            # v is real
 
    # transfer of operators to the eigen basis. The eigenvectors are reshaped as (charge, Fock, level)
    # and the operators act on them directly
    V = v.reshape(nc, N, nl)
    aV = np.zeros_like(V)
    aV[:, :-1, :] = np.sqrt(np.arange(1, N))[None, :, None]*V[:, 1:, :]       # (1 x a) V
    A = v.T @ aV.reshape(dim, nl)                                             # <i|a|j>;  <i|at|j> = A.T
    n_mat = v.T @ (n[:, None, None]*V).reshape(dim, nl)                       # <i|n|j>
    phi_c = v.T @ np.tensordot(-p, V, axes=(1, 0)).reshape(dim, nl)           # <i|φ|j> on the charge grid
 
    # after polaron transformation:
    # φ -> φ + i*alpha*(at - a), n -> n
    # θ -> i*((1+eps)/gamma)^0.25*(a - at), m -> 0.5*(gamma/(1 + eps))^0.25*(at + a + 2*alpha*n)
    th_zpf = ((1 + eps)/gamma)**0.25
    phi = phi_c + 1j*alpha*(A.T - A)
    theta = 1j*th_zpf*(A - A.T)
    m = (A + A.T)/(2*th_zpf) + n_mat/(1 + eps)
 
    out = [phi, n_mat, theta, m]
    if exp_phi:
        W = np.zeros_like(V)
        W[1:] = np.einsum('kl,clj->ckj', D.T, V[:-1])
        out.append(v.T @ W.reshape(dim, nl))
    # all operators are returned as complex matrices (n, m and exp(iφ) are real in this gauge)
    out = [np.asarray(x, dtype=complex) for x in out]
    if rounding:
        out = [np.around(x, 11) for x in out]
 
    # spectrum alignment
    spectrum = spectrum - spectrum[0]
 
    res = (spectrum, (out[0], out[1]), (out[2], out[3]))
    return res + (out[4],) if exp_phi else res

    
def transmon_search(f01, alpha, bounds=[(2, 100), (0.05, 0.8)], x0=[40, 0.2], truncation=5):
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