import numpy as np
from scipy.optimize import minimize, dual_annealing
from pandas import DataFrame

# physical constants
e=1.6*1e-19
hbar=1.05*1e-34
                
# some staff to manage capacitance-energy matrixes

# sum non-diognal elements and return 1D array of full elements capacitances
def sum_to_diag(M):

    diag = np.zeros(M.shape[0])
    for n in range(M.shape[0]):
        for m in range(M.shape[0]):
            diag[n] += M[n, m] + M[m, n]
        diag[n] -= M[n, n]

    return diag

# subtract non-diognal elements for diognal ones
def subtract_from_diag(M):

    M_out = np.copy(M)
    
    for n in range(M.shape[0]):
        for m in range(M.shape[0]):
            M_out[n, n] -= (M[n, m] + M[m, n])
        M_out[n, n] += 2*M[n, n]

    return M_out

# need to be used after subspace on capacitance matrix to obtain an upper triangular matrix
def sym_to_triangle(M_in):

    M = np.copy(M_in)

    # symmetrisation
    for n in range(M.shape[0]):
        for m in range(M.shape[1]):
            
            if(M[n, m] != 0): M[m, n] = M[n, m]

    # to triangle
    for n in range(M.shape[0]):
        for m in range(n):
            M[n, m] = 0

    return M


def lom_c_matrix_convert(C_init, island_sequence, pins=[], physical=False):
    """
       Converts capacitance matrix given by qiskit_metal.analyses.quantization.LOManalysis 
       - excludes Ground, separates islands and pins, and gives convinient names

       Parameters:
       
       C_init : DataFrame
       island_sequence : python list in the form ['old_name -> new_name (optional)', ...]
           defines the desired order of islands in the resulting matrix,
           also can rename them (optional)
       pins : python list in the form ['old_name -> new_name (optional)', ...], default: []
           defines which elements are treated as pins,
           ATTENTION len(island_sequence) + len(pins) must be equial to 
           the C_init dimention - 1 (subtracting ground)
       physical : bool, default: False
           defines output C form: 
               True - returns C in the physical form, 
               False - returns C in the reduced form

       Returns:
       C : DataFrame
       C_pins : DataFrame (only if not empty)
    """
    
    # complete island set check
    if(C_init.columns.size - 1 != len(island_sequence) + len(pins)): raise ValueError('The system is not complete!')

    indices = []
    isl_name_new = []

    # renaming and reordering islands
    for n in range(len(island_sequence)):
        name = island_sequence[n].replace('->', ' ')
        name = name.split()
        isl_name_new.append(f'{n}: ' + name[-1])
        indices.append(C_init.columns.get_loc(name[0]))

    # projecting
    indices = np.asarray(indices)
    C = np.asarray(C_init)
    C = C[indices, :][:, indices]

    # convet C to reduced form
    C = np.diag(sum_to_diag(np.triu(C))) - np.triu(C, k=1)

    # transform to dataframe
    C = DataFrame(C, columns=isl_name_new, index=isl_name_new)
    
    if(len(pins)==0): return C

    # renaming and reordering pins
    indices_pins = []
    pin_name_new = []
    for n in range(len(pins)):
        name = pins[n].replace('->', ' ')
        name = name.split()
        pin_name_new.append(f'{n}: ' + name[-1])
        indices_pins.append(C_init.columns.get_loc(name[0]))

    # projecting
    indices_pins = np.asarray(indices_pins)
    C_pins = np.asarray(C_init)
    C_pins = np.abs(C_pins[indices, :][:, indices_pins])

    # transform to dataframe
    C_pins = DataFrame(C_pins, columns=pin_name_new, index=isl_name_new)
    
    return C, C_pins
    

def legendre_forward(C, C_pins=None, S=None):
    """
       performe legendre transformation of the capacitance circuit,
       computing Ec matrix and effective pin charges 

       Parameters:

       C : 2-D np.array, fF
           circuit capacitance matrix in physical or reduced form
       C_pins : 2-D np.array, fF
           array of C vectors for pins
       S : variable transfer matrix in form S*v_0 -> v

       Returns:

       Ec : 2-D np.array, GHz
           circuit capacitance energy matrix in reduced form
           (upper triangle part - g between nodes, and
            diagonal part – Ec for nodes in 4*Ec*n^2 notation)
       pin_q : 2-D np.array, GHz/mV
           array of effective charge vectors for pins (only if C_pins!=None)      
    """        
    if(np.any(S==None)): S = np.eye(C.shape[0])

    # check input C matrix form and transfer to physical C_phys
    if(np.any(np.diag(C) < 0)):
        raise ValueError('Invalid C matrix!')
    elif(np.all(np.triu(C, k=1)!=np.transpose(np.tril(C, k=-1))) and np.all(np.triu(C, k=1) <= 0)):
        C_ph = np.copy(C)
    elif(np.all(np.tril(C, k=-1) == 0) and np.all(np.triu(C, k=1) >= 0)):
        C_ph = np.diag(sum_to_diag(C)) - np.triu(C, k=1) - np.transpose(np.triu(C, k=1))
        if(np.all(C_pins!=None)): C_ph = C_ph + np.diag(np.sum(C_pins, axis=1))
    else: 
        raise ValueError('Invalid C matrix!')

    # capacitance matrix transform
    C_inv_ph = np.linalg.inv(C_ph)
    C_inv = S@C_inv_ph@np.transpose(S)

    # transfer to fancy form and right scales
    Ec = 4*e**2*(np.diag(np.diag(C_inv))/8 + np.triu(C_inv, k=1))/ (2*np.pi*hbar) * 1e6
    if(np.any(C_pins==None)): return Ec
    
    # compute effective pin charges and transfer them to GHz/mV scale
    pin_q = 2*e*S@C_inv_ph@C_pins/ (2*np.pi*hbar) * 1e-12
    return Ec, pin_q

    
def legendre_backward(Ec, pin_q=None, S=None):
    """
       performe backward legendre transformation of the Ec matrix,
       computing capacitance circuit and pin capacitors

       Parameters:

       Ec : 2-D np.array, GHz
           circuit capacitance energy matrix in reduced form
           (upper triangle part - g between nodes, and
            diagonal part – Ec for nodes in 4*Ec*n^2 notation)
       pin_q : 2-D np.array, GHz/mV
           array of effective charge vectors for pins
       S : variable transfer matrix in form S*v_0 -> v

       Returns:

       C : 2-D np.array, fF
           circuit capacitance matrix in physical or reduced form
       C_pins : 2-D np.array, fF
           array of C vectors for pins (only if pin_q!=None)
    """  
    if(np.any(S==None)): S = np.eye(Ec.shape[0])

    # check input Ec matrix form and transfer to physical C_inv
    if(np.any(np.diag(Ec) < 0)):
        raise ValueError('Invalid Ec matrix!')
    elif(np.all(np.abs(np.triu(Ec, k=1)) - np.transpose(np.abs(np.tril(Ec, k=-1))) < 0)):
        raise ValueError('Invalid Ec matrix!')
    else:
        C_inv = 2*np.pi*hbar/(4*e**2)*(np.diag(np.diag(Ec))*8 + np.triu(Ec, k=1) +\
                                       np.transpose(np.triu(Ec, k=1))) * 1e-6

    # capacitance matrix transform
    C = np.linalg.inv(C_inv)
    C_ph = np.transpose(S)@C@S
    
    # convet C_ph to reduced form
    C_re = np.diag(sum_to_diag(np.triu(C_ph))) - np.triu(C_ph, k=1)
    if(np.any(pin_q==None)): return C_re
    
    # compute effective pin capacitances
    C_pins = 2*np.pi*hbar/(2*e) * np.transpose(S)@C@pin_q * 1e12
    return C_re, C_pins


def legendre_forward_opt(C_0, C_d, Ec_tg, Ec_weights=None,
                         C_pins_0=None, C_pins_d=None, pin_q_tg=None, pin_q_weights=None,
                         S=None, maxiter=500):
    """
       performe optimization of physical capacitance matrix and pin capacitors
       to get the target Ec matrix and pin_q (use dual annealing); deviation from
       the target is taken as Frobenius norm of (M_target - M)*weights
       
       Parameters:

       C_0 : 2-D np.array, fF
           initial capacitance matrix in reduced form
       C_d : 2-D np.array, fF
           capacitance matrix in reduced form defining maximal capacitance change
           (search border is [C_0 - C_d, C_0 + C_d], where lower one kept 0 if C_d > C_0)
       Ec_tg : 2-D np.array, GHz
           target Ec matrix in reduced form
       Ec_weights : 2-D np.array, (by default matrix of ones)
           weights for Ec matrix deviation
       C_pins_0 : 2-D np.array, fF
           array of initial C vectors for pins
       C_pins_d : 2-D np.array, fF
           array of maximal capacitance change for pins (similar to C_d)
       pin_q_tg : 2-D np.array, GHz/mV
           array of target charge vectors for pins
       pin_q_weigths : 2-D np.array, (by default matrix of ones)
           array of weights for charge vectors for pins
       S : 2-D np.array
           variable transfer matrix in form S*v_0 -> v
       maxiter : int,
           defines maxiter in scipy.optimize.dual_annealing
       
       Returns:
       Ec : 2-D np.array, GHz
           optimal capacitance energy matrix in reduced form
           (upper triangle part - g between nodes, and
            diagonal part – Ec for nodes in 4*Ec*n^2 notation)
       pin_q : 2-D np.array, GHz/mV
           array of optimal charge vectors for pins (only if C_pins_0!=None)
       C : 2-D np.array, fF
           optimal capacitance matrix in physical or reduced form
       C_pins : 2-D np.array, fF
           array of optimal C vectors for pins (only if C_pins_0!=None)
    """
    if(np.any(Ec_weights==None)): Ec_weights = np.ones(Ec_tg.shape)
    if(np.any(pin_q_weights==None) and np.all(pin_q_tg!=None)): pin_q_weights = np.ones(pin_q_tg.shape)
            
    # set bounds
    bounds = []

    C_0_l = np.copy(C_0.reshape(C_0.shape[0]*C_0.shape[1]))
    C_d_l = np.copy(C_d.reshape(C_d.shape[0]*C_d.shape[1]))
    C_d_ind = C_d_l.nonzero()[0]
    
    for n in C_d_ind: bounds.append((max(C_0_l[n] - C_d_l[n], 0), C_0_l[n] + C_d_l[n]))
    
    if(np.all(C_pins_0!=None)):
        C_pins_0_l = np.copy(C_pins_0.reshape(C_pins_0.shape[0]*C_pins_0.shape[1]))
        C_pins_d_l = np.copy(C_pins_d.reshape(C_pins_d.shape[0]*C_pins_d.shape[1]))
        C_pins_d_ind = C_pins_d_l.nonzero()[0]

        for n in C_pins_d_ind: bounds.append((max(C_pins_0_l[n] - C_pins_d_l[n], 0), C_pins_0_l[n] + C_pins_d_l[n]))

    # define loss function
    def loss(x):
        
        C_0_l[C_d_ind] = x[range(C_d_ind.shape[0])]
        C = C_0_l.reshape(C_0.shape)

        # check for singularity
        if(np.linalg.det(C)==0): return 1e10

        if(np.any(C_pins_0==None)):
            Ec = legendre_forward(C, S=S)
            return np.linalg.norm(Ec_weights*(Ec - Ec_tg))
        else:
            C_pins_0_l[C_pins_d_ind] = x[range(C_d_ind.shape[0], C_d_ind.shape[0] + C_pins_d_ind.shape[0])]
            C_pins = C_pins_0_l.reshape(C_pins_0.shape)
            Ec, pin_q = legendre_forward(C, C_pins=C_pins, S=S)
            return np.linalg.norm(Ec_weights*(Ec - Ec_tg)) + np.linalg.norm(pin_q_weights*(pin_q - pin_q_tg))

    # optimization of C and C_pins
    sol = dual_annealing(loss, bounds=bounds, maxiter=maxiter)

    C_0_l[C_d_ind] = sol.x[range(C_d_ind.shape[0])]
    C = C_0_l.reshape(C_0.shape)

    if(np.any(C_pins_0==None)):
        Ec = legendre_forward(C, S=S)
        return Ec, C
    if(np.all(C_pins_0!=None)):
        C_pins_0_l[C_pins_d_ind] = sol.x[range(C_d_ind.shape[0], C_d_ind.shape[0] + C_pins_d_ind.shape[0])]
        C_pins = C_pins_0_l.reshape(C_pins_0.shape)
        Ec, pin_q = legendre_forward(C, C_pins=C_pins, S=S)
        return Ec, pin_q, C, C_pins


def cross_talk_solver(M):
    """
       compute effective pin force coefficients and vectors to drive 
       degrees of freedom independently in presence of crosstalk.
       If H = M@F, where F - force vecotor in pin bvasis, H - Hamiltonian 
       perturbation, and M - crosstalk matrix, then effective force coef on 
       the pin n is:

       pin_ef_n = 1/(M^-1)_nn

       and corresponding force vector:

       v_force_n = (M^-1)_nm * pin_ef_m

       Parameters:

       M : 2-D np.array, GHz/force_unit
           crosstalk matrix

       Returns:

       pin_ef : 1-D np.array, GHz/force_unit
           effective force coef per pin
       v_force : 2-D np.array, float
           forve vectors for independent drive
       
    """
    
    M_inv = np.linalg.inv(M)
    pin_ef = 1/np.diag(M_inv)
    v_force = np.einsum('ij, j -> ij', M_inv, pin_ef)

    return pin_ef, v_force

    