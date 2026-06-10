import numpy as np
from .math import kron, dagger
from scipy.sparse.linalg import eigsh


def dressed_states_map(states, dim, get_strings=False, minimal_st_impact=0.002):
    """
       map dressed states on the bare state basis, gives "key" enabling to find 
       the dressed state corresponding to the given bare state, also computes 
       dresseed state "selfhood" which represents the degree of a bare state
       conservation and is defined as |<bare_st|dressed_st>|^2
       
       Parameters: 
           states : 2-D np.array
               array of dressed state vecotrs in the bare state basis
           dim : 1-D np.array or list
               sizes of the subspaces dimensions, the function process only 2-D and 3-D 
               cases like AxB and AxBxC -> [dim(A), dim(B)] or [dim(A), dim(B), dim(C)]
           get_strings : bool
               turn the regime in which the function computs the list of strings 
               with bare state decomposition of dressed states
           minimal_st_impact : float
               in get_strings=True regime, defines the minimnal hybridization 
               of bare states listed for each dressed state
               
       Returns:
           key : N-D np.array with N=len(dim)
               map: bare -> dressed state, key[n, m] or key[n, m, k]->N in the dressed basis
               in case of complete state degeneration returns None
           selfhood : 1-D array
               contains selfhood values for all dressed states
           st_strings : list of strings
               gives explicit strings with the dressed state decomposition
    """
    tmp = np.asarray(dim)
    st_strings = []
    # 2-D case
    if(tmp.shape[0] == 2):
        N1 = tmp[0]
        N2 = tmp[1]
        
        key = np.zeros((N1, N2), dtype=object)
        selfhood = np.zeros(N1*N2)
        
        for n in range(states.shape[1]):
            s = abs(states[:, n])
            s = s.reshape(N1, N2)
            
            old_num = key[np.unravel_index(s.argmax(), s.shape)]
            selfhood[n] = abs(s[np.unravel_index(s.argmax(), s.shape)])**2 
            
            if(old_num != 0):
                if(selfhood[old_num] <= selfhood[n]):
                    key[np.unravel_index(s.argmax(), s.shape)] = int(n)
                    
            else:
                key[np.unravel_index(s.argmax(), s.shape)] = int(n)
                    
            if(get_strings):
                string = str(n) + ': '
                
                while(True):
                    local_impact = abs(s[np.unravel_index(s.argmax(), s.shape)])**2
                    if(local_impact > minimal_st_impact):
                        un_ind = np.unravel_index(int(s.argmax()), s.shape)
                        string += str(local_impact * 100)+'% of '+ f'({int(un_ind[0])}, {int(un_ind[1])})\n'
                    else:
                        break
                    
                    s[np.unravel_index(s.argmax(), s.shape)] = 0
                    
                st_strings.append(string)
            
        # lable suppressed states with None
        for n in range(key.shape[0]):
            for m in range(key.shape[1]):
                    if(n + m != 0 and key[n, m] == 0):
                        key[n, m] = None
        
        if(get_strings):
            return (key, selfhood, st_strings)
        else:
            return (key, selfhood)
    # 3-D case            
    elif(tmp.shape[0] == 3):
        N1 = tmp[0]
        N2 = tmp[1]
        N3 = tmp[2]
        
        key = np.zeros((N1, N2, N3), dtype=object)
        selfhood = np.zeros(N1*N2*N3)
        
        for n in range(states.shape[1]):
            s = abs(states[:, n])
            s = s.reshape(N1, N2, N3)
            
            old_num = key[np.unravel_index(s.argmax(), s.shape)]
            selfhood[n] = abs(s[np.unravel_index(s.argmax(), s.shape)])**2 
            
            if(old_num != 0):
                if(selfhood[old_num] <= selfhood[n]):
                    key[np.unravel_index(s.argmax(), s.shape)] = int(n)
                    
            else:
                key[np.unravel_index(s.argmax(), s.shape)] = int(n)
                    
            if(get_strings):
                string = str(n) + ': '
                
                while(True):
                    local_impact = abs(s[np.unravel_index(s.argmax(), s.shape)])**2
                    if(local_impact > minimal_st_impact):
                        un_ind = np.unravel_index(int(s.argmax()), s.shape)
                        string += str(local_impact * 100)+'% of '+ f'({int(un_ind[0])}, {int(un_ind[1])}, {int(un_ind[2])})\n'
                    else:
                        break
                    
                    s[np.unravel_index(s.argmax(), s.shape)] = 0
                    
                st_strings.append(string)
            
        # lable suppressed states with None
        for n in range(key.shape[0]):
            for m in range(key.shape[1]):
                for k in range(key.shape[2]):
                    if(n + m + k != 0 and key[n, m, k] == 0):
                        key[n, m, k] = None
                        
        if(get_strings):
            return (key, selfhood, st_strings)
        else:
            return (key, selfhood)
        
    else: raise ValueError('Too many dimentions!')
        

def mix_two_sys(spectrum_1, spectrum_2, V_12, V_21, truncation=4, 
                operators_1=[], operators_2=[], eig_states_output=False,
                dressed_st_map=True, get_strings=False, minimal_st_impact=0.002):
    """
       diagonalizes two-part system defined by Hamiltonian:
       H = H_1 + H_2 + V_12*V_21
       phase for each eigen wf psi is chosen to get psi[argmax(psi)] - real positive
       this condition is based on <psi_0|psi> - real positive, which is common for 
       the perturbatio theory
       
       Parameters: 
           spectrum_1 : 1-D np.array, spectrum of the 1-subsystem
           spectrum_2 : 1-D np.array, spectrum of the 2-subsystem
           V_12 : 2-D np.array, 1-subsystem interaction operator
           V_21 : 2-D np.array, 2-subsystem interaction operator
           truncation : int
               number of levels in the final spectrum
           operators_1 : list of 2-D np.arrays
               list of the 1-subsystem operators in original Hilbert space 
               which are transfered to the dressed state basis
           operators_2 : list of 2-D np.arrays
               list of the 2-subsystem operators in original Hilbert space 
               which are transfered to the dressed state basis
           eig_states_output : bool,
               if True, adds dressed states in bare basis to returns
           dressed_st_map : 2-D np.array
               if True, compute dressed_state_map (see dressed_state_map fun)
           get_strings : bool, (dressed_states_map fun parameter)
           minimal_st_impact : float, (dressed_states_map fun parameter)
               
       Returns:
           spectrum : 1-D np.array with len=truncation
           eig_states : 2-D np.array (exists only if eig_states_output=True)
               dressed states in bare basis
           dressed_st_info : (key, selfhood, st_strings(if get_strings=True)) 
               (exists only if dressed_st_map=True)
               output of dressed_state_map fun
           new_operators_1 : list of 2-D np.arrays if len(operators_1)>1 
               and 2-D np.array if len(operators_1)=1 (exists only if len(operators_1)>0),
               the first subsystem operators in the dressed state basis
           new_operators_2 : list of 2-D np.arrays if len(operators_2)>1 
               and 2-D np.array if len(operators_2)=1 (exists only if len(operators_1)>0),
               the second subsystem operators in the dressed state basis
    """
    dim_1 = spectrum_1.shape[0]
    dim_2 = spectrum_2.shape[0]
    
    E_1 = np.eye(dim_1)
    E_2 = np.eye(dim_2)
    
    H_1 = np.diag(spectrum_1)
    H_2 = np.diag(spectrum_2)
    
    # complete Hamiltonian
    H = kron(H_1, E_2) + kron(E_1, H_2) + kron(V_12, V_21)

    # diagonalization
    (e, v) = eigsh(H, k=truncation, which='SA', maxiter=5000)
    sorted_indices = np.argsort(e)
    spectrum = e[sorted_indices]
    eig_states = v[:, sorted_indices]

    # phase gauge
    for k in range(eig_states.shape[1]):
        psi_0 = np.abs(eig_states[:, k]).argmax()
        rot = np.conjugate(eig_states[psi_0, k]/abs(eig_states[psi_0, k]))
        eig_states[:, k]=eig_states[:, k]*rot

    # spectrum alignment
    spectrum = spectrum - spectrum[0]
    output = [spectrum]

    # add eig_states to output
    if(eig_states_output): output.append(eig_states)

    # get dressed state map
    if(dressed_st_map):
        dressed_st_info = dressed_states_map(eig_states, (dim_1, dim_2), get_strings=get_strings, 
                                             minimal_st_impact=minimal_st_impact)
        output.append(dressed_st_info)
        
    # transfer operators to the new dressed eigen basis
    new_operators_1 = []    
    for oper in operators_1:
        M = kron(oper, E_2)
        new_operators_1.append(dagger(eig_states) @ M @ eig_states)                    
    if(len(operators_1) == 1):
        output.append(new_operators_1[0])
    elif(len(operators_1) > 1):
        output.append(new_operators_1)
    
    new_operators_2 = []    
    for oper in operators_2:
        M = kron(E_1, oper)
        new_operators_2.append(dagger(eig_states) @ M @ eig_states)                    
    if(len(operators_2) == 1):
        output.append(new_operators_2[0])
    elif(len(operators_2) > 1):
        output.append(new_operators_2)

    return output
    

def mix_three_sys(spectrum_1, spectrum_2, spectrum_3, V_12=None, V_13=None, V_21=None, V_23=None, V_31=None, V_32=None, 
                  truncation=8, operators_1=[], operators_2=[], operators_3=[], eig_states_output=False,
                  dressed_st_map=True, get_strings=False, minimal_st_impact=0.002):
    """
       diagonalizes three-part system defined by Hamiltonian:
       H = H_1 + H_2 + H_3 + V_12*V_21 + V_23*V_32 + V_13*V_31
       phase for each eigen wf psi is chosen to get psi[argmax(psi)] - real positive
       this condition is based on <psi_0|psi> - real positive, which is common for 
       the perturbatio theory
       
       Parameters: 
           spectrum_1 : 1-D np.array, spectrum of the 1-ubsystem
           spectrum_2 : 1-D np.array, spectrum of the 2-subsystem
           spectrum_2 : 1-D np.array, spectrum of the 3-subsystem
           V_12, V_13 : 2-D np.array, 1-subsystem interaction operator
           V_21, V_23 : 2-D np.array, 2-subsystem interaction operator
           V_31, V_32 : 2-D np.array, 3-subsystem interaction operator
           truncation : int
               number of levels in the final spectrum
           operators_1 : list of 2-D np.arrays
               list of the 1-subsystem operators in original Hilbert space 
               which are transfered to the dressed state basis
           operators_2 : list of 2-D np.arrays
               list of the 2-subsystem operators in original Hilbert space 
               which are transfered to the dressed state basis
           operators_3 : list of 2-D np.arrays
               list of the 3-subsystem operators in original Hilbert space 
               which are transfered to the dressed state basis
           eig_states_output : bool
               if True, adds dressed states in bare basis to returns
           dressed_st_map : 2-D np.array
               if True, compute dressed_state_map (see dressed_state_map fun)
           get_strings : bool (dressed_states_map fun parameter)
           minimal_st_impact : float (dressed_states_map fun parameter)
               
       Returns:
           spectrum : 1-D np.array with len=truncation
           eig_states : 2-D np.array (exists only if eig_states_output=True)
               dressed states in bare basis
           dressed_st_info : (key, selfhood, st_strings(if get_strings=True)) 
               (exists only if dressed_st_map=True)
               output of dressed_state_map fun
           new_operators_1 : list of 2-D np.arrays if len(operators_1)>1 
               and 2-D np.array if len(operators_1)=1 (exists only if len(operators_1)>0),
               the first subsystem operators in the dressed state basis
           new_operators_2 : list of 2-D np.arrays if len(operators_2)>1 
               and 2-D np.array if len(operators_2)=1 (exists only if len(operators_2)>0),
               the second subsystem operators in the dressed state basis
           new_operators_3 : list of 2-D np.arrays if len(operators_3)>1 
               and 2-D np.array if len(operators_3)=1 (exists only if len(operators_3)>0),
               the second subsystem operators in the dressed state basis
    """
    dim_1 = spectrum_1.shape[0]
    dim_2 = spectrum_2.shape[0]
    dim_3 = spectrum_3.shape[0]
    
    E_1 = np.eye(dim_1)
    E_2 = np.eye(dim_2)
    E_3 = np.eye(dim_3)
    
    H_1 = np.diag(spectrum_1)
    H_2 = np.diag(spectrum_2)
    H_3 = np.diag(spectrum_3)
    
    # unperturbed Hamiltonian
    H = kron(H_1, E_2, E_3) + kron(E_1, H_2, E_3) + kron(E_1, E_2, H_3)

    # perturbation
    if(np.any(V_12!=None)): H = H + kron(V_12, V_21, E_3)
    if(np.any(V_23!=None)): H = H + kron(E_1, V_23, V_32)
    if(np.any(V_12!=None)): H = H + kron(V_13, E_2, V_31)
    
    # diagonalization
    (e, v) = eigsh(H, k=truncation, which='SA', maxiter=5000)
    sorted_indices = np.argsort(e)
    spectrum = e[sorted_indices]
    eig_states = v[:, sorted_indices]

    # phase gauge
    for k in range(eig_states.shape[1]):
        psi_0 = np.abs(eig_states[:, k]).argmax()
        rot = np.conjugate(eig_states[psi_0, k]/abs(eig_states[psi_0, k]))
        eig_states[:, k]=eig_states[:, k]*rot
        
    # spectrum alignment
    spectrum = spectrum - spectrum[0]
    output = [spectrum]
    
    if(eig_states_output): output.append(eig_states)
    
    if(dressed_st_map):
        dressed_st_info = dressed_states_map(eig_states, (dim_1, dim_2, dim_3), get_strings=get_strings, 
                                             minimal_st_impact=minimal_st_impact)
        output.append(dressed_st_info)

    # transfer operators to the new dressed eigen basis
    new_operators_1 = []    
    for oper in operators_1:
        M = kron(oper, E_2, E_3)
        new_operators_1.append(dagger(eig_states) @ M @ eig_states)                    
    if(len(operators_1) == 1):
        output.append(new_operators_1[0])
    elif(len(operators_1) > 1):
        output.append(new_operators_1)
    
    new_operators_2 = []    
    for oper in operators_2:
        M = kron(E_1, oper, E_3)
        new_operators_2.append(dagger(eig_states) @ M @ eig_states)                    
    if(len(operators_2) == 1):
        output.append(new_operators_2[0])
    elif(len(operators_2) > 1):
        output.append(new_operators_2)

    new_operators_3 = []    
    for oper in operators_3:
        M = kron(E_1, E_2, oper)
        new_operators_3.append(dagger(eig_states) @ M @ eig_states)                    
    if(len(operators_3) == 1):
        output.append(new_operators_3[0])
    elif(len(operators_3) > 1):
        output.append(new_operators_3)
            
    return output


# class for combining a set of many key-arrays
class keys:

    def __init__(self, *args):
        # args – pairs (key, next_node) or just (key, None), where next_node is a target of the next embedding
        
        self.keys_set = []
        self.next_nodes = []
        self.size = 0

        for n in range(len(args)):
            
            self.keys_set.append(args[n][0])
            self.size += args[n][0].ndim
            
            if(args[n][1] != None):
                
                self.next_nodes.append(args[n][1])
                self.size -= 1
            
        
    def get(self, str):
        # str – is a string with a bare state

        # just in case
        str = str.replace(' ', '')
        str = str.replace('|', '')
        str = str.replace('>', '')
        if(len(str) != self.size): raise TypeError('invalid bare state format')

        # first step
        car = 0
        for node in self.next_nodes: car += node
        loc_keys = self.keys_set[-1]
        leap = loc_keys.ndim - 1
        
        loc_state = []
        for n in range(loc_keys.ndim): loc_state.append(int(str[car + n]))
        index = loc_keys[tuple(loc_state)]
        
        for n in range(len(self.keys_set) - 1):
            
            car -= self.next_nodes[-n-1]
            loc_keys = self.keys_set[-n-2]

            loc_state = []
            for k in range(loc_keys.ndim): 
                if(k < self.next_nodes[-n-1]):
                    loc_state.append(int(str[car + k]))
                elif(k == self.next_nodes[-n-1]):
                    loc_state.append(index)
                else:
                    loc_state.append(int(str[car + k + leap]))
                    
            leap += self.keys_set[-n-2].ndim - 1

            index = loc_keys[tuple(loc_state)]
            
        return index

        
def trans_isolation(init_st_list, target_st_list, pert_oper, spectrum, border, other_st_list=[], mod='k^2/d', 
                    rounding=3, multiphoton_trigger=1e-6):

    # mod 0: search based on k**2/delta, where k = m_tr/m_aim (inspired by three-lvl Rabi), here border=(k**2/delta)_min
    # mod 1: search based on k**2/delta**2, where k = m_tr/m_aim (inspired by three-lvl Rabi), here border=(k**2/delta**2)_min
    # mod 2: search with border[0] – minimal value of k and border[1] – maximum transition's frequencies delta
    # mod 3: TWO-PHOTON LEAKAGE with border[0] – minimum of k=sum_v(|k_iv*k_vf/(f-2f_virt)|) and border[1] – maximum |f_signal-f/2|
    
    if(mod=='k^2/d' or mod==0):
        mod = 0
    elif(mod=='k^2/d^2' or mod==1):
        mod = 1
    elif(mod=='k_min, d_max' or mod==2):
        mod = 2
    elif(mod=='two-photon' or mod==3):
        mod = 3
        
    # output: leakage_st[0] – init leakage states, leakage_st[1] – target leakage states; leakage_param[0] – k, leakage_param[1] – delta

    full_st_list = init_st_list + target_st_list + other_st_list
    full_st_list = np.asarray(full_st_list, dtype=int)

    m_0 = abs(pert_oper[init_st_list[0], target_st_list[0]])
    f_0 = abs(spectrum[init_st_list[0]] - spectrum[target_st_list[0]])

    # arrays for output
    leakage_trans = []
    leakage_k = []
    leakage_delta = []

    
    # transitions init -> fin
    for init in full_st_list:
        for fin in range(spectrum.shape[0]):

            # first filter
            flag = False
            for n in range(len(init_st_list)):
                if(init==init_st_list[n] and fin==target_st_list[n]): flag = True

            for n in range(len(target_st_list)):
                if(fin==init_st_list[n] and init==target_st_list[n]): flag = True
                
            if(flag): continue


            # trans params writing down
            m = abs(pert_oper[init, fin])
            k = m/m_0
            delta = abs(abs(spectrum[init] - spectrum[fin]) - f_0)

            # analysis
            if(mod==0 and k**2/delta > border):

                flag = True
                for trans in leakage_trans: 
                    if(trans[0] == init and trans[1] == fin): flag = False
                    if(trans[0] == fin and trans[1] == init): flag = False
                if(flag):
                    leakage_trans.append([init, fin])
                    leakage_k.append(k)
                    leakage_delta.append(delta)
                
            elif(mod==1 and k**2/delta**2 > border):
                
                flag = True
                for trans in leakage_trans: 
                    if(trans[0] == init and trans[1] == fin): flag = False
                    if(trans[0] == fin and trans[1] == init): flag = False
                if(flag):
                    leakage_trans.append([init, fin])
                    leakage_k.append(k)
                    leakage_delta.append(delta)

            elif(mod==2 and k > border[0] and delta < border[1]):
                
                flag = True
                for trans in leakage_trans: 
                    if(trans[0] == init and trans[1] == fin): flag = False
                    if(trans[0] == fin and trans[1] == init): flag = False
                if(flag):
                    leakage_trans.append([init, fin])
                    leakage_k.append(k)
                    leakage_delta.append(delta)
               
            elif(mod==3):
                
                k_multi = 0
                
                for virt in range(spectrum.shape[0]):
                    
                    flag = False
                    for st in full_st_list:
                        if(virt == st):
                            flag = True

                    if(flag): continue
                        
                    k_multi += abs(pert_oper[init, virt]*pert_oper[virt, fin]/m_0**2\
                    /(spectrum[fin] - 2*spectrum[virt] + spectrum[init]))
                    
                delta = abs(abs(spectrum[init] - spectrum[fin])/2 - f_0)
                
                flag = True
                for trans in leakage_trans: 
                    if(trans[0] == init and trans[1] == fin): flag = False
                    if(trans[0] == fin and trans[1] == init): flag = False
                if(flag):
                    if(k_multi > border[0] and delta < border[1]):
                        leakage_trans.append([init, fin])
                        leakage_k.append(k_multi)
                        leakage_delta.append(delta)

    if(mod==0):
        tmp = np.asarray(leakage_k)**2/np.asarray(leakage_delta)
        sort = np.argsort(np.asarray(tmp))
        sort = np.flip(sort)
        
    if(mod==1):
        tmp = np.asarray(leakage_k)**2/np.asarray(leakage_delta)**2
        sort = np.argsort(np.asarray(tmp))
        sort = np.flip(sort)

    if(mod==2 or mod==3):
        sort = np.argsort(np.asarray(leakage_delta))
        

    leakage_st = np.zeros((sort.shape[0], 2), int)
    leakage_param = np.zeros((sort.shape[0], 2))
    string_list = []
    
    for i in range(sort.shape[0]):
        
        leakage_st[i, 0] = leakage_trans[sort[i]][0]
        leakage_st[i, 1] = leakage_trans[sort[i]][1]
        
        leakage_param[i, 0] = leakage_k[sort[i]]
        leakage_param[i, 1] = leakage_delta[sort[i]]
        
        tmp_1 = leakage_param[i, 0]**2/leakage_param[i, 1]
        tmp_2 = leakage_param[i, 0]**2/leakage_param[i, 1]**2

        if(mod!=3):
            string_list.append("{0} -> {1} : k={2}, ∆={3}, k**2/∆={4}, k**2/∆**2={5}".format(leakage_st[i, 0], 
                                                                                        leakage_st[i, 1], 
                                                                                       around(leakage_param[i, 0], rounding), 
                                                                                       around(leakage_param[i, 1], rounding), 
                                                                                       around(tmp_1, rounding), 
                                                                                       around(tmp_2, rounding)))
        else:
            string_list.append("{0} -> {1} : ∑|k_iv*k_vf/(fr_f-2fr_v)|={2}, ∆={3}".format(leakage_st[i, 0], leakage_st[i, 1], 
                                                                  around(leakage_param[i, 0], rounding), 
                                                                  around(leakage_param[i, 1], rounding)))
    
    return leakage_st, leakage_param, string_list

    
def side_transitions(init_st_list, target_st_list, 
                     pert_oper, spectrum, constraint, 
                     other_st_list=[], mode='one-photon'):

    # constraint – f: (k, delta, init) -> {0, 1}, give a constraint for thr leakages selection
    # mode – setup a n-photon leakages search (one-photon, two-photon)


    full_st_list = init_st_list + target_st_list + other_st_list
    full_st_list = np.asarray(full_st_list, dtype=int)

    # establishig of the main target transition parameters
    m_0 = abs(pert_oper[init_st_list[0], target_st_list[0]])
    f_0 = abs(spectrum[init_st_list[0]] - spectrum[target_st_list[0]])

    # arrays of the leakages parameters
    leakage_trans = []
    leakage_k = []
    leakage_delta = []


    # transitions init -> fin
    for init in full_st_list:
        for fin in range(spectrum.shape[0]):

            # excluding of the target transitions
            flag = False
            for n in range(len(init_st_list)): 
                if((init==init_st_list[n] or init==target_st_list[n]) and (fin==init_st_list[n] or fin==target_st_list[n])): flag = True
                    
            if(flag): continue

            
            # writing down of the leakage parameters
            if(mode=='one-photon'):
                
                k = abs(pert_oper[init, fin])/m_0
                delta = abs(abs(spectrum[init] - spectrum[fin]) - f_0)
                
            elif(mode=='two-photon'):
                # calculation of the k analogue for the two-photon leakage
                k = 0
                for virt in range(spectrum.shape[0]):
                    if(virt != init and virt != fin):           
                        k += abs(pert_oper[init, virt]*pert_oper[virt, fin]/m_0**2\
                        /(spectrum[fin] - 2*spectrum[virt] + spectrum[init]))
                    
                delta = abs(abs(spectrum[init] - spectrum[fin])/2 - f_0)
                
  
            # application of the constraint
            if(constraint(k, delta, init)):

                # excluding of the repeating leakages
                flag = True
                for trans in leakage_trans: 
                    if(trans[0] == init and trans[1] == fin): flag = False
                    if(trans[0] == fin and trans[1] == init): flag = False

                # wirting down of the leakage
                if(flag):
                    leakage_trans.append([init, fin])
                    leakage_k.append(k)
                    leakage_delta.append(delta)


    # sorting of the leakages accroding to their delta values
    sort = np.argsort(np.asarray(leakage_delta))
    leakage_delta = np.asarray(leakage_delta)[sort]
    leakage_k = np.asarray(leakage_k)[sort]
    if(len(leakage_trans) > 0):
        leakage_trans = np.asarray(leakage_trans)[sort, :]
    else: leakage_trans = np.asarray(leakage_trans)
    
    return leakage_trans, leakage_k, leakage_delta

