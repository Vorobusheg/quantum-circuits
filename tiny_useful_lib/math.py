import numpy as np
import numdifftools as nd
from scipy.optimize import dual_annealing

def dagger(a): return np.conjugate(a.transpose())


def subspace(M, indices): return M[np.asarray(indices), :][:, np.asarray(indices)]


def kron(*opers):
    """
       compute kronecker product of arbitrary length operator sequence
       
       Parameters: 
           opers : *args
       Returns: 
           prod : np.array
    """
    prod = np.kron(opers[0], opers[1])
    for n in range(len(opers) - 2): prod =np.kron(prod, opers[n + 2])

    return prod


def expand_space(H, expansion):
    """
       expand space of matrix with new basis states
       
       Parameters: 
           H : 2-dimentional np.array
           expansion : int
               number of new states
       Returns:
           prod : 2-dimentional np.array
    """
    H_out = np.append(H, np.zeros((expansion, H.shape[0])), axis=0)
    H_out = np.append(H_out, np.zeros((H_out.shape[0], expansion)), axis=1)
    
    return H_out
    

def index_linear(coord, dim):
    """
       transform coordinate of N-dimentional tensor A to the corresponding integer 
       index of the 1-dimentional tensor B = np.reshape(A, A.size)
       
       Parameters: 
           coord : 1-dimentional np.array or list, coord.size=N
           dim : 1-dimentional np.array or list, dim.size=N
               contain dimention sizes (A.shape)
       Returns:
           index : int
    """
    coord_loc = np.asarray(coord)
    dim_loc = np.asarray(dim)

    if(np.any(coord_loc >= dim_loc)): raise ValueError('DIMENTIONS INCOMPATIBILITY!')
    
    index = 0
    
    for n in range(coord_loc.shape[0]-1): index = (index + coord_loc[n])*dim_loc[n + 1]
        
    index += coord_loc[-1]
    
    return int(index)


def index_coord(index, dim):
    """
       transform integer index of the 1-dimentional tensor B = np.reshape(A, A.size)
       to the corresponding coordinate of N-dimentional tensor A
       
       Parameters: 
           index : int
           dim : 1-dimentional np.array or list, dim_in.size=N
               contain dimention sizes (A.shape)
       Returns:
           coord : 1-dimentional np.array, coord.size=N
    """
    
    dim_loc = np.copy(np.asarray(dim))
    coord = np.zeros(dim_loc.shape[0], int)

    dim_loc[0] = 1
    
    for it in range(dim_loc.shape[0]):

        d_prod = np.prod(dim_loc)
        coord[it] = int(index//d_prod)
        index -= coord[it]*d_prod
        
        if(it < dim_loc.shape[0] - 1):
            dim_loc[it + 1] = 1
    
    return coord


def around(x, n):
    """
       round x up to the n values after the first non-zero number
       
       Parameters: 
           x : float
           n : int
       Returns:
           x : float
    """
    deg = np.log(np.abs(x))/np.log(10)
    
    if(deg > 0): deg = int(deg)
    else: deg = int(np.ceil(deg))
        
    deg -= n
    if(abs(x)>=1): deg += 1
    
    return np.around(x, -deg)


# make all non-zero matrix elements equal to 1
def to_ones(x):
    if(x!=0): return 1
    else: return 0
to_ones = np.vectorize(to_ones)


def parabolic_fit(X_in, Y_in, bounds, maxiter=300, no_local_search=False, x0=None):
    """
       search of parabolic fit in form of:
       y = a*x^2 + b*x + c
       (optimizer stronger than in scipy.optimize.curve_fit)

       Parameters: 
           X_in : array of dot x-coordinates
           Y_in : array of dot y-coordinates
           bounds : [(a_min, a_max), (b_min, b_max), (c_min, c_max)]
               defines area of parabolic parameters search
           no_local_search : bool, default: False
               parameter of scipy.optimize.dual_annealing wich control additional local search
           x0 : np.array, [a_0, b_0, c_0]
       Returns:
           ans : 1-D np.array
               optimal (a, b, c) parametric dot
           cov : 2-D np.array
               covariation matrix at the optimal point, needed to get fit errors
    """
    X = np.asarray(X_in)
    Y = np.asarray(Y_in)
    # model A*(x - x_0)**2 + B**2
    def loss(x):
        
        a = x[0]
        b = x[1]
        c = x[2]
    
        return np.sum((Y - a*X**2 - b*X - c)**2)
    
    # оптимизируем
    sol = dual_annealing(loss, bounds=bounds, 
                         maxiter=maxiter,
                         no_local_search=no_local_search,
                         x0=x0)
    ans = sol.x
    
    hess = nd.Hessian(loss, step=1e-4, method='central', order=2)(ans)
    # covariance computation similar to scipy.optimize.curve_fit
    cov = np.linalg.inv(hess)*sol.fun/(X.shape[0] - ans.shape[0]) * 2

    return ans, cov
    

def linear_regression_fit(X_in, Y_in):
    """
       analitical search of arbitrary-dimentional linear fit in form of:
       y = beta@[1, x_1, x_2, ...]
       (source - wikipedia)

       Parameters: 
           X_in : 2-D array
               dots and their coordinates in the arbitrary-dimentional space 
               X_in.shape[0] -> dots, X_in.shape[1] -> dimensions (constant is included)
           Y_in : 1-D array of function values
           
       Returns:
           beta : 1-D np.array
               optimal regression vector
           mse : float
               mean square error
    """
    Y = np.copy(np.asarray(Y_in))
    X_0 = np.copy(np.asarray(X_in))
    
    if(len(X_0.shape) == 1):
        
        X = np.ones((X_0.shape[0], 2))
        X[:,1] = X_0
        
    else:
    
        X = np.ones((X_0.shape[0], X_0.shape[1] + 1))
        X[:,1:] = X_0
    
    tmp = np.einsum('mi,mk->ik', X, X)
    tmp = np.linalg.inv(tmp)
    tmp = np.einsum('ik,nk->in', tmp, X)
    
    beta = np.einsum('in,n->i', tmp, Y)
        
    mse = np.sqrt(np.sum((Y - np.einsum('k,ik->i', beta, X))**2))/Y.shape[0]

    return beta, mse

