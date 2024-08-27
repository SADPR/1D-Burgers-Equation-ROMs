import numpy as np

def get_sym(qi):
    ''' 
    Auxiliary function to get the symmetric part of q kron q (Kronecker product).
    This function generates a vector containing the unique elements of the symmetric
    Kronecker product of a given vector qi with itself.
    
    Parameters:
    - qi: np.array, the input vector.
    
    Returns:
    - vec: np.array, the symmetric part of the Kronecker product of qi with itself.
    '''
    size = qi.shape[0]
    vec = []

    for i in range(size):
        for j in range(size):
            if j >= i:
                vec.append(qi[i]*qi[j])

    return np.array(vec)

def getQ(modes, Ns, q):
    ''' 
    Constructs the Q matrix, which contains all the symmetric quadratic terms
    for each snapshot in the reduced coordinate space.
    
    Parameters:
    - modes: int, number of modes (size of q_p).
    - Ns: int, number of snapshots.
    - q: np.array, the matrix of reduced coordinates where each column is a reduced snapshot.
    
    Returns:
    - Q: np.array, the matrix where each column contains the symmetric quadratic terms
         for a corresponding snapshot.
    '''
    k = int(modes*(modes+1)/2)
    Q = np.empty((k, Ns))

    for i in range(Ns):
        Q[:,i] = get_sym(q[:,i])

    return Q

def get_single_Q(modes, q):
    ''' 
    Constructs the Q vector for a single snapshot, containing all the symmetric quadratic terms
    for the reduced coordinate space.
    
    Parameters:
    - modes: int, number of modes (size of q_p).
    - q: np.array, the vector of reduced coordinates for a single snapshot.
    
    Returns:
    - Q: np.array, the vector containing the symmetric quadratic terms for the input snapshot.
    '''
    k = int(modes*(modes+1)/2)
    Q = np.empty(k)

    Q = get_sym(q)

    return Q

def getE(N, Ns, S, Phi_p, q):
    ''' 
    Constructs the error matrix E, which represents the difference between the original
    snapshots and their linear POD approximations.
    
    Parameters:
    - N: int, number of spatial points.
    - Ns: int, number of snapshots.
    - S: np.array, the snapshot matrix where each column is a snapshot.
    - Phi_p: np.array, the linear basis matrix (Phi_p).
    - q: np.array, the matrix of reduced coordinates where each column is a reduced snapshot.
    
    Returns:
    - E: np.array, the error matrix where each column is the difference between an original
         snapshot and its linear POD approximation.
    '''
    E = np.empty((N,Ns))
    for i in range(Ns):
        E[:,i] = S[:,i] - Phi_p @ q[:,i]  # u_i - Vq_i - u_ref

    return E

def getH(Q, E, modes, N, alpha):
    ''' 
    Constructs the matrix H, which captures the influence of the secondary modes
    on the quadratic manifold approximation.
    
    Parameters:
    - Q: np.array, the matrix containing all the symmetric quadratic terms for each snapshot.
    - E: np.array, the error matrix (S - Phi_p * q).
    - modes: int, number of modes (size of q_p).
    - N: int, number of spatial points.
    - alpha: float, regularization parameter to stabilize the SVD inversion.
    
    Returns:
    - H: np.array, the matrix H capturing the effect of secondary modes on the quadratic approximation.
    '''
    Uq, Sigmaq, Yq_T = np.linalg.svd(Q, full_matrices=False)
    Sigmaq_Squared = np.square(Sigmaq)

    k = int(modes*(modes+1)/2)
    H = np.empty((N,k))

    for i in range(N):
        # print(f"{i} out of {N}")
        h_i = 0
        for l in range(Sigmaq.shape[0]):
            h_i += (Sigmaq_Squared[l] / (Sigmaq_Squared[l] + alpha ** 2)) * (np.dot(Yq_T[l], E[i, :].T) / Sigmaq[l]) * Uq[:, l]
        H[i, :] = h_i

    return H

def relative_error(S_exact, S_reconstructed):
    ''' 
    Calculate the relative squared Frobenius-norm error between the exact and reconstructed snapshots.
    
    Parameters:
    - S_exact: np.array, the exact snapshot matrix.
    - S_reconstructed: np.array, the reconstructed snapshot matrix.
    
    Returns:
    - error: float, the relative squared Frobenius-norm error.
    '''
    return np.linalg.norm(S_exact - S_reconstructed) / np.linalg.norm(S_exact)

#################################################

def get_dQ_dq(modes, q):
    ''' 
    Compute the derivative of the quadratic terms with respect to the reduced coordinates q_p.
    This function gives a matrix where each row corresponds to the derivative of a specific quadratic term
    with respect to the components of q_p.
    
    Parameters:
    - modes: int, number of modes (size of q_p).
    - q: np.array, the vector q_p of reduced coordinates.
    
    Returns:
    - dQ_dq: np.array, the derivative of the quadratic terms with respect to q_p.
    '''
    k = int(modes * (modes + 1) / 2)
    dQ_dq = np.zeros((k, modes))
    
    index = 0
    for i in range(modes):
        for j in range(i, modes):
            if i == j:
                dQ_dq[index, i] = 2 * q[i]  # Derivative of q_i^2 w.r.t q_i
            else:
                dQ_dq[index, i] = q[j]  # Derivative of q_i * q_j w.r.t q_i
                dQ_dq[index, j] = q[i]  # Derivative of q_i * q_j w.r.t q_j
            index += 1

    return dQ_dq

def compute_derivative(U_p, H, q_p):
    ''' 
    Compute the derivative of the quadratic manifold approximation with respect to q_p.
    
    Parameters:
    - U_p: np.array, the linear basis matrix (Phi_p).
    - H: np.array, the matrix H capturing the effect of secondary modes.
    - q_p: np.array, the vector of reduced coordinates in the primary space.
    
    Returns:
    - derivative: np.array, the derivative of the quadratic manifold approximation.
    '''
    modes = len(q_p)
    dQ_dq = get_dQ_dq(modes, q_p)
    
    # The derivative of the quadratic manifold approximation
    derivative = U_p + H @ dQ_dq

    return derivative

