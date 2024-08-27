import numpy as np

def get_sym(qi):
    ''' Auxiliary function to get the symmetric part of q kron q '''
    size = qi.shape[0]
    vec = []

    for i in range(size):
        for j in range(size):
            if j >= i:
                vec.append(qi[i] * qi[j])

    return np.array(vec)

def getQ(modes, Ns, q):
    ''' Populates Q row by row '''
    k = int(modes * (modes + 1) / 2)
    Q = np.empty((k, Ns))

    for i in range(Ns):
        Q[:, i] = get_sym(q[:, i])

    return Q

def get_single_Q(modes, q):
    ''' Populates Q row by row '''
    k = int(modes * (modes + 1) / 2)
    Q = np.empty(k)

    Q = get_sym(q)

    return Q

def getE(N, Ns, S, phi, q):
    ''' Populates E row by row '''
    E = np.empty((N, Ns))
    for i in range(Ns):
        E[:, i] = S[:, i] - phi @ q[:, i]  # u_i - Vq_i - uref

    return E

def getH(Q, E, modes, N, alpha):
    ''' Populates H row by row '''

    Uq, Sigmaq, Yq_T = np.linalg.svd(Q, full_matrices=False)
    Nq = Sigmaq.shape[0]
    Sigmaq_Squared = np.square(Sigmaq)

    k = int(modes * (modes + 1) / 2)
    H = np.empty((N, k))

    for i in range(N):
        print(f"{i} out of {N}")
        h_i = 0
        for l in range(Nq):
            h_i += (Sigmaq_Squared[l] / (Sigmaq_Squared[l] + alpha**2)) * (np.dot(Yq_T[l], E[i, :].T) / Sigmaq[l]) * Uq[:, l]
        H[i, :] = h_i

    return H

def relative_error(S_exact, S_reconstructed):
    """Calculate the relative squared Frobenius-norm error."""
    return np.linalg.norm(S_exact - S_reconstructed) / np.linalg.norm(S_exact)

#################################################

def get_dQ_dq(modes, q):
    """
    Compute the derivative of the quadratic terms with respect to the reduced coordinates q_p.
    This will give a matrix where each row corresponds to the derivative of a specific quadratic term
    with respect to the components of q_p.
    
    Parameters:
    - modes: int, number of modes (size of q_p).
    - q: np.array, the vector q_p of reduced coordinates.

    Returns:
    - dQ_dq: np.array, the derivative of the quadratic terms with respect to q_p.
    """
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
    """
    Compute the derivative of the quadratic manifold approximation with respect to q_p.

    Parameters:
    - U_p: np.array, the linear basis matrix (Phi_p).
    - H: np.array, the matrix H capturing the effect of secondary modes.
    - q_p: np.array, the vector of reduced coordinates in the primary space.

    Returns:
    - derivative: np.array, the derivative of the quadratic manifold approximation.
    """
    modes = len(q_p)
    dQ_dq = get_dQ_dq(modes, q_p)
    
    # The derivative of the quadratic manifold approximation
    derivative = U_p + H @ dQ_dq

    return derivative

def finite_difference_derivative(U, H, q_p, h=1e-5):
    """
    Compute the finite difference approximation of the derivative.

    Parameters:
    - U: np.array, the linear basis matrix (Phi_p).
    - H: np.array, the matrix H capturing the effect of secondary modes.
    - q_p: np.array, the vector of reduced coordinates in the primary space.
    - h: float, the small perturbation for finite difference.

    Returns:
    - approx_derivative: np.array, the finite difference approximation of the derivative.
    """
    modes = len(q_p)
    approx_derivative = np.zeros((U.shape[0], modes))

    # Compute the original quadratic manifold approximation
    original_output = U @ q_p + H @ get_single_Q(modes, q_p)
    
    for i in range(modes):
        q_p_plus = q_p.copy()
        q_p_minus = q_p.copy()

        q_p_plus[i] += h
        q_p_minus[i] -= h

        output_plus = U @ q_p_plus + H @ get_single_Q(modes, q_p_plus)
        output_minus = U @ q_p_minus + H @ get_single_Q(modes, q_p_minus)

        # Finite difference approximation
        approx_derivative[:, i] = (output_plus - output_minus) / (2 * h)
    
    return approx_derivative

def check_derivative(U, H, q_p, analytical_derivative):
    """
    Compare the analytical derivative with the finite difference approximation.

    Parameters:
    - U: np.array, the linear basis matrix (Phi_p).
    - H: np.array, the matrix H capturing the effect of secondary modes.
    - q_p: np.array, the vector of reduced coordinates in the primary space.
    - analytical_derivative: np.array, the derivative computed analytically.

    Returns:
    - difference: np.array, the difference between analytical and finite difference derivatives.
    """
    approx_derivative = finite_difference_derivative(U, H, q_p)
    difference = np.linalg.norm(analytical_derivative - approx_derivative)
    print(f"Difference between analytical and finite difference derivatives: {difference}")
    return difference

def compare_derivatives(U, H, q_p, analytical_derivative, h=1e-5):
    """
    Compare each component of the analytical and finite difference derivatives.

    Parameters:
    - U: np.array, the linear basis matrix (Phi_p).
    - H: np.array, the matrix H capturing the effect of secondary modes.
    - q_p: np.array, the vector of reduced coordinates in the primary space.
    - analytical_derivative: np.array, the derivative computed analytically.
    - h: float, the small perturbation for finite difference.

    Returns:
    - max_difference: float, the maximum difference found.
    """
    modes = len(q_p)
    approx_derivative = finite_difference_derivative(U, H, q_p, h)

    # Print out each component's difference
    max_difference = 0.0
    for i in range(modes):
        for j in range(U.shape[0]):
            difference = abs(analytical_derivative[j, i] - approx_derivative[j, i])
            print(f"Component ({j},{i}) - Analytical: {analytical_derivative[j, i]}, Finite Difference: {approx_derivative[j, i]}, Difference: {difference}")
            if difference > max_difference:
                max_difference = difference

    return max_difference

if __name__ == "__main__":
    # Assume we have already computed these variables
    Phi_p = np.load("U_truncated.npy")  # Example loading of U
    H = np.load("H_quadratic.npy")  # Example loading of H
    q_p = np.random.randn(Phi_p.shape[1])  # Example q_p

    # Analytical derivative computed by your function
    analytical_derivative = compute_derivative(Phi_p, H, q_p)

    # Compare the analytical derivative with the finite difference approximation
    max_diff = compare_derivatives(Phi_p, H, q_p, analytical_derivative)

    print(f"Maximum difference between analytical and finite difference derivatives: {max_diff}")
    
    # If the maximum difference is small (e.g., less than a tolerance), the derivative is likely correct
    tolerance = 1e-6
    if max_diff < tolerance:
        print("The derivative seems to be correctly implemented.")
    else:
        print("There might be an issue with the derivative implementation.")

