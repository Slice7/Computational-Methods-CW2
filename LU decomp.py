import numpy as np

def my_LU(A):
    
    # Return an error if matrix is not square
    if not A.shape[0] == A.shape[1]:
        raise ValueError("Input matrix must be square")
        
    n = A.shape[0]  # The number of columns of A, (also number of rows as A is square)
    
    L = np.zeros((n, n), dtype='float64')  # Reserve space for L
    U = np.zeros((n, n), dtype='float64')  # Reserve space for U
    U[:] = A  # Copy A into U as we do not want to modify A
    np.fill_diagonal(L, 1)  # fill the diagonal of L with 1
    perm = list(range(n))
    
    for i in range(n - 1):
        pos_max = i + np.argmax(np.abs(U[i:, i]))  # Finding the positions of the maximum values in U
        perm[i], perm[pos_max] = perm[pos_max], perm[i]
        U[[i, pos_max], :] = U[[pos_max, i], :]  # Swap ith row with pos_max row
        L[[i, pos_max], :i] = L[[pos_max, i], :i]  # Swap ith row with pos_max row only up to ith column
        for j in range(i + 1, n):
            L[j, i] = U[j, i] / U[i, i]
            U[j, i:] -= L[j, i] * U[i, i:]
            
    P = np.eye(n)[perm, :]  # change the position of 1s in the identity matrix
    return P, L, U
    
