import numpy as np

def my_LU(A):
    # this function is taken from the 'LU decomp.py' file
    if not A.shape[0] == A.shape[1]:
        raise ValueError("Input matrix must be square")
        
    n = A.shape[0]
    L = np.zeros((n, n), dtype='float64')
    U = np.zeros((n, n), dtype='float64')
    U[:] = A
    np.fill_diagonal(L, 1)
    perm = list(range(n))
    
    for i in range(n - 1):
        pos_max = i + np.argmax(np.abs(U[i:, i]))
        perm[i], perm[pos_max] = perm[pos_max], perm[i]
        U[[i, pos_max], :] = U[[pos_max, i], :]
        L[[i, pos_max], :i] = L[[pos_max, i], :i]
        for j in range(i + 1, n):
            L[j, i] = U[j, i] / U[i, i]
            U[j, i:] -= L[j, i] * U[i, i:]
            
    P = np.eye(n)[perm, :]
    return P, L, U

def solve(A, b):
    # Solving Ax = b
    P, L, U = my_LU(A)
    n = A.shape[0]
    m = b.shape[1]
    x = np.zeros((n, m), dtype='float64')  # Reserve space for x
    y = np.zeros((n, m), dtype='float64')  # Reserve space for y
    # nxm dimension of x and y allow us to solve for multiple columns in b (and, by extension, x)
    b = np.dot(P, b)  # Permute b

    # Forward substitution
    for i in range(n):
        sum_Ly = 0
        for j in range(i):
            sum_Ly += L[i, j] * y[j]  # multiplying L by y
        y[i] = b[i] - sum_Ly

    # Backward substitution
    for i in range(n-1, -1, -1):
        sum_Ux = 0
        for j in range(i+1, n):
            sum_Ux += U[i, j] * x[j]  # multiplying U by x
        x[i] = (y[i] - sum_Ux) / U[i, i]

    return x
    
