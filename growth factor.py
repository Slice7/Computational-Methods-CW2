import numpy as np
from matplotlib import pyplot as plt
from numpy import mean

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


def growth_factor(A):
    # computes growth factor of matrix A
    rho = np.max(np.abs(U)) / np.max(np.abs(A))
    return rho


m = 20  # Number of matrices to take averages over
nvals = range(10, 501, 10)  # Sizes of different matrices
growth = np.zeros_like(nvals, dtype='float64')  # Reserve space for average growth factors

for i, n in enumerate(nvals):
    growth_factors = np.zeros(m, dtype='float64')  # Reserve space for current dimension
    A = np.random.randn(n, n)  # Gaussian distributed numbers
    # Note: Uses my_LU(A) from "LU decomp.py" file
    P, L, U = my_LU(A)
    for j in range(m):
        growth_factors[j] = growth_factor(A)  # Make a list of growth factors for current dimension
    growth[i] = mean(growth_factors)  # Make a list of average growth factors
    
plt.plot(nvals, growth)
plt.title('Growth factor')
plt.xlabel('Dimension')
plt.ylabel('Average growth factor')
plt.show()
"""As the dimensions increase, we see that the growth factor seems to increase linearly."""
