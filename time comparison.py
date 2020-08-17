import numpy as np
from scipy.linalg import solve as scipy_solve
import timeit
from matplotlib import pyplot as plt

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
    # this function is taken from the 'solve.py' file
    P, L, U = my_LU(A)
    n = A.shape[0]
    m = b.shape[1]
    x = np.zeros((n, m), dtype='float64')
    y = np.zeros((n, m), dtype='float64')
    b = np.dot(P, b)
    
    for i in range(n):
        sum_Ly = 0
        for j in range(i):
            sum_Ly += L[i, j] * y[j]
        y[i] = b[i] - sum_Ly
        
    for i in range(n-1, -1, -1):
        sum_Ux = 0
        for j in range(i+1, n):
            sum_Ux += U[i, j] * x[j]
        x[i] = (y[i] - sum_Ux) / U[i, i]
        
    return x

dims = range(10, 501, 10)
m = 5  # number of columns in b
repeats = 5

my_solve_times = np.zeros_like(dims, dtype='float64')
scipy_times = np.zeros_like(dims, dtype='float64')

for i, n in enumerate(dims):
    
    A = np.random.rand(n, n)
    b = np.random.rand(n, m)
    
    def my_solve_test():
        # Note: Uses solve(A, b) from "solve.py" file
        x = solve(A, b)
    
    def scipy_solve_test():
        x = scipy_solve(A, b)
    
    my_solve_times[i] = timeit.Timer(my_solve_test).timeit(repeats)*1./repeats
    scipy_times[i] = timeit.Timer(scipy_solve_test).timeit(repeats)*1./repeats
    
plt.semilogy(dims, my_solve_times, 'r', label='My solve function')
plt.semilogy(dims, scipy_times, 'b', label='Scipy solve function')
plt.legend(loc='upper left')
plt.title('Me vs scipy')
plt.xlabel('Dimension')
plt.ylabel('Average time (sec)')
plt.show()
"""We can clearly see that the in-built solve function is a lot faster than my own implementation,
by almost an order of 10^3."""
