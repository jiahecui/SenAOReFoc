import numpy as np

A = np.random.randint(1, 7, size=(3, 4))
B = np.ravel(A.copy())
print(A)
print(B)
C = B.copy()
print(np.size(C))
D = np.arange(0,12,3)
print(D)
E = C[D]
print(E)
B[1] = 5
print(np.ravel(A))
print(B)