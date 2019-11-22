import numpy as np

def array_integrate(A, B, A_start, B_start, B_end):
    """
    Integrates array A into array B and returns array B

    Args:
        A as numpy array
        B as numpy array
        A_start: index with respect to A of the upper left corner of the overlap
        B_start: index with respect to B of the upper left corner of the overlap
        B_end：index with respect to B of the lower right corner of the overlap
    """
    A_start, B_start, B_end = map(np.asarray, [A_start, B_start, B_end])
    shape = B_end - B_start
    B_slices = tuple(map(slice, B_start, B_end + 1))
    A_slices = tuple(map(slice, A_start, A_start + shape + 1))
    print('A_start:', A_start)
    print('B_start:', B_start)
    print('B_end', B_end)
    print('shape', shape)
    print('(A_slices', A_slices)
    print('B_slices', B_slices)
    B[B_slices] = A[A_slices]

    return B

A = np.zeros((21,15))
B = np.ones((16,15))
A_start = [11, 5]
B_start = [6, 0]
B_end = [15, 9]
array_integrate(A, B, A_start, B_start, B_end)
print(B)