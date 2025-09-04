import numpy as np
from numpy import linalg
import math
np.set_printoptions(precision=5, suppress=True) 
 
def extrinsic_matrix(T):
    T0 = np.delete(T, 3, 1)
    if np.linalg.det(T0) <= 0:
        T = -1 * T
    T0 = np.delete(T, 3, 1)
    T0_inv = np.linalg.inv(T0)

    Q, R = np.linalg.qr(T0_inv)

    if R[0][0] <= 0:
        R = np.matmul(np.diag([-1,1,1]), R)
        Q = np.matmul(Q, np.diag([-1, 1, 1]))
    if R[1][1] <= 0:
        R = np.matmul(R, np.diag([1,-1,1]))
        Q = np.matmul(Q, np.diag([1,-1,1]))
    if R[2][2] <= 0:
        R = np.matmul(np.diag([1,1,-1]), R)
        Q = np.matmul(Q, np.diag([1,1,-1]))

    A = Q 
    A = np.where(np.isclose(A, 0) , 0.0 , A)
    return A
 

