import numpy as np
from numpy import linalg
import math
np.set_printoptions(precision=5, suppress=True) 
 
def centre(T):

    C1 = np.linalg.det(np.delete(T, 0, 1))
    C2 = np.linalg.det(np.delete(T, 1, 1))
    C3 = np.linalg.det(np.delete(T, 2, 1))
    C4 = np.linalg.det(np.delete(T, 3, 1))

    C = np.array([C1, -C2, C3, -C4]) * (-1/C4)
 
    C = np.where(np.isclose(C, 0) , 0.0 , C)
    return C
 

