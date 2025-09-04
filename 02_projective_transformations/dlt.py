import numpy as np
from numpy import linalg  #zbog SVD algoritma
np.set_printoptions(precision=5, suppress=True)
 
eps = 0.000001

# norming of the matrix
def norm_matrix(matrix):
    coef = matrix[-1][-1]
    matrix = matrix/coef

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if abs(matrix[i][j]) <= eps:
                matrix[i][j] = abs(matrix[i][j])
    return matrix

 # DLT algorithm
def DLT(origs, imgs):
    P = []
    for i in range(len(origs)):
        x_1 = origs[i][0]
        x_2 = origs[i][1]
        x_3 = origs[i][2]

        x_1_ = imgs[i][0]
        x_2_ = imgs[i][1]
        x_3_ = imgs[i][2]


        P.append([0, 0, 0, -x_3_*x_1, -x_3_*x_2, -x_3_*x_3, x_2_*x_1, x_2_*x_2, x_2_*x_3])
        P.append([x_3_*x_1, x_3_*x_2, x_3_*x_3, 0, 0, 0, -x_1_*x_1, -x_1_*x_2, -x_1_*x_3])       
        
    P = np.array(P)
    U, D, V = np.linalg.svd(P)
    mat = np.array(V[8]).reshape(3,3)
    mat = norm_matrix(mat)
 
    return mat
 

