import numpy as np
import math
from numpy import linalg  #zbog SVD algoritma
np.set_printoptions(precision=5, suppress=True)
 
# help functions
def afinize(p):
    return [p[0] / p[2], p[1] / p[2], 1]

def norm(p):
    return math.sqrt(p[0]**2 + p[1]**2)
 
def scale(factor):
    return np.array([[factor, 0, 0], [0, factor, 0], [0, 0, 1]])

def translate(p):
    return np.array([[1, 0, -p[0]], [0, 1, -p[1]], [0, 0, 1]])

def normMatrix(points):
    points = np.array([afinize(p) for p in points])
    B = np.mean(points, axis = 0)
    G = translate(B)
    translated_points = np.array([np.array(p-B) for p in points])
   
    average_distance = np.mean(np.array([norm(tp) for tp in translated_points]))
    factor = math.sqrt(2) / average_distance
    S = scale(factor)
  
    mat = np.dot(S, G)   
    return mat
 
def fix_matrix(matrix):

    coef = matrix[-1][-1]
    matrix = matrix/coef
    matrix[np.isclose(matrix, 0)] = 0
    return matrix

 
 
 
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
    mat = fix_matrix(mat)
 
    return mat


def DLTwithNormalization(origs, imgs):
# vaš kod
    T= np.array(normMatrix(origs))
    transformed_origs = [np.dot(T, orig) for orig in origs]
    T_ = np.array(normMatrix(imgs))
    transformed_imgs = [np.dot(T_, im) for im in imgs]
    P = DLT(transformed_origs, transformed_imgs)

    mat = np.linalg.inv(T_).dot(P).dot(T)
    mat = fix_matrix(mat)
    return mat
 
 

