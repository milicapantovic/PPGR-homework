import numpy as np
from numpy import linalg
import math
np.set_printoptions(precision=5, suppress=True) 

def fix_matrix(matrix):
    coef = matrix[-1][-1]
    matrix = matrix/coef

    return matrix

def two_equations(img, orig):
    zeros = np.array([0,0,0,0])
    fst = np.array(np.concatenate((zeros, -img[2]*orig, img[1]*orig)))
    snd = np.array(np.concatenate((img[2]*orig, zeros, -img[0]*orig)))

    return fst, snd

def create_matrix_from_equations(imgs, origs):
    mat = []
    for img, orig in zip(imgs, origs):
            fst, snd = two_equations(img, orig)
            mat.append(fst)
            mat.append(snd)
    return mat

def camera_matrix(pts2D, pts3D):
    a = create_matrix_from_equations(pts2D, pts3D)
    a = np.array(a)
    U, S, Vh = np.linalg.svd(a)
    T = np.array(Vh[11])
    T = T.reshape(3, 4)
    T = fix_matrix(T)
    T = np.where(np.isclose(T, 0) , 0.0 , T)
    return T
 

