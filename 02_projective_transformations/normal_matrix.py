import numpy as np
from numpy import linalg
import math

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
 



