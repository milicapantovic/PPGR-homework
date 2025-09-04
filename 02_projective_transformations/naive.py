import numpy as np
from itertools import combinations

np.set_printoptions(precision=5, suppress=True)

eps = 0.000001

#cross product
def product(p_1, p_2, p_3):
    return np.cross(np.array(p_1), np.array(p_2)).dot(np.array(p_3))

# are the points non-collinear
def check_position(points):
    for p_1, p_2, p_3 in combinations(points, 3):
        if product(p_1, p_2, p_3) == 0:
            return False
    return True

def fix_matrix(matrix):

    coef = matrix[-1][-1]
    matrix = matrix/coef

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if abs(matrix[i][j]) <= eps:
                matrix[i][j] = abs(matrix[i][j])
    return matrix

# find matrix of the projective transformation
def find_p(points):
    eq_1 = np.array([points[0][0], points[1][0], points[2][0]])
    eq_2 = np.array([points[0][1], points[1][1], points[2][1]])
    eq_3 = np.array([points[0][2], points[1][2], points[2][2]])
    coef = np.array([eq_1, eq_2, eq_3])
    b = np.array(points[3])
    x = np.linalg.solve(coef, b)
    points = np.array(points[:3])
    p = np.array([x[0]*points[0], x[1]*points[1], x[2]*points[2]]).transpose()
    return p


# naive algorithm
def naivni(origs, imgs):

    if not check_position(origs):
        return 'Losi originali!'
    elif not check_position(imgs):
        return 'Lose slike!'

    g = find_p(origs)
    h = find_p(imgs)
    f = np.dot(h, np.linalg.inv(g))

    return fix_matrix(f)




