import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# left image vertex coordinates
p1l = [360, 678]
p2l = [955, 1038]
p3l = [1815, 585]
p4l = [1245, 330]
p5l = [1298, 858]
p6l = [538, 1283]
p7l = [1068, 1665]
p8l = [1813, 1133]
p9l = [2103, 498]
p10l = [2628, 873]
p11l = [3375, 610]
p12l = [2790, 318]
p13l = [2700, 808]
p14l = [2068, 1025]
p15l = [2530, 1458]
p16l = [3203, 1148]

# right image vertex coordinates
p1d = [393, 618]
p2d = [708, 950]
p3d = [1650, 630]
p4d = [1260, 380]
p5d = [1323, 858]
p6d = [553, 1173]
p7d = [853, 1535]
p8d = [1683, 1145]
p9d = [1960, 570]
p10d = [2268, 953]
p11d = [3188, 775]
p12d = [2740, 450]
p13d = [2675, 935]
p14d = [1958, 1080]
p15d = [2238, 1520]
p16d = [3060, 1313]


left = [p1l, p2l, p3l, p4l, p5l, p6l, p7l, p8l, p9l, p10l, p11l, p12l, p13l, p14l, p15l, p16l]
right = [p1d, p2d, p3d, p4d, p5d, p6d, p7d, p8d, p9d, p10d, p11d, p12d, p13d, p14d, p15d, p16d]


#img width: 4000 pixels

def fix_coords(point):
    return [4000 - point[0], point[1], 1]

left_fixed = list(map(fix_coords, left))
right_fixed = list(map(fix_coords, right))

def equation(left_point, right_point):
    return [a * b for a in left_point for b in right_point]

matrixForm = [equation(left, right) for left, right in zip(left_fixed, right_fixed)]
U, D, V = np.linalg.svd(matrixForm)
F = np.array(V[8])
FF = F.reshape(3, 3).T
print("fundamental matrix FF:")
print(FF)

# epipoles
UU, DD, VV = np.linalg.svd(FF)
e1 = np.array(VV[2])
e1 = e1 * (1 / e1[2])

e2 = np.array(UU[:,2])
e2 = e2 * (1 / e2[2])

print("first epipole: ", e1)
print("second epipole: ", e2)

# fundamental matrix fix
DD = np.diag(DD)
DD1 = np.diag([1, 1, 0])
DD1 = np.dot(DD1, DD)

FF1 = np.dot(np.dot(UU, DD1), VV)
# print("determinants:")
# print(np.linalg.det(FF), np.linalg.det(FF1))
print("Fixed fundamental matrix FF1:")
print(FF1)


def test(x, y):
    return np.dot(np.dot(y, FF1), x)

# for left, right in zip(left_fixed, right_fixed):
#     print(test(left, )


#essential matrix 

# camera's calibration matrix 
K1 = np.array( [[3190.79476, 7.26940398, 2122.85594],
 [0.00000000, 3050.98387,701.838781],
 [0.00000000, 0.00000000, 1.00000000]])

K2 = K1
EE = np.dot(np.dot(K2.T, FF1) ,K1)
print("Osnovna matrica EE:")
print(EE)
# essential matrix decomposition
Q0 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
E0 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])

U, SS, V = np.linalg.svd(-EE)

# print(np.linalg.det(U))
# print(np.linalg.det(V))
# print(SS)


EC = np.dot(np.dot(U, E0), U.T)
AA = np.dot(np.dot(U, Q0.T), V)

print("Skew-symmetric matrix EC (representing the position of the first camera in the coordinate system of the second camera):")
print(EC)

print("Orientation matrix of the first camera in the coordinate system of the second camera AA:")
print(AA)


# ECA = np.dot(EC, AA)
# scalar = EE[0,0] / ECA[0,0]
# print(scalar)
# print(ECA)
# print(ECA.dot(scalar))
# print()
# print(EE)


def skew2vec(matrix):
    return [matrix[2, 1], matrix[0, 2], matrix[1, 0]]

CC = skew2vec(EC)
print("Coordinates of the first camera center in the coordinate system of the second camera CC:")
print(CC)

# Camera matrices (in the coordinate system of the second camera)

# Position of the first camera in the second camera's coordinate system
CC1 = -np.dot(AA.T, CC)
print("Position of the first camera in the coordinate system of the second camera CC1:")
print(CC1)

T1 = np.row_stack((np.dot(K1, AA.T).T, np.dot(K1, CC1))).T
print("First camera matrix T1:")
print(T1)


T2 = np.array([
    [3190.79476, 7.26940398, 2122.85594, 0.0],
    [0.0, 3050.98387, 701.838781, 0.0],
    [0.0, 0.0, 1.0, 0.0]
])

print("Camera calibration matrix T2:")
print(T2)

def equations(t1, t2, m1, m2):
    result = np.array([
        m1[1]*t1[2] - m1[2]*t1[1],
        -m1[0]*t1[2] + m1[2]*t1[0],
        m2[1]*t2[2] - m2[2]*t2[1],
        -m2[0]*t2[2] + m2[2]*t2[0]
    ])
    return result


def to_affine(mat):
    return mat[:-1] / mat[-1]


def triangulate(t1, t2, m1, m2):
    linear_system = equations(t1, t2, m1, m2)
    _, _, V = np.linalg.svd(linear_system)
    M = to_affine(np.array(V[3]))
    return M

points3D = [triangulate(T1, T2, m1, m2) for m1, m2 in zip(left_fixed, right_fixed)]


p = {}  # Empty list to store variables
print("3D coordinates:")
for i in range(0, 16):
    p[i+1] = points3D[i]
    print(p[i+1])

sides_cube_1 = [[p[6], p[7], p[8], p[5]],
                [p[1], p[2], p[3], p[4]],
                [p[6], p[7], p[2], p[1]],
                [p[8], p[5], p[4], p[3]],
                [p[7], p[8], p[3], p[2]],
                [p[6], p[5], p[4], p[1]]
                ]


sides_cube_2 = [[p[14], p[15], p[16], p[13]], 
[p[9], p[10], p[11], p[12]], 
[p[14], p[15], p[10], p[9]], 
[p[16], p[13], p[12], p[11]], 
[p[15], p[16], p[11], p[10]], 
[p[14], p[13], p[12], p[9]]
]


# The relationship between the two camera coordinate systems is:
# MC = AA * M + CC
# where M represents the coordinates of a point in the second camera's coordinate system,
# and MC represents the coordinates of the same point in the first camera's coordinate system.
# Conversely, we can compute M from MC as:
# M = AA⁻¹ * MC - AA⁻¹ * CC  (or equivalently, M = AA_T * MC - AA_T * CC)
# The matrix AA_T encodes the rotation of the first camera relative to the second camera's coordinate system.

camera_center = CC

x_axis_camera = AA.T[0]
y_axis_camera = AA.T[1]
z_axis_camera = AA.T[2]


figure = plt.figure()
axes = figure.add_subplot(111, projection='3d')
axes.add_collection3d(Poly3DCollection(sides_cube_1, facecolors='blue', linewidths=0.5, edgecolors='black', alpha=0.4))
axes.add_collection3d(Poly3DCollection(sides_cube_2, facecolors='red', linewidths=0.5, edgecolors='black', alpha=0.4))

axes.quiver(*camera_center, *x_axis_camera, color = 'blue')
axes.quiver(*camera_center, *y_axis_camera, color = 'green')
axes.quiver(*camera_center, *z_axis_camera, color = 'red')


axes.set_xlabel('X osa')
axes.set_ylabel('Y osa')
axes.set_zlabel('Z osa')

axes.set_xlim([-2, 2])
axes.set_ylim([-2, 2])
axes.set_zlim([-7, 0])

plt.show()
