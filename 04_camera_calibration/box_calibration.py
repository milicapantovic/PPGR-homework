import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D

def centre(T):
    C1 = np.linalg.det(np.delete(T, 0, 1))
    C2 = np.linalg.det(np.delete(T, 1, 1))
    C3 = np.linalg.det(np.delete(T, 2, 1))
    C4 = np.linalg.det(np.delete(T, 3, 1))

    C = np.array([C1, -C2, C3, -C4]) * (-1/C4)
 
    C = np.where(np.isclose(C, 0) , 0.0 , C)
    return C
 
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
 
def fix_matrix(matrix):
    coef = matrix[-1][-1]
    matrix = matrix/coef

    return matrix
 
def calibration_matrix(T):

    T0 = np.delete(T, 3, 1)
    if np.linalg.det(T0) <= 0:
        T = -1 * T
    
    T0 = np.delete(T, 3, 1)
    T0_inv = np.linalg.inv(T0)

    Q, R = np.linalg.qr(T0_inv)

    if R[0][0] < 0:
        R = np.matmul(np.diag([-1, 1, 1]), R)
        Q = np.matmul(Q, np.diag([-1, 1, 1]))
    if R[1][1] < 0:
        R = np.matmul(np.diag([1, -1, 1]), R)
        Q = np.matmul(Q, np.diag([1, -1, 1]))
    if R[2][2] < 0:
        R = np.matmul(np.diag([1, 1,-1]), R)
        Q = np.matmul(Q, np.diag([1, 1,-1]))

    K = np.linalg.inv(R)
    K = fix_matrix(K)
    K = np.where(np.isclose(K, 0) , 0.0 , K)
    return K
 
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

# image width: 1792 pixels
# testing
origs = np.array([1, -1, -1]) * (np.array([1700, 0, 0]) - np.array([[904, 1877, 1], [1435, 2033, 1], [916, 2249, 1], [377, 2044, 1],  
                                                            [423, 2696, 1], [923, 3001, 1], [1405, 2687, 1],  [907, 1965, 1], [910, 2088, 1]]))

imgs = np.array([[0, 0, 3, 1], [0, 3, 3, 1], [3, 3, 3, 1], [3, 0, 3, 1], [3, 0, 0, 1], [3, 3, 0, 1], [0, 3, 0, 1], [1, 1, 3, 1], [2, 2, 3, 1]])

                #     v5     	 v6           v7        v8         v4         v3        v2         v1
vertices = np.array([[3, 0, 0], [3, 3, 0], [0, 3, 0], [0, 0, 0], [3, 0, 3], [3, 3, 3], [0, 3, 3], [0, 0, 3]])

edges = [[vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
         [vertices[4], vertices[5], vertices[6], vertices[7]],  # up
         [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
         [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
         [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
         [vertices[0], vertices[3], vertices[7], vertices[4]]]  # left

origin = np.array([0, 0, 0])

x_box_axis = np.array([7, 0, 0])
y_box_axis = np.array([0, 7, 0])
z_box_axis = np.array([0, 0, 7])

T1 = camera_matrix(origs, imgs)
calibrationMatrix = calibration_matrix(T1)
centre_position = centre(T1)
centre_position = centre_position[:3]
extrinsicMatrix = extrinsic_matrix(T1)


x_camera_axis = extrinsicMatrix[0]
y_camera_axis = extrinsicMatrix[1]
z_camera_axis = extrinsicMatrix[2]


figura = plt.figure()
ose = figura.add_subplot(111, projection='3d')
ose.add_collection3d(Poly3DCollection(edges, facecolors='orchid', linewidths=0.5, edgecolors='black', alpha=1))

ose.quiver(*origin, *x_box_axis, color = 'blue')
ose.quiver(*origin, *y_box_axis, color = 'green')
ose.quiver(*origin, *z_box_axis, color = 'red')

ose.quiver(*centre_position, *x_camera_axis, color = 'blue')
ose.quiver(*centre_position, *y_camera_axis, color = 'green')
ose.quiver(*centre_position, *z_camera_axis, color = 'red')


ose.set_xlabel('X axis')
ose.set_ylabel('Y axis')
ose.set_zlabel('Z axis')

ose.set_xlim([-1, 15])
ose.set_ylim([-1, 15])
ose.set_zlim([0, 15])

plt.show()
