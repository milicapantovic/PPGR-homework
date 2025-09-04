# user function for cross product
def cross(p1, p2):
    x = p1[1] * p2[2] - p1[2] * p2[1]
    y = p1[0] * p2[2] - p1[2] * p2[0]
    z = p1[0] * p2[1] - p1[1] * p2[0]
    return [x, y, z]

# coordinates transformation (homogeneous -> affine)
def afinize(p):
    return [round(p[0] / p[2]), round(p[1] / p[2]), 1]

# function that calculates the coordinates of the invisible eighth vertex on a image
def osmoteme(temena):
    p = []
    for teme in temena:
        teme.append(1)
        p.append(teme)
    
    p_1 = p[4]
    p_2 = p[5]
    p_3 = p[6]
    p_5 = p[0]
    p_6 = p[1]
    p_7 = p[2]
    p_8 = p[3]

    xb_1 = afinize(cross(cross(p_2, p_6), cross(p_1, p_5)))
    xb_2 = afinize(cross(cross(p_2, p_6), cross(p_3, p_7)))
    xb_3 = afinize(cross(cross(p_1, p_5), cross(p_3, p_7)))
    xb = [round((xb_1[0] + xb_2[0] + xb_3[0]) / 3), round((xb_1[1] + xb_2[1] + xb_3[1]) / 3), 1]
    yb = cross(cross(p_5, p_6), cross(p_7, p_8))
    P_4 = afinize(cross(cross(p_8, xb), cross(p_3, yb)))
    return [round(P_4[0]), round(P_4[1])]
    
# testing
print(osmoteme([[32, 70], [195, 144], [195, 538], [30, 307], [251, 40], [454, 78], [455, 337]]))





