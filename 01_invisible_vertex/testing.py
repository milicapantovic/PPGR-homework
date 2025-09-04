# Point class 
class Point():
    def __init__(self, x, y, z):
        self.X = x
        self.Y = y
        self.Z = z

    def getX(self):
        return self.X

    def getY(self):
        return self.Y
    
    def getZ(self):
        return self.Z
        
    # cross product
    def cross(self, other):
        x = self.Y * other.Z - self.Z * other.Y
        y = self.X * other.Z - self.Z * other.X
        z = self.X * other.Y - self.Y * other.X
        return Point(x, y, z)
    
    def afinize(self):
        return Point(round(self.X / self.Z), round(self.Y / self.Z), 1)
    

# function that calculates the coordinates of the invisible eighth vertex of a box
def nevidljivo(P_1, P_2, P_3, P_5, P_6, P_7, P_8):
    xb_1 = (P_2.cross(P_6)).cross(P_1.cross(P_5)).afinize()
    xb_2 = (P_2.cross(P_6)).cross(P_3.cross(P_7)).afinize()
    xb_3 = (P_1.cross(P_5)).cross(P_3.cross(P_7)).afinize()
    xb = Point(round((xb_1.X + xb_2.X + xb_3.X) / 3), round((xb_1.Y + xb_2.Y + xb_3.Y) / 3), round((xb_1.Z + xb_2.Z + xb_3.Z) / 3))
    yb = (P_5.cross(P_6)).cross(P_7.cross(P_8))
    P_4 = (P_8.cross(xb)).cross(P_3.cross(yb)).afinize()
    return P_4

# example points
p_1 = Point(2687,985,1)
p_2 = Point(1170,1710,1)
p_3 = Point(1231,1436,1)
p_5 = Point(2714,317,1)
p_6 = Point(1667,856,1)
p_7 = Point(1085,658,1)
p_8 = Point(2230,236,1)

# testing
p_4 = nevidljivo(p_1, p_2, p_3, p_5, p_6, p_7, p_8)
print('({0}, {1}, {2})'.format(p_4.X, p_4.Y, p_4.Z))




