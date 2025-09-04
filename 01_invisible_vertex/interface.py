import cv2
import sys

p = []

# user function for cross product
def cross(p1, p2):
    x = p1[1] * p2[2] - p1[2] * p2[1]
    y = p1[0] * p2[2] - p1[2] * p2[0]
    z = p1[0] * p2[1] - p1[1] * p2[0]
    return [x, y, z]

# coordinates transformation (homogeneous -> affine)
def afinize(p):
    return [round(p[0] / p[2]), round(p[1] / p[2]), 1]

# function that calculates the coordinates of the invisible eighth vertex of a box
def osmoteme(p):
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
    p_4 = afinize(cross(cross(p_8, xb), cross(p_3, yb)))
    return [round(p_4[0]), round(p_4[1])]

# after the user clicks on the image, the coordinates of the pixel clicked on are stocked in a list
# when the user clicks on the 7th vertex, application showes 8th vertex coordinates
def click_event(event, x, y, flags, params):
        font = cv2.FONT_HERSHEY_SIMPLEX
        if len(p) == 7:
                tacka = osmoteme(p)
                xt = tacka[0]
                yt = tacka[1]
                cv2.putText(img, f'{xt} {yt}', (xt, yt), font, 1, (255, 12, 0), 2)
                cv2.circle(img, (xt,yt), 3, (0,100,255), -1)
                cv2.imshow('Osmo teme', img)
        if event == cv2.EVENT_LBUTTONDOWN:
                p.append([x, y, 1])
                cv2.putText(img, f'{x}, {y}', (x,y), font, 1, (255, 255, 0), 2)
                cv2.circle(img, (x,y), 3, (0,255,255), -1)
                cv2.imshow('Osmo teme', img)

           
# main function
if __name__ == "__main__":
    img = cv2.imread('/change/to/your/image/path', 1)  
    img = cv2.resize(img, (0, 0), fx = 0.4, fy = 0.4)
    cv2.imshow('Osmo teme', img)

    cv2.setMouseCallback('Osmo teme', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
