import cv2
import sys
import numpy as np
# import dlt module
import dlt_normal as dltn

points = []

def select_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append(np.array([x, y, 1]))
            cv2.circle(img, (x ,y), 5, (0,0,255), -1)
            cv2.imshow("Removal of projective distortion", img)
        if len(points) == 4:
            print('4 points selected.')

           

if __name__ == "__main__":
    img = cv2.imread('original.jpg', 1)
    img =  cv2.resize(img, dsize=(1170,900), interpolation=cv2.INTER_CUBIC)
    if img is None:
        print("Error: Image not found.")
        sys.exit(1)

    cv2.imshow('Removal of projective distortion', img)
    cv2.setMouseCallback("Removal of projective distortion", select_points)
  
    while len(points) < 4:
        cv2.waitKey(1)

    cv2.destroyAllWindows()

    img_height, img_width, _ = img.shape
    
    dst_pts = np.array([[0, 0, 1], [img_width - 1, 0, 1], [img_width - 1, img_height - 1, 1], [0, img_height - 1, 1]])

    transform_matrix = dltn.DLTwithNormalization(points, dst_pts)

    result = cv2.warpPerspective(img, transform_matrix, (img_width, img_height), flags = cv2.INTER_LINEAR)
    cv2.imwrite('result.jpg', result)
    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
