import numpy as np
import cv2
from scipy import linalg

def drawbox(img, pts):
    start_pt = pts[0]
    end_pt = pts[1]
    output = cv2.line(img, tuple(start_pt), tuple(end_pt), (0, 0, 255), 3)

    start_pt = pts[0]
    end_pt = pts[3]
    output = cv2.line(output, tuple(start_pt), tuple(end_pt), (0, 0, 255), 3)

    start_pt = pts[1]
    end_pt = pts[2]
    output = cv2.line(output, tuple(start_pt), tuple(end_pt), (0, 0, 255), 3)

    start_pt = pts[2]
    end_pt = pts[3]
    output = cv2.line(output, tuple(start_pt), tuple(end_pt), (0, 0, 255), 3)
    return output


def calculate_homography_matrix(left, pts_right):
    # P * h = x
    # h = inverse(P) * x
    P = np.array([
            [left[0][0], left[0][1], 1, 0, 0, 0, -left[0][0]*pts_right[0][0], -left[0][1]*pts_right[0][0]],
            [0, 0, 0, left[0][0], left[0][1], 1, -left[0][0]*pts_right[0][1], -left[0][1]*pts_right[0][1]],
            [left[1][0], left[1][1], 1, 0, 0, 0, -left[1][0]*pts_right[1][0], -left[1][1]*pts_right[1][0]],
            [0, 0, 0, left[1][0], left[1][1], 1, -left[1][0]*pts_right[1][1], -left[1][1]*pts_right[1][1]],
            [left[2][0], left[2][1], 1, 0, 0, 0, -left[2][0]*pts_right[2][0], -left[2][1]*pts_right[2][0]],
            [0, 0, 0, left[2][0], left[2][1], 1, -left[2][0]*pts_right[2][1], -left[2][1]*pts_right[2][1]],
            [left[3][0], left[3][1], 1, 0, 0, 0, -left[3][0]*pts_right[3][0], -left[3][1]*pts_right[3][0]],
            [0, 0, 0, left[3][0], left[3][1], 1, -left[3][0]*pts_right[3][1], -left[3][1]*pts_right[3][1]]])
    x = np.array([
            [pts_right[0][0]],
            [pts_right[0][1]],
            [pts_right[1][0]],
            [pts_right[1][1]],
            [pts_right[2][0]],
            [pts_right[2][1]],
            [pts_right[3][0]],
            [pts_right[3][1]]])
    
    P_inverse = np.linalg.inv(P)
    h = np.dot(P_inverse, x)
    h = np.append(h, 1)
    H = h.reshape(3,3)
    return H


def calculate_bilinear_pixel(img, x, y):
    dx = x - int(x)
    dx = round(dx) # dx=0.48
    dy = y - int(y)
    dy = round(dy) # fy=0.08
    
    vector = np.zeros((3,))
    vector += img[int(y), int(x)] * (1-dx) * (1-dy)
    vector += img[int(y), int(x)+1] * dx * (1-dy)
    vector += img[int(y)+1, int(x)+1] * dx * dy
    vector += img[int(y)+1, int(x)] * (1-dx) * dy

    return vector


def backward_warpping(img, rectified_img, selected_point):
    width, height = rectified_img.shape[1], rectified_img.shape[0]
    corners = np.array([[0, 0], [width-1, 0],[width-1, height-1], [0, height-1]])
    H = calculate_homography_matrix(corners, selected_point)
    print("Homography Matrix:")
    print(H)

    for y in range(height):
        for x in range(width):
            A_transpose = np.array([[x, y, 1]]).T
            B = np.dot(H, A_transpose)
            alpha = B[2][0]
            new_x = B[0][0] / alpha
            new_y = B[1][0] / alpha
            feature_vector = calculate_bilinear_pixel(img, new_x, new_y)
            rectified_img[y][x] = feature_vector
    return rectified_img

def main():
    img = cv2.imread('Delta-Building.jpg')
    selected_point = np.array([[458, 427], [776, 289], [775, 950], [448, 820]])
    img_copy = img.copy()

    selected_img = drawbox(img_copy, selected_point)
    rectified_img = np.zeros((400, 400, 3))
    rectified_img = backward_warpping(img, rectified_img, selected_point)

    # cv2.imshow(selected_img)
    cv2.imwrite('output/selected_img.jpg', selected_img)
    # cv2.imshow(rectified_img)
    cv2.imwrite('output/rectified_img.jpg', rectified_img)

if __name__ == '__main__':
    main()