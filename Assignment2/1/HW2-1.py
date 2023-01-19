import numpy as np
import cv2
from scipy import linalg
import math

def read_points_files():  
    file_left_path = "pt_2D_1.txt"
    file_right_path = "pt_2D_2.txt"
    cor1 = []
    cor2 = []
    # ====== file1 (left) ===========================
    f = open(file_left_path, "r")
    cnt = 0
    num_pair = 0
    for line in f.readlines():
      if cnt==0:
        num_pair = int(line)
        cnt += 1
      else:
        pair = line.split(" ")
        x, y = float(pair[0]), float(pair[1])
        cor1.append((x, y))
    f.close()
    # ====== file2 (right) ==========================
    f = open(file_right_path, "r")
    cnt = 0
    num_pair = 0
    for line in f.readlines():
      if cnt==0:
        num_pair = int(line)
        cnt += 1
      else:
        pair = line.split(" ")
        x, y = float(pair[0]), float(pair[1])
        cor2.append((x, y))
    f.close()
    return cor1, cor2

def calculate_fundamental_matrix(x1, x2):
    A = np.zeros((46, 9))
    for i in range(46):
        # [x'*x, x'*y, x', y'*x, y'*y, y', x, y, 1]
        A[i] = [
            x2[i][0]*x1[i][0], x2[i][0]*x1[i][1], x2[i][0],
            x2[i][1]*x1[i][0], x2[i][1]*x1[i][1], x2[i][1],
            x1[i][0],      x1[i][1],      1
        ]
    
    # linear least-squares
    U, S, V = linalg.svd(A)
    V_transpose = V[-1]
    F = V_transpose.reshape(3,3)

    # change to rank 2 
    U, S, V = linalg.svd(F)
    S[2] = 0
    tmp = np.dot(np.diag(S),V)
    F = np.dot(U, tmp)
    F = F / F[2,2]
    return F

def calculate_epipolar_lines(cor1, cor2, F):
    lines_left = []
    lines_right = []
    for i in range(46):
        x, y = cor1[i][0], cor1[i][1]
        z = 1
        x_vector = [x, y, z]
        ll = np.dot(F, x_vector)
        lines_right.append(ll)

        # F.transpose
        xx_vector = [cor2[i][0], cor2[i][1], 1]
        # l = np.dot(F, xx_vector)
        l = np.dot(F.transpose(), xx_vector)
        lines_left.append(l)
    return lines_left, lines_right

def drawlines(img, lines, cor1, cor2):
    img_copy = img.copy()
    row, col, channel = img_copy.shape
    color_line = (0, 0, 255)
    color_dot = (255, 255, 0)
    for row, pt1, pt2 in zip(lines, cor1, cor2):
        x0, y0 = map(int, [0, -row[2]/row[1]])
        x1, y1 = map(int, [col, -(row[2]+row[0]*col)/row[1]])
        img_copy = cv2.line(img_copy, (x0, y0), (x1, y1), color_line, 1)
        img_copy = cv2.circle(img_copy, (int(pt1[0]), int(pt1[1])), 4, color_dot, -1)
    return img_copy

def coordinates_normalized(x1, x2):
    x_left_sum = 0
    y_left_sum = 0
    x_right_sum = 0
    y_right_sum = 0
    for i in range(len(x1)):
        x_left_sum += x1[i][0]
        y_left_sum += x1[i][1]
        x_right_sum += x2[i][0]
        x_right_sum += x2[i][1]

    x_left_mean = x_left_sum / 46
    y_left_mean = y_left_sum / 46
    x_right_mean = x_right_sum / 46
    y_right_mean = y_right_sum / 46

    tmp_left = sum([((x-x_left_mean)**2 + (y-y_left_mean)**2)**0.5 for x,y in x1])
    tmp_right = sum([((x-x_right_mean)**2 + (y-y_right_mean)**2)**0.5 for x,y in x2])
    s_left = (46*2)**0.5/tmp_left
    s_right = (46*2)**0.5/tmp_right

    
    T_left = np.array([
                        [1/s_left, 0, -x_left_mean*(1/s_left)], 
                        [0, 1/s_left, -y_left_mean*(1/s_left)], 
                        [0, 0, 1]])
    T_right = np.array([
                        [1/s_right, 0, -x_right_mean*(1/s_right)], 
                        [0, 1/s_right, -y_right_mean*(1/s_right)], 
                        [0, 0, 1]])

    x1_normalized = []
    x2_normalized = []
    for i in range(len(x1)):
        x = x1[i][0]
        y = x1[i][1]
        vector = np.array([x, y, 1])
        pts_left = np.matmul(T_right, vector)
        x1_normalized.append((pts_left[0], pts_left[1]))

        x = x2[i][0]
        y = x2[i][1]
        vector = np.array([x, y, 1])
        pts_right = np.matmul(T_left, vector)
        x2_normalized.append((pts_right[0], pts_right[1]))
    
    return x1_normalized, x2_normalized, T_left, T_right

def restore_fundamental_matrix(f_tmp, t1, t2):
    t2_transpose = t2.transpose()
    F_matrix = t2_transpose @ f_tmp @ t1
    # F_matrix = F_matrix / F_matrix[2][2]
    return F_matrix

def calculate_mean_distance(pts, lines):
    avg = 0
    d = 0
    for i in range(len(pts)):
        a = lines[i][0]
        b = lines[i][1]
        c = lines[i][2]
        x = pts[i][0]
        y = pts[i][1]
        d += math.fabs((a*x + b*y + c))/(math.sqrt(a*a + b*b))
    avg = d / 46
    return avg


def main():
    image_left = cv2.imread("image1.jpg")
    image_right = cv2.imread("image2.jpg")

    width, height = image_left.shape[0], image_left.shape[1]

    # ====== question (a) ===========================================
    cor_left, cor_right = read_points_files()
    f_matrix = calculate_fundamental_matrix(cor_left, cor_right)
    print("(a) Fundamental Matrix")
    print(f_matrix)
    a_img1_lines, a_img2_lines = calculate_epipolar_lines(cor_left, cor_right, f_matrix)
    a_img1 = drawlines(image_left, a_img1_lines, cor_left, cor_right)
    a_img2 = drawlines(image_right, a_img2_lines, cor_right, cor_left)
    cv2.imwrite('output/a_img1.jpg', a_img1)
    cv2.imwrite('output/a_img2.jpg', a_img2)

    # ====== question (b) ===========================================
    cor_left_normalized, cor_right_normalized, T_left, T_right = coordinates_normalized(cor_left, cor_right)
    f_matrix_normalized = calculate_fundamental_matrix(cor_left_normalized, cor_right_normalized)
    f_matrix_normalized = restore_fundamental_matrix(f_matrix_normalized, T_right, T_left)
    print("\n(b) Fundamental Matrix")
    print(f_matrix_normalized)
    b_img1_lines, b_img2_lines = calculate_epipolar_lines(cor_left, cor_right, f_matrix_normalized)
    b_img1 = drawlines(image_left, b_img1_lines, cor_left, cor_right)
    b_img2 = drawlines(image_right, b_img2_lines, cor_right, cor_left)
    cv2.imwrite('output/b_img1.jpg', b_img1)
    cv2.imwrite('output/b_img2.jpg', b_img2)

    # ====== question (c) ===========================================
    a_img1_dist = calculate_mean_distance(cor_left, a_img1_lines)
    a_img2_dist = calculate_mean_distance(cor_right, a_img2_lines)
    b_img1_dist = calculate_mean_distance(cor_left, b_img1_lines)
    b_img2_dist = calculate_mean_distance(cor_right, b_img2_lines)
    print("\na_img1_dist: ", a_img1_dist)
    print("a_img2_dist: ", a_img2_dist)
    print("b_img1_dist: ", b_img1_dist)
    print("b_img2_dist: ", b_img2_dist)


if __name__ == '__main__':
    main()