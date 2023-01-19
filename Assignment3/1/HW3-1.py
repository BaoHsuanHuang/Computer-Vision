import cv2
import numpy as np
import math
import random



def calculate_distance(pt1, pt2):
    dist = np.sqrt(np.sum((pt1 - pt2) ** 2))
    return dist

def SIFT_feature_matching(img1, kp1, d1, img2, kp2, d2, threshold):
    distances, neighbors = [], []
    match, good = [], []

    for i in range(len(d2)):
        min_dist, min_idx = 1e20, -1
        pt2 = d2[i]
        for j in range(len(d1)):
            pt1 = d1[j]
            dist = calculate_distance(pt1, pt2)
            if dist < min_dist:
                min_dist = dist
                min_idx = j
        if min_dist < threshold:
            neighbors.append((min_idx, i))
            distances.append(min_dist)
    for i in range(len(distances)):
        pair = neighbors[i]
        match = cv2.DMatch(pair[0], pair[1], distances[i])
        good.append(match)
    print("# of match: ", len(good))
    # matching = cv2.drawMatches(img2, kp2, img1, kp1, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    matching = cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return good, matching


def find_correspoinding_pts(kp1, kp2, matches):
    corres_pts = []
    for match in matches:
        (x1, y1) = kp1[match.queryIdx].pt
        (x2, y2) = kp2[match.trainIdx].pt
        corres_pts.append([x1, y1, x2, y2])
    return corres_pts


def draw_BoundingBox(img, H, flag):
    dst = []
    if flag==1:
        pts = [[15,45,1], [10,320,1], [435,315,1], [425,40,1]]
        # pts = np.float32([[15,45], [10,320], [435,315], [425,40]]).reshape(-1, 1, 2)
    elif flag==2:
        pts = [[25,15,1], [25,320,1], [425,320,1], [410,10,1]]
    else:
        pts = [[15,30,1], [15,300,1], [420,300,1], [420,30,1]]
    for pt in pts:
        tmp = np.dot(H, pt)  # tmp=H @ np.array(pt).transpose()
        if tmp[2] != 0:
            transformed_pt = (tmp/tmp[2])[0:2]
        else:
            transformed_pt = tmp[0:2]
        dst.append(transformed_pt)
    img = cv2.polylines(img, [np.int32(dst)], True, (125,255,0), 3, cv2.LINE_AA)
    return img

def show_DeviationVector(img, pairs):
    for pair in pairs:
        start_pt = (int(pair[0]), int(pair[1]))
        end_pt = (int(pair[2]), int(pair[3]))
        img = cv2.arrowedLine(img, start_pt, end_pt, (0, 255, 0), 2)
    return img


def choose_4points(src_pts, dst_pts):
    four_points = []
    idx = random.sample(range(len(src_pts)), 4)
    for i in idx:
      point = [src_pts[i][0], src_pts[i][1], dst_pts[i][0], dst_pts[i][1]]
      four_points.append(point)
    return np.array(four_points)

def find_HomographyMatrix(pairs):
    P = []
    x1,  y1, x1_, y1_  = pairs[0][0], pairs[0][1], pairs[0][2], pairs[0][3]
    x2,  y2, x2_, y2_  = pairs[1][0], pairs[1][1], pairs[1][2], pairs[1][3]
    x3,  y3, x3_, y3_  = pairs[2][0], pairs[2][1], pairs[2][2], pairs[2][3]
    x4,  y4, x4_, y4_  = pairs[3][0], pairs[3][1], pairs[3][2], pairs[3][3]
    P = [
        [x1, y1, 1, 0, 0, 0, -x1*x1_, -y1*x1_, -x1_],
        [0, 0, 0, x1, y1, 1, -x1*y1_, -y1*y1_, -y1_],
        [x2, y2, 1, 0, 0, 0, -x2*x2_, -y2*x2_, -x2_],
        [0, 0, 0, x2, y2, 1, -x2*y2_, -y2*y2_, -y2_],
        [x3, y3, 1, 0, 0, 0, -x3*x3_, -y3*x3_, -x3_],
        [0, 0, 0, x3, y3, 1, -x3*y3_, -y3*y3_, -y3_],
        [x4, y4, 1, 0, 0, 0, -x4*x4_, -y4*x4_, -x4_],
        [0, 0, 0, x4, y4, 1, -x4*y4_, -y4*y4_, -y4_]
    ]
    # P = [
    #     [-x1, -y1, -1, 0, 0, 0, x1*x1_, y1*x1_, x1_],
    #     [0, 0, 0, -x1, -y1, -1, x1*y1_, y1*y1_, y1_],
    #     [-x2, -y2, -1, 0, 0, 0, x2*x2_, y2*x2_, x2_],
    #     [0, 0, 0, -x2, -y2, -1, x2*y2_, y2*y2_, y2_],
    #     [-x3, -y3, -1, 0, 0, 0, x3*x3_, y3*x3_, x3_],
    #     [0, 0, 0, -x3, -y3, -1, x3*y3_, y3*y3_, y3_],
    #     [-x4, -y4, -1, 0, 0, 0, x4*x4_, y4*x4_, x4_],
    #     [0, 0, 0, -x4, -y4, -1, x4*y4_, y4*y4_, y4_]
    # ]
    U, s, V = np.linalg.svd(P)
    H = V[-1].reshape(3,3)
    return H / H[2][2]

def RANSAC_calculate_error(src_pts, dst_pts, H):
    n_points = len(src_pts)
    dist_all = []
    deviation_pts = []
    for i in range(n_points):
        dist = 0
        x1, y1 = src_pts[i][0], src_pts[i][1]
        x2, y2 = dst_pts[i][0], dst_pts[i][1]
        pt1 = [x1, y1, 1]
        pt1 = np.array(pt1).T
        pt2 = np.array([x2, y2])
        pt1_tmp = np.dot(H, pt1)
        # pt1_tmp = H @ pt1
        if pt1_tmp[2] != 0:
            pt1_transformed = (pt1_tmp/pt1_tmp[2])[0:2]
        else:
            pt1_transformed = pt1_tmp[0:2]
        x1, y1 = pt1_transformed[0], pt1_transformed[1]
        dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
        # dist = np.sqrt(np.sum((pt1 - pt2) ** 2))
        # dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)
        dist_all.append(dist)
        deviation_pts.append([x1, y1, x2, y2])
    dist_all = np.array(dist_all)
    return dist_all, deviation_pts

def RANSAC_algorithm(good, src_pts, dst_pts, threshold, iters):
    n_best_inliers = 0
    
    for i in range(iters):
        points = choose_4points(src_pts, dst_pts)
        H = find_HomographyMatrix(points)
        
        errors, deviation_pts = RANSAC_calculate_error(src_pts, dst_pts, H)
        mask = np.where(errors < threshold)[0]
        # inliers = matches[mask]
        # inliers = np.array(good)[mask.astype(int)]
        inliers = []
        inliers_deviation_pts = []
        for idx in mask:
            inliers.append(good[idx])
            inliers_deviation_pts.append(deviation_pts[idx])
        # print("len(inliers): ", len(inliers))

        n_inliers = len(inliers)
        if n_inliers > n_best_inliers:
            best_inliers = inliers.copy()
            n_best_inliers = n_inliers
            best_H = H.copy()
            best_deviation_pts = inliers_deviation_pts
    return best_inliers, best_H, best_deviation_pts



def main():
    img = cv2.imread("1-image.jpg")
    img_book1 = cv2.imread("1-book1.jpg")
    img_book2 = cv2.imread("1-book2.jpg")
    img_book3 = cv2.imread("1-book3.jpg")

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_book1_gray = cv2.cvtColor(img_book1, cv2.COLOR_BGR2GRAY)
    img_book2_gray = cv2.cvtColor(img_book2, cv2.COLOR_BGR2GRAY)
    img_book3_gray = cv2.cvtColor(img_book3, cv2.COLOR_BGR2GRAY)

    ############## Problem A ##############
    # SIFT
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=1500)
    sift_book = cv2.xfeatures2d.SIFT_create(nfeatures=800)

    # extract features and keypoints
    keypoints_img, descriptors_img = sift.detectAndCompute(img_gray, None)
    keypoints_book1, descriptors_book1 = sift_book.detectAndCompute(img_book1_gray, None)
    keypoints_book2, descriptors_book2 = sift_book.detectAndCompute(img_book2_gray, None)
    keypoints_book3, descriptors_book3 = sift_book.detectAndCompute(img_book3_gray, None)

    # # feature matching
    good_1, matched_1 = SIFT_feature_matching(img, keypoints_img, descriptors_img, img_book1, keypoints_book1, descriptors_book1, 130)
    good_2, matched_2 = SIFT_feature_matching(img, keypoints_img, descriptors_img, img_book2, keypoints_book2, descriptors_book2, 130)
    good_3, matched_3 = SIFT_feature_matching(img, keypoints_img, descriptors_img, img_book3, keypoints_book3, descriptors_book3, 180)
    cv2.imwrite("output/A_book1.jpg", matched_1)
    cv2.imwrite("output/A_book2.jpg", matched_2)
    cv2.imwrite("output/A_book3.jpg", matched_3)

    ############## Problem B : book1 ##############
    RANSAC_threshold, RANSAC_n_iters = 1.0, 2000
    src_pts = [keypoints_img[match.queryIdx].pt for match in good_1]
    dst_pts = [keypoints_book1[match.trainIdx].pt for match in good_1]
    inliers, H, deviation_vector = RANSAC_algorithm(good_1, dst_pts, src_pts, RANSAC_threshold, RANSAC_n_iters)
    print("[B_book1] # of inliers: {}/{}".format(len(inliers), len(good_1)))

    img_bounding = img.copy()
    img_bounding = draw_BoundingBox(img_bounding, H, 1)
    
    matching = cv2.drawMatches(img_bounding, keypoints_img, img_book1, keypoints_book1, inliers, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite("output/B_book1.jpg", matching)

    img_vector = img.copy()
    deviation_img = show_DeviationVector(img_vector, deviation_vector)
    cv2.imwrite("output/C_book1_vector.jpg", deviation_img)

    ############## Problem B : book2 ##############
    RANSAC_threshold, RANSAC_n_iters = 1.0, 2000
    src_pts = [keypoints_img[match.queryIdx].pt for match in good_2]
    dst_pts = [keypoints_book2[match.trainIdx].pt for match in good_2]
    inliers, H, deviation_vector = RANSAC_algorithm(good_2, dst_pts, src_pts, RANSAC_threshold, RANSAC_n_iters)
    print("[B_book2] # of inliers: {}/{}".format(len(inliers), len(good_2)))

    img_bounding = img.copy()
    img_bounding = draw_BoundingBox(img_bounding, H, 1)
    
    matching = cv2.drawMatches(img_bounding, keypoints_img, img_book2, keypoints_book2, inliers, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite("output/B_book2.jpg", matching)

    img_vector = img.copy()
    deviation_img = show_DeviationVector(img_vector, deviation_vector)
    cv2.imwrite("output/C_book2_vector.jpg", deviation_img)

    ############## Problem B : book3 ##############
    RANSAC_threshold, RANSAC_n_iters = 1.0, 2000
    src_pts = [keypoints_img[match.queryIdx].pt for match in good_3]
    dst_pts = [keypoints_book3[match.trainIdx].pt for match in good_3]
    inliers, H, deviation_vector = RANSAC_algorithm(good_3, dst_pts, src_pts, RANSAC_threshold, RANSAC_n_iters)
    print("[B_book3] # of inliers: {}/{}".format(len(inliers), len(good_3)))

    img_bounding = img.copy()
    img_bounding = draw_BoundingBox(img_bounding, H, 1)
    
    matching = cv2.drawMatches(img_bounding, keypoints_img, img_book3, keypoints_book3, inliers, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite("output/B_book3.jpg", matching)

    img_vector = img.copy()
    deviation_img = show_DeviationVector(img_vector, deviation_vector)
    cv2.imwrite("output/C_book3_vector.jpg", deviation_img)


if __name__ == '__main__':
    main()