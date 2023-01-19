import numpy as np
import cv2
import math
from scipy import signal


def SIFT_detector(img, gray, id):
  if id=='a':
    sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.232) # 115 keypoints
  if id=='b':
    # sift = cv2.xfeatures2d.SIFT_create(edgeThreshold = 1.07)
    sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.2555) # 115 keypoints
  kp = sift.detect(gray, None)
  keypoints, descriptors = sift.detectAndCompute(gray, None)
  # print(len(keypoints))
  result = cv2.drawKeypoints(img ,
                             kp ,
                             img , (0, 0, 255),
                             flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
  return keypoints, descriptors, result



def SIFT_feature_matching(img1, img2, kp1, descriptor1, kp2, descriptor2):
	isMatch = [-1] * len(descriptor2)
	table = [-1] * len(descriptor1)

	for i in range(len(descriptor1)):
		distance = []
		min_value = -1
		min_index = -1
		# point1 = np.array((kp1[i].pt[0], kp1[i].pt[1]))
		point1 = descriptor1[i]
		for j in range(len(descriptor2)):
			point2 = descriptor2[j]
			# point2 = np.array((kp2[j].pt[0], kp2[j].pt[1]))
			# d = np.linalg.norm(point1 - point2)
			d = np.sqrt(np.sum(np.square(point1-point2)))
			distance.append((j, d))

		distance = sorted(distance, key=lambda x: x[1])
		for j in range(len(distance)):
			idx, dist = distance[j]
			if isMatch[idx] == -1:
				isMatch[idx] = 0
				table[i] = idx
				break
			else:
				continue
	matches = []
	for i in range(len(table)):
		matches.append(cv2.DMatch(i, table[i], 1))

	draw_params = dict(matchColor = (255,0,0), singlePointColor = (0,0,255), flags = 0)
	matching = cv2.drawMatches(img1, kp1, img2, kp2,matches, None, **draw_params)
	cv2.imwrite("output/B_matching.jpg", matching)


def reduce_SIFT_feature_matching(img1, img2, kp1, descriptor1, kp2, descriptor2):
	ratio = 0.85
	isMatch = [-1] * len(descriptor2)
	table = [-1] * len(descriptor1)

	for i in range(len(descriptor1)):
		distance = []
		min_value = -1
		min_index = -1
		# point1 = np.array((kp1[i].pt[0], kp1[i].pt[1]))
		point1 = descriptor1[i]
		for j in range(len(descriptor2)):
			point2 = descriptor2[j]
			# point2 = np.array((kp2[j].pt[0], kp2[j].pt[1]))
			# d = np.linalg.norm(point1 - point2)
			d = np.sqrt(np.sum(np.square(point1-point2)))
			distance.append((j, d))

		distance = sorted(distance, key=lambda x: x[1])
		idx_1, dist_1 = distance[0]
		idx_2, dist_2 = distance[1]
		if dist_1/dist_2 < ratio and isMatch[idx_1]==-1:
			isMatch[idx_1] = 0
			table[i] = idx_1

	matches = []
	for i in range(len(table)):
		if table[i] != -1:
			matches.append(cv2.DMatch(i, table[i], 1))

	draw_params = dict(matchColor = (0,255,0), singlePointColor = (0,0,255), flags = 0)
	matching = cv2.drawMatches(img1, kp1, img2, kp2,matches, None, **draw_params)
	cv2.imwrite("output/C_matching.jpg", matching)



def main():
    img1 = cv2.imread("1a_notredame.jpg")
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.imread("1b_notredame.jpg")
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # ======== question(a) : SIFT interest point detection =========================================
    kp1, descriptor1, result1 = SIFT_detector(img1, gray1, 'a')
    cv2.imwrite("output/A_1a_keypoints.jpg", result1)
    kp2, descriptor2, result2 = SIFT_detector(img2, gray2, 'b')
    cv2.imwrite("output/A_1b_keypoints.jpg", result2)

    # ======== question(b) : SIFT feature matching =================================================
    SIFT_feature_matching(img1, img2, kp1, descriptor1, kp2, descriptor2)
    reduce_SIFT_feature_matching(img1, img2, kp1, descriptor1, kp2, descriptor2)


if __name__ == '__main__':
    main()
