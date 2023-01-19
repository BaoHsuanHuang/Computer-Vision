# CV HW1 : Image Features

## Part1 : Harris Corner Detection
>  `Implement in HW1-1.py`
>  `Run this file directly, then can get the results.`
### Step1 : Gaussian Smooth
* Construct two Gaussian kernels with different kernel sizes, the kernel sizes are 5 and 10 respectively. After that, using Gaussian kernel to smooth the images and obtain the blurred images. 
* Function: *gaussian_blur(image, ksize, sigma)*
* Blurred images are named as '1a_a_k5.jpg', '1a_a_k10.jpg', 'chessboard_a_k5.jpg', and 'chessboard_a_k10.jpg'.
    * 1a: represents the image '1a_notredame.jpg'
    * a: represents for the question 'a' in Part1-A
    * k5: represents that Gaussian kernel size = 5

### Step2 : Intensity Gradient (Sobel edge detection)
* Compute the magnitude and direction/orientation of gradient by using Sobel edge detection algorithm.
* Apply horizontal filter and vertical filter first. Calculate the gradient magnitude and the gradient direction value of each pixel.
* Draw the gradient direction of the image in HSV form to visualize the edges
* Function: *sobel_edge_detection(image, filterx, filtery)*
* Named as '1a_b_k5_dir.jpg', '1a_b_k5_mag.jpg'...
    * 1a: represents the image '1a_notredame.jpg'
    * b: represents for the question 'b' in Part1-A
    * k5: represents that the input image is filter with the Gaussian kernel size=5
    * dir, mag: represents 'direction' and 'magnitude' respectively

### Step3 : Structure Tensor
* Compute the structure tensor 𝐻 of each pixel and show the image with different window size.
* Use the formula mentioned in the lecture, calculating *Ixx*, *Ixy*, *Iyy* first, then calculate *determinant value* and *trace value* with 2 different window size, three and five respectively.
* Calculate response value 'R' of each value.
* Function: *structure_tensor(img, filterx, filtery, ratio, wsize)*
* Named as '1a_c_k10_w3.jpg', '1a_c_k10_w5.jpg' ...
    * 1a: represents the image '1a_notredame.jpg'
    * c: represents for the question 'c' in Part1-A
    * w3: represnets that window size = 3

### Step4 : Non-maximal Suppression
* Perform Non-maximal Suppression and get the corner detection result.
* Find all pixels that has response value greater than a threshold and are the local maximum with a window size.
* Function: *non_max_suppression(img, dataR, window, maxR, ratio, rrr, mask1)*
* Named as '1a_d_w3.jpg', '1a_d_w5.jpg', 'chessboard_d_w3.jpg', and 'chessboard_d_w5.jpg'
    * 1a: represents the image '1a_notredame.jpg'
    * d: represents for the question 'd' in Part1-A
    * w3: represnets that window size = 3


## Part2 : SIFT interest point detection and matching
>  `Implement in HW1-2.py`
>  `Run this file directly, then can get the results.`
### Step1 : SIFT interest point detection
* Apply *cv2.xfeatures2d.SIFT_create.detectAndCompute()* to detect around 100 points on 2 images.
* Plot the detected feature keypoints with red dot.
* Function: *SIFT_detector(img, gray, id)*
* Named as 'A_1a_keypoints.jpg' and 'A_1b_keypoints.jpg'

### Step2 : SIFT feature matching
* With the keypoints obtained in Step1, calculating the distance of all the keypoint pairs(feature vectors pairs) in the two images.
* Use nearest-neighbor matching algorithm, for each keypoint in image1, to match with another keypoint(in image2) which has the shortest distance value.
* Function: *SIFT_feature_matching(img1, img2, kp1, descriptor1, kp2, descriptor2)*
* Named as 'B_matching.jpg'

### Step3: Reduce the mis-matches
* Calculate the ratio of the two shortest distances, instead of only using the shortest distance for matching.
* Function: *reduce_SIFT_featurematching(img1, img2, kp1, descriptor1, kp2, descriptor2)*
* Named as 'C_matching.jpg'