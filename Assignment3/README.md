# CV HW3 : Image Alignment with RANSAC & Image Segmentation
> student ID: 111062543

## Part1: Image Alignment with RANSAC
* Create folder *ouptut/* under the folder HW3_111062543/1/
* Output images will be saved in HW3_111062543/1/output/
* Run program "HW3-1.py" with command
    ```
    python HW3-1.py
    ```
* The output images named in the following format:
    * A_book1.jpg
        * A: SIFT feature matching for Question(A)
        * book1: SIFT feature matching for image '1-book1.jpg'
    * B_book1.jpg
        * B: RANSAC feature matching for Question(B)
        * book1: RANSAC feature matching for image '1-book1.jpg'
    * C_book1_vector.jpg
        * C: deviation vector result
        * book1_vector: deviation vector result for image '1-book1.jpg'


## Part2: Image Segmentation
* Create folders *ouptut/image/* and *output/masterpiece* under the folder HW3_111062543/2/
* Output images will be saved in HW3_111062543/2/output/image/ and HW3_111062543/2/output/masterpiece respectively
* Run program "HW3-2.py" with command
    ```
    python HW3-2.py
    ```
* The output images named in the following format:
    * A_K4.jpg
        * A: K-Means result for Question(A)
        * K4: K is 4 in K-Means algorithm
        * (with 20 initial guesses)
    * A_K4_iter1.jpg
        * A: K-Means result for Question(A)
        * K4: K is 4 in K-Means algorithm
        * iter1: with only 1 initial guess
    * C_bandwidth3.jpg
        * C: Mean-Shift result (RGB feature space) for Question( C )
        * bandwidth3: with Gaussian kernel bandwidth is 3
    * C_pixel_distributions_before.jpg:
        * C: pixel distributions result for Question( C )
        * before/after: distributions result before/after clustering
    * D_bandwidth3.jpg
        * D: Mean-Shift result (RGC+spatial information) for Question(D)
        * bandwidth3: with Gaussian kernel bandwidth is 3
    * E_bandwidth5.jpg
        * E: Mean-Shift result (RGB feature space) for Question(E)
        * bandwidth3/bandwidth5/bandwidth10: with Gaussian kernel bandwidth is 3/5/10