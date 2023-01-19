import numpy as np
import cv2
import math
from scipy import signal

# def rotate_img(img):
#     cols = img.shape[1]
#     rows = img.shape[0]
#     M = cv2.getRotationMatrix2D((cols/2,rows/2), 30, 1) 
#     img_rotate = cv2.warpAffine(img, M, (cols,rows)) 
#     return img_rotate

# def scale_img(img):
#     scale_ratio = 0.5
#     width = int(img.shape[1] * scale_ratio)
#     height = int(img.shape[0] * scale_ratio)
#     img_scale = cv2.resize(img, (width, height))
#     return img_scale

def rotate_scale_img(img):
    cols = img.shape[1]
    rows = img.shape[0]
    M = cv2.getRotationMatrix2D((cols/2,rows/2), 30, 1) 
    img_rotate = cv2.warpAffine(img, M, (cols,rows)) 

    scale_ratio = 0.5
    width = int(img.shape[1] * scale_ratio)
    height = int(img.shape[0] * scale_ratio)
    img_scale = cv2.resize(img_rotate, (width, height))
    return img_scale

# ======== question(a) : Gaussian Smooth =====================================================================
def convolution(img, kernel):
    img_row = img.shape[0]
    img_col = img.shape[1]
    kernel_row = kernel.shape[0]
    kernel_col = kernel.shape[1]
 
    output = np.zeros(img.shape)
 
    if kernel_row%2 == 1:
        pad_height = int((kernel_row - 1) / 2)
        pad_width = int((kernel_col - 1) / 2)
    else:
        pad_height = int(kernel_row / 2)
        pad_width = int(kernel_col / 2)
 
    padded_img = np.zeros((img_row + (2 * pad_height), img_col + (2 * pad_width)))
 
    padded_img[pad_height:padded_img.shape[0] - pad_height, pad_width:padded_img.shape[1] - pad_width] = img

    for row in range(img_row):
        for col in range(img_col):
            output[row, col] = np.sum(kernel * padded_img[row:row + kernel_row, col:col + kernel_col])
            output[row, col] /= kernel.shape[0] * kernel.shape[1]
 
    return output

def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)
def gaussian_kernel(ksize, sigma):
    # kernel_1D = np.linspace(-(ksize // 2), ksize // 2, ksize)
    # for i in range(ksize):
    #     kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    # kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
    # kernel_2D *= 1.0 / kernel_2D.max()
    # return kernel_2D
    kernel = np.zeros((ksize, ksize), np.float64)
    if ksize==10:
        radius = (ksize-1)//2
    else:
        radius = ksize//2
    for y in range(-radius, radius + 1):  # [-r, r]
        for x in range(-radius, radius + 1):
            v = 1.0 / (2 * np.pi * sigma ** 2) * np.exp(-1.0 / (2 * sigma ** 2) * (x ** 2 + y ** 2))
            kernel[y + radius, x + radius] = v
    kernel2 = kernel / round(np.sum(kernel), 3)
    return kernel2  

def gaussian_blur(image, ksize, sigma):
    kernel = gaussian_kernel(ksize, sigma)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blurred = convolution(gray, kernel)
    blurred = signal.convolve2d(gray, kernel, mode="same", boundary="symm")
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred

# ======== question(b) : Sobel Edge Detection ===============================================================
def sobel_edge_detection(image, filter_x, filter_y):
    new_image_x = convolution(image, filter_x)
    new_image_y = convolution(image, filter_y)

    # if verbose:
    #     plt.imshow(new_image_x, cmap='gray')
    #     plt.title("Horizontal Edge")
    #     plt.show()
    # if verbose:
    #     plt.imshow(new_image_y, cmap='gray')
    #     plt.title("Vertical Edge")
    #     plt.show()
    height = image.shape[0]
    width =image.shape[1]

    image_mag = np.zeros((height,width),np.uint8)
    image_dir = np.zeros((height,width),np.uint8)
    hsv = np.zeros((height, width, 3),np.uint8)

    blue = np.array([255, 0, 0])
    yellow = np.array([0, 255, 255])
    red = np.array([0, 0, 255])
    cyan = np.array([255, 255, 0])
    black = np.array([0, 0, 0])

    for i in range(height):
        for j in range(width):
            gY = new_image_y[i][j]
            gX = new_image_x[i][j]
            mag = np.sqrt(gX ** 2 + gY ** 2)
            image_mag[i][j] = mag

            dir = np.arctan2(gY, gX) * (180/np.pi)

            if mag>=10.0:
                image_dir[i][j] = dir
                if dir <= 180.0 and dir > 90.0:
                    hsv[i, j, :] = red
                elif dir <= 90.0 and dir > 0.0:
                    hsv[i, j, :] = yellow
                elif dir == 0.0:
                    hsv[i, j, :] = cyan
                elif dir < 0.0 and dir > -90.0:
                    hsv[i, j, :] = cyan
                elif dir <= -90.0 and dir >= -180.0:
                    hsv[i, j, :] = blue
                else:
                    hsv[i, j, :] = black
            else:
                hsv[i, j, :] = black
    return image_mag, hsv

# ======== question(c) : Structure Tensor ==============================================================
def gradient(img, filter_x, filter_y):
    I_x = signal.convolve2d(img, filter_x, mode='same')
    I_y = signal.convolve2d(img, filter_y, mode='same')
    return I_x, I_y

def harris_response_calculate(img, Ix, Iy):
    Ixx = Ix * Ix
    Ixy = Ix * Iy
    Iyy = Iy * Iy
    Ixx = Ixx.astype('float64')
    Ixy = Ixy.astype('float64')
    Iyy = Iyy.astype('float64')

    height = int(img.shape[0] - 13)
    width = int(img.shape[1] - 13)
    k = 0.04
    data = []
    max_R = 0.0

    for i in range(1, int(img.shape[0] - 2)):
        for j in range(1, int(img.shape[1] - 2)):
            # Wx = Ixx[i-4 : i+5 , j-4 : j+5]
            # Wy = Iyy[i-4 : i+5 , j-4 : j+5]
            # Wxy = Ixy[i-4 : i+5 , j-4 : j+5]
            
            # Wx = Ixx[i-2 : i+3 , j-2 : j+3]
            # Wy = Iyy[i-2 : i+3 , j-2 : j+3]
            # Wxy = Ixy[i-2 : i+3 , j-2 : j+3]
            Wx = Ixx[i-1 : i+2 , j-1 : j+2]
            Wy = Iyy[i-1 : i+2 , j-1 : j+2]
            Wxy = Ixy[i-1 : i+2 , j-1 : j+2]
            sum_x = np.sum(Wx)
            sum_y = np.sum(Wy)
            sum_xy = np.sum(Wxy)
            # print((sum_x * sum_y) - (sum_xy * sum_xy))
            detA = (sum_x * sum_y) - (sum_xy * sum_xy)
            traceA = sum_x + sum_y
            R = detA - (k * traceA * traceA)
            data.append((i, j, R))
            if float(R) > float(max_R):
                max_R = R
    return data, max_R

def harris_response_calculate_w5(img, Ix, Iy):
    Ixx = Ix * Ix
    Ixy = Ix * Iy
    Iyy = Iy * Iy
    Ixx = Ixx.astype('float64')
    Ixy = Ixy.astype('float64')
    Iyy = Iyy.astype('float64')

    # height = int(img.shape[0] - 2)
    # width = int(img.shape[1] - 2)
    height = int(img.shape[0] - 14)
    width = int(img.shape[1] - 15)
    k = 0.04
    data = []
    max_R = 0.0

    for i in range(2, int(img.shape[0] - 3)):
        for j in range(2, int(img.shape[1] - 3)):
            # Wx = Ixx[i-12 : i+13 , j-12 : j+13]
            # Wy = Iyy[i-12 : i+13 , j-12 : j+13]
            # Wxy = Ixy[i-12 : i+13 , j-12 : j+13]
            # Wx = Ixx[i-4 : i+5 , j-4 : j+5]
            # Wy = Iyy[i-4 : i+5 , j-4 : j+5]
            # Wxy = Ixy[i-4 : i+5 , j-4 : j+5]
            Wx = Ixx[i-2 : i+3 , j-2 : j+3]
            Wy = Iyy[i-2 : i+3 , j-2 : j+3]
            Wxy = Ixy[i-2 : i+3 , j-2 : j+3]
            sum_x = np.sum(Wx)
            sum_y = np.sum(Wy)
            sum_xy = np.sum(Wxy)
            detA = (sum_x * sum_y) - (sum_xy * sum_xy)
            traceA = sum_x + sum_y
            R = detA - (k * traceA * traceA)
            data.append((i, j, R))
            if R>max_R:
                max_R = R
    return data, max_R

def structure_tensor(img, filter_x, filter_y, ratio, wsize):
    match = []
    Ix, Iy = gradient(img, filter_x, filter_y)
    if wsize==3:
        data_R, max_R = harris_response_calculate(img, Ix, Iy)
    else:
        data_R, max_R = harris_response_calculate_w5(img, Ix, Iy)
    threshold = max_R * ratio

    tmp_array = np.zeros(img.shape)
    rrr = np.zeros((img.shape[0], img.shape[1]))
    mask = np.zeros((img.shape[0], img.shape[1]))

    for instance in data_R:
        i, j, R = instance
        rrr[i][j] = R
        if R > threshold:
            tmp_array[i][j] = 255
            mask[i][j] = 1
    window_img = tmp_array
    return data_R, window_img, max_R, rrr, mask

# ======== question(d) : Non-maximal Suppression =======================================================
def non_max_suppression(img, data_R, window, max_R, ratio, rrr, mask1):
    threshold = max_R * ratio
    detected_img = np.copy(img)
    red = np.array([0, 0, 255])
    mask2 = np.zeros((img.shape[0], img.shape[1]))
    height = img.shape[0]
    width = img.shape[1]

    # for i in range(4, height-5):
    #     for j in range(4, width-5):
    #         box = rrr[i-4 : i+5 , j-4 : j+5]
    #         maximum = np.max(box)
    #         if rrr[i][j] >= maximum and maximum!=0:
    #             mask2[i][j] = 1
    # for i in range(2, height-3):
    #     for j in range(2, width-3):
    #         box = rrr[i-2 : i+3 , j-2 : j+3]
    #         maximum = np.max(box)
    #         if rrr[i][j] >= maximum:
    #             mask2[i][j] = 1
    # for i in range(1, height-2):
    #     for j in range(1, width-2):
    #         box = rrr[i-1 : i+2 , j-1 : j+2]
    #         maximum = np.max(box)
    #         if rrr[i][j] >= maximum:
    #             mask2[i][j] = 1
    # for i in range(len(mask1)):
    #     for j in range(len(mask1[i])):
    #         if mask1[i][j]==1 and mask2[i][j]==1:
    #             detected_img[i, j, :] = red
    for i in range(len(mask1)):
        for j in range(len(mask1[i])):
            if mask1[i][j]==1 and rrr[i][j]>=threshold:
                detected_img[i, j, :] = red
    return detected_img



def Harris_Corner_Detection(img, id):
    filter_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    filter_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    if id==1:
        blur_img_k5 = gaussian_blur(img, 5, 5)
        blur_img_k10 = gaussian_blur(img, 10, 5)
        cv2.imwrite("output/normal/1a_a_k5.jpg", blur_img_k5)
        cv2.imwrite("output/normal/1a_a_k10.jpg", blur_img_k10)

        k5_mag, k5_dir = sobel_edge_detection(blur_img_k5, filter_x, filter_y)
        k10_mag, k10_dir = sobel_edge_detection(blur_img_k10, filter_x, filter_y)
        cv2.imwrite("output/normal/1a_b_k5_mag.jpg", k5_mag)
        cv2.imwrite("output/normal/1a_b_k5_dir.jpg", k5_dir)
        cv2.imwrite("output/normal/1a_b_k10_mag.jpg", k10_mag)
        cv2.imwrite("output/normal/1a_b_k10_dir.jpg", k10_dir)

        match_k10_w3, img_k10_w3, max_R_w3, rrr_w3, mask_w3 = structure_tensor(k10_mag, filter_x, filter_y, 0.00000001, 3)  #0.00001 0.000005 0.000001
        match_k10_w5, img_k10_w5, max_R_w5, rrr_w5, mask_w5 = structure_tensor(k10_mag, filter_x, filter_y, 0.00000001, 5) # 0.00025 0.00015  0.000001
        cv2.imwrite("output/normal/1a_c_k10_w3.jpg", img_k10_w3)
        cv2.imwrite("output/normal/1a_c_k10_w5.jpg", img_k10_w5)

        detected_w3 = non_max_suppression(img, match_k10_w3, img_k10_w3, max_R_w3, 0.0000008, rrr_w3, mask_w3)
        detected_w5 = non_max_suppression(img, match_k10_w5, img_k10_w5, max_R_w5, 0.0000008, rrr_w5, mask_w5)
        cv2.imwrite("output/normal/1a_d_w3.jpg", detected_w3)
        cv2.imwrite("output/normal/1a_d_w5.jpg", detected_w5)

    elif id==2:
        blur_img_k5 = gaussian_blur(img, 5, 5)
        blur_img_k10 = gaussian_blur(img, 10, 5)
        cv2.imwrite("output/normal/chessboard_a_k5.jpg", blur_img_k5)
        cv2.imwrite("output/normal/chessboard_a_k10.jpg", blur_img_k10)

        k5_mag, k5_dir = sobel_edge_detection(blur_img_k5, filter_x, filter_y)
        k10_mag, k10_dir = sobel_edge_detection(blur_img_k10, filter_x, filter_y)
        cv2.imwrite("output/normal/chessboard_b_k5_mag.jpg", k5_mag)
        cv2.imwrite("output/normal/chessboard_b_k5_dir.jpg", k5_dir)
        cv2.imwrite("output/normal/chessboard_b_k10_mag.jpg", k10_mag)
        cv2.imwrite("output/normal/chessboard_b_k10_dir.jpg", k10_dir)

        match_k10_w3, img_k10_w3, max_R_w3, rrr_w3, mask_w3 = structure_tensor(k10_mag, filter_x, filter_y, 0.00005, 3) # 0.0005 0.0001
        match_k10_w5, img_k10_w5, max_R_w5, rrr_w5, mask_w5 = structure_tensor(k10_mag, filter_x, filter_y, 0.00005, 5) # 0.0055 0.0001
        cv2.imwrite("output/normal/chessboard_c_k10_w3.jpg", img_k10_w3)
        cv2.imwrite("output/normal/chessboard_c_k10_w5.jpg", img_k10_w5)

        detected_w3 = non_max_suppression(img, match_k10_w3, img_k10_w3, max_R_w3, 0.0005, rrr_w3, mask_w3) # 0.0008 0.0003
        detected_w5 = non_max_suppression(img, match_k10_w5, img_k10_w5, max_R_w5, 0.0005, rrr_w5, mask_w5) # 0.0058 0.0003
        cv2.imwrite("output/normal/chessboard_d_w3.jpg", detected_w3)
        cv2.imwrite("output/normal/chessboard_d_w5.jpg", detected_w5)

    elif id==3:
        blur_img_k5 = gaussian_blur(img, 5, 5)
        blur_img_k10 = gaussian_blur(img, 10, 5)
        cv2.imwrite("output/transformed/1a_a_k5.jpg", blur_img_k5)
        cv2.imwrite("output/transformed/1a_a_k10.jpg", blur_img_k10)

        k5_mag, k5_dir = sobel_edge_detection(blur_img_k5, filter_x, filter_y)
        k10_mag, k10_dir = sobel_edge_detection(blur_img_k10, filter_x, filter_y)
        cv2.imwrite("output/transformed/1a_b_k5_mag.jpg", k5_mag)
        cv2.imwrite("output/transformed/1a_b_k5_dir.jpg", k5_dir)
        cv2.imwrite("output/transformed/1a_b_k10_mag.jpg", k10_mag)
        cv2.imwrite("output/transformed/1a_b_k10_dir.jpg", k10_dir)

        match_k10_w3, img_k10_w3, max_R_w3, rrr_w3, mask_w3 = structure_tensor(k10_mag, filter_x, filter_y, 0.000001, 3) # 0000001
        match_k10_w5, img_k10_w5, max_R_w5, rrr_w5, mask_w5 = structure_tensor(k10_mag, filter_x, filter_y, 0.000001, 5) # 0000001
        cv2.imwrite("output/transformed/1a_c_k10_w3.jpg", img_k10_w3)
        cv2.imwrite("output/transformed/1a_c_k10_w5.jpg", img_k10_w5)

        detected_w3 = non_max_suppression(img, match_k10_w3, img_k10_w3, max_R_w3, 0.00005, rrr_w3, mask_w3)
        detected_w5 = non_max_suppression(img, match_k10_w5, img_k10_w5, max_R_w5, 0.00005, rrr_w5, mask_w5)
        cv2.imwrite("output/transformed/1a_d_w3.jpg", detected_w3)
        cv2.imwrite("output/transformed/1a_d_w5.jpg", detected_w5)
  
    elif id==4:
        blur_img_k5 = gaussian_blur(img, 5, 5)
        blur_img_k10 = gaussian_blur(img, 10, 5)
        cv2.imwrite("output/transformed/chessboard_a_k5.jpg", blur_img_k5)
        cv2.imwrite("output/transformed/chessboard_a_k10.jpg", blur_img_k10)

        k5_mag, k5_dir = sobel_edge_detection(blur_img_k5, filter_x, filter_y)
        k10_mag, k10_dir = sobel_edge_detection(blur_img_k10, filter_x, filter_y)
        cv2.imwrite("output/transformed/chessboard_b_k5_mag.jpg", k5_mag)
        cv2.imwrite("output/transformed/chessboard_b_k5_dir.jpg", k5_dir)
        cv2.imwrite("output/transformed/chessboard_b_k10_mag.jpg", k10_mag)
        cv2.imwrite("output/transformed/chessboard_b_k10_dir.jpg", k10_dir)

        match_k10_w3, img_k10_w3, max_R_w3, rrr_w3, mask_w3 = structure_tensor(k10_mag, filter_x, filter_y, 0.05, 3) # 0005
        match_k10_w5, img_k10_w5, max_R_w5, rrr_w5, mask_w5 = structure_tensor(k10_mag, filter_x, filter_y, 0.05, 5) # 0005
        cv2.imwrite("output/transformed/chessboard_c_k10_w3.jpg", img_k10_w3)
        cv2.imwrite("output/transformed/chessboard_c_k10_w5.jpg", img_k10_w5)

        detected_w3 = non_max_suppression(img, match_k10_w3, img_k10_w3, max_R_w3, 0.0005, rrr_w3, mask_w3)
        detected_w5 = non_max_suppression(img, match_k10_w5, img_k10_w5, max_R_w5, 0.0005, rrr_w5, mask_w5)
        cv2.imwrite("output/transformed/chessboard_d_w3.jpg", detected_w3)
        cv2.imwrite("output/transformed/chessboard_d_w5.jpg", detected_w5)
  
    else:
        print("Notice: Image is Wrong!")

def Part1_Harris_Corner_Detection():
    img_1 = cv2.imread("1a_notredame.jpg")
    img_2 = cv2.imread("chessboard-hw1.jpg")
    
    # [rotate + scale]
    img_3 = rotate_scale_img(img_1)
    img_4 = rotate_scale_img(img_2)

    # [detect]
    Harris_Corner_Detection(img_1, 1) # [1a]
    Harris_Corner_Detection(img_2, 2) # [chessboard]
    Harris_Corner_Detection(img_3, 3) # [1a + rotate + scale]
    Harris_Corner_Detection(img_4, 4) # [chessboard + rotate + scale]
    # Harris_Corner_Detection(img_5, 5) # [1a + scale]
    # Harris_Corner_Detection(img_6, 6) # [chessboard + scale]


def main():
    Part1_Harris_Corner_Detection()


if __name__ == '__main__':
    main()