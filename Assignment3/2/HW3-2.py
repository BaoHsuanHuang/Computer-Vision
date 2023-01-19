import cv2
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import colors
np.random.seed(42)

#################### Load Images ####################
def load_image(filename):
    img = cv2.imread(filename)
    # print("img.shape: ", img.shape)
    # img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixel_values = img.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    return img, pixel_values

def plot_image(img1, img2):
    fig = plt.figure(figsize=(12, 12)) 
    fg1 = fig.add_subplot(1,2,1)
    fg1.imshow(img1)
    fg2 = fig.add_subplot(1,2,2)
    fg2.imshow(img2)
    plt.show()

def show_image(img1, img2):
    h, w = img1.shape[0], img1.shape[1]
    img1 = cv2.resize(img1, (int(w/1.5), int(h/1.5)), interpolation=cv2.INTER_AREA)
    h, w = img2.shape[0], img2.shape[1]
    img2 = cv2.resize(img2, (int(w/1.5), int(h/1.5)), interpolation=cv2.INTER_AREA)

#################### K-Means & K-Means++ ####################
def calculate_distance(x1, x2):
    dist = np.sqrt(np.sum((x1 - x2)**2))
    return dist

def initialization(pixel_values, n_clusters):
    n_pixels = pixel_values.shape[0]
    n_features = pixel_values.shape[1]
    random_1st_idx = np.random.randint(0, n_pixels-1)
    # print("random_1st_idx: ", random_1st_idx)

    centers = []
    centers.append(pixel_values[random_1st_idx])

    for cluster_idx in range(n_clusters-1):
        distances = []
        for j in range(n_pixels):
            point = pixel_values[j]
            min_dist = 1e9

            for c in range(len(centers)):
                tmp_dist = calculate_distance(point, centers[c])
                min_dist = min(tmp_dist, min_dist)
            distances.append(min_dist)
        distances = np.array(distances)
        next_center_idx = np.argmax(distances)
        next_center = pixel_values[next_center_idx]
        centers.append(next_center)
    return centers

class K_Means_algorithm():
    def __init__(self, K=5, n_iters=20):
        self.K = K
        self.n_iters = n_iters
        self.clusters = [ [] for i in range(K) ]
        self.centers = []
    def predict_label(self, pixel_values):
        self.pixel_values = pixel_values
        self.n_pixels = pixel_values.shape[0]
        self.n_features = pixel_values.shape[1]

        # initialize K centers (K-Means)
        init_pixel_idxs = np.random.choice(self.n_pixels, self.K, replace=False)
        self.centers = [self.pixel_values[idx] for idx in init_pixel_idxs]
        # initialize K centers (K-Means_++)
        # self.centers = initialization(self.pixel_values, self.K)

        # update K centers
        self.update_centers(self.centers)
        return self.get_predicted_cluster_labels(self.clusters)
    def predict_label_PlusPlus(self, pixel_values):
        self.pixel_values = pixel_values
        self.n_pixels = pixel_values.shape[0]
        self.n_features = pixel_values.shape[1]

        # initialize K centers (K-Means_++)
        self.centers = initialization(self.pixel_values, self.K)

        # update K centers
        self.update_centers(self.centers)
        return self.get_predicted_cluster_labels(self.clusters)
    def update_centers(self, centers):
        for i in range(self.n_iters):
            # construct clusters
            self.clusters = [[] for j in range(self.K)]
            for idx, pixel in enumerate(self.pixel_values):
                center_idx = self.find_closest_center(pixel, centers)
                self.clusters[center_idx].append(idx)
            original_centers = self.centers
            self.centers = self.get_new_centers(self.clusters)

            # check whether center moves or not
            all_centers_dists = [calculate_distance(original_centers[i], self.centers[i]) for i in range(self.K)]
            if sum(all_centers_dists) == 0:
                break
    def find_closest_center(self, pixel, centers):
        dists = []
        for i in range(len(centers)):
            c = centers[i]
            d = calculate_distance(pixel, c)
            dists.append(d)
        cloest_center_idx = np.argmin(dists)
        return cloest_center_idx
    def get_new_centers(self, clusters):
        new_centers = np.zeros((self.K, self.n_features))
        for idx, cluster in enumerate(clusters):
            new_mean = np.mean(self.pixel_values[cluster], axis=0)
            new_centers[idx] = new_mean
        return new_centers
    def get_predicted_cluster_labels(self, clusters):
        labels = np.zeros(self.n_pixels)
        for idx, cluster in enumerate(clusters):
            for i in cluster:
                labels[i] = idx
        return labels
    def show_all_centers(self):
        return self.centers

def run_KMeans(img, pixel_values, k):
    if k==6:
        iters = 100
    else:
        iters = 60
    objectK = K_Means_algorithm(K = k, n_iters = iters)  
    predicted_cluster_label = objectK.predict_label(pixel_values)
    centers = np.uint8(objectK.show_all_centers())
    predicted_cluster_label = predicted_cluster_label.astype(int)
    labels = predicted_cluster_label.flatten()
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(img.shape)

    ##### calculate error #####
    dist = 0
    for idx in range(len(pixel_values)):
        pt = pixel_values[idx]
        cluster_idx = predicted_cluster_label[idx]
        center = centers[cluster_idx]
        dist += ((pt[0]-center[0])**2) + (pt[1]-center[1])**2 + (pt[2]-center[2])**2
    error = int(np.sqrt(dist))
    return segmented_image, error

def run_KMeans_plus(img, pixel_values, k):
    if k==6:
        iters = 100
    else:
        iters = 60
    objectK = K_Means_algorithm(K = k, n_iters = iters)  
    predicted_cluster_label = objectK.predict_label_PlusPlus(pixel_values)
    centers = np.uint8(objectK.show_all_centers())
    predicted_cluster_label = predicted_cluster_label.astype(int)
    labels = predicted_cluster_label.flatten()
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(img.shape)
    return segmented_image


def load_image_Resize(filename, flag):
    img = mpimg.imread(filename)
    h, w = img.shape[0], img.shape[1]
    if flag==1:
        img = cv2.resize(img, (528, 300), interpolation=cv2.INTER_AREA)
        # img = cv2.resize(img, (64, 36), interpolation=cv2.INTER_AREA)
    else:
        img = cv2.resize(img, (500, 375), interpolation=cv2.INTER_AREA)
        # img = cv2.resize(img, (64, 48), interpolation=cv2.INTER_AREA)
    # print("img.shape: ", img.shape)
    # img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixel_values = img.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    return img, pixel_values

def euclidean_dist(pointA, pointB):
    total = float(0)
    for d in range(len(pointA)):
        total += (pointA[d] - pointB[d])**2
    dist = math.sqrt(total)
    return dist

def gaussian_kernel(distance, bandwidth):
    euclidean_distance = np.sqrt(((distance)**2).sum(axis=1))
    val = (1/(bandwidth*math.sqrt(2*math.pi))) * np.exp(-0.5*((euclidean_distance / bandwidth))**2)
    return val


# MIN_DISTANCE = 0.000001
# GROUP_DISTANCE_THRESHOLD = 0.1

# MIN_DISTANCE = 0.001
GROUP_DISTANCE_THRESHOLD = 20
MIN_DISTANCE = 0.01
wSize = 10

class PointGrouper(object):
    def group_points(self, points):
        group_assignment = []
        groups = []
        group_index = 0
        for i in range(0, len(points)):
            point = points[i]
            nearest_group_index = self._determine_nearest_group(point, groups)
            if nearest_group_index == -1:
                groups.append([point])
                group_assignment.append(group_index)
                group_index += 1
            else:
                group_assignment.append(nearest_group_index)
                groups[nearest_group_index].append(point)
        group_assignment = np.array(group_assignment)
        return group_assignment

    def _determine_nearest_group(self, point, groups):
        nearest_group_index, index = -1, 0
        for group in groups:
            dist = self._distance_to_group(point, group)
            # print("dist: ", distance_to_group)
            if GROUP_DISTANCE_THRESHOLD > dist:
                nearest_group_index = index
            index += 1
        return nearest_group_index

    def _distance_to_group(self, point, group):
        min_dist = 1e20
        for i in range(len(group)):
            pt = group[i]
            dist = euclidean_dist(point, pt)
            if min_dist > dist:
                min_dist = dist
        return min_dist


class MeanShift(object):
    def __init__(self, kernel=gaussian_kernel):
        self.kernel = kernel

    def cluster(self, points, kernel_bandwidth):
        points = np.array([[float(v) for v in point] for point in points])
        shift_points = np.array(points)
        max_min_dist = 1
        n_iters = 0
        n_points = points.shape[0]

        still_shifting = [True] * n_points
        # for iter in range(1):
            # print("iter: ", iter)
        while max_min_dist > MIN_DISTANCE:
            n_iters += 1
            # print("[MeanShift.cluster] n_iters: ", n_iters)
            # print max_min_dist
            max_min_dist = 0
            for i in range(len(shift_points)):
                # print("  current pixel id: ", i)
                # if not still_shifting[i]:
                #     continue
                if still_shifting[i] == True:
                    new_pt = shift_points[i]
                    new_pt_start = new_pt
                    new_pt = self._shift_point(new_pt, points, kernel_bandwidth)
                    dist = euclidean_dist(new_pt, new_pt_start)
                    # print("[MeanShift.cluster] dist: ", dist)
                    if dist < MIN_DISTANCE:
                        still_shifting[i] = False
                    if dist > max_min_dist:
                        max_min_dist = dist
                    shift_points[i] = new_pt
        point_grouper = PointGrouper()
        group_assignments = point_grouper.group_points(shift_points.tolist())
        return MeanShiftResult(group_assignments)

    def _shift_point(self, point, points, kernel_bandwidth):
        points = np.array(points)

        point_weights = self.kernel(point-points, kernel_bandwidth)
        tiled_weights = np.tile(point_weights, [len(point), 1])

        denominator = sum(point_weights)
        shifted_point = np.multiply(tiled_weights.transpose(), points).sum(axis=0) / denominator
        return shifted_point

class MeanShiftResult:
    def __init__(self, cluster_ids):
        self.cluster_ids = cluster_ids

#################### Mean-Shift (RGB) ####################
def Mean_Shift_algorithm_RGB(pixels):
    mean_shifter = MeanShift()
    mean_shift_result = mean_shifter.cluster(pixels, kernel_bandwidth=3)

    labels = mean_shift_result.cluster_ids
    lables_unique = np.unique(labels)
    n_labels = len(lables_unique)
    print("n_labels: ", n_labels)

    centers = [[0, 0, 0] for i in range(n_labels)]
    cnts = [0 for i in range(n_labels)]
    for i in range(len(pixels)):
        label = mean_shift_result.cluster_ids[i]
        rgb = pixels[i].astype(int)
        centers[label] += rgb
        cnts[label] += 1
    for i in range(n_labels):
        centers[i][0] /= cnts[i]
        centers[i][1] /= cnts[i]
        centers[i][2] /= cnts[i]

    seg_img = []
    for i in range(len(pixels)):
        label = mean_shift_result.cluster_ids[i]
        seg_img.append(centers[label])
    seg_img = np.array(seg_img)
    # print("seg_img.shape: ", seg_img.shape)
    # seg_img = seg_img.reshape((90, 160, 3))
    return seg_img

def Mean_Shift_algorithm_RGBXY(pixels):
    mean_shifter = MeanShift()
    mean_shift_result = mean_shifter.cluster(pixels, kernel_bandwidth=3)

    labels = mean_shift_result.cluster_ids
    lables_unique = np.unique(labels)
    n_labels = len(lables_unique)
    print("n_labels: ", n_labels)

    centers = [[0, 0, 0] for i in range(n_labels)]
    cnts = [0 for i in range(n_labels)]
    for i in range(len(pixels)):
        label = mean_shift_result.cluster_ids[i]
        rgbxy = pixels[i].astype(int)
        centers[label] += rgbxy[0:3]
        cnts[label] += 1
    for i in range(n_labels):
        centers[i][0] /= cnts[i]
        centers[i][1] /= cnts[i]
        centers[i][2] /= cnts[i]

    seg_img = []
    for i in range(len(pixels)):
        label = mean_shift_result.cluster_ids[i]
        seg_img.append(centers[label])
    seg_img = np.array(seg_img)
    # print("seg_img.shape: ", seg_img.shape)
    # seg_img = seg_img.reshape((90, 160, 3))
    return seg_img

def plot_pixel_distributions_before(img, flag):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r, g, b = cv2.split(img)
    pixel_colors_black = np.zeros((img.shape[0] * img.shape[1], 3))

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(r.flatten(), g.flatten(), b.flatten(), facecolor=pixel_colors_black, marker=".")
    ax.set_xlabel("Red")
    ax.set_ylabel("Green")
    ax.set_zlabel("Blue")
    plt.title("pixel distributions [Before]")
    # plt.show()
    if flag==1:
        fig.savefig("output/image/C_pixel_distributions_before.jpg")
    else:
        fig.savefig("output/masterpiece/C_pixel_distributions_before.jpg")
    return True

def plot_pixel_distributions_after(img, clusters, flag):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r, g, b = cv2.split(img)
    
    pixel_colors = clusters.reshape((np.shape(clusters)[0]*np.shape(clusters)[1], 3))
    norm = colors.Normalize(vmin=-1.0,vmax=1.0)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(r.flatten(), g.flatten(), b.flatten(), facecolor=pixel_colors, marker=".")
    ax.set_xlabel("Red")
    ax.set_ylabel("Green")
    ax.set_zlabel("Blue")
    plt.title("pixel distributions [After]")
    # plt.show()
    if flag==1:
        fig.savefig("output/image/C_pixel_distributions_after.jpg")
    else:
        fig.savefig("output/masterpiece/C_pixel_distributions_after.jpg")
    return True

def recoverImage(pixels, total, flag):
    recover = []
    for i in range(len(pixels)):
        rgb = pixels[i]
        last = i*wSize + wSize
        if last > total:
            num = total - i*wSize
            for i in range(num):
                recover.append(rgb)
        else:
            for j in range(wSize):
                recover.append(rgb)
    recover = np.array(recover)
    # print("recover.shape: ", recover.shape)
    if flag==1:
        recover = recover.reshape((300, 528, 3))
        # recover = recover.reshape((36, 64, 3))
    else:
        recover = recover.reshape((375, 500, 3))
        # recover = recover.reshape((48, 64, 3))
    return recover


def A_KMeans():
    img1, pixel_values1 = load_image("2-image.jpg")
    img2, pixel_values2 = load_image("2-masterpiece.jpg")

    n_iters = 20
    KList = [4, 5, 6]
    # n_iters = 1
    # KList = [4]
    for K in KList:
        print("==== (K-Means) K is {} ====".format(K))
        min_error1, min_error2 = 1e9, 1e9
        for i in range(n_iters):
            seg_img1, error1 = run_KMeans(img1, pixel_values1, K)
            seg_img2, error2 = run_KMeans(img2, pixel_values2, K)
            if error1 < min_error1:
                min_error1 = error1
                final_seg_img1 = seg_img1
                final_iter1 = i
            if error2 < min_error2:
                min_error2 = error2
                final_seg_img2 = seg_img2
                final_iter2 = i
        # show_image(final_seg_img1, final_seg_img2)
        if K == 4:
            cv2.imwrite('output/image/A_K4.jpg', final_seg_img1)
            cv2.imwrite('output/masterpiece/A_K4.jpg', final_seg_img2)
        elif K == 5:
            cv2.imwrite('output/image/A_K5.jpg', final_seg_img1)
            cv2.imwrite('output/masterpiece/A_K5.jpg', final_seg_img2)
        else:
            cv2.imwrite('output/image/A_K6.jpg', final_seg_img1)
            cv2.imwrite('output/masterpiece/A_K6.jpg', final_seg_img2)
    return True

def B_KMeans_plus():
    img1, pixel_values1 = load_image("2-image.jpg")
    img2, pixel_values2 = load_image("2-masterpiece.jpg")

    KList = [4, 5, 6]
    # KList = [4]
    for K in KList:
        print("\n==== (K-Means++) K is {} ====".format(K))
        seg_img1= run_KMeans_plus(img1, pixel_values1, K)
        seg_img2= run_KMeans_plus(img2, pixel_values2, K)
        # show_image(seg_img1, seg_img2)
        if K == 4:
            cv2.imwrite('output/image/B_K4.jpg', seg_img1)
            cv2.imwrite('output/masterpiece/B_K4.jpg', seg_img2)
        elif K == 5:
            cv2.imwrite('output/image/B_K5.jpg', seg_img1)
            cv2.imwrite('output/masterpiece/B_K5.jpg', seg_img2)
        else:
            cv2.imwrite('output/image/B_K6.jpg', seg_img1)
            cv2.imwrite('output/masterpiece/B_K6.jpg', seg_img2)
    return True


def run_Image1():
    img, pixel_values = load_image_Resize("2-image.jpg", 1)
    origin = pixel_values
    pixel_values = []
    for i in range(0, len(origin), wSize):
        pixel_values.append(origin[i])
    seg_RGB = Mean_Shift_algorithm_RGB(pixel_values)
    seg_RGB = recoverImage(seg_RGB, len(origin), 1)
    plt.imsave('output/image/C_bandwidth3.jpg', seg_RGB.astype(np.uint8))
    plot_pixel_distributions_before(img, 1)
    plot_pixel_distributions_after(img, seg_RGB, 1)
    return True

def run_Image2():
    img, pixel_values = load_image_Resize("2-masterpiece.jpg", 2)
    origin = pixel_values
    pixel_values = []
    for i in range(0, len(origin), wSize):
        pixel_values.append(origin[i])
    seg_RGB = Mean_Shift_algorithm_RGB(pixel_values)
    seg_RGB = recoverImage(seg_RGB, len(origin), 2)
    plt.imsave('output/masterpiece/C_bandwidth3.jpg', seg_RGB.astype(np.uint8))
    plot_pixel_distributions_before(img, 2)
    plot_pixel_distributions_after(img, seg_RGB, 2)
    return True


def C_MeanShift():
    run_Image1()  #[Bao]
    run_Image2()   #[Bao]
    return True

def D_MeanShift_Image1():
    img, pixel_values = load_image_Resize("2-image.jpg", 1)
    origin = pixel_values
    pixel_values = []
    for i in range(0, len(origin), wSize):
        r, g, b = origin[i][0], origin[i][1], origin[i][2]
        x, y = int(i/img.shape[1]), int(i%img.shape[1])
        feature = np.array([r, g, b, x, y])
        pixel_values.append(feature)
    seg_RGB = Mean_Shift_algorithm_RGBXY(pixel_values)
    seg_RGB = recoverImage(seg_RGB, len(origin), 1)
    plt.imsave('output/image/D_bandwidth3.jpg', seg_RGB.astype(np.uint8))
    return True

def D_MeanShift_Image2():
    img, pixel_values = load_image_Resize("2-masterpiece.jpg", 2)
    origin = pixel_values
    pixel_values = []
    for i in range(0, len(origin), wSize):
        r, g, b = origin[i][0], origin[i][1], origin[i][2]
        x, y = int(i/img.shape[1]), int(i%img.shape[1])
        feature = np.array([r, g, b, x, y])
        pixel_values.append(feature)
    seg_RGB = Mean_Shift_algorithm_RGBXY(pixel_values)
    seg_RGB = recoverImage(seg_RGB, len(origin), 2)
    plt.imsave('output/masterpiece/D_bandwidth3.jpg', seg_RGB.astype(np.uint8))
    return True

def Mean_Shift_algorithm_RGB_bandwidth(pixels, bandwidth):
    mean_shifter = MeanShift()
    mean_shift_result = mean_shifter.cluster(pixels, kernel_bandwidth=bandwidth)

    labels = mean_shift_result.cluster_ids
    lables_unique = np.unique(labels)
    n_labels = len(lables_unique)
    print("n_labels: ", n_labels)

    centers = [[0, 0, 0] for i in range(n_labels)]
    cnts = [0 for i in range(n_labels)]
    for i in range(len(pixels)):
        label = mean_shift_result.cluster_ids[i]
        rgb = pixels[i].astype(int)
        centers[label] += rgb
        cnts[label] += 1
    for i in range(n_labels):
        centers[i][0] /= cnts[i]
        centers[i][1] /= cnts[i]
        centers[i][2] /= cnts[i]

    seg_img = []
    for i in range(len(pixels)):
        label = mean_shift_result.cluster_ids[i]
        seg_img.append(centers[label])
    seg_img = np.array(seg_img)
    # print("seg_img.shape: ", seg_img.shape)
    # seg_img = seg_img.reshape((90, 160, 3))
    return seg_img


def E_image_bandwidth(bandwidth):
    img, pixel_values = load_image_Resize("2-image.jpg", 1)
    origin = pixel_values
    pixel_values = []
    for i in range(0, len(origin), wSize):
        pixel_values.append(origin[i])
    seg_RGB = Mean_Shift_algorithm_RGB_bandwidth(pixel_values, bandwidth)
    seg_RGB = recoverImage(seg_RGB, len(origin), 1)
    if bandwidth == 3:
        plt.imsave('output/image/E_bandwidth3.jpg', seg_RGB.astype(np.uint8))
    if bandwidth == 5:
        plt.imsave('output/image/E_bandwidth5.jpg', seg_RGB.astype(np.uint8))
    if bandwidth == 10:
        plt.imsave('output/image/E_bandwidth10.jpg', seg_RGB.astype(np.uint8))
    return True


def E_masterpiece_bandwidth(bandwidth):
    start = time.time()

    img, pixel_values = load_image_Resize("2-masterpiece.jpg", 2)
    print("img1.shape: ", img.shape)
    print("# of pixels: ", len(pixel_values))
    origin = pixel_values
    pixel_values = []
    for i in range(0, len(origin), wSize):
        pixel_values.append(origin[i])
    print("# of pixels: ", len(pixel_values))
    seg_RGB = Mean_Shift_algorithm_RGB_bandwidth(pixel_values, bandwidth)
    seg_RGB = recoverImage(seg_RGB, len(origin), 2)
    print("seg_RGB.shape: ", seg_RGB.shape)
    if bandwidth == 3:
        plt.imsave('output/masterpiece/E_bandwidth3.jpg', seg_RGB.astype(np.uint8))
    if bandwidth==5:
        plt.imsave('output/masterpiece/E_bandwidth5.jpg', seg_RGB.astype(np.uint8))
    if bandwidth==10:
        plt.imsave('output/masterpiece/E_bandwidth10.jpg', seg_RGB.astype(np.uint8))

    end = time.time()
    print("\nexecution time: ", int(end-start)/60)
    return True

def main():
    A_KMeans() 
    B_KMeans_plus()
    C_MeanShift() 
    D_MeanShift_Image1() 
    D_MeanShift_Image2()
    E_image_bandwidth(3)
    E_image_bandwidth(5)
    E_image_bandwidth(10)
    E_masterpiece_bandwidth(3)
    E_masterpiece_bandwidth(5)
    E_masterpiece_bandwidth(10)


if __name__ == '__main__':
    main()
