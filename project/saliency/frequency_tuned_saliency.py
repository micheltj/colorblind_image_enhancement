import cv2 as cv
import numpy as np
import math
from sklearn.cluster import MeanShift
from sklearn.cluster import  estimate_bandwidth

from project.util import VideoReader


class FrequencyTunedSalientRegionDetection:
    
    def __init__(self, reader: VideoReader):
        self.reader = reader

    def transform(self):
        for image in self.reader.get_images():
            yield image, self.get_dog_saliency(image)

    def transform_single(self, image):
        return image, self.get_dog_saliency(image)

    @property
    def provides_type(self):
        return 'gray8'

    # Mean shift clustering 
    def mean_shift(self, X):
        bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
        msc = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        msc.fit(X)
        cluster_centers = msc.cluster_centers_
        labels = msc.labels_
        cluster_label = np.unique(labels)
        n_clusters = len(cluster_label)
        return labels

    def get_dog_saliency(self, img):
        if np.max(img) == 0:
            # Algorithm doesn't support 0-arrays/black images, return accordingly shaped 0-array
            return np.repeat(np.uint64(0), img.shape[0] * img.shape[1]).reshape(img.shape[0], -1)
        # Convert to Lab color space, where perceived distance corresponds to euclidean distance
        lab_img = cv.cvtColor(img, cv.COLOR_BGR2LAB)
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        sigma_l = cv.GaussianBlur(lab_img, (5,5), sigmaX=1.6*(np.pi/2.75))
        sigma_h = cv.GaussianBlur(lab_img, (5,5), sigmaX=(np.pi/2.75))

        # Calculate Difference of Gaussian
        dog = sigma_l - sigma_h
        saliency_map = np.zeros(gray_img.shape, dtype="float32")

        # Mean pixel values of corresponding channels of Lab image
        img_mean_l = np.average(lab_img[:,:,0])
        img_mean_a = np.average(lab_img[:,:,1])
        img_mean_b2 = np.average(lab_img[:,:,2])
        
        # Compute Saliency: S(x,y) = I_mu - I_sigmah(x,y)
        for row in range(dog.shape[0]):
            for col in range(dog.shape[1]):
                #saliency_val = np.abs(img_mean - dog[row, col])
                # Saliency using LAB color space and L2 norm
                saliency_val = math.sqrt((img_mean_l - dog[row, col, 0])**2 + (img_mean_a - dog[row, col, 1])**2 + (img_mean_b2 - dog[row, col, 2])**2)
                saliency_map[row, col] = saliency_val
        sal_normalized = saliency_map

        # Do Mean Shift clustering also in Lab
        flatImg=np.reshape(sal_normalized, [-1, 1])
        labels = self.mean_shift(flatImg)

        segmentedImg = np.reshape(labels, sal_normalized.shape[:2])

        # Adaptive Thresholding
        thresh = (2/(segmentedImg.shape[0]*segmentedImg.shape[1])) * np.average(segmentedImg)

        for row in range(segmentedImg.shape[0]):
            for col in range(segmentedImg.shape[1]):
                val = segmentedImg[row, col]
                if val > thresh:
                    segmentedImg[row, col] = 255
                else:
                    segmentedImg[row, col] = 0

        segmentedImg = cv.medianBlur(segmentedImg.astype('uint8'), 5)

        return segmentedImg
