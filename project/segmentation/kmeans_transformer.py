import cv2 as cv
import numpy as np

from project.util import Transformer, VideoReader


class KMeansTransformer(Transformer):
    mode_normal = 'normal'
    mode_saliency_region_only = 'saliency_region_only'

    def __init__(self, reader: VideoReader, clusters=6, mode=mode_normal, saliency_transformer: Transformer = None,
                 iterations=10, saliency_only_threshold=100,
                 intermediate_conversion_format=None, final_conversion_format=None):
        """
        @param reader: The input reader to use
        @param clusters: The amount of color clusters / centers to find
        @param mode: Which KMeansTransformer.mode_* to use
        @param saliency_transformer: The saliency method / transformer to use if mode is not set to 'normal'
        @param iterations: Maximum amount of iterations to find cluster centers
        @param saliency_only_threshold: Threshold for saliency only mode
        @param intermediate_conversion_format: The image format / color channels to use for clustering (e. g. cv.BGR2HSV)
        @param final_conversion_format: The back conversion format to use for clustering if intermediate_conversion_format is set (e. g. cv.HSV2BGR)
        """
        self.reader = reader
        self.clusters = clusters
        self.previous_centers = np.array([])
        self.saliency_only_threshold = saliency_only_threshold
        self.saliency_transformer = None
        if mode == self.mode_normal:
            self.kmeans = self.kmeans_normal
        elif mode == self.mode_saliency_region_only:
            self.kmeans = self.kmeans_saliency_only
            self.transformer = saliency_transformer
        self.intermediate_conversion_format = intermediate_conversion_format
        self.final_conversion_format = final_conversion_format
        self.iterations = iterations

    def transform(self) -> (np.array, np.array):
        for frame in self.reader.get_images():
            # Apply channel conversion only if requested
            converted_frame = frame if self.intermediate_conversion_format is None else cv.cvtColor(frame, self.intermediate_conversion_format)
            yield frame, cv.cvtColor(self.kmeans(converted_frame), cv.COLOR_BGR2RGB)

    @property
    def provides_type(self) -> str:
        return 'bgr888'

    def provides_shape(self):
        # Underlying transformer may return a different shape than the input video supplies.
        if hasattr(self.transformer, 'provides_shape'):
            return self.transformer.provides_shape()
        else:
            return self.reader.width, self.reader.height

    def kmeans_normal(self, image: np.array):
        img = image.reshape((-1, 3)).astype('float32')
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, self.iterations, 1.0)
        # Find centers and label the image
        ret, labels, centers = cv.kmeans(img, self.clusters, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        self.previous_centers = centers
        # Assign the centers to the labeled image / convert to segmented image
        clustered = centers[labels.flatten()]
        clustered = clustered.reshape(image.shape)
        if self.final_conversion_format is not None:
            clustered = cv.cvtColor(clustered, self.final_conversion_format)
        return clustered

    def kmeans_saliency_only(self, image: np.array):
        """ Uses saliency masks only for kmeans training and applies centers to the original image """
        original, salienced = self.transformer.transform_single(image)
        # mask the image and train kmeans only on saliency values
        img = image[salienced > self.saliency_only_threshold].reshape((-1, 3)).astype('float32')
        if len(img) < self.clusters:
            img = image
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, labels, centers = cv.kmeans(img, self.clusters, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
        center_int = np.int32(centers)
        center_uint = np.uint8(centers)
        # apply saliency centers to the original image
        predicted = np.apply_along_axis(
            lambda x: center_uint[np.argmin(np.sum(np.square(center_int - x), axis=1))], 2, image)
        if self.final_conversion_format is not None:
            predicted = cv.cvtColor(predicted, self.final_conversion_format)
        return predicted



