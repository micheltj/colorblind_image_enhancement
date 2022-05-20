import numpy as np

from project.saliency import static_fine_saliency_core
from project.util import Transformer, VideoReader


class StaticFineSaliency(Transformer):
    def __init__(self, reader: VideoReader, thresholded=True):
        """Uses static fine saliency. transform() returns a thresholded map if thresholded is true, otherwise plain saliency map"""
        self.reader = reader
        self.thresholded = thresholded

    @property
    def provides_type(self) -> str:
        return 'gray8'

    def transform(self) -> (np.array, np.array):
        for image in self.reader.get_images():
            yield self.transform_single(image)

    def transform_single(self, image: np.array):
        saliency_map, thresholded_map = static_fine_saliency_core(image)
        return image, (thresholded_map * 255).astype('uint8') if self.thresholded else (saliency_map * 255).astype('uint8')