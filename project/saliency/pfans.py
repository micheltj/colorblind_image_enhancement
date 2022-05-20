from project.NN import get_predictions
from project.util import Transformer, VideoReader

import numpy as np


class PyramidFeatureAttentionNetwork(Transformer):
    """Wrapper around the imported pyramid feature attention neural network in project.NN"""
    def __init__(self, reader: VideoReader):
        self.reader = reader

    def transform(self) -> (np.array, np.array):
        for image in self.reader.get_images():
            yield image, self.transform_single(image)

    def transform_single(self, image: np.array):
        return (get_predictions(image) * 255).astype('uint8')

    @property
    def provides_type(self) -> str:
        return 'gray8'

    def provides_shape(self):
        return 512, 512
