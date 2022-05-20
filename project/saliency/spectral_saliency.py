import cv2 as cv

from project.util import VideoReader


class SpectralSaliencyDetection:
    
    def __init__(self, reader: VideoReader):
        self.reader = reader

    def transform(self):
        for image in self.reader.get_images():
            yield image, self.static_spectral_saliency(image)

    def transform_single(self, image):
        return image, self.static_spectral_saliency(image)

    @property
    def provides_type(self):
        return 'gray8'

    # Based on paper: https://www.sciencedirect.com/science/article/abs/pii/S0262885609001371
    def static_spectral_saliency(self, img):
        # https://github.com/ivanred6/image_saliency_opencv
        saliency = cv.saliency.StaticSaliencySpectralResidual_create()
        (success, saliencyMap) = saliency.computeSaliency(img)
        saliencyMap = (saliencyMap * 255).astype("uint8")
        return saliencyMap
