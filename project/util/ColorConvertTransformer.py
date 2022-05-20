import cv2 as cv

from project.util import Transformer


class ColorConvertTransformer(Transformer):
    """Simple transformer wrapper to convert image format on the fly"""
    def __init__(self, transformer, cvt_type=cv.COLOR_BGR2RGB):
        self._transformer = transformer
        self.cvt_type = cvt_type

    def transform(self):
        for original, transformed in self._transformer.transform():
            yield original, cv.cvtColor(transformed, self.cvt_type)

    @property
    def provides_type(self):
        if self.cvt_type == cv.COLOR_BGR2RGB:
            return 'rgb888'
        elif self.cvt_type == cv.COLOR_RGB2BGR:
            return 'bgr888'
        else:
            raise 'Unsupported format (implementation for cv format {} is missing)'.format(self.cvt_type)
