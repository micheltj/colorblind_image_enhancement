import cv2 as cv


class ImageReader:
    """Simple reader around an image to act like a video reader"""
    def __init__(self, filename: str):
        self.filename = filename
        self.image = cv.imread(self.filename)
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]
        self.frame_count = 1
        self.frame_rate = 1

    def get_images(self, color_mode=None):
        if color_mode is not None:
            image = cv.cvtColor(self.image, color_mode)
        else:
            image = self.image
        return [image]

    @staticmethod
    def is_image(filepath: str) -> bool:
        return filepath.endswith('.png') or filepath.endswith('.jpg') or filepath.endswith('.jpeg') \
                or filepath.endswith('.svg') or filepath.endswith('.gif') or filepath.endswith('.webp')
