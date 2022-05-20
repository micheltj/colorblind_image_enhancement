import cv2 as cv


class VideoReader:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.cap = cv.VideoCapture(filepath)
        self.frame_count = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
        self.frame_rate = int(self.cap.get(cv.CAP_PROP_FPS))
        self.estimated_duration = float(self.frame_count) / self.frame_rate
        self.height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))

    def get_images(self, color_mode=None):
        """Returns a generator to read the provided video capture image by image.
        Parameters:
            color_mode: The color mode for frame conversion. Can be set to cv2.COLOR_BGR2RGB or others.
        """
        if not self.cap.isOpened():
            self.cap.open(self.filepath)
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if frame is None:
                return []
            if color_mode is None:
                yield frame
            else:
                yield cv.cvtColor(frame, color_mode)
        self.cap.release()

