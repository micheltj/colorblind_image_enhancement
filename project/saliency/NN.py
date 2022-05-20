from project.util.VideoReader import VideoReader
from project.NN import get_predictions


class NeuralNetworkSaliency:
    def __init__(self, reader: VideoReader):
        self.reader = reader

    def transform(self):
        for image in self.reader.get_images():
            yield image, get_predictions(image)

