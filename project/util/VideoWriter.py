import cv2 as cv
import numpy as np


class VideoWriter:
    def __init__(self, output_file: str, frame_rate: int, frame_size: (int, int), is_color=True):
        # Alternatively codec for motion jpg: 'M', 'J', 'P', 'G'
        if not output_file.endswith('mp4'):
            output_file = output_file+'.mp4'
            print('Supported format is currently MP4, naming file "{}" instead'.format(output_file))
        self.writer = cv.VideoWriter(output_file, cv.VideoWriter_fourcc(*'mp4v'), frame_rate, frame_size, is_color)

    def write_frame(self, frame: np.array):
        self.writer.write(frame)

    def close(self):
        self.writer.release()
