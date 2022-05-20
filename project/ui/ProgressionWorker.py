import traceback

import cv2 as cv
import numpy as np
from PySide6.QtCore import QObject, Signal

from project.util import VideoWriter
from project.util.ColorConvertTransformer import ColorConvertTransformer


class ProgressionWorker(QObject):
    finished = Signal(bool)
    next_image = Signal(np.ndarray)
    frame_progressed = Signal(int)

    def __init__(self, transformer, destination_filepath: str, frame_rate: int, size):
        super().__init__()
        self.stopped = False
        self.destination_filepath = destination_filepath
        self.frame_rate = frame_rate
        self.width = size[0]
        self.height = size[1]
        if transformer.provides_type == 'bgr888':
            # convert on the fly to rgb using a proxy wrapper
            self.transformer = ColorConvertTransformer(transformer, cv.COLOR_BGR2RGB)
        else:
            self.transformer = transformer
        self._running = True  # Set to false to prematurely stop the worker

    def _init_writer(self):
        if hasattr(self.transformer, 'provides_shape'):
            shape = self.transformer.provides_shape()
        else:
            shape = (self.width, self.height)
        self.writer = VideoWriter(
            self.destination_filepath,
            self.frame_rate,
            shape,
            False if self.transformer.provides_type == 'gray8' else True)  # is_color?

    def run(self):
        self._init_writer()
        try:
            for i, (original_frame, transformed_frame) in enumerate(self.transformer.transform()):
                if not self._running:
                    break
                self.writer.write_frame(transformed_frame)
                self.next_image.emit(transformed_frame)
                self.frame_progressed.emit(i)
            self.writer.close()
        except Exception as e:
            print(traceback.format_exc())
            print(e)
            self.writer.close()
        self.finished.emit(True)

    def stop(self):
        self._running = False
