from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QImage, QPixmap, QColor, QPalette
from PySide6.QtWidgets import QVBoxLayout, QLabel, QWidget, QApplication, QScrollArea

from project.ui.ProgressionWorker import ProgressionWorker


class ProgressionWindow(QWidget):
    finished = Signal(bool)

    def __init__(self, transformer, destination_filepath: str):
        super().__init__()
        self.display_format = None
        self.transformer = transformer
        self.set_format_from_transformer(transformer)
        self.destination_filepath = destination_filepath

        self.main_layout = QVBoxLayout()
        self.header = QLabel('Fortschrittsanzeige')
        self.header.setAlignment(Qt.AlignCenter)
        self.header.setStyleSheet('font-size: 18px; font-weight: bold;')
        self.main_layout.addWidget(self.header)

        #self._init_video_widget()  # Uncomment blocks if video player becomes necessary; currently using image label
        self.image_label = QLabel()
        self.image_scroll_area = QScrollArea()
        self.image_scroll_area.setBackgroundRole(QPalette.Dark)
        self.image_scroll_area.setWidget(self.image_label)
        self.main_layout.addWidget(self.image_scroll_area)
        preferred_image_dimensions = self._get_preferred_display_image_size(
            transformer.reader.width, transformer.reader.height)
        self.image_label.setFixedSize(transformer.reader.width, transformer.reader.height)

        # Frame counter
        self._frame_counter_label = QLabel('Frame 0 / {}'.format(self.transformer.reader.frame_count))
        self._frame_counter_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self._frame_counter_label)

        self.setLayout(self.main_layout)
        self._init_worker_thread()
        self.window().setWindowTitle('Fortschrittsanzeige')
        self.window().setBaseSize(preferred_image_dimensions[0], preferred_image_dimensions[1])

        #self.display_video_stream()

    def _get_preferred_display_image_size(self, image_width: int, image_height: int):
        """ Naive way to limit the displayed image size. """
        primary_size = QApplication.primaryScreen().size()
        width = image_width
        height = image_height
        while primary_size.width() * 0.9 < width or primary_size.height() * 0.9 < height:
            width = width / 2
            height = height / 2
        return width, height

    def _init_worker_thread(self):
        self._worker_thread = QThread()

        self._progression_worker = ProgressionWorker(
            self.transformer, self.destination_filepath,
            self.transformer.reader.frame_rate,
            (self.transformer.reader.width, self.transformer.reader.height))

        self._worker_thread.started.connect(self._progression_worker.run)
        self._progression_worker.finished.connect(self._worker_thread.quit)
        self._progression_worker.finished.connect(self._worker_thread.deleteLater)
        self._progression_worker.finished.connect(self._emit_finished)
        self._worker_thread.finished.connect(self._worker_thread.deleteLater)

        self._progression_worker.moveToThread(self._worker_thread)
        self._progression_worker.next_image.connect(self.display_frame)
        self._progression_worker.frame_progressed.connect(self._update_progression)
        self._progression_worker.finished.connect(self._on_finished)

        self._worker_thread.start()

    def set_format_from_transformer(self, transformer):
        if not hasattr(transformer, 'provides_type'):
            print("Warning: Transformer doesn't provide output type, assuming RGB888")
            self.display_format = QImage.Format_RGB888
        elif transformer.provides_type == 'gray8':
            self.display_format = QImage.Format_Grayscale8
        elif transformer.provides_type == 'rgb888':
            self.display_format = QImage.Format_RGB888
        elif transformer.provides_type == 'bgr888':
            self.display_format = QImage.Format_BGR888
        else:
            raise Exception('Unsupported format')

    def _update_progression(self, frame: int):
        self._frame_counter_label.setText('Frame {} / {}'.format(frame, self.transformer.reader.frame_count))

    def _emit_finished(self):
        self.window().close()
        self.finished.emit(True)

    def stop(self):
        self._on_finished()

    def _on_finished(self):
        self._progression_worker.stop()

    #def _init_video_widget(self):
    #    self._scene = QGraphicsScene()
    #    self._media_player = QMediaPlayer()
    #    self._video_widget = QVideoWidget()
    #    self._graphics_item = QGraphicsItem()
    #    self._graphics_view = QGraphicsView(self._scene)
    #    self._graphics_view.scene().addItem(self._graphics_item)
    #    self.layout().addWidget(self._graphics_view)

    def display_frame(self, frame):
        image = QImage(frame, frame.shape[1], frame.shape[0],
                       frame.strides[0], self.display_format)
        self.image_label.setPixmap(QPixmap.fromImage(image))

    def display_video_stream(self):
        """Manually triggers synchronous processing and displaying of the video/image.

        Prefer using the constructor or _init_worker_thread for asynchronous processing.
        """
        count = 0
        for frame in self.transformer.transform():
            count += 1
            image = QImage(frame, frame.shape[1], frame.shape[0],
                           frame.strides[0], self.display_format)
            self.image_label.setPixmap(QPixmap.fromImage(image))
            self.writer.write_frame(frame)
            if count > 20:
                self.writer.close()
                break

