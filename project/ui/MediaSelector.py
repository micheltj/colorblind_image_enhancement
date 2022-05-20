from PySide6.QtCore import Slot
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget, QGraphicsVideoItem
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QFileDialog, QHBoxLayout, QLineEdit, QGraphicsView, QGraphicsScene

from project.util.ImageReader import ImageReader


class MediaSelector(QWidget):
    _video_widget = None
    _video_player = None
    file = ''

    def __init__(self):
        super().__init__()
        self.main_layout = QVBoxLayout()
        file_dialog_button = QPushButton('Bild oder Video auswählen')
        file_dialog_button.clicked.connect(self.open_file_dialog)
        self.main_layout.addWidget(file_dialog_button)

        self._media_player = None
        self._video_player = None

        self._media_player = QMediaPlayer()
        self._video_widget = QVideoWidget()
        self.scene = QGraphicsScene()
        self._graphics_view = QGraphicsView(self.scene)
        self._graphics_item = QGraphicsVideoItem()
        self._video_widget.setUpdatesEnabled(True)
        self._media_player.errorOccurred.connect(self.print_error)
        self._graphics_view.scene().addItem(self._graphics_item)
        self._graphics_view.setFixedHeight(240)
        self.main_layout.addWidget(self._graphics_view)
        self.main_layout.addSpacing(10)

        save_layout = QHBoxLayout()
        self.save_location_text = QLineEdit()
        save_dialog_button = QPushButton('Speicherort auswählen')
        save_dialog_button.clicked.connect(self.open_save_dialog)
        save_layout.addWidget(self.save_location_text)
        save_layout.addWidget(save_dialog_button)
        self.main_layout.addLayout(save_layout)

        self.main_layout.addSpacing(10)

        self.setLayout(self.main_layout)

    @Slot()
    def open_file_dialog(self):
        file_dialog = QFileDialog(caption='Bild oder Video auswählen',
                                  filter='Image/Video (*.png *.jpg *.mp4 *.mkv *.avi)')
        if file_dialog.exec() == QFileDialog.Accepted:
            self.file = file_dialog.selectedFiles()[0]
            print('Opening {}'.format(self.file))
            self._media_player.setVideoOutput(self._graphics_item)
            if not ImageReader.is_image(self.file):
                self._media_player.setSource(self.file)
                self._media_player.play()
        else:
            print('Media selection aborted')

    def get_source_file(self):
        return self.file

    @Slot()
    def open_save_dialog(self):
        save_location = QFileDialog.getSaveFileName(self, caption='Speicherort auswählen')
        print('Save location will be {} with mimetype', save_location[0], save_location[1])
        self.save_location_text.setText(save_location[0])

    def get_save_file_path(self):
        return self.save_location_text.text().strip()

    @Slot()
    def print_error(self, error, error_string):
        print('An error during media load/play occurred:')
        print(error_string)
