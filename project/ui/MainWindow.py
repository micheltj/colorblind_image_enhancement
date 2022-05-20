from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget, QMainWindow, QHBoxLayout, QPushButton, QMessageBox

from . import OptionGroup
from .States import ExecState
from .ProgressionWindow import ProgressionWindow
from .MediaSelector import MediaSelector
from project.util.VideoReader import VideoReader
from project.Machado_Colorblind_Simulation import MachadoColorConverter
from project.saliency import FrequencyTunedSalientRegionDetection, MinimumBarrierSaliency, StaticFineSaliency, \
    SpectralSaliencyDetection
from project.saliency.rbd_ft import RobustBackgroundDetection
from project.recolor import Recoloring
from project.segmentation import KMeansTransformer
from project.saliency import PyramidFeatureAttentionNetwork
from project.util.ImageReader import ImageReader


class MainWindow(QMainWindow):

    _exec_saliency = 'Saliency (Maske)'
    _exec_segmentation = 'Bildsegmentierung'
    _exec_color_correction = 'Farbanpassung für Achromasie'

    _saliency_mbd = 'Minimum Barrier Salient'
    _saliency_rbd = 'Robust Background Detection'
    _saliency_ft = 'Frequency-tuned Salient Region Detection'
    _saliency_static_fine = "Static Fine Saliency"
    _spectral_saliency = 'Spectral Saliency'
    _saliency_pfans = 'Pyramid Feature Attention Network for Saliency Detection'
    _saliency_none_segm = 'Keine - reine Segmentierung'

    _cb_protan = 'Protan'
    _cb_deutan = 'Deutan'
    _cb_tritan = 'Tritan'

    _cb_extended_normal = 'Normal'
    _cb_extended_recolor = 'Recolor'
    _cb_extended_back_transform = 'Inkl. Rücktransformation'

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Digitale Bildverarbeitung 21/22 Gruppe 06')
        self.state = ExecState.INIT

        self._second_option = None
        self._third_option = None
        self._media_selector = None

        self.header = QLabel('BFFS')
        self.header.setAlignment(Qt.AlignCenter)
        self.header.setStyleSheet('font-size: 18px; font-weight: bold;')
        self.description = QLabel('<b>B</b>ildsegmentierung / <b>F</b>arbanpassung für <b>F</b>arbenblinde mithilfe von <b>S</b>aliency')
        self.description.setAlignment(Qt.AlignCenter)

        self.main_layout = QVBoxLayout(self)
        self.main_layout.addWidget(self.header)
        self.main_layout.addWidget(self.description)
        self.main_layout.addSpacing(32)

        self.options_layout = QHBoxLayout(self)
        self.options_layout.addWidget(self._init_exec_options())
        self.options_layout.addStretch(1)

        self.main_layout.addLayout(self.options_layout)

        self._execute_button = QPushButton('Ausführen')
        self._execute_button.clicked.connect(self._open_progression_window)
        self.main_layout.addWidget(self._execute_button)

        # Prevent child components from stretching by adding a spacer filling remaining space
        self.main_layout.addStretch(1)

        main_widget = QWidget()
        main_widget.setLayout(self.main_layout)

        self.setGeometry(0, 0, 800, 600)
        self.setCentralWidget(main_widget)

        self.selected_exec_option = None
        self.selected_color_correction_option = None
        self.selected_saliency_option = None
        self.selected_color_correction_extended_option = None

        self._second_option = self._init_saliency_options()
        #self._second_option.setCheckable(False)
        self._second_option.setDisabled(True)
        self.options_layout.insertWidget(1, self._second_option)

        self._add_media_selector()

    @staticmethod
    def _init_options(title, options, option_selected_function):
        group_widget = OptionGroup(title, options)
        group_widget.optionSelected.connect(option_selected_function)
        return group_widget

    def _init_exec_options(self):
        return self._init_options('Ausführung', [
                self._exec_saliency,
                self._exec_segmentation,
                self._exec_color_correction
            ], self.exec_option_selected)

    def _init_saliency_options(self):
        if self._third_option is not None:
            self._third_option.deleteLater()
            self._third_option = None
        if self.selected_exec_option == self._exec_segmentation:
            return self._init_options('Saliency Methode', [
                self._saliency_ft,
                self._saliency_mbd,
                self._saliency_rbd,
                self._saliency_static_fine,
                self._spectral_saliency,
                self._saliency_pfans,
                self._saliency_none_segm  # Optionally provide none to run without saliency step
            ], self.saliency_option_selected)
        else:
            return self._init_options('Saliency Methode', [
                self._saliency_ft,
                self._saliency_mbd,
                self._saliency_rbd,
                self._saliency_static_fine,
                self._spectral_saliency,
                self._saliency_pfans
            ], self.saliency_option_selected)

    def _init_cb_options(self):
        return self._init_options('Typ der Achromasie', [
            self._cb_deutan, self._cb_protan, self._cb_tritan
        ], self.cb_option_selected)

    def _init_cb_extended_options(self):
        return self._init_options('Transformationsschritte', [
            self._cb_extended_normal,
            self._cb_extended_recolor,
            self._cb_extended_back_transform,
        ], self.cb_extended_option_selected)

    def _add_media_selector(self):
        if self._media_selector is not None:
            return
        self._media_selector = MediaSelector()
        self.main_layout.insertWidget(4, self._media_selector)

    @Slot()
    def _open_progression_window(self):
        if self.state == ExecState.INIT:
            filepath: str = self._media_selector.get_source_file()
            if ImageReader.is_image(filepath):
                reader = ImageReader(filepath)
            else:
                reader = VideoReader(filepath)
            # Debug code to skip frames; algorithms were partially unable to process black/0-frames before
            #for i, img in enumerate(reader.get_images()):
            #    if i > 620:
            #        break
            #    i += 1
            transformer = self._create_transformer(reader)
            self.progression_window = ProgressionWindow(transformer, self._media_selector.get_save_file_path())
            self.progression_window.finished.connect(self._on_transform_finished)
            self.progression_window.show()
            self._execute_button.setText('Stop')
            self.state = ExecState.RUNNING
        elif self.state == ExecState.RUNNING:
            message_box = self._create_cancel_message_box()
            should_abort = message_box.exec()
            if should_abort == QMessageBox.Yes:
                self.progression_window.stop()

    def _create_transformer(self, reader: VideoReader):
        if self.selected_exec_option == self._exec_saliency:
            return self._create_saliency_transformer(reader)
        elif self.selected_exec_option == self._exec_segmentation:
            return self._create_segmentation_transformer(reader)
        elif self.selected_exec_option == self._exec_color_correction:
            return self._create_colorblind_transformer(reader)
        else:
            print('Unsupported execution function {}'.format(self.selected_exec_option))

    def _create_saliency_transformer(self, reader: VideoReader):
        if self.selected_saliency_option == self._saliency_ft:
            return FrequencyTunedSalientRegionDetection(reader)
        elif self.selected_saliency_option == self._saliency_rbd:
            return RobustBackgroundDetection(reader)
        elif self.selected_saliency_option == self._saliency_mbd:
            return MinimumBarrierSaliency(reader)
        elif self.selected_saliency_option == self._saliency_static_fine:
            return StaticFineSaliency(reader, thresholded=False)
        elif self.selected_saliency_option == self._spectral_saliency:
            return SpectralSaliencyDetection(reader)
        elif self.selected_saliency_option == self._saliency_pfans:
            return PyramidFeatureAttentionNetwork(reader)
        else:
            raise Exception('Unknown saliency option {}'.format(self.saliency_option_selected))

    def _create_segmentation_transformer(self, reader: VideoReader):
        if self.selected_saliency_option != self._saliency_none_segm:
            saliency_transformer = self._create_saliency_transformer(reader)
            return KMeansTransformer(
                reader, saliency_transformer=saliency_transformer, mode=KMeansTransformer.mode_saliency_region_only)
        else:
            return KMeansTransformer(reader)

    def _create_colorblind_transformer(self, reader: VideoReader):
        if self.selected_color_correction_option == self._cb_tritan:
            color_option = MachadoColorConverter.conversion_tritan
        elif self.selected_color_correction_option == self._cb_protan:
            color_option = MachadoColorConverter.conversion_protan
        elif self.selected_color_correction_option == self._cb_deutan:
            color_option = MachadoColorConverter.conversion_deutan
        else:
            raise Exception('Unsupported option for color blindness correction {}'
                            .format(self.selected_color_correction_option))

        machado_color_converter = MachadoColorConverter(
            reader, color_option,
            self.selected_color_correction_extended_option == self._cb_extended_back_transform)
        if self.selected_color_correction_extended_option == self._cb_extended_recolor:
            return Recoloring(machado_color_converter)
        else:
            return machado_color_converter

    def _on_transform_finished(self, _):
        self._execute_button.setText('Ausführen')
        self.state = ExecState.INIT

    @staticmethod
    def _create_cancel_message_box():
        message_box = QMessageBox()
        message_box.window().setWindowTitle('Berechnung stoppen')
        message_box.setText('<b>Berechnung vorzeitig beenden</b>')
        message_box.setInformativeText('Sind Sie sicher, dass sie den Prozess beenden möchten? Die Datei wird in '
                                       'ihrem jetzigen Zustand gespeichert, der restliche Arbeitsaufwand wird '
                                       '<b>nach Berechnung des derzeitigen Frames</b> verworfen.')
        message_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        message_box.setDefaultButton(QMessageBox.No)
        return message_box

    @Slot()
    def exec_option_selected(self, event):
        self.selected_exec_option = event
        second_option = None
        # Set the next set of choosable options depending on the execution goal
        if event == self._exec_saliency or event == self._exec_segmentation:
            second_option = self._init_saliency_options()
        if event == self._exec_color_correction:
            second_option = self._init_cb_options()

        if second_option is not None:
            if self._second_option is not None:
                self.options_layout.replaceWidget(self._second_option, second_option)
                self._second_option.deleteLater()
            else:
                self.options_layout.insertWidget(1, second_option)
            self._second_option = second_option

    @Slot()
    def saliency_option_selected(self, event):
        self.selected_saliency_option = event
        self._add_media_selector()
        # todo: segint
        #if self.selected_exec_option == self._exec_segmentation:
        #    third_option = QVBoxLayout()
        #    if self._third_option is not None:
        #        self.options_layout.replaceWidget(self._third_option, third_option)
        #        self._third_option.deleteLater()
        #    else:
        #        self.options_layout.insertWidget(2, third_option)
        #    self._third_option = third_option

    @Slot()
    def cb_option_selected(self, event):
        third_option = self._init_cb_extended_options()
        if self._third_option is not None:
            self.options_layout.replaceWidget(self._third_option, third_option)
            self._third_option.deleteLater()
        else:
            self.options_layout.insertWidget(2, third_option)

        self._third_option = third_option
        self.selected_color_correction_option = event
        self._add_media_selector()

    @Slot()
    def cb_extended_option_selected(self, event):
        self.selected_color_correction_extended_option = event

