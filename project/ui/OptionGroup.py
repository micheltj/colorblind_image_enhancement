from PySide6.QtCore import Slot, Signal
from PySide6.QtWidgets import QGroupBox, QVBoxLayout, QButtonGroup, QRadioButton
from typing import List


class OptionGroupOption:
    def __init__(self, value: str, text: str):
        self.text = text
        self.value = value


class OptionGroup(QGroupBox):
    """Simple RadioButton selection list surrounded by a grouping box.

    Events triggered upon each option selection.
    """

    optionSelected = Signal(str)

    def __init__(self, title, options: List[str]):
        """
        Parameters:
            title: The title to show inside the grouping box.
            options: A list of objects defined as [{value:str, text:str}]
        """
        super().__init__(title)
        self.setStyleSheet("QGroupBox {color: #0336ff; font-weight: bold; font-size: 14px}")
        # self.setStyleSheet("QGroupBox::title {color: green;}")

        layout = QVBoxLayout()
        self.button_group = QButtonGroup()  # by default exclusive
        for option in options:
            option_button = QRadioButton(option)
            self.button_group.addButton(option_button)
            layout.addWidget(option_button)
        self.button_group.buttonClicked.connect(self.proxy_throw_event)
        self.setLayout(layout)

    @Slot()
    def proxy_throw_event(self, event: QRadioButton):
        self.optionSelected.emit(event.text())

