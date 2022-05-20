# Projekt
In diesem Ordner sollen alle Dateien untergebracht werden, die zur Ausführung ihres Projektes notwendig sind (Source Code, Build-Dateien, Modelle, Assets, etc.). 

## Installationsanleitung
Welche Voraussetzungen sind notwendig, um das Projekt auszuführen? Geben Sie eine detaillierte Installationsanleitung mit allen Abhängigkeiten.

### Installation
Aus dem Arbeitsverzeichnis (WS2122_CG06) heraus:
- `python -m venv env` zur Erstellung einer virtual environment und Vermeidung von Abhängigkeitskollisionen
- `source env/bin/activate` unter Unix Shells bzw. `env\bin\activate.bat` unter Windows
- `pip install .` installiert Abhängigkeiten, die in `setup.py` gelistet sind
- Falls eine CUDA-fähige Grafikkarte vorliegt und [`nvcc`](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html) vorinstalliert ist, kann zur schnelleren Ausführung des PatchMatch Algorithmus PyTorch installiert werden: `pip install .[cuda]`
- `python project/ui.py`

Zudem ist eine Internetverbindung während der ersten Ausführung einiger Algorithmen erforderlich, da benötigte Modelle (z. B. VGG16) zur Laufzeit heruntergeladen werden.

### Bibliotheken
Eine manuelle Installation der Bibliotheken ist nur notwendig, wenn die vorliegende [setup.py](../setup.py) nicht zur Installation genutzt wird.

| Bibliothek (PyPI Bezeichnung) | Version | Beschreibung |
| ---- | ---- | ---- |
| numpy | `>=1.20.0` | - |
| opencv-contrib-python | any | Community Contributions für OpenCV |
| matplotlib | any | - |
| scikit-learn | any | - |
| scikit-image | any | - |
| pyside6 | any | Benutzeroberfläche Qt |
| shiboken6 | Selbe wie pyside6 | Benutzeroberfläche Qt |
| torch | any | ML / Neuronale Netze |
| torchvision | any | - |
| tqdm | any | Ladeanzeige für CLI |
| scipy | any | - |

Weiterhin die optionalen CUDA- und ML-Bibliotheken für deutlich erhöhte Performance von Algorithmen, die neuronale Netze nutzen:

| Bibliothek (PyPI Bezeichnung) | Version |
| ---- | ---- |
| pycuda | `>=2021.1` |
| cupy-cuda | Abhängig von installierter CUDA Version: `cupy-cuda<CUDA_VERSION>` |

### Weitere Hinweise
Es sollten Pfadanpassungen unter `project/NN/inference.py` gemacht werden falls das Program das Model nicht finden sollte.
Dafür einfach in `inference.py` unter `parser_arguments() __model_path` den Pfad `/model/best-model_epoch-204_mae-0.0505_loss-0.1370.pth` durch `C:/Users/.../WS2122_G06/project/NN/model/best-model_epoch-204_mae-0.0505_loss-0.1370.pth` ersetzen.
