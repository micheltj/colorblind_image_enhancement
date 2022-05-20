import cv2
import numpy as np
from util import VideoReader

"""
Die Methode der Colorblind Simulation nach
[Gustavo M. Machado, Manuel M. Oliveira, and Leandro A. F. Fernandes "A Physiologically-based Model for Simulation of Color Vision Deficiency". 
IEEE Transactions on Visualization and Computer Graphics. Volume 15 (2009), Number 6, November/December 2009. pp. 1291-1298.]

wurde selbstständig Entwickelt wobei Teile aus https://github.com/MaPePeR/jsColorblindSimulator mit eingeflossen sind.
Die Matritzen unter machado_params stammen aus https://www.inf.ufrgs.br/~oliveira/pubs_files/CVD_Simulation/CVD_Simulation.html und wurden für die
entsprechenden Werte 1 zu 1 übernommen.
"""

class MachadoColorConverter:
    conversion_protan = 'protan'
    conversion_tritan = 'tritan'
    conversion_deutan = 'deutan'

    def sRGB_to_Linear_RGB(self, v):
        fv = v / 255.0

        if (fv < 0.04045):
            return fv / 12.92

        return pow((fv + 0.055) / 1.055, 2.4)

    def Linear_RGB_to_sRGB(self, v):
        if (v <= 0.):
            return 0

        if (v >= 1.):
            return 255

        if (v < 0.0031308):
            return 0.5 + (v * 12.92 * 255)

        return 0 + 255 * (pow(v, 1.0 / 2.4) * 1.055 - 0.055)

    machado_params = {
        "protan": {
            "0.6": np.array([[0.385450, 0.769005, -0.154455],
                             [0.100526, 0.829802, 0.069673],
                             [-0.007442, -0.022190, 1.029632]]),

            "1.0": np.array([[0.152286, 1.052583, -0.204868],
                             [0.114503, 0.786281, 0.099216],
                             [-0.003882, -0.048116, 1.051998]])
        },

        "deutan": {
            "0.6": np.array([[0.498864, 0.674741, -0.173604],
                             [0.205199, 0.754872, 0.039929],
                             [-0.011131, 0.030969, 0.980162]]),

            "1.0": np.array([[0.367322, 0.860646, -0.227968],
                             [0.280085, 0.672501, 0.047413],
                             [-0.011820, 0.042940, 0.968881]])
        },

        "tritan": {
            "0.6": np.array([[1.104996, -0.046633, -0.058363],
                             [-0.032137, 0.971635, 0.060503],
                             [0.001336, 0.317922, 0.680742]]),

            "1.0": np.array([[1.255528, -0.076749, -0.178779],
                             [-0.078411, 0.930809, 0.147602],
                             [0.004733, 0.691367, 0.303900]])
        }
    }

    def machado(self, srgb, t, severity):
        rgb = []
        rgb.append(self.Lookup[srgb[0]])
        rgb.append(self.Lookup[srgb[1]])
        rgb.append(self.Lookup[srgb[2]])

        cvd_matrix = self.machado_params[t][str(severity)]
        rgb = np.array(rgb)
        rgb_cvd = np.matmul(cvd_matrix, rgb)

        rgb_cvd[0] = self.Linear_RGB_to_sRGB(rgb_cvd[0])
        rgb_cvd[1] = self.Linear_RGB_to_sRGB(rgb_cvd[1])
        rgb_cvd[2] = self.Linear_RGB_to_sRGB(rgb_cvd[2])

        return rgb_cvd

    def inverse_machado(self, srgb, t, severity):
        rgb = []
        rgb.append(self.Lookup[srgb[0]])
        rgb.append(self.Lookup[srgb[1]])
        rgb.append(self.Lookup[srgb[2]])

        cvd_matrix = self.machado_params[t][str(severity)]
        rgb = np.array(rgb)
        rgb_normal = np.matmul(np.linalg.inv(cvd_matrix), rgb)

        rgb_normal[0] = self.Linear_RGB_to_sRGB(rgb_normal[0])
        rgb_normal[1] = self.Linear_RGB_to_sRGB(rgb_normal[1])
        rgb_normal[2] = self.Linear_RGB_to_sRGB(rgb_normal[2])

        return rgb_normal

    def convert_to_cvd(self, image: np.array, conversion_type: str = conversion_protan):
        # grab the image dimensions
        w = image.shape[0]
        h = image.shape[1]

        # loop over the image, pixel by pixel
        for x in range(0, w):
            for y in range(0, h):
                # threshold the pixel
                image[x, y] = self.machado(image[x, y], conversion_type, 1.0)

        return image

    def __init__(self, reader: VideoReader, conversion_type: str, convert_back: bool = False):
        self.reader = reader
        self.Lookup = []
        self.conversion_type = conversion_type
        for i in range(0, 256):
            self.Lookup.append(self.sRGB_to_Linear_RGB(i))

        # no class owned transform method as it's set dynamically by constructor parameter request here
        if convert_back:
            self.transform = self.transform_back_to_original
        else:
            self.transform = self.transform_normal

    @property
    def provides_type(self):
        return 'bgr888'

    def transform_normal(self):
        for image in self.reader.get_images(cv2.COLOR_BGR2RGB):
            yield image, self.convert_to_cvd(image, self.conversion_type)

    def transform_back_to_original(self):
        for image in self.reader.get_images():
            yield image, self.convert_to_original(image, self.conversion_type)

    def convert_to_original(self, image: np.array, conversion_type: str = conversion_protan):
        # grab the image dimensions
        w = image.shape[0]
        h = image.shape[1]

        # loop over the image, pixel by pixel
        for x in range(0, w):
            for y in range(0, h):
                # threshold the pixel
                image[x, y] = self.inverse_machado(image[x, y], conversion_type, 1.0)

        return image
