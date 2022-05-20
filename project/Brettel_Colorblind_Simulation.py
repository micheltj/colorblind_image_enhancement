import numpy as np
import cv2 as cv
import matplotlib as mp
import openCVTools as cvt
import matplotlib.pyplot as plt

"""
Die Methode der Colorblind Simulation nach 
[Hans Brettel, Françoise Viénot, and John D. Mollon, "Computerized simulation of color appearance for dichromats," J. Opt. Soc. Am. A 14, 2647-2655 (1997)]
wurde von https://github.com/MaPePeR/jsColorblindSimulator übernommen.
Dabei haben wir den Code nur in Python übersetz und geringfügig verändert wobei ein Großerteil noch erhalten blieb.
"""

def sRGB_to_Linear_RGB(v):
    fv = v / 255.0

    if (fv < 0.04045): 
        return fv / 12.92

    return pow((fv + 0.055) / 1.055, 2.4)

def Linear_RGB_to_sRGB(v):
    if (v <= 0.): 
        return 0

    if (v >= 1.): 
        return 255

    if (v < 0.0031308): 
        return 0.5 + (v * 12.92 * 255)

    return 0 + 255 * (pow(v, 1.0 / 2.4) * 1.055 - 0.055)

Lookup = []
for i in range(0, 256):
    Lookup.append(sRGB_to_Linear_RGB(i))

brettel_params = {
    "protan": {
        "rgbCvdFromRgb_1": np.array([[0.14510, 1.20165, -0.34675],
                                     [0.10447, 0.85316, 0.04237],
                                     [0.00429, -0.00603, 1.00174]]),

        "rgbCvdFromRgb_2": np.array([[0.14115, 1.16782, -0.30897],
                                     [0.10495, 0.85730, 0.03776],
                                     [0.00431, -0.00586, 1.00155]]),

        "separationPlaneNormal": np.array([0.00048, 0.00416, -0.00464])
    },

    "deutan": {
        "rgbCvdFromRgb_1": np.array([[0.36198, 0.86755, -0.22953],
                                     [0.26099, 0.64512, 0.09389],
                                     [-0.01975, 0.02686, 0.99289]]),

        "rgbCvdFromRgb_2": np.array([[0.37009, 0.88540, -0.25549],
                                    [0.25767, 0.63782, 0.10451],
                                    [-0.01950, 0.02741, 0.99209]]),

        "separationPlaneNormal": np.array([-0.00293, -0.00645, 0.00938])
    },

    "tritan": {
        "rgbCvdFromRgb_1": np.array([[1.01354, 0.14268, -0.15622],
                                     [-0.01181, 0.87561, 0.13619],
                                     [0.07707, 0.81208, 0.11085]]),

        "rgbCvdFromRgb_2": np.array([[0.93337, 0.19999, -0.13336],
                                     [0.05809, 0.82565, 0.11626],
                                     [-0.37923, 1.13825, 0.24098]]),

        "separationPlaneNormal": np.array([0.03960, -0.02831, -0.01129 ])
    },
}

"""
@param paper: use the matrix values from the paper
"""
def brettel(srgb, t, severity, paper):

    rgb = []
    rgb.append(Lookup[srgb[0]])
    rgb.append(Lookup[srgb[1]])
    rgb.append(Lookup[srgb[2]])

    params = brettel_params[t]
    separationPlaneNormal = params["separationPlaneNormal"]
    rgbCvdFromRgb_1 = params["rgbCvdFromRgb_1"]
    rgbCvdFromRgb_2 = params["rgbCvdFromRgb_2"]

    dotWithSepPlane = rgb[0] * separationPlaneNormal[0] + rgb[1] * separationPlaneNormal[1] + rgb[2] * separationPlaneNormal[2]
    rgbCvdFromRgb = rgbCvdFromRgb_1 if dotWithSepPlane >= 0 else rgbCvdFromRgb_2

    if(paper):
        rgbCvdFromRgb = np.array([[0.347, 0.598, -0.365],
                                 [-0.007, -0.113, -1.185],
                                 [1.185, -1.570, 0.383]])

    rgb_cvd = []
    rgb_cvd.append(rgbCvdFromRgb[0, 0] * rgb[0] + rgbCvdFromRgb[0, 1] * rgb[1] + rgbCvdFromRgb[0, 2] * rgb[2])
    rgb_cvd.append(rgbCvdFromRgb[1, 0] * rgb[0] + rgbCvdFromRgb[1, 1] * rgb[1] + rgbCvdFromRgb[1, 2] * rgb[2])
    rgb_cvd.append(rgbCvdFromRgb[2, 0] * rgb[0] + rgbCvdFromRgb[2, 1] * rgb[1] + rgbCvdFromRgb[2, 2] * rgb[2])

    rgb_cvd[0] = rgb_cvd[0] * severity + rgb[0] * (1.0-severity)
    rgb_cvd[1] = rgb_cvd[1] * severity + rgb[1] * (1.0-severity)
    rgb_cvd[2] = rgb_cvd[2] * severity + rgb[2] * (1.0-severity)

    rgb_cvd[0] = Linear_RGB_to_sRGB(rgb_cvd[0])
    rgb_cvd[1] = Linear_RGB_to_sRGB(rgb_cvd[1])
    rgb_cvd[2] = Linear_RGB_to_sRGB(rgb_cvd[2])

    return rgb_cvd

"""
@param paper: use the matrix values from the paper
"""
def inverse_brettel(srgb, paper, t):
    rgb_cvd = []
    rgb_cvd.append(Lookup[srgb[0]])
    rgb_cvd.append(Lookup[srgb[1]])
    rgb_cvd.append(Lookup[srgb[2]])

    params = brettel_params[t]
    separationPlaneNormal = params["separationPlaneNormal"]
    rgbCvdFromRgb_1 = params["rgbCvdFromRgb_1"]
    rgbCvdFromRgb_2 = params["rgbCvdFromRgb_2"]

    # Invert the matrix to get original rgb value
    rgbCvdFromRgb_1_inv = np.linalg.inv(rgbCvdFromRgb_1)
    rgbCvdFromRgb_2_inv = np.linalg.inv(rgbCvdFromRgb_2)

    inverse_matrix = rgbCvdFromRgb_1_inv
    # Use values from Equation 4 (lamda, Y-B, R-G color space via LMS space using CIECAM02 model) https://link.springer.com/article/10.1007/s41095-020-0172-x
    if(paper):
        inverse_matrix = np.array([[1.225, -0.221, 0.482],
                                     [0.901, -0.364, -0.267],
                                     [-0.093, -0.807, 0.022]])
    rgb = np.matmul(inverse_matrix, rgb_cvd)

    rgb[0] = Linear_RGB_to_sRGB(rgb[0])
    rgb[1] = Linear_RGB_to_sRGB(rgb[1])
    rgb[2] = Linear_RGB_to_sRGB(rgb[2])

    return rgb

def convert_to_cvd(image):
    w = image.shape[0]
    h = image.shape[1]

    # loop over the image, pixel by pixel
    for x in range(0, w):
        for y in range(0, h):
            # threshold the pixel
            image[x, y] = brettel(image[x, y], "deutan", 1.0, True)

    # return the thresholded image
    return image

def convert_to_original(image):
    w = image.shape[0]
    h = image.shape[1]

    # loop over the image, pixel by pixel
    for x in range(0, w):
        for y in range(0, h):
            # threshold the pixel
            image[x, y] = inverse_brettel(image[x, y], True, "deutan")

    # return the thresholded image
    return image
