import cv2
import numpy as np

from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *


def rescale(img):
    # Get the dimensions of the image
    height, width, channel = img.shape
    bytesPerLine = 3 * width

    # Convert cv2 image to QImage
    qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format.Format_RGB888)

    # Convert QImage to QPixmap for display
    pixmap = QPixmap.fromImage(qImg)
    pixmap = pixmap.scaled(500, 500)

    return pixmap


def rescaleOneChannel(img):
    # Get the dimensions of the image
    height, width = img.shape
    bytesPerLine = width

    # Convert cv2 image to QImage
    qImg = QImage(
        img.data, width, height, bytesPerLine, QImage.Format.Format_Grayscale8
    )

    # Convert QImage to QPixmap for display
    pixmap = QPixmap.fromImage(qImg)
    pixmap = pixmap.scaled(500, 500)

    return pixmap


def openImage():
    # Open the file dialog box
    filename, _ = QFileDialog.getOpenFileName()

    # Read image
    bgrImg = cv2.imread(filename)

    # Convert to rgb
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
    return rgbImg


##############################################################################
# COLORSPACE FUNCTIONS
def cmyConvert(img):
    R = img[..., 0]
    G = img[..., 1]
    B = img[..., 2]

    # K = 1 - max(R, G, B)
    K = 1 - np.max(img)
    # C = 1 - red / K
    C = (1 - R - K) / (1 - K)
    # M = 1 - green / K
    M = (1 - G - K) / (1 - K)
    # Y = 1 - blue / K
    Y = (1 - B - K) / (1 - K)

    CMY = (np.dstack((C, M, Y)) * 255).astype(np.uint8)
    C, M, Y = cv2.split(CMY)
    return [C, M, Y]


def grayConvert(img):
    grayImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return grayImg


def hsvConvert(img):
    HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(HSV)
    return [H, S, V]


def luvConvert(img):
    LUV = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    L, U, V = cv2.split(LUV)
    return [L, U, V]


##############################################################################
# FILTER FUNCTIONS
def averageBlur(img):
    average = cv2.blur(img, (3, 3))
    return average


def medianBlur(img):
    median = cv2.medianBlur(img, 3)
    return median


def gaussianBlur(img):
    gaussian = cv2.GaussianBlur(img, (3, 3), 0)
    return gaussian


def sharpen(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    sharpen = cv2.filter2D(img, -1, kernel)
    return sharpen


##############################################################################
# EDGE DETECTION FUNCTIONS
def sobel(img):
    sobelX = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=3)
    sobelY = cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=3)

    sobelXY = sobelX + sobelY
    return sobelXY


def laplacian(img):
    laplacian = cv2.Laplacian(img, cv2.CV_8U, 3)
    return laplacian


def log(img):
    gaussian = cv2.GaussianBlur(img, (3, 3), 0)
    log = cv2.Laplacian(gaussian, cv2.CV_8U, 3)
    return log


def canny(img):
    canny = cv2.Canny(img, 260, 550)
    return canny


##############################################################################
# THRESHOLDING FUNCTIONS
def binaryThreshold(img):
    ret, threshold = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    return threshold


def adaptiveThreshold(img):
    # thresh1 = cv2.adaptiveThreshold(splitHSV[1], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 1015, 5)
    ret, threshold = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 415, 5
    )
    return threshold


def otsuThreshold(img):
    ret, threshold = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return threshold


##############################################################################
# MORPHOLOGICAL OPERATION FUNCTIONS
def erode(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded = cv2.erode(img, kernel, iterations=3)
    return eroded


def dilate(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(img, kernel, iterations=3)
    return dilated


def open(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return opened


def close(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return closed
