import cv2 as cv
import pathlib
import glob
import numpy as np
import matplotlib.pyplot as plt


def kmeans_segmentation(image):
    # Reshape the image to a 2D array of pixels
    pixels = image.reshape(-1, 3).astype(np.float32)

    # Define the criteria and perform k-means clustering
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv.kmeans(
        pixels, 2, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS
    )

    # Calculate the average pixel value for each cluster
    average_colors = np.uint8(centers)

    # Find the index of the cluster with the lower average pixel value
    cluster_index = np.argmin(average_colors)

    # Convert the label image to 8-bit and reshape it to the original image shape
    labels = labels.reshape(image.shape[:2]).astype(np.uint8)

    white_area = np.sum(labels == 1)

    # Convert the label image to 8-bit and reshape it to the original image shape
    labels = labels.reshape(image.shape[:2]).astype(np.uint8)

    # Invert the segmentation mask if the white area is larger than the segmented part
    if white_area > (labels.size - white_area):
        labels = 1 - labels

    # Segment the image based on the labels
    segmented_image = np.where(labels == 1, 255, 0).astype(np.uint8)

    return segmented_image


def spot_cleanup(image):
    # Convert the image to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Perform adaptive thresholding to obtain a binary image
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # Perform morphological operations to clean the spots in the background
    kernel_background = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    cleaned_background = cv.morphologyEx(
        binary, cv.MORPH_OPEN, kernel_background, iterations=2
    )

    # Perform morphological operations to close the spots in the foreground
    kernel_foreground = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    closed_foreground = cv.morphologyEx(
        cleaned_background, cv.MORPH_CLOSE, kernel_foreground, iterations=2
    )

    # Invert the binary image to obtain the final mask
    mask = cv.bitwise_not(closed_foreground)

    # Apply the mask to the original image
    segmented_image = cv.bitwise_and(image, image, mask=mask)

    return segmented_image


def preprocess_image(image, label):
    # Perform spot cleanup
    image = np.array(image)
    preprocessed_image = kmeans_segmentation(image)
    preprocessed_image = spot_cleanup(preprocessed_image)
    preprocessed_image = preprocessed_image.convert_to_tensor()

    return preprocessed_image, label
