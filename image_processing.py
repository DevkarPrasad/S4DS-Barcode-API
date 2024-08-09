# image_processing.py
import cv2
import numpy as np
from GPyOpt.methods import BayesianOptimization
import ot

# Image preprocessing
def image_processing(image):

    # Convert RGBA to RGB if necessary
    if len(image.shape) == 3 and image.shape[2] == 4:  # RGBA image
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        print(f"Converted RGBA to RGB, new shape: {image.shape}")

    # Convert RGB to grayscale if necessary
    if len(image.shape) == 3 and image.shape[2] == 3:  # RGB image
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        print(f"Converted RGB to grayscale, new shape: {image.shape}")
    elif len(image.shape) == 2:  # Grayscale image
        # If the image is already grayscale, no action is needed
        print(f"Image is already grayscale, shape: {image.shape}")
    else:
        raise ValueError("Unsupported image format")


    # Ensure the image is in 8-bit format
    if image.dtype != np.uint8:
        print("Converting image to 8-bit format")
        image = cv2.convertScaleAbs(image)  # Convert to 8-bit format

    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    try:
        equalized = clahe.apply(blur)
    except cv2.error as e:
        raise RuntimeError(f"CLAHE error: {str(e)}")

    return equalized


# Helper function to compute Sinkhorn distance
def sinkhorn_distance(hist1, hist2, epsilon=0.1):
    M = ot.dist(hist1.reshape(-1, 1), hist2.reshape(-1, 1), metric='euclidean')
    return ot.sinkhorn2(hist1, hist2, M, reg=epsilon)

import itertools

# Objective function for Bayesian optimization
def objective_function(params):
    global fingerprint  # Declare fingerprint as global
    # Unpack hyperparameters
    sigma = params[0][0]
    alpha = params[0][1]
    beta = params[0][2]

    # Apply semantic sketching
    sketched = apply_semantic_sketching(fingerprint, sigma, alpha, beta)

# Compute Sinkhorn distance

    # Compute histogram of edge intensities
    hist, _ = np.histogram(fingerprint, bins=256, range=(0, 256), density=True)

    # Reference histogram (e.g., uniform distribution)
    ref_hist, _ = np.histogram(sketched, bins=256, range=(0, 256), density=True)

    distance = sinkhorn_distance(hist, ref_hist)
    return -distance


# Semantic sketching function
def apply_semantic_sketching(image, sigma, alpha, beta):
    # Apply Gaussian smoothing
    smoothed = cv2.GaussianBlur(image, (0, 0), sigma)
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(smoothed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, alpha)
    # Apply dilation and erosion
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    # Adjust intensity
    adjusted = cv2.addWeighted(eroded, beta, image, 1 - beta, 0)

    return adjusted



def preprocess_image(image):
    global fingerprint  # Declare fingerprint as global
    fingerprint = image_processing(image)

    # Define hyperparameter bounds
    bounds = [{'name': 'sigma', 'type': 'continuous', 'domain': (0.1, 2)},
              {'name': 'alpha', 'type': 'continuous', 'domain': (0.1, 2)},
              {'name': 'beta', 'type': 'continuous', 'domain': (0.1, 2)}]

    # Perform Bayesian optimization
    optimizer = BayesianOptimization(f=objective_function, domain=bounds, maximize=False)
    optimizer.run_optimization(max_iter=100)

    # Get optimal hyperparameters
    optimal_params = optimizer.x_opt

    # Apply semantic sketching with optimal hyperparameters
    sketched = apply_semantic_sketching(fingerprint, *optimal_params)
    
    return sketched