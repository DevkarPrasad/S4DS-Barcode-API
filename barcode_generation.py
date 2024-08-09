import os
import cv2
import numpy as np
import itertools
import skimage
from scipy.stats import wasserstein_distance
from skimage.feature import corner_harris, corner_peaks
from skimage.morphology import skeletonize, convex_hull_image, erosion, square
from skimage.measure import label, regionprops
from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix
import gudhi as gd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from skimage import feature, morphology
from PIL import Image
import base64

plt.rcParams['text.usetex'] = False
# Function to detect ridges using Hessian matrix
def detect_ridges(gray, sigma=0.1):
    H_elems = skimage.feature.hessian_matrix(gray, sigma=sigma, order='rc')
    maxima_ridges, minima_ridges = skimage.feature.hessian_matrix_eigvals(H_elems)
    return maxima_ridges, minima_ridges

# Function to get termination and bifurcation points
def getTerminationBifurcation(img, mask):
    img = img == 255
    (rows, cols) = img.shape
    minutiaeTerm = np.zeros(img.shape)
    minutiaeBif = np.zeros(img.shape)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if img[i][j] == 1:
                block = img[i - 1:i + 2, j - 1:j + 2]
                block_val = np.sum(block)
                if block_val == 2:
                    minutiaeTerm[i, j] = 1
                elif block_val == 4:
                    minutiaeBif[i, j] = 1

    mask = skimage.morphology.convex_hull_image(mask > 0)
    mask = skimage.morphology.erosion(mask, skimage.morphology.square(5))
    minutiaeTerm = np.uint8(mask) * minutiaeTerm
    return minutiaeTerm, minutiaeBif


class MinutiaeFeature:
    def __init__(self, locX, locY, Orientation, Type):
        self.locX = locX
        self.locY = locY
        self.Orientation = Orientation
        self.Type = Type

def computeAngle(block, minutiaeType):
    angle = 0
    (blkRows, blkCols) = np.shape(block)
    CenterX, CenterY = (blkRows - 1) / 2, (blkCols - 1) / 2
    if minutiaeType.lower() == 'termination':
        sumVal = 0
        for i in range(blkRows):
            for j in range(blkCols):
                if (i == 0 or i == blkRows - 1 or j == 0 or j == blkCols - 1) and block[i][j] != 0:
                    angle = -np.degrees(np.arctan2(i - CenterY, j - CenterX))
                    sumVal += 1
                    if sumVal > 1:
                        angle = float('nan')
        return angle
    elif minutiaeType.lower() == 'bifurcation':
        angle = []
        sumVal = 0
        for i in range(blkRows):
            for j in range(blkCols):
                if (i == 0 or i == blkRows - 1 or j == 0 or j == blkCols - 1) and block[i][j] != 0:
                    angle.append(-np.degrees(np.arctan2(i - CenterY, j - CenterX)))
                    sumVal += 1
        if sumVal != 3:
            angle = float('nan')
        return angle


def extractMinutiaeFeatures(skel, minutiaeTerm, minutiaeBif):
    FeaturesTerm = []
    minutiaeTerm = label(minutiaeTerm, connectivity=2)
    RP = regionprops(minutiaeTerm)
    WindowSize = 2
    for i in RP:
        (row, col) = np.int16(np.round(i['Centroid']))
        block = skel[row - WindowSize:row + WindowSize + 1, col - WindowSize:col + WindowSize + 1]
        angle = computeAngle(block, 'Termination')
        FeaturesTerm.append(MinutiaeFeature(row, col, angle, 'Termination'))

    FeaturesBif = []
    minutiaeBif = label(minutiaeBif, connectivity=2)
    RP = regionprops(minutiaeBif)
    WindowSize = 1
    for i in RP:
        (row, col) = np.int16(np.round(i['Centroid']))
        block = skel[row - WindowSize:row + WindowSize + 1, col - WindowSize:col + WindowSize + 1]
        angle = computeAngle(block, 'Bifurcation')
        FeaturesBif.append(MinutiaeFeature(row, col, angle, 'Bifurcation'))
    return FeaturesTerm, FeaturesBif

def AddMinutiae(skel, FeaturesTerm, FeaturesBif):
    minutiaeBif = np.zeros(skel.shape)
    minutiaeTerm = np.zeros(skel.shape)

    (rows, cols) = skel.shape
    DispImg = np.zeros((rows, cols, 3), np.uint8)
    DispImg[:, :, 0] = skel
    DispImg[:, :, 1] = skel
    DispImg[:, :, 2] = skel

    # Assuming you have already obtained FeaturesTerm and FeaturesBif
    minutiae_points = []
    for feature in FeaturesTerm + FeaturesBif:
        minutiae_points.append([feature.locX, feature.locY])
    return minutiae_points


def reduce_minutiae_points(points, max_points=100):
    if len(points) > max_points:
        kmeans = KMeans(n_clusters=max_points, random_state=0).fit(points)
        reduced_points = kmeans.cluster_centers_
    else:
        reduced_points = points
    return reduced_points

def compute_distance_matrix(points):
    return distance_matrix(points, points)  # Euclidean distance (L2 norm)

def construct_vietoris_rips_filtration(dist_matrix, max_dimension=1, max_filtration_value=np.inf):
    rips_complex = gd.RipsComplex(distance_matrix=dist_matrix, max_edge_length=max_filtration_value)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension)
    return simplex_tree

def compute_persistent_homology(simplex_tree):
    return simplex_tree.persistence()



def generate_barcode(img):

    # Ensure the image is in grayscale (single channel)
    if len(img.shape) == 3 and img.shape[2] == 3:  # If image has 3 channels (color image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    elif len(img.shape) == 2:  # Grayscale image
        pass
    else:
        raise ValueError("Unsupported image format")

    # Apply Otsu's thresholding
    ret, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Skeletonize the binary image
    skel = skeletonize(img_bin // 255)
    skel = np.uint8(skel) * 255

    # Create a mask from the binary image
    mask = img_bin * 255

    # Get termination and bifurcation points
    minutiaeTerm, minutiaeBif = getTerminationBifurcation(skel, mask)
    FeaturesTerm, FeaturesBif = extractMinutiaeFeatures(skel, minutiaeTerm, minutiaeBif)

    # Label the features for visualization
    BifLabel = label(minutiaeBif, connectivity=1)
    TermLabel = label(minutiaeTerm, connectivity=1)
    minutiae_points = AddMinutiae(skel, FeaturesTerm, FeaturesBif)
    # Reduce the number of minutiae points
    reduced_minutiae_points = reduce_minutiae_points(minutiae_points, max_points=100)

    # Compute the distance matrix and construct the Vietoris-Rips filtration
    dist_matrix = compute_distance_matrix(reduced_minutiae_points)
    simplex_tree = construct_vietoris_rips_filtration(dist_matrix)

    # Compute the persistent homology and plot the barcode
    persistence = compute_persistent_homology(simplex_tree)
    # Save the plot to a BytesIO object
    img_stream = BytesIO()

    # Plot the persistence barcode
    gd.plot_persistence_barcode(persistence)
    plt.savefig(img_stream, format='jpg')  # Save the figure to the BytesIO object
    plt.close()  # Close the figure to avoid display

    # Reset the stream position to the beginning
    img_stream.seek(0)

    # Now you can use img_stream as needed, e.g., convert to PIL Image, or base64

    img_base64 = base64.b64encode(img_stream.getvalue()).decode('utf-8')
    return img_base64