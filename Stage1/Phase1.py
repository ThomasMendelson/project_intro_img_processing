import cv2
import numpy as np
from sklearn.mixture import GaussianMixture
import pickle
import Training
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__)) # Get the current directory (directory of gui.py)
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir)) # Get the parent directory (project directory)
sys.path.append(parent_dir) # Add the parent directory to the Python path
from utils import colors, Masks


def Detect_Lego(image, gmm, color, n):
    # Preprocess test image
    test_data = Training.Preprocess_Image(image)

    # Predict labels using GMM
    labels = gmm.predict(test_data)

    # Reshape labels to the shape of the original image
    labels = labels.reshape(image.shape[:2])
    mask = MaskPicture(image, color)
    k = find_index(colors, color)
    labels = np.where((mask != 0) | (labels == k + 1), 1, 0)

    # Create mask for the specific color regions
    mask = (labels == 1)  # Assuming the color we want is the first component

    # Bitwise-AND mask and original image
    areas = cv2.bitwise_and(image, image, mask=np.uint8(mask * 255))

    return areas


def DetectGmm(img, color, n, Model = "Read"):
    model_path = "Trained_All.pkl"

    # Check if the model file exists
    if Model == "Read":
        # If model file exists, load the trained GMM model
        with open(model_path, 'rb') as f:
            gmm = pickle.load(f)
            print("Model Loaded")

    elif Model == "Write":
        # If model file doesn't exist, train a new GMM model
        train_data = Training.load_data()
        # Fit Gaussian Mixture Model (GMM)
        gmm = GaussianMixture(n_components=n, covariance_type='full')
        gmm.fit(train_data)
        # Save trained GMM model
        with open(model_path, 'wb') as f:
            pickle.dump(gmm, f)
            print("Model Saved")

    # Load test image
    test_image = cv2.imread(img)  # Path to your test image
    cv2.imshow('Image', test_image)

    # Detect yellow LEGO pieces
    detected = Detect_Lego(test_image, gmm, color, n)

    # Display result
    cv2.imshow(f"{color} Detection", detected)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()


def MaskPicture(image, color, read=False):
    if read:
        image = cv2.imread(image)

    # Define the desired width and height for the resized image
    desired_width = 640
    desired_height = 480

    # Resize the image
    image = cv2.resize(image, (desired_width, desired_height))

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define a broader HSV range for black, capturing variations
    Mask_lower = Masks[color][0]
    Mask_upper = Masks[color][1]

    # Create the mask for black color
    mask = cv2.inRange(hsv, Mask_lower, Mask_upper)

    if color == "Red":
        Mask_lower2 = Masks[color + "2"][0]
        Mask_upper2 = Masks[color + "2"][1]
        mask2 = cv2.inRange(hsv, Mask_lower2, Mask_upper2)
        mask = cv2.bitwise_or(mask, mask2)

    # Apply morphological operations to refine the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Close small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Remove small noise

    # Apply the mask to the original image
    Masked_Image = cv2.bitwise_and(image, image, mask=mask)
    if read:
        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.imshow('Mask', Masked_Image)
        cv2.waitKey(0)

    return mask


def find_index(colors, color_to_find):
    for i, color in enumerate(colors):
        if color.lower() == color_to_find:
            return i
    return -1


def All_Gmm(img):
    for color in colors:
        DetectGmm(img, color, 10)
    cv2.waitKey(0)
    cv2.destroyAllWindows()