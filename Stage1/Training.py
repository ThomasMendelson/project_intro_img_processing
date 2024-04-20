import cv2
import numpy as np
import sys
import os
from sklearn.mixture import GaussianMixture
import pickle

current_dir = os.path.dirname(os.path.abspath(__file__)) # Get the current directory (directory of gui.py)
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir)) # Get the parent directory (project directory)
sys.path.append(parent_dir) # Add the parent directory to the Python path
from utils import colors


Types = {}
Types["Green"] = (4, 6, 8)
Types["Blue"] = (4, 6, 8, 12)
Types["Yellow"] = (4, 6, 8, 12, 20)
Types["Black"] = (4, 6, 8, 12)
Types["Red"] = (4, 8, 16)
Types["Orange"] = (4, 6, 16)
Types["White"] = (4, 6, 8, 12)

global gmm


def load_data():
    dataset_images = []
    # if color == "all":
    for color in colors:
        for size in Types[color]:  # choose the size Types of every color
            for j in range(10):  # Assuming each type has 10 pictures at the Data Base
                # if j == 0 or j == 5:
                #     continue
                img_path = f"data base/Library/NoBG/{color}NoBG/{color}{size}{j}-removebg-preview.png"  # Path to your dataset images of the specific color and size
                image = cv2.imread(img_path)
                dataset_images.append(image)
        for j in range(5):
            img_path = f"data base/Library/NoBG/{color}NoBG/all_{color}_{j}-removebg-preview.png"  # Path to your dataset images of the specific color and size
            image = cv2.imread(img_path)
            dataset_images.append(image)
    preprocessed_data = np.vstack([Preprocess_Image(img) for img in dataset_images])

    return preprocessed_data


def Preprocess_Image(image):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Convert image to Lab color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    # Flatten the HSV and Lab images to a 2D array and concatenate them
    flattened_hsv = hsv.reshape((-1, 3))
    flattened_lab = lab[:, :, [0, 1, 2]].reshape((-1, 3))
    flattened_rgb = image.reshape((-1, 3))

    # Get image shape and repeat it for each pixel
    image_shape = np.repeat(np.array(image.shape[:2]).reshape((1, 2)), np.prod(image.shape[:2]), axis=0)

    # Concatenate all features
    return np.concatenate((flattened_hsv, flattened_lab, flattened_rgb, image_shape), axis=1)


def load_model(n, Model = "Read"):
    model_path = "../Trained_All.pkl"

    # Check if the model file exists
    if Model == "Read":
        # If model file exists, load the trained GMM model
        with open(model_path, 'rb') as f:
            gmm = pickle.load(f)
            print("Model Loaded")

    elif Model == "Write":
        # If model file doesn't exist, train a new GMM model
        train_data = load_data()
        # Fit Gaussian Mixture Model (GMM)
        gmm = GaussianMixture(n_components=n, covariance_type='full')
        gmm.fit(train_data)
        # Save trained GMM model
        with open(model_path, 'wb') as f:
            pickle.dump(gmm, f)
            print("Model Saved")

    return gmm


gmm = load_model(10)