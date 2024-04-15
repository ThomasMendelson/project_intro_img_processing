import cv2
import numpy as np
from sklearn.mixture import GaussianMixture
import pickle
import matplotlib.pyplot as plt
import sys
import os
# import Training as T
# from get_brick import find_and_add_bricks

current_dir = os.path.dirname(os.path.abspath(__file__)) # Get the current directory (directory of gui.py)
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir)) # Get the parent directory (project directory)
sys.path.append(parent_dir) # Add the parent directory to the Python path
from utils import colors, Masks, DEBUG, stock, long_side, IMG_HEIGHT, IMG_WIDTH


def Detect_Color(image, gmm, color):
    # Preprocess test image
    test_data = Preprocess_Image(image)

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
    detected_colors = cv2.bitwise_and(image, image, mask=np.uint8(mask * 255))

    return detected_colors

def MaskPicture(image, color, read=False):
    if read:
        image = cv2.imread(image)

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


def All_Gmm(original_img_path):
    original_img = perform_projective_transformation(cv2.imread(original_img_path))  # Path to your test image
    gmm = load_model(10)                                        #todo to move from here
    img = original_img

    for color in colors:
        masked_image = Detect_Color(original_img, gmm, color)
        img = find_and_add_bricks(img, masked_image, color)

    if DEBUG:
        plt.figure()
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()

    return img


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

















#////////////////////////////////////////////////////////////////////////////////
Types = {}
Types["Green"] = (4, 6, 8)
Types["Blue"] = (4, 6, 8, 12)
Types["Yellow"] = (4, 6, 8, 12, 20)
Types["Black"] = (4, 6, 8, 12)
Types["Red"] = (4, 8, 16)
Types["Orange"] = (4, 6, 16)
Types["White"] = (4, 6, 8, 12)


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

#///////////////////////////////////////////////////////////////////////////////////////////////////////////
def add_brick(color, long):
    if long < long_side[0]:
        stock[(color, 2)].add_one()
    elif long < long_side[1]:
        stock[(color, 3)].add_one()
    elif long < long_side[2]:
        stock[(color, 4)].add_one()
    else:
        stock[(color, 6)].add_one()

def perform_projective_transformation(image):
    # Define source and destination points for the transformation
    src_points = np.float32(
        [[0, 0], [image.shape[1] - 1, 0], [0, image.shape[0] - 1], [image.shape[1] - 1, image.shape[0] - 1]])
    dst_points = np.float32(
        [[0, 0], [image.shape[1] - 1, 0], [50, image.shape[0] - 1], [image.shape[1] - 51, image.shape[0] - 1]])

    # Calculate the homography matrix using corresponding points
    homography_matrix, _ = cv2.findHomography(src_points, dst_points)

    # Perform the projective transformation
    transformed_image = cv2.warpPerspective(image, homography_matrix, (image.shape[1], image.shape[0]))

    return transformed_image

def find_and_add_bricks(image, masked_image, color):
    # masked_image = perform_projective_transformation(masked_image)
    # Convert image to HSV color space
    hsv = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)
    # Define lower and upper bounds for the color mask
    lower_bound = np.array(Masks[color][0])
    upper_bound = np.array(Masks[color][1])
    mask_result = cv2.inRange(hsv, lower_bound, upper_bound)

    # Additional condition for Red mask
    if color == "Red":
        lower_bound2 = Masks[color + "2"][0]
        upper_bound2 = Masks[color + "2"][1]
        mask2 = cv2.inRange(hsv, lower_bound2, upper_bound2)
        mask_result = cv2.bitwise_or(mask_result, mask2)

    # Find contours in the masked image
    contours, _ = cv2.findContours(mask_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for contour in contours:
        # Get the minimum area rectangle bounding the contour
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        if cv2.contourArea(contour) > 450:
            long = np.linalg.norm(box[0] - box[1])
            long2 = np.linalg.norm(box[3] - box[2])
            short = np.linalg.norm(box[1] - box[2])
            short2 = np.linalg.norm(box[0] - box[3])
            long = (long + long2) / 2
            short = (short + short2) / 2
            if short > long:
                short, long = long, short
            if short > long_side[0]:                #we got few bricks together
                # Draw the rotated bounding box
                cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
            else:
                # Draw the rotated bounding box
                cv2.drawContours(image, [box], 0, (255, 255, 255), 2)
                add_brick(color, long)
            if DEBUG:
                # Add text next to the rectangle
                area = cv2.contourArea(contour)
                text = f"{area}"
                cv2.putText(image, text, (box[1][0], box[1][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                print(f"long: {long:.2f} short: {short:.2f} area: {area:.2f} \n")
    return image