import tkinter as tk
from tkinter import PhotoImage
import Brick
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.mixture import GaussianMixture
import cv2
import numpy as np
import time as time




stock = []
stock.append(Brick.LegoBrick("Red",1,2,3))
stock.append(Brick.LegoBrick("Red",2,4,3))
stock.append(Brick.LegoBrick("Blue",4,2,2))
stock.append(Brick.LegoBrick("white",1,1,3))


def measure_lego_brick_size(image):
    # # Load the image
    # image = cv2.imread(image_path)

    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    for mask in Brick.Masks:
        print(mask)
        # Define lower and upper bounds for the color red (LEGO brick color)
        lower_bound = np.array(Brick.Masks[mask][0])
        upper_bound = np.array(Brick.Masks[mask][1])

        # Threshold the HSV image to get only red colors
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # Find contours in the masked image
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Loop through the contours
        for contour in contours:
            # Get the bounding box of the contour
            x, y, w, h = cv2.boundingRect(contour)
            # Filter out small contours (noise)
            if cv2.contourArea(contour) > 100:
                # Draw a rectangle around the detected LEGO brick
                cv2.rectangle(image, (x, y), (x + w, y + h),  (255, 255, 255), 2)
                # Add text next to the rectangle

                text = f"{cv2.contourArea(contour)}"  # Text showing width x height
                cv2.putText(image, text, (x + w + 10, y + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                print("width: {} height: {} area: {} \n" .format(w, h, cv2.contourArea(contour)))
        # # Display the image with the detected bricks
        # cv2.imshow('LEGO Bricks', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Return the number of detected bricks and the size of the largest brick
        # return len(contours), (w, h) if contours else None
    return image


def live_measure():
    # Initialize webcam
    cap = cv2.VideoCapture(1)

    # Time delay in milliseconds (100 milliseconds = 0.1 second)
    time_delay = 100

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Call the function to measure LEGO brick size
        processed_frame = measure_lego_brick_size(frame)

        # Display the resulting frame with rectangles and text
        cv2.imshow("Live Preview", processed_frame)

        # Check for any key press to break the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Display the resulting frame
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()

import cv2
import numpy as np
from sklearn.mixture import GaussianMixture
def load_data():
    dataset_images = []
    for i in range(1, 11):  # Assuming your images are named as 1.jpg, 2.jpg, ..., 5.jpg
        img_path = f"data base\green\captured_photo_{i}-removebg-preview.png"  # Path to your dataset images of the specific color
        # img_path1 = f"data base\yellow\captured_photo_{i}.jpg"  # Path to your dataset images of the specific color
        image = cv2.imread(img_path)
        dataset_images.append(image)
        # img_path1 = "data base\green\\all_Green-removebg-preview.png"  # Path to your dataset images of the specific color
        # image1 = cv2.imread(img_path1)
        # dataset_images.append(image1)
        # image1 = cv2.imread(img_path1)
        # dataset_images.append(image1)
    preprocessed_data = np.vstack([preprocess_image(img) for img in dataset_images])

    return preprocessed_data

def preprocess_image(image):
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
    return np.concatenate((flattened_hsv, flattened_lab,flattened_rgb, image_shape), axis=1)



def detect_yellow_lego(image, gmm):
    # Preprocess test image
    test_data = preprocess_image(image)

    # Predict labels using GMM
    labels = gmm.predict(test_data)

    # Reshape labels to the shape of the original image
    labels = labels.reshape(image.shape[:2])
    print(labels)
    # Create mask for yellow regions
    for i in range(3):
        yellow_mask = (labels == i)  # Assuming yellow is the first component
        # Bitwise-AND mask and original image
        yellow_areas = cv2.bitwise_and(image, image, mask=np.uint8(yellow_mask * 255))
        cv2.imshow("Yellow LEGO Detection {}".format(i), yellow_areas)
        gray = cv2.cvtColor(yellow_areas, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection
        number = 120
        edges = cv2.Canny(gray, number * 0.3, number, apertureSize=3)
        print("Value is ", number)
        cv2.imshow('Canny', edges)
        cv2.waitKey(0)
    return yellow_areas

def detectgmm(img):
    # Load training image

    # Preprocess training image
    train_data = load_data()

    # Fit Gaussian Mixture Model (GMM)
    gmm = GaussianMixture(n_components=3, covariance_type='full')

    gmm.fit(train_data)
    # dump(gmm, 'fitted_gmm_model.joblib')

    # Load test image
    test_image = cv2.imread(img)  # Path to your test image

    # Detect yellow LEGO pieces
    yellow_pieces = detect_yellow_lego(test_image,gmm)

    # Display result
    cv2.imshow('Yellow LEGO Detection', yellow_pieces)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
