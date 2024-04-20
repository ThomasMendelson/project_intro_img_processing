import cv2
import numpy as np
from sklearn.mixture import GaussianMixture
import pickle
import sys
import os
from time import time
from threading import Thread
from queue import Queue
from copy import deepcopy
from Stage1.Training import load_data, Preprocess_Image


current_dir = os.path.dirname(os.path.abspath(__file__)) # Get the current directory (directory of gui.py)
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir)) # Get the parent directory (project directory)
sys.path.append(parent_dir) # Add the parent directory to the Python path
from utils import colors, Masks, DEBUG, stock, long_side, IMG_HEIGHT, IMG_WIDTH, cap, cap2


def Detect_Color(image, labels, color, Mode):
    # Reshape labels to the shape of the original image
    mask = MaskPicture(image, color)
    k = find_index(colors, color)
    if Mode == "both":
        labels = labels.reshape(image.shape[:2])
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

    # Convert the image to HLS color space
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

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

    if color == "White":
        Mask_lower_hls = Masks[color + "2"][0]
        Mask_upper_hls = Masks[color + "2"][1]
        mask_hls = cv2.inRange(hls, Mask_lower_hls, Mask_upper_hls)
        mask = cv2.bitwise_and(mask, mask_hls)

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
        if colors[i] == color_to_find:
            return i
    return -1


def All_Gmm(original_img, gmm, Mode):
    # original_img = perform_projective_transformation(cv2.imread(original_img_path))  # Path to your test image
    img = deepcopy(original_img)
    labels = []

    if Mode == "both":
        # Preprocess test image
        test_data = Preprocess_Image(original_img)
        # Predict labels using GPU-accelerated GMM
        labels = gmm.predict(test_data)

    for color in colors:
        masked_image = Detect_Color(original_img, labels, color, Mode)
        img = find_and_add_bricks(img, masked_image, color)

    if DEBUG:
        # plt.figure()
        # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # plt.show()
        cv2.imshow("Detected Bricks", img)
    return img




#/////////////////////////////////////////////////////////////////////////////////////////////////////////// Brick //////////////////
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
    # cv2.imshow(f"mask_image", masked_image)
    # cv2.waitKey(0)

    tmp = distance_transform(cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY))
    kernel = np.ones((5, 5), np.uint8)  # Define the erosion kernel (structuring element)
    tmp2 = cv2.erode(tmp, kernel, iterations=3)

    # Find contours in the masked image
    contours, _ = cv2.findContours(cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for contour in contours:
        # Get the minimum area rectangle bounding the contour
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        if cv2.contourArea(contour) > 450:
            short, long = FindSides(box)
            if short > long_side[0]:  # we got few bricks together
                split_Contours = Seperate_Bricks(contour, image, tmp2)
                count = 0
                if len(split_Contours) > 1:
                    for C in split_Contours:
                        mask = np.zeros_like(image)
                        cv2.drawContours(mask, [C], -1, 255, thickness=-2)
                        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                        kernel = np.ones((5, 5), np.uint8)  # Define the dilation kernel
                        dilated_image = cv2.dilate(mask, kernel, iterations=3)
                        T, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        T = sorted(T, key=cv2.contourArea, reverse=True)

                        # cv2.imshow(f"split contour {count}", mask)
                        # cv2.waitKey(0)
                        # cv2.imshow(f"split contour dilated {count}", dilated_image)
                        # cv2.waitKey(0)

                        count = count + 1
                        rectS = cv2.minAreaRect(T[0])
                        boxS = cv2.boxPoints(rectS)
                        boxS = np.intp(boxS)
                        short2, long2 = FindSides(boxS)
                        # print(long2, short2)
                        # print(long2, short2)
                        if short2 > long_side[0]:
                            cv2.drawContours(image, [boxS], 0, (0, 0, 255), 2)
                            continue
                        add_brick(color, long2)
                        cv2.drawContours(image, [boxS], 0, (255, 0, 255), 2)
                        if DEBUG:
                            # Add text next to the rectangle
                            area2 = cv2.contourArea(contour)
                            text = f"{area2}"
                            cv2.putText(image, text, (boxS[1][0], boxS[1][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (255, 255, 255), 1)
                            print(f"long: {long2:.2f} short: {short2:.2f} area: {area2:.2f} \n")
                else:
                    cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
            else:
                # Draw the rotated bounding box
                cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
                add_brick(color, long)
                mask = np.zeros_like(image)
                cv2.drawContours(mask, [contour], -1, 255, thickness=-2)
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                _, binary_image = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

                # Normalize the binary image to have values of 0 and 1
                binary_image_normalized = (binary_image / 255)
                binary_image_normalized = 1 - binary_image_normalized

                # Apply the mask to the original image to remove the contour
                result_image = cv2.bitwise_and(tmp, tmp, mask=binary_image_normalized.astype(np.uint8))
                tmp = result_image
                result_image = cv2.bitwise_and(tmp2, tmp2, mask=binary_image_normalized.astype(np.uint8))
                tmp2 = result_image
                if DEBUG:
                    # Add text next to the rectangle
                    area = cv2.contourArea(contour)
                    text = f"{area}"
                    cv2.putText(image, text, (box[1][0], box[1][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    print(f"long: {long:.2f} short: {short:.2f} area: {area:.2f} \n")

    return image


def Seperate_Bricks(contour, image, processed_img):

    mask = np.zeros_like(image)
    cv2.drawContours(mask, [contour], -1, 255, thickness=-2)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    binary_image = cv2.bitwise_and(processed_img, processed_img,mask=mask)
    # cv2.imshow('temporary image after remove', binary_image)

    Split_Contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return Split_Contours

def distance_transform(image):
    distance_transform = cv2.distanceTransform(image, cv2.DIST_L2, 3)

    normalized_distance_transform = cv2.normalize(distance_transform, None, 0, 255, cv2.NORM_MINMAX,
                                                  cv2.CV_8UC1)
    _, binary_image = cv2.threshold(normalized_distance_transform,
                                    0.2 * normalized_distance_transform.max(), 255,
                                    cv2.THRESH_BINARY)

    # Normalize the distance transform to visualize
    cv2.normalize(distance_transform, distance_transform, 0, 1, cv2.NORM_MINMAX)

    return binary_image
##//////////////////////////////////////////////////////////////////////////////////////////////////////////////// Camera ////////////////

def New_Measure(gmm, Mode = "mask"):

    # Create a queue to communicate between threads
    frame_queue = Queue()

    #takes a frame
    ret, frame = cap.read()

    # Start a new thread to process the frame
    frame_processing_thread = Thread(target=process_frame, args=(frame, frame_queue, gmm, Mode))
    frame_processing_thread.start()

    # Return the processed frame from the queue
    return frame_queue.get()


def process_frame(frame, frame_queue, gmm, Mode):
    # Call the function to measure LEGO brick size
    processed_frame = All_Gmm(frame, gmm, Mode)
    # Put the processed frame into the queue
    frame_queue.put(processed_frame)


##///////////////////////////////////////////////////// Eran ///////////////////////////////////////

def FindSides(box):
    long = np.linalg.norm(box[0] - box[1])
    long2 = np.linalg.norm(box[3] - box[2])
    short = np.linalg.norm(box[1] - box[2])
    short2 = np.linalg.norm(box[0] - box[3])
    long = (long + long2) / 2
    short = (short + short2) / 2
    if short > long:
        short, long = long, short
    return short, long