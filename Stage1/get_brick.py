import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__)) # Get the current directory (directory of gui.py)
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir)) # Get the parent directory (project directory)
sys.path.append(parent_dir) # Add the parent directory to the Python path
from utils import stock, long_side, DEBUG, Masks

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
    masked_image = perform_projective_transformation(masked_image)
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




def t():
    img = find_and_add_bricks(cv2.imread("test_get_size/Tomas15.jpg"))

    plt.figure()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

    for (color, long) in stock:
        print(stock[(color, long)].describe())

if __name__ == "_main_":
    t()
