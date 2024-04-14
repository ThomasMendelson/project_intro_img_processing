import tkinter as tk
from tkinter import PhotoImage
from Stage1 import Brick
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np




stock = []
stock.append(Brick.LegoBrick("Red",1,2,3))
stock.append(Brick.LegoBrick("Red",2,4,3))
stock.append(Brick.LegoBrick("Blue",4,2,2))
stock.append(Brick.LegoBrick("white",1,1,3))


import cv2
import numpy as np

def findPeices(image_path):
    # Load the image
    image = cv2.imread(image_path)
    # Define the desired width and height for the resized image
    desired_width = 800
    desired_height = 700

    # Resize the image
    image = cv2.resize(image, (desired_width, desired_height))
    cv2.imshow('Canny', image)
    cv2.waitKey(0)

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define a broader HSV range for black, capturing variations
    black_lower = Brick.Masks["Blue"][0]
    black_upper = Brick.Masks["Blue"][1]

    # Create the mask for black color
    mask = cv2.inRange(hsv, black_lower, black_upper)

    # Apply morphological operations to refine the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Close small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove small noise

    # Apply the mask to the original image
    image = cv2.bitwise_and(image, image, mask=mask)
    cv2.imshow('Canny', image)
    cv2.waitKey(0)


    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    number = 120
    edges = cv2.Canny(gray, number*0.3, number, apertureSize=3)
    print("Value is ",number)
    cv2.imshow('Canny', edges)
    cv2.waitKey(0)

    # Apply Hough transform to detect lines
    lines = cv2.HoughLinesP(edges, 0.5, np.pi / 360, threshold=10, minLineLength=45, maxLineGap=13)
    print(lines)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Display the result
    cv2.imshow('Lego pieces detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def start_game():
    # Add your code to start the Lego Land game here
    print("Lego Land game started!")

def show_stock():
    # Add your code to start the Lego Land game here
    for b in stock:
        b.describe()

# find pices
findPeices("bricks.jpg")


