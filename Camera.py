import cv2
import time as time
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__)) # Get the current directory (directory of gui.py)
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir)) # Get the parent directory (project directory)
sys.path.append(parent_dir) # Add the parent directory to the Python path
from utils import cap, cap2


def capture_photos(cap_idx):
    if cap_idx == 1:
        ret, frame = cap.read()
        return frame
    else:
        ret, frame = cap2.read()
        return frame