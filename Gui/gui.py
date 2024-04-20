import tkinter as tk
from tkinter import PhotoImage
from PIL import Image, ImageTk
import sys
import os
import cv2
from time import time, sleep
import numpy as np
from copy import deepcopy

current_dir = os.path.dirname(os.path.abspath(__file__))  # Get the current directory (directory of gui.py)
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))  # Get the parent directory (project directory)
sys.path.append(parent_dir)  # Add the parent directory to the Python path

from utils import stock, IMG_HEIGHT, IMG_WIDTH, buildings, colors, longs, clear_stock, start_mes, time0, old_image, \
    count, cap2, stock_for_show, DEBUG, DEBUG2
from Stage1.Training import gmm, load_model
from Stage1.Brick import LegoBrick
from Stage1.Phase1 import New_Measure
from Camera import capture_photos
from Stage2.Building import LegoBuilding
from Stage2.Phase2 import calc_score, find_close_up_image
from Stage1.stopwatch import Stopwatch

global stock_for_show
stock_for_show = deepcopy(stock)


def clear_window(window):
    # Destroy all widgets in the window
    for widget in window.winfo_children():
        widget.destroy()


def numpy_to_photoimage(img):
    # Convert NumPy array to PIL Image
    pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Convert PIL Image to Tkinter PhotoImage
    photo_image = ImageTk.PhotoImage(pil_image)

    return photo_image

def update_scan(window):
    global stock_for_show
    image = New_Measure(gmm)
    sleep(0.01)
    new_image = New_Measure(gmm)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    clear_stock()

    if np.mean((gray_image - gray_new_image) ** 2) > 12:
        clear_window(window)
        new_image = New_Measure(gmm)
        rotated_image = np.rot90(new_image, 2)
        new_image = numpy_to_photoimage(rotated_image)
        image_label = tk.Label(window, image=new_image, bg="gray")
        image_label.image = new_image
        image_label.pack(side=tk.LEFT)

        for i, (color, long) in enumerate(stock):
            brick = stock[(color, long)]
            txt_label = tk.Label(window, text=brick.describe(), fg=brick.color, bg="gray",
                                 font=("Helvetica", 10, "bold"))
            txt_label.place(x=IMG_WIDTH + 10, y=i * 17)
            stock_for_show[(color, long)] = deepcopy(brick)
        clear_stock()

    # Schedule the next update after 0.5 seconds
    window.after(50, update_scan, window)


def update_scan_old(window):
    time0 = time()
    clear_window(window)
    clear_stock()

    image = New_Measure(gmm)
    image = numpy_to_photoimage(image)
    image_label = tk.Label(window, image=image, bg="gray")
    image_label.image = image
    image_label.pack(side=tk.LEFT)

    for i, (color, long) in enumerate(stock):
        brick = stock[(color, long)]
        txt_label = tk.Label(window, text=brick.describe(), fg=brick.color, bg="gray", font=("Helvetica", 10, "bold"))
        txt_label.place(x=IMG_WIDTH + 10, y=i * 17)

    clear_window(window)

    # Schedule the next update after 0.5 seconds
    window.after(50, update_scan_old, window)


def update_build_step(window, building, started, stopwatch, camera_image, score_label, success_label):
    # Create frames to organize widgets
    stopwatch_frame = tk.Frame(window)
    stopwatch_frame.pack()

    middle_frame = tk.Frame(window)
    middle_frame.pack()

    bottom_frame = tk.Frame(window)
    bottom_frame.pack()

    success_frame = tk.Frame(window)
    success_frame.pack()

    image = capture_photos(2)
    # Resize the image to 400x400
    image_for_show = cv2.resize(image, (400, 400))
    rotated_image = np.rot90(image_for_show, 2)
    new_photo_image = numpy_to_photoimage(rotated_image)
    camera_image.config(image=new_photo_image)
    camera_image.image = new_photo_image  # Keep a reference to prevent garbage collection

    if stopwatch.is_running:
        success_label.config(text="")
        # Update the stopwatch label
        elapsed_seconds = int(stopwatch.elapsed_time)
        hours = elapsed_seconds // 3600
        minutes = (elapsed_seconds % 3600) // 60
        seconds = elapsed_seconds % 60
        stopwatch.label.config(text=f"{hours:02d}:{minutes:02d}:{seconds:02d}")
        txt, score = calc_score(image, building)
        # Repack the frames to ensure correct positioning
        stopwatch_frame.pack(side=tk.LEFT)
        middle_frame.pack(side=tk.LEFT)
        bottom_frame.pack(side=tk.LEFT)
        success_frame.pack(side=tk.LEFT)
        # Update the score label
        if DEBUG2:
            score_label.config(text="Score: " + str(score) + "\n" + txt, font=("Arial", 16))
        else:
            score_label.config(text=txt, font=("Arial", 16))
        # else:
        #     score_label.config(text="Score: " + str(score), font=("Arial", 16))
        if "Good Job" in str(txt):
            success_label.config(text="You did it ! Good Job", font=("Arial", 26), fg="green")
            # Stop the stopwatch
            stopwatch.start_stop()

    window.after(100, update_build_step, window, building, started, stopwatch, camera_image, score_label, success_label)




def build_step(building):
    # L - label, B - Button, I - Image
    # -----------------------
    # |  L.time   L.score    |
    # | I.build     I.camera |
    # |        B.start       |
    # ------------------------
    started = [False]
    build_step_window = tk.Toplevel()
    build_step_window.title("Let build")
    build_step_window.geometry("800x800")

    # Create frames to organize widgets
    stopwatch_frame = tk.Frame(build_step_window)
    stopwatch_frame.pack()

    middle_frame = tk.Frame(build_step_window)
    middle_frame.pack()

    bottom_frame = tk.Frame(build_step_window)
    bottom_frame.pack()

    success_frame = tk.Frame(build_step_window)
    success_frame.pack()

    # Create an instance of the Stopwatch class inside the frame
    stopwatch = Stopwatch(stopwatch_frame)

    # Add images to the middle frame
    image_left = PhotoImage(file=building.img_path)
    build_image = tk.Label(middle_frame, image=image_left, text="Build Image")
    build_image.image = image_left
    build_image.pack(side=tk.LEFT)

    camera_image = tk.Label(middle_frame, text="Camera Image")
    # get image from camera
    camera_image.pack(side=tk.RIGHT)

    success_label = tk.Label(success_frame, text="Waiting for you to start", font=("Arial", 16))
    success_label.pack(side=tk.RIGHT)

    score_label = tk.Label(bottom_frame, text="", font=("Arial", 16))
    score_label.pack(side=tk.RIGHT)

    # Call the update function of the stopwatch
    stopwatch.update()

    # Call the update_build_step function with initial parameters
    update_build_step(build_step_window, building, started, stopwatch, camera_image, score_label, success_label)


def update_build(window, construction_options, idx=0):
    clear_window(window)
    # Create a frame for buttons

    if not construction_options:  # empty
        txt_label = tk.Label(window, text="There are no options", font=("Helvetica", 12, "bold"))
        txt_label.pack()
    else:
        building_image_path = construction_options[idx].img_path
        building_image = Image.open(building_image_path)
        building_image_tk = ImageTk.PhotoImage(building_image)
        building_image_label = tk.Label(window, image=building_image_tk, bg="white")
        building_image_label.image = building_image_tk  # Keep a reference to the image
        building_image_label.pack(side=tk.TOP)

        build_button_frame = tk.Frame(window)
        build_button_frame.pack(side=tk.BOTTOM)

        build_image2 = PhotoImage(file="../Data Base/Gui/build2_button.png")
        build_button2 = tk.Button(build_button_frame,
                                  command=lambda: (window.destroy(), build_step(construction_options[idx])),
                                  image=build_image2, borderwidth=0, highlightthickness=0, bg="white")
        build_button2.image = build_image2
        build_button2.grid(row=0, column=1)

        if idx == 0 and len(construction_options) > 1:
            next_image = PhotoImage(file="../Data Base/Gui/next_button.png")  # change to next
            next_button = tk.Button(build_button_frame,
                                    command=lambda: update_build(window, construction_options, idx + 1),
                                    image=next_image, borderwidth=0, highlightthickness=0, bg="white")
            next_image.image = next_image
            next_button.grid(row=0, column=2)
        elif idx != 0 and idx == len(construction_options) - 1:
            prev_image = PhotoImage(file="../Data Base/Gui/prev_button.png")  # change to next
            prev_button = tk.Button(build_button_frame,
                                    command=lambda: update_build(window, construction_options, idx - 1),
                                    image=prev_image, borderwidth=0, highlightthickness=0, bg="white")
            prev_button.image = prev_image
            prev_button.grid(row=0, column=0)
        elif idx != 0:
            next_image = PhotoImage(file="../Data Base/Gui/next_button.png")  # change to next
            next_button = tk.Button(build_button_frame,
                                    command=lambda: update_build(window, construction_options, idx + 1),
                                    image=next_image, borderwidth=0, highlightthickness=0, bg="white")
            next_image.image = next_image
            next_button.grid(row=0, column=2)

            prev_image = PhotoImage(file="../Data Base/Gui/prev_button.png")  # change to next
            prev_button = tk.Button(build_button_frame,
                                    command=lambda: update_build(window, construction_options, idx - 1),
                                    image=prev_image, borderwidth=0, highlightthickness=0, bg="white")
            prev_button.image = prev_image
            prev_button.grid(row=0, column=0)


def scan():
    window_width = int(IMG_WIDTH * 1.5)

    # Create a new window for scanning
    scan_window = tk.Toplevel()
    scan_window.title("Scanning...")
    scan_window.geometry(f"{window_width}x{IMG_HEIGHT}")
    scan_window.configure(bg="gray")
    update_scan(scan_window)
    scan_window.mainloop()


def build():
    global stock_for_show
    build_window = tk.Toplevel()
    build_window.title("Building...")
    build_window.attributes("-fullscreen", True)
    build_window.configure(bg="white")

    construction_options = []  # get list of opnional images to build
    # #####################################################################################todo delete
    # for color in colors:
    #     for long in longs:
    #         stock[(color, long)].change_quantity(100)
    # ####################################################################################
    for i, (color, long) in enumerate(stock_for_show):
        brick = stock_for_show[(color, long)]
    for building in buildings:
        if building.have_all_bricks(stock_for_show):
            construction_options.append(building)
    update_build(build_window, construction_options)
    build_window.mainloop()


# Create the main window
root = tk.Tk()
root.title("LEGO LAND GAME")
root.configure(bg="white")
root.attributes("-fullscreen", True)
# Load the image
image_path = "../data base/Gui/LegoLand_Logo-removebg-preview.png"
image = PhotoImage(file=image_path)

scan_image = PhotoImage(file="../data base/Gui/scan_button.png")
build_image = PhotoImage(file="../data base/Gui/build_button.png")

# Display the image
image_label = tk.Label(root, image=image, bg="white")
image_label.pack()

# Create a frame for buttons
button_frame = tk.Frame(root)
button_frame.pack()

# Create the buttons with round edges
scan_button = tk.Button(button_frame, command=scan, image=scan_image, borderwidth=0, highlightthickness=0)
scan_button.pack(side=tk.LEFT)

build_button = tk.Button(button_frame, command=build, image=build_image, borderwidth=0, highlightthickness=0)
build_button.pack(side=tk.LEFT)

# Run the Tkinter event loop
root.mainloop()
