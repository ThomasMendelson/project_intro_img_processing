import random
import tkinter as tk
from tkinter import PhotoImage
from PIL import Image, ImageTk
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__)) # Get the current directory (directory of gui.py)
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir)) # Get the parent directory (project directory)
sys.path.append(parent_dir) # Add the parent directory to the Python path

from utils import stock, IMG_HEIGHT, IMG_WIDTH, buildings, colors, longs
from Stage1.Brick import LegoBrick
from Stage1.Phase1 import All_Gmm
from Stage2.Building import LegoBuilding


def clear_window(window):
    # Destroy all widgets in the window
    for widget in window.winfo_children():
        widget.destroy()

def numpy_to_photoimage(numpy_array):
    # Convert NumPy array to PIL Image
    pil_image = Image.fromarray(numpy_array)

    # Convert PIL Image to Tkinter PhotoImage
    photo_image = ImageTk.PhotoImage(pil_image)

    return photo_image

def update_scan(window):
    clear_window(window)

    image = All_Gmm(f"../data base/Library/mix/MixColors_46.jpg")   # todo we need to reset stock
    image = numpy_to_photoimage(image)
    image_label = tk.Label(window, image=image, bg="gray")
    image_label.pack()
    # we need to update also the photo...............................................
    # color = random.choice(list(colors))
    # long = random.choice(list(longs))

    # stock[(color, long)].change_quantity(random.randint(0, 121))
    # stock.sort(key=lambda brick: brick.count, reverse=True)
    # # For demonstration, let's update the label text every 0.5 seconds
    # for i, brick in enumerate(stock):
    #     txt_label = tk.Label(window, text=brick.describe(), fg=brick.color, bg="gray", font=("Helvetica", 12, "bold"))
    #     txt_label.place(x=IMG_WIDTH + 10, y=i * 20)
    for i, (color, long) in enumerate(stock):
        brick = stock[(color, long)]
        txt_label = tk.Label(window, text=brick.describe(), fg=brick.color, bg="gray", font=("Helvetica", 8, "bold"))
        txt_label.place(x=IMG_WIDTH + 10, y=i * 15)

    # Schedule the next update after 0.5 seconds
    window.after(500, update_scan, window)


def build_step(bulding):
    print("in build steps")
    print(bulding.name)
    # to complite 2 stage////////////////////////////////////////////

    build_step_window = tk.Toplevel()
    build_step_window.title("Let build step by step")


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

        build_image2 = PhotoImage(file="../data base/Gui/build2_button.png")
        build_button2 = tk.Button(build_button_frame,
                                  command=lambda: (window.destroy(), build_step(construction_options[idx])),
                                  image=build_image2, borderwidth=0, highlightthickness=0, bg="white")
        build_button2.image = build_image2
        build_button2.grid(row=0, column=1)

        if idx == 0 and len(construction_options) > 1:
            next_image = PhotoImage(file="../data base/Gui/next_button.png")  # change to next
            next_button = tk.Button(build_button_frame,
                                    command=lambda: update_build(window, construction_options, idx + 1),
                                    image=next_image, borderwidth=0, highlightthickness=0, bg="white")
            next_image.image = next_image
            next_button.grid(row=0, column=2)
        elif idx == len(construction_options) - 1:
            prev_image = PhotoImage(file="../data base/Gui/prev_button.png")  # change to next
            prev_button = tk.Button(build_button_frame,
                                    command=lambda: update_build(window, construction_options, idx - 1),
                                    image=prev_image, borderwidth=0, highlightthickness=0, bg="white")
            prev_button.image = prev_image
            prev_button.grid(row=0, column=0)
        else:
            next_image = PhotoImage(file="../data base/Gui/next_button.png")  # change to next
            next_button = tk.Button(build_button_frame,
                                    command=lambda: update_build(window, construction_options, idx + 1),
                                    image=next_image, borderwidth=0, highlightthickness=0, bg="white")
            next_image.image = next_image
            next_button.grid(row=0, column=2)

            prev_image = PhotoImage(file="../data base/Gui/prev_button.png")  # change to next
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
    build_window = tk.Toplevel()
    build_window.title("Building...")
    build_window.attributes("-fullscreen", True)
    build_window.configure(bg="white")

    construction_options = []  # get list of opnional images to build
    #####################################################################################todo delete
    for color in colors:
        for long in longs:
            stock[(color, long)].change_quantity(100)
    ####################################################################################
    for building in buildings:
        if building.have_all_bricks(stock):
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