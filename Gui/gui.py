import tkinter as tk
from tkinter import PhotoImage

from ..Stage1 import Brick

# Now you can use the Brick class

# import sys
# sys.path.append("../Stage1")
# import Brick.py as Brick
def scan():
    print("Button scan clicked")
    stock = []
    stock.append(Brick.LegoBrick("Red", 1, 2, 3))
    stock.append(Brick.LegoBrick("Red", 2, 4, 3))
    stock.append(Brick.LegoBrick("Blue", 4, 2, 2))
    stock.append(Brick.LegoBrick("white", 1, 1, 3))


def build():
    print("Button build clicked")


# Create the main window
root = tk.Tk()
root.title("LEGO LAND GAME")
root.configure(bg="white")

# Load the image
image_path = "../images/LegoLand_Logo-removebg-preview.png"
image = PhotoImage(file=image_path)

scan_image = PhotoImage(file="../images/scan_button.png")
# build_image = PhotoImage(file="../images/build_button.png")

# Display the image
image_label = tk.Label(root, image=image, bg="white")
image_label.pack()

# Create a frame for buttons
button_frame = tk.Frame(root)
button_frame.pack()

# Create the buttons with round edges
button1 = tk.Button(button_frame, command=scan, image=scan_image, borderwidth=0, highlightthickness=0)
button1.pack(side=tk.LEFT)

# button2 = tk.Button(button_frame, command=build, image=build_image, borderwidth=0, highlightthickness=0)
# button2.pack(side=tk.LEFT)

# Run the Tkinter event loop
root.mainloop()



#
# from kivy.app import App
# from kivy.uix.image import Image
# from kivy.uix.button import Button
# from kivy.uix.boxlayout import BoxLayout
# from kivy.uix.label import Label
#
# class LegoLandApp(App):
#     def scan(self):
#         print("Button scan clicked")
#
#     def build_the_lego(self):
#         print("Button build clicked")
#
#     def build(self):
#         # Create the main layout
#         main_layout = BoxLayout(orientation='vertical')
#
#         # Load the image
#         image = Image(source="LegoLand_Logo-removebg-preview.png", size_hint_y=None, height=400)
#         main_layout.add_widget(image)
#
#         # Create a label for button descriptions
#         label = Label(text="Scan", size_hint_y=None, height=50)
#         main_layout.add_widget(label)
#
#         # Create the scan button
#         scan_button = Button(background_normal="scan_button.png", size_hint_y=None)
#         scan_button.bind(on_press=self.scan)
#         main_layout.add_widget(scan_button)
#
#         # Create a label for button descriptions
#         label = Label(text="Build", size_hint_y=None, height=50)
#         main_layout.add_widget(label)
#
#         # Create the build button
#         build_button = Button(background_normal="build_button.png", size_hint_y=None)
#         build_button.bind(on_press=self.build_the_lego)
#         main_layout.add_widget(build_button)
#
#         return main_layout
#
# if __name__ == '__main__':
#     LegoLandApp().run()


# from kivy.app import App
# from kivy.uix.button import Button
# from kivy.uix.boxlayout import BoxLayout
#
#
# class LegoLandApp(App):
#     def scan(self, instance):
#         print("Scan button clicked")
#
#     def build(self):
#         # Create a vertical box layout
#         layout = BoxLayout(orientation='vertical')
#
#         # Create a scan button
#         scan_button = Button(text="Scan", size_hint=(None, None), size=(200, 50))
#         scan_button.bind(on_press=self.scan)
#         layout.add_widget(scan_button)
#
#         # Create a build button
#         build_button = Button(text="Build", size_hint=(None, None), size=(200, 50))
#         build_button.bind(on_press=self.build)
#         layout.add_widget(build_button)
#
#         return layout
#
#
# if __name__ == '__main__':
#     LegoLandApp().run()


# from kivy.app import App
# from kivy.uix.button import Button
# from kivy.uix.boxlayout import BoxLayout
# from kivy.graphics import RoundedRectangle
#
#
# class RoundedButton(Button):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.background_color = (0, 0, 1, 1)  # blue color
#         self.bind(size=self.update_canvas)
#
#     def update_canvas(self, *args):
#         self.canvas.before.clear()
#         with self.canvas.before:
#             RoundedRectangle(pos=self.pos, size=self.size, radius=[10, 10, 10, 10])
#
#
# class LegoLandApp(App):
#     def scan(self, instance):
#         print("Scan button clicked")
#
#     def build(self):
#         # Create a vertical box layout
#         layout = BoxLayout(orientation='vertical')
#
#         # Create a scan button with rounded corners
#         scan_button = RoundedButton(text="Scan", size_hint=(None, None), size=(200, 50))
#         scan_button.bind(on_press=self.scan)
#         layout.add_widget(scan_button)
#
#         return layout
#
#
# if __name__ == '__main__':
#     LegoLandApp().run()


