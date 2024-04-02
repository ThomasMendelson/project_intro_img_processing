class LegoBrick:
    def __init__(self, color, width, height, length, shape = "Rectangular"):
        self.color = color
        self.width = width
        self.height = height
        self.length = length
        self.shape = shape

    def describe(self):
        print(f"This Lego brick is {self.color}, {self.width}x{self.height}x{self.length}, and {self.shape}.")

    def stack(self):
        print("Stacking the Lego brick.")


Masks = {}
Masks["Green"] = ((50, 0, 0), (90, 255, 120))
Masks["Blue"] = ((105, 0, 0), (145, 255, 125))
Masks["Yellow"] = ((15, 100, 150), (50, 255, 200))
#Masks["Black"] = ((0, 0, 0), (100, 180, 80))
Masks["Red"] = ((0, 100, 150), (9, 255, 190))
Masks["Orange"] = ((10, 100, 120), (17, 255, 255))

color_dict = {}
color_dict["Green"] = (0, 255, 0)
color_dict["Blue"] = (0, 0, 255)
color_dict["Yellow"] = (255, 255, 0)
color_dict["Black"] = (0, 0, 0)
color_dict["Red"] = (255, 0, 0)
color_dict["Orange"] = (255, 165, 0)


# lower_red = np.array([0, 100, 100])
# upper_red = np.array([10, 255, 255])