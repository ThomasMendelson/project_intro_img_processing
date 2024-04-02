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
Masks["Blue"] = ((110, 0, 0), (150, 255, 120))