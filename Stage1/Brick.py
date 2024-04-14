class LegoBrick:
    def __init__(self, color, long, short, count=0):
        self.color = color
        self.long = long
        self.short = short
        self.count = count

    def describe(self):
        return f"{self.color:8s}, {self.long:3d} x{self.short:3d} we found: {self.count:2d}     "

    def stack(self):
        print("Stacking the Lego brick.")

    def change_quantity(self, quantity):
        self.count = quantity

    def add_one(self):
        self.count += 1




color_dict = {}
color_dict["Green"] = (0, 255, 0)
color_dict["Blue"] = (0, 0, 255)
color_dict["Yellow"] = (255, 255, 0)
color_dict["Black"] = (0, 0, 0)
color_dict["Red"] = (255, 0, 0)
color_dict["Orange"] = (255, 165, 0)

# lower_red = np.array([0, 100, 100])
# upper_red = np.array([10, 255, 255])
