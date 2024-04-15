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


