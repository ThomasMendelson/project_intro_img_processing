import sys
import os

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Now you can perform the import
from Stage1 import Brick


class LegoBuilding:
    def __init__(self, name, img_path, bricks):
        self.name = name
        self.img_path = img_path
        self.bricks = bricks


    def have_all_bricks(self,stock):
        for (color, long) in stock:
            if stock[(color, long)].count < self.bricks[(color, long)].count:
                return False
        return True

