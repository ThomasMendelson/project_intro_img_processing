
from copy import deepcopy
from Stage1 import Brick
from Stage2 import Building

DEBUG = True
IMG_long = 480
IMG_WIDTH = 640
# BUILD_SIZE = 800
long_side = [36, 47, 65]
longs = {2, 3, 4, 6}
colors = {"Yellow", "Black", "Red", "Blue", "White", "Orange", "Green"}
stock = {}
for color in colors:
    for long in longs:
        stock[(color, long)] = Brick.LegoBrick(color, long, 2)


mushroom = deepcopy(stock)
mushroom[("Blue", 4)].change_quantity(5)
mushroom[("Red", 4)].change_quantity(9)
mushroom[("Orange", 2)].change_quantity(1)
mushroom[("Orange", 3)].change_quantity(1)

house = deepcopy(stock)
house[("Red", 4)].change_quantity(7)
house[("Blue", 4)].change_quantity(1)
house[("Blue", 2)].change_quantity(1)
house[("Black", 4)].change_quantity(1)
house[("Yellow", 4)].change_quantity(9)

ballon = deepcopy(stock)
ballon[("Yellow", 4)].change_quantity(2)
ballon[("Yellow", 2)].change_quantity(1)
ballon[("Red", 2)].change_quantity(1)
ballon[("Red", 4)].change_quantity(22)

buildings = []
buildings.append(Building.LegoBuilding("mushroom", "../images/LegoLand_Logo-removebg-preview.png", mushroom))
buildings.append(Building.LegoBuilding("house", "../images/LegoLand_Logo-removebg-preview.png", house))
buildings.append(Building.LegoBuilding("ballon", "../images/LegoLand_Logo-removebg-preview.png", ballon))


Masks = {}
Masks["Green"] = ((45, 60, 60), (100, 255, 255))
Masks["Blue"] = ((100, 90, 90), (150, 255, 255))
Masks["Yellow"] = ((15, 75, 75), (60, 255, 255))
Masks["Black"] = ((55, 60, 0), (170, 110, 208))
Masks["Red"] = ((0, 40, 0), (4, 255, 255))
Masks["Red2"] = ((150, 40, 0), (180, 255, 255))
Masks["Orange"] = ((5, 5, 200), (20, 255, 255))
Masks["White"] = ((60, 0, 240), (255, 60, 255))



pkl_dict = {}
pkl_dict["Green"] = "Trained_Green.pkl"
pkl_dict["Blue"] = "Trained_Blue.pkl"
pkl_dict["Yellow"] = "Trained_Yellow.pkl"
pkl_dict["Black"] = "Trained_Black.pkl"
pkl_dict["Red"] = "Trained_Red.pkl"
pkl_dict["Orange"] = "Trained_Orange.pkl"
pkl_dict["White"] = "Trained_White.pkl"
pkl_dict["All1"] = "Trained_All.pkl"
pkl_dict["All2"] = "Trained_All2.pkl"