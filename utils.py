from copy import deepcopy
from Stage1 import Brick
from Stage2 import Building
import cv2


DEBUG = True
DEBUG2 = True
IMG_HEIGHT = 480
IMG_WIDTH = 640
# BUILD_SIZE = 800
long_side = [40, 47, 65]
longs = {2, 3, 4, 6}
colors = ("Yellow", "Black", "Red", "Blue", "White", "Orange", "Green")
stock = {}
for color in colors:
    for long in longs:
        stock[(color, long)] = Brick.LegoBrick(color, long, 2)
stock_for_show = deepcopy(stock)
start_mes = True
time0 = 0
old_image = 0
count = 3


House = deepcopy(stock)
House[("Red", 2)].change_quantity(1)
House[("Red", 4)].change_quantity(8)
House[("Blue", 4)].change_quantity(2)
House[("Black", 4)].change_quantity(3)
House[("Yellow", 2)].change_quantity(4)
House[("Yellow", 4)].change_quantity(11)

alien = deepcopy(stock)
alien[("Black", 2)].change_quantity(2)
alien[("Black", 4)].change_quantity(2)
alien[("Red", 2)].change_quantity(2)
alien[("Red", 4)].change_quantity(2)
alien[("Yellow", 2)].change_quantity(4)
alien[("Yellow", 3)].change_quantity(2)
alien[("Yellow", 4)].change_quantity(12)
alien[("Yellow", 6)].change_quantity(1)
alien[("Blue", 2)].change_quantity(1)
alien[("Blue", 4)].change_quantity(9)
alien[("Blue", 6)].change_quantity(1)

israeli_road = deepcopy(stock)
israeli_road[("White", 2)].change_quantity(1)
israeli_road[("White", 3)].change_quantity(1)
israeli_road[("White", 4)].change_quantity(1)
israeli_road[("Blue", 2)].change_quantity(1)
israeli_road[("Blue", 3)].change_quantity(1)
israeli_road[("Blue", 4)].change_quantity(1)
israeli_road[("Orange", 3)].change_quantity(3)

Tammy = deepcopy(stock)
Tammy[("Yellow", 2)].change_quantity(3)
Tammy[("Yellow", 4)].change_quantity(4)
Tammy[("Black", 4)].change_quantity(2)
Tammy[("Blue", 2)].change_quantity(1)
Tammy[("Blue", 4)].change_quantity(5)
Tammy[("Red", 2)].change_quantity(3)
Tammy[("Red", 4)].change_quantity(3)
Tammy[("Orange", 3)].change_quantity(1)

Balloon = deepcopy(stock)
Balloon[("White", 4)].change_quantity(1)
Balloon[("White", 3)].change_quantity(1)
Balloon[("White", 2)].change_quantity(2)
Balloon[("Red", 2)].change_quantity(3)
Balloon[("Red", 3)].change_quantity(2)
Balloon[("Red", 4)].change_quantity(8)

Smily = deepcopy(stock)
Smily[("Yellow", 6)].change_quantity(1)
Smily[("Yellow", 4)].change_quantity(10)
Smily[("Yellow", 3)].change_quantity(2)
Smily[("Yellow", 2)].change_quantity(2)
Smily[("Red", 2)].change_quantity(2)
Smily[("Red", 4)].change_quantity(1)
Smily[("Black", 4)].change_quantity(2)

Fire = deepcopy(stock)
Fire[("Orange", 3)].change_quantity(3)
Fire[("Yellow", 2)].change_quantity(1)
Fire[("Red", 2)].change_quantity(1)
Fire[("Red", 4)].change_quantity(7)

Man = deepcopy(stock)
Man[("Black", 4)].change_quantity(1)
Man[("Red", 2)].change_quantity(1)
Man[("Red", 4)].change_quantity(11)

Heart = deepcopy(stock)
Heart[("Red", 2)].change_quantity(1)
Heart[("Red", 4)].change_quantity(11)


Pyramid = deepcopy(stock)
Pyramid[("Yellow", 2)].change_quantity(1)
Pyramid[("Yellow", 4)].change_quantity(1)
Pyramid[("Yellow", 6)].change_quantity(1)


JapanFlag = deepcopy(stock)
JapanFlag[("Black", 2)].change_quantity(2)
JapanFlag[("Black", 4)].change_quantity(1)
JapanFlag[("Black", 6)].change_quantity(1)
JapanFlag[("Red", 2)].change_quantity(2)
JapanFlag[("Red", 4)].change_quantity(6)
JapanFlag[("White", 2)].change_quantity(1)

Pencil = deepcopy(stock)
Pencil[("Black", 2)].change_quantity(1)
Pencil[("Yellow", 2)].change_quantity(1)
Pencil[("Yellow", 4)].change_quantity(3)
Pencil[("Black", 6)].change_quantity(1)
Pencil[("Red", 4)].change_quantity(1)
Pencil[("White", 4)].change_quantity(2)

Invader = deepcopy(stock)
Invader[("Yellow", 2)].change_quantity(4)
Invader[("Yellow", 4)].change_quantity(4)
Invader[("Red", 2)].change_quantity(2)
Invader[("Red", 3)].change_quantity(4)
Invader[("Red", 4)].change_quantity(13)

USAFlag = deepcopy(stock)
USAFlag[("Black", 4)].change_quantity(3)
USAFlag[("Black", 6)].change_quantity(1)
USAFlag[("White", 2)].change_quantity(3)
USAFlag[("White", 4)].change_quantity(3)
USAFlag[("White", 6)].change_quantity(1)
USAFlag[("Red", 4)].change_quantity(8)
USAFlag[("Blue", 2)].change_quantity(2)

BlueMonster = deepcopy(stock)
BlueMonster[("Blue", 2)].change_quantity(3)
BlueMonster[("Blue", 4)].change_quantity(9)
BlueMonster[("Blue", 6)].change_quantity(1)
BlueMonster[("White", 2)].change_quantity(2)

RedMonster = deepcopy(stock)
RedMonster[("Red", 2)].change_quantity(1)
RedMonster[("Red", 4)].change_quantity(12)
RedMonster[("White", 2)].change_quantity(2)

Pacman = deepcopy(stock)
Pacman[("Yellow", 2)].change_quantity(1)
Pacman[("Yellow", 4)].change_quantity(12)
Pacman[("Black", 2)].change_quantity(2)

O = deepcopy(stock)
O[("White", 2)].change_quantity(3)
O[("White", 4)].change_quantity(3)
O[("White", 6)].change_quantity(1)

L = deepcopy(stock)
L[("Blue", 4)].change_quantity(3)
L[("Blue", 6)].change_quantity(1)

G = deepcopy(stock)
G[("Yellow", 2)].change_quantity(1)
G[("Yellow", 3)].change_quantity(2)
G[("Yellow", 4)].change_quantity(5)

E = deepcopy(stock)
E[("Red", 4)].change_quantity(7)
E[("Red", 3)].change_quantity(1)


Cyborg = deepcopy(stock)
Cyborg[("Blue", 4)].change_quantity(5)
Cyborg[("Orange", 3)].change_quantity(2)
Cyborg[("Black", 2)].change_quantity(2)
Cyborg[("Black", 4)].change_quantity(2)
Cyborg[("Yellow", 2)].change_quantity(4)
Cyborg[("Yellow", 3)].change_quantity(2)
Cyborg[("Yellow", 4)].change_quantity(13)
Cyborg[("Red", 4)].change_quantity(13)

Car = deepcopy(stock)
Car[("Orange", 3)].change_quantity(1)
Car[("Black", 2)].change_quantity(2)
Car[("Black", 4)].change_quantity(3)
Car[("Yellow", 4)].change_quantity(1)
Car[("Red", 4)].change_quantity(15)
Car[("Red", 3)].change_quantity(4)
Car[("Red", 2)].change_quantity(3)

Mushroom = deepcopy(stock)
Mushroom[("White", 2)].change_quantity(3)
Mushroom[("White", 4)].change_quantity(2)
Mushroom[("White", 6)].change_quantity(1)
Mushroom[("Red", 4)].change_quantity(9)
Mushroom[("Red", 3)].change_quantity(2)
Mushroom[("Red", 2)].change_quantity(1)
Mushroom[("Orange", 3)].change_quantity(2)

RedCross = deepcopy(stock)
RedCross[("Red", 4)].change_quantity(4)
RedCross[("Red", 2)].change_quantity(1)

buildings = []
buildings.append(Building.LegoBuilding("Smily", "../Data Base/Gui/Smily.png", Smily))
buildings.append(Building.LegoBuilding("Fire", "../Data Base/Gui/Fire.png", Fire))
buildings.append(Building.LegoBuilding("Heart", "../Data Base/Gui/Heart.png", Heart))
# buildings.append(Building.LegoBuilding("Pyramid", "../Data Base/Gui/Pyramid.png", Pyramid))
# buildings.append(Building.LegoBuilding("JapanFlag", "../Data Base/Gui/JapanFlag.png", JapanFlag))
buildings.append(Building.LegoBuilding("Pencil", "../Data Base/Gui/Pencil.png", Pencil))
buildings.append(Building.LegoBuilding("Invader", "../Data Base/Gui/Invader.png", Invader))
buildings.append(Building.LegoBuilding("USAFlag", "../Data Base/Gui/USAFlag.png", USAFlag))
buildings.append(Building.LegoBuilding("BlueMonster", "../Data Base/Gui/BlueMonster.png", BlueMonster))
buildings.append(Building.LegoBuilding("RedMonster", "../Data Base/Gui/RedMonster.png", RedMonster))
buildings.append(Building.LegoBuilding("Mushroom", "../Data Base/Gui/Mushroom.png", Mushroom))
buildings.append(Building.LegoBuilding("L", "../Data Base/Gui/L.png", L))
buildings.append(Building.LegoBuilding("E", "../Data Base/Gui/E.png", E))
buildings.append(Building.LegoBuilding("G", "../Data Base/Gui/G.png", G))
buildings.append(Building.LegoBuilding("O", "../Data Base/Gui/O.png", O))
buildings.append(Building.LegoBuilding("House", "../Data Base/Gui/House.png", House))
buildings.append(Building.LegoBuilding("Cyborg", "../Data Base/Gui/Cyborg.png", Cyborg))
buildings.append(Building.LegoBuilding("Balloon", "../Data Base/Gui/Balloon.png", Balloon))
buildings.append(Building.LegoBuilding("Alien", "../Data Base/Gui/Alien.png", alien))
buildings.append(Building.LegoBuilding("Car", "../Data Base/Gui/Car.png", Car))
buildings.append(Building.LegoBuilding("IR", "../Data Base/Gui/IR.png", israeli_road))
buildings.append(Building.LegoBuilding("Tammy", "../Data Base/Gui/Tammy.png", Tammy))
buildings.append(Building.LegoBuilding("RedCross", "../Data Base/Gui/RedCross.png", RedCross))



Masks = {}
Masks["Green"] = ((45, 60, 60), (100, 255, 255))
Masks["Blue"] = ((100, 90, 90), (150, 255, 255))
Masks["Yellow"] = ((21, 70, 70), (60, 255, 255))
Masks["Black"] = ((25, 15, 0), (170, 110, 80))
Masks["Red"] = ((0, 40, 0), (4, 255, 255))
Masks["Red2"] = ((150, 40, 0), (180, 255, 255))
Masks["Orange"] = ((5, 5, 200), (20, 255, 255))
Masks["White"] = ((60, 0, 225), (100, 50, 255))
Masks["White2"] = ((0, 200, 80), (120, 255, 255))



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


def clear_stock():
    for (color, long) in stock:
        stock[(color, long)].change_quantity(0)

# Initialize webcam
cap = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(0)