import cv2
import random

scale = 0.5
circle = []
counter = 0
counter2 = 0
point1 = []
point2 = []
myPoints = []
myColor = []

def mousePoints(event, x, y, flags, params):
    global counter, point1, point2, counter2, circle, myColor
    if event == cv2.EVENT_LBUTTONDOWN:
        if counter == 0:
            point1 = int(x//scale), int(y//scale);
            counter += 1
            myColor = (random.randint(0, 2) * 200, random.randint(0, 2) * 200, random.randint(0, 2) * 200)
        elif counter == 1:
            point2 = int(x//scale), int(y//scale)
            type = input("Enter Type: ")
            name = input("Enter Name: ")
            myPoints.append([point1, point2, type, name])
            counter = 0
        circle.append([x, y, myColor])
        counter2 += 1

img = cv2.imread('C:\\Users\\Shakib\\Documents\\Python\\pythonProject\\Scanned\\myImage.jpg')
img = cv2.resize(img, (0, 0), None, scale, scale)

while True:
    #to display points
    for x, y, color in circle:
        cv2.circle(img, (x, y), 3, color, cv2.FILLED)
    cv2.imshow("Original Image", img)
    cv2.setMouseCallback("Original Image", mousePoints)
    if cv2.waitKey(0):
        print(myPoints)
        break

#and 0xFF == ord('s')