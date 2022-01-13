import cv2
import numpy as np
import pytesseract
import os
import processImage



#-------delete file-------
if os.path.exists('DataOutput.csv'):
    os.remove('DataOutput.csv')

per = 25
#-------field to crop-------
roi = [[(1706, 272), (2090, 332), 'text', 'Reg No.'],
       [(684, 588), (2080, 672), 'text', 'Name'],
       [(542, 708), (2082, 792), 'text', 'Father Name'],
       [(224, 814), (2082, 900), 'text', 'Mother Name'],
       [(190, 928), (2082, 1016), 'text', 'College Name'],
       [(420, 1058), (1152, 1148), 'text', 'Area'],
       [(1282, 1062), (1672, 1150), 'text', 'Roll No.'],
       [(1228, 1174), (1672, 1260), 'text', 'Group'],
       [(276, 1298), (382, 1350), 'text', 'GPA']]


pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

imgQ = cv2.imread('temp\\query.jpg')
h, w, c = imgQ.shape

orb = cv2.ORB_create(1000)
kp1, des1 = orb.detectAndCompute(imgQ, None)

#-------store user image-------
path = 'image'

#-------list of images-------
myPicList = os.listdir(path)
print(myPicList)

for j, y in enumerate(myPicList):
    img = cv2.imread(path + "/" + y)
    imgScan = processImage.cropImage(img)

    myData = []

    imgShow = imgScan.copy()
    imgMask = np.zeros_like(imgShow)

    for x, r in enumerate(roi):
        cv2.rectangle(imgMask, (r[0][0], r[0][1]), (r[1][0], r[1][1]), (0, 255, 0), cv2.FILLED)
        imgShow = cv2.addWeighted(imgShow, 0.99, imgMask, 0.1, 0)

#-------crop info-------
        imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]
        # cv2.imshow(y, imgShow)

#-------extract data from crop image-------

        #change mad to md
        data = pytesseract.image_to_string(imgCrop)
        data = str(data)
        data = data.replace('Mad', 'Md')
        print(f'{r[3]} : {data}')
        myData.append(data)

#-------create and write data in csv file-------
    with open('DataOutput.csv', 'a+') as f:
        for data in myData:
            data = str(data)
            data = data.replace('Mad', 'Md')
            f.write((data+', '))
        f.write('\n')

    imgShow = cv2.resize(imgShow, (w // 3, h // 3))
    cv2.imshow(y, imgShow)
    cv2.waitKey(0)
    print(myData)


