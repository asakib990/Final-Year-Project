import cv2
import numpy as np
import pytesseract
import os

per = 25
#-------field to crop-------
roi = [[(1682, 312), (2050, 362), 'text', 'Reg No'],
       [(710, 598), (2028, 674), 'text', 'Name'],
       [(580, 706), (2030, 782), 'text', "Father's Name"],
       [(278, 798), (2022, 874), 'text', "Mother's Name"],
       [(242, 904), (2022, 978), 'text', "College's Name"],
       [(456, 1014), (1144, 1106), 'text', 'Area'],
       [(1272, 1018), (1638, 1104), 'text', 'Roll No.'],
       [(1234, 1124), (1650, 1202), 'text', 'Group']]


pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

imgQ = cv2.imread('temp\\query.jpg')
h, w, c = imgQ.shape
#imgQ = cv2.resize(imgQ, (w//3, h//3))

orb = cv2.ORB_create(1000)
kp1, des1 = orb.detectAndCompute(imgQ, None)
#imgKp1 = cv2.drawKeypoints(imgQ, kp1, None)

#-------store user image-------
path = 'temp'

#-------list of images-------
myPicList = os.listdir(path)
print(myPicList)

for j, y in enumerate(myPicList):
    img = cv2.imread(path + "/" + y)
    #img = cv2.resize(img, (w // 3, h // 3))
    kp2, des2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des2, des1)
    #matches.sort(key=lambda x: x.distance)
    good = matches[:int(len(matches)*(per/100))]
    imgMatch = cv2.drawMatches(img, kp2, imgQ, kp1, good[:100], None, flags=2)
    #imgMatch = cv2.resize(imgMatch, (w // 3, h // 3))
    #cv2.imshow(y, imgMatch)

    srcPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dstPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
    imgScan = cv2.warpPerspective(img, M, (w, h))

    #cv2.imshow(y, imgScan)

    myData = []

    imgShow = imgScan.copy()
    imgMask = np.zeros_like(imgShow)

    for x, r in enumerate(roi):
        cv2.rectangle(imgMask, (r[0][0], r[0][1]), (r[1][0], r[1][1]), (0, 255, 0), cv2.FILLED)
        imgShow = cv2.addWeighted(imgShow, 0.99, imgMask, 0.1, 0)

        #crop info
        imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]
        #cv2.imshow(str(x), imgCrop)

#-------extract data from crop image-------

        #print(f'{r[3]} : {pytesseract.image_to_string(imgCrop)}')
        #myData.append(pytesseract.image_to_string(imgCrop))

        #changed to md from mad
        data = pytesseract.image_to_string(imgCrop)
        data = str(data)
        data = data.replace('Mad', 'Md')
        print(f'{r[3]} : {data}')
        myData.append(pytesseract.image_to_string(imgCrop))

#-------write data in csv file-------
    with open('DataOutput.csv', 'a+') as f:
        for data in myData:
            data = str(data)
            data = data.replace('Mad', 'Md')
            f.write((data+','))
        f.write('\n')

    #imgShow = cv2.resize(imgShow, (w // 3, h // 3))
    #cv2.imshow(y, imgShow)
    print(myData)

