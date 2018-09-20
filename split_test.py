import cv2
import glob
import numpy as np
import os

data_path = 'data'


def split(img):
    img = cv2.resize(img, (400, 100))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)

    _, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)
    erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), 2)
    dilate = cv2.dilate(erode, np.ones((3, 3), np.uint8), 1)

    _, conts, _ = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.imshow('ba', dilate)
    signs = []

    for con in conts:
        area = cv2.contourArea(con)
        if area > 100:
            M = cv2.moments(con)
            cx = int(M['m10'] / M['m00'])
            mini, maxi = cx - 25, cx + 25
            if mini < 0:
                mini, maxi = 0, 50
            elif maxi > erode.shape[1]:
                mini, maxi = erode.shape[1]-50, erode.shape[1]
            frag = erode[5:95, mini:maxi]
            y, x = frag.shape
            if y > 0 and x > 0:
                signs.append([frag, cx, area])

    if len(signs) > 1:
        signs = sorted(signs, key=lambda a_entry: a_entry[1])

        aha = []
        j = 1
        old = signs[0]
        aha.append(old)
        while j < len(signs):
            if signs[j][1]-old[1] > 30:
                aha.append(signs[j])
                old = signs[j]
            j += 1
        return aha

    print('nie odczytałem')
    return None


def transform(img, points):
    pts1 = np.float32([[points[0], points[1]], [points[2], points[3]], [points[4], points[5]], [points[6], points[7]]])
    pts2 = np.float32([[0, 0], [400, 0], [0, 100], [400, 100]])

    M = cv2.getPerspectiveTransform(pts1, pts2)

    dst = cv2.warpPerspective(img, M, (400, 100))

    return dst



cars = np.load(os.path.join(data_path, 'arrays', 'cars.npy'))[95:150]


i = 0

for i, car in enumerate(cars):
    plate = transform(car[0],[car[1], car[2], car[3], car[4], car[5], car[6], car[7], car[8]])
    chars = split(plate)

    cv2.imshow('a', plate)
    if chars is not None:
        k = 0
        while k < len(chars):
            cv2.imshow('char', chars[k][0])
            cv2.waitKey(0)
            k += 1

