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
    erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), 1)

    _, conts, _ = cv2.findContours(erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    signs = []

    for i, con in enumerate(conts):
        area = cv2.contourArea(con)
        if 1000 < area < 5000:
            M = cv2.moments(con)
            cx = int(M['m10'] / M['m00'])
            frag = erode[5:95, cx - 25:cx + 25]
            y, x = frag.shape
            if y > 0 and x > 00:
                signs.append((frag, cx, area))

    if len(signs) > 3:
        signs = sorted(signs, key=lambda a_entry: a_entry[1])
        global z
        aha = []
        i = 0
        while i < len(signs) - 1:
            if signs[i + 1][1] - signs[i][1] < 20:
                if signs[i][2] > signs[i + 1][2]:
                    aha.append(signs[i])
                    i += 1
            else:
                aha.append(signs[i])
            i += 1

        if abs(signs[len(signs) - 1][1] - signs[len(signs) - 2][1]) > 10:
            aha.append(signs[len(signs) - 1])

        return aha

    return None


def transform(img, points):
    pts1 = np.float32([[points[0], points[1]], [points[2], points[3]], [points[4], points[5]], [points[6], points[7]]])
    pts2 = np.float32([[0, 0], [400, 0], [0, 100], [400, 100]])

    M = cv2.getPerspectiveTransform(pts1, pts2)

    dst = cv2.warpPerspective(img, M, (400, 100))

    return dst



cars = np.load(os.path.join(data_path, 'arrays', 'cars.npy'))[0:10]

array = []

i = 0

for i, car in enumerate(cars):
    plate = transform(car[0],[car[1], car[2], car[3], car[4], car[5], car[6], car[7], car[8]])
    chars = split(plate)
    cv2.imshow('image', plate)

    if chars is not None:
        k = 0
        while k < len(chars):
            cv2.imshow('char', chars[k][0])
            cv2.waitKey(1)
            type = input('Znak: ')

            if not type or len(type)>1:
                continue
            array.append([chars[k][0], type.upper()])
            k += 1


if os.path.isfile(os.path.join(data_path, 'arrays', 'chars.npy')):
    old_array = np.load(os.path.join(data_path, 'arrays', 'aa.npy'))
    new_array = old_array.tolist() + array
    print(len(new_array))
    np.save(os.path.join(data_path, 'arrays', 'chars'), new_array)
    np.save(os.path.join(data_path, 'arrays', 'chars_only_new'), array)
else:
    np.save(os.path.join(data_path, 'arrays', 'chars'), array)
