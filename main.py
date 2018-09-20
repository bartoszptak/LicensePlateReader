import numpy as np
import cv2
from keras.models import load_model
import os

div = 10
div2 = 1
z = 0

car_model = load_model(os.path.join('data', 'models', 'cars'))
car_model.load_weights(os.path.join('data', 'weights', 'cars'))

char_model = load_model(os.path.join('data', 'models', 'chars'))
char_model.load_weights(os.path.join('data', 'weights', 'chars'))


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


def char_predict(img):
    im = cv2.resize(img, (int(50 / div2), int(90 / div2)))

    Xx = im * 1. / 255
    Xx = np.array(Xx)
    X = Xx[np.newaxis, ..., np.newaxis]
    L = char_model.predict_classes(X)

    return chr(int(L[0]))


def show(frame, plate, chars):

    L = []
    for ch in chars:
        L.append(char_predict(ch[0]))

    text = ''

    for i, sig in enumerate(chars):
        if i == 0:
            text += L[i]
        else:
            if chars[i][1] - chars[i - 1][1] < 65:
                text += L[i]
            else:
                text += ' '+L[i]

    plate = cv2.resize(plate, (200, 50))

    cv2.rectangle(frame, (0, 0), (200, 90), (255, 255, 255), -1)
    frame[0:50, 0:200] = plate
    cv2.putText(frame, text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Plates reader', frame)
    cv2.waitKey(0)


def transform(img, points):
    pts1 = np.float32([[points[0], points[1]], [points[2], points[3]], [points[4], points[5]], [points[6], points[7]]])
    pts2 = np.float32([[0, 0], [400, 0], [0, 100], [400, 100]])

    M = cv2.getPerspectiveTransform(pts1, pts2)

    dst = cv2.warpPerspective(img, M, (400, 100))

    return dst


def car_predict(img):
    im = cv2.resize(img, (int(1280 / div), int(720 / div)))
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    Xx = gray * 1. / 255
    Xx = np.array(Xx)
    X = Xx[np.newaxis, ..., np.newaxis]
    K = car_model.predict(X)
    K *= div

    return [int(K[0][0]), int(K[0][1]), int(K[0][2]), int(K[0][3]), int(K[0][4]), int(K[0][5]), int(K[0][6]),
            int(K[0][7])]


def main():
    its = np.load('data/arrays/cars_only_new.npy')
    for i,it in enumerate(its):
        frame = it[0]
        K = [it[1], it[2], it[3], it[4], it[5], it[6], it[7], it[8]]


        frame = cv2.resize(frame, (1280, 720))
        #K = car_predict(test_img)

        plate = transform(frame, K)
        chars = split(plate)
        if chars is not None:
            show(frame, plate, chars)
            print(i)
    cv2.destroyAllWindows()


def plates_cut():
    table = np.load('data/arrays/cars.npy')
    for it in table:
        plate = transform(it[0], [it[1], it[2], it[3], it[4], it[5], it[6], it[7], it[8]])
        split(plate)


main()

