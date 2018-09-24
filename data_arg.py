import cv2
import glob
import numpy as np


i = 1200
cars = glob.glob('data/cars/toarg/*.jpg')

for path in cars:
    img = cv2.imread(path)
    #img = cv2.resize(img,None,None,0.2,0.2)
    rows, cols, _ = img.shape

    M = np.float32([[1, 0, -100], [0, 1, -200]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    # M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 11, 1)
    # dst = cv2.warpAffine(img, M, (cols, rows))

    # cv2.imshow('okno2', img)
    # cv2.imshow('okno', dst)
    # cv2.waitKey(0)
    cv2.imwrite('./data/cars/new/arg' + str(i) + '.jpg', dst)
    i += 1
