import cv2
import glob
import numpy as np


i = 184
cars = glob.glob('data/cars/new/*.jpg')

for path in cars:
    img = cv2.imread(path)
    rows, cols, _ = img.shape

    M = np.float32([[1, 0, -400], [0, 1, 350]])
    dst = cv2.warpAffine(img, M, (cols, rows))

    cv2.imwrite('./data/cars/new/' + str(i) + '.jpg', dst)
    i+=1
