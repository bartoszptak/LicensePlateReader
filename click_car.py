import cv2
import glob
import numpy as np
import os

data_path = 'data'

imgs = glob.glob(os.path.join(data_path, 'cars', 'new', '*'))

array = []

i = 0


def transform(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONUP:
        Xs.append(x)
        Xs.append(y)


cv2.namedWindow('image')
cv2.setMouseCallback('image', transform)

while i < imgs.__len__():
    Xs = []
    im = cv2.imread(imgs[i])
    im = cv2.resize(im, (1280, 720))

    cv2.imshow('image', im)
    cv2.waitKey(0)

    if len(Xs) < 8:
        continue
    array.append([im, Xs[0], Xs[1], Xs[2], Xs[3], Xs[4], Xs[5], Xs[6], Xs[7]])
    print(imgs.__len__() - i)
    i += 1

if os.path.isfile(os.path.join(data_path, 'arrays', 'cars.npy')):
    old_array = np.load(os.path.join(data_path, 'arrays', 'cars.npy'))
    new_array = old_array.tolist() + array
    print(len(new_array))
    np.save(os.path.join(data_path, 'arrays', 'cars'), new_array)
    np.save(os.path.join(data_path, 'arrays', 'cars_only_new'), array)
else:
    np.save(os.path.join(data_path, 'arrays', 'cars'), array)
