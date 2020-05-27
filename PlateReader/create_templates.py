import numpy as np
from PIL import ImageFont, ImageDraw, Image
import cv2
import string
import os

chars = sorted(string.ascii_lowercase+string.digits)

fontpath = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(
    __file__))), 'data', 'arklatrs.ttf')

font = ImageFont.truetype(fontpath, 128)

base = []
for ch in chars:
    # Make canvas and set the color
    img = np.zeros((100, 60), np.uint8)

    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text((-5, -18),  ch, font=font, fill=255)
    im = np.array(img_pil)

    im = cv2.resize(im, (50, 100))
    im = cv2.erode(im, np.ones((3, 3)))
    im = cv2.threshold(im, 80, 255, cv2.THRESH_BINARY)[1]

    aa = np.argwhere(im == 255)
    miny, minx = aa.min(axis=0)
    maxy, maxx = aa.max(axis=0)
    im = cv2.resize(im[miny:maxy, minx:maxx], (50, 100))
    im = cv2.threshold(im, 80, 255, cv2.THRESH_BINARY)[1]

    base.append(im)

base = np.array(base)
np.save(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(
    __file__))), 'data', 'chars_templates.npy'), base, allow_pickle=True)

