import numpy as np
import cv2
import os
from typing import Tuple, List
from scipy.spatial.distance import cdist
import string


class PlateReader:
    """[summary]
    """

    def __init__(self) -> None:
        """Class constructor
        """

        self.plate_size = (520, 114)
        self.dst_matrix = np.array(
            [
                [0, 0],
                [self.plate_size[0], 0],
                self.plate_size,
                [0, self.plate_size[1]]
            ], dtype=np.float32)

        self.char_size = (50, 100)
        self.char_tresh = 80
        self.char_targets = sorted(string.ascii_lowercase+string.digits)
        self.char_base = np.load(
            os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(
                __file__))), 'data', 'chars_templates.npy'),
            allow_pickle=True).reshape(
                len(self.char_targets),
                np.multiply(*self.char_size))

        self.dep_right = {
            'B': '8',
            'D': '0',
            'I': '1',
            'O': '0',
            'Z': '2'
        }
        self.dep_left = {v: k for k, v in self.dep_right.items()}

    def read(self, img: np.ndarray) -> str:
        """The main function for reading car license plates.

        Parameters
        ----------
        img : np.ndarray
            input image, where should be the car with the license plate

        Returns
        -------
        str
            car license plate numbers
        """

        pkt = np.zeros((4, 2))
        plate = np.zeros((400, 321), dtype=np.uint8)

        pkt = self.order_points(pkt)
        plate = self.normalize_plate_size(plate, pkt)
        plate = self.less_dummy_threshold(plate)

        chars, dists = self.split_plate_to_chars(plate)
        predicted = self.predict_chars(chars, dists)
        result = self.check_character_dependencies(predicted)

        return result

    @staticmethod
    def order_points(pts: np.ndarray) -> np.ndarray:
        """Set the points always in the same order.

        Parameters
        ----------
        pts : np.array
            Array of 4 points [(x0,y0), (x1,y1), (x2,y2), (x3,y3)]

        Returns
        -------
        np.array
            Sorted array of 4 points [(x0,y0), (x1,y1), (x2,y2), (x3,y3)]
        """

        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    def normalize_plate_size(self, plate: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Warped and normalized plate image by perspective transformation.

        Parameters
        ----------
        plate : np.ndarray
            Unormalized plate image
        pts : np.ndarray
            Sorted points

        Returns
        -------
        np.ndarray
            Warped and normalized plate image
        """

        M = cv2.getPerspectiveTransform(pts, self.dst_matrix)
        warped = cv2.warpPerspective(plate, M, self.plate_size)

        return warped

    def less_dummy_threshold(self, plate: np.ndarray) -> np.ndarray:
        """Threshold image to get the most border colors.

        Parameters
        ----------
        plate : np.ndarray
            Grey-scaled image

        Returns
        -------
        np.ndarray
            Binary image
        """

        plate = cv2.adaptiveThreshold(
            plate, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 33, 3)

        # add padding
        plate[:5, :] = 0
        plate[:, :5] = 0
        plate[-5:, :] = 0
        plate[:, -5:] = 0

        plate = cv2.morphologyEx(plate, cv2.MORPH_OPEN, np.ones(
            (3, 3), dtype=np.float32), iterations=2)

        return plate

    def split_plate_to_chars(self, plate: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Divide the license plate image into individual characters, keep the distance between them.

        Parameters
        ----------
        plate : np.ndarray
            Normalized plate image

        Returns
        -------
        np.ndarray
            Splitted chars images
        np.ndarray
            Distances between chars
        """

        conts, _ = cv2.findContours(
            plate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        conts = sorted(conts, key=lambda ctr: cv2.boundingRect(ctr)[0])

        chars = []
        dists = [0]

        for i, ctr in enumerate(conts):
            if cv2.contourArea(ctr) > 1000:
                x, y, w, h = cv2.boundingRect(ctr)

                roi = plate[y-1:y+h+1, x-1:x+w+1]
                roi = cv2.resize(roi, self.char_size)
                roi = cv2.erode(roi, np.ones((3, 3)))
                roi = cv2.threshold(roi, self.char_tresh,
                                    255, cv2.THRESH_BINARY)[1]

                mask = np.argwhere(roi == 255)
                miny, minx = mask.min(axis=0)
                maxy, maxx = mask.max(axis=0)

                roi = cv2.resize(roi[miny:maxy, minx:maxx], self.char_size)
                roi = cv2.threshold(roi, self.char_tresh,
                                    255, cv2.THRESH_BINARY)[1]

                M = cv2.moments(ctr)
                cX = int(M["m10"] / M["m00"])

                dists.append(cX)
                chars.append(roi)
            else:
                cv2.drawContours(plate, conts, i, 0, -1)

        return np.array(chars), np.diff(np.array(dists))

    def predict_chars(self, chars: np.ndarray, dists: np.ndarray) -> List:
        """Infer character information from images.

        Parameters
        ----------
        chars : np.ndarray
            Splitted chars images
        dists : np.ndarray
            Distances between chars

        Returns
        -------
        List
            List of dicts like {'char': 'P, 'left' True}. Left is true if the char is befor the sticker
        """

        res = cdist(self.char_base, chars.reshape(
            (-1, np.multiply(*self.char_size))), metric="cosine")

        plate = []
        flag = True
        for iaaa, d in zip(res.T[-7:], dists[-7:]):
            if flag:
                flag = d < 85

            plate.append({
                'char': chars[iaaa.argmin()].upper(),
                'left': flag
            })

        return plate

    def check_character_dependencies(self, plate: List) -> str:
        """Check character dependencies. The characters before the sticker are reserved. If too few
        characters were read, fill in with "_".

        Parameters
        ----------
        plate : List
            List of dicts like {'char': 'P, 'left' True}

        Returns
        -------
        str
            String with a length of 7
        """

        results = ''
        for p in plate:
            if p['left'] and p['char'] in self.dep_left:
                results += self.dep_left[p['char']]
            elif not p['left'] and p['char'] in self.dep_right:
                results += self.dep_right[p['char']]
            else:
                results += p['char']

        return results.ljust(7, '_')
