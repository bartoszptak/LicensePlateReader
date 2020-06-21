"""
    Author: Bartosz Ptak - 2020

    It is a project for the subject of Vision Systems conducted at Robots and Autonomous Systems
    specialization at the Poznan University of Technology.

    Task:
        Writing a program without using machine learning:
            - find car plates in the image
            - read the marks on them

    Assumptions:
        - law in Poland
        - boards with only 7 characters
        - images for max. 45 degrees

    Final results on private test set:
        - Find bbox accurancy: 87.50% (42 readed plates for 48 total)
        - Total accurancy: 83.97% (288 good chars per 343 total)
        - Execution time: 10.32s (per 48 images)
"""
import numpy as np
import cv2
import os
from typing import Tuple, List
from scipy.spatial.distance import cdist
from skimage import measure
from skimage.measure import regionprops
import string
import imutils


class PlateReader:
    """Class to search for license plates and to describe them.

    Example:
        import cv2
        from PlateReader import PlateReader

        pr = PlateReader()

        img = cv2.imread('path_to_image.png')
        plate_chars = pr.read(img)
        print(plate_chars)
    """

    def __init__(self) -> None:
        """Class constructor for image processing. Requires defining some constants for the problem.

        Elements
        ----------
        plate_size
            This is the size of the license plates (official size in mm), default 520x114 mm
        plate_numbers
            Number of characters on the license plate (here fixed 7)
        char_size
            The size of one character to which each of them will be normalized
        char_base
            Character pattern generated from arklatrs.ttf font for char_size 50x100
        dep_right
            Restrictions resulting from restrictions on the possibility of the char being
            abbreviated as a region
        """

        self.plate_size = (520, 114)
        self.plate_numbers = 7
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
            os.path.join(os.path.dirname(os.path.realpath(
                __file__)), 'chars_templates.npy'),
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

        plate, pkt = self.get_plate_from_image(img)

        if plate is None:
            plate, pkt = self.get_plate_from_image2(img)

            if plate is None:
                return self.check_character_dependencies([])

        pkt = self.order_points(pkt)
        plate = self.normalize_plate_size(plate, pkt)
        plate = self.less_dummy_threshold(plate)

        chars, dists = self.split_plate_to_chars(plate)
        predicted = self.predict_chars(chars, dists)

        result = self.check_character_dependencies(predicted)

        return result

    @staticmethod
    def try_get_contour(gray_org, region):
        try:
            eq = gray_org[region[0]-5:region[2]+5, region[1]-5:region[3]+5]

            edged = cv2.Canny(eq, 80, 200, apertureSize=3)
            edged = cv2.morphologyEx(
                edged, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.float32))
            cnts = cv2.findContours(
                edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

            screenCnt = None

            for c in cnts:
                approx = cv2.approxPolyDP(
                    c, 0.010 * cv2.arcLength(c, True), True)
                if len(approx) == 4 and cv2.contourArea(c) > 2000:
                    screenCnt = approx[:, 0]
                    break

            return eq, screenCnt
        except:
            return None, None

    def get_plate_from_image(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """This is the first method of searching for car registrations. Fast and quite effective.

        Parameters
        ----------
        img : np.ndarray
            Imput image

        Returns
        -------
        np.ndarray
            Unormalized plate image
        np.ndarray
            Unsorted points
        """
        img = cv2.resize(img, (1280, 1280*img.shape[0]//img.shape[1]))
        gray_org = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_org = cv2.bilateralFilter(gray_org, 11, 17, 17)
        tresh = cv2.threshold(gray_org, 120, 255, 0)[1]

        label_image = measure.label(tresh)

        screenCnt = None

        for i, region in enumerate(sorted(regionprops(label_image), key=lambda x: x.area, reverse=True)):
            if region.area < 30000:
                # if the region is so small then it's likely not a license plate
                continue

            # the bounding box coordinates
            minRow, minCol, maxRow, maxCol = region.bbox

            if 520/114*1.8 < (maxCol-minCol)/(maxRow-minRow) or \
                520/114*0.2 > (maxCol-minCol)/(maxRow-minRow) or \
                (maxRow-minRow) > (maxCol-minCol) or \
                    minCol == 0 or minRow == 0:
                continue

            eq, screenCnt = self.try_get_contour(gray_org, region.bbox)
            if screenCnt is not None:
                break

        if screenCnt is None:
            return None, None

        mask = np.zeros(eq.shape, np.uint8)
        new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1,)
        new_image = cv2.bitwise_and(eq, eq, mask=mask)

        return new_image, screenCnt

    def get_plate_from_image2(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """This method is more of a brute force type, it can find special cases.

        Parameters
        ----------
        img : np.ndarray
            Imput image

        Returns
        -------
        np.ndarray
            Unormalized plate image
        np.ndarray
            Unsorted points
        """
        img = cv2.resize(img, (1280, 720))
        gray_org = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_org = cv2.bilateralFilter(gray_org, 11, 17, 17)

        screenCnt = None

        cannies = [10, 30, 50, 80]
        approxies = np.arange(0.014, 0.020, 0.002)
        scales = [1.5, 1.0, 0.8, 0.5]

        cannies_c = 0
        approxies_c = 0
        scales_c = 0

        scale = None

        while screenCnt is None:
            gray = cv2.resize(
                gray_org, None, fx=scales[scales_c], fy=scales[scales_c]).copy()
            edged = cv2.Canny(gray, cannies[cannies_c], 180)
            cnts = cv2.findContours(
                edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

            for c in cnts:
                approx = cv2.approxPolyDP(
                    c, approxies[approxies_c] * cv2.arcLength(c, True), True)
                if len(approx) == 4 and cv2.contourArea(c) > 10000:
                    screenCnt = approx[:, 0]
                    scale = scales[scales_c]
                    break

            approxies_c += 1

            if approxies_c >= len(approxies):
                approxies_c = 0
                cannies_c += 1

            if cannies_c >= len(cannies):
                approxies_c = 0
                cannies_c = 0
                scales_c += 1

            if scales_c >= len(scales):
                break

        if screenCnt is None:
            return None, None

        gray = cv2.resize(gray_org, None, fx=scale, fy=scale).copy()
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1,)
        new_image = cv2.bitwise_and(gray, gray, mask=mask)

        return new_image, screenCnt

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

        The function uses contour search. Then they are sorted from those with the largest area.
        Then iteration over the detected contours occurs and if they have sufficient surface area,
        they are processed:
            1. Create ROI
            2. Scale ROI
            3. Erosion to get rid of outliers
            4. Binarization
            5. Crop the char to the image size
            6. Re-binarization
            7. Calculate the center of the contour
        """

        conts, _ = cv2.findContours(
            plate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        conts = sorted(conts, key=lambda ctr: cv2.boundingRect(ctr)[0])

        chars = []
        dists = [0]

        for i, ctr in enumerate(conts):
            if cv2.contourArea(ctr) > 900:
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
            List of dicts like {'char': 'P, 'left' True}. Left is true if the char is befor 
            the sticker

        Algorithm:
             - Calculates the cosine similarity between input and pattern.
             - Checks if the character is a character describing the region.
        """

        res = cdist(self.char_base, chars.reshape(
            (-1, np.multiply(*self.char_size))), metric="cosine")

        plate = []
        flag = True
        for i, (ch, d) in enumerate(zip(res.T[-self.plate_numbers:], dists[-self.plate_numbers:])):
            if (flag and i > 2) or i < 3:
                flag = d < 90

            plate.append({
                'char': self.char_targets[ch.argmin()].upper(),
                'left': flag
            })

        return plate

    def check_character_dependencies(self, plate: List) -> str:
        """Check character dependencies. The characters before the sticker are reserved. If too few
        characters were read, fill in with "?".

        Parameters
        ----------
        plate : List
            List of dicts like {'char': 'P, 'left' True}

        Returns
        -------
        str
            String with a length of self.plate_numbers
        """

        results = ''
        for i, p in enumerate(plate):
            if (p['left'] and p['char'] in self.dep_left) and i < 2:
                results += self.dep_left.get(p['char'], p['char'])
            elif (not p['left'] and p['char'] in self.dep_right) and i > 2:
                results += self.dep_right.get(p['char'], p['char'])
            else:
                results += p['char']

        return results.ljust(self.plate_numbers, '?')
