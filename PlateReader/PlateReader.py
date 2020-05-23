import numpy as np
import cv2


class PlateReader:
    """[summary]
    """

    def __init__(self) -> None:
        """Class constructor
        """
        pass

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
        
        return ""