#####################################################################

# The files in this directory are based off several tutorials and
# scripts provided by Toby Breckon, toby.breckon@durham.ac.uk,
# which may be found at:
#
#       https://github.com/tobybreckon/stereo-disparity
#
# They are available under the LGPL licence
# (http://www.gnu.org/licenses/lgpl.html). Where tutorials have been
# used, or OpenCV documentation, they are referenced in-line.

#####################################################################


import cv2
import numpy as np


class Disparity:
    """
    Superclass for calculating disparity. Contains helper functions and variables for more advanced subclasses.

    """

    @staticmethod
    def to_image(disparity: np.ndarray) -> np.ndarray:
        """
        Convert a disparity map to form suitable for display: an 8-bit greyscale image. Brighter areas are closer.
        """

        image = cv2.normalize(src=disparity, dst=disparity, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)

        return np.uint8(image)

    def __init__(self, histogram="default", clip_limit=3, tile_grid_size=(12, 12)):

        self.histogram = histogram
        self.offset_x = 135
        self.max_y = 395

        self.baseline_m = 0.2090607502              # in metres
        self.focal_length_px = 399.9745178222656    # in pixels
        self.focal_length_m = 0.0048                # in metres

        self.max_disparity = 128

        # https://docs.opencv.org/3.4/d6/db6/classcv_1_1CLAHE.html
        # https://docs.opencv.org/master/d5/daf/tutorial_py_histogram_equalization.html

        if histogram == "CLAHE":

            self.histogram_equaliser = cv2.createCLAHE(
                clipLimit=clip_limit,          # Sets threshold for contrast limiting
                tileGridSize=tile_grid_size    # Input image will be divided into equally sized rectangular tiles
            )

    def get_box_distance(self, disparity: np.ndarray, box: tuple) -> float:
        """
        Given the disparity map of an image, and a bounding box in that image, get the distance of the object contained
        within that box. Make adjustments for blank areas of disparity and bad boxes.
        """

        x, y, width, height = box

        if x < self.offset_x:                   # If we're in the left-images region with no disparity

            width = width + x - self.offset_x
            x = self.offset_x

        if y + height > self.max_y:             # If we're in the bottom region with no disparity

            height = self.max_y - y

        disparity_box = disparity[y: y + height, x: x + width]
        disparity_box_average = np.average(disparity_box)

        if disparity_box_average != 0:

            # depth (metres) = baseline (metres) * focal_length (pixels) / disparity (pixels)
            return self.focal_length_px * self.baseline_m / disparity_box_average

        else:

            return 0.0

    def postprocess(self, disparity: np.ndarray) -> np.ndarray:

        _, disparity = cv2.threshold(disparity, 0, self.max_disparity * 16, cv2.THRESH_TOZERO)

        disparity = (disparity / 16.).astype(np.uint8)

        # https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#bilateralfilter
        disparity = cv2.bilateralFilter(disparity, 5, 25, 25)

        return disparity

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess an image to improve disparity calculation. Convert to greyscale, smooth using a bilateral filter,
        apply histogram equalisation, and raise to the power of 3/4.
        """

        # Convert to greyscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html#median-filtering
        image = cv2.medianBlur(image, 3)

        if self.histogram == "CLAHE":

            image = self.histogram_equaliser.apply(image)

        else:

            image = cv2.equalizeHist(image)

        return image
