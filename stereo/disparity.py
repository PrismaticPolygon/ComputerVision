import cv2
import numpy as np
import os
# from stereo.wls import WLS
# from stereo.sgbm import SGBM
import time


class Disparity:

    @staticmethod
    def to_image(disparity):

        image = cv2.normalize(src=disparity, dst=disparity, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)

        return np.uint8(image)

    def __init__(self):

        self.offset_x = 135
        self.max_y = 396

        self.baseline_m = 0.2090607502              # in metres
        self.focal_length_px = 399.9745178222656    # in pixels
        self.focal_length_m = 4.8 / 1000            # in metres

        self.image_centre_h = 262.0  # rectified image centre height, in pixels
        self.image_centre_w = 474.5  # rectified image centre width, in pixels

        # https://docs.opencv.org/3.4/d6/db6/classcv_1_1CLAHE.html
        # https://docs.opencv.org/master/d5/daf/tutorial_py_histogram_equalization.html

        self.histogram_equaliser = cv2.createCLAHE(
            clipLimit=2.0,          # Sets threshold for contrast limiting
            tileGridSize=(8, 8)     # Input image will be divided into equally sized rectangular tiles
        )

    def get_box_distance(self, disparity, box):

        x, y, width, height = box

        if x < self.offset_x:                   # If we're in the left region with no disparity

            width = width - (self.offset_x - x)
            x = self.offset_x

        if y + height > self.max_y:             # If we're in the bottom region with no disparity

            height = self.max_y - y

        disparity_box = disparity[y: y + height, x: x + width]
        disparity_median = np.nanmedian(disparity_box)

        # depth (metres) = baseline (metres) * focal_length (pixels) / disparity (pixels)

        if disparity_median != 0:   # Ah. This is just the median value of the box.

            distance = (self.focal_length_px / self.baseline_m) / disparity_median
            distance = distance / 100

            return distance

        else:

            return 0

    def preprocess(self, image):

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Appears subjectively to improve performance

        # image = np.power(image, 0.75).astype('uint8')

        # https://docs.opencv.org/3.4/d5/d69/tutorial_py_non_local_means.html
        image = cv2.fastNlMeansDenoising(image, h=10, templateWindowSize=7, searchWindowSize=21)

        image = self.histogram_equaliser.apply(image)

        return image


# if __name__ == "__main__":
#
#     processors = [SGBM(), WLS()]
#
#     left_path = os.path.join("images", "left", "1506942473.484027_L.png")
#     right_path = os.path.join("images", "right", "1506942473.484027_R.png")
#
#     left = cv2.imread(left_path, cv2.IMREAD_COLOR)
#     right = cv2.imread(right_path, cv2.IMREAD_COLOR)
#
#     cv2.imshow("Left", left)
#     cv2.imshow("Right", right)
#
#     for processor in processors:
#
#         start = time.time()
#
#         disparity = processor.calculate(left, right)
#
#         print("{} took {:.2f} seconds".format(processor.__name__, time.time() - start))
#
#         cv2.imshow('{} disparity'.format(processor.__name__), Disparity.to_image(disparity))
#
#         cv2.waitKey(0)



