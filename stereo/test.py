import cv2
import time
import os
import numpy as np

from stereo.sgbm import SGBM
from stereo.wls import WLS
from stereo.disparity import Disparity

# Helper script for generating report TTBB-durham-02-10-17-sub10

left_path = os.path.join("..", "TTBB-durham-02-10-17-sub10", "left-images", "1506943569.478977_L.png")  # Colour
right_path = os.path.join("..", "TTBB-durham-02-10-17-sub10", "right-images", "1506943569.478977_R.png")  # Greyscale

left = cv2.imread(left_path, cv2.IMREAD_COLOR)
right = cv2.imread(right_path, cv2.IMREAD_COLOR)


def processor_comparison():

    processors = [SGBM(), WLS()]
    processor_image = None

    for processor in processors:

        disparity = processor.calculate(left, right)

        if processor_image is None:

            processor_image = processor.to_image(disparity)

        else:

            processor_image = np.vstack((processor_image,  processor.to_image(disparity)))

    cv2.imshow("Processor image", processor_image)
    cv2.imwrite("../output/tests/processor_comparison.png", processor_image)

    cv2.waitKey(0)


def histogram_comparison():

    grey_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)

    clahe = Disparity(histogram="CLAHE")
    default = Disparity()

    left_clahe = clahe.preprocess(left)
    left_default = default.preprocess(left)

    both = np.hstack((grey_left, left_default, left_clahe))

    cv2.imshow("Both", both)
    cv2.imwrite("../output/tests/histogram_comparison.png", both)

    cv2.waitKey(0)


def tile_grid_optimisation():

    tile_grid_image = None

    for i in range(2, 16, 2):

        tile_grid_size = (i, i)

        clahe = Disparity(histogram="CLAHE", tile_grid_size=tile_grid_size)

        left_clahe = clahe.preprocess(left)

        if tile_grid_image is None:

            tile_grid_image = left_clahe

        else:

            tile_grid_image = np.vstack((tile_grid_image, left_clahe))

    cv2.imshow("Tile_grid_image", tile_grid_image)
    cv2.imwrite("../output/tests/tile_grid_image.png", tile_grid_image)

    cv2.waitKey(0)


def median_filter_comparison():

    print(left.shape)

    grey_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)

    median = cv2.medianBlur(grey_left, 3)

    median_comparison = np.hstack((grey_left, median))

    cv2.imshow("median_comparison", median_comparison)
    cv2.imwrite("../output/tests/median_comparison.png", median_comparison)

    cv2.waitKey(0)

    # There doesn't seem to be a particularly salt-and-peppery

def clip_limit_optimisation():

    clip_limit_image = None

    for i in range(0, 6):

        clip_limit = i / 2

        clahe = Disparity(histogram="CLAHE", clip_limit=clip_limit)

        left_clahe = clahe.preprocess(left)

        if clip_limit_image is None:

            clip_limit_image = left_clahe

        else:

            clip_limit_image = np.vstack((clip_limit_image, left_clahe))

    cv2.imshow("clip_limit_image", clip_limit_image)
    cv2.imwrite("../output/tests/clip_limit_image.png", clip_limit_image)

    cv2.waitKey(0)


# histogram_comparison()
processor_comparison()