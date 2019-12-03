import cv2
import time
from stereo.sgbm import SGBM
from stereo.wls import WLS
from stereo.disparity import Disparity
import os
import numpy as np


def processor_comparison():

    processors = [SGBM(), WLS()]

    left_path = "/home/dom/PycharmProjects/ComputerVision/images/left/1506942473.484027_L.png"
    right_path = "/home/dom/PycharmProjects/ComputerVision/images/right/1506942473.484027_R.png"

    # left_path = os.path.join("images", "left", "1506942473.484027_L.png")
    # right_path = os.path.join("images", "right", "1506942473.484027_R.png")

    left = cv2.imread(left_path, cv2.IMREAD_COLOR)
    right = cv2.imread(right_path, cv2.IMREAD_COLOR)

    cv2.imshow("Left", left)
    cv2.imshow("Right", right)

    for processor in processors:

        start = time.time()

        disparity = processor.calculate(left, right)

        print("{} took {:.2f} seconds".format(processor.__class__.__name__, time.time() - start))

        cv2.imshow('{} disparity'.format(processor.__class__.__name__), Disparity.to_image(disparity))

    cv2.waitKey(0)


def histogram_comparison():

    left_path = os.path.join("..", "TTBB-durham-02-10-17-sub10", "left-images", "1506943569.478977_L.png")     # Colour
    right_path = os.path.join("..", "TTBB-durham-02-10-17-sub10", "right-images", "1506943569.478977_R.png")   # Greyscale

    left = cv2.imread(left_path, cv2.IMREAD_COLOR)
    right = cv2.imread(right_path, cv2.IMREAD_COLOR)

    clahe = Disparity(histogram="CLAHE")
    default = Disparity()

    left_clahe = clahe.preprocess(left)
    left_default = default.preprocess(left)

    both = np.hstack((cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), left_default, left_clahe))

    cv2.imshow("Both", both)
    cv2.imwrite("../output/tests/histogram_comparison.png", both)

    cv2.waitKey(0)


def tile_grid_optimisation():

    left_path = os.path.join("..", "TTBB-durham-02-10-17-sub10", "left-images", "1506943569.478977_L.png")     # Colour
    right_path = os.path.join("..", "TTBB-durham-02-10-17-sub10", "right-images", "1506943569.478977_R.png")   # Greyscale

    left = cv2.imread(left_path, cv2.IMREAD_COLOR)

    # Let's make two concurrently, then join them.

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


def clip_limit_optimisation():

    left_path = os.path.join("..", "TTBB-durham-02-10-17-sub10", "left-images", "1506943569.478977_L.png")     # Colour
    right_path = os.path.join("..", "TTBB-durham-02-10-17-sub10", "right-images", "1506943569.478977_R.png")   # Greyscale

    # Having a clip_limit of 0 completely removes the dark areas. That might actually prove to be superior. It depends
    # what's desirable.

    #

    left = cv2.imread(left_path, cv2.IMREAD_COLOR)

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


histogram_comparison()
