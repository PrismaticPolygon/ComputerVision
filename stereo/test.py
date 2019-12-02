import cv2
import time
from stereo.sgbm import SGBM
from stereo.wls import WLS
from stereo.disparity import Disparity


if __name__ == "__main__":

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
