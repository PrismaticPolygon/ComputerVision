import cv2
import os
import numpy as np

from stereo.sgbm import SGBM
from stereo.wls import WLS
from stereo.disparity import Disparity

# Helper script for generating report images

left_path = os.path.join("..", "TTBB-durham-02-10-17-sub10", "left-images", "1506943569.478977_L.png")  # Colour
right_path = os.path.join("..", "TTBB-durham-02-10-17-sub10", "right-images", "1506943569.478977_R.png")  # Greyscale

left = cv2.imread(left_path, cv2.IMREAD_COLOR)
right = cv2.imread(right_path, cv2.IMREAD_COLOR)

annotated = cv2.imread(os.path.join("..", "output", "images", "1506943569.478977_L.png"))

cv2.imshow("annotated", annotated)

cv2.imwrite("../output/tests/annotated.png", annotated)

cv2.waitKey(0)


def processor_comparison():

    processors = [SGBM(), WLS(histogram="CLAHE")]
    processor_image = None

    for processor in processors:

        disparity = processor.calculate(left, right)

        cv2.imwrite("../output/tests/processor_comparison_{}.png".format(processor.__class__.__name__),  processor.to_image(disparity))

        if processor_image is None:

            processor_image = processor.to_image(disparity)

        else:

            processor_image = np.hstack((processor_image,  processor.to_image(disparity)))

    cv2.imshow("Processor image", processor_image)
    cv2.imwrite("../output/tests/processor_comparison.png", processor_image)

    # cv2.waitKey(0)

def bilateral_filter_comparison():

    bilateral = WLS(histogram="CLAHE", bilateral=True)
    wls = WLS(histogram="CLAHE", bilateral=False)

    disparity = wls.calculate(left, right)
    disparity_b = bilateral.calculate(left, right)

    cv2.imwrite("../output/tests/bilateral_comparison_bilateral.png", bilateral.to_image(disparity_b))
    cv2.imwrite("../output/tests/bilateral_comparison_default.png", wls.to_image(disparity))

    bilateral_comparison = np.hstack((wls.to_image(disparity), bilateral.to_image(disparity_b)))

    cv2.imshow("Bilateral comparison", bilateral_comparison)
    cv2.imwrite("../output/tests/bilateral_comparison.png", bilateral_comparison)

    # cv2.waitKey(0)


def histogram_comparison():

    grey_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)

    clahe = Disparity(histogram="CLAHE")
    default = Disparity()

    left_clahe = clahe.preprocess(left)
    left_default = default.preprocess(left)

    cv2.imwrite("../output/tests/histogram_comparison_CLAHE.png", clahe.to_image(left_clahe))
    cv2.imwrite("../output/tests/histogram_comparison_default.png", default.to_image(left_default))

    both = np.hstack((grey_left, left_default, left_clahe))

    cv2.imshow("Both", both)
    cv2.imwrite("../output/tests/histogram_comparison.png", both)

    # cv2.waitKey(0)


def median_filter_comparison():

    print(left.shape)

    grey_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)

    median = cv2.medianBlur(grey_left, 3)

    median_comparison = np.hstack((grey_left, median))

    cv2.imwrite("../output/tests/source.png", grey_left)
    cv2.imwrite("../output/tests/median.png", median)

    cv2.imshow("median_comparison", median_comparison)
    cv2.imwrite("../output/tests/median_comparison.png", median_comparison)

    # cv2.waitKey(0)

    # There doesn't seem to be a particularly salt-and-peppery

median_filter_comparison()
histogram_comparison()
processor_comparison()
bilateral_filter_comparison()