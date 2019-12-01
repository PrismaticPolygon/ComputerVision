import cv2
import numpy as np
import time
import os

# https://docs.opencv.org/master/d3/d14/tutorial_ximgproc_disparity_filtering.html
# http://timosam.com/python_opencv_depthimage

# I just want to be sure that I'm not wasting my time.

class Stereo:

    def __init__(self):

        print("Initialising stereo...", end="")

        start = time.time()

        # Camera variables

        self.baseline_m = 0.2090607502           # in metres
        self.focal_length_px = 399.9745178222656 # in pixels
        self.focal_length_m = 4.8 / 1000         # in metres

        self.image_centre_h = 262.0              # rectified image centre height, in pixels
        self.image_centre_w = 474.5              # rectified image centre width, in pixels

        # Processor variables

        self.window_size = 3
        self.min_disparity = 0                      # Minimum possible disparity value. Defaults to 0.
        self.num_disparities = 160                  # Maximum disparity - minimum disparity. Defaults to 16.
        self.block_size = 5                         # Matched blocked size. Must be an odd number >= 1. Normally between 3 and 11. Defaults to 3.
        self.P1 = 8 * 1 * self.window_size ** 2     # First parameter controlling disparity smoothness. The penalty on the disparity change by +- 1 between pixels.
        self.P2 = 32 * 1 * self.window_size ** 2    # Second parameter controlling disparity smoothness. The larger the values, the smoother the disparity. Must be greater than P1.
        self.disp_12_max_diff = 1                   # The maximum allowed difference (in integer pixel units) in the left-right disparity check. Set to non-positive to disable.
        self.pre_filter_cap = 63                    # Truncation value for the prefiltered image pixels
        self.uniqueness_ratio = 15                  # Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct. Normally, a value within the 5-15 range is good enou
        self.speckle_window_size = 200              # The maximum size of smooth disparity regions to consider noise speckles and invalidate. Set to 0. to disable speckling. Should be somewhere between 50 - 200.
        self.speckle_range = 2                      # maximum disparity variation within each connected component. If speckle filtering, set to positive. will be implicitly multiplied by 16. Normally, 1 or 2 is good enough.
        self.mode = cv2.STEREO_SGBM_MODE_HH         # Mode. Defaults to STEREO_SGBM_MODE_SGBM.

        # https://docs.opencv.org/master/d2/d85/classcv_1_1StereoSGBM.html
        self.stereo_processor = cv2.StereoSGBM_create(
            minDisparity=self.min_disparity,
            numDisparities=self.num_disparities,
            blockSize=self.block_size,
            P1=self.P1,
            P2=self.P2,
            disp12MaxDiff=self.disp_12_max_diff,
            preFilterCap=self.pre_filter_cap,
            uniquenessRatio=self.uniqueness_ratio,
            speckleWindowSize=self.speckle_window_size,
            speckleRange=self.speckle_range,
            mode=self.mode
        )

        # Histogram variables

        self.clip_limit = 2.0
        self.tile_grid_size = (8, 8)

        # https://docs.opencv.org/master/d5/daf/tutorial_py_histogram_equalization.html
        self.histogram_equaliser = cv2.createCLAHE(
            clipLimit=self.clip_limit,          #
            tileGridSize=self.tile_grid_size    #
        )

        print("DONE ({:.2f}s)".format(time.time() - start))

    def disparity_to_distance(self, disparity):

        # https://stackoverflow.com/questions/26248654/how-to-return-0-with-divide-by-zero
        return np.divide(self.focal_length_px * self.baseline_m, disparity, out=np.zeros_like(disparity), where=disparity > 0)

    def preprocess(self, left, right):

        images = (left, right)

        # In many papers, a median filter is adopted to suppress noise.
        # I imagine implementing my own in C would give the best results.
        # I wonder if their code is available...

        # For SSD / SAD, some form of band-pass filtering is typically used, such as a LoG.
        # Only the high-pass component which

        for image in images:

            # Median filter.
            # Wiener filter
            # Histogram filter

            # small Gaussian serves as a low pass filter and the differencing serves as a high pass filter.
            # For images of good quality, the noise suppression provided by the low pass filter
            # is generally unnecessary.
            # image = cv2.medianBlur(image, 5)

            image = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)

            # https://docs.opencv.org/3.4/d5/d69/tutorial_py_non_local_means.html
            image = cv2.fastNlMeansDenoising(image, h=10, templateWindowSize=7, searchWindowSize=21)

            image = self.histogram_equaliser.apply(image)

            cv2.imshow("Image", image)

            pass

        # grey_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)  # (544, 1024)
        # grey_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)  # (544, 1024)

        # Perform pre-processing. Raise to the power, as this subjectively appears to improve disparity calculation.

        # cv2.imshow("Left", grey_left)
        # cv2.imshow("Right", grey_right)

        # grey_left = np.power(grey_left, 0.75).astype('uint8')
        # grey_right = np.power(grey_right, 0.75).astype('uint8')

        # Use histogram equalisation to improve contrast

        # grey_left = cv2.equalizeHist(grey_left)
        # grey_right = cv2.equalizeHist(grey_right)
        #
        # cv2.imshow("Left equalised", grey_left)
        # cv2.imshow("Right equalised", grey_right)

        # Use CLAHE to improve contrast

        # grey_left = self.histogram_equaliser.apply(grey_left)
        # grey_right = self.histogram_equaliser.apply(grey_right)


        return images



    def distance(self, left, right, crop_disparity=True):

        grey_left, grey_right = self.preprocess(left, right)

        # Convert both to grayscale, as this is what the algorithm uses. Could also downscale to improve speed at the cost of quality.
        #
        # grey_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)  # (544, 1024)
        # grey_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)  # (544, 1024)
        #
        # # Perform pre-processing. Raise to the power, as this subjectively appears to improve disparity calculation.
        #
        # # cv2.imshow("Left", grey_left)
        # # cv2.imshow("Right", grey_right)
        #
        # grey_left = np.power(grey_left, 0.75).astype('uint8')
        # grey_right = np.power(grey_right, 0.75).astype('uint8')
        #
        # # Use histogram equalisation to improve contrast
        #
        # # grey_left = cv2.equalizeHist(grey_left)
        # # grey_right = cv2.equalizeHist(grey_right)
        # #
        # # cv2.imshow("Left equalised", grey_left)
        # # cv2.imshow("Right equalised", grey_right)
        #
        # # Use CLAHE to improve contrast
        #
        # grey_left = self.histogram_equaliser.apply(grey_left)
        # grey_right = self.histogram_equaliser.apply(grey_right)

        # cv2.imshow("Left equalised CLAHE", grey_left)
        # cv2.imshow("Right equalised CLAHE", grey_right)

        # Compute disparity from undistorted and rectified stereo images. This is returned scaled by 16.

        disparity = self.stereo_processor.compute(grey_left, grey_right) / 16.0

        # Filter out noise and speckles. Adjust parameters as needed. The higher disparityNoiseFilter the more aggressive it is.

        # disparity_noise_filter = 5
        # cv2.filterSpeckles(disparity, 0, 4000, 16 - disparity_noise_filter)

        # The range of values should then be (0, max disparity), but is instead (-1, max disparity - 1).
        # Fix this by using an initial threshold between 0 and max disparity as disparity = -1 means none is available.


        # Could also use adaptive thresholding: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html

        # _, disparity = cv2.threshold(disparity, 0, 128 * 16, cv2.THRESH_TOZERO)

        # Crop to remove left part where there is no disparity (as are is not seen by both cameras) and the bottom area
        # (the front of the car bonnet)

        if crop_disparity:

            disparity = disparity[0:396, 135:1024]

        return self.disparity_to_distance(disparity)

    def get_box_distance(self, distance, x, y, box_width, box_height):

        # Could use nanmedian. Peter also divides by 100; I'm not sure why.

        height, width = distance.shape  # (396, 889)

        x_offset = 135
        max_y = 396

        if x < x_offset:
            x = 0
            box_width -= (x_offset - x)

        if x + box_width > width:
            box_width = width - x

        if y + box_height > max_y:
            box_height = max_y - y

        result = np.nanmedian(distance[y: y + box_height, x: x + box_width])

        return result

if __name__ == "__main__":

    stereo = Stereo()

    left_path = os.path.join("images", "left", "1506942473.484027_L.png")
    right_path = os.path.join("images", "right", "1506942473.484027_R.png")


    left = cv2.imread(left_path, cv2.IMREAD_COLOR)
    right = cv2.imread(right_path, cv2.IMREAD_COLOR)

    cv2.imshow("Left", left)
    cv2.imshow("Right", right)

    start = time.time()

    distance = stereo.distance(left, right)

    print("SGBM took {:.2f} seconds".format(time.time() - start))

    # cv2.imshow("Disparity scaled", distance_to_image(disparity_scaled))

    cv2.waitKey(0)