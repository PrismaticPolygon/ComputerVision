import cv2
import numpy as np
import time

baseline_m = 0.2090607502           # in metres
focal_length_px = 399.9745178222656 # in pixels
focal_length_m = 4.8 / 1000         # in metres

image_centre_h = 262.0              # rectified image centre height, in pixels
image_centre_w = 474.5              # rectified image centre width, in pixels

# depth (metres) = baseline (metres) * focal_length (pixels) / disparity (pixels)

# https://github.com/vmarquet/opencv-disparity-map-tuner. Only works on Linux / Mac.

# max_disp has to be divisable by 16.

class WLS:

    def __init__(self):

        print("Initialising WLS...", end="")

        start = time.time()

        # Camera variables

        self.baseline_m = 0.2090607502  # in metres
        self.focal_length_px = 399.9745178222656  # in pixels
        self.focal_length_m = 4.8 / 1000  # in metres

        self.image_centre_h = 262.0  # rectified image centre height, in pixels
        self.image_centre_w = 474.5  # rectified image centre width, in pixels

        # Processor variables

        self.window_size = 5
        self.min_disparity = 0  # Minimum possible disparity value. Defaults to 0.
        self.num_disparities = 160  # Maximum disparity - minimum disparity. Defaults to 16.
        self.block_size = 5  # Matched blocked size. Must be an odd number >= 1. Normally between 3 and 11. Defaults to 3.
        self.P1 = 8 * 1 * self.window_size ** 2  # First parameter controlling disparity smoothness. The penalty on the disparity change by +- 1 between pixels.
        self.P2 = 32 * 1 * self.window_size ** 2  # Second parameter controlling disparity smoothness. The larger the values, the smoother the disparity. Must be greater than P1.
        self.disp_12_max_diff = 1  # The maximum allowed difference (in integer pixel units) in the left-right disparity check. Set to non-positive to disable.
        self.pre_filter_cap = 63  # Truncation value for the prefiltered image pixels
        self.uniqueness_ratio = 15  # Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct. Normally, a value within the 5-15 range is good enou
        self.speckle_window_size = 200  # The maximum size of smooth disparity regions to consider noise speckles and invalidate. Set to 0. to disable speckling. Should be somewhere between 50 - 200.
        self.speckle_range = 2  # maximum disparity variation within each connected component. If speckle filtering, set to positive. will be implicitly multiplied by 16. Normally, 1 or 2 is good enough.
        self.mode = cv2.STEREO_SGBM_MODE_HH  # Mode. Defaults to STEREO_SGBM_MODE_SGBM.

        # https://docs.opencv.org/master/d2/d85/classcv_1_1StereoSGBM.html

        self.left_matcher = cv2.StereoSGBM_create(
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

        self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)

        # Histogram variables

        self.clip_limit = 2.0
        self.tile_grid_size = (8, 8)

        # https://docs.opencv.org/master/d5/daf/tutorial_py_histogram_equalization.html
        self.histogram_equaliser = cv2.createCLAHE(
            clipLimit=self.clip_limit,  #
            tileGridSize=self.tile_grid_size  #
        )

        # WLS variables

        self.lmbda = 80000
        self.sigma = 1.2
        self.visual_multiplier = 1.0

        # http://timosam.com/python_opencv_depthimage

        self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=self.left_matcher)

        self.wls_filter.setLambda(self.lmbda)
        self.wls_filter.setSigmaColor(self.sigma)

        print("DONE ({:.2f}s)".format(time.time() - start))

    def preprocess(self, left, right):

        for image in (left, right):

            # Convert to greyscale

            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            image = self.histogram_equaliser.apply(image)

        return left, right

    def calculate(self, left, right, crop_disparity=True):

        left, right = self.preprocess(left, right)

        disparity_left = self.left_matcher.compute(left, right).astype(np.float32)/16
        disparity_right = self.right_matcher.compute(right, left).astype(np.float32)/16

        disparity = self.wls_filter.filter(disparity_left, left, None, disparity_right)

        if crop_disparity:

            disparity = disparity[0:396, 135:1024]

        return disparity

    def to_image(self, disparity):

        image = cv2.normalize(src=disparity, dst=disparity, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)

        return np.uint8(image)


if __name__ == "__main__":

    imgL = cv2.imread("images/left/1506942473.484027_L.png", cv2.IMREAD_COLOR)
    imgR = cv2.imread("images/right/1506942473.484027_R.png", cv2.IMREAD_COLOR)

    wls = WLS()

    start = time.time()

    disparity = wls.calculate(imgL, imgR)

    print("WLS took {:.2f} seconds".format(time.time() - start))

    cv2.imshow('Disparity Map', wls.to_image(disparity))

    cv2.waitKey(0)

    # window_size = 3  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
    #
    # left_matcher = cv2.StereoSGBM_create(
    #     minDisparity=0,
    #     numDisparities=160,  # max_disp has to be dividable by 16 f. E. HH 192, 256
    #     blockSize=5,
    #     P1=8 * 3 * window_size ** 2,
    #     P2=32 * 3 * window_size ** 2,
    #     disp12MaxDiff=1,
    #     uniquenessRatio=15,
    #     speckleWindowSize=0,
    #     speckleRange=2,
    #     preFilterCap=63,
    #     mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    # )
    #
    # right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    #
    # lmbda = 80000
    # sigma = 1.2
    # visual_multiplier = 1.0
    #
    # wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    # wls_filter.setLambda(lmbda)
    # wls_filter.setSigmaColor(sigma)
    #
    # print('computing disparity...')
    # displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
    # dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
    # displ = np.int16(displ)
    # dispr = np.int16(dispr)
    # filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!
    #
    # # cv2.imshow("Disparity", filteredImg)
    # #
    # # cv2.waitKey(0)
    #
    # # /filteredImg, out=np.zeros_like(filteredImg),
    #
    # filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    # filteredImg = np.uint8(filteredImg)
    # cv2.imshow('Disparity Map', filteredImg)
    #
    # # cv2.imwrite("Disparity map WLS.png", filteredImg)
    #
    # cv2.waitKey()
    # cv2.destroyAllWindows()


    # disparity = calculate(left, right)
    #
    # filtered_img = cv2.normalize(disparity, disparity, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    #
    # print(filtered_img)
    #
    # # It's all just... white.
    # # But it works!
    #
    # cv2.imshow("Distance", filtered_img)
    #
    # cv2.waitKey(0)
