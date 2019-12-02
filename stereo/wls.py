import cv2
import numpy as np
from stereo.disparity import Disparity


class WLS(Disparity):

    def __init__(self):

        super().__init__()

        window_size = 3

        # https://docs.opencv.org/master/d2/d85/classcv_1_1StereoSGBM.html

        self.left_matcher = cv2.StereoSGBM_create(
            minDisparity=0,                 # Minimum possible disparity value. Defaults to 0.
            numDisparities=160,             # Maximum disparity - minimum disparity. Defaults to 16.
            blockSize=5,                    # Matched blocked size. Must be an odd number >= 1. Normally between 3 and 11. Defaults to 3.
            P1=8 * 1 * window_size ** 2,    # First parameter controlling disparity smoothness. The penalty on the disparity change by +- 1 between pixels.
            P2=32 * 1 * window_size ** 2,   # Second parameter controlling disparity smoothness. The larger the values, the smoother the disparity. Must be greater than P1
            disp12MaxDiff=1,                # The maximum allowed difference (in integer pixel units) in the left-right disparity check. Set to non-positive to disable.
            preFilterCap=63,                # Truncation value for the prefiltered image pixels
            uniquenessRatio=15,             # Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct. Normally, a value within the 5-15 range is good enough
            speckleWindowSize=200,          # The maximum size of smooth disparity regions to consider noise speckles and invalidate. Set to 0. to disable speckling. Should be somewhere between 50 - 200.
            speckleRange=2,                 # maximum disparity variation within each connected component. If speckle filtering, set to positive. will be implicitly multiplied by 16. Normally, 1 or 2 is good enough.
            mode=cv2.STEREO_SGBM_MODE_HH    # Mode. Defaults to STEREO_SGBM_MODE_SGBM.
        )

        self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)

        # http://timosam.com/python_opencv_depthimage
        # https://docs.opencv.org/3.4/d9/d51/classcv_1_1ximgproc_1_1DisparityWLSFilter.html#ab26fa73918b84d1a0e57951e00704708

        self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=self.left_matcher)

        self.wls_filter.setLambda(8000)     # The amount of regularisation. Large values force filtered disparity map edges to adhere more to source image edges. Typically 8000.
        self.wls_filter.setSigmaColor(1.6)  # How sensitive the filtering process is to source image edges. Large values can lead to disparity leakage through low-contrast edges. Small values can make the filter too sensitive to noise and textures in the source image. Typical values range fmro 0.8 to 2.0

    def calculate(self, left, right):

        left = self.preprocess(left)
        right = self.preprocess(right)

        disparity_left = self.left_matcher.compute(left, right).astype(np.float32) / 16.
        disparity_right = self.right_matcher.compute(right, left).astype(np.float32) / 16.

        return self.wls_filter.filter(disparity_left, left, None, disparity_right)
