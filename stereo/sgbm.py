import cv2
import numpy as np
from stereo.disparity import Disparity


class SGBM(Disparity):
    """
    Class encapsulating the Semi-Global Block Matching stereo filter.
    """

    def __init__(self):

        super().__init__()

        # Processor variables

        window_size = 3

        self.sgbm_filter = cv2.StereoSGBM_create(
            minDisparity=0,                 # Minimum possible disparity value. Defaults to 0.
            numDisparities=self.max_disparity,             # Maximum disparity - minimum disparity. Defaults to 16.
            blockSize=5,                    # Matched blocked size. Must be an odd number >= 1. Normally between 3 and 11. Defaults to 3.
            P1=8 * 1 * window_size ** 2,    # First parameter controlling disparity smoothness. The penalty on the disparity change by +- 1 between pixels.
            P2=32 * 1 * window_size ** 2,   # Second parameter controlling disparity smoothness. The larger the values, the smoother the disparity. Must be greater than P1
            disp12MaxDiff=1,                # The maximum allowed difference (in integer pixel units) in the left-images-right-images disparity check. Set to non-positive to disable.
            preFilterCap=63,                # Truncation value for the prefiltered image pixels
            uniquenessRatio=15,             # Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct. Normally, a value within the 5-15 range is good enough
            speckleWindowSize=200,          # The maximum size of smooth disparity regions to consider noise speckles and invalidate. Set to 0. to disable speckling. Should be somewhere between 50 - 200.
            speckleRange=2,                 # maximum disparity variation within each connected component. If speckle filtering, set to positive. will be implicitly multiplied by 16. Normally, 1 or 2 is good enough.
            mode=cv2.STEREO_SGBM_MODE_HH  # Mode. Defaults to STEREO_SGBM_MODE_SGBM.
        )

    def calculate(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:

        left = self.preprocess(left)
        right = self.preprocess(right)

        disparity = self.sgbm_filter.compute(left, right)

        return self.postprocess(disparity)
