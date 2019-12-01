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

# https://docs.opencv.org/master/d2/d85/classcv_1_1StereoSGBM.html
window_size = 5
min_disparity = 0                # Minimum possible disparity value. Defaults to 0.
num_disparities = 160            # Maximum disparity - minimum disparity. Defaults to 16.
block_size = 5            # Matched blocked size. Must be an odd number >= 1. Normally between 3 and 11. Defaults to 3.
P1 = 8 * 1 * window_size ** 2    # First parameter controlling disparity smoothness. The penalty on the disparity change by +- 1 between pixels.
P2 = 32 * 1 * window_size ** 2   # Second parameter controlling disparity smoothness. The larger the values, the smoother the disparity. Must be greater than P1.
disp_12_max_diff = 1             # The maximum allowed difference (in integer pixel units) in the left-right disparity check. Set to non-positive to disable.
pre_filter_cap = 63               # Truncation value for the prefiltered image pixels
uniqueness_ratio = 15            # Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct. Normally, a value within the 5-15 range is good enou
speckle_window_size = 200         # The maximum size of smooth disparity regions to consider noise speckles and invalidate. Set to 0. to disable speckling. Should be somewhere between 50 - 200.
speckle_range = 2                # maximum disparity variation within each connected component. If speckle filtering, set to positive. will be implicitly multiplied by 16. Normally, 1 or 2 is good enough.
mode = cv2.STEREO_SGBM_MODE_HH   # Mode. Defaults to STEREO_SGBM_MODE_SGBM.

left_matcher = cv2.StereoSGBM_create(
    minDisparity=min_disparity,
    numDisparities=num_disparities,
    blockSize=block_size,
    P1=P1,
    P2=P2,
    disp12MaxDiff=disp_12_max_diff,
    preFilterCap=pre_filter_cap,
    uniquenessRatio=uniqueness_ratio,
    speckleWindowSize=speckle_window_size,
    speckleRange=speckle_range,
    mode=mode
)

right_matcher = cv2.StereoSGBM_create(
    minDisparity=min_disparity,
    numDisparities=num_disparities,
    blockSize=block_size,
    P1=P1,
    P2=P2,
    disp12MaxDiff=disp_12_max_diff,
    preFilterCap=pre_filter_cap,
    uniquenessRatio=uniqueness_ratio,
    speckleWindowSize=speckle_window_size,
    speckleRange=speckle_range,
    mode=mode
)

lmbda = 80000
sigma = 1.2
visual_multiplier = 1.0

# http://timosam.com/python_opencv_depthimage
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)

print(wls_filter.filter)    # I don't know enough about how this works.

# Just waiting for my seg fault...

# Interestingly, it seems to be these two that causes the error.

wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

def disparity_to_distance(disparity):

   # https://stackoverflow.com/questions/26248654/how-to-return-0-with-divide-by-zero
   return np.divide(focal_length_px * baseline_m, disparity, out=np.zeros_like(disparity), where=disparity > 0)

def distance_to_image(distance):

    image = cv2.normalize(src=distance, dst=distance, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)

    return np.uint8(image)


def calculate(left, right, crop_disparity=True):

    # Convert both to grayscale, as this is what the algorithm uses. Could also downscale to improve speed at the cost of quality.

    grey_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)      # (544, 1024)
    grey_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)    # (544, 1024)

    cv2.imshow("Left", grey_left)
    cv2.imshow("Right", grey_right)

    disparity_left = left_matcher.compute(grey_left, grey_right).astype(np.float32)/16
    disparity_right = right_matcher.compute(grey_right, grey_left).astype(np.float32)/16

    disparity = wls_filter.filter(disparity_left, grey_left, None, disparity_right)

    print(disparity)

    return disparity
    # # Wait. What?
    #
    # # So it's definitely the filter itself.
    #
    # if crop_disparity:
    #
    #     disparity = disparity[0:396, 135:1024]
    #
    # return disparity_to_distance(disparity)


if __name__ == "__main__":

    print("Running")

    imgL = cv2.imread("images/left/1506942473.484027_L.png", cv2.IMREAD_COLOR)
    imgR = cv2.imread("images/right/1506942473.484027_R.png", cv2.IMREAD_COLOR)
#
    # image = cv2.medianBlur(image, 5)

    imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    clip_limit = 2.0
    tile_grid_size = (8, 8)

    # https://docs.opencv.org/master/d5/daf/tutorial_py_histogram_equalization.html
    histogram_equaliser = cv2.createCLAHE(
        clipLimit=clip_limit,  #
        tileGridSize=tile_grid_size  #
    )

    # And crop. The near ones are throwing off the value too
    # much.

    imgL = histogram_equaliser.apply(imgL)
    imgR = histogram_equaliser.apply(imgR)

    start = time.time()

    window_size = 3  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=160,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=5,
        P1=8 * 3 * window_size ** 2,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    lmbda = 80000
    sigma = 1.2
    visual_multiplier = 1.0

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    print('computing disparity...')
    displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, imgL, None, dispr) # important to put "imgL" here!!!

    filteredImg = filteredImg[0:396, 180:1024]

    # baseline_m = 0.2090607502  # in metres
    # focal_length_px = 399.9745178222656  # in pixels
    # focal_length_m = 4.8 / 1000  # in metres
    #
    # # This is when we have the wrong types
    #
    # filteredImg = np.divide(focal_length_px * baseline_m, filteredImg, out=np.zeros_like(filteredImg),
    #                         where=filteredImg > 0)

    # Remember: this is disparity, NOT depth.
    # Let's try with depth now.

    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    filteredImg = np.uint8(filteredImg)
    cv2.imshow('Disparity Map', filteredImg)

    cv2.imwrite("Disparity map WLS.png", filteredImg)

    cv2.waitKey()
    cv2.destroyAllWindows()


    # disparity = calculate(left, right)
    #
    # print("WLS took {:.2f} seconds".format(time.time() - start))
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
