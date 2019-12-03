import cv2
import numpy as np


class Disparity:

    @staticmethod
    def to_image(disparity):

        image = cv2.normalize(src=disparity, dst=disparity, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)

        return np.uint8(image)

    def __init__(self, histogram="default", clip_limit=3, tile_grid_size=(16, 16)):

        self.histogram = histogram
        self.offset_x = 160
        self.max_y = 395

        self.baseline_m = 0.2090607502              # in metres
        self.focal_length_px = 399.9745178222656    # in pixels
        self.focal_length_m = 0.0048                # in metres

        self.left_camera_matrix = np.array([
            [399.9745178222656,               0.0, 474.5, 0.0],
            [              0.0, 399.9745178222656, 262.0, 0.0],
            [              0.0,               0.0,   1.0, 0.0]
        ])

        self.right_camera_matrix = np.array([
            [399.9745178222656,               0.0, 474.5, -83.61897277832031],
            [              0.0, 399.9745178222656, 262.0,                0.0],
            [              0.0,               0.0,   1.0,                0.0]
        ])

        # https://docs.opencv.org/3.4/d6/db6/classcv_1_1CLAHE.html
        # https://docs.opencv.org/master/d5/daf/tutorial_py_histogram_equalization.html

        if histogram == "CLAHE":

            self.histogram_equaliser = cv2.createCLAHE(
                clipLimit=clip_limit,          # Sets threshold for contrast limiting
                tileGridSize=tile_grid_size    # Input image will be divided into equally sized rectangular tiles
            )

    def get_box_distance(self, disparity, box):

        x, y, width, height = box

        if x < self.offset_x:                   # If we're in the left region with no disparity

            width = width + x - self.offset_x
            x = self.offset_x

        if y + height > self.max_y:             # If we're in the bottom region with no disparity

            height = self.max_y - y

        # tar_left, tar_right = int(x, + width / 3)

        disparity_box = disparity[y: y + height, x: x + width]

        disparity_box_average = np.average(disparity_box)

        disparity_box[disparity_box < 0] = 0    # Remove values below 0

        distance = self.focal_length_px * self.baseline_m / (disparity_box_average * 4)


        # distance_box = np.divide(self.focal_length_px * self.baseline_m, disparity_box, out=np.zeros_like(disparity_box),
        #                  where=disparity_box > 0)

        # disparity_median = np.nanmedian(disparity_box)

        # depth (metres) = baseline (metres) * focal_length (pixels) / disparity (pixels)

        # SO how do we get Nan values out of this?

        return distance

        # https://stackoverflow.com/questions/26248654/how-to-return-0-with-divide-by-zero
        # return np.divide(self.focal_length_px * self.baseline_m, disparity, out=np.zeros_like(disparity),
        #                  where=disparity > 0)

        # So this gets the distance.
        #
        #
        # if disparity_median != 0:
        #
        #     distance = (self.baseline_m * self.focal_length_px) / disparity_median
        #
        #     return distance
        #
        # else:
        #
        #     return 0

    # Now we look into bilateral filters.
    # Normally applies a Laplacian of Gaussian, approximated
    # by a difference of Gaussian.

    def postprocess(self, image):

        pass

    def preprocess(self, image):

        # Convert to greyscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#bilateralfilter
        image = cv2.bilateralFilter(image, 5, 10, 10)

        if self.histogram == "CLAHE":

            image = self.histogram_equaliser.apply(image)

        else:

            image = cv2.equalizeHist(image)

        # Appears to subjectively improve disparity calculation

        # image = np.power(image, 0.75)

        return image
