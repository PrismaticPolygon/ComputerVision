import cv2
import numpy as np


class Stereo:

    def __init__(self):

        self.offset_x = 135

        self.baseline_m = 0.2090607502              # in metres
        self.focal_length_px = 399.9745178222656    # in pixels
        self.focal_length_m = 4.8 / 1000            # in metres

        self.image_centre_h = 262.0  # rectified image centre height, in pixels
        self.image_centre_w = 474.5  # rectified image centre width, in pixels

    def to_image(self):

        image = cv2.normalize(src=self.disparity, dst=self.disparity, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)

        return np.uint8(image)

    def get_box_distance(self, disparity, x, y, box_width, box_height):

        # Lol. How did I end up with that ridiculous value?

        max_y = 396

        x = x - self.offset_x

        if x < 0:               # If we're in the left region with no disparity

            box_width = box_width - (self.offset_x - x)
            x = 0

        if y + box_height > max_y:  # If we're in the bottom region with no disparity

            box_height = max_y - y

        disparity_box = disparity[y: y + box_height, x: x + box_width]

        disparity_median = np.nanmedian(disparity_box)

        if disparity_median != 0:

            distance = (self.focal_length_px / self.baseline_m) / disparity_median
            distance = distance / 100

            return distance

        else:

            return 0
