import os
import cv2
from yolo import YOLO
from stereo.wls import WLS
import numpy as np

# TODO: investigate "statistics" and "heuristics" to produce an accurate object range.
# TODO: investigate "frame-to-frame temporal constraints"
# TODO: process / project only the region in front of the car
# TODO: test on more images
# TODO: further investigate pre- and post- filtering algorithms.

MASTER_PATH_TO_DATASET = "images"
LEFT_DIR = "left"
RIGHT_DIR = "right"


def images(start=""):

    left_path, right_path = os.path.join(MASTER_PATH_TO_DATASET, LEFT_DIR), os.path.join(MASTER_PATH_TO_DATASET, RIGHT_DIR)

    for left_file in sorted(os.listdir(left_path)):

        if len(start) > 0 and not start in left_file:

            continue

        start = ""
        right_file = left_file.replace("_L", "_R")
        left_file_path = os.path.join(left_path, left_file)
        right_file_path = os.path.join(right_path, right_file)

        if left_file[-4:] == ".png" and os.path.isfile(right_file_path):

            yield left_file, right_file, left_file_path, right_file_path


yolo = YOLO()
wls = WLS()

crop_disparity = True
pause_playback = False

for left_file, right_file, left_file_path, right_file_path in images():

    print("")
    print(left_file)

    left = cv2.imread(left_file_path, cv2.IMREAD_COLOR)
    right = cv2.imread(right_file_path, cv2.IMREAD_COLOR)

    disparity_map = wls.calculate(left, right)

    # distance_map = stereo.distance(left, right)
    class_IDs, confidences, boxes = yolo.predict(left)

    smallest_distance = np.inf

    for i, box in enumerate(boxes):

        class_id = class_IDs[i]
        confidence = confidences[i]

        distance = wls.get_box_distance(disparity_map, box)

        if distance < smallest_distance:

            smallest_distance = distance    # There we go.

        yolo.draw_prediction(left, class_id, confidence, box, distance)

    if smallest_distance == np.inf:

        smallest_distance = 0

    print(right_file + " : nearest detected scene object ({:.2f}m)".format(smallest_distance))

    cv2.imshow("Disparity", wls.to_image(disparity_map))
    cv2.imshow("Result", left)

    out_image_path = os.path.join("output", "images", left_file)
    out_disparity_path = os.path.join("output", "disparities", left_file)

    cv2.imwrite(out_disparity_path, wls.to_image(disparity_map))
    cv2.imwrite(out_image_path, left)

    key = cv2.waitKey(40 * (not pause_playback)) & 0xFF  # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)

    if key == ord('x'):  # exit

        break  # exit

    elif key == ord(' '):  # pause (on next frame)

        pause_playback = not pause_playback
