import os
import cv2
import shutil
import numpy as np
from yolo import YOLO
from stereo.wls import WLS

MASTER_PATH_TO_DATASET = "TTBB-durham-02-10-17-sub10"
LEFT_DIR = "left-images"
RIGHT_DIR = "right-images"


def images(start=""):
    """
    A generator yielding file names and paths for both left and right images
    """

    left_path, right_path = os.path.join(MASTER_PATH_TO_DATASET, LEFT_DIR), os.path.join(MASTER_PATH_TO_DATASET, RIGHT_DIR)

    for left_file in sorted(os.listdir(left_path)):

        if len(start) > 0 and start not in left_file:

            continue

        start = ""
        right_file = left_file.replace("_L", "_R")
        left_file_path = os.path.join(left_path, left_file)
        right_file_path = os.path.join(right_path, right_file)

        if left_file[-4:] == ".png" and os.path.isfile(right_file_path):

            yield left_file, right_file, left_file_path, right_file_path


yolo = YOLO()
wls = WLS(histogram="CLAHE")

PAUSE_PLAYBACK = False
SAVE_IMAGES = True

if SAVE_IMAGES:

    image_output_path = os.path.join("output", "images")
    disparities_output_path = os.path.join("output", "disparities")

    if not os.path.exists("output"):

        os.mkdir("output")

    if os.path.exists(image_output_path):

        shutil.rmtree(image_output_path)

    os.mkdir(image_output_path)

    if os.path.exists(disparities_output_path):

        shutil.rmtree(disparities_output_path)

    os.mkdir(disparities_output_path)


for left_file, right_file, left_file_path, right_file_path in images(start="1506943035.478214_L.png"):

    print("")
    print(left_file)

    left = cv2.imread(left_file_path, cv2.IMREAD_COLOR)
    right = cv2.imread(right_file_path, cv2.IMREAD_COLOR)

    disparity_map = wls.calculate(left, right)

    left_filtered = cv2.bilateralFilter(left, 5, 25, 25)

    class_IDs, confidences, boxes = yolo.predict(left_filtered[0:390, :])

    smallest_distance = np.inf

    for i, box in enumerate(boxes):

        x, y, width, height = box

        if x + width > 135:  # No disparity information in this range.

            confidence = confidences[i]
            class_id = class_IDs[i]

            distance = wls.get_box_distance(disparity_map, box)

            if distance < smallest_distance:

                smallest_distance = distance

            yolo.draw_prediction(left, class_id, confidence, box, distance)

    if smallest_distance == np.inf:

        smallest_distance = 0

    print(right_file + " : nearest detected scene object ({:.1f}m)".format(smallest_distance))

    cv2.imshow("Disparity", wls.to_image(disparity_map))
    cv2.imshow("Result", left)

    if SAVE_IMAGES:

        out_image_path = os.path.join("output", "images", left_file)
        out_disparity_path = os.path.join("output", "disparities", left_file)

        cv2.imwrite(out_disparity_path, wls.to_image(disparity_map))
        cv2.imwrite(out_image_path, left)

    key = cv2.waitKey(40 * (not PAUSE_PLAYBACK)) & 0xFF  # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)

    if key == ord('x'):    # exit

        break  # exit

    elif key == ord(' '):  # pause (on next frame)

        PAUSE_PLAYBACK = not PAUSE_PLAYBACK
