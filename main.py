import os
import cv2
from my_yolo import YOLO
from my_stereo_disparity import Stereo

MASTER_PATH_TO_DATASET = "images"
LEFT_DIR = "left"
RIGHT_DIR = "right"

def images(start=""):

    left_path, right_path = os.path.join(MASTER_PATH_TO_DATASET, LEFT_DIR), os.path.join(MASTER_PATH_TO_DATASET, RIGHT_DIR)

    for left_file in os.listdir(left_path):

        if len(start) > 0 and not start in left_file:

            continue

        start = ""
        right_file = left_file.replace("_L", "_R")
        left_file_path = os.path.join(left_path, left_file)
        right_file_path = os.path.join(right_path, right_file)

        if left_file[-4:] == ".png" and os.path.isfile(right_file_path):

            yield left_file_path, right_file_path

yolo = YOLO()
stereo = Stereo()

crop_disparity = True
pause_playback = False

for left_file_path, right_file_path in images():

    print(left_file_path)

    left = cv2.imread(left_file_path, cv2.IMREAD_COLOR)
    right = cv2.imread(right_file_path, cv2.IMREAD_COLOR)

    distance_map = stereo.distance(left, right)
    class_IDs, confidences, boxes = yolo.predict(left)

    nearest_object = 0

    for i, box in enumerate(boxes):

        x, y, width, height = box

        class_id = class_IDs[i]
        confidence = confidences[i]
        distance = stereo.get_box_distance(distance_map, x, y, width, height)

        yolo.draw_prediction(left, class_id, confidence, box, distance)

    cv2.imshow("Result", left)

    key = cv2.waitKey(40 * (not pause_playback)) & 0xFF  # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)

    if key == ord('x'):  # exit

        break  # exit

    elif key == ord('c'):  # crop

        crop_disparity = not crop_disparity

    elif key == ord(' '):  # pause (on next frame)

        pause_playback = not pause_playback