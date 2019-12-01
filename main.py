import os
import cv2
from my_yolo import YOLO
from my_stereo_disparity import Stereo

root = "images"
left_dir = "left"
right_dir = "right"

def images(start=""):

    left_path, right_path = os.path.join(root, left_dir), os.path.join(root, right_dir)

    for left_file in os.listdir(left_path):

        if len(start) > 0 and not start in left_file:

            continue

        start = ""
        right_file = left_file.replace("_L", "_R")
        left_file_path = os.path.join(left_path, left_file)
        right_file_path = os.path.join(right_path, right_file)

        if left_file[-4:] == ".png" and os.path.isfile(right_file_path):

            yield cv2.imread(left_file_path, cv2.IMREAD_COLOR), cv2.imread(right_file_path, cv2.IMREAD_COLOR)

yolo = YOLO()
stereo = Stereo()
# And we'll make a stereo object, naturally.

for left, right in images():

    distance_map = stereo.distance(left, right)
    class_IDs, confidences, boxes = yolo.predict(left)

    for i, box in enumerate(boxes):

        x, y, width, height = box

        class_id = class_IDs[i]
        confidence = confidences[i]
        distance = stereo.get_box_distance(distance_map, x, y, width, height)

        yolo.draw_prediction(left, class_id, confidence, box, distance)

    cv2.imshow("Result", left)

    cv2.waitKey(0)