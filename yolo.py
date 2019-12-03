################################################################################

# This script is based off work by Toby Breckon, toby.breckon@durham.ac.uk,
# available at:
#
#       https://github.com/tobybreckon/python-examples-cv/blob/master/yolo.py
#
# under the LGPL licence. It implements the You Only Look Once (YOLO)
# object detection algorithm described in:
#
# Redmon, J., & Farhadi, A. (2018). Yolov3: An incremental improvement. arXiv preprint arXiv:1804.02767.
#
# and available at https://pjreddie.com/media/files/papers/YOLOv3.pdf
#
# The script expects to find the following files in a directory named "yolo-coco":
#
# https://pjreddie.com/media/files/yolov3.weights
# https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg?raw=true
# https://github.com/pjreddie/darknet/blob/master/data/coco.names?raw=true

################################################################################

import cv2
import os
import numpy as np

COCO_NAMES_PATH = os.path.join("yolo-coco", "coco.names")
COCO_CONFIG_PATH = os.path.join("yolo-coco", "yolov3.cfg")
COCO_WEIGHTS_PATH = os.path.join("yolo-coco", "yolov3.weights")



class YOLO:
    """
    Class that encapsulates You Only Look Once (YOLO), a state-of-the-art object detection technique.
    """

    def __init__(self):

        self.confidence_T = 0.55  # Confidence threshold
        self.nms_T = 0.4  # Non-maxima suppression threshold

        self.input_height = 416  # Height of the network's input image
        self.input_width = 416  # Width of the network's input image

        with open(COCO_NAMES_PATH) as coco_names:

            self.labels = coco_names.read().rstrip('\n').split('\n')

        self.net = cv2.dnn.readNetFromDarknet(COCO_CONFIG_PATH, COCO_WEIGHTS_PATH)

        self.layer_names = self.net.getLayerNames()

        self.output_layer_names = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        # defaults DNN_BACKEND_INFERENCE_ENGINE if Intel Inference Engine lib available or DNN_BACKEND_OPENCV otherwise

        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)

        # change to cv2.dnn.DNN_TARGET_CPU (slower) if this causes issues (should fail gracefully if OpenCL not available)

        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

        # print(" DONE ({:.2f}s)".format(time.time() - start))  # 6.35 seconds. Very not good.

    def get_output_names(self):

        layer_names = self.net.getLayerNames()

        return [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def draw_prediction(self, image: np.ndarray, class_id: int, confidence: float, box: tuple, distance: float) -> None:
        """
        Draw a prediction on an image, as as a bounding box labelled with the class of object, its distance from
        the camera, and the certainty of the prediction.
        """

        # If the WHOLe box is in that range.

        color = (0, 0, 255)
        left, top, width, height = box

        label = self.labels[class_id]

        if label != "train":    # Curious number of false positives of trains

            cv2.rectangle(image, (left, top), (left + width, top + height), color, 3)

            label = '{} - {:.1f}m ({:.0f}%)'.format(label, distance, confidence * 100)  # construct label

            cv2.putText(image, label, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def predict(self, frame: np.ndarray):
        """
        Given an image, generate predictions for it using YOLO.
        """

        # Create a 4D tensor (OpenCV "blob") from the image frame (with pixels scaled  0 -> and the image resized)

        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (self.input_width, self.input_height), [0, 0, 0], 1, crop=False)

        # Set the input to the CNN

        self.net.setInput(blob)

        # Run forward inference to get output of the final layer

        results = self.net.forward(self.output_layer_names)

        class_IDs, confidences, boxes = self.postprocess(frame, results)

        return class_IDs, confidences, boxes

    def postprocess(self, frame: np.ndarray, results):
        """
        Scan through all the bounding boxes output from the network.
        Remove those with lose confidence. Assign a box class label as the class with the highest score
        Construct a list of bounding boxes, class labels, and confidence scores.
        """

        H, W, _ = frame.shape

        boxes = []          # Bounding boxes around the object
        confidences = []    # Confidences. Object below 0.5 are filtered out
        class_IDs = []      # The detected objects' class labels

        for result in results:

            for detection in result:

                scores = detection[5:]

                class_ID = np.argmax(scores)    # So this is wrong. Or my scores are... maybe it's the wrong structure?

                confidence = scores[class_ID]   # This is the error here.

                if confidence > self.confidence_T:

                    center_x = int(detection[0] * W)
                    center_y = int(detection[1] * H)
                    width = int(detection[2] * W)
                    height = int(detection[3] * H)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)

                    class_IDs.append(class_ID)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        class_IDs_nms = []
        confidences_nms = []
        boxes_nms = []

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_T, self.nms_T)

        # Perform non-maximum suppression to eliminate redundant overlapping boxes with lower confidence.
        for i in indices:

            i = i[0]

            class_IDs_nms.append(class_IDs[i])
            confidences_nms.append(confidences[i])
            boxes_nms.append(boxes[i])

        return class_IDs_nms, confidences_nms, boxes_nms