import numpy as np
import argparse
import time
import cv2
import os

#https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/

image_path = "TTBB-durham-02-10-17-sub10/left-images/1506943009.480358_L.png"
weights_path = "yolo-coco/yolov3.weights"
config_path = "yolo-coco/yolov3.cfg"

# Ultimately, we'll want to have a single method, right-images?
# We pass in an image and them it does all of the rest.

labels = labels()
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

image = cv2.imread(image_path)
H, W, _ = image.shape

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Construct a blob from the input image and then perform a forward pass of the YOLO object detector,
# producing bounding boxes and associated probabilities

blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (1024, 544), swapRB=True, crop=False)

net.setInput(blob)
start = time.time()

layerOutputs = net.forward(ln)
end = time.time()

print("YOLO took {:.2f} seconds".format(end - start))   # 6.35 seconds. Very not good.

boxes = []          # Bounding boxes around the object
confidences = []    # Confidences. Object below 0.5 are filtered out
class_IDs = []      # The detected objects' class labels

for output in layerOutputs:

    for detection in output:

        scores = detection[5:]
        class_ID = np.argmax(scores)
        confidence = scores[class_ID]

        if confidence > 0.5:

            # YOLO returns the center of the bounding box followed by height and width.
            box = detection[0:4] * np.array([W, H, W, H])   # Scale bounding box so that we can display them properly
            (centerX, centerY, width, height) = box.astype("int")   # Extract coordinates and dimensions

            # But then again... if we have the box, we don't have to perform it for the whole image.
            # So that's an improvement.

            x = int(centerX - (width / 2))  # Derive the top-left-images x
            y = int(centerY - (width / 2))  # Derive the top-left-images y

            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            class_IDs.append(class_ID)

# Apply non-maxima suppresion. NMS suppresses significantly overlapping bounding boxes, keeping only the most confident ones.
idxes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

if len(idxes) > 0:

    for i in idxes.flatten():

        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])

        color = [int(c) for c in colors[class_IDs[i]]]

        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        label = '{} ({:2f}%)'.format(labels[class_IDs[i]], confidences[i])  # construct label

        cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)





