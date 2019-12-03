import cv2
import os

left_path = "output/disparities/1506942473.484027_L.png"

left = cv2.imread(left_path, cv2.IMREAD_COLOR)

cv2.imshow("left", left)

# I'm being stupid. It's the disparity that has to be be modiefied

cv2.waitKey(0)
