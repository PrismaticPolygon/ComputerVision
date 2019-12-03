import cv2

left_path = "output/disparities/1506942473.484027_L.png"

left = cv2.imread(left_path, cv2.IMREAD_COLOR)

cv2.imshow("left", left)

cv2.waitKey(0)
