import cv2
import os

video = cv2.VideoWriter("ffgt86.avi", 0, 1, (1024, 544))

for image in sorted(os.listdir("annotated")):

    frame = cv2.imread(os.path.join("annotated", image))

    video.write(frame)

cv2.destroyAllWindows()
video.release()
