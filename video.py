import cv2
import os
import numpy as np

def images():

    images_path = os.path.join("output", "images")
    disparities_path = os.path.join("output", "disparities")

    for filename in sorted(os.listdir(images_path)):

        image_path = os.path.join(images_path, filename)
        disparity_path = os.path.join(disparities_path, filename)

        yield image_path, disparity_path


disparity_video_path = os.path.join("output", "videos", "disparity.avi")
image_video_path = os.path.join("output", "videos", "annotated.avi")
combined_video_path = os.path.join("output", "videos", "ffgt86.avi")

frame_shape = (1024, 544)

disparity_video = cv2.VideoWriter(disparity_video_path, 0, 10, frame_shape)
image_video = cv2.VideoWriter(image_video_path, 0, 10, frame_shape)
combined_video = cv2.VideoWriter(combined_video_path, 0, 1, frame_shape)

for image_path, disparity_path in images():

    image_frame = cv2.imread(image_path)
    disparity_frame = cv2.imread(disparity_path)    # That's weird. Why is it by3?

    # print(disparity_frame.shape)

    image_video.write(image_frame)
    disparity_video.write(disparity_frame)

    # disparity_frame = cv2.cvtColor(disparity_frame, cv2.COLOR_GRAY2BGR)

    # combined = np.hstack((disparity_frame, image_frame))
    #
    # combined_video.write(combined)


image_video.release()
disparity_video.release()
combined_video.release()
