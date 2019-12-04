import cv2
import os
import numpy as np


def images():
    """
    Generator that yields paths to output disparity and annotated TTBB-durham-02-10-17-sub10
    """

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

codec = cv2.VideoWriter_fourcc("M", "J", "P", "G")
fps = 2

# disparity_video = cv2.VideoWriter(disparity_video_path, codec, fps, frame_shape)
# image_video = cv2.VideoWriter(image_video_path, codec, fps, frame_shape)
combined_video = cv2.VideoWriter(combined_video_path, codec, fps, (2048, 544))

i = 0

for image_path, disparity_path in images():

    if i < 170:

        image_frame = cv2.imread(image_path, cv2.IMREAD_COLOR)
        disparity_frame = cv2.imread(disparity_path, cv2.IMREAD_COLOR)

        combined = np.hstack((disparity_frame, image_frame))

        combined_video.write(combined)

        i += 1

    else:

        break

combined_video.release()