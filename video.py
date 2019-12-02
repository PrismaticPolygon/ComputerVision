import cv2
import os

# So I want to produce three videos. One disparity, one annotated, and one both.
# The both is for submissions.
# The second is for me.

fps = 30


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

disparity_video = cv2.VideoWriter(disparity_video_path, 0, 1, (1024, 544))
image_video = cv2.VideoWriter(image_video_path, 0, 1, (1024, 544))
combined_video = cv2.VideoWriter(combined_video_path, 0, 1, (1024, 544))

for image_path, disparity_path in images():

    print(image_path, disparity_path)

    image_frame = cv2.imread(image_path)
    disparity_frame = cv2.imread(disparity_path)

    print(image_frame.shape)
    print(disparity_frame.shape)

    combined_video.write(disparity_frame)
    combined_video.write(image_frame)

    image_video.write(image_frame)
    disparity_video.write(disparity_frame)


image_video.release()
disparity_video.release()
combined_video.release()
