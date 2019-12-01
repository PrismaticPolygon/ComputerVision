1. Automatically detect objects
2. Estimate their distance from the vehicle (ranging) by integrating the use of depth (disparity) information

## Task
* Design a prototype computer vision system to estimate the range of specific objects
of interest from the vehicle at any given point in a journey
* Can use YOLO (You Only Look Once) or another object detection approach
* For each detected object, make a single estimate of its distance from the object, using either 
dense stereo (provided), sparse (feature-point based) or some other variant.
* Solution must cope with noise.
* Initially, only the identification of pedestrians and vehicles is necessary.
Extra credit is available for detecting other types of object.
* Performance is more important than efficiency.

## Marks

* any image pre-filtering or optimization performed (or similar first stage processing)
to improve either/both object detection or stereo depth estimation  [10]
* effective integration of (existing) object detection and dense stereo ranging [10]
* object range estimation strategy for challenging conditions   [10]
* general performance [20]
* good code [5]
* discussion / detail of solution design and choices made in report [10]
* qualitative and quantitative evidence of performance [10]
* additional credit is given for an alternative sparse stereo based ranging approach, another variant approach to 
stereo-based ranging, and the use of heuristics or advanced processing / optimisation to improve performance.

## Notes
* Automatic detection of objects and the estimation of their distance from the vehicle
within stereo video imagery from an on-board facing stereo camera.
* Works by integrating depth (disparity) information recovered from an existing stereo vision algorithm
with an object detection algorithm.
* Low cost and high granularity means classification and distance of objects is superior to LIDAR.
* Set of still image pairs.

* Only required to identify two types (class) of dynamic objects - pedestrians and vehicles. 

## Submission
* Full program source code, together with any additional files, as a Python script. Include clear instructions in 
a README on how to run it on the TTBB stereo dataset.
* Example video file showing general performance.
* 750 word report detailing approach taken and success.

So we're basically merging object detection and stereo vision.
Nice! SVN for object detection.

## Advice

It sound quite similar to our coursework last year. I’m assuming you are using SGBM for stereo matching and 
SVN for object detection. The simplest way is to just get the distance of the centre pixel of your bounding box. 
That doesn’t work too well because of the noise. It’s better to do something like a mean or median of 
distances over the whole bounding box or a section of it. The distance is bf/d where b is distance 
between two cameras, f is focal length of a camera and d is the disparity of the pixel.
To get better stereo you will need to tweak the parameters of SGBM or use something like WLS filtering or a 
newer stereo algorithm but SGBM does the job just fine. To get better detection accuracy you can 
you’ll have to experiment with SVN or any other detection strategy you are using. For SVN I used 
HOG and radial basis kernel to get best performance. I also used stereo to filter out some false positives. 
Don’t go overboard with stereo parameters as the more aggressive you go the slower it will get.

Use HOG with YOLO.

Stereo to 3D projects a single example stereo pair
to 3D. Presumably? 

## Files
* The stereo disparity files can be found [here](https://github.com/tobybreckon/stereo-disparity).
* Other files can be found [here](https://github.com/tobybreckon/python-examples-cv).

## Glossary

##### Stereo SBGM

A modified Hirschmuller (1990) algorithm. It differs from the original by:
* By default, the algorithm is single-pass, so only 5 directions are considered, instead of 8.
* The algorithm matches blocks, not individual pixels.
* The mutual information cost function is not implemented. A simpler Birchfield-Tomasi sub-pixel metric (Birchfield and Tomasi, 1998)
* Some pre-filtering (Sobel) and post-filtering (uniqueness check, quadratic interpolation, and speckle filtering), as in StereoBM.

Could use WLS filtering to improve performance. The more aggressive the stereo, the slower it will be


## Scripts
* `stereo_disparity.py` loads, displays and computes SGBM disparity from the dataset. 


# YOLO
Given an S x S grid on input, split it into bounding boxes and confidence, a class probability map,
and into the final detections.

There are three primary object detectors that you'll encounter:
* R-CNN and their variants. Incredibly slow: only 5FPS on a GPU. Two-stage.
* Single Shot Detectors (SSDs). One-stage. Treats object detection as a regression problem: taking a given input image
and simultaneously learning bounding box coordinates and corresponding class label probabilities.
* YOLO

Single-stage detectors tend to be less accurate than two-stage detectors but are significantly faster.

YOLO performs joint training for both object detection and classification.

COCO consists of 80 labels.

Folder contains 4 static images. We'll perform object detection on
for testing and evaluation purposes.

YOLO does not always handle small objects well.
It especially does not handle objects grouped close together. 

Easy trade-off between speed and accuracy by changing the size of the model (can I do this in practicality?)

Applies a single neural network to the full image. It divides the images into regions and predics
bounding boxes and probabilities for each region. Each bounding box is weighted by the predicte
probabilities

Predictions are informed by global context. Extremely faster.
Each grid cell or anchor represents a classifier which is responsible for generating
K bounding boxes around potential objects whose ground truth cetner falls within that cell.

# Packages

Updating packages within a venv can be tricky. Use `python -m pip install -U --force-reinstall <package>`.

* `numpy`
* `opencv`
* `opencv-contrib-python`

