Detail approach to the problem and the success in the task specified.

Provide illustrative images of the intermediate results of the system (e.g. overlays), results of the processing stages.
Titles, captions, references and graphs do not count towards the total word count of the report.

# Pre-processing

The pre-processing steps were based off the results found in [this study](https://ieeexplore.ieee.org/document/8284645),
namely the use of a median filter, Weiner filter, and histogram equalisation.


## Stereo-based.

Much research has been done into choosing optimal parameters for *StereoBM*

https://jayrambhia.com/blog/disparity-mpas
`
sgbm.SADWindowSize = 5;
sgbm.numberOfDisparities = 192;
sgbm.preFilterCap = 4;
sgbm.minDisparity = -64;
sgbm.uniquenessRatio = 1;
sgbm.speckleWindowSize = 150;
sgbm.speckleRange = 2;
sgbm.disp12MaxDiff = 10;
sgbm.fullDP = false;
sgbm.P1 = 600;
sgbm.P2 = 2400;
`

sparse - feature point based. I could prbably creack one out.

Scene changes in terrain type, illumination conditions, clutter, and road markings.

Efficiency is less important performance.



A tool was used to tune StereoBM, available [here](https://github.com/vmarquet/opencv-disparity-map-tuner)


`cv::filterSpeckles` can be integrated into `SGBM` by setting `speckleWindowSize` and `speckleRange`; this is done here
for conciseness. 

maxDisparity should be chosen based on your camera's setup.

Pre-processing is typically performed to reduce photometric variations between the images.

Weighted Least Squares (WLS) was chosen as the alternative dense stereo ranging method.
It was found to be far superior to the basic SGBM method.


# Sources
* https://www.ncbi.nlm.nih.gov/pubmed/28350382
* https://hal.archives-ouvertes.fr/hal-01609038/file/moving-object-detection.pdf
* http://openaccess.thecvf.com/content_ECCV_2018/papers/Peiliang_LI_Stereo_Vision-based_Semantic_ECCV_2018_paper.pdf
* https://ieeexplore.ieee.org/document/8284645
