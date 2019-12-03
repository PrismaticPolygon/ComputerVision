Detail approach to the problem and the success in the task specified.

Provide illustrative images of the intermediate results of the system (e.g. overlays), results of the processing stages.
Titles, captions, references and graphs do not count towards the total word count of the report.



# Pre-processing

The main purpose of pre-processing is to remove photometric distortion.

The pre-processing steps were based off the results found in [this study](https://ieeexplore.ieee.org/document/8284645),
namely the use of a median filter, Weiner filter, and histogram equalisation.


The purpose of histogram equalisation is used to increase the global contrast of images. The images in the dataset
are ideal for equalisation - they have backgrounds and foregrounds that are both light and dark.

Histogram equalisation works better when applied to 16-bit grayscale images.
It can produce undesirable effects (like a visible image gradient) when applied to images with a low color depth.
Applying it to an 8-bit image will further reduce the color depth of the image. 

Results are below. The default operation in the scripts provided - raising the images to the power of `3/4` - seemed
to be incompatible with subsequent operations, and so was discarded. A comparison of two histogram equalisation
techniques is imaged below, using the default parameters recommended on OpenCV.

Contrast-limiting adaptive adaptive histogram equalisation (CLAHE)

The left-hand image is a `left` image from the dataset, converted to greyscale, and chosen for high contrast.
Objects in the shaded right-hand region are difficult to distinguish; 
details in the upper left-hand region are obscured by bright light.

The centre image is 'default' histogram equalisation `cv2.equalizeHist`. 
It appears 'washed out'.

The right image uses CLAHE. Details are a lot sharper, and details in the upper left-hand
region have been preserved, while the lower right-hand region has been simultaneously lightened.

CLAHE was found to be superior. Parameters were tweaked to compensate for the challenges of this dataset.

Following this, a range of CLAHE parameters were tweaked to improve results. 

AHE computes several histograms, each corresponding to a distcint section of the image,
and uses them to re-distribute the lightness values of the image.
It is therefore suitable for improving the local contrast and enhancing the definitions of edges 
in each region. Has a tendency of over-amplify noise in relatively
homogeneous regions of an image. CLAHE prevents this by limiting the amplification
by clipping the histogram at a pre-defined value.

% Then we can arrange them into a nice stack. 


Typical operations include Laplacian of Gaussian (LoG) filtering and bilateral filtering.


Laplacian of Gaussian (LoG) filtering (T. Kanade, H. Kato, S. Kimura, A. Yoshida, and K. Oda, Development of a Video-Rate Stereo Machine International Robotics and Systems Conference (IROS '95), Human Robot Interaction and Cooperative Robots, 1995 )
Subtraction of mean values computed in nearby pixels ( O. Faugeras, B. Hotz, H. Mathieu, T. Viville, Z. Zhang, P. Fua, E. Thron, L. Moll, G. Berry, Real-time correlation-based stereo: Algorithm. Implementation and Applications, INRIA TR n. 2013, 1993)
Bilateral filtering (A. Ansar, A. Castano, L. Matthies, Enhanced real-time stereo using bilateral filtering IEEE Conference on Computer Vision and Pattern Recognition 2004)
Census transform


# Dense stereo

Two dense stereo approaches were comapred. The first, Semi-Global Block Matching (SGBM), is...
The second, Weighted Least Squares (WLS) is....




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

Regular equalisation performs poorly when the image contains regions that are significantly ligher or darker than most of the image.
As the contrast in those regions will not be sufficiently enhanced. 

Contrast at smaller scales is enhanced. Contrast at larger scales is reduced.

Both M and N must be at least 2.

Some research has been performed into automating the selection of parameters for CLAHE, but was deemed out-of-scope for this
project; parameters were tweaked by hand, in reasonable ranges. 

Best results were found with higher-than-expected tile-grid-sizes; around 16 seemed to be optimal.

3 seemed to be optimal. Values of 3 - 4 are recommended [here](https://en.wikipedia.org/wiki/Adaptive_histogram_equalization#Contrast_Limited_AHE)

A study of CLAHE here recommends a value of 0.25; however, in some implementations (e.g. MATLAB, the allowable range is `[0, 1]`),
But then it's not clear what my limits are either. 


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
