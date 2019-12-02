### Enhancement Technique for Improving the Reliability of Disparity Map Under Low light Condition

Histogram-based enhancement techniques along with filtering to improve reliability. 
Histogram equalisation combined with a low-pass and median filter is used to process
the stereo images before disparity map creation. 

Histogram equalisation - gaussian filtering - median filtering - median filtering - gaussian filtering

Histogram equalisation can result in noisy images that need to be filtered properly
before calculating disparity. And that's for low-light as well; it might not be suitable. 

Should I be projecting onto a point cloud?



# https://github.com/vmarquet/opencv-disparity-map-tuner. Only works on Linux / Mac.

Median filter for denoising, Wiener filter for deblurring,
and contrast enhancement by Histogram Equalisation. 
Many algorithms give integer disparity values. This can result in 
discontinuous disparity maps and lead to a lot of information loss,
particularly at more consierable distances. Can be handled using
gradient descent and curve fitting. 


Disparity is the difference between the x co-ordinate of the two
corresponding points. Typically encoded with greyscale.

Depth measured by a stereo vision system is discretised into parallel
planes, one for each disparity value. 

Stereo correspondence: finding homologous points in the stereo pair.

Difficult because of photometric distortions and noise,
specular surfaces, foreshortening, perspective distortions,
uniform / ambiguous regions,
repetitive / ambiguous patterns, transparent objects,
occlusions and discontinuities

Middlesbury stereo evaluation site http://vision.middlebury.edu/stereo/eval/
holds schizzle for evaluation.

Includes Tsukuba, Venus, Teddy, and Cones stereo pairs.

Most stereo algorithms:
1. Matching cost computation
2. Cost aggregation
3. Disparity computation / optimisation
4. Disparity refinement

Local algorithms perform 1 -> 2 ->, with a simple winner-takes-all
Global algorithms perform 1 -> (2) -> 3 with global or semi-global reasoning.

Pre-processing is sometimes deployed to compensate for photometric distortions.
Typical operations include:
 
 * Laplacian of Gaussian (LoG) filtering (T. Kanade, H. Kato, S. Kimura, A. Yoshida, and K. Oda, Development of a Video-Rate Stereo Machine
International Robotics and Systems Conference (IROS '95), Human Robot Interaction and Cooperative Robots, 1995 )
 * Subtraction of mean values computed in nearby pixels ( O. Faugeras, B. Hotz, H. Mathieu, T. Viville, Z. Zhang, P. Fua, E. Thron, L. Moll, G. Berry,
Real-time correlation-based stereo: Algorithm. Implementation and Applications, INRIA TR n. 2013, 1993)
* Bilateral filtering (A. Ansar, A. Castano, L. Matthies, Enhanced real-time stereo using bilateral filtering
IEEE Conference on Computer Vision and Pattern Recognition 2004)
* Census transform

Matching cost: pixel-based absolute difference between pixel intensities.

Disparity computation: WTA

Global (and semi-global*) algorithms search for disparity assignments that 
minimize an energy function over the whole stereo pair using a pixel-based 
matching cost (sometime the matching cost is aggregated over a support).

There's lots of area-based matching costs: sum of absolute differences (SAD), sum of squared differences (SSD), sum of truncated absolute differences (STAD)

State-of-the-art cost aggregation strategies aim at shaping the support in order to
include points with the same (unknown) disparity. Then we could get our bounding boxes
and search for disparity within. That might improve results a little. 

Box-filtering

Post-processing: median filtering, morphological operators, bilateral filtering.

Bi-directional matching is a widely used technique for detecting outliers in stereo (local and global)

Holes can be filled using the left-right disparity consistency check, i.e. two disparity maps

## Sources

* [Literature review](https://www.hindawi.com/journals/js/2016/8742920/)
* [Slides](http://vision.deis.unibo.it/~smatt/Seminars/StereoVision.pdf)
* [More stuff](https://www.intechopen.com/online-first/efficient-depth-estimation-using-sparse-stereo-vision-with-other-perception-techniques)