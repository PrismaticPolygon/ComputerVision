Detail approach to the problem and the success in the task specified.

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

A tool was used to tune StereoBM, available [here](https://github.com/vmarquet/opencv-disparity-map-tuner)


`cv::filterSpeckles` can be integrated into `SGBM` by setting `speckleWindowSize` and `speckleRange`; this is done here
for conciseness. 

maxDisparity should be chosen based on your camera's setup.