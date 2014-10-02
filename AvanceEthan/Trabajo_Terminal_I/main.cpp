#include <iostream>
#include <string>

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "frame.h"
#include "lucas_kanade.h"
#include "VideoFactory.h"

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cout << "Video input file and output directory required.";
    return 0;
  }

  cv::VideoCapture vcapture;

  vcapture.open(argv[1]);
  if (!vcapture.isOpened()) {
    std::cout << "Could not initialize capturing.\n";
    return 0;
  }
  std::string dir = std::string(argv[2]);

  LucasKanade lk;
  cv::Mat capture;
  Frame frame(true);
  VideoFactory vF(dir + "\\salida.avi", 
	  (int) vcapture.get(CV_CAP_PROP_FRAME_WIDTH), 
	  (int) vcapture.get(CV_CAP_PROP_FRAME_HEIGHT),
	  vcapture.get(CV_CAP_PROP_FPS));
  /*
  VideoFactory vFx(dir + "\\salidaX.avi", 
	  (int) vcapture.get(CV_CAP_PROP_FRAME_WIDTH), 
	  (int) vcapture.get(CV_CAP_PROP_FRAME_HEIGHT),
	  vcapture.get(CV_CAP_PROP_FPS));
  VideoFactory vFy(dir + "\\salidaY.avi", 
	  (int) vcapture.get(CV_CAP_PROP_FRAME_WIDTH), 
	  (int) vcapture.get(CV_CAP_PROP_FRAME_HEIGHT),
	  vcapture.get(CV_CAP_PROP_FPS));
  VideoFactory vFt(dir + "\\salidaT.avi", 
	  (int) vcapture.get(CV_CAP_PROP_FRAME_WIDTH), 
	  (int) vcapture.get(CV_CAP_PROP_FRAME_HEIGHT),
	  vcapture.get(CV_CAP_PROP_FPS));
  */
  VideoFactory vFf(dir + "\\salidaF.avi",
	  (int)vcapture.get(CV_CAP_PROP_FRAME_WIDTH),
	  (int)vcapture.get(CV_CAP_PROP_FRAME_HEIGHT),
	  vcapture.get(CV_CAP_PROP_FPS));

  for (int i = 0; i < 50; ++i) {
    std::cout << "Processing frame " << i << ".\n";

    vcapture >> capture;
    if (capture.empty()) break;
		frame.SetMatrix(&capture);
    
    std::string path_and_index = std::string(argv[2]) + std::to_string(static_cast<long long>(i));
	vF.agregaFrame( lk.AddFrame(&frame) );

	//*cv::Mat gradX = lk.GradientEstimationAtX();
	//cv::Mat gradY = lk.GradientEstimationAtX();
	//cv::Mat gradT = lk.GradientEstimationAtT();
	cv::Mat velX, velY, vel(frame.Rows(), frame.Columns(), CV_8U);
	//gradX.copyTo(vel);
	lk.CalculateFlow(velX, velY);

	//vFy.agregaFrame( gradX );
	//vFx.agregaFrame( gradY );
	//vFt.agregaFrame( gradT );

	for (int ii = 0; ii < velX.rows; ii++) {
		for (int jj = 0; jj < velY.cols; jj++) {
			double x = velX.at<double>(ii, jj), y = velY.at<double>(ii, jj);
			vel.at<uchar>(ii, jj) = (uchar)(sqrt(x * x + y * y));
		}
	}
	vFf.agregaFrame( vel );

  }

  return 0;
}