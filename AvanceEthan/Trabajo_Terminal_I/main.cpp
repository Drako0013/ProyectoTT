#include <string>
#include <iostream>
#include <cstdio>

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "frame.h"
#include "lucas_kanade.h"
#include "video_factory.h"
#include "horn_schunck.h"


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

  cv::Mat capture;
  Frame frame(true);
  Frame redFrame(false);
  std::string dir = std::string(argv[1]);

  int width = vcapture.get(CV_CAP_PROP_FRAME_WIDTH);
  int height = vcapture.get(CV_CAP_PROP_FRAME_HEIGHT);

  //VideoFactory vf(dir + "flow_hs.avi", width, height, vcapture.get(CV_CAP_PROP_FPS));

  /*
  LucasKanade lk;
  
  cv::Mat vx, vy;
  cv::Mat v(height, width, CV_8U);
  std::cout << "\n\nStarting process.\n";
  for (int i = 0; true; ++i) {
    std::cout << "Processing frame " << i << ".\n";

    vcapture >> capture;
    if (capture.empty()) break;
  	frame.SetMatrix(&capture);
    lk.AddFrame(&frame);

    lk.CalculateFlow(vx, vy);
    for (int ii = 0; ii < vx.rows; ++ii) {
      for (int jj = 0; jj < vx.cols; ++jj) {
        double x = vx.at<double>(ii, jj);
        double y = vy.at<double>(ii, jj);
        uchar flow = (x > 3.5 || y > 3.5)? 255: 0;
        v.at<uchar>(ii, jj) = flow;
      }
    }
    vf.AddFrame(v);
  }
  */
  HornSchunck hs;
  VideoFactory min(dir + "min.avi", 100, 80, vcapture.get(CV_CAP_PROP_FPS));
  cv::Mat vx, vy;
  cv::Mat v(80, 100, CV_8U);
  std::cout << "\n\nStarting process.\n";
  for (int i = 0; i < 100; ++i) {
    std::cout << "Processing frame " << i << ".\n";

    vcapture >> capture;
    if (capture.empty()) break;
  	frame.SetMatrix(&capture);
	cv::Mat redMatrix = frame.reduceImageSize(100, 80);
	//min.AddFrame(redMatrix);
	redFrame.SetMatrix(&redMatrix);
	hs.AddFrame(&redFrame);
    hs.CalculateFlow(vx, vy);
    for (int ii = 0; ii < vx.rows; ++ii) {
      for (int jj = 0; jj < vx.cols; ++jj) {
        double x = vx.at<double>(ii, jj);
        double y = vy.at<double>(ii, jj);
        uchar flow = (x > 3.5 || y > 3.5)? 255: 0;
        v.at<uchar>(ii, jj) = flow;
      }
    }
    min.AddFrame(v);

  }

  return 0;
}