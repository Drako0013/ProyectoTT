#include <stdio.h>
#include <string>
#include <iostream>

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
  std::string dir = std::string(argv[1]);

  int width = 320;
  int height = 240;

  VideoFactory vf(dir + "-lk-flow.avi", width, height, vcapture.get(CV_CAP_PROP_FPS));

  LucasKanade lk;
  
  cv::Mat vx, vy;
  cv::Mat v(height, width, CV_8U);
  std::cout << "\n\nStarting process.\n";

  Frame* result = new Frame(false);
  for (int i = 0; true && i < 1000; ++i) {
    std::cout << "Processing frame " << i << ".\n";

    vcapture >> capture;
    if (capture.empty()) break;

    if (!i) {
      result->SetMatrix(&capture);
      result->Rescale(width, height);
      result->GetMatrixOnCache();
    }

  	Frame* frame = new Frame(&capture);
    frame->Rescale(width, height);
    frame->GetMatrixOnCache();
    lk.AddFrame(frame);

    lk.CalculateFlow(vx, vy);
    for (int x = 0; x < height; ++x) {
      uchar* ptr = v.ptr<uchar>(x);
      double* ptr_vx = vx.ptr<double>(x);
      double* ptr_vy = vy.ptr<double>(x);
      for (int y = 0; y < width; ++y, ++ptr, ++ptr_vx, ++ptr_vy) {
        double X = *ptr_vx, Y = *ptr_vy;
        uchar flow_x = (X < -50 || 50 < X)? 255: 0;
        uchar flow_y = (Y < -50 || 50 < X)? 255: 0;
        result->SetPixel(x, y, flow_x << 16 | flow_y);
      }
    }
    result->GetCacheOnMatrix();
    vf.AddFrame(result->GetMatrix());
  }
  
  /*HornSchunck hs;
  int widthF, heightF, ratio;
  ratio = width / 100;
  if(ratio == 0){
	widthF = width;
	heightF = height;
  } else {
	widthF = width / ratio;
	heightF = height / ratio;
  }
  VideoFactory min(dir + "min.avi", widthF, heightF, vcapture.get(CV_CAP_PROP_FPS));
  cv::Mat vx, vy;
  cv::Mat v(heightF, widthF, CV_8U);
  std::cout << "\n\nStarting process.\n";
  std::cout << heightF << " x " << widthF << std::endl;
  std::cout << vcapture.get(CV_CAP_PROP_FPS) << " FPS" << std::endl;
  for (int i = 0; i < 500; ++i) {
    //std::cout << "Processing frame " << i << ".\n";

    vcapture >> capture;
    if (capture.empty()) break;
  	frame.SetMatrix(&capture);
	cv::Mat redMatrix = frame.Rescale(widthF, heightF);
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

  }*/

  return 0;
}