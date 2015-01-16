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
#include "simple_flow.h"

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

  /*VideoFactory vf(dir + "-lk-flow.avi", width, height, vcapture.get(CV_CAP_PROP_FPS));

  //LucasKanade lk;
  SimpleFlow sf;
  
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
    sf.AddFrame(frame);

	if(i == 0) continue;

    sf.CalculateFlow(vx, vy);
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
  }*/
  
  HornSchunck hs;

  VideoFactory min(dir + "min.avi", width, height, vcapture.get(CV_CAP_PROP_FPS));

  double* vx = NULL, *vy = NULL;
  cv::Mat v(height, width, CV_8U);
  std::cout << "\n\nStarting process.\n";
  for (int i = 0; true; ++i) {
    std::cout << "Processing frame " << i << ".\n";

    vcapture >> capture;
    if (capture.empty()) break;
    Frame* frame = new Frame(&capture);
	  frame->Rescale(width, height);
    frame->GetMatrixOnCache();
    hs.AddFrame(frame);

    if (i % 1 == 0) {
      delete [] vx;
      delete [] vy;
      hs.CalculateFlow(&vx, &vy);
    }

    int rows = frame->Rows();
    int cols = frame->Columns();
    for (int x = 0; x < rows; ++x) {
      uchar* ptr = v.ptr<uchar>(x);
      for (int y = 0; y < cols; ++y, ++ptr) {
        double X = vx[x * cols + y];
        double Y = vy[x * cols + y];
        uchar flow = (2 < X || 2 < Y)? 255: 0;
        *ptr = flow;
      }
    }
    min.AddFrame(v);
  }

  return 0;
}