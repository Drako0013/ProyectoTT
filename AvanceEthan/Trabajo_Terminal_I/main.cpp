#include <iostream>

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

  LucasKanade lk;
  cv::Mat capture;
  Frame frame(true);
  VideoFactory vF(std::string("C:\\Users\\Drako\\Desktop\\salida.avi"), 
	  (int) vcapture.get(CV_CAP_PROP_FRAME_WIDTH), 
	  (int) vcapture.get(CV_CAP_PROP_FRAME_HEIGHT),
	  vcapture.get(CV_CAP_PROP_FPS));

  for (int i = 0; i < 100; ++i) {
    std::cout << "Processing frame " << i << ".\n";

    vcapture >> capture;
    if (capture.empty()) break;
		frame.SetMatrix(&capture);
    
    std::string path_and_index = std::string(argv[2]) + std::to_string(static_cast<long long>(i));
    //imwrite(path_and_index + "_original.jpg", capture);
    //imwrite(path_and_index + "_smooth.jpg", lk.AddFrame(&frame));
	vF.agregaFrame( frame.GetMatrix() );
  }

  return 0;
}