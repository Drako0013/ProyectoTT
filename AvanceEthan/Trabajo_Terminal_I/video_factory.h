#ifndef VIDEO_FACTORY_H_
#define VIDEO_FACTORY_H_

#include "opencv2/core/core.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

class VideoFactory {
 public:
  VideoFactory(int, int, int);
  VideoFactory(std::string&, int, int, int);
  void AddFrame(cv::Mat&);
		
 private:
  std::string filename_;
  cv::VideoWriter output_;
};

#endif