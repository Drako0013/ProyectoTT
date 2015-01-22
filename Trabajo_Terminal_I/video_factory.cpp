#include "video_factory.h"

#include "opencv2/core/core.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

VideoFactory::VideoFactory(int width, int height, int fps) {
  filename_ = "default.avi";
  cv::Size size = cv::Size(width, height);
  output_.open(filename_, CV_FOURCC('M','S','V','C'), fps, size, true);
}

VideoFactory::VideoFactory(std::string& filename, int width, int height, int fps) {
  filename_ = filename;
  cv::Size size = cv::Size(width, height);
  output_.open(filename, CV_FOURCC('M','S','V','C'), fps, size, true);
}

void VideoFactory::AddFrame(cv::Mat& frame) {
	output_ << frame;
}