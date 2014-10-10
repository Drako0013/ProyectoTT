#pragma once

#include <vector>
#include <algorithm>

#include "opencv2/imgproc/imgproc.hpp"

#include "frame.h"

class HornSchunck
{
public:
	HornSchunck();
	~HornSchunck();
	cv::Mat AddFrame(Frame* frame);
	void RemoveFrame();
	void CalculateFlow(cv::Mat &, cv::Mat &);
	cv::Mat GradientEstimationAtX();
	cv::Mat GradientEstimationAtY();
	cv::Mat GradientEstimationAtT();
	cv::Mat LocalAverage(cv::Mat &);
private:
	static const int avgKernel[3][3];
	std::vector<Frame> frames_;
};

