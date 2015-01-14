#ifndef SIMPLEFLOW_H_
#define SIMPLEFLOW_H_

#include <algorithm>
#include <cmath>
#include <map>
#include <utility>
#include <limits>

#include "opencv2/imgproc/imgproc.hpp"

#include "frame.h"

class SimpleFlow{
public:
	// Frame control methods
	cv::Mat AddFrame(Frame*);
	void RemoveFrame();
	// Flow calculation method
	void CalculateFlow(cv::Mat& vel_x, cv::Mat& vel_y);
	double getSmoothness(Frame &f1, Frame &f2, int x0, int y0, int x, int y);
private:
	std::vector<Frame*> frames;
	int GetEnergy(Frame &f1, int x1, int y1, Frame &f2, int x2, int y2);
	double GetWr(std::vector<int> &energyArray);
	double GetWd(int x0, int y0, int x, int y);
	double GetWc(Frame &f1, int x0, int y0, int x, int y);
	double SimpleFlow::GetWc(cv::Mat &f1, int x0, int y0, int x, int y);
	static const int NeighborhoodSize = 5;
	static const double rd;
	static const double rc;
	static const double occlusion_limit;
};


#endif