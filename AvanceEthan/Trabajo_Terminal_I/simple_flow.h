#ifndef SIMPLEFLOW_H_
#define SIMPLEFLOW_H_

#include <algorithm>
#include <cmath>
#include <vector>
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

private:
	std::vector<Frame*> frames;
	void BuildPyramid(Frame &src, std::vector<Frame>& pyramid);
	int GetEnergy(Frame &f1, int x1, int y1, Frame &f2, int x2, int y2);
	double GetWr(std::vector<int> &energyArray);
	double GetWd(int x0, int y0, int x, int y);
	double GetWc(Frame &f1, int x0, int y0, int x, int y);
	//double GetWc(cv::Mat &f1, int x0, int y0, int x, int y);
	double getSmoothness(Frame &f1, Frame &f2, int x0, int y0, int x, int y);
	cv::Mat UpscaleFlow(cv::Mat& flow, int new_cols, int new_rows, Frame &image, cv::Mat& confidence);
	void CalcStageFlow(Frame& cur, Frame& next, cv::Mat& flow_x, cv::Mat& flow_y);
	void BilateralFilter(cv::Mat flow_x, cv::Mat flow_y, Frame cur, Frame next, cv::Mat confidence, std::vector< std::vector<bool> >& isOccludedPixel);
	void CrossBilateralFilter(cv::Mat &orig, Frame &edge, cv::Mat& confidence, cv::Mat& dest);
	void CalcOcclusion(Frame& cur, Frame& next, cv::Mat& vel_x, cv::Mat& vel_y, cv::Mat& vel_x_inv, cv::Mat& vel_y_inv, std::vector< std::vector<bool> >& isOccludedPixel);
	void CalcConfidence(Frame& cur, Frame& next, cv::Mat& confidence);
	static const int NeighborhoodSize = 5;
	static const int Layers = 5;
	static const double rd;
	static const double rc;
	static const double occlusion_limit;
};


#endif